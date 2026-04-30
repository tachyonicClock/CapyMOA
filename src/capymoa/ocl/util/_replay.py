from typing import Dict, Sequence, OrderedDict, Tuple
from typing_extensions import override
from torch import Tensor, nn
from torch.utils.data import TensorDataset
from abc import abstractmethod, ABC
from capymoa.ocl.util._buffer import BufferDict
import torch


def _detach(**batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Detach a batch of tensors from the computation graph."""
    return {key: tensor.detach() for key, tensor in batch.items()}


class ReplayBuffer(ABC, nn.Module):
    @abstractmethod
    def update(self, **batch: Dict[str, Tensor]) -> None:
        """Update the replay buffer with new examples.

        :param batch: Dictionary of tensors representing a batch of examples.
        """
        ...

    def sample(self, n: int) -> OrderedDict[str, Tensor]:
        """Sample ``n`` examples from the replay buffer.

        :param n: Number of examples to sample
        :return: Dictionary of tensors representing the sampled examples.
        """
        indices = torch.randint(0, self.count, (n,))
        return {key: buffer[indices] for key, buffer in self._buffer.items()}

    def array(self) -> Dict[str, Tensor]:
        """Return the replay buffer as a dictionary of tensors."""
        return {key: buffer[: self.count] for key, buffer in self._buffer.items()}

    def dataset_view(self) -> TensorDataset:
        """Return a TensorDataset view of the replay buffer."""
        return TensorDataset(*self.array().values())

    @property
    def capacity(self) -> int:
        """Return the maximum number of samples that can be stored in the coreset."""
        return self._capacity

    @property
    def count(self) -> int:
        """Return the current number of samples in the coreset."""
        assert self._count <= self._capacity
        return self._count

    @property
    def device(self) -> torch.device:
        return next(iter(self._buffer.values())).device

    def __init__(
        self,
        capacity: int,
        buffers: Dict[str, Tuple[Sequence[int], torch.dtype]],
        rng: torch.Generator = torch.default_generator,
    ) -> None:
        super().__init__()
        self._capacity = capacity
        self._rng = rng
        self._count = 0
        self._i = 0
        self._buffer = BufferDict(
            {
                key: torch.zeros((capacity, *shape), dtype=dtype)
                for key, (shape, dtype) in buffers.items()
            }
        )

    @classmethod
    def new_xybuffer(
        cls,
        capacity: int,
        x_shape: Sequence[int],
        rng: torch.Generator = torch.default_generator,
    ) -> "ReplayBuffer":
        """Standard buffer with features ``x: (n, *x_shape) f32`` and labels ``y: (n,) i64``."""
        return cls(
            capacity, {"x": (x_shape, torch.float32), "y": ((), torch.long)}, rng
        )


class ReservoirSampler(ReplayBuffer):
    @override
    def update(self, **batch: Dict[str, Tensor]) -> None:
        assert set(batch.keys()) == set(self._buffer.keys())
        batch = _detach(**batch)
        batch_size = next(iter(batch.values())).shape[0]
        for key, values in batch.items():
            assert values.shape[0] == batch_size
            batch[key] = values.to(self._buffer[key].device)

        for i in range(batch_size):
            if self.count < self.capacity:
                # Fill the reservoir.
                for key, values in batch.items():
                    self._buffer[key][self.count] = values[i]
                self._count += 1
            else:
                # Standard reservoir sampling replacement.
                index = int(
                    torch.randint(0, self._i + 1, (1,), generator=self._rng).item()
                )
                if index < self.capacity:
                    for key, values in batch.items():
                        self._buffer[key][index] = values[i]
            self._i += 1


class GreedySampler(ReplayBuffer):
    """Update the buffer with every new example, replacing a random example from the
    majority class if the buffer is full.
    """

    @override
    def update(self, **batch: Dict[str, Tensor]) -> None:
        assert set(batch.keys()) == set(self._buffer.keys())
        assert "y" in batch
        batch = _detach(**batch)

        batch_size = next(iter(batch.values())).shape[0]
        prepared_batch: Dict[str, Tensor] = {}
        for key, values in batch.items():
            assert values.shape[0] == batch_size
            prepared_batch[key] = values.to(self._buffer[key].device)

        for i in range(batch_size):
            if self.count < self.capacity:
                # Room left in the coreset for this example.
                for key, values in prepared_batch.items():
                    self._buffer[key][self.count] = values[i]
                self._count += 1
                continue

            # Coreset is full, replace a random example from the majority class.
            y_buffer = self._buffer["y"][: self.count]
            classes, counts = y_buffer.unique(return_counts=True)
            replace_class = classes[counts.argmax()]
            candidate_indices = (y_buffer == replace_class).nonzero(as_tuple=True)[0]
            idx = int(
                torch.randint(
                    0, len(candidate_indices), (1,), generator=self._rng
                ).item()
            )
            replace_idx = int(candidate_indices[idx].item())
            for key, values in prepared_batch.items():
                self._buffer[key][replace_idx] = values[i]


class SlidingWindow(ReplayBuffer):
    """Update the buffer with every new example, replacing the oldest example if the
    buffer is full.
    """

    @override
    def update(self, **batch: Dict[str, Tensor]) -> None:
        assert set(batch.keys()) == set(self._buffer.keys())
        batch = _detach(**batch)

        batch_size = next(iter(batch.values())).shape[0]
        prepared_batch: Dict[str, Tensor] = {}
        for key, values in batch.items():
            assert values.shape[0] == batch_size
            prepared_batch[key] = values.to(self._buffer[key].device)

        # Calculate where the batch ends
        end_idx = self._i + batch_size

        if end_idx <= self.capacity:
            # Case 1: Simple slice (no wrap-around)
            for key, values in prepared_batch.items():
                self._buffer[key][self._i : end_idx] = values
        else:
            # Case 2: Wrap-around (split the batch)
            mid_point = self.capacity - self._i

            # Fill the end of the buffer
            for key, values in prepared_batch.items():
                self._buffer[key][self._i :] = values[:mid_point]

            # Wrap the remainder to the start
            wrap_size = batch_size - mid_point
            for key, values in prepared_batch.items():
                self._buffer[key][:wrap_size] = values[mid_point:]

        # Update index and count
        self._i = (self._i + batch_size) % self.capacity
        self._count = min(self._count + batch_size, self.capacity)
