from typing import Dict, Sequence, OrderedDict, Tuple
from torch import Tensor, nn
from torch.utils.data import TensorDataset
from abc import abstractmethod, ABC
from capymoa.ocl.util._buffer import BufferDict
import torch


def _detach(**batch: Tensor) -> Dict[str, Tensor]:
    """Detach a batch of tensors from the computation graph."""
    return {key: tensor.detach() for key, tensor in batch.items()}


class ReplayBuffer(ABC, nn.Module):
    """Abstract base class for replay buffers."""

    @abstractmethod
    def update(self, **batch: Tensor) -> None:
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


class ReplayBuilder(ABC):
    """Capymoa interface for building replay buffer strategies.

    We use a builder pattern to separate hyperparameter configuration from configuring
    the objects and dtype of the buffer. The learner
    :py:class:`~capymoa.base.Classifier` performs construction
    :py:meth:`ReplayBuilder.build` with the appropriate configuration when it is time to
    construct the buffer.

    .. seealso::

        - :py:class:`~capymoa.ocl.strategy._experience_replay.ExperienceReplay`
    """

    @abstractmethod
    def build(
        self,
        capacity: int,
        buffers: Dict[str, Tuple[Sequence[int], torch.dtype]],
        rng: torch.Generator = torch.default_generator,
    ) -> ReplayBuffer:
        """Build a replay buffer from the given configuration."""
        ...

    def new_xybuffer(
        self,
        capacity: int,
        x_shape: Sequence[int],
        rng: torch.Generator = torch.default_generator,
    ) -> "ReplayBuffer":
        """Standard buffer with features ``x: (n, *x_shape) f32`` and labels ``y: (n,) i64``."""
        return self.build(
            capacity, {"x": (x_shape, torch.float32), "y": ((), torch.long)}, rng
        )
