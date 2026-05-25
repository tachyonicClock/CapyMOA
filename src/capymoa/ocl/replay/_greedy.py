from typing import Dict
from typing_extensions import override
from torch import Tensor
import torch

from ._base import ReplayBuilder, ReplayBuffer, _detach


class GreedySampler(ReplayBuilder):
    """Replay buffer that greedily stores every example.

    When the buffer is full, a random example from the current majority class is
    evicted to keep class representation balanced.
    """

    class _GreedySampler(ReplayBuffer):
        @override
        def update(self, **batch: Tensor) -> None:
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
                candidate_indices = (y_buffer == replace_class).nonzero(as_tuple=True)[
                    0
                ]
                idx = int(
                    torch.randint(
                        0, len(candidate_indices), (1,), generator=self._rng
                    ).item()
                )
                replace_idx = int(candidate_indices[idx].item())
                for key, values in prepared_batch.items():
                    self._buffer[key][replace_idx] = values[i]

    def build(self, *args, **kwargs) -> ReplayBuffer:
        return self._GreedySampler(*args, **kwargs)
