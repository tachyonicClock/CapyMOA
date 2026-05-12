from typing_extensions import override
from torch import Tensor
import torch

from ._base import ReplayBuilder, ReplayBuffer, _detach


class ReservoirSampler(ReplayBuilder):
    """Replay buffer using reservoir sampling.

    Each incoming example is stored with equal probability, ensuring a uniform
    random sample of all examples seen so far regardless of arrival order.
    """

    class _ReservoirSampler(ReplayBuffer):
        @override
        def update(self, **batch: Tensor) -> None:
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

    def build(self, *args, **kwargs) -> ReplayBuffer:
        return self._ReservoirSampler(*args, **kwargs)
