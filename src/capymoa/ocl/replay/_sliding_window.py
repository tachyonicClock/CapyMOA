from typing import Dict
from typing_extensions import override
from torch import Tensor

from ._base import ReplayBuilder, ReplayBuffer, _detach


class SlidingWindow(ReplayBuilder):
    """Replay buffer that retains the most recent examples.

    Stores incoming examples in a circular buffer, evicting the oldest example
    once capacity is reached.
    """

    class _SlidingWindow(ReplayBuffer):
        @override
        def update(self, **batch: Tensor) -> None:
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

    def build(self, *args, **kwargs) -> ReplayBuffer:
        return self._SlidingWindow(*args, **kwargs)
