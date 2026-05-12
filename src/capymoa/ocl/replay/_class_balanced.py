from typing import Dict, Sequence, OrderedDict, Tuple
from typing_extensions import override
from torch import Tensor
import torch

from capymoa.ocl.util._buffer import BufferDict
from ._base import ReplayBuilder, ReplayBuffer, _detach


class ClassBalanced(ReplayBuilder):
    """Class-balanced replay buffer using per-class reservoir sampling.

    This builder maintains equal representation of classes in the replay buffer
    by tracking per-class samples and using reservoir sampling within each class.
    When the buffer is full, samples are replaced from the class with the most
    samples to maintain balance.

    This is particularly useful for class-imbalanced streaming data.
    """

    class _ClassBalancedBuffer(ReplayBuffer):
        """Maintains class-balanced replay buffer using per-class tracking."""

        def __init__(
            self,
            capacity: int,
            buffers: Dict[str, Tuple[Sequence[int], torch.dtype]],
            rng: torch.Generator = torch.default_generator,
        ) -> None:
            super().__init__(capacity, buffers, rng)
            # Track per-class sample counts and indices
            # class_id -> count
            self._class_counts: Dict[int, int] = {}
            # class_id -> buffer indices
            self._class_indices: Dict[int, list[int]] = {}
            # class_id -> total updates seen
            self._class_update_count: Dict[int, int] = {}

        @override
        def update(self, **batch: Tensor) -> None:
            assert set(batch.keys()) == set(self._buffer.keys())
            assert "y" in batch, "Class balanced buffer requires 'y' (labels) in batch"
            batch = _detach(**batch)

            batch_size = next(iter(batch.values())).shape[0]
            prepared_batch: Dict[str, Tensor] = {}
            for key, values in batch.items():
                assert values.shape[0] == batch_size
                prepared_batch[key] = values.to(self._buffer[key].device)

            y_batch = prepared_batch["y"]

            for i in range(batch_size):
                class_id = int(y_batch[i].item())

                # Initialize class tracking if needed
                if class_id not in self._class_counts:
                    self._class_counts[class_id] = 0
                    self._class_indices[class_id] = []
                    self._class_update_count[class_id] = 0

                self._class_update_count[class_id] += 1

                if self.count < self.capacity:
                    # Buffer not full, add sample
                    buffer_idx = self.count
                    for key, values in prepared_batch.items():
                        self._buffer[key][buffer_idx] = values[i]
                    self._class_indices[class_id].append(buffer_idx)
                    self._class_counts[class_id] += 1
                    self._count += 1
                else:
                    # Buffer full, use class-balanced replacement strategy
                    # Find the class with the most samples
                    majority_class = max(
                        self._class_counts.keys(), key=lambda c: self._class_counts[c]
                    )

                    # Use reservoir sampling within the majority class
                    majority_indices = self._class_indices[majority_class]
                    total_seen_majority = self._class_update_count[majority_class]

                    # Reservoir sampling probability
                    prob = self._class_counts[majority_class] / total_seen_majority
                    if torch.rand(1, generator=self._rng).item() < prob:
                        # Replace a random sample from majority class
                        replace_pos = int(
                            torch.randint(
                                0, len(majority_indices), (1,), generator=self._rng
                            ).item()
                        )
                        buffer_idx = majority_indices[replace_pos]

                        # Remove old sample from tracking
                        for key, values in prepared_batch.items():
                            self._buffer[key][buffer_idx] = values[i]

                        # Update class tracking
                        self._class_indices[class_id].append(buffer_idx)
                        self._class_counts[class_id] += 1
                        self._class_counts[majority_class] -= 1
                        majority_indices[replace_pos] = -1  # Mark as removed
                        # Clean up removed indices
                        self._class_indices[majority_class] = [
                            idx for idx in majority_indices if idx != -1
                        ]

        @override
        def sample(self, n: int) -> OrderedDict[str, Tensor]:
            """Sample n examples with class balance.

            Samples uniformly across available classes to maintain class representation.
            """
            if self.count == 0:
                raise ValueError("Cannot sample from empty buffer")

            # Get list of all available classes
            available_classes = [
                c for c in self._class_counts if self._class_counts[c] > 0
            ]

            if len(available_classes) == 0:
                raise ValueError("No samples available in buffer")

            # Determine samples per class
            samples_per_class = n // len(available_classes)
            remainder = n % len(available_classes)

            indices = []
            for class_idx, class_id in enumerate(available_classes):
                class_samples = samples_per_class + (1 if class_idx < remainder else 0)
                class_indices = self._class_indices[class_id]

                if class_samples > 0 and len(class_indices) > 0:
                    # Sample from this class with replacement if needed
                    sampled = torch.randint(
                        0,
                        len(class_indices),
                        (class_samples,),
                        generator=self._rng,
                    )
                    for sample_idx in sampled:
                        indices.append(class_indices[int(sample_idx)])

            # If not enough samples (edge case), fill with random sampling
            while len(indices) < n:
                indices.append(
                    int(torch.randint(0, self.count, (1,), generator=self._rng).item())
                )

            indices_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
            result = BufferDict(
                {key: buffer[indices_tensor] for key, buffer in self._buffer.items()}
            )
            return result

    def build(self, *args, **kwargs) -> ReplayBuffer:
        return self._ClassBalancedBuffer(*args, **kwargs)
