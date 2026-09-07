"""Dataset helpers that require PyTorch.

Separated from :mod:`capymoa.datasets._utils` so that importing
:mod:`capymoa.datasets` does not import PyTorch, which is an optional extra.
Only :mod:`capymoa.ocl.datasets` uses these, and OCL requires PyTorch anyway.
"""

from collections.abc import Callable

import torch


class TensorDatasetWithTransform(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]
):
    """A PyTorch dataset that applies a transformation to the data."""

    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[object], object] | None = None,
    ):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y
