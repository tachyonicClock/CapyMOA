"""Memory Aware Synapses (MAS).

References:
* Original:  https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
* FACIL:     https://github.com/mmasana/FACIL/blob/e09d2c83320a1aa945a6157d4875437515824dc9/src/approach/mas.py
* Avalanche: https://github.com/ContinualAI/avalanche/blob/eb075be393e1f458b2c352514ff6c17b5a2c0f4e/avalanche/training/plugins/mas.py
"""

from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from capymoa.base import BatchClassifier
from capymoa.base.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.replay import ReplayBuilder, SlidingWindow
from capymoa.ocl.util._buffer import BufferList
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.stream._stream import Schema

from ._ewc import trainable_params, weighted_l2_reg

NEG_INF = float("-inf")


def _mas_compute_importance(
    model: nn.Module,
    forward_fn: Callable[[Tensor], Tensor],
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    device: torch.device,
) -> Sequence[Tensor]:
    """Estimate MAS parameter importance from the given data loader."""
    model = model.train().to(device)
    importances = [torch.zeros_like(param) for param in trainable_params(model)]

    for x, _ in dataloader:
        x = x.to(device)
        model.zero_grad()

        # MAS importance uses gradients of the squared output norm.
        outputs = forward_fn(x)
        loss = outputs.norm(p=2, dim=1).pow(2).mean()
        loss.backward()

        # Accumulate absolute gradients, safeguarding for None grads
        for imp, param in zip(importances, trainable_params(model), strict=True):
            assert param.grad is not None
            imp.add_(param.grad.data.abs())

    # Average over the number of batches
    for imp in importances:
        imp.div_(len(dataloader))
    return importances


def _mas_update_importance(
    importance: Iterable[Tensor], new_importance: Iterable[Tensor], alpha: float
):
    """Update importance via exponential moving average."""
    for imp, new_imp in zip(importance, new_importance, strict=True):
        imp.mul_(alpha).add_(new_imp, alpha=1 - alpha)


class MAS(BatchClassifier, nn.Module, Handler):
    """Memory Aware Synapses learner."""

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        lambda_: float,
        alpha: float = 0.5,
        buffer_capacity: int = 256,
        importance_replay_builder: Optional[ReplayBuilder] = None,
        importance_batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
        task_mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__(schema, 0)
        nn.Module.__init__(self)
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        self.device = device

        # Hyperparameters
        self._lambda = lambda_
        self._alpha = alpha
        self._importance_batch_size = importance_batch_size

        # Modules
        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()
        if importance_replay_builder is None:
            importance_replay_builder = SlidingWindow()
        self._buffer = importance_replay_builder.new_xybuffer(
            buffer_capacity, schema.shape
        )

        # Buffers used by MAS regularisation
        self._anchor_params = BufferList(
            [param.clone().detach() for param in trainable_params(model)]
        )
        self._importance = BufferList(
            [torch.zeros_like(param) for param in trainable_params(model)]
        )

        # Task tracking
        self._train_task = 0
        self._test_task = 0
        if task_mask is None:
            self._task_mask = None
        else:
            self._task_mask = nn.Buffer(task_mask)

        # Move all model parameters and buffers to the specified device
        self.to(device)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._buffer.update(x=x, y=y)
        self._model.train()
        self._optimiser.zero_grad()
        y_hat = self._train_forward(x)
        loss = self._criterion(y_hat, y)
        total_loss = loss + self._lambda * self._regularisation_loss()
        total_loss.backward()
        self._optimiser.step()

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        y_hat = self._test_forward(x)
        return torch.softmax(y_hat, dim=1)

    def attach_with(self, source: Dispatcher) -> None:
        source.subscribe(TrainTaskBegin, self.on_train_task)
        source.subscribe(TestTaskBegin, self.on_test_task)

    def on_train_task(self, event: TrainTaskBegin) -> None:
        reset_optimizer_state(self._optimiser)
        if event.train_task > 0:
            self._update_importance()
            self._update_anchor_params()
        self._train_task = event.train_task

    def on_test_task(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    def _update_importance(self) -> None:
        """Estimate and accumulate MAS importance from the replay buffer."""
        dataset = self._buffer.dataset_view()
        dataloader = DataLoader(
            dataset, batch_size=self._importance_batch_size, shuffle=False
        )
        with torch.enable_grad():
            task_importance = _mas_compute_importance(
                self._model,
                self._mas_forward,
                dataloader,  # type: ignore[arg-type]
                self.device,
            )
            _mas_update_importance(self._importance, task_importance, self._alpha)

    def _update_anchor_params(self) -> None:
        """Update anchored parameters to the current model weights."""
        for param, anchor_param in zip(
            trainable_params(self._model), self._anchor_params, strict=True
        ):
            anchor_param.copy_(param.detach())

    def _test_forward(self, x: Tensor) -> Tensor:
        """Compute logits for inference, optionally applying a test-task mask."""
        y_hat = self._model(x)
        if self._task_mask is not None:
            y_hat = y_hat.masked_fill(self._task_mask[self._test_task] == 0, NEG_INF)
        return y_hat

    def _train_forward(self, x: Tensor) -> Tensor:
        """Compute logits for training, optionally applying a train-task mask."""
        y_hat = self._model(x)
        if self._task_mask is not None:
            y_hat = y_hat.masked_fill(self._task_mask[self._train_task] == 0, NEG_INF)
        return y_hat

    def _mas_forward(self, x: Tensor) -> Tensor:
        # MAS does not like infinite values, so we simply omit masked-out logits instead
        # of setting them to -inf.
        y_hat = self._model(x)
        if self._task_mask is not None:
            y_hat = y_hat[:, self._task_mask[self._train_task]]
        return y_hat

    def _regularisation_loss(self) -> Tensor:
        """Return the MAS regularisation loss for the current task."""
        if self._train_task < 1:
            return torch.tensor(0.0, device=self.device)

        return weighted_l2_reg(
            params=trainable_params(self._model),
            anchor_params=self._anchor_params,
            weight=self._importance,
            device=self.device,
        )

    def __str__(self) -> str:
        return f"MAS(lambda_={self._lambda}, alpha={self._alpha})"
