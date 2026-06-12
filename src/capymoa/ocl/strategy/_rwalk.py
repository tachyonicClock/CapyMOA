"""Riemannian Walk (RWalk)

References:
* Original:  https://github.com/facebookresearch/agem/tree/main
* FACIL:     https://github.com/mmasana/FACIL/blob/e09d2c83320a1aa945a6157d4875437515824dc9/src/approach/r_walk.py
* Avalanche: https://github.com/ContinualAI/avalanche/blob/eb075be393e1f458b2c352514ff6c17b5a2c0f4e/avalanche/training/plugins/rwalk.py#L12

"""

from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.functional import relu

from capymoa.base import BatchClassifier
from capymoa.base.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.util._buffer import BufferList
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.stream._stream import Schema

from ._ewc import trainable_params, weighted_l2_reg

EPSILON = torch.finfo(torch.float32).eps
NEG_INF = float("-inf")


def _rwalk_update_importances(
    importances: Sequence[Tensor], params: Sequence[Tensor], alpha: float
):
    """Update the EMA of squared unregularised gradients."""
    for importance, param in zip(importances, params, strict=True):
        importance.mul_(1 - alpha).add_(param.square(), alpha=alpha)


@torch.no_grad()
def _rwalk_update_score(
    scores: Sequence[Tensor],
    params: Sequence[Tensor],
    losses: Sequence[Tensor],
    importances: Sequence[Tensor],
    old_params: Sequence[Tensor],
):
    """Accumulate checkpoint scores using the RWalk denominator."""
    for score, param, loss, imp, old_param in zip(
        scores, params, losses, importances, old_params, strict=True
    ):
        assert score.shape == param.shape == loss.shape == imp.shape == old_param.shape
        score.add_(loss / (0.5 * imp * (param - old_param).square() + EPSILON))


def _rwalk_update_task_scores(
    scores: Sequence[Tensor],
    old_scores: Sequence[Tensor],
):
    """Blend task scores with the scores from previous tasks."""
    for score, old_score in zip(scores, old_scores, strict=True):
        score.add_(old_score).mul_(0.5)


def _set_penalties(
    penalties: Sequence[Tensor],
    importances: Sequence[Tensor],
    scores: Sequence[Tensor],
):
    """Combine importance and positive scores into the RWalk penalty weights."""
    max_score = max(s.max() for s in scores).clamp_min(EPSILON)
    max_importance = max(i.max() for i in importances).clamp_min(EPSILON)

    for penalty, importance, score in zip(penalties, importances, scores, strict=True):
        penalty.copy_(importance / max_importance + relu(score) / max_score)


@torch.no_grad()
def _copy_params_(module: nn.Module, dst: Sequence[Tensor]) -> None:
    """Copy trainable parameters into a pre-allocated list."""
    for param, dst_tensor in zip(trainable_params(module), dst, strict=True):
        dst_tensor.copy_(param.detach())


@torch.no_grad()
def _copy_grads_(module: nn.Module, dst: Sequence[Tensor]) -> None:
    """Copy parameter gradients into a pre-allocated list."""
    for param, dst_tensor in zip(trainable_params(module), dst, strict=True):
        assert param.grad is not None
        dst_tensor.copy_(param.grad.detach())


@torch.no_grad()
def _accumulate_loss_(
    losses: Sequence[Tensor],
    params: Iterable[Tensor],
    old_params: Sequence[Tensor],
    grads: Sequence[Tensor],
) -> None:
    """Update the first-order approximation of the loss variation."""
    for loss, param, old_param, grad in zip(
        losses, params, old_params, grads, strict=True
    ):
        loss.sub_(grad * (param.detach() - old_param))


@torch.no_grad()
def _zero_(buffers: Sequence[Tensor]) -> None:
    """Zero a sequence of tensors in place."""
    for buffer in buffers:
        buffer.zero_()


class RWalk(BatchClassifier, nn.Module, Handler):
    """Riemannian Walk learner.

    RWalk augments task loss with a weighted quadratic penalty like EWC, while the
    penalty weights combine a moving average of squared gradients with trajectory
    scores that estimate how sensitive the loss is to parameter updates.
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        lambda_: float,
        alpha: float = 0.9,
        delta_t: int = 10,
        device: torch.device = torch.device("cpu"),
        task_mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__(schema, 0)
        nn.Module.__init__(self)
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if delta_t < 1:
            raise ValueError("delta_t must be at least 1.")

        self.device = device

        # Hyperparameters
        self._lambda = lambda_
        self._alpha = alpha
        self._delta_t = delta_t

        # Modules
        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()

        # Persistent regularisation buffers
        params = list(trainable_params(model))
        self._anchor_params = BufferList([param.clone().detach() for param in params])
        self._penalties = BufferList([torch.zeros_like(param) for param in params])
        self._task_scores = BufferList([torch.zeros_like(param) for param in params])

        # Task-local running statistics
        self._iter_importances = BufferList([torch.zeros_like(p) for p in params])
        self._iter_grads = BufferList([torch.zeros_like(param) for param in params])
        self._pre_step_params = BufferList([torch.zeros_like(p) for p in params])
        self._checkpoint_params = BufferList([p.clone().detach() for p in params])
        self._checkpoint_losses = BufferList([torch.zeros_like(p) for p in params])
        self._checkpoint_scores = BufferList([torch.zeros_like(p) for p in params])

        # Task tracking
        self._train_task = 0
        self._test_task = 0
        self._train_steps = 0
        self._steps_since_checkpoint = 0
        self._has_completed_task = False
        if task_mask is None:
            self._task_mask = None
        else:
            self._task_mask = nn.Buffer(task_mask)

        self.to(device)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._model.train()

        self._optimiser.zero_grad()
        y_hat = self._train_forward(x)
        loss = self._criterion(y_hat, y)
        loss.backward()

        _copy_params_(self._model, self._pre_step_params)
        _copy_grads_(self._model, self._iter_grads)
        _rwalk_update_importances(self._iter_importances, self._iter_grads, self._alpha)

        if self._train_task > 0:
            reg_loss = self._lambda * weighted_l2_reg(
                trainable_params(self._model),
                self._anchor_params,
                self._penalties,
                device=self.device,
            )
            reg_loss.backward()

        self._optimiser.step()

        _accumulate_loss_(
            self._checkpoint_losses,
            trainable_params(self._model),
            self._pre_step_params,
            self._iter_grads,
        )

        self._train_steps += 1
        self._steps_since_checkpoint += 1
        if self._steps_since_checkpoint >= self._delta_t:
            self._flush_checkpoint_scores()

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
            self._finalise_previous_task()
        self._train_task = event.train_task

    def on_test_task(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    @torch.no_grad()
    def _finalise_previous_task(self) -> None:
        """Turn the just-finished task statistics into penalties for the next task."""
        if self._steps_since_checkpoint > 0:
            self._flush_checkpoint_scores()

        if self._has_completed_task:
            _rwalk_update_task_scores(self._checkpoint_scores, self._task_scores)

        for task_score, checkpoint_score in zip(
            self._task_scores, self._checkpoint_scores, strict=True
        ):
            task_score.copy_(checkpoint_score)

        _set_penalties(self._penalties, self._iter_importances, self._task_scores)
        _copy_params_(self._model, self._anchor_params)

        _zero_(self._checkpoint_losses)
        _zero_(self._checkpoint_scores)
        _zero_(self._iter_importances)
        _copy_params_(self._model, self._checkpoint_params)
        self._steps_since_checkpoint = 0
        self._has_completed_task = True

    @torch.no_grad()
    def _flush_checkpoint_scores(self) -> None:
        """Commit the current checkpoint segment into the task scores."""
        _rwalk_update_score(
            scores=self._checkpoint_scores,
            params=list(trainable_params(self._model)),
            losses=self._checkpoint_losses,
            importances=self._iter_importances,
            old_params=self._checkpoint_params,
        )
        _zero_(self._checkpoint_losses)
        _copy_params_(self._model, self._checkpoint_params)
        self._steps_since_checkpoint = 0

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

    def __str__(self) -> str:
        return f"RWalk(lambda_={self._lambda}, alpha={self._alpha}, delta_t={self._delta_t})"
