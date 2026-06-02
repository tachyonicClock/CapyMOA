from typing import Iterable, Optional

import torch
from torch import Tensor, nn

from capymoa.base import BatchClassifier
from capymoa.base.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.util._buffer import BufferList
from capymoa.stream._stream import Schema

NEG_INF = float("-inf")


def weighted_l2_reg(
    params: Iterable[Tensor],
    anchor_params: Iterable[Tensor],
    importances: Iterable[Tensor],
    device: torch.device,
) -> Tensor:
    """Compute an SI-style weighted L2 regularisation term."""
    l2 = torch.tensor(0.0, device=device)
    for param, anchor_param, importance in zip(
        params, anchor_params, importances, strict=True
    ):
        assert param.shape == anchor_param.shape
        l2 += (importance * (param - anchor_param) ** 2).sum()
    return l2 / 2.0


class SI(BatchClassifier, nn.Module, Handler):
    """Synaptic Intelligence learner.

    Synaptic Intelligence (SI) is a regularisation-based continual learning strategy
    that accumulates per-parameter importance online from optimization trajectories,
    then penalises changes to parameters that were important for previous tasks [#f1]_.

    ..  [#f1] Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through
        Synaptic Intelligence. International Conference on Machine Learning, 3987–3995.
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        lambda_: float,
        eps: float = 1e-7,
        device: torch.device = torch.device("cpu"),
        mask_test: bool = False,
        mask_train: bool = False,
        task_mask: Optional[Tensor] = None,
    ) -> None:
        """Construct an SI learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param lambda_: Weight of the SI regularisation term.
        :param eps: Damping value used in SI importance consolidation.
        :param device: Compute device.
        :param mask_test: Whether to apply per-task masking during testing. This is a
            task incremental scenario.
        :param mask_train: Whether to apply per-task masking during training. This is
            also known as the labels trick.
        :param task_mask: Optional per-task mask applied to output logits.
        :raises ValueError: If task-specific masking is requested without ``task_mask``.
        """
        super().__init__(schema, 0)
        nn.Module.__init__(self)
        if (mask_train or mask_test) and task_mask is None:
            raise ValueError(
                "Task schedule must be provided for task incremental or labels trick scenarios."
            )
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be greater than zero.")

        self.device = device

        # Hyperparameters
        self._lambda = lambda_
        self._eps = eps
        self._mask_train = mask_train
        self._mask_test = mask_test

        # Modules
        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()

        # Buffers for SI regularisation
        self._anchor_params = BufferList(
            [param.clone().detach() for param in model.parameters()]
        )
        self._omegas = BufferList(
            [torch.zeros_like(param) for param in model.parameters()]
        )
        self._trajectory = BufferList(
            [torch.zeros_like(param) for param in model.parameters()]
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
        self._model.train()
        self._optimiser.zero_grad()

        y_hat = self._train_forward(x)
        loss = self._criterion(y_hat, y)
        total_loss = loss + self._lambda * self._regularisation_loss()
        total_loss.backward()

        with torch.no_grad():
            old_params = [param.detach().clone() for param in self._model.parameters()]
            grads = []
            for param in self._model.parameters():
                if param.grad is None:
                    raise ValueError(
                        "Parameter gradients must be computed before updating SI trajectory."
                    )
                grads.append(param.grad.detach().clone())

        self._optimiser.step()
        self._accumulate_trajectory(old_params, grads)

    @torch.no_grad()
    def _accumulate_trajectory(
        self, old_params: list[Tensor], grads: list[Tensor]
    ) -> None:
        """Accumulate SI path integrals from parameter updates."""
        for param, old_param, grad, trajectory in zip(
            self._model.parameters(), old_params, grads, self._trajectory, strict=True
        ):
            delta = param.detach() - old_param
            trajectory.add_(-grad * delta)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        y_hat = self._test_forward(x)
        return torch.softmax(y_hat, dim=1)

    def attach_with(self, source: Dispatcher) -> None:
        source.subscribe(TrainTaskBegin, self.on_train_task)
        source.subscribe(TestTaskBegin, self.on_test_task)

    def on_train_task(self, event: TrainTaskBegin) -> None:
        if event.train_task > 0:
            self._update_omegas()
            self._update_anchor_params()
            self._reset_trajectory()
        self._train_task = event.train_task

    def on_test_task(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    @torch.no_grad()
    def _update_omegas(self) -> None:
        """Consolidate trajectory information into parameter importances."""
        for omega, trajectory, param, anchor_param in zip(
            self._omegas,
            self._trajectory,
            self._model.parameters(),
            self._anchor_params,
            strict=True,
        ):
            delta = param.detach() - anchor_param
            contribution = trajectory / (delta.pow(2) + self._eps)
            omega.add_(contribution)
            omega.clamp_(min=0.0)

    @torch.no_grad()
    def _update_anchor_params(self) -> None:
        """Update anchored parameters to the current model weights."""
        for param, anchor_param in zip(
            self._model.parameters(), self._anchor_params, strict=True
        ):
            anchor_param.copy_(param.detach())

    @torch.no_grad()
    def _reset_trajectory(self) -> None:
        for i in range(len(self._trajectory)):
            self._trajectory[i].zero_()

    def _test_forward(self, x: Tensor) -> Tensor:
        """Compute logits for inference, optionally applying a test-task mask."""
        y_hat = self._model(x)
        if self._task_mask is not None and self._mask_test:
            y_hat = y_hat.masked_fill(self._task_mask[self._test_task] == 0, NEG_INF)
        return y_hat

    def _train_forward(self, x: Tensor) -> Tensor:
        """Compute logits for training, optionally applying a train-task mask."""
        y_hat = self._model(x)
        if self._task_mask is not None and self._mask_train:
            y_hat = y_hat.masked_fill(self._task_mask[self._train_task] == 0, NEG_INF)
        return y_hat

    def _regularisation_loss(self) -> Tensor:
        """Return the SI regularisation loss for the current task."""
        if self._train_task < 1:
            return torch.tensor(0.0, device=self.device)
        return weighted_l2_reg(
            self._model.parameters(),
            self._anchor_params,
            self._omegas,
            device=self.device,
        )

    def __str__(self) -> str:
        return f"SI(lambda_={self._lambda}, eps={self._eps})"
