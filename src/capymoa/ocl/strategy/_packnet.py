from torch.func import functional_call, grad
from torch import Tensor, nn
import torch
from typing import Dict, Optional, override

from capymoa.base import BatchClassifier
from capymoa.base.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin, TrainTaskEnd
from capymoa.stream import Schema

TensorDict = Dict[str, Tensor]


def masked_forward(
    module: nn.Module, params: TensorDict, params_mask: TensorDict, *args, **kwargs
) -> Tensor:
    masked_params = {k: params[k] * params_mask[k] for k in params}
    return functional_call(module, masked_params, args, kwargs)


class PackNet(BatchClassifier, Handler):
    """PackNet Classifier.

    PackNet [#f0]_ trains a single network across tasks while freezing previously
    allocated parameters and pruning trainable parameters by magnitude at task
    boundaries.

    .. [#f0] Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a
        Single Network by Iterative Pruning. 2018 IEEE/CVF Conference on Computer Vision
        and Pattern Recognition, 7765–7773. https://doi.org/10.1109/CVPR.2018.00810
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        prune_fraction: float = 0.5,
        mask_test: bool = True,
        mask_train: bool = False,
        ensemble_output: bool = False,
        task_mask: Optional[Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Construct a PackNet continual learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param prune_fraction: Fraction of currently trainable parameters to prune at
            each task end, defaults to ``0.5``.
        :param mask_test: Whether to use task-specific parameter masks during
            inference, defaults to ``True``.
        :param mask_train: Whether to use task-specific output masking during
            training (labels trick), defaults to ``False``.
        :param ensemble_output: Whether to ensemble predictions by averaging logits
            from all stored task masks during inference, defaults to ``False``.
        :param task_mask: Optional per-task class mask of shape
            ``(num_tasks, num_classes)`` applied to logits in task-incremental
            settings, defaults to ``None``.
        :param device: Compute device, defaults to ``torch.device("cpu")``.
        """
        super().__init__(schema, 0)
        if prune_fraction < 0.0 or prune_fraction > 1.0:
            raise ValueError("prune_fraction must be in [0, 1].")
        if (mask_train or mask_test) and task_mask is None:
            raise ValueError(
                "Task schedule must be provided for task incremental or labels trick scenarios."
            )

        self.device = device
        self._model = model.to(device)
        self._optimiser = optimiser
        self._criterion = torch.nn.CrossEntropyLoss()

        self._prune_fraction = prune_fraction
        self._mask_train = mask_train
        self._mask_test = mask_test
        self._ensemble_output = ensemble_output

        self._train_task = 0
        self._test_task = 0
        if task_mask is None:
            self._task_mask = None
        else:
            self._task_mask = task_mask.to(device)

        named_params = dict(self._model.named_parameters())
        self._frozen_mask: Dict[str, Tensor] = {
            name: torch.zeros_like(param, dtype=torch.bool, device=self.device)
            for name, param in named_params.items()
        }
        self._prunable_mask: Dict[str, Tensor] = {
            name: torch.ones_like(param, dtype=torch.bool, device=self.device)
            for name, param in named_params.items()
        }

        # Stores frozen masks at each task boundary for task-specific inference.
        self._task_param_masks: Dict[int, Dict[str, Tensor]] = {}

    def attach_with(self, source: Dispatcher) -> None:
        source.subscribe(TrainTaskBegin, self.on_train_task_begin)
        source.subscribe(TrainTaskEnd, self.on_train_task_end)
        source.subscribe(TestTaskBegin, self.on_test_task_begin)

    def on_train_task_begin(self, event: TrainTaskBegin) -> None:
        self._train_task = event.train_task

    def on_train_task_end(self, event: TrainTaskEnd) -> None:
        self._finalise_task(event.train_task)

    def on_test_task_begin(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    def _named_params(self) -> Dict[str, Tensor]:
        return {name: param for name, param in self._model.named_parameters()}

    def _current_training_mask(self) -> Dict[str, Tensor]:
        # During training all parameters are used in forward pass, while frozen
        # parameters are protected by gradient masking and post-step restoration.
        return {
            name: torch.ones_like(param, dtype=param.dtype, device=self.device)
            for name, param in self._named_params().items()
        }

    def _task_forward_mask(self, task: int, train: bool) -> Dict[str, Tensor]:
        if train and self._mask_train:
            return self._task_param_masks.get(task, self._current_training_mask())
        if (not train) and self._mask_test:
            return self._task_param_masks.get(task, self._current_training_mask())
        return self._current_training_mask()

    def _apply_output_task_mask(self, logits: Tensor, task: int, train: bool) -> Tensor:
        if self._task_mask is None:
            return logits
        if train and self._mask_train:
            return self._task_mask[task] * logits
        if (not train) and self._mask_test:
            return self._task_mask[task] * logits
        return logits

    def _compute_loss(
        self,
        params: TensorDict,
        params_mask: TensorDict,
        x: Tensor,
        y: Tensor,
        task: int,
        train: bool,
    ) -> Tensor:
        logits = masked_forward(self._model, params, params_mask, x)
        logits = self._apply_output_task_mask(logits, task=task, train=train)
        return self._criterion(logits, y)

    @torch.no_grad()
    def _restore_frozen_params(self, previous_params: Dict[str, Tensor]) -> None:
        for name, param in self._named_params().items():
            frozen = self._frozen_mask[name]
            param[frozen] = previous_params[name][frozen]

    @torch.no_grad()
    def _prune_current_trainable_weights(self) -> Dict[str, Tensor]:
        pruned_mask: Dict[str, Tensor] = {
            name: torch.zeros_like(mask, dtype=torch.bool)
            for name, mask in self._frozen_mask.items()
        }
        if self._prune_fraction <= 0.0:
            return pruned_mask

        for name, param in self._named_params().items():
            eligible = (~self._frozen_mask[name]) & self._prunable_mask[name]
            num_eligible = int(eligible.sum().item())
            if num_eligible <= 0:
                continue

            k = int(self._prune_fraction * num_eligible)
            if k <= 0:
                continue
            if k >= num_eligible:
                to_prune = eligible
            else:
                flat_abs = param.detach().abs()[eligible]
                prune_local_idx = torch.topk(flat_abs, k, largest=False).indices
                eligible_idx = eligible.nonzero(as_tuple=False)
                selected_idx = eligible_idx[prune_local_idx]
                to_prune = torch.zeros_like(eligible)
                to_prune[tuple(selected_idx.t())] = True

            param[to_prune] = 0.0
            pruned_mask[name] = to_prune

        return pruned_mask

    @torch.no_grad()
    def _finalise_task(self, task: int) -> None:
        pruned_now = self._prune_current_trainable_weights()
        new_frozen_mask: Dict[str, Tensor] = {}

        for name, param in self._named_params().items():
            prev_frozen = self._frozen_mask[name]
            trainable = ~prev_frozen

            committed_prunable = (
                trainable & self._prunable_mask[name] & (~pruned_now[name])
            )
            committed_non_prunable = torch.zeros_like(prev_frozen)

            new_frozen_mask[name] = (
                prev_frozen | committed_prunable | committed_non_prunable
            )

        self._frozen_mask = new_frozen_mask

        # Task-specific inference replays the exact frozen superposition for this task.
        self._task_param_masks[task] = {
            name: mask.to(dtype=self._named_params()[name].dtype)
            for name, mask in self._frozen_mask.items()
        }

    @override
    def batch_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._model.train()
        self._optimiser.zero_grad()

        x = x.to(self.device)
        y = y.to(self.device)

        params = self._named_params()
        params_mask = self._task_forward_mask(self._train_task, train=True)
        grads = grad(self._compute_loss)(
            params, params_mask, x, y, self._train_task, True
        )

        # Save current parameters to restore frozen ones after the step. This is
        # necessary because some optimisers may update frozen parameters due to momentum
        # or weight decay.
        previous_params = {
            name: param.detach().clone() for name, param in params.items()
        }
        for name, param in params.items():
            trainable = (~self._frozen_mask[name]).to(dtype=param.dtype)
            param.grad = grads[name].detach() * trainable

        self._optimiser.step()
        self._restore_frozen_params(previous_params)

    @override
    def batch_predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self._model.eval()
        x = x.to(self.device)
        params = self._named_params()
        if self._ensemble_output and len(self._task_param_masks) > 0:
            probs = []
            for task_id in self._task_param_masks:
                params_mask = self._task_forward_mask(task_id, train=False)
                logits = masked_forward(self._model, params, params_mask, x)
                logits = self._apply_output_task_mask(logits, task=task_id, train=False)
                # Apply softmax BEFORE appending
                probs.append(torch.softmax(logits, dim=1))
            return torch.stack(probs, dim=0).mean(dim=0)
        else:
            params_mask = self._task_forward_mask(self._test_task, train=False)
            logits = masked_forward(self._model, params, params_mask, x)
            logits = self._apply_output_task_mask(
                logits, task=self._test_task, train=False
            )
            return torch.softmax(logits, dim=1)

    def __str__(self) -> str:
        return (
            f"PackNet(prune_fraction={self._prune_fraction}, "
            f"ensemble_output={self._ensemble_output})"
        )
