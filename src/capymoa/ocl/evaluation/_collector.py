"""Statistics collector used while evaluating OCL learners."""

from typing import Optional

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from capymoa.base.events import Handler, Dispatcher, LogScalar
from capymoa.evaluation.results import PrequentialResults
from capymoa.ocl.evaluation import events
from capymoa.type_alias import LabelIndex

from ._metrics import (
    OCLMetrics,
    _backwards_transfer,
    _forwards_transfer,
    _get_ttt_windowed_task_index,
)


class _OCLEvaluator(Handler):
    """A builder used to collect statistics during online continual learning evaluation."""

    cm: torch.Tensor
    """Confusion 'Matrix' of shape:
    ``(train_task_id, eval_step_id, test_task_id, y_true, y_pred)``.
    """

    def __init__(self, task_count: int, eval_step_count: int, class_count: int):
        self.task_count = task_count
        self.class_count = class_count
        self.seen_tasks = 0
        self.step_count = eval_step_count
        self.cm = torch.zeros(
            task_count,
            eval_step_count,
            task_count,
            class_count,
            class_count,
            dtype=torch.int,
        )

    def holdout_update(
        self,
        train_task_id: int,
        eval_step_id: int,
        test_task_id: int,
        y_true: LabelIndex,
        y_pred: Optional[LabelIndex],
    ):
        """Record a prediction when using holdout evaluation."""
        if y_pred is not None:
            self.cm[train_task_id, eval_step_id, test_task_id, y_true, y_pred] += 1
        # TODO: handle missing predictions

    def attach_with(self, source: Dispatcher) -> "Handler":
        self.dispatcher = source

        source.subscribe(events.TestBatchPredict, self._on_eval_batch)
        source.subscribe(events.TrainBatchPredict, self._on_train_batch)
        source.subscribe(events.TestEnd, self._on_test_end)
        return self

    def log_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.dispatcher.notify(
            LogScalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        )

    def _on_train_batch(self, event: events.TrainBatchPredict) -> None:
        accuracy = (event.y == event.y_pred).float().mean().item()
        ce = cross_entropy(event.y_proba, event.y).item()
        self.log_scalar(
            f"ocl_train/accuracy_{event.train_task:02d}", accuracy, event.global_step
        )
        self.log_scalar(
            f"ocl_train/cross_entropy_{event.train_task:02d}", ce, event.global_step
        )

    def _on_eval_batch(self, event: events.TestBatchPredict) -> None:
        for y_true, y_pred in zip(event.y, event.y_hat, strict=True):
            self.holdout_update(
                event.train_task,
                event.continual_eval,
                event.test_task,
                int(y_true.item()),
                int(y_pred.item()),
            )

    def _on_test_end(self, event: events.TestEnd) -> None:
        correct = self.cm.diagonal(dim1=3, dim2=4).sum(-1)
        total = self.cm.sum((3, 4))
        anytime_acc = correct / total  # (train task, step_id, test task)

        # Compute the accuracy on the current training task at the current evaluation
        # step, and log it as a scalar.
        for t in range(self.task_count):
            self.log_scalar(
                f"ocl_test/accuracy_task_{t:02d}",
                anytime_acc[event.train_task, event.continual_eval, t].item(),
                event.global_step,
            )

        # Compute the average accuracy on seen tasks at the current evaluation step, and
        # log it as a scalar.
        self.log_scalar(
            "ocl_test/accuracy_seen",
            anytime_acc[event.train_task, event.continual_eval, : event.train_task + 1]
            .mean()
            .item(),
            event.global_step,
        )

        # Compute the average accuracy on all tasks at the current evaluation step, and
        # log it as a scalar.
        self.log_scalar(
            "ocl_test/accuracy_all",
            anytime_acc[event.train_task, event.continual_eval, :].mean().item(),
            event.global_step,
        )

    def build(
        self, ttt: PrequentialResults, boundary_instances: torch.Tensor
    ) -> OCLMetrics:
        """Creates metrics using collected statistics."""
        correct = self.cm.diagonal(dim1=3, dim2=4).sum(-1)
        total = self.cm.sum((3, 4))
        anytime_acc = correct / total  # (train task, step_id, test task)
        accuracy_matrix = anytime_acc[:, -1, :]  # (train task, test task)

        anytime_accuracy_seen = torch.zeros(self.task_count, self.step_count)
        anytime_accuracy_all = torch.zeros(self.task_count, self.step_count)
        for t_train in range(self.task_count):
            for s_step in range(self.step_count):
                anytime_accuracy_seen[t_train, s_step] = (
                    anytime_acc[t_train, s_step, : t_train + 1].mean().item()
                )
                anytime_accuracy_all[t_train, s_step] = (
                    anytime_acc[t_train, s_step, :].mean().item()
                )

        def _accuracy_seen(t: int) -> float:
            return accuracy_matrix[t, : t + 1].mean().item()

        def _accuracy_all(t: int) -> float:
            return accuracy_matrix[t, :].mean().item()

        tasks = np.arange(self.task_count, dtype=int)

        accuracy_seen = np.array([_accuracy_seen(t) for t in tasks])
        accuracy_all = np.array([_accuracy_all(t) for t in tasks])
        boundaries = boundary_instances.numpy()

        ttt_windowed_task_index = None
        if ttt.windowed is not None:
            ttt_windowed_task_index = _get_ttt_windowed_task_index(
                boundaries, ttt.windowed.window_size
            )
            assert len(ttt_windowed_task_index) == len(ttt.windowed.accuracy())

        return OCLMetrics(
            accuracy_seen=accuracy_seen,
            accuracy_all=accuracy_all,
            accuracy_final=_accuracy_all(self.task_count - 1),
            accuracy_all_avg=np.mean(accuracy_all),
            accuracy_seen_avg=np.mean(accuracy_seen),
            accuracy_matrix=accuracy_matrix.numpy(),
            class_cm=self.cm[:, -1].sum(1).numpy(),
            anytime_accuracy_all=anytime_accuracy_all.flatten().numpy(),
            anytime_accuracy_seen=anytime_accuracy_seen.flatten().numpy(),
            anytime_accuracy_all_avg=anytime_accuracy_all.mean().item(),
            anytime_accuracy_seen_avg=anytime_accuracy_seen.mean().item(),
            anytime_task_index=np.linspace(
                0, self.task_count, self.step_count * self.task_count + 1
            )[1:],
            task_index=np.arange(self.task_count) + 1,
            anytime_accuracy_matrix=anytime_acc.flatten(end_dim=1).numpy(),
            backward_transfer=_backwards_transfer(accuracy_matrix),
            forward_transfer=_forwards_transfer(accuracy_matrix),
            ttt=ttt,
            boundaries=boundaries,
            ttt_windowed_task_index=ttt_windowed_task_index,
            n_tasks=self.task_count,
            n_continual_evaluations=self.step_count,
            n_classes=self.class_count,
        )
