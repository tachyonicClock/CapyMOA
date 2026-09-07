"""Evaluate online continual learning in classification tasks."""

from . import events
from ._loop import ocl_train_eval_loop
from ._metrics import OCLMetrics

__all__ = ["OCLMetrics", "events", "ocl_train_eval_loop"]
