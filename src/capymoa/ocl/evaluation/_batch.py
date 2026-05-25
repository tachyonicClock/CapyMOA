"""Batch-level train/test helpers for OCL evaluation."""

import numpy as np
from torch import Tensor

from capymoa.base import BatchClassifier, Classifier
from capymoa.instance import Instance, LabeledInstance
from capymoa.type_alias import LabelIndex
from typing import Sequence


def _abstain_prediction_uniform(rng: np.random.Generator, n_classes: int) -> LabelIndex:
    return int(rng.integers(0, n_classes))


def _batch_test(rng: np.random.Generator, learner: Classifier, x: Tensor) -> np.ndarray:
    """Test a batch of instances using the learner."""
    batch_size = x.shape[0]
    if isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        return learner.batch_predict(x).cpu().numpy()
    else:
        x = x.view(batch_size, -1)
        yb_pred = np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            instance = Instance.from_array(learner.schema, x[i].numpy())
            y_pred = learner.predict(instance)
            if y_pred is None:
                y_pred = _abstain_prediction_uniform(
                    rng, learner.schema.get_num_classes()
                )
            yb_pred[i] = y_pred
        return yb_pred


def _batch_train(learner: Classifier, x: Tensor, y: Tensor, x_shape: Sequence[int]):
    """Train a batch of instances using the learner."""
    bs = x.shape[0]
    if isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device).view(bs, *x_shape)
        y = y.to(dtype=learner.y_dtype, device=learner.device)
        learner.batch_train(x, y)
    else:
        x = x.view(bs, -1)
        for i in range(bs):
            instance = LabeledInstance.from_array(
                learner.schema, x[i].numpy(), int(y[i].item())
            )
            learner.train(instance)
