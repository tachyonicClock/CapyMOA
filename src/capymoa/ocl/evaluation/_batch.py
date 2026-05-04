"""Batch-level train/test helpers for OCL evaluation."""

import numpy as np
from torch import Tensor

from capymoa.base import BatchClassifier, Classifier
from capymoa.instance import Instance, LabeledInstance
from capymoa.type_alias import LabelIndex


def _abstain_prediction_uniform(rng: np.random.Generator, n_classes: int) -> LabelIndex:
    return int(rng.integers(0, n_classes))


def _batch_test(rng: np.random.Generator, learner: Classifier, x: Tensor) -> np.ndarray:
    """Test a batch of instances using the learner."""
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    if isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        return learner.batch_predict(x).cpu().numpy()
    else:
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


def _batch_train(learner: Classifier, x: Tensor, y: Tensor):
    """Train a batch of instances using the learner."""
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    if isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        y = y.to(dtype=learner.y_dtype, device=learner.device)
        learner.batch_train(x, y)
    else:
        for i in range(batch_size):
            instance = LabeledInstance.from_array(
                learner.schema, x[i].numpy(), int(y[i].item())
            )
            learner.train(instance)
