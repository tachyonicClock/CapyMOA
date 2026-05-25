"""Batch-level train/test helpers for OCL evaluation."""

import numpy as np
from torch import Tensor

from capymoa.base import BatchClassifier, Classifier
from capymoa.instance import Instance, LabeledInstance
from capymoa.type_alias import LabelIndex
from typing import Sequence
import torch


def _abstain_prediction_uniform(rng: np.random.Generator, n_classes: int) -> LabelIndex:
    return int(rng.integers(0, n_classes))


@torch.no_grad()
def _batch_test(
    rng: np.random.Generator, learner: Classifier, x: Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Test a batch and return integer predictions and class probabilities."""
    batch_size = x.shape[0]
    n_classes = learner.schema.get_num_classes()
    if isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        yb_logits = learner.batch_predict_proba(x).cpu().numpy()
        yb_pred = yb_logits.argmax(axis=1).astype(int)
        return yb_pred, yb_logits
    else:
        x = x.view(batch_size, -1)
        yb_pred = np.zeros(batch_size, dtype=int)
        yb_logits = np.zeros((batch_size, n_classes), dtype=np.float64)
        for i in range(batch_size):
            instance = Instance.from_array(learner.schema, x[i].numpy())
            y_logits = learner.predict_proba(instance)
            if y_logits is None or len(y_logits) != n_classes:
                # The classifier cannot create a probability distribution or returns an
                # invalid one, so we fall back to predicting a single class or
                # abstaining, depending on whether a prediction is available.

                # TODO: This should be simplified once this issue is resolved:
                # https://github.com/adaptive-machine-learning/backlog/issues/96
                y_pred = learner.predict(instance)
                if y_pred is None:
                    # Abstain
                    yb_pred[i] = _abstain_prediction_uniform(rng, n_classes)
                    yb_logits[i] = np.full(n_classes, 1 / n_classes)
                else:
                    # Predict but no probabilities
                    yb_pred[i] = int(y_pred)
                    yb_logits[i][int(y_pred)] = 1.0
            else:
                yb_pred[i] = int(np.argmax(y_logits))
                yb_logits[i] = y_logits
        return yb_pred, yb_logits


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
