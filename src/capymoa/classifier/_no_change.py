from __future__ import annotations

from capymoa.base import (
    MOAClassifier,
)


class NoChange(MOAClassifier):
    """NoChange classifier.

    Always predicts the last class seen.

    Example usages:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import NoChange
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = NoChange(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    85.9
    """

    def __init__(
        self,
    ):
        """NoChange class classifier."""

        super(NoChange, self).__init__(
            java_learner_class="moa.classifiers.functions.NoChange",
            CLI="",
        )
