from __future__ import annotations

from capymoa.base import (
    MOAClassifier,
)


class MajorityClass(MOAClassifier):
    """Majority class classifier.

    Always predicts the class that has been observed most frequently the in the training data.

    Example usages:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import MajorityClass
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = MajorityClass(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    50.2
    """

    def __init__(
        self,
    ):
        """Majority class classifier.

        :param schema: The schema of the stream.
        """
        super(MajorityClass, self).__init__(
            java_learner_class="moa.classifiers.functions.MajorityClass",
            CLI="",
        )
