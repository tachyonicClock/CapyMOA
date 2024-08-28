from __future__ import annotations

from capymoa.base import MOAClassifier



class NaiveBayes(MOAClassifier):
    """Naive Bayes incremental learner.
    Performs classic Bayesian prediction while making the naive assumption that all inputs are independent. Naive Bayes is a classifier algorithm known for its simplicity and low computational cost. Given n different classes, the trained Naive Bayes classifier predicts, for every unlabeled instance I, the class C to which it belongs with high accuracy.

    :param random_seed: The random seed passed to the MOA learner, defaults to 0.
    """

    def __init__(self, random_seed: int = 0):
        super(NaiveBayes, self).__init__(
            "moa.classifiers.bayes.NaiveBayes", "", random_seed
        )
