from capymoa.base import MOAClassifier


class KNN(MOAClassifier):
    """
    The default number of neighbors (k) is set to 3 instead of 10 (as in MOA)
    """

    def __init__(self, k: int = 3, window_size: int = 1000):
        """Construct a k-Nearest Neighbors (kNN) Classifier

        :param k: Number of neighbors to consider, defaults to 3
        :param window_size: The size of the window for the kNN classifier, defaults to 1000
        """
        cli = [f"-k {k}", f"-w {window_size}"]
        super().__init__(
            java_learner_class="moa.classifiers.lazy.kNN",
            random_seed=0,
            CLI=" ".join(cli),
        )
