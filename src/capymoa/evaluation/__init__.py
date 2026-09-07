"""Evaluation procedures and evaluators for CapyMOA learners.

This module provides prequential evaluation functions and evaluator classes for
classification, regression, prediction interval, anomaly detection, and
clustering tasks.
"""

from . import results
from .evaluation import (
    AnomalyDetectionEvaluator,
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
    ClusteringEvaluator,
    PredictionIntervalEvaluator,
    PredictionIntervalWindowedEvaluator,
    RegressionEvaluator,
    RegressionWindowedEvaluator,
    prequential_evaluation,
    prequential_evaluation_anomaly,
    prequential_evaluation_multiple_learners,
    prequential_ssl_evaluation,
)

__all__ = [
    "AnomalyDetectionEvaluator",
    "ClassificationEvaluator",
    "ClassificationWindowedEvaluator",
    "ClusteringEvaluator",
    "PredictionIntervalEvaluator",
    "PredictionIntervalWindowedEvaluator",
    "RegressionEvaluator",
    "RegressionWindowedEvaluator",
    "prequential_evaluation",
    "prequential_evaluation_anomaly",
    "prequential_evaluation_multiple_learners",
    "prequential_ssl_evaluation",
    "results",
]
