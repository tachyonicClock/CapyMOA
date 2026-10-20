"""Drift detection.

Drift detection identifies changes in the underlying data distribution over
time. In data stream learning, detecting drift promptly allows learners to
adapt or reset so that predictions remain accurate as conditions change.
"""

from capymoa.drift import evaluate
from capymoa.drift._evaluate import evaluate_drift
from capymoa.drift.evaluate import (
    DriftCumulative,
    DriftDetectionMetrics,
    DriftResults,
    EvaluateDriftDetector,
)

__all__ = [
    "DriftCumulative",
    "DriftDetectionMetrics",
    "DriftResults",
    "EvaluateDriftDetector",
    "evaluate",
    "evaluate_drift",
]
