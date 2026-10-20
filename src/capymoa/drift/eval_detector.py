"""Deprecated. Use :mod:`capymoa.drift.evaluate` instead."""

import warnings

from capymoa.drift.evaluate import DriftDetectionMetrics, EvaluateDriftDetector

warnings.warn(
    "capymoa.drift.eval_detector is deprecated and will be removed in a future "
    "release. Import from capymoa.drift.evaluate instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DriftDetectionMetrics", "EvaluateDriftDetector"]
