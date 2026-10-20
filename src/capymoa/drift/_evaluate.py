import time
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np

from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.evaluate import DriftResults, EvaluateDriftDetector


def evaluate_drift(
    values: Sequence[float] | np.ndarray,
    detector: BaseDriftDetector,
    drifts: Sequence[int] | Sequence[tuple[int, int]] | np.ndarray,
    max_delay: int,
    rate_period: int = 1000,
    max_early_detection: int = 0,
    stream_name: str = "values",
    reset: bool = True,
) -> DriftResults:
    """Feed values to a drift detector and score its detections.

    Each value is passed to :meth:`~capymoa.drift.base_detector.BaseDriftDetector.add_element`
    in order. The detections are then compared with the true drifts by
    :class:`~capymoa.drift.evaluate.EvaluateDriftDetector`.

    >>> import numpy as np
    >>> from capymoa.drift import evaluate_drift
    >>> from capymoa.drift.detectors import ADWIN
    >>> rng = np.random.default_rng(4)
    >>> values = rng.integers(2, size=2000)
    >>> for i in range(999, 2000):
    ...     values[i] = rng.integers(4, high=8)
    >>> results = evaluate_drift(values, ADWIN(delta=0.001), drifts=[1000], max_delay=50)
    >>> print(results["cumulative"]["f1"], results["detections"][0] >= 1000)
    1.0 True

    :param values: The stream of values to feed to the detector, in order.
    :param detector: The drift detector to test. It is reset first if ``reset``.
    :param drifts: The true drift positions. Either a list of instance indexes, or
        a list of ``(start, end)`` pairs for gradual drifts. Must not be empty.
    :param max_delay: Number of instances after a drift in which a detection still
        counts as correct. See :class:`~capymoa.drift.evaluate.EvaluateDriftDetector`.
    :param rate_period: Number of instances used for the alarm and false alarm
        rates.
    :param max_early_detection: Number of instances before a drift in which a
        detection still counts as correct.
    :param stream_name: Name of the stream, stored in the results.
    :param reset: If ``True``, clear the detector and its history before the run.
    :return: The metrics, with the detector's detections in ``detections``.
    :raises ValueError: If ``drifts`` is empty or ``max_delay`` is not positive.
    """
    if len(drifts) == 0:
        raise ValueError("drifts must not be empty")
    if reset:
        detector.reset(clean_history=True)

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    for value in values:
        detector.add_element(value)
    wallclock = time.perf_counter() - start_wall
    cpu_time = time.process_time() - start_cpu

    detections = [int(i) for i in detector.detection_index]
    evaluator = EvaluateDriftDetector(
        max_delay=max_delay,
        rate_period=rate_period,
        max_early_detection=max_early_detection,
    )
    metrics = evaluator.calc_performance(
        trues=drifts, preds=detections, tot_n_instances=len(values)
    )
    return DriftResults(
        learner=type(detector).__name__,
        stream=stream_name,
        instances=len(values),
        wallclock=wallclock,
        cpu_time=cpu_time,
        cumulative=asdict(metrics),  # type: ignore[typeddict-item]
        detections=detections,
    )
