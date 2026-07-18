from __future__ import annotations

import numpy as np


DEFAULT_PHASE_BOUNDS = (12, 24, 48)


def phase_ids(hours: np.ndarray, phase_bounds: tuple[int, ...] = DEFAULT_PHASE_BOUNDS) -> np.ndarray:
    """Map one-based ICU hours to threshold phases."""
    return np.digitize(np.asarray(hours, dtype=float), bins=np.asarray(phase_bounds), right=True)


def apply_global_threshold(scores_by_patient: list[np.ndarray], threshold: float) -> list[np.ndarray]:
    return [(np.asarray(scores) >= threshold).astype(np.int8) for scores in scores_by_patient]


def apply_time_phased_thresholds(
    scores_by_patient: list[np.ndarray],
    hours_by_patient: list[np.ndarray],
    thresholds: tuple[float, ...] | list[float] | np.ndarray,
    phase_bounds: tuple[int, ...] = DEFAULT_PHASE_BOUNDS,
) -> list[np.ndarray]:
    threshold_values = np.asarray(thresholds, dtype=float)
    expected_count = len(phase_bounds) + 1
    if len(threshold_values) != expected_count:
        raise ValueError(f"Expected {expected_count} phase thresholds, received {len(threshold_values)}")
    if len(scores_by_patient) != len(hours_by_patient):
        raise ValueError("Scores and ICU-hour arrays must contain the same patients")

    predictions: list[np.ndarray] = []
    for scores, hours in zip(scores_by_patient, hours_by_patient):
        scores_array = np.asarray(scores, dtype=float)
        patient_phases = phase_ids(np.asarray(hours), phase_bounds)
        if len(scores_array) != len(patient_phases):
            raise ValueError("Scores and ICU-hour arrays must have equal length for each patient")
        predictions.append((scores_array >= threshold_values[patient_phases]).astype(np.int8))
    return predictions
