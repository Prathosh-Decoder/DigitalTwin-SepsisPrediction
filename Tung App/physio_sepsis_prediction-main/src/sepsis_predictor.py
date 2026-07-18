from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sepsis_features import RAW_COLUMNS, FeatureConfig, build_feature_frame
from .sepsis_policies import phase_ids


class SepsisRiskPredictor:
    """Callable patient-history predictor packaged inside the deployment pickle."""

    artifact_version = "1.0"

    def __init__(
        self,
        model: Any,
        calibrator: Any,
        feature_names: list[str],
        feature_config: FeatureConfig,
        global_threshold: float,
        phase_bounds: tuple[int, ...],
        phase_thresholds: tuple[float, ...],
        metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.feature_names = list(feature_names)
        self.feature_config = feature_config
        self.global_threshold = float(global_threshold)
        self.phase_bounds = tuple(int(value) for value in phase_bounds)
        self.phase_thresholds = tuple(float(value) for value in phase_thresholds)
        self.metadata = dict(metadata)

    @staticmethod
    def _load_patient(patient_history: Any) -> pd.DataFrame:
        if isinstance(patient_history, pd.DataFrame):
            dataframe = patient_history.copy()
        elif isinstance(patient_history, (str, Path)):
            path = Path(patient_history).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Patient file does not exist: {path}")
            dataframe = pd.read_csv(path, sep="|" if path.suffix.lower() == ".psv" else ",")
        elif isinstance(patient_history, list):
            dataframe = pd.DataFrame.from_records(patient_history)
        elif isinstance(patient_history, dict):
            dataframe = pd.DataFrame(patient_history)
        else:
            raise TypeError("patient_history must be a DataFrame, PSV/CSV path, list of records, or column dictionary")

        if dataframe.empty:
            raise ValueError("patient_history must contain at least one hourly row")
        if dataframe.columns.duplicated().any():
            duplicates = dataframe.columns[dataframe.columns.duplicated()].tolist()
            raise ValueError(f"patient_history has duplicate columns: {duplicates}")
        if not set(dataframe.columns).intersection(RAW_COLUMNS):
            raise ValueError("patient_history does not contain any recognized PhysioNet clinical columns")

        if "ICULOS" not in dataframe or pd.to_numeric(dataframe["ICULOS"], errors="coerce").isna().all():
            dataframe["ICULOS"] = np.arange(1, len(dataframe) + 1, dtype=float)
        else:
            icu_hours = pd.to_numeric(dataframe["ICULOS"], errors="coerce")
            observed = icu_hours.dropna().to_numpy(dtype=float)
            if len(observed) > 1 and np.any(np.diff(observed) < 0):
                raise ValueError("patient_history must be ordered chronologically by ICULOS")
        return dataframe.reset_index(drop=True)

    @staticmethod
    def _raw_to_logit(raw_scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(raw_scores, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def _calibrated_probability(self, raw_scores: np.ndarray) -> np.ndarray:
        return np.asarray(self.calibrator.predict_proba(self._raw_to_logit(raw_scores))[:, 1], dtype=float)

    def predict_trajectory(self, patient_history: Any) -> pd.DataFrame:
        """Return one causal risk estimate for every supplied patient hour."""
        dataframe = self._load_patient(patient_history)
        engineered = build_feature_frame(dataframe, config=self.feature_config, include_label=False)
        X = engineered.reindex(columns=self.feature_names)
        raw_scores = np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        probabilities = self._calibrated_probability(raw_scores)
        fallback_hours = pd.Series(np.arange(1, len(dataframe) + 1, dtype=float))
        hours = pd.to_numeric(dataframe["ICULOS"], errors="coerce").where(
            pd.to_numeric(dataframe["ICULOS"], errors="coerce").notna(), fallback_hours
        )
        phases = phase_ids(hours.to_numpy(dtype=float), self.phase_bounds)
        thresholds = np.asarray(self.phase_thresholds, dtype=float)[phases]
        alarms = probabilities >= thresholds
        return pd.DataFrame(
            {
                "ICULOS": hours.to_numpy(dtype=float),
                "sepsis_probability_next_6h": probabilities,
                "decision_threshold": thresholds,
                "sepsis_alarm": alarms,
                "raw_model_score": raw_scores,
            }
        )

    def predict_patient(self, patient_history: Any) -> dict[str, Any]:
        """Return the latest risk result from all patient history available so far."""
        latest = self.predict_trajectory(patient_history).iloc[-1]
        probability = float(latest["sepsis_probability_next_6h"])
        threshold = float(latest["decision_threshold"])
        return {
            "icu_hour": float(latest["ICULOS"]),
            "sepsis_probability_next_6h": probability,
            "sepsis_risk_percent": probability * 100.0,
            "decision_threshold": threshold,
            "sepsis_alarm": bool(latest["sepsis_alarm"]),
            "raw_model_score": float(latest["raw_model_score"]),
            "model_version": self.metadata.get("model_version", self.artifact_version),
            "target_definition": self.metadata.get("target_definition"),
        }

    def predict_proba(self, patient_history: Any) -> float:
        """Return only the latest calibrated probability as a float."""
        return float(self.predict_patient(patient_history)["sepsis_probability_next_6h"])

    def __call__(self, patient_history: Any) -> float:
        return self.predict_proba(patient_history)
