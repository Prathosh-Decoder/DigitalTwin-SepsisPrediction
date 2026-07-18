"""Robust wrapper around Tung's packaged sepsis model.

Two problems with loading Tung's cloudpickle artifact and calling it directly:

  1. The XGBoost booster inside was serialized by a build whose in-memory binary format
     segfaults on `predict` against the installed xgboost (loading is fine; only inference
     crashes). Fix: re-export the booster with `save_model` and reload it -- xgboost's own
     recommended cross-version path -- which normalizes the format.
  2. The predictor's own methods (`_load_patient`, `predict_trajectory`, ...) were cloudpickled
     *by value*; their serialized bytecode segfaults when executed under this Python build.
     Fix: never call those methods. Reconstruct the (simple, documented) inference here from
     the real `src` source modules + the pickle's plain data attributes + the healed model.

This module must be imported from the `.venv-tung` environment (Python 3.12,
xgboost/cloudpickle/shap, numpy<2.4, pandas<2.3). Do NOT import lightgbm in the same
process (LightGBM + XGBoost OpenMP double-load segfaults).
"""
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
TUNG_DIR = ROOT / "Tung App" / "physio_sepsis_prediction-main"
TUNG_PKL = TUNG_DIR / "models" / "sepsis_next_6h_predictor.pkl"

if str(TUNG_DIR) not in sys.path:
    sys.path.insert(0, str(TUNG_DIR))

# real source modules (safe to execute -- imported from disk, not the pickled bytecode)
from src.sepsis_features import LITERATURE_CORE_CONFIG, build_feature_frame  # noqa: E402
from src.sepsis_policies import phase_ids  # noqa: E402


def _heal_booster(clf):
    """Round-trip the xgboost booster through JSON so predict is stable in this build."""
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        clf.get_booster().save_model(tmp)
        healed = xgb.XGBClassifier()
        healed.load_model(tmp)
        return healed
    finally:
        os.unlink(tmp)


class TungModel:
    """Tung's model, reconstructed for stable inference + SHAP in this environment."""

    def __init__(self):
        with open(TUNG_PKL, "rb") as f:
            pred = pickle.load(f)          # loading is safe; we just read attributes
        self.model = _heal_booster(pred.model)             # healed XGBClassifier (native)
        self.calibrator = pred.calibrator                  # sklearn LogisticRegression (native)
        self.feature_names = list(pred.feature_names)      # plain data
        self.phase_bounds = tuple(int(b) for b in pred.phase_bounds)
        self.phase_thresholds = np.asarray(pred.phase_thresholds, dtype=float)
        self.global_threshold = float(pred.global_threshold)
        self.config = LITERATURE_CORE_CONFIG               # from source
        self._explainer = None

    # -- feature engineering (mirrors SepsisRiskPredictor._load_patient + predict_trajectory) --
    def _engineer(self, patient_df):
        df = patient_df.copy()
        icu = pd.to_numeric(df.get("ICULOS"), errors="coerce") if "ICULOS" in df else None
        if icu is None or icu.isna().all():
            df["ICULOS"] = np.arange(1, len(df) + 1, dtype=float)
        df = df.reset_index(drop=True)
        engineered = build_feature_frame(df, config=self.config, include_label=False)
        X = engineered.reindex(columns=self.feature_names)
        hours = pd.to_numeric(df["ICULOS"], errors="coerce").to_numpy(dtype=float)
        return X, hours

    def _calibrate(self, raw_scores):
        clipped = np.clip(np.asarray(raw_scores, dtype=float), 1e-7, 1 - 1e-7)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return np.asarray(self.calibrator.predict_proba(logit)[:, 1], dtype=float)

    def trajectory(self, patient_df):
        """Per-hour causal risk for one patient. Columns: ICULOS, tung_prob, tung_raw, threshold, alarm."""
        X, hours = self._engineer(patient_df)
        raw = np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        prob = self._calibrate(raw)
        phases = phase_ids(hours, self.phase_bounds)
        thr = self.phase_thresholds[phases]
        return pd.DataFrame({
            "ICULOS": hours,
            "tung_prob": prob,
            "tung_raw": raw,
            "threshold": thr,
            "alarm": prob >= thr,
        })

    def explain_row(self, patient_df, k=5):
        """Top-k SHAP drivers for the LATEST hour. Returns (prob, raw, threshold, alarm, drivers)."""
        import shap  # lazy: only the app tab needs it
        X, hours = self._engineer(patient_df)
        raw = np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        prob = self._calibrate(raw)
        phases = phase_ids(hours, self.phase_bounds)
        thr = float(self.phase_thresholds[phases][-1])
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        row = X.iloc[[-1]]
        sv = np.asarray(self._explainer.shap_values(row))[0]
        vals = row.to_numpy()[0]
        order = np.argsort(-np.abs(sv))[:k]
        drivers = [{
            "feature": self.feature_names[i],
            "value": float(vals[i]) if np.isfinite(vals[i]) else None,
            "shap": float(sv[i]),
            "direction": "↑" if sv[i] > 0 else "↓",
        } for i in order]
        return float(prob[-1]), float(raw[-1]), thr, bool(prob[-1] >= thr), drivers
