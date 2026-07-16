"""Criticality / prioritization layer on top of the trained sepsis model.

This is a LAYER on top of the frozen model -- it never retrains or modifies it. It turns the
model's raw score (which is NOT a probability, range ~ -0.1..+1.5) into interpretable triage
outputs:

  - calibrated_probability : honest P(pre-sepsis window), via isotonic regression fit on validation
  - criticality_score      : 0-100 percentile rank vs a training-set reference (triage headline;
                             a RELATIVE RISK RANK, not a probability)
  - tier_from_score        : LOW / MODERATE / HIGH / CRITICAL bands
  - criticality_trend      : rising / steady / falling over recent hours (causal)
  - top_shap_drivers       : top-k SHAP features in plain clinical language

Importable module; the fitted calibrator + reference are produced by 07_criticality.py and saved
to artifacts/criticality_calibrator.joblib.
"""
import re

import numpy as np
from sklearn.isotonic import IsotonicRegression

# --- tier bands (display defaults; validated in 07_criticality.py) ---
DEFAULT_TIER_BANDS = [("LOW", 0, 50), ("MODERATE", 50, 75), ("HIGH", 75, 90), ("CRITICAL", 90, 101)]

# --- plain-language names for the raw clinical variables ---
_VAR_PLAIN = {
    "HR": "heart rate", "O2Sat": "oxygen saturation", "Temp": "temperature",
    "SBP": "systolic BP", "MAP": "mean arterial pressure", "DBP": "diastolic BP",
    "Resp": "respiration rate", "EtCO2": "end-tidal CO2",
    "BaseExcess": "base excess", "HCO3": "bicarbonate", "FiO2": "inspired oxygen",
    "pH": "blood pH", "PaCO2": "arterial CO2", "SaO2": "arterial O2 sat",
    "AST": "AST", "BUN": "blood urea nitrogen", "Alkalinephos": "alkaline phosphatase",
    "Calcium": "calcium", "Chloride": "chloride", "Creatinine": "creatinine",
    "Bilirubin_direct": "direct bilirubin", "Glucose": "glucose", "Lactate": "lactate",
    "Magnesium": "magnesium", "Phosphate": "phosphate", "Potassium": "potassium",
    "Bilirubin_total": "bilirubin", "TroponinI": "troponin", "Hct": "hematocrit",
    "Hgb": "hemoglobin", "PTT": "PTT", "WBC": "white blood cell count",
    "Fibrinogen": "fibrinogen", "Platelets": "platelets",
    "Age": "age", "Gender": "gender", "HospAdmTime": "time since hospital admission",
    "ICULOS": "hours in ICU", "Unit1": "ICU unit", "Unit2": "ICU unit",
}
_SPECIAL_PLAIN = {
    "shock_index": "shock index (HR/SBP)",
    "bun_creatinine_ratio": "BUN/creatinine ratio",
    "partial_sofa": "organ-dysfunction score (partial SOFA)",
    "sofa_delta_24h": "rising organ-dysfunction score (24h)",
    "sofa_worsening_flag": "worsening organ-dysfunction flag",
    "vitals_missing_count": "number of unrecorded vitals",
}


def plain_name(feature: str) -> str:
    """Render one engineered feature name in plain clinical English."""
    if feature in _SPECIAL_PLAIN:
        return _SPECIAL_PLAIN[feature]
    if feature in _VAR_PLAIN:
        return _VAR_PLAIN[feature]

    m = re.match(r"^(?P<var>.+?)_(?P<suf>ffill|measured|delta_1h|hours_since_measured|"
                 r"roll(?P<win>\d+)_(?P<stat>mean|min|max|std))$", feature)
    if not m:
        return feature.replace("_", " ")
    var = m.group("var")
    v = _VAR_PLAIN.get(var, var.replace("_", " "))
    suf = m.group("suf")
    if suf == "ffill":
        return f"latest {v}"
    if suf == "measured":
        return f"{v} recorded this hour"
    if suf == "delta_1h":
        return f"change in {v} (1h)"
    if suf == "hours_since_measured":
        return f"hours since {v} drawn"
    win, stat = m.group("win"), m.group("stat")
    stat_word = {"mean": "avg", "min": "lowest", "max": "peak", "std": "variability of"}[stat]
    return f"{stat_word} {v} ({win}h)"


def build_plain_names(feature_names) -> dict:
    return {f: plain_name(f) for f in feature_names}


# ---------------------------------------------------------------------------
# Calibration + reference (fit in 07_criticality.py)
# ---------------------------------------------------------------------------

def fit_calibrator(val_raw_scores, val_labels) -> IsotonicRegression:
    """Isotonic calibration: raw score -> P(SepsisLabel=1). Fit on VALIDATION only."""
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(np.asarray(val_raw_scores, dtype=float), np.asarray(val_labels, dtype=float))
    return cal


def build_reference_quantiles(train_raw_scores, n_points=1001) -> np.ndarray:
    """Compact reference distribution of raw scores (sorted quantile grid) for percentile lookup."""
    return np.quantile(np.asarray(train_raw_scores, dtype=float), np.linspace(0, 1, n_points))


# ---------------------------------------------------------------------------
# The four triage views
# ---------------------------------------------------------------------------

def calibrated_probability(calibrator, raw_scores) -> np.ndarray:
    """Calibrated P(pre-sepsis) in [0,1]."""
    p = calibrator.predict(np.atleast_1d(np.asarray(raw_scores, dtype=float)))
    return np.clip(p, 0.0, 1.0)


def criticality_score(raw_scores, reference_quantiles) -> np.ndarray:
    """0-100 percentile rank of each raw score vs the reference distribution. Monotonic with the
    raw score -- a RELATIVE RISK RANK, not a probability. 94 = riskier than ~94% of reference hours."""
    raw = np.atleast_1d(np.asarray(raw_scores, dtype=float))
    ref = np.asarray(reference_quantiles, dtype=float)
    pct = np.searchsorted(ref, raw, side="right") / len(ref) * 100.0
    return np.clip(pct, 0.0, 100.0)


def tier_from_score(criticality, bands=DEFAULT_TIER_BANDS):
    """Map criticality (scalar or array) to a tier label."""
    def one(c):
        for name, lo, hi in bands:
            if lo <= c < hi:
                return name
        return bands[-1][0]
    arr = np.atleast_1d(criticality)
    tiers = [one(float(c)) for c in arr]
    return tiers[0] if np.isscalar(criticality) or arr.shape == (1,) else tiers


def criticality_trend(criticality_series, lookback=3, steady_band=5.0) -> str:
    """rising / steady / falling based on the change in criticality over the last `lookback` hours.
    Expects a sequence ordered oldest->newest (past hours only; causal)."""
    vals = np.asarray(criticality_series, dtype=float)
    if len(vals) < 2:
        return "steady"
    past = vals[-1 - lookback] if len(vals) > lookback else vals[0]
    delta = vals[-1] - past
    if delta > steady_band:
        return "rising"
    if delta < -steady_band:
        return "falling"
    return "steady"


def top_shap_drivers(shap_row, feature_row, feature_names, k=3) -> list:
    """Top-k features by |SHAP| for one prediction, in plain language.
    Returns [{feature, plain_name, value, shap, direction}] where direction is an up/down arrow
    for whether the feature pushed predicted risk up or down."""
    shap_row = np.asarray(shap_row, dtype=float)
    feature_row = np.asarray(feature_row, dtype=float)
    order = np.argsort(-np.abs(shap_row))[:k]
    out = []
    for i in order:
        out.append({
            "feature": feature_names[i],
            "plain_name": plain_name(feature_names[i]),
            "value": float(feature_row[i]),
            "shap": float(shap_row[i]),
            "direction": "↑" if shap_row[i] > 0 else "↓",
        })
    return out
