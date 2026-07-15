"""Causal, per-patient feature engineering. Every feature for a row uses only that
patient's own rows up to and including that row (verified in 05_sanity_checks.py).

Importable module (not run directly) so both 02_feature_engineering.py and
05_sanity_checks.py's causality spot-check can share the same logic.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config


def _points_above_cutoffs(values: np.ndarray, cutoffs) -> np.ndarray:
    """SOFA-style points for a lab where a HIGHER value is worse (e.g. Creatinine,
    Bilirubin): one point per ascending cutoff exceeded. NaN (never measured) -> 0
    points, i.e. assumed normal (documented simplification)."""
    pts = np.zeros(len(values), dtype=float)
    for c in cutoffs:
        pts += (values > c).fillna(False).to_numpy().astype(float)
    return pts


def _points_below_cutoffs(values: np.ndarray, cutoffs) -> np.ndarray:
    """SOFA-style points for a lab where a LOWER value is worse (Platelets): one
    point per descending cutoff undershot. NaN -> 0 points."""
    pts = np.zeros(len(values), dtype=float)
    for c in cutoffs:
        pts += (values < c).fillna(False).to_numpy().astype(float)
    return pts


def build_patient_features(patient_df: pd.DataFrame) -> pd.DataFrame:
    """patient_df: all rows for ONE patient, any order (will be sorted here) with
    the original PSV columns plus patient_id/hospital. Returns the engineered
    feature rows for that patient, same row count, sorted by ICULOS ascending.

    Columns are accumulated in a plain dict and assembled into a DataFrame once
    at the end (rather than incremental `out[col] = ...` assignment) to avoid
    pandas' per-insert fragmentation cost -- this function runs once per patient
    (~40k times for the full dataset), so the per-call cost matters.
    """
    g = patient_df.sort_values("ICULOS").reset_index(drop=True)
    cols: dict[str, pd.Series] = {}

    cols["patient_id"] = g["patient_id"]
    cols["hospital"] = g["hospital"]
    cols["ICULOS"] = g["ICULOS"]
    cols[config.LABEL_COL] = g[config.LABEL_COL]

    # --- static passthrough ---
    cols["Age"] = g["Age"]
    cols["Gender"] = g["Gender"]
    cols["HospAdmTime"] = g["HospAdmTime"]
    cols["Unit1"] = g["Unit1"].ffill().bfill()
    cols["Unit2"] = g["Unit2"].ffill().bfill()

    # --- LOCF-filled variables + measured flags (34 vars) ---
    for var in config.MEASURED_COLS:
        raw = g[var]
        cols[f"{var}_measured"] = raw.notna().astype(int)
        cols[f"{var}_ffill"] = raw.ffill()

    # --- missingness pattern: hours since last measured (26 labs only) ---
    for var in config.LAB_COLS:
        measured = cols[f"{var}_measured"] == 1
        last_measured_iculos = g["ICULOS"].where(measured).ffill()
        hours_since = g["ICULOS"] - last_measured_iculos
        cols[f"{var}_hours_since_measured"] = hours_since.fillna(config.HOURS_SINCE_SENTINEL)

    cols["vitals_missing_count"] = g[config.VITAL_COLS].isna().sum(axis=1)

    # --- rolling stats on RAW (pre-ffill) values, curated subset ---
    for var in config.ROLLING_VARS:
        raw = g[var]
        for w in config.ROLLING_WINDOWS:
            roll = raw.rolling(window=w, min_periods=1)
            cols[f"{var}_roll{w}_mean"] = roll.mean()
            cols[f"{var}_roll{w}_min"] = roll.min()
            cols[f"{var}_roll{w}_max"] = roll.max()
            cols[f"{var}_roll{w}_std"] = roll.std()

    # --- deltas on RAW values, curated subset ---
    for var in config.ROLLING_VARS:
        cols[f"{var}_delta_1h"] = g[var].diff(1)

    # --- clinical scores (from ffill columns) ---
    hr = cols["HR_ffill"]
    sbp = cols["SBP_ffill"]
    bun = cols["BUN_ffill"]
    creat = cols["Creatinine_ffill"]
    platelets = cols["Platelets_ffill"]
    bili = cols["Bilirubin_total_ffill"]
    map_ = cols["MAP_ffill"]

    cols["shock_index"] = hr / sbp
    cols["bun_creatinine_ratio"] = bun / creat

    platelet_pts = _points_below_cutoffs(platelets, config.SOFA_PLATELETS_CUTOFFS)
    bilirubin_pts = _points_above_cutoffs(bili, config.SOFA_BILIRUBIN_CUTOFFS)
    creatinine_pts = _points_above_cutoffs(creat, config.SOFA_CREATININE_CUTOFFS)
    map_pts = (map_ < config.SOFA_MAP_THRESHOLD).fillna(False).to_numpy().astype(float)

    partial_sofa = platelet_pts + bilirubin_pts + creatinine_pts + map_pts
    cols["partial_sofa"] = partial_sofa

    partial_sofa_s = pd.Series(partial_sofa, index=g.index)
    sofa_delta = partial_sofa_s.diff(config.SOFA_DELTA_WINDOW_HOURS)
    cols["sofa_delta_24h"] = sofa_delta
    cols["sofa_worsening_flag"] = (sofa_delta >= config.SOFA_WORSENING_DELTA).fillna(False).astype(int)

    # --- assemble once; raw (pre-ffill) measured columns are intentionally
    # excluded, superseded by the _ffill versions above ---
    return pd.DataFrame(cols, index=g.index)


def build_features_dataframe(raw_df: pd.DataFrame, n_jobs: int = 8) -> pd.DataFrame:
    """Runs build_patient_features across all patients in parallel and concatenates."""
    groups = [g for _, g in raw_df.groupby("patient_id", sort=False)]
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(build_patient_features)(g) for g in groups
    )
    return pd.concat(results, ignore_index=True)


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Columns that are actual model features. ICULOS IS included as a feature
    (elapsed ICU time is clinically predictive) even though it's also used
    separately as metadata for splitting/utility-score computation. `patient_id`
    and `hospital` are excluded -- hospital deliberately so, to avoid the model
    learning a hospital-identity shortcut instead of transferable clinical signal."""
    exclude = {"patient_id", "hospital", config.LABEL_COL}
    return [c for c in features_df.columns if c not in exclude]
