"""Score YOUR LightGBM model on the val + test patients and dump per-row predictions.

Run with Python 3.12 (lightgbm only -- NO xgboost import in this process, to avoid the
OpenMP double-load segfault). Reuses the already-engineered features.parquet so the
numbers reproduce sepsis_pipeline/artifacts/metrics.json exactly.

Output: combined/eval/results/user_scores.parquet
    patient_id, ICULOS, split, SepsisLabel, user_raw, user_prob
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

PIPE = Path(__file__).resolve().parents[2] / "sepsis_pipeline"
OUT = Path(__file__).resolve().parent / "results" / "user_scores.parquet"

# import the pipeline's criticality helper for the isotonic raw->prob mapping
sys.path.insert(0, str(PIPE))
from criticality import calibrated_probability  # noqa: E402


def main() -> None:
    features = pd.read_parquet(PIPE / "data_cache" / "features.parquet")
    bundle = load(PIPE / "artifacts" / "model_bundle.joblib")
    crit = load(PIPE / "artifacts" / "criticality_calibrator.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    val_ids = set(json.loads((PIPE / "artifacts" / "val_patient_ids.json").read_text()))
    test_ids = set(json.loads((PIPE / "artifacts" / "test_patient_ids.json").read_text()))

    split = np.where(
        features["patient_id"].isin(test_ids), "test",
        np.where(features["patient_id"].isin(val_ids), "val", "other"),
    )
    features = features.assign(split=split)
    df = features[features["split"].isin(["val", "test"])].copy()
    print(f"Scoring {len(df):,} rows ({df['patient_id'].nunique():,} val+test patients).")

    raw = model.predict(df[feature_names])
    prob = calibrated_probability(crit["calibrator"], raw)

    out = pd.DataFrame({
        "patient_id": df["patient_id"].to_numpy(),
        "ICULOS": df["ICULOS"].to_numpy(),
        "split": df["split"].to_numpy(),
        "SepsisLabel": df["SepsisLabel"].to_numpy().astype(int),
        "user_raw": raw,
        "user_prob": prob,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(out):,} rows).")

    # self-check: reproduce test precision/recall at the bundle's native raw threshold
    thr = bundle["threshold"]
    t = out[out["split"] == "test"]
    pred = (t["user_raw"] >= thr).astype(int)
    tp = int(((pred == 1) & (t["SepsisLabel"] == 1)).sum())
    fp = int(((pred == 1) & (t["SepsisLabel"] == 0)).sum())
    fn = int(((pred == 0) & (t["SepsisLabel"] == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"[self-check] test @raw>={thr:.3f}: precision={prec:.4f} recall={rec:.4f} "
          f"(metrics.json expects ~0.080 / ~0.669)")


if __name__ == "__main__":
    main()
