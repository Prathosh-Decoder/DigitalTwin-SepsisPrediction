"""LOCAL STUDY ONLY -- does not touch artifacts/model_bundle.joblib, config.py, the saved split
files, or anything committed/pushed.

Retrains the production-config LightGBM on 10 different random 80/10/10 groupings, with the 8
Project-1 digital-twin demo patients PINNED to the test set in every grouping. Reports the full
spread (mean +/- std, min-max) of test metrics.

Framing: the model config is identical across all 10 runs -- only the patient grouping changes.
The variation is split noise, so the "best" grouping is the lucky draw, NOT a better model. The
spread is the answer. Nothing is adopted.

Usage:
    python3 -u experiments/split_robustness.py
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import sepsis_utils

OUT_DIR = config.BASE_DIR / "experiments" / "split_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
DEMO_PATIENT_IDS = {4880, 1072, 17091, 7057, 14527, 295, 10756, 11623}

FEATURE_EXCLUDE = {"patient_id", "hospital", config.LABEL_COL,
                   "row_u1", "row_u0", "row_target_u_diff"}


def split_patients_pinned(patient_table, seed):
    """Per-hospital stratified 80/10/10 on the NON-demo patients, then the 8 demo
    patients are appended to test. Sizes stay ~80/10/10; demo always in test."""
    pool = patient_table[~patient_table["patient_id"].isin(DEMO_PATIENT_IDS)]
    train_ids, val_ids, test_ids = [], [], []
    rel_test = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    for hosp, g in pool.groupby("hospital"):
        train_g, temp_g = train_test_split(
            g, test_size=(1 - config.TRAIN_FRAC), stratify=g["ever_septic"], random_state=seed)
        val_g, test_g = train_test_split(
            temp_g, test_size=rel_test, stratify=temp_g["ever_septic"], random_state=seed)
        train_ids += train_g["patient_id"].tolist()
        val_ids += val_g["patient_id"].tolist()
        test_ids += test_g["patient_id"].tolist()
    test_ids += list(DEMO_PATIENT_IDS)  # pin demo patients to test
    return set(train_ids), set(val_ids), set(test_ids)


def main():
    print(f"Loading {config.FEATURES_PARQUET} ...", flush=True)
    features = pd.read_parquet(config.FEATURES_PARQUET)
    features = sepsis_utils.add_row_utilities(features)
    feature_cols = [c for c in features.columns if c not in FEATURE_EXCLUDE]
    print(f"{len(features):,} rows, {len(feature_cols)} features. Demo patients pinned to test.\n", flush=True)

    patient_table = (features.groupby(["patient_id", "hospital"])[config.LABEL_COL]
                     .max().rename("ever_septic").reset_index())

    rows = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        train_ids, val_ids, test_ids = split_patients_pinned(patient_table, seed)
        assert DEMO_PATIENT_IDS <= test_ids, "demo patients not all in test!"
        assert not (train_ids & test_ids) and not (val_ids & test_ids) and not (train_ids & val_ids)

        tr = features[features["patient_id"].isin(train_ids)]
        va = features[features["patient_id"].isin(val_ids)]
        te = features[features["patient_id"].isin(test_ids)]

        model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        model.fit(tr[feature_cols], tr["row_target_u_diff"],
                  eval_set=[(va[feature_cols], va["row_target_u_diff"])],
                  eval_metric="l2",
                  callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
                             lgb.log_evaluation(0)])

        val_score = model.predict(va[feature_cols])
        thr, val_util, _ = sepsis_utils.threshold_grid_search(
            val_score, va["row_u1"].to_numpy(), va["row_u0"].to_numpy())
        test_score = model.predict(te[feature_cols])
        m = sepsis_utils.evaluate_predictions(
            te[config.LABEL_COL].to_numpy(), test_score,
            te["row_u1"].to_numpy(), te["row_u0"].to_numpy(), thr)

        rows.append({"seed": seed, "val_utility": val_util, "threshold": thr,
                     "AUROC": m["auroc"], "AUPRC": m["auprc"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "Lift@10%": m["lift_at_10pct"],
                     "Utility": m["utility"]})
        print(f"seed {seed}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} Recall={m['recall']:.4f} "
              f"Utility={m['utility']:.4f} (val_util={val_util:.4f})  [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    metric_cols = ["AUROC", "AUPRC", "Precision", "Recall", "F1", "Lift@10%", "Utility"]

    print("\n=== SPREAD across 10 groupings (demo patients pinned to test) ===", flush=True)
    print(df[["seed"] + metric_cols].round(4).to_string(index=False), flush=True)
    print("\nmean :", {k: round(df[k].mean(), 4) for k in metric_cols}, flush=True)
    print("std  :", {k: round(df[k].std(), 4) for k in metric_cols}, flush=True)
    print("min  :", {k: round(df[k].min(), 4) for k in metric_cols}, flush=True)
    print("max  :", {k: round(df[k].max(), 4) for k in metric_cols}, flush=True)

    # reference lines
    prod = json.loads(config.METRICS_PATH.read_text())["test"]
    print(f"\nREFERENCE -- current production single split: AUROC={prod['auroc']:.4f} "
          f"AUPRC={prod['auprc']:.4f} Recall={prod['recall']:.4f} Utility={prod['utility']:.4f}", flush=True)
    print("REFERENCE -- nested-CV honest estimate: AUROC=0.8381+/-0.0114 Utility=0.4070+/-0.0233", flush=True)

    df.to_csv(OUT_DIR / "spread.csv", index=False)
    (OUT_DIR / "results.json").write_text(json.dumps({
        "per_seed": rows,
        "summary": {k: {"mean": float(df[k].mean()), "std": float(df[k].std()),
                        "min": float(df[k].min()), "max": float(df[k].max())} for k in metric_cols},
        "production_reference": prod,
    }, indent=2, default=float))
    print(f"\nSaved to {OUT_DIR}/. Production model NOT modified; nothing adopted.", flush=True)


if __name__ == "__main__":
    main()
