"""Patient-level 80/10/10 split, train a LightGBM regressor on the utility-gain
target (mimicking the winning "Can I get your signature?" team's approach minus
path-signature features), select a decision threshold on validation, evaluate on
test (overall + per-hospital), and save the model bundle.

Usage:
    python3 03_train_model.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

import config
import sepsis_utils
from feature_engineering import get_feature_columns


def split_patients(patient_table: pd.DataFrame, seed: int):
    """80/10/10 patient-level split, stratified by ever_septic, independently
    within each hospital then unioned (see plan: avoids patient-level leakage,
    preserves per-hospital positive rates, keeps both hospitals in all 3 sets)."""
    train_ids, val_ids, test_ids = [], [], []
    rel_test = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    for hosp, g in patient_table.groupby("hospital"):
        train_g, temp_g = train_test_split(
            g, test_size=(1 - config.TRAIN_FRAC), stratify=g["ever_septic"], random_state=seed
        )
        val_g, test_g = train_test_split(
            temp_g, test_size=rel_test, stratify=temp_g["ever_septic"], random_state=seed
        )
        train_ids.extend(train_g["patient_id"].tolist())
        val_ids.extend(val_g["patient_id"].tolist())
        test_ids.extend(test_g["patient_id"].tolist())
    return train_ids, val_ids, test_ids


def main():
    print(f"Loading {config.FEATURES_PARQUET} ...")
    features = pd.read_parquet(config.FEATURES_PARQUET)
    print(f"Loaded {len(features):,} rows, {features['patient_id'].nunique():,} patients.")

    # --- utility-gain target, computed once on the full dataset (per-patient causal) ---
    features = sepsis_utils.add_row_utilities(features)

    feature_cols = get_feature_columns(features)
    # row_u1/row_u0/row_target_u_diff are outputs of add_row_utilities, not model inputs
    feature_cols = [c for c in feature_cols if c not in ("row_u1", "row_u0", "row_target_u_diff")]
    print(f"{len(feature_cols)} model features.")

    # --- 80/10/10 patient-level split ---
    patient_table = (
        features.groupby(["patient_id", "hospital"])[config.LABEL_COL]
        .max()
        .rename("ever_septic")
        .reset_index()
    )
    train_ids, val_ids, test_ids = split_patients(patient_table, config.RANDOM_SEED)
    assert set(train_ids) & set(val_ids) == set()
    assert set(train_ids) & set(test_ids) == set()
    assert set(val_ids) & set(test_ids) == set()
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test patients.")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.TRAIN_IDS_PATH.write_text(json.dumps(train_ids))
    config.VAL_IDS_PATH.write_text(json.dumps(val_ids))
    config.TEST_IDS_PATH.write_text(json.dumps(test_ids))

    train_df = features[features["patient_id"].isin(train_ids)]
    val_df = features[features["patient_id"].isin(val_ids)]
    test_df = features[features["patient_id"].isin(test_ids)]
    print(f"Rows: {len(train_df):,} train / {len(val_df):,} val / {len(test_df):,} test.")

    X_train, y_train = train_df[feature_cols], train_df["row_target_u_diff"]
    X_val, y_val = val_df[feature_cols], val_df["row_target_u_diff"]
    X_test = test_df[feature_cols]

    # --- train LightGBM regressor on the utility-gain target (Morrill et al. approach) ---
    print("Training LightGBM ...")
    t0 = time.time()
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS), lgb.log_evaluation(20)],
    )
    print(f"Trained in {time.time() - t0:.1f}s, best_iteration_={model.best_iteration_}")

    # --- threshold selection on VALIDATION set, maximizing normalized utility ---
    val_score = model.predict(X_val)
    best_threshold, best_val_utility, sweep = sepsis_utils.threshold_grid_search(
        val_score, val_df["row_u1"].to_numpy(), val_df["row_u0"].to_numpy()
    )
    print(f"Selected threshold={best_threshold:.3f} (validation utility={best_val_utility:.4f})")

    # --- evaluation: train / val / test (+ test hospital A/B slices) ---
    metrics = {}
    for name, df, X in [("train", train_df, X_train), ("val", val_df, X_val), ("test", test_df, X_test)]:
        score = model.predict(X)
        metrics[name] = sepsis_utils.evaluate_predictions(
            df[config.LABEL_COL].to_numpy(), score, df["row_u1"].to_numpy(), df["row_u0"].to_numpy(), best_threshold
        )
    test_score = model.predict(X_test)
    for hosp in ("A", "B"):
        mask = (test_df["hospital"] == hosp).to_numpy()
        metrics[f"test_hospital_{hosp}"] = sepsis_utils.evaluate_predictions(
            test_df.loc[mask, config.LABEL_COL].to_numpy(),
            test_score[mask],
            test_df.loc[mask, "row_u1"].to_numpy(),
            test_df.loc[mask, "row_u0"].to_numpy(),
            best_threshold,
        )

    for name, m in metrics.items():
        print(f"[{name}] AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} "
              f"utility={m['utility']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f}")

    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    sepsis_utils.save_roc_pr_threshold_plots(
        test_df[config.LABEL_COL].to_numpy(), test_score, sweep, config.PLOTS_DIR
    )

    # --- save model bundle ---
    # No median imputation: LightGBM handles remaining NaN natively via learned
    # split defaults, mirroring the winning team's own imputation choice
    # (forward-fill, then let LightGBM handle the rest).
    dump(
        {
            "model": model,
            "feature_names": feature_cols,
            "threshold": best_threshold,
            "impute_medians": {},
        },
        config.MODEL_BUNDLE_PATH,
    )
    print(f"Saved model bundle to {config.MODEL_BUNDLE_PATH}")


if __name__ == "__main__":
    main()
