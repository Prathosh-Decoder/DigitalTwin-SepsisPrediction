"""End-to-end verification suite: run AFTER 01-04 have produced their artifacts.
Exits non-zero (via AssertionError) on any hard failure; prints WARN for softer,
eyeball-worthy flags. See plan's Verification section for what each check guards.

Usage:
    python3 05_sanity_checks.py
"""
import json

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score, average_precision_score

import config
from feature_engineering import build_patient_features


def check_row_file_reconciliation(raw: pd.DataFrame):
    print("\n--- Row/file reconciliation ---")
    total_expected_rows = 0
    for (pid, hosp), _ in raw.groupby(["patient_id", "hospital"]):
        data_dir = config.DATA_DIR_A if hosp == "A" else config.DATA_DIR_B
        path = data_dir / f"p{pid:06d}.psv"
        assert path.exists(), f"Missing source file for patient_id={pid} hospital={hosp}: {path}"
        with open(path) as f:
            n_lines = sum(1 for _ in f)
        total_expected_rows += n_lines - 1  # minus header
    assert total_expected_rows == len(raw), (
        f"Row count mismatch: source files imply {total_expected_rows:,} rows, "
        f"raw parquet has {len(raw):,}"
    )
    n_files = raw.groupby(["patient_id", "hospital"]).ngroups
    n_unique_patients = raw["patient_id"].nunique()
    assert n_unique_patients == n_files, "patient_id is not unique across hospitals as assumed"
    print(f"OK: {len(raw):,} rows reconcile exactly against {n_files:,} source files.")


def check_schema_and_label(raw: pd.DataFrame):
    print("\n--- Schema / label check ---")
    expected_cols = set(config.MEASURED_COLS) | set(config.STATIC_COLS) | {
        config.LABEL_COL, "patient_id", "hospital"
    }
    missing = expected_cols - set(raw.columns)
    assert not missing, f"Missing expected columns: {missing}"
    labels = raw[config.LABEL_COL]
    assert labels.isna().sum() == 0, "SepsisLabel contains NaN"
    assert set(labels.unique()) <= {0, 1}, f"SepsisLabel has unexpected values: {labels.unique()}"
    print("OK: all expected columns present, SepsisLabel is clean binary.")


def check_no_unexpected_nan(features: pd.DataFrame):
    print("\n--- NaN-shouldn't-leak checks ---")
    always_known = ["patient_id", "hospital", "ICULOS", config.LABEL_COL, "Age", "Gender"]
    for col in always_known:
        n_nan = features[col].isna().sum()
        assert n_nan == 0, f"Column '{col}' should never be NaN but has {n_nan}"
    for var in config.MEASURED_COLS:
        n_nan = features[f"{var}_measured"].isna().sum()
        assert n_nan == 0, f"'{var}_measured' flag should never be NaN but has {n_nan}"
    print("OK: metadata/label/_measured columns have zero NaN.")

    print("NaN%% after LOCF for a few representative _ffill columns:")
    for var in ["HR", "O2Sat", "Temp", "TroponinI", "Fibrinogen"]:
        pct = 100 * features[f"{var}_ffill"].isna().mean()
        print(f"  {var}_ffill: {pct:.1f}% NaN")
    hr_nan_pct = 100 * features["HR_ffill"].isna().mean()
    if hr_nan_pct > 5:
        print(f"WARN: HR_ffill NaN% ({hr_nan_pct:.1f}%) higher than expected for a near-hourly vital.")


def check_causality(raw: pd.DataFrame, n_patients: int = 5, n_spot_checks: int = 3):
    print("\n--- Causality / no-lookahead spot check ---")
    rng = np.random.default_rng(config.RANDOM_SEED)
    patient_ids = raw["patient_id"].unique()
    sample_pids = rng.choice(patient_ids, size=min(n_patients, len(patient_ids)), replace=False)

    for pid in sample_pids:
        pdf = raw[raw["patient_id"] == pid].sort_values("ICULOS").reset_index(drop=True)
        if len(pdf) < 4:
            continue
        full_result = build_patient_features(pdf)

        candidate_hours = pdf["ICULOS"].iloc[1:].sample(
            n=min(n_spot_checks, len(pdf) - 1), random_state=config.RANDOM_SEED
        )
        for t in candidate_hours:
            truncated = pdf[pdf["ICULOS"] <= t]
            truncated_result = build_patient_features(truncated)

            full_row = full_result[full_result["ICULOS"] == t].iloc[0]
            trunc_row = truncated_result[truncated_result["ICULOS"] == t].iloc[0]

            for col in full_row.index:
                a, b = full_row[col], trunc_row[col]
                if pd.isna(a) and pd.isna(b):
                    continue
                if isinstance(a, (int, float, np.floating, np.integer)):
                    assert np.isclose(a, b, equal_nan=True), (
                        f"Causality violation for patient={pid}, hour={t}, column={col}: "
                        f"full-stay value={a} != truncated-at-t value={b} "
                        "(a feature is peeking at future rows)"
                    )
                else:
                    assert a == b, f"Causality violation for patient={pid}, hour={t}, column={col}"
    print(f"OK: {len(sample_pids)} patients, feature values identical whether computed "
          "from the full stay or truncated at each spot-checked hour.")


def check_split_integrity():
    print("\n--- Split integrity ---")
    train_ids = set(json.loads(config.TRAIN_IDS_PATH.read_text()))
    val_ids = set(json.loads(config.VAL_IDS_PATH.read_text()))
    test_ids = set(json.loads(config.TEST_IDS_PATH.read_text()))

    assert train_ids.isdisjoint(val_ids), "train/val patient overlap"
    assert train_ids.isdisjoint(test_ids), "train/test patient overlap"
    assert val_ids.isdisjoint(test_ids), "val/test patient overlap"

    total = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"OK: no overlap. {len(train_ids)} train ({100*len(train_ids)/total:.1f}%) / "
          f"{len(val_ids)} val ({100*len(val_ids)/total:.1f}%) / "
          f"{len(test_ids)} test ({100*len(test_ids)/total:.1f}%).")

    features = pd.read_parquet(config.FEATURES_PARQUET, columns=["patient_id", "hospital", config.LABEL_COL])
    patient_table = features.groupby(["patient_id", "hospital"])[config.LABEL_COL].max()
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        sub = patient_table[patient_table.index.get_level_values("patient_id").isin(ids)]
        pos_rate = sub.mean()
        hosp_counts = sub.index.get_level_values("hospital").value_counts(normalize=True)
        print(f"  {name}: {len(sub)} patients, {100*pos_rate:.1f}% ever-septic, "
              f"hospital mix A={hosp_counts.get('A', 0):.2f} B={hosp_counts.get('B', 0):.2f}")


def check_model_sanity():
    print("\n--- Model sanity ---")
    bundle = load(config.MODEL_BUNDLE_PATH)
    model, feature_names = bundle["model"], bundle["feature_names"]
    assert model.n_features_in_ == len(feature_names), "model/feature_names length mismatch"

    metrics = json.loads(config.METRICS_PATH.read_text())
    test_m = metrics["test"]
    baseline_ap = test_m["positive_rate"]
    assert test_m["auprc"] > baseline_ap, (
        f"Test AUPRC ({test_m['auprc']:.4f}) does not beat the trivial base-rate "
        f"baseline ({baseline_ap:.4f}) -- model may not be learning useful signal."
    )
    print(f"OK: test AUPRC {test_m['auprc']:.4f} > base-rate baseline {baseline_ap:.4f}.")

    if test_m["auroc"] > 0.95:
        print(f"WARN: test AUROC={test_m['auroc']:.4f} is unusually high for this task -- "
              "double-check for feature leakage.")
    if test_m["auroc"] < 0.55:
        print(f"WARN: test AUROC={test_m['auroc']:.4f} is close to chance -- "
              "double-check feature/label alignment.")
    print(f"model.n_features_in_={model.n_features_in_} matches feature_names length.")


def check_shap_shapes():
    print("\n--- SHAP shape check ---")
    bundle = load(config.MODEL_BUNDLE_PATH)
    feature_names = bundle["feature_names"]
    shap_values = np.load(config.SHAP_VALUES_PATH)
    sample = pd.read_parquet(config.SHAP_SAMPLE_FEATURES_PATH)

    assert shap_values.shape[0] == len(sample), "shap_values row count != sample row count"
    assert shap_values.shape[1] == len(feature_names), "shap_values column count != feature count"
    print(f"OK: shap_values.shape={shap_values.shape} matches ({len(sample)}, {len(feature_names)}).")
    return bundle, shap_values, sample


def spot_check_predictions(bundle, shap_values, sample, n=4):
    print("\n--- Individual prediction spot-check ---")
    model, feature_names, threshold = bundle["model"], bundle["feature_names"], bundle["threshold"]
    X_sample = sample[feature_names]
    scores = model.predict(X_sample)

    pos_idx = np.where(sample[config.LABEL_COL].to_numpy() == 1)[0]
    neg_idx = np.where(sample[config.LABEL_COL].to_numpy() == 0)[0]
    rng = np.random.default_rng(config.RANDOM_SEED)
    picks = list(rng.choice(pos_idx, size=min(2, len(pos_idx)), replace=False)) + \
            list(rng.choice(neg_idx, size=min(2, len(neg_idx)), replace=False))

    for i in picks[:n]:
        row = sample.iloc[i]
        score = scores[i]
        pred = int(score >= threshold)
        top_feats = np.argsort(-np.abs(shap_values[i]))[:5]
        print(f"\n  patient_id={row['patient_id']} hour={row['ICULOS']} "
              f"true_label={row[config.LABEL_COL]} pred={pred} score={score:.3f} (thr={threshold:.3f})")
        for j in top_feats:
            fname = feature_names[j]
            print(f"    {fname}: value={row[fname]:.3f}, shap={shap_values[i, j]:+.4f}")


def main():
    print("Loading artifacts ...")
    raw = pd.read_parquet(config.RAW_PARQUET)
    features = pd.read_parquet(config.FEATURES_PARQUET)

    check_row_file_reconciliation(raw)
    check_schema_and_label(raw)
    check_no_unexpected_nan(features)
    check_causality(raw)
    check_split_integrity()
    check_model_sanity()
    bundle, shap_values, sample = check_shap_shapes()
    spot_check_predictions(bundle, shap_values, sample)

    print("\n=== All sanity checks passed ===")


if __name__ == "__main__":
    main()
