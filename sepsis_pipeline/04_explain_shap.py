"""SHAP explainability artifacts, computed on a stratified sample of the
untouched TEST set (all available positive rows capped at SHAP_MAX_POSITIVES,
plus random negatives up to SHAP_MAX_TOTAL total).

Usage:
    python3 04_explain_shap.py
"""
import json

import numpy as np
import pandas as pd
import shap
from joblib import dump, load

import config


def stratified_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    pos = df[df[config.LABEL_COL] == 1]
    neg = df[df[config.LABEL_COL] == 0]
    if len(pos) > config.SHAP_MAX_POSITIVES:
        pos = pos.sample(n=config.SHAP_MAX_POSITIVES, random_state=seed)
    n_neg = max(config.SHAP_MAX_TOTAL - len(pos), 0)
    if len(neg) > n_neg:
        neg = neg.sample(n=n_neg, random_state=seed)
    sample = pd.concat([pos, neg], ignore_index=True)
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    print(f"Loading model bundle from {config.MODEL_BUNDLE_PATH} ...")
    bundle = load(config.MODEL_BUNDLE_PATH)
    model, feature_names = bundle["model"], bundle["feature_names"]

    print(f"Loading {config.FEATURES_PARQUET} ...")
    features = pd.read_parquet(config.FEATURES_PARQUET)

    test_ids = set(json.loads(config.TEST_IDS_PATH.read_text()))
    test_df = features[features["patient_id"].isin(test_ids)]
    print(f"Test set: {len(test_df):,} rows across {test_df['patient_id'].nunique():,} patients.")

    sample = stratified_sample(test_df, config.RANDOM_SEED)
    n_pos = int((sample[config.LABEL_COL] == 1).sum())
    print(f"SHAP sample: {len(sample):,} rows ({n_pos:,} positive, {len(sample) - n_pos:,} negative).")

    X_sample = sample[feature_names]

    print("Building TreeExplainer and computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=True)

    assert shap_values.shape == (len(sample), len(feature_names)), (
        f"Unexpected SHAP output shape {shap_values.shape}, expected "
        f"{(len(sample), len(feature_names))}"
    )

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.SHAP_VALUES_PATH, shap_values)

    # ICULOS is already included in feature_names (it's a model feature); avoid duplicating it.
    metadata_cols = ["patient_id", "hospital", config.LABEL_COL]
    sample_to_save = sample[metadata_cols + [c for c in feature_names if c not in metadata_cols]]
    sample_to_save.to_parquet(config.SHAP_SAMPLE_FEATURES_PATH, engine="pyarrow", index=False)

    dump(explainer, config.SHAP_EXPLAINER_PATH)

    print("Saving SHAP summary plots ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP artifacts to {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
