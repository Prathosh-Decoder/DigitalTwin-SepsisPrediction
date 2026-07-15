"""Turn combined_raw.parquet into features.parquet using the causal per-patient
feature logic in feature_engineering.py.

Usage:
    python3 02_feature_engineering.py
"""
import time

import pandas as pd

import config
from feature_engineering import build_features_dataframe, get_feature_columns


def main():
    print(f"Loading {config.RAW_PARQUET} ...")
    raw = pd.read_parquet(config.RAW_PARQUET)
    print(f"Loaded {len(raw):,} rows, {raw['patient_id'].nunique():,} patients.")

    t0 = time.time()
    features = build_features_dataframe(raw, n_jobs=8)
    print(f"Feature engineering done in {time.time() - t0:.1f}s.")

    feature_cols = get_feature_columns(features)
    print(f"{len(features):,} rows x {len(feature_cols)} feature columns "
          f"(+ patient_id/hospital/ICULOS/{config.LABEL_COL} metadata).")

    config.FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.FEATURES_PARQUET, engine="pyarrow", index=False)
    print(f"Saved to {config.FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
