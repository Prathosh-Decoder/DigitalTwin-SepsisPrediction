"""LOCAL STUDY ONLY -- does not touch artifacts/model_bundle.joblib, config.py, the saved splits,
or anything committed/pushed.

Rebuilds the pinned split (8 demo patients in test) but backfills the train/val slots the demo
patients vacated with CLINICALLY SIMILAR patients (nearest-neighbor on a per-patient summary,
same hospital + same septic outcome) instead of random ones. Retrains the production-config model
and compares to the current (random-backfill) production model.

Usage:
    python3 -u experiments/matched_replacement.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import sepsis_utils

OUT_DIR = config.BASE_DIR / "experiments" / "matched_replacement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEMO_PATIENT_IDS = [4880, 1072, 17091, 7057, 14527, 295, 10756, 11623]

# per-patient summary columns for the similarity match (raw clinical values)
SUMMARY_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
SUMMARY_LABS = ["WBC", "Lactate", "Creatinine", "Platelets", "BUN"]

FEATURE_EXCLUDE = {"patient_id", "hospital", config.LABEL_COL,
                   "row_u1", "row_u0", "row_target_u_diff"}


def natural_split(patient_table, seed):
    """The original seed-42 80/10/10 split BEFORE any demo pinning (reproduces how the
    production split's base was built)."""
    train_ids, val_ids, test_ids = [], [], []
    rel_test = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    for hosp, g in patient_table.groupby("hospital"):
        train_g, temp_g = train_test_split(
            g, test_size=(1 - config.TRAIN_FRAC), stratify=g["ever_septic"], random_state=seed)
        val_g, test_g = train_test_split(
            temp_g, test_size=rel_test, stratify=temp_g["ever_septic"], random_state=seed)
        train_ids += train_g["patient_id"].tolist()
        val_ids += val_g["patient_id"].tolist()
        test_ids += test_g["patient_id"].tolist()
    return set(train_ids), set(val_ids), set(test_ids)


def build_patient_summary():
    """Per-patient clinical summary vector from the raw hourly data."""
    raw = pd.read_parquet(config.RAW_PARQUET)
    agg = {v: "mean" for v in SUMMARY_VITALS + SUMMARY_LABS}
    agg.update({"Age": "first", "Gender": "first", "HospAdmTime": "first",
                "ICULOS": "max", config.LABEL_COL: "max", "hospital": "first"})
    s = raw.groupby("patient_id").agg(agg).rename(columns={"ICULOS": "LOS",
                                                            config.LABEL_COL: "ever_septic"})
    return s.reset_index()


def matched_backfill(summary, departing_pids, candidate_pool_pids, exclude=set()):
    """For each departing demo patient, find the nearest candidate patient in the same hospital
    with the same ever_septic outcome (Euclidean on standardized summary features). Greedy,
    without replacement. Returns {departing_pid: matched_pid}."""
    feat_cols = SUMMARY_VITALS + SUMMARY_LABS + ["Age", "Gender", "HospAdmTime", "LOS"]
    S = summary.set_index("patient_id").copy()
    S[feat_cols] = S[feat_cols].fillna(S[feat_cols].median())
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(S[feat_cols]), index=S.index, columns=feat_cols)

    used = set(exclude)
    matches = {}
    for dp in departing_pids:
        hosp, sep = S.loc[dp, "hospital"], S.loc[dp, "ever_septic"]
        cands = [p for p in candidate_pool_pids
                 if p not in used and S.loc[p, "hospital"] == hosp and S.loc[p, "ever_septic"] == sep]
        if not cands:  # relax outcome constraint if no same-outcome candidate (shouldn't happen)
            cands = [p for p in candidate_pool_pids if p not in used and S.loc[p, "hospital"] == hosp]
        d = ((Z.loc[cands] - Z.loc[dp]) ** 2).sum(axis=1)
        best = d.idxmin()
        matches[dp] = int(best)
        used.add(best)
    return matches


def evaluate_split(features, feature_cols, train_ids, val_ids, test_ids):
    tr = features[features["patient_id"].isin(train_ids)]
    va = features[features["patient_id"].isin(val_ids)]
    te = features[features["patient_id"].isin(test_ids)]
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    model.fit(tr[feature_cols], tr["row_target_u_diff"],
              eval_set=[(va[feature_cols], va["row_target_u_diff"])], eval_metric="l2",
              callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
                         lgb.log_evaluation(0)])
    val_score = model.predict(va[feature_cols])
    thr, _, _ = sepsis_utils.threshold_grid_search(
        val_score, va["row_u1"].to_numpy(), va["row_u0"].to_numpy())
    test_score = model.predict(te[feature_cols])
    return sepsis_utils.evaluate_predictions(
        te[config.LABEL_COL].to_numpy(), test_score,
        te["row_u1"].to_numpy(), te["row_u0"].to_numpy(), thr)


def main():
    print("Loading features + building per-patient summaries ...", flush=True)
    features = pd.read_parquet(config.FEATURES_PARQUET)
    features = sepsis_utils.add_row_utilities(features)
    feature_cols = [c for c in features.columns if c not in FEATURE_EXCLUDE]
    summary = build_patient_summary()

    patient_table = (features.groupby(["patient_id", "hospital"])[config.LABEL_COL]
                     .max().rename("ever_septic").reset_index())

    demo = set(DEMO_PATIENT_IDS)
    nat_train, nat_val, nat_test = natural_split(patient_table, config.RANDOM_SEED)

    depart_train = sorted(demo & nat_train)   # demo patients that must leave train
    depart_val = sorted(demo & nat_val)       # demo patients that must leave val
    print(f"Demo patients leaving train: {depart_train}", flush=True)
    print(f"Demo patients leaving val:   {depart_val}", flush=True)
    print(f"Demo already in test:        {sorted(demo & nat_test)}\n", flush=True)

    # candidate backfill pool = natural test patients that are NOT demo
    pool = [p for p in nat_test if p not in demo]
    match_train = matched_backfill(summary, depart_train, pool)
    match_val = matched_backfill(summary, depart_val, pool, exclude=set(match_train.values()))

    print("Matched backfill (departing demo -> similar patient pulled into train/val):", flush=True)
    for dp, mp in {**match_train, **match_val}.items():
        ds, ms = summary.set_index("patient_id").loc[dp], summary.set_index("patient_id").loc[mp]
        print(f"  demo p{dp:06d} (hosp {ds['hospital']}, septic={int(ds['ever_septic'])}, "
              f"age {ds['Age']:.0f}, LOS {ds['LOS']:.0f})  ->  p{mp:06d} "
              f"(hosp {ms['hospital']}, septic={int(ms['ever_septic'])}, age {ms['Age']:.0f}, "
              f"LOS {ms['LOS']:.0f})", flush=True)

    matched_in = set(match_train.values()) | set(match_val.values())
    train_m = (nat_train - set(depart_train)) | set(match_train.values())
    val_m = (nat_val - set(depart_val)) | set(match_val.values())
    test_m = (nat_test - matched_in) | demo

    assert demo <= test_m and not (train_m & test_m) and not (val_m & test_m) and not (train_m & val_m)
    assert len(train_m) == len(nat_train) and len(val_m) == len(nat_val) and len(test_m) == len(nat_test)

    print("\nRetraining on matched-backfill split ...", flush=True)
    m_matched = evaluate_split(features, feature_cols, train_m, val_m, test_m)

    prod = json.loads(config.METRICS_PATH.read_text())["test"]
    metric_keys = ["auroc", "auprc", "precision", "recall", "f1", "lift_at_10pct", "utility"]
    labels = ["AUROC", "AUPRC", "Precision", "Recall", "F1", "Lift@10%", "Utility"]

    print("\n=== Current production (random backfill) vs Matched backfill ===", flush=True)
    print(f"{'metric':12s} {'random (current)':>18s} {'matched':>12s}", flush=True)
    for k, lab in zip(metric_keys, labels):
        print(f"{lab:12s} {prod[k]:>18.4f} {m_matched[k]:>12.4f}", flush=True)

    (OUT_DIR / "results.json").write_text(json.dumps(
        {"production_random": prod, "matched": m_matched,
         "matches": {int(k): int(v) for k, v in {**match_train, **match_val}.items()}},
        indent=2, default=float))
    print(f"\nSaved to {OUT_DIR}/. Production model NOT modified; nothing adopted.", flush=True)


if __name__ == "__main__":
    main()
