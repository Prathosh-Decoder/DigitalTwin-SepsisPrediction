"""LOCAL STUDY ONLY -- does not touch artifacts/model_bundle.joblib, config.py, or anything
committed/pushed. Two experiments, both reusing the exact features + U1-U0 target as production:

  (a) Standalone Optuna search (broad LightGBM space, objective = validation AUPRC) on the fixed
      train/val split, best config then scored on the LOCKED test set.
  (b) Nested cross-validation (StratifiedGroupKFold outer folds, inner Optuna per fold) for an
      honest, generalization-robust estimate of AUROC/AUPRC/Utility.

Writes only to experiments/optuna_nested_cv/. Nothing is adopted or pushed.

Usage:
    python3 experiments/optuna_nested_cv.py
"""
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import sepsis_utils

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = config.BASE_DIR / "experiments" / "optuna_nested_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = config.RANDOM_SEED
N_TRIALS_STANDALONE = 50
N_TRIALS_INNER = 20
N_OUTER_FOLDS = 5
EARLY_STOP = config.EARLY_STOPPING_ROUNDS

# Fixed LightGBM params not part of the search (kept as in production).
FIXED_PARAMS = dict(
    boosting_type="gbdt",
    subsample_freq=1,
    subsample_for_bin=200000,
    random_state=SEED,
    n_jobs=8,
    verbosity=-1,
)


def suggest_params(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 200.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 200.0, log=True),
        subsample=trial.suggest_float("subsample", 0.3, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
    )


def train_one(params, X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMRegressor(**FIXED_PARAMS, **params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def load_data():
    features = pd.read_parquet(config.FEATURES_PARQUET)
    features = sepsis_utils.add_row_utilities(features)
    feature_cols = [c for c in features.columns
                    if c not in ("patient_id", "hospital", config.LABEL_COL,
                                 "row_u1", "row_u0", "row_target_u_diff")]
    return features, feature_cols


# ---------------------------------------------------------------------------
# (a) Standalone Optuna search on the fixed split
# ---------------------------------------------------------------------------

def run_standalone(features, feature_cols):
    train_ids = set(json.loads(config.TRAIN_IDS_PATH.read_text()))
    val_ids = set(json.loads(config.VAL_IDS_PATH.read_text()))
    test_ids = set(json.loads(config.TEST_IDS_PATH.read_text()))

    tr = features[features["patient_id"].isin(train_ids)]
    va = features[features["patient_id"].isin(val_ids)]
    te = features[features["patient_id"].isin(test_ids)]

    X_tr, y_tr = tr[feature_cols], tr["row_target_u_diff"]
    X_va, y_va = va[feature_cols], va["row_target_u_diff"]
    X_te = te[feature_cols]
    val_label = va[config.LABEL_COL].to_numpy()

    def objective(trial):
        params = suggest_params(trial)
        model = train_one(params, X_tr, y_tr, X_va, y_va)
        val_score = model.predict(X_va)
        return average_precision_score(val_label, val_score)  # AUPRC = discrimination target

    print(f"[standalone] Optuna search, {N_TRIALS_STANDALONE} trials, objective=validation AUPRC ...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS_STANDALONE, show_progress_bar=False)
    print(f"[standalone] done in {time.time()-t0:.0f}s, best val AUPRC={study.best_value:.4f}")

    best_params = study.best_params
    model = train_one(best_params, X_tr, y_tr, X_va, y_va)

    # Select threshold on validation (utility-optimal), evaluate on test -- same as production.
    val_score = model.predict(X_va)
    best_threshold, _, _ = sepsis_utils.threshold_grid_search(
        val_score, va["row_u1"].to_numpy(), va["row_u0"].to_numpy())
    test_score = model.predict(X_te)
    m = sepsis_utils.evaluate_predictions(
        te[config.LABEL_COL].to_numpy(), test_score,
        te["row_u1"].to_numpy(), te["row_u0"].to_numpy(), best_threshold)
    print(f"[standalone] TEST: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
          f"Recall={m['recall']:.4f} Utility={m['utility']:.4f}")
    return {"best_params": best_params, "best_val_auprc": study.best_value,
            "threshold": best_threshold, "test_metrics": m}


# ---------------------------------------------------------------------------
# (b) Nested CV
# ---------------------------------------------------------------------------

def run_nested_cv(features, feature_cols):
    patient_table = (features.groupby("patient_id")[config.LABEL_COL].max()
                     .rename("ever_septic").reset_index())
    pid_to_ever = dict(zip(patient_table["patient_id"], patient_table["ever_septic"]))

    all_pids = patient_table["patient_id"].to_numpy()
    y_strat = patient_table["ever_septic"].to_numpy()
    groups = all_pids  # each patient is its own group

    sgkf = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_pidx, test_pidx) in enumerate(sgkf.split(all_pids, y_strat, groups), 1):
        outer_train_pids = set(all_pids[train_pidx])
        outer_test_pids = set(all_pids[test_pidx])

        # carve an inner validation split (patient-level, stratified) from outer-train
        otr_pids = np.array(sorted(outer_train_pids))
        otr_strat = np.array([pid_to_ever[p] for p in otr_pids])
        inner_tr_pids, inner_va_pids = train_test_split(
            otr_pids, test_size=0.1111, stratify=otr_strat, random_state=SEED)  # ~10% of full
        inner_tr_pids, inner_va_pids = set(inner_tr_pids), set(inner_va_pids)

        tr = features[features["patient_id"].isin(inner_tr_pids)]
        va = features[features["patient_id"].isin(inner_va_pids)]
        te = features[features["patient_id"].isin(outer_test_pids)]

        X_tr, y_tr = tr[feature_cols], tr["row_target_u_diff"]
        X_va, y_va = va[feature_cols], va["row_target_u_diff"]
        X_te = te[feature_cols]
        val_label = va[config.LABEL_COL].to_numpy()

        def objective(trial):
            params = suggest_params(trial)
            model = train_one(params, X_tr, y_tr, X_va, y_va)
            return average_precision_score(val_label, model.predict(X_va))

        t0 = time.time()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=N_TRIALS_INNER, show_progress_bar=False)

        model = train_one(study.best_params, X_tr, y_tr, X_va, y_va)
        val_score = model.predict(X_va)
        best_threshold, _, _ = sepsis_utils.threshold_grid_search(
            val_score, va["row_u1"].to_numpy(), va["row_u0"].to_numpy())
        test_score = model.predict(X_te)
        m = sepsis_utils.evaluate_predictions(
            te[config.LABEL_COL].to_numpy(), test_score,
            te["row_u1"].to_numpy(), te["row_u0"].to_numpy(), best_threshold)
        fold_metrics.append(m)
        print(f"[nested cv] fold {fold}/{N_OUTER_FOLDS} done in {time.time()-t0:.0f}s: "
              f"AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} Utility={m['utility']:.4f}")

    def agg(key):
        vals = [fm[key] for fm in fold_metrics]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {k: agg(k) for k in ["auroc", "auprc", "precision", "recall", "f1", "lift_at_10pct", "utility"]}
    print("[nested cv] SUMMARY (mean +/- std across outer folds):")
    for k, v in summary.items():
        print(f"    {k:15s} {v['mean']:.4f} +/- {v['std']:.4f}")
    return {"per_fold": fold_metrics, "summary": summary}


def main():
    print(f"Loading {config.FEATURES_PARQUET} ...")
    features, feature_cols = load_data()
    print(f"{len(features):,} rows, {len(feature_cols)} features.")

    before = json.loads(config.METRICS_PATH.read_text())["test"]
    print(f"\nCurrent production TEST: AUROC={before['auroc']:.4f} AUPRC={before['auprc']:.4f} "
          f"Recall={before['recall']:.4f} Utility={before['utility']:.4f}\n")

    standalone = run_standalone(features, feature_cols)
    print()
    nested = run_nested_cv(features, feature_cols)

    results = {"production_test": before, "standalone_optuna": standalone, "nested_cv": nested}
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))

    # comparison table
    so = standalone["test_metrics"]
    ns = nested["summary"]
    print("\n=== COMPARISON (test set) ===")
    print(f"{'':30s} {'AUROC':>8s} {'AUPRC':>8s} {'Recall':>8s} {'Utility':>8s}")
    print(f"{'Current production':30s} {before['auroc']:8.4f} {before['auprc']:8.4f} "
          f"{before['recall']:8.4f} {before['utility']:8.4f}")
    print(f"{'Optuna-best (single split)':30s} {so['auroc']:8.4f} {so['auprc']:8.4f} "
          f"{so['recall']:8.4f} {so['utility']:8.4f}")
    print(f"{'Nested CV (mean)':30s} {ns['auroc']['mean']:8.4f} {ns['auprc']['mean']:8.4f} "
          f"{ns['recall']['mean']:8.4f} {ns['utility']['mean']:8.4f}")
    print(f"\nSaved to {OUT_DIR}/results.json. Production model NOT modified.")


if __name__ == "__main__":
    main()
