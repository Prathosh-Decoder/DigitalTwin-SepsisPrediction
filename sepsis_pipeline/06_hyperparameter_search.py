"""Light grid search over n_estimators / num_leaves / learning_rate, evaluated on
the EXISTING train/val split (test set is never touched). Reports whether any
combination beats the current published-hyperparameter baseline on validation
utility; does not modify config.py or retrain the production model itself.

Usage:
    python3 06_hyperparameter_search.py
"""
import csv
import itertools
import time

import lightgbm as lgb
import pandas as pd

import config
import sepsis_utils
from feature_engineering import get_feature_columns

N_ESTIMATORS_GRID = [100, 200, 300]
NUM_LEAVES_GRID = [31, 49, 70]
LEARNING_RATE_GRID = [0.05, 0.1, 0.15]


def main():
    print(f"Loading {config.FEATURES_PARQUET} ...")
    features = pd.read_parquet(config.FEATURES_PARQUET)
    features = sepsis_utils.add_row_utilities(features)

    feature_cols = get_feature_columns(features)
    feature_cols = [c for c in feature_cols if c not in ("row_u1", "row_u0", "row_target_u_diff")]

    import json
    train_ids = set(json.loads(config.TRAIN_IDS_PATH.read_text()))
    val_ids = set(json.loads(config.VAL_IDS_PATH.read_text()))

    train_df = features[features["patient_id"].isin(train_ids)]
    val_df = features[features["patient_id"].isin(val_ids)]
    print(f"Rows: {len(train_df):,} train / {len(val_df):,} val (same split as production model).")

    X_train, y_train = train_df[feature_cols], train_df["row_target_u_diff"]
    X_val, y_val = val_df[feature_cols], val_df["row_target_u_diff"]
    val_label = val_df[config.LABEL_COL].to_numpy()
    val_u1 = val_df["row_u1"].to_numpy()
    val_u0 = val_df["row_u0"].to_numpy()

    combos = list(itertools.product(N_ESTIMATORS_GRID, NUM_LEAVES_GRID, LEARNING_RATE_GRID))
    print(f"Searching {len(combos)} combinations ...")

    results = []
    for i, (n_estimators, num_leaves, learning_rate) in enumerate(combos, 1):
        params = dict(config.LGBM_PARAMS)
        params.update(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate)

        t0 = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        elapsed = time.time() - t0

        val_score = model.predict(X_val)
        best_threshold, best_utility, _ = sepsis_utils.threshold_grid_search(val_score, val_u1, val_u0)
        m = sepsis_utils.evaluate_predictions(val_label, val_score, val_u1, val_u0, best_threshold)

        row = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "best_iteration": model.best_iteration_,
            "val_utility": best_utility,
            "val_auroc": m["auroc"],
            "val_auprc": m["auprc"],
            "val_recall": m["recall"],
            "val_lift_at_10pct": m["lift_at_10pct"],
            "threshold": best_threshold,
            "seconds": round(elapsed, 1),
        }
        results.append(row)
        is_baseline = (n_estimators, num_leaves, learning_rate) == (100, 49, 0.1)
        tag = " <- current baseline" if is_baseline else ""
        print(f"[{i:2d}/{len(combos)}] n_estimators={n_estimators:3d} num_leaves={num_leaves:2d} "
              f"lr={learning_rate:.2f} best_iter={model.best_iteration_:3d} "
              f"val_utility={best_utility:.4f} val_auroc={m['auroc']:.4f} ({elapsed:.1f}s){tag}")

    results.sort(key=lambda r: r["val_utility"], reverse=True)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.ARTIFACTS_DIR / "hyperparam_search_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved full results to {out_path}")

    baseline = next(r for r in results if (r["n_estimators"], r["num_leaves"], r["learning_rate"]) == (100, 49, 0.1))
    baseline_rank = results.index(baseline) + 1

    print("\nTop 5 combinations by validation utility:")
    for rank, r in enumerate(results[:5], 1):
        print(f"  #{rank}: n_estimators={r['n_estimators']}, num_leaves={r['num_leaves']}, "
              f"learning_rate={r['learning_rate']} -> val_utility={r['val_utility']:.4f}, "
              f"val_auroc={r['val_auroc']:.4f}")

    print(f"\nCurrent baseline (n_estimators=100, num_leaves=49, learning_rate=0.1) "
          f"ranks #{baseline_rank} of {len(results)} with val_utility={baseline['val_utility']:.4f}")

    best = results[0]
    if best["val_utility"] > baseline["val_utility"]:
        improvement = best["val_utility"] - baseline["val_utility"]
        print(f"\n>>> Best combination beats baseline by {improvement:+.4f} validation utility.")
        print(f">>> n_estimators={best['n_estimators']}, num_leaves={best['num_leaves']}, "
              f"learning_rate={best['learning_rate']}")
    else:
        print("\n>>> No combination in this grid beats the current baseline. "
              "Published (mimicked) hyperparameters already look locally optimal here.")


if __name__ == "__main__":
    main()
