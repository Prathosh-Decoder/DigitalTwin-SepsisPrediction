"""Shared helpers: the official PhysioNet/CinC 2019 utility scoring function,
threshold selection, and small evaluation/plotting utilities.

Utility function reference: Reyna et al., "Early Prediction of Sepsis From Clinical
Data: The PhysioNet/Computing in Cardiology Challenge 2019," Crit Care Med 2020;
official reference implementation at
https://github.com/physionetchallenges/evaluation-2019/blob/master/evaluate_sepsis_score.py
This is our own from-scratch reimplementation of that public formula, not vendored code.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score,
)

import config

# ---------------------------------------------------------------------------
# Official utility function
# ---------------------------------------------------------------------------

def _tp_fn_scores(dt, u_fp=None, max_u_tp=None, min_u_fn=None):
    """Vectorized per-row utility for a SEPTIC patient's row at relative time
    dt = ICULOS - t_sepsis (hours). Returns (score_if_predict_1, score_if_predict_0).

    Piecewise-linear reward for a true positive ramps 0 -> max_u_tp between
    dt_early and dt_optimal, then decays back to 0 by dt_late. Piecewise-linear
    penalty for a false negative is 0 until dt_optimal, then ramps to min_u_fn by
    dt_late. Outside [dt_early, dt_late] there is no more reward/penalty for
    timing; predicting positive there still costs the flat false-alarm penalty.

    `u_fp`/`max_u_tp`/`min_u_fn` override the config.* constants for this call
    (used by the reward/penalty sensitivity sweeps in the notebook, to re-grade
    the already-trained model's EXISTING scores under different severity
    settings without retraining it). dt_early/dt_optimal/dt_late are NOT
    parameterized here -- dt_optimal is structurally tied to how t_sepsis
    itself is reconstructed from the label (see compute_t_sepsis_table), so
    changing it would change what "onset" means, not just reward severity.
    """
    dt = np.asarray(dt, dtype=float)
    u_fp = config.U_FP if u_fp is None else u_fp
    max_u_tp = config.MAX_U_TP if max_u_tp is None else max_u_tp
    min_u_fn = config.MIN_U_FN if min_u_fn is None else min_u_fn

    m1 = max_u_tp / (config.DT_OPTIMAL - config.DT_EARLY)
    b1 = -m1 * config.DT_EARLY
    m2 = -max_u_tp / (config.DT_LATE - config.DT_OPTIMAL)
    b2 = -m2 * config.DT_LATE
    m3 = min_u_fn / (config.DT_LATE - config.DT_OPTIMAL)
    b3 = -m3 * config.DT_OPTIMAL

    score_pred1 = np.full(dt.shape, u_fp, dtype=float)
    score_pred0 = np.full(dt.shape, config.U_TN, dtype=float)

    zone_ramp = (dt > config.DT_EARLY) & (dt <= config.DT_OPTIMAL)
    zone_decay = (dt > config.DT_OPTIMAL) & (dt <= config.DT_LATE)
    zone_late = dt > config.DT_LATE

    score_pred1[zone_ramp] = m1 * dt[zone_ramp] + b1
    score_pred0[zone_ramp] = 0.0

    score_pred1[zone_decay] = m2 * dt[zone_decay] + b2
    score_pred0[zone_decay] = m3 * dt[zone_decay] + b3

    score_pred1[zone_late] = 0.0
    score_pred0[zone_late] = 0.0

    return score_pred1, score_pred0


def compute_t_sepsis_table(df: pd.DataFrame) -> pd.Series:
    """Per-patient clinical sepsis onset hour (Sepsis-3 t_sepsis), derived from the
    already-6h-shifted SepsisLabel column: t_sepsis = first hour SepsisLabel==1, + 6.
    Patients who are never septic are absent from the returned Series (map -> NaN).
    """
    septic_rows = df.loc[df[config.LABEL_COL] == 1, ["patient_id", "ICULOS"]]
    first_positive = septic_rows.groupby("patient_id")["ICULOS"].min()
    return first_positive - config.DT_OPTIMAL  # DT_OPTIMAL is -6, so this is +6


def add_row_utilities(df: pd.DataFrame, u_fp=None, max_u_tp=None, min_u_fn=None) -> pd.DataFrame:
    """Adds row_u1 (utility if predicted positive), row_u0 (utility if predicted
    negative), and row_target_u_diff (= row_u1 - row_u0, the LightGBM regression
    target) to a copy of df. Requires columns: patient_id, ICULOS, SepsisLabel.
    Fully vectorized (no per-patient Python loop) via a t_sepsis lookup merge.

    `u_fp`/`max_u_tp`/`min_u_fn` override the config.* constants for this call
    (see _tp_fn_scores docstring).
    """
    df = df.copy()
    u_fp = config.U_FP if u_fp is None else u_fp
    t_sepsis_map = compute_t_sepsis_table(df)
    df["t_sepsis"] = df["patient_id"].map(t_sepsis_map)
    is_septic_row = df["t_sepsis"].notna().to_numpy()

    dt = (df["ICULOS"] - df["t_sepsis"]).to_numpy()

    u1 = np.full(len(df), u_fp, dtype=float)
    u0 = np.full(len(df), config.U_TN, dtype=float)
    if is_septic_row.any():
        u1_valid, u0_valid = _tp_fn_scores(
            dt[is_septic_row], u_fp=u_fp, max_u_tp=max_u_tp, min_u_fn=min_u_fn
        )
        u1[is_septic_row] = u1_valid
        u0[is_septic_row] = u0_valid

    df["row_u1"] = u1
    df["row_u0"] = u0
    df["row_target_u_diff"] = u1 - u0
    df = df.drop(columns=["t_sepsis"])
    return df


def normalized_utility(row_u1, row_u0, y_pred_binary) -> float:
    """Cohort-level normalized utility: sum per-row observed/inaction/best utility
    across ALL rows (equivalent to summing per-patient then combining, since the
    official score is additive), then normalize so 'always predict negative' = 0
    and an oracle with perfect timing = 1.
    """
    row_u1 = np.asarray(row_u1, dtype=float)
    row_u0 = np.asarray(row_u0, dtype=float)
    y_pred_binary = np.asarray(y_pred_binary)

    observed = np.where(y_pred_binary == 1, row_u1, row_u0).sum()
    inaction = row_u0.sum()
    best = np.maximum(row_u1, row_u0).sum()

    denom = best - inaction
    if denom == 0:
        return 0.0
    return float((observed - inaction) / denom)


def threshold_grid_search(y_score, row_u1, row_u0, grid=None):
    """Sweeps candidate decision thresholds and returns the one maximizing
    normalized_utility, plus the full sweep (for plotting/diagnostics).
    Functionally the same objective the winning team optimized with nevergrad;
    implemented here as a plain grid search since it's a 1-D search.
    """
    if grid is None:
        grid = np.arange(
            config.THRESHOLD_GRID_MIN,
            config.THRESHOLD_GRID_MAX + 1e-9,
            config.THRESHOLD_GRID_STEP,
        )
    y_score = np.asarray(y_score)
    sweep = []
    best_threshold, best_utility = float(grid[0]), -np.inf
    for thr in grid:
        pred = (y_score >= thr).astype(int)
        util = normalized_utility(row_u1, row_u0, pred)
        sweep.append((float(thr), util))
        if util > best_utility:
            best_utility = util
            best_threshold = float(thr)
    return best_threshold, best_utility, sweep


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def lift_at_k(y_true_binary, y_score, k_frac: float = 0.1) -> float:
    """Lift@k: (positive rate among the top k_frac highest-scored rows) /
    (overall positive rate). 1.0 = no better than random ranking."""
    y_true_binary = np.asarray(y_true_binary)
    y_score = np.asarray(y_score)
    n = len(y_true_binary)
    overall_rate = y_true_binary.mean()
    if n == 0 or overall_rate == 0:
        return float("nan")
    k = max(int(np.ceil(n * k_frac)), 1)
    top_k_idx = np.argsort(-y_score)[:k]
    return float(y_true_binary[top_k_idx].mean() / overall_rate)


def evaluate_predictions(y_true_binary, y_score, row_u1, row_u0, threshold) -> dict:
    """One consistent metrics dict, reused across train/val/test/hospital slices
    in 03_train_model.py and re-checked in 05_sanity_checks.py."""
    y_true_binary = np.asarray(y_true_binary)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    n_pos = int(y_true_binary.sum())
    n = len(y_true_binary)

    metrics = {
        "n_rows": n,
        "n_positive_rows": n_pos,
        "positive_rate": n_pos / n if n else float("nan"),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true_binary, y_score)) if 0 < n_pos < n else float("nan"),
        "auprc": float(average_precision_score(y_true_binary, y_score)) if 0 < n_pos < n else float("nan"),
        "utility": normalized_utility(row_u1, row_u0, y_pred),
        "precision": float(precision_score(y_true_binary, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_binary, y_pred, zero_division=0)),
        "lift_at_10pct": lift_at_k(y_true_binary, y_score, 0.1),
    }
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_roc_pr_threshold_plots(y_true_binary, y_score, threshold_sweep, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = str(out_dir)
    y_true_binary = np.asarray(y_true_binary)
    y_score = np.asarray(y_score)

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png", dpi=150)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true_binary, y_score)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pr_curve.png", dpi=150)
    plt.close()

    thr, util = zip(*threshold_sweep)
    plt.figure(figsize=(6, 4))
    plt.plot(thr, util)
    plt.xlabel("Decision threshold (on U1-U0 score)")
    plt.ylabel("Normalized utility (validation set)")
    plt.title("Threshold sweep")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/threshold_sweep.png", dpi=150)
    plt.close()
