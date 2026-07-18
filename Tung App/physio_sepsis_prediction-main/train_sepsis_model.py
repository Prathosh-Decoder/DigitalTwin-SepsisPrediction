#!/usr/bin/env python3
"""Codex baseline model for PhysioNet Challenge 2019 sepsis early warning."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.sepsis_features import (
    ADVANCED_CONFIG,
    DEFAULT_CONFIG,
    ENHANCED_CONFIG,
    LABEL_COLUMN,
    LAB_DYNAMICS_CONFIG,
    LITERATURE_TREND_CONFIG,
    TARGETED_ADVANCED_CONFIG,
    FeatureConfig,
    build_feature_frame,
    compact_feature_names,
    expected_feature_names,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "challenge-2019" / "training"
CODEX_ROOT = Path(__file__).resolve().parent
MODEL_DIR = CODEX_ROOT / "models"
OUTPUT_DIR = CODEX_ROOT / "output"


def patient_files(data_root: Path) -> list[Path]:
    files = sorted(path for path in data_root.rglob("*.psv") if not path.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No patient .psv files found under {data_root}")
    return files


def patient_label(path: Path) -> int:
    labels = pd.read_csv(path, sep="|", usecols=[LABEL_COLUMN])
    return int(pd.to_numeric(labels[LABEL_COLUMN], errors="coerce").fillna(0).max())


def stratified_patient_split(
    files: list[Path],
    test_size: float,
    validation_size: float,
    seed: int,
    max_patients: int | None,
) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    positives: list[Path] = []
    negatives: list[Path] = []
    for path in files:
        if patient_label(path) == 1:
            positives.append(path)
        else:
            negatives.append(path)

    rng.shuffle(positives)
    rng.shuffle(negatives)

    if max_patients is not None:
        positive_target = max(1, round(max_patients * len(positives) / len(files)))
        negative_target = max_patients - positive_target
        positives = positives[:positive_target]
        negatives = negatives[:negative_target]

    def split_group(group: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
        test_count = max(1, round(len(group) * test_size))
        validation_count = max(1, round(len(group) * validation_size))
        test = group[:test_count]
        validation = group[test_count : test_count + validation_count]
        train = group[test_count + validation_count :]
        return train, validation, test

    train_pos, validation_pos, test_pos = split_group(positives)
    train_neg, validation_neg, test_neg = split_group(negatives)
    train_files = train_pos + train_neg
    validation_files = validation_pos + validation_neg
    test_files = test_pos + test_neg
    rng.shuffle(train_files)
    rng.shuffle(validation_files)
    rng.shuffle(test_files)
    return train_files, validation_files, test_files


def load_feature_rows(
    files: list[Path],
    negative_sample_rate: float,
    seed: int,
    config: FeatureConfig = DEFAULT_CONFIG,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0
    positive_rows = 0

    for path in files:
        df = pd.read_csv(path, sep="|")
        features = build_feature_frame(df, config=config)
        labels = features.pop(LABEL_COLUMN).astype(int)
        total_rows += len(features)
        positive_rows += int(labels.sum())

        if negative_sample_rate < 1:
            keep_mask = labels.eq(1).to_numpy() | (rng.random(len(labels)) < negative_sample_rate)
            features = features.loc[keep_mask]
            labels = labels.loc[keep_mask]

        features[LABEL_COLUMN] = labels
        kept_rows += len(features)
        frames.append(features)

    table = pd.concat(frames, ignore_index=True)
    y = table.pop(LABEL_COLUMN).astype(int)
    X = table.reindex(columns=feature_names or expected_feature_names(config))
    stats = {
        "patients": len(files),
        "total_rows_before_sampling": total_rows,
        "rows_after_sampling": kept_rows,
        "positive_rows_before_sampling": positive_rows,
        "positive_row_rate_before_sampling": round(positive_rows / total_rows, 6) if total_rows else 0,
        "positive_rows_after_sampling": int(y.sum()),
        "positive_row_rate_after_sampling": round(float(y.mean()), 6) if len(y) else 0,
    }
    return X, y, stats


def make_model(model_type: str, seed: int) -> Any:
    if model_type == "hist_gbdt":
        return HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=160,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=seed,
            class_weight="balanced",
        )
    if model_type == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=-1,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_type == "extra_trees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_leaf=30,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Install it or choose --model-type hist_gbdt.")
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_estimators=500,
            learning_rate=0.045,
            max_depth=4,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            reg_alpha=0.1,
            n_jobs=-1,
            random_state=seed,
        )
    if model_type == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed. Install it or choose another --model-type.")
        return LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            reg_alpha=0.1,
            scale_pos_weight=3.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def get_feature_config(feature_set: str) -> FeatureConfig:
    if feature_set == "baseline":
        return DEFAULT_CONFIG
    if feature_set == "lab_dynamics":
        return LAB_DYNAMICS_CONFIG
    if feature_set == "literature_trend":
        return LITERATURE_TREND_CONFIG
    if feature_set in {"enhanced", "shap_compact"}:
        return ENHANCED_CONFIG
    if feature_set == "targeted_advanced":
        return TARGETED_ADVANCED_CONFIG
    if feature_set == "advanced":
        return ADVANCED_CONFIG
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def get_feature_names(feature_set: str, config: FeatureConfig) -> list[str]:
    if feature_set == "shap_compact":
        return compact_feature_names(config)
    return expected_feature_names(config)


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    return probabilities[:, 1]


def choose_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, {"f1": 0, "precision": 0, "recall": 0}
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1))
    threshold = float(thresholds[best_index])
    return threshold, {
        "f1": round(float(f1[best_index]), 4),
        "precision": round(float(precision[best_index]), 4),
        "recall": round(float(recall[best_index]), 4),
    }


def compute_prediction_utility_values(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: int = -12,
    dt_optimal: int = -6,
    dt_late: int = 3,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
) -> np.ndarray:
    """Official PhysioNet/CinC 2019 utility contribution for each patient-hour."""
    if np.any(labels):
        is_septic = True
        t_sepsis = int(np.argmax(labels)) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float("inf")

    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    utility = np.zeros(len(labels))
    for t in range(len(labels)):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    utility[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    utility[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                utility[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    utility[t] = 0
                elif t <= t_sepsis + dt_late:
                    utility[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                utility[t] = u_tn
    return utility


def compute_prediction_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: int = -12,
    dt_optimal: int = -6,
    dt_late: int = 3,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
) -> float:
    """Official PhysioNet/CinC 2019 per-patient utility function."""
    utility = compute_prediction_utility_values(
        labels,
        predictions,
        dt_early=dt_early,
        dt_optimal=dt_optimal,
        dt_late=dt_late,
        max_u_tp=max_u_tp,
        min_u_fn=min_u_fn,
        u_fp=u_fp,
        u_tn=u_tn,
    )
    return float(np.sum(utility))


def normalized_challenge_utility(cohort_labels: list[np.ndarray], cohort_predictions: list[np.ndarray]) -> dict[str, float]:
    observed_utilities = []
    best_utilities = []
    inaction_utilities = []

    for labels, observed_predictions in zip(cohort_labels, cohort_predictions):
        num_rows = len(labels)
        best_predictions = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if np.any(labels):
            t_sepsis = int(np.argmax(labels)) - (-6)
            best_predictions[max(0, t_sepsis - 12) : min(t_sepsis + 3 + 1, num_rows)] = 1

        observed_utilities.append(compute_prediction_utility(labels, observed_predictions))
        best_utilities.append(compute_prediction_utility(labels, best_predictions))
        inaction_utilities.append(compute_prediction_utility(labels, inaction_predictions))

    observed = float(np.sum(observed_utilities))
    best = float(np.sum(best_utilities))
    inaction = float(np.sum(inaction_utilities))
    normalized = (observed - inaction) / (best - inaction) if best != inaction else 0.0
    return {
        "utility": round(float(normalized), 4),
        "unnormalized_observed_utility": round(observed, 4),
        "unnormalized_best_utility": round(best, 4),
        "unnormalized_inaction_utility": round(inaction, 4),
    }


def evaluate(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    base_rate = float(np.mean(y_true))
    lift_at_k: dict[str, dict[str, float | int]] = {}
    order = np.argsort(-y_score)
    sorted_labels = np.asarray(y_true)[order]
    for fraction in (0.01, 0.05, 0.10, 0.20):
        top_n = max(1, int(np.ceil(len(sorted_labels) * fraction)))
        top_labels = sorted_labels[:top_n]
        top_rate = float(np.mean(top_labels))
        lift = top_rate / base_rate if base_rate > 0 else 0.0
        lift_at_k[f"top_{int(fraction * 100)}pct"] = {
            "rows": int(top_n),
            "positive_rate": round(top_rate, 6),
            "captured_positives": int(top_labels.sum()),
            "lift": round(float(lift), 4),
        }
    return {
        "auroc": round(float(roc_auc_score(y_true, y_score)), 4) if y_true.nunique() > 1 else None,
        "average_precision": round(float(average_precision_score(y_true, y_score)), 4) if y_true.nunique() > 1 else None,
        "brier_score": round(float(brier_score_loss(y_true, y_score)), 4),
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "positive_rate": round(base_rate, 6),
        "prediction_positive_rate": round(float(np.mean(y_pred)), 6),
        "lift_at_k": lift_at_k,
    }


def evaluate_patient_files(
    model: Any,
    files: list[Path],
    threshold: float,
    config: FeatureConfig = DEFAULT_CONFIG,
    feature_names: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_by_patient: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    lengths: list[int] = []

    for path in files:
        df = pd.read_csv(path, sep="|")
        features = build_feature_frame(df, config=config)
        labels = features.pop(LABEL_COLUMN).astype(int).to_numpy()
        X = features.reindex(columns=feature_names or expected_feature_names(config))

        labels_by_patient.append(labels)
        frames.append(X)
        lengths.append(len(labels))

    X_all = pd.concat(frames, ignore_index=True)
    y_true = pd.Series(np.concatenate(labels_by_patient))
    y_score = predict_proba(model, X_all)
    scores_by_patient = np.split(y_score, np.cumsum(lengths)[:-1])
    predictions_by_patient = [(scores >= threshold).astype(int) for scores in scores_by_patient]
    metrics = evaluate(y_true, y_score, threshold)
    utility = normalized_challenge_utility(labels_by_patient, predictions_by_patient)
    metrics.update(utility)
    stats = {
        "patients": len(files),
        "rows_after_sampling": int(len(y_true)),
        "positive_rows_after_sampling": int(y_true.sum()),
        "positive_row_rate_after_sampling": round(float(y_true.mean()), 6) if len(y_true) else 0,
    }
    return metrics, stats


def write_report(payload: dict[str, Any], path: Path) -> None:
    train_metrics = payload["metrics"]["train"]
    validation_metrics = payload["metrics"]["validation"]
    test_metrics = payload["metrics"]["test"]

    def metric_value(metrics: dict[str, Any], key: str) -> str:
        value = metrics[key]
        return "n/a" if value is None else str(value)

    def lift_rows(metrics: dict[str, Any]) -> str:
        return "\n".join(
            [
                "| Top 1% highest risk | {rows:,} | {rate:.2%} | {captured:,} | {lift}x |".format(
                    rows=metrics["lift_at_k"]["top_1pct"]["rows"],
                    rate=metrics["lift_at_k"]["top_1pct"]["positive_rate"],
                    captured=metrics["lift_at_k"]["top_1pct"]["captured_positives"],
                    lift=metrics["lift_at_k"]["top_1pct"]["lift"],
                ),
                "| Top 5% highest risk | {rows:,} | {rate:.2%} | {captured:,} | {lift}x |".format(
                    rows=metrics["lift_at_k"]["top_5pct"]["rows"],
                    rate=metrics["lift_at_k"]["top_5pct"]["positive_rate"],
                    captured=metrics["lift_at_k"]["top_5pct"]["captured_positives"],
                    lift=metrics["lift_at_k"]["top_5pct"]["lift"],
                ),
                "| Top 10% highest risk | {rows:,} | {rate:.2%} | {captured:,} | {lift}x |".format(
                    rows=metrics["lift_at_k"]["top_10pct"]["rows"],
                    rate=metrics["lift_at_k"]["top_10pct"]["positive_rate"],
                    captured=metrics["lift_at_k"]["top_10pct"]["captured_positives"],
                    lift=metrics["lift_at_k"]["top_10pct"]["lift"],
                ),
                "| Top 20% highest risk | {rows:,} | {rate:.2%} | {captured:,} | {lift}x |".format(
                    rows=metrics["lift_at_k"]["top_20pct"]["rows"],
                    rate=metrics["lift_at_k"]["top_20pct"]["positive_rate"],
                    captured=metrics["lift_at_k"]["top_20pct"]["captured_positives"],
                    lift=metrics["lift_at_k"]["top_20pct"]["lift"],
                ),
            ]
        )

    comparison_rows = "\n".join(
        [
            "| Train | {patients:,} | {rows:,} | {positive_rate:.2%} | {auroc} | {ap} | {brier} | {precision} | {recall} | {f1} | {lift}x | {utility} |".format(
                patients=payload["train_stats"]["patients"],
                rows=payload["train_stats"]["rows_after_sampling"],
                positive_rate=train_metrics["positive_rate"],
                auroc=metric_value(train_metrics, "auroc"),
                ap=metric_value(train_metrics, "average_precision"),
                brier=train_metrics["brier_score"],
                precision=train_metrics["precision"],
                recall=train_metrics["recall"],
                f1=train_metrics["f1"],
                lift=train_metrics["lift_at_k"]["top_10pct"]["lift"],
                utility=train_metrics["utility"],
            ),
            "| Validation | {patients:,} | {rows:,} | {positive_rate:.2%} | {auroc} | {ap} | {brier} | {precision} | {recall} | {f1} | {lift}x | {utility} |".format(
                patients=payload["validation_stats"]["patients"],
                rows=payload["validation_stats"]["rows_after_sampling"],
                positive_rate=validation_metrics["positive_rate"],
                auroc=metric_value(validation_metrics, "auroc"),
                ap=metric_value(validation_metrics, "average_precision"),
                brier=validation_metrics["brier_score"],
                precision=validation_metrics["precision"],
                recall=validation_metrics["recall"],
                f1=validation_metrics["f1"],
                lift=validation_metrics["lift_at_k"]["top_10pct"]["lift"],
                utility=validation_metrics["utility"],
            ),
            "| Test | {patients:,} | {rows:,} | {positive_rate:.2%} | {auroc} | {ap} | {brier} | {precision} | {recall} | {f1} | {lift}x | {utility} |".format(
                patients=payload["test_stats"]["patients"],
                rows=payload["test_stats"]["rows_after_sampling"],
                positive_rate=test_metrics["positive_rate"],
                auroc=metric_value(test_metrics, "auroc"),
                ap=metric_value(test_metrics, "average_precision"),
                brier=test_metrics["brier_score"],
                precision=test_metrics["precision"],
                recall=test_metrics["recall"],
                f1=test_metrics["f1"],
                lift=test_metrics["lift_at_k"]["top_10pct"]["lift"],
                utility=test_metrics["utility"],
            ),
        ]
    )
    feature_set = payload.get("feature_set", "baseline")
    feature_engineering_bullets = [
        "- Raw current vitals, labs, demographics, and `ICULOS`",
        "- Missingness indicators for each vital/lab channel",
        "- Forward-filled latest observed values from hours `<= t`",
        "- Hours since last measurement for each vital/lab channel",
        "- Vital deltas over 1 and 3 hours",
        "- Vital rolling mean/min/max over 3 and 6 hours",
        "- Log-transformed ICU hour and absolute hospital-to-ICU lag",
    ]
    if feature_set == "lab_dynamics":
        feature_engineering_bullets.extend(
            [
                "- Lab-result abnormality flags for Lactate, WBC, BUN, Creatinine, Platelets, Bilirubin_total, pH, PaCO2, HCO3, BaseExcess, and FiO2",
                "- Time since each key lab was last abnormal",
                "- Deltas and percentage deltas between observed lab measurements",
                "- Rolling observed lab min/max over the last three measured values",
                "- Lab-panel intensity and organ-system lab scores",
            ]
        )
    if feature_set in {"enhanced", "shap_compact", "targeted_advanced", "advanced"}:
        feature_engineering_bullets.extend(
            [
                "- Extended vital deltas over 6 and 12 hours",
                "- Extended vital rolling windows over 12 and 24 hours",
                "- Clinical derived features such as shock index, pulse pressure, SpO2/FiO2, qSOFA/SIRS flags, and lactate-high status",
            ]
        )
    if feature_set in {"targeted_advanced", "advanced"}:
        feature_engineering_bullets.extend(
            [
                "- Targeted measurement-intensity features for core vitals, FiO2, Lactate, WBC, BUN, and Creatinine",
                "- Targeted instability-burden counts for MAP, HR, respiratory rate, O2Sat, and temperature",
                "- Clinical summary scores and directional deterioration features over 6 hours",
            ]
        )
    if feature_set == "advanced":
        feature_engineering_bullets.append(
            "- Additional broad abnormality flags and broad measurement-density features for exploratory testing"
        )
    feature_engineering_text = "\n".join(feature_engineering_bullets)
    text = f"""# Codex Sepsis Early-Warning Baseline Model

## Prediction Task

For each ICU patient at each hour `t`, use only current and historical data up to `t` to predict:

```text
P(sepsis within the next 6 hours)
```

The target is `SepsisLabel`. PhysioNet shifted this label six hours early, so the model is trained as an hourly early-warning classifier.

## Feature Engineering

{feature_engineering_text}

All features are causal. No future row is used.

## Model

- Model type: `{payload["model_type"]}`
- Feature set: `{feature_set}`
- Train patients: {payload["train_stats"]["patients"]:,}
- Validation patients: {payload["validation_stats"]["patients"]:,}
- Test patients: {payload["test_stats"]["patients"]:,}
- Train rows used for fitting after negative sampling: {payload["train_fit_stats"]["rows_after_sampling"]:,}
- Train evaluation rows: {payload["train_stats"]["rows_after_sampling"]:,}
- Validation rows: {payload["validation_stats"]["rows_after_sampling"]:,}
- Test rows: {payload["test_stats"]["rows_after_sampling"]:,}
- Threshold selected on validation set: {test_metrics["threshold"]}

## Train / Validation / Test Metrics

| Split | Patients | Rows | Positive rate | AUROC | AUPRC | Brier | Precision | Recall | F1 | Lift@10% | Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{comparison_rows}

## Confusion Matrices

| Split | True positives | False positives | False negatives | True negatives |
| --- | ---: | ---: | ---: | ---: |
| Train | {train_metrics["confusion_matrix"]["tp"]} | {train_metrics["confusion_matrix"]["fp"]} | {train_metrics["confusion_matrix"]["fn"]} | {train_metrics["confusion_matrix"]["tn"]} |
| Validation | {validation_metrics["confusion_matrix"]["tp"]} | {validation_metrics["confusion_matrix"]["fp"]} | {validation_metrics["confusion_matrix"]["fn"]} | {validation_metrics["confusion_matrix"]["tn"]} |
| Test | {test_metrics["confusion_matrix"]["tp"]} | {test_metrics["confusion_matrix"]["fp"]} | {test_metrics["confusion_matrix"]["fn"]} | {test_metrics["confusion_matrix"]["tn"]} |

## Lift Analysis

Lift shows how much richer the highest-risk ranked rows are compared with random review. A lift of `5x` means that reviewed group has five times the baseline sepsis-positive rate.

### Validation Lift

Baseline positive row rate in the validation set: {validation_metrics["positive_rate"]:.2%}

| Review group | Rows | Positive row rate | Captured positives | Lift |
| --- | ---: | ---: | ---: | ---: |
{lift_rows(validation_metrics)}

### Test Lift

Baseline positive row rate in the test set: {test_metrics["positive_rate"]:.2%}

| Review group | Rows | Positive row rate | Captured positives | Lift |
| --- | ---: | ---: | ---: | ---: |
{lift_rows(test_metrics)}

## Official PhysioNet Utility

The official PhysioNet/CinC Challenge 2019 score is `Utility`, a custom metric for early sepsis warning. It is designed around clinical timing and alarm burden, not raw classification accuracy. The Utility values above are computed on complete patient time series for each split.

Utility behavior:

- Correct sepsis alarms in the early warning window before clinical onset receive positive reward.
- Earlier useful warnings receive more value than late warnings because they create more treatment time.
- Late alarms and missed septic cases are penalized.
- False alarms on non-septic patients are penalized, reflecting alarm fatigue.
- Correctly staying quiet on non-septic patients receives zero reward and zero penalty.

The score is summed over patient-hours and normalized so `1.0` is an optimal early-warning predictor and `0.0` is an inactive model that never alarms. Negative utility means the alert policy is worse than doing nothing.

For example, a Utility of `0.375` would mean the model captures about 37.5% of the possible value between an inactive model and a perfect early-warning model. This can sound modest, but the official first-ranked team paper reported a full-test Utility score of `0.360`, so Utility scores in this range are meaningful for this challenge.

Utility best reflects the digital twin goal: early, well-timed warnings without excessive false alarms.

Official metric source: https://physionet.org/content/challenge-2019/1.0.0/
First-ranked team paper: https://physionet.org/files/challenge-2019/1.0.0/papers/CinC2019-014.pdf

## Notes

- Patient-level train/test split is used to avoid leakage across rows from the same patient.
- The target is highly imbalanced. AUPRC, recall, calibration, and false-alarm burden are more informative than accuracy.
- This Codex baseline is intentionally isolated from the main API/Claude work under `Codex_model/`.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--model-type",
        choices=["hist_gbdt", "logistic", "extra_trees", "xgboost", "lightgbm"],
        default="hist_gbdt",
    )
    parser.add_argument(
        "--feature-set",
        choices=[
            "baseline",
            "lab_dynamics",
            "literature_trend",
            "enhanced",
            "shap_compact",
            "targeted_advanced",
            "advanced",
        ],
        default="baseline",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--negative-sample-rate", type=float, default=0.08)
    parser.add_argument("--max-patients", type=int, default=5000, help="Use 0 for all patients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "sepsis_baseline.joblib")
    parser.add_argument("--metrics-path", type=Path, default=OUTPUT_DIR / "sepsis_baseline_metrics.json")
    parser.add_argument("--report-path", type=Path, default=OUTPUT_DIR / "sepsis_model_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.max_patients is not None and args.max_patients <= 0:
        args.max_patients = None

    feature_config = get_feature_config(args.feature_set)
    feature_names = get_feature_names(args.feature_set, feature_config)
    files = patient_files(args.data_root)
    train_files, validation_files, test_files = stratified_patient_split(
        files,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
        max_patients=args.max_patients,
    )
    print(
        "Training patients: {train:,}; validation patients: {validation:,}; testing patients: {test:,}".format(
            train=len(train_files),
            validation=len(validation_files),
            test=len(test_files),
        )
    )

    X_train, y_train, train_stats = load_feature_rows(
        train_files,
        negative_sample_rate=args.negative_sample_rate,
        seed=args.seed,
        config=feature_config,
        feature_names=feature_names,
    )
    X_validation, y_validation, _ = load_feature_rows(
        validation_files,
        negative_sample_rate=1.0,
        seed=args.seed + 1,
        config=feature_config,
        feature_names=feature_names,
    )
    print(f"Train rows: {len(X_train):,}; positives: {int(y_train.sum()):,} ({y_train.mean():.3%})")
    print(
        f"Validation rows: {len(X_validation):,}; positives: {int(y_validation.sum()):,} ({y_validation.mean():.3%})"
    )

    model = make_model(args.model_type, args.seed)
    model.fit(X_train, y_train)

    validation_scores = predict_proba(model, X_validation)
    threshold, threshold_stats = choose_threshold(y_validation, validation_scores)
    train_metrics, train_eval_stats = evaluate_patient_files(
        model,
        train_files,
        threshold,
        config=feature_config,
        feature_names=feature_names,
    )
    validation_metrics, validation_eval_stats = evaluate_patient_files(
        model,
        validation_files,
        threshold,
        config=feature_config,
        feature_names=feature_names,
    )
    test_metrics, test_eval_stats = evaluate_patient_files(
        model,
        test_files,
        threshold,
        config=feature_config,
        feature_names=feature_names,
    )
    print(
        "Test rows: {rows:,}; positives: {positive:,} ({rate:.3%})".format(
            rows=test_eval_stats["rows_after_sampling"],
            positive=test_eval_stats["positive_rows_after_sampling"],
            rate=test_eval_stats["positive_row_rate_after_sampling"],
        )
    )
    metrics = {
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "threshold": threshold,
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "feature_config": asdict(feature_config),
        "training_seed": args.seed,
        "negative_sample_rate": args.negative_sample_rate,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "target": LABEL_COLUMN,
        "prediction_task": "Hourly probability of sepsis within the next 6 hours",
    }
    joblib.dump(artifact, args.model_path)

    payload = {
        "model_type": args.model_type,
        "model_path": str(args.model_path),
        "feature_set": args.feature_set,
        "feature_count": len(feature_names),
        "train_fit_stats": train_stats,
        "train_stats": train_eval_stats,
        "validation_stats": validation_eval_stats,
        "test_stats": test_eval_stats,
        "threshold_selection_validation": threshold_stats,
        "metrics": metrics,
    }
    args.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, args.report_path)

    print(f"Saved model -> {args.model_path}")
    print(f"Saved metrics -> {args.metrics_path}")
    print(f"Saved report -> {args.report_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
