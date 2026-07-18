#!/usr/bin/env python3
"""Train, calibrate, and package the final six-hour sepsis predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

CODEX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODEX_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "challenge-2019" / "training"
CACHE_DIR = CODEX_ROOT / "cache" / "literature_core"
NESTED_METRICS = CODEX_ROOT / "output" / "nested_cv_5fold_metrics.json"
NESTED_MODEL_DIR = CODEX_ROOT / "models" / "nested_cv"
OUTPUT_MODEL = CODEX_ROOT / "models" / "sepsis_next_6h_predictor.pkl"
OUTPUT_REPORT = CODEX_ROOT / "output" / "sepsis_next_6h_predictor_report.md"
OUTPUT_METADATA = CODEX_ROOT / "models" / "sepsis_next_6h_predictor_metadata.json"

sys.path.insert(0, str(CODEX_ROOT))
from run_literature_training_plan import build_utility_cache, evaluate_predictions  # noqa: E402
from run_nested_cv import (  # noqa: E402
    apply_policy,
    feature_columns,
    fit_fold_model,
    load_or_build_cache,
    merge_scored,
    patient_rows,
    sampled_training_rows,
    score_patients,
)
from src.sepsis_features import LITERATURE_CORE_CONFIG, expected_feature_names  # noqa: E402
from src.sepsis_policies import DEFAULT_PHASE_BOUNDS, phase_ids  # noqa: E402
from src.sepsis_predictor import SepsisRiskPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--nested-metrics", type=Path, default=NESTED_METRICS)
    parser.add_argument("--nested-model-dir", type=Path, default=NESTED_MODEL_DIR)
    parser.add_argument("--output-model", type=Path, default=OUTPUT_MODEL)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--output-metadata", type=Path, default=OUTPUT_METADATA)
    parser.add_argument("--negative-sample-rate", type=float, default=0.08)
    parser.add_argument("--score-batch-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def raw_to_logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_calibrator(scores: np.ndarray, labels: np.ndarray, seed: int) -> LogisticRegression:
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=seed)
    calibrator.fit(raw_to_logit(scores), labels)
    return calibrator


def calibration_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": round(float(roc_auc_score(labels, scores)), 6),
        "auprc": round(float(average_precision_score(labels, scores)), 6),
        "brier_score": round(float(brier_score_loss(labels, scores)), 6),
        "log_loss": round(float(log_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7))), 6),
        "mean_probability": round(float(np.mean(scores)), 6),
        "observed_positive_rate": round(float(np.mean(labels)), 6),
    }


def calibrated_threshold_grid() -> np.ndarray:
    fine = np.arange(0.001, 0.101, 0.001)
    coarse = np.arange(0.105, 0.501, 0.005)
    return np.unique(np.round(np.concatenate([fine, coarse]), 4))


def tune_calibrated_policy(scored: dict[str, Any]) -> dict[str, Any]:
    grid = calibrated_threshold_grid()
    best_global = 0.5
    best_global_utility = -float("inf")
    for threshold in grid:
        predictions = (scored["scores"] >= threshold).astype(np.int8)
        utility = float(scored["utility_cache"].normalized(predictions))
        if utility > best_global_utility:
            best_global = float(threshold)
            best_global_utility = utility

    bounds = tuple(DEFAULT_PHASE_BOUNDS)
    phases = phase_ids(scored["hours"], bounds)
    thresholds = np.repeat(best_global, len(bounds) + 1)
    for phase in range(len(thresholds)):
        best_threshold = float(thresholds[phase])
        best_utility = -float("inf")
        for threshold in grid:
            candidate = thresholds.copy()
            candidate[phase] = threshold
            predictions = (scored["scores"] >= candidate[phases]).astype(np.int8)
            utility = float(scored["utility_cache"].normalized(predictions))
            if utility > best_utility:
                best_threshold = float(threshold)
                best_utility = utility
        thresholds[phase] = best_threshold

    global_predictions = (scored["scores"] >= best_global).astype(np.int8)
    phased_predictions = (scored["scores"] >= thresholds[phases]).astype(np.int8)
    global_metrics = evaluate_predictions(scored, global_predictions, "calibrated_global_oof_selected")
    phased_metrics = evaluate_predictions(scored, phased_predictions, "calibrated_phased_oof_selected")
    return {
        "global_threshold": best_global,
        "phase_bounds": list(bounds),
        "phase_thresholds": thresholds.round(4).tolist(),
        "global_oof_metrics_after_policy_tuning": global_metrics,
        "phased_oof_metrics_after_policy_tuning": phased_metrics,
    }


def collect_nested_oof(
    nested_metrics: dict[str, Any],
    metadata: dict[str, Any],
    features: np.memmap,
    labels: np.memmap,
    hours: np.memmap,
    model_dir: Path,
    score_batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    import joblib

    offsets = np.asarray(metadata["patient_offsets"], dtype=np.int64)
    outcomes = np.asarray(metadata["patient_outcomes"], dtype=np.int8)
    all_names = list(metadata["feature_names"])
    patient_indices = np.arange(len(outcomes), dtype=np.int64)
    outer_cv = StratifiedKFold(
        n_splits=int(nested_metrics["protocol"]["outer_folds"]), shuffle=True, random_state=seed
    )
    parts: list[dict[str, Any]] = []
    for fold, (_, test_positions) in enumerate(outer_cv.split(patient_indices, outcomes), start=1):
        artifact = joblib.load(model_dir / f"literature_nested_outer_fold_{fold}.joblib")
        positions = {name: index for index, name in enumerate(all_names)}
        columns = np.asarray([positions[name] for name in artifact["feature_names"]], dtype=np.int64)
        test_patients = patient_indices[test_positions]
        print(f"Scoring nested outer fold {fold} for calibration...", flush=True)
        parts.append(
            score_patients(
                artifact["model"],
                test_patients,
                offsets,
                features,
                labels,
                hours,
                columns,
                score_batch_size,
            )
        )
    return parts


def cross_fitted_calibration(parts: list[dict[str, Any]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    calibrated_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for held_out in range(len(parts)):
        train_scores = np.concatenate([part["scores"] for index, part in enumerate(parts) if index != held_out])
        train_labels = np.concatenate([part["labels"] for index, part in enumerate(parts) if index != held_out])
        calibrator = fit_calibrator(train_scores, train_labels, seed + held_out)
        calibrated = calibrator.predict_proba(raw_to_logit(parts[held_out]["scores"]))[:, 1]
        calibrated_parts.append(calibrated)
        label_parts.append(parts[held_out]["labels"])
    return np.concatenate(calibrated_parts), np.concatenate(label_parts)


def write_report(payload: dict[str, Any], path: Path) -> None:
    calibration = payload["calibration"]
    policy = payload["deployment_policy"]
    nested = payload["nested_cv_reference"]
    text = f"""# Deployable Six-Hour Sepsis Predictor

## Artifact

`{payload['artifact']}`

The pickle contains one `SepsisRiskPredictor` object. It accepts a patient DataFrame, a PSV/CSV path, a list of hourly records, or a dictionary of columns.

## Probability Calibration

Calibration was learned from patient-level nested out-of-fold scores. The calibration evaluation below is cross-fitted: each fold was calibrated without using that fold's labels.

| Score | AUROC | AUPRC | Brier | Log loss | Mean probability | Positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw XGBoost | {calibration['raw_oof']['auroc']:.4f} | {calibration['raw_oof']['auprc']:.4f} | {calibration['raw_oof']['brier_score']:.4f} | {calibration['raw_oof']['log_loss']:.4f} | {calibration['raw_oof']['mean_probability']:.4f} | {calibration['raw_oof']['observed_positive_rate']:.4f} |
| Calibrated | {calibration['cross_fitted_calibrated_oof']['auroc']:.4f} | {calibration['cross_fitted_calibrated_oof']['auprc']:.4f} | {calibration['cross_fitted_calibrated_oof']['brier_score']:.4f} | {calibration['cross_fitted_calibrated_oof']['log_loss']:.4f} | {calibration['cross_fitted_calibrated_oof']['mean_probability']:.4f} | {calibration['cross_fitted_calibrated_oof']['observed_positive_rate']:.4f} |

## Deployment Policy

- Global calibrated threshold: `{policy['global_threshold']}`
- ICU phase bounds: `{policy['phase_bounds']}`
- Calibrated phase thresholds: `{policy['phase_thresholds']}`
- Nested-CV Utility reference: `{nested['pooled_utility']}` with 95% fold interval `{nested['utility_ci95']}`

The deployment thresholds were tuned on pooled cross-fitted out-of-fold probabilities after the nested evaluation. They are operational settings, not a new unbiased performance estimate.

## Python Usage

```python
import pickle

with open("sepsis_next_6h_predictor.pkl", "rb") as handle:
    predictor = pickle.load(handle)

result = predictor.predict_patient(patient_dataframe)
probability = predictor.predict_proba(patient_dataframe)
trajectory = predictor.predict_trajectory(patient_dataframe)
```

## Target Definition

The probability estimates the PhysioNet Challenge hourly sepsis target, whose label begins six hours before clinical sepsis onset and remains positive afterward. Before onset, this functions as a next-six-hour warning probability; it is not a pure incident-onset label after the patient has already become septic.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    nested_metrics = json.loads(args.nested_metrics.read_text(encoding="utf-8"))
    cache_args = SimpleNamespace(data_root=args.data_root, cache_dir=args.cache_dir, rebuild_cache=False)
    metadata, features, labels, hours = load_or_build_cache(cache_args)
    oof_parts = collect_nested_oof(
        nested_metrics,
        metadata,
        features,
        labels,
        hours,
        args.nested_model_dir,
        args.score_batch_size,
        args.seed,
    )
    oof_scored = merge_scored(oof_parts)
    raw_oof_metrics = calibration_metrics(oof_scored["labels"], oof_scored["scores"])
    cross_fitted_probabilities, cross_fitted_labels = cross_fitted_calibration(oof_parts, args.seed)
    calibrated_oof_metrics = calibration_metrics(cross_fitted_labels, cross_fitted_probabilities)

    final_calibrator = fit_calibrator(oof_scored["scores"], oof_scored["labels"], args.seed)
    calibrated_scored = dict(oof_scored)
    calibrated_scores = final_calibrator.predict_proba(raw_to_logit(oof_scored["scores"]))[:, 1]
    calibrated_scored["scores"] = calibrated_scores
    calibrated_scored["scores_by_patient"] = list(
        np.split(calibrated_scores, np.cumsum([len(x) for x in oof_scored["labels_by_patient"]])[:-1])
    )
    calibrated_scored["utility_cache"] = build_utility_cache(oof_scored["labels_by_patient"])
    deployment_policy = tune_calibrated_policy(calibrated_scored)

    offsets = np.asarray(metadata["patient_offsets"], dtype=np.int64)
    all_patients = np.arange(int(metadata["patient_count"]), dtype=np.int64)
    all_names = list(metadata["feature_names"])
    final_names = expected_feature_names(LITERATURE_CORE_CONFIG)
    positions = {name: index for index, name in enumerate(all_names)}
    final_columns = np.asarray([positions[name] for name in final_names], dtype=np.int64)
    train_rows = sampled_training_rows(
        all_patients,
        offsets,
        labels,
        args.negative_sample_rate,
        args.seed + 900_000,
    )
    print(f"Training final literature-core model on {len(train_rows):,} sampled rows...", flush=True)
    final_model = fit_fold_model(
        features,
        labels,
        train_rows,
        final_columns,
        args.seed + 900_000,
    )

    model_metadata = {
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBoost with sigmoid probability calibration",
        "feature_set": "literature_core",
        "feature_count": len(final_names),
        "training_patients": int(metadata["patient_count"]),
        "training_hourly_rows_before_sampling": int(metadata["row_count"]),
        "negative_sample_rate_for_model_fit": args.negative_sample_rate,
        "calibration_source": "five-fold nested out-of-fold patient predictions",
        "target_definition": (
            "Probability of the PhysioNet Challenge hourly sepsis label, shifted six hours before "
            "clinical onset and remaining positive afterward."
        ),
        "nested_cv_pooled_utility": nested_metrics["pooled_outer_predictions"]["utility"],
        "nested_cv_utility_ci95": [
            nested_metrics["fold_summary"]["utility"]["ci95_low"],
            nested_metrics["fold_summary"]["utility"]["ci95_high"],
        ],
        "required_packages": {
            "python": ">=3.11",
            "cloudpickle": ">=3.1",
            "numpy": ">=2.0",
            "pandas": ">=2.0",
            "scikit-learn": ">=1.5",
            "xgboost": ">=3.0",
        },
    }
    predictor = SepsisRiskPredictor(
        model=final_model,
        calibrator=final_calibrator,
        feature_names=final_names,
        feature_config=LITERATURE_CORE_CONFIG,
        global_threshold=deployment_policy["global_threshold"],
        phase_bounds=tuple(deployment_policy["phase_bounds"]),
        phase_thresholds=tuple(deployment_policy["phase_thresholds"]),
        metadata=model_metadata,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    import src.sepsis_features as sepsis_features_module
    import src.sepsis_policies as sepsis_policies_module
    import src.sepsis_predictor as sepsis_predictor_module

    cloudpickle.register_pickle_by_value(sepsis_features_module)
    cloudpickle.register_pickle_by_value(sepsis_policies_module)
    cloudpickle.register_pickle_by_value(sepsis_predictor_module)
    with args.output_model.open("wb") as handle:
        cloudpickle.dump(predictor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    artifact_sha256 = hashlib.sha256(args.output_model.read_bytes()).hexdigest()
    try:
        artifact_display_path = str(args.output_model.resolve().relative_to(CODEX_ROOT.resolve()))
    except ValueError:
        artifact_display_path = str(args.output_model)

    payload = {
        "artifact": artifact_display_path,
        "artifact_bytes": args.output_model.stat().st_size,
        "artifact_sha256": artifact_sha256,
        "model_metadata": model_metadata,
        "calibration": {
            "raw_oof": raw_oof_metrics,
            "cross_fitted_calibrated_oof": calibrated_oof_metrics,
        },
        "deployment_policy": deployment_policy,
        "nested_cv_reference": {
            "pooled_utility": nested_metrics["pooled_outer_predictions"]["utility"],
            "utility_ci95": model_metadata["nested_cv_utility_ci95"],
        },
    }
    args.output_metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, args.output_report)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
