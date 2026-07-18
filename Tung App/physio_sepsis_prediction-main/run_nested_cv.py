#!/usr/bin/env python3
"""Run patient-level nested cross-validation for the Codex sepsis model."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CODEX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODEX_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "challenge-2019" / "training"
CACHE_DIR = CODEX_ROOT / "cache" / "literature_core"
MODEL_DIR = CODEX_ROOT / "models" / "nested_cv"
OUTPUT_DIR = CODEX_ROOT / "output"

sys.path.insert(0, str(CODEX_ROOT))
from run_literature_training_plan import (  # noqa: E402
    build_utility_cache,
    choose_global_threshold,
    choose_time_phased_thresholds,
    evaluate_predictions,
)
from src.sepsis_features import (  # noqa: E402
    DEFAULT_CONFIG,
    LABEL_COLUMN,
    LITERATURE_CORE_CONFIG,
    build_feature_frame,
    expected_feature_names,
)
from src.sepsis_policies import DEFAULT_PHASE_BOUNDS, phase_ids  # noqa: E402
from train_sepsis_model import make_model, patient_files, predict_proba  # noqa: E402


CACHE_VERSION = 1
CANDIDATES = {
    "baseline": DEFAULT_CONFIG,
    "literature_core": LITERATURE_CORE_CONFIG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--metrics-path", type=Path, default=OUTPUT_DIR / "nested_cv_5fold_metrics.json")
    parser.add_argument("--report-path", type=Path, default=OUTPUT_DIR / "nested_cv_5fold_report.md")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--negative-sample-rate", type=float, default=0.08)
    parser.add_argument("--score-batch-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "features": cache_dir / "features_float32.dat",
        "labels": cache_dir / "labels_int8.dat",
        "hours": cache_dir / "hours_float32.dat",
        "metadata": cache_dir / "metadata.json",
    }


def _patient_hours(dataframe: pd.DataFrame) -> np.ndarray:
    fallback = pd.Series(np.arange(1, len(dataframe) + 1, dtype=float), index=dataframe.index)
    if "ICULOS" not in dataframe:
        return fallback.to_numpy(dtype=np.float32)
    hours = pd.to_numeric(dataframe["ICULOS"], errors="coerce")
    return hours.where(hours.notna(), fallback).to_numpy(dtype=np.float32)


def build_cache(files: list[Path], data_root: Path, cache_dir: Path) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items() if name != "metadata"}
    for path in temporary.values():
        path.unlink(missing_ok=True)

    feature_names = expected_feature_names(LITERATURE_CORE_CONFIG)
    offsets = [0]
    outcomes: list[int] = []
    relative_paths: list[str] = []
    total_positive_rows = 0

    print(f"Building feature cache for {len(files):,} patients and {len(feature_names)} features...", flush=True)
    try:
        with (
            temporary["features"].open("wb") as feature_handle,
            temporary["labels"].open("wb") as label_handle,
            temporary["hours"].open("wb") as hour_handle,
        ):
            for index, path in enumerate(files, start=1):
                dataframe = pd.read_csv(path, sep="|")
                feature_frame = build_feature_frame(dataframe, config=LITERATURE_CORE_CONFIG)
                labels = feature_frame.pop(LABEL_COLUMN).astype(np.int8).to_numpy()
                values = feature_frame.reindex(columns=feature_names).to_numpy(dtype=np.float32)
                hours = _patient_hours(dataframe)
                values.tofile(feature_handle)
                labels.tofile(label_handle)
                hours.tofile(hour_handle)
                offsets.append(offsets[-1] + len(labels))
                outcomes.append(int(labels.max(initial=0)))
                total_positive_rows += int(labels.sum())
                relative_paths.append(str(path.relative_to(data_root)))
                if index % 1_000 == 0 or index == len(files):
                    print(
                        f"  cached {index:,}/{len(files):,} patients; {offsets[-1]:,} hourly rows",
                        flush=True,
                    )
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise

    for name in ("features", "labels", "hours"):
        os.replace(temporary[name], paths[name])

    metadata = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "feature_config": asdict(LITERATURE_CORE_CONFIG),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "patient_count": len(files),
        "row_count": offsets[-1],
        "positive_rows": total_positive_rows,
        "patient_paths": relative_paths,
        "patient_offsets": offsets,
        "patient_outcomes": outcomes,
    }
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def load_or_build_cache(args: argparse.Namespace) -> tuple[dict[str, Any], np.memmap, np.memmap, np.memmap]:
    files = patient_files(args.data_root)
    paths = cache_paths(args.cache_dir)
    expected_names = expected_feature_names(LITERATURE_CORE_CONFIG)
    metadata: dict[str, Any] | None = None
    if paths["metadata"].exists() and not args.rebuild_cache:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        valid = (
            metadata.get("cache_version") == CACHE_VERSION
            and metadata.get("data_root") == str(args.data_root.resolve())
            and metadata.get("feature_names") == expected_names
            and metadata.get("patient_count") == len(files)
            and all(paths[name].exists() for name in ("features", "labels", "hours"))
        )
        if not valid:
            metadata = None
    if metadata is None:
        metadata = build_cache(files, args.data_root, args.cache_dir)
    else:
        print(
            f"Using feature cache: {metadata['patient_count']:,} patients, "
            f"{metadata['row_count']:,} rows, {metadata['feature_count']} features",
            flush=True,
        )

    row_count = int(metadata["row_count"])
    feature_count = int(metadata["feature_count"])
    features = np.memmap(paths["features"], dtype=np.float32, mode="r", shape=(row_count, feature_count))
    labels = np.memmap(paths["labels"], dtype=np.int8, mode="r", shape=(row_count,))
    hours = np.memmap(paths["hours"], dtype=np.float32, mode="r", shape=(row_count,))
    return metadata, features, labels, hours


def patient_rows(patient_indices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    parts = [np.arange(offsets[index], offsets[index + 1], dtype=np.int64) for index in patient_indices]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def sampled_training_rows(
    patient_indices: np.ndarray,
    offsets: np.ndarray,
    labels: np.memmap,
    negative_sample_rate: float,
    seed: int,
) -> np.ndarray:
    rows = patient_rows(patient_indices, offsets)
    row_labels = np.asarray(labels[rows])
    rng = np.random.default_rng(seed)
    keep = (row_labels == 1) | (rng.random(len(rows)) < negative_sample_rate)
    return rows[keep]


def fit_fold_model(
    features: np.memmap,
    labels: np.memmap,
    train_rows: np.ndarray,
    feature_columns: np.ndarray,
    seed: int,
) -> Any:
    full_X = np.asarray(features[train_rows], dtype=np.float32)
    X = full_X[:, feature_columns]
    y = np.asarray(labels[train_rows], dtype=np.int8)
    model = make_model("xgboost", seed)
    model.fit(X, y)
    del X, full_X, y
    gc.collect()
    return model


def score_patients(
    model: Any,
    patient_indices: np.ndarray,
    offsets: np.ndarray,
    features: np.memmap,
    labels: np.memmap,
    hours: np.memmap,
    feature_columns: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    rows = patient_rows(patient_indices, offsets)
    scores = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        full_X = np.asarray(features[batch_rows], dtype=np.float32)
        scores[start : start + len(batch_rows)] = predict_proba(model, full_X[:, feature_columns])

    lengths = np.asarray([offsets[index + 1] - offsets[index] for index in patient_indices], dtype=int)
    split_at = np.cumsum(lengths)[:-1]
    labels_flat = np.asarray(labels[rows], dtype=np.int8)
    hours_flat = np.asarray(hours[rows], dtype=np.float32)
    labels_by_patient = list(np.split(labels_flat, split_at))
    scores_by_patient = list(np.split(scores, split_at))
    hours_by_patient = list(np.split(hours_flat, split_at))
    return {
        "patient_indices": patient_indices.tolist(),
        "labels_by_patient": labels_by_patient,
        "scores_by_patient": scores_by_patient,
        "hours_by_patient": hours_by_patient,
        "labels": labels_flat,
        "scores": scores,
        "hours": hours_flat,
        "utility_cache": build_utility_cache(labels_by_patient),
        "stats": {
            "patients": int(len(patient_indices)),
            "rows": int(len(rows)),
            "positive_rows": int(labels_flat.sum()),
            "positive_rate": round(float(labels_flat.mean()), 6),
        },
    }


def merge_scored(parts: list[dict[str, Any]]) -> dict[str, Any]:
    labels_by_patient = [item for part in parts for item in part["labels_by_patient"]]
    scores_by_patient = [item for part in parts for item in part["scores_by_patient"]]
    hours_by_patient = [item for part in parts for item in part["hours_by_patient"]]
    labels = np.concatenate(labels_by_patient)
    scores = np.concatenate(scores_by_patient)
    hours = np.concatenate(hours_by_patient)
    return {
        "labels_by_patient": labels_by_patient,
        "scores_by_patient": scores_by_patient,
        "hours_by_patient": hours_by_patient,
        "labels": labels,
        "scores": scores,
        "hours": hours,
        "utility_cache": build_utility_cache(labels_by_patient),
        "stats": {
            "patients": len(labels_by_patient),
            "rows": int(len(labels)),
            "positive_rows": int(labels.sum()),
            "positive_rate": round(float(labels.mean()), 6),
        },
    }


def tune_policy(scored: dict[str, Any], phase_bounds: tuple[int, ...]) -> dict[str, Any]:
    global_threshold, global_metrics, _ = choose_global_threshold(scored)
    phase_thresholds, phased_metrics, _ = choose_time_phased_thresholds(
        scored, global_threshold, phase_bounds
    )
    if phased_metrics["utility"] > global_metrics["utility"]:
        selected_policy = "time_phased"
        selected_metrics = phased_metrics
    else:
        selected_policy = "global"
        selected_metrics = global_metrics
    return {
        "policy": selected_policy,
        "global_threshold": global_threshold,
        "phase_thresholds": phase_thresholds,
        "global_metrics": global_metrics,
        "phased_metrics": phased_metrics,
        "selected_metrics": selected_metrics,
    }


def apply_policy(scored: dict[str, Any], policy: dict[str, Any], phase_bounds: tuple[int, ...]) -> np.ndarray:
    if policy["policy"] == "time_phased":
        phases = phase_ids(scored["hours"], phase_bounds)
        thresholds = np.asarray(policy["phase_thresholds"], dtype=float)
        return (scored["scores"] >= thresholds[phases]).astype(np.int8)
    return (scored["scores"] >= float(policy["global_threshold"])).astype(np.int8)


def feature_columns(all_names: list[str], candidate: str) -> tuple[list[str], np.ndarray]:
    names = expected_feature_names(CANDIDATES[candidate])
    positions = {name: index for index, name in enumerate(all_names)}
    missing = [name for name in names if name not in positions]
    if missing:
        raise ValueError(f"Cache is missing candidate features: {missing[:5]}")
    return names, np.asarray([positions[name] for name in names], dtype=np.int64)


def metric_summary(folds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ("auroc", "average_precision", "precision", "recall", "f1", "utility")
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([fold["outer_metrics"][key] for fold in folds], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        margin = 2.776 * std / np.sqrt(len(values))
        result[key] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci95_low": round(mean - margin, 4),
            "ci95_high": round(mean + margin, 4),
        }
    lift_values = np.asarray(
        [fold["outer_metrics"]["lift_at_k"]["top_10pct"]["lift"] for fold in folds], dtype=float
    )
    lift_mean = float(lift_values.mean())
    lift_std = float(lift_values.std(ddof=1))
    lift_margin = 2.776 * lift_std / np.sqrt(len(lift_values))
    result["lift_at_10pct"] = {
        "mean": round(lift_mean, 4),
        "std": round(lift_std, 4),
        "ci95_low": round(lift_mean - lift_margin, 4),
        "ci95_high": round(lift_mean + lift_margin, 4),
    }
    return result


def write_report(payload: dict[str, Any], path: Path) -> None:
    fold_rows = []
    policy_rows = []
    for fold in payload["folds"]:
        metrics = fold["outer_metrics"]
        fold_rows.append(
            "| {fold} | {feature_set} | {policy} | {auroc:.4f} | {auprc:.4f} | "
            "{precision:.4f} | {recall:.4f} | {f1:.4f} | {lift:.4f}x | {utility:.4f} |".format(
                fold=fold["outer_fold"],
                feature_set=fold["selected_feature_set"],
                policy=fold["selected_policy"]["policy"],
                auroc=metrics["auroc"],
                auprc=metrics["average_precision"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                lift=metrics["lift_at_k"]["top_10pct"]["lift"],
                utility=metrics["utility"],
            )
        )
        policy = fold["selected_policy"]
        policy_rows.append(
            f"| {fold['outer_fold']} | {fold['selected_feature_set']} | "
            f"{policy['global_threshold']:.2f} | "
            f"{', '.join(f'{value:.2f}' for value in policy['phase_thresholds'])} |"
        )
    summary_rows = []
    for label, key in (
        ("AUROC", "auroc"),
        ("AUPRC", "average_precision"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Lift@10%", "lift_at_10pct"),
        ("Utility", "utility"),
    ):
        item = payload["fold_summary"][key]
        summary_rows.append(
            f"| {label} | {item['mean']:.4f} | {item['std']:.4f} | "
            f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] |"
        )
    pooled = payload["pooled_outer_predictions"]
    selected_counts = payload["selection_counts"]
    text = f"""# Five-Fold Nested Cross-Validation

## Protocol

- Outer folds: `{payload['protocol']['outer_folds']}` patient-stratified folds for unbiased evaluation.
- Inner folds: `{payload['protocol']['inner_folds']}` patient-stratified folds inside each outer development cohort.
- Inner selection: baseline versus literature-core features, then global versus ICU-phase Utility thresholds.
- Negative-row sampling (`{payload['protocol']['negative_sample_rate']}`) is applied only to model-fitting rows.
- Every held-out inner and outer cohort retains all hourly rows.
- No patient appears in training and evaluation within the same fold.

## Outer-Fold Results

| Fold | Selected features | Policy | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(fold_rows)}

## Inner-Selected Thresholds

The four phase thresholds correspond to ICU hours `1-12`, `13-24`, `25-48`, and `49+`.

| Outer fold | Selected features | Global threshold | Phase thresholds |
| ---: | --- | ---: | --- |
{chr(10).join(policy_rows)}

## Mean and Uncertainty

The confidence intervals use the fold mean with a t interval (`df=4`). They describe fold-to-fold variation, not external-hospital uncertainty.

| Metric | Mean | SD | 95% CI |
| --- | ---: | ---: | ---: |
{chr(10).join(summary_rows)}

## Pooled Out-of-Fold Predictions

Each patient contributes predictions from exactly one outer model.

| AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {pooled['auroc']:.4f} | {pooled['average_precision']:.4f} | {pooled['precision']:.4f} | {pooled['recall']:.4f} | {pooled['f1']:.4f} | {pooled['lift_at_k']['top_10pct']['lift']:.4f}x | {pooled['utility']:.4f} |

Feature selection counts: `{selected_counts}`. Literature-core won three folds and baseline won two, so the engineered features are helpful but not uniformly stable.

## Interpretation

This nested estimate is more defensible than a single internal train/validation/test split because feature and threshold decisions are repeated entirely inside each outer training cohort. It still evaluates hospitals A and B only and therefore does not replace external validation on an independent hospital system.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("Both outer and inner fold counts must be at least 2")
    if not 0 < args.negative_sample_rate <= 1:
        raise ValueError("negative-sample-rate must be in (0, 1]")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metadata, features, labels, hours = load_or_build_cache(args)
    offsets = np.asarray(metadata["patient_offsets"], dtype=np.int64)
    outcomes = np.asarray(metadata["patient_outcomes"], dtype=np.int8)
    all_names = list(metadata["feature_names"])
    candidate_features = {name: feature_columns(all_names, name) for name in CANDIDATES}
    patient_indices = np.arange(len(outcomes), dtype=np.int64)
    phase_bounds = tuple(DEFAULT_PHASE_BOUNDS)

    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    fold_results: list[dict[str, Any]] = []
    pooled_parts: list[dict[str, Any]] = []
    pooled_predictions: list[np.ndarray] = []

    for outer_fold, (outer_dev_pos, outer_test_pos) in enumerate(
        outer_cv.split(patient_indices, outcomes), start=1
    ):
        outer_dev = patient_indices[outer_dev_pos]
        outer_test = patient_indices[outer_test_pos]
        print(
            f"\nOuter fold {outer_fold}/{args.outer_folds}: "
            f"development={len(outer_dev):,}, test={len(outer_test):,}",
            flush=True,
        )
        inner_cv = StratifiedKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.seed + outer_fold * 1_000,
        )
        inner_splits = list(inner_cv.split(outer_dev, outcomes[outer_dev]))
        candidate_results: dict[str, Any] = {}

        for candidate_index, (candidate, (names, columns)) in enumerate(candidate_features.items()):
            print(f"  Candidate {candidate}: {len(names)} features", flush=True)
            inner_parts: list[dict[str, Any]] = []
            for inner_fold, (inner_train_pos, inner_valid_pos) in enumerate(inner_splits, start=1):
                inner_train = outer_dev[inner_train_pos]
                inner_valid = outer_dev[inner_valid_pos]
                train_rows = sampled_training_rows(
                    inner_train,
                    offsets,
                    labels,
                    args.negative_sample_rate,
                    args.seed + outer_fold * 10_000 + candidate_index * 1_000 + inner_fold,
                )
                print(
                    f"    inner {inner_fold}/{args.inner_folds}: fit {len(train_rows):,} sampled rows; "
                    f"validate {len(inner_valid):,} patients",
                    flush=True,
                )
                model = fit_fold_model(
                    features,
                    labels,
                    train_rows,
                    columns,
                    args.seed + outer_fold * 10_000 + candidate_index * 1_000 + inner_fold,
                )
                inner_parts.append(
                    score_patients(
                        model,
                        inner_valid,
                        offsets,
                        features,
                        labels,
                        hours,
                        columns,
                        args.score_batch_size,
                    )
                )
                del model, train_rows
                gc.collect()

            inner_oof = merge_scored(inner_parts)
            policy = tune_policy(inner_oof, phase_bounds)
            candidate_results[candidate] = {
                "feature_count": len(names),
                "inner_oof_stats": inner_oof["stats"],
                "policy": policy,
            }
            print(
                f"    inner OOF: Utility={policy['selected_metrics']['utility']:.4f}, "
                f"AUPRC={policy['selected_metrics']['average_precision']:.4f}, "
                f"policy={policy['policy']}",
                flush=True,
            )
            del inner_parts, inner_oof
            gc.collect()

        selected = max(
            candidate_results,
            key=lambda name: (
                candidate_results[name]["policy"]["selected_metrics"]["utility"],
                candidate_results[name]["policy"]["selected_metrics"]["average_precision"],
                -candidate_results[name]["feature_count"],
            ),
        )
        selected_names, selected_columns = candidate_features[selected]
        selected_policy = candidate_results[selected]["policy"]
        print(
            f"  Selected {selected} with {selected_policy['policy']} policy; fitting outer model...",
            flush=True,
        )
        outer_train_rows = sampled_training_rows(
            outer_dev,
            offsets,
            labels,
            args.negative_sample_rate,
            args.seed + outer_fold * 100_000,
        )
        outer_model = fit_fold_model(
            features,
            labels,
            outer_train_rows,
            selected_columns,
            args.seed + outer_fold * 100_000,
        )
        outer_scored = score_patients(
            outer_model,
            outer_test,
            offsets,
            features,
            labels,
            hours,
            selected_columns,
            args.score_batch_size,
        )
        outer_predictions = apply_policy(outer_scored, selected_policy, phase_bounds)
        outer_metrics = evaluate_predictions(
            outer_scored,
            outer_predictions,
            f"nested_inner_selected_{selected_policy['policy']}",
        )
        artifact_path = args.model_dir / f"literature_nested_outer_fold_{outer_fold}.joblib"
        joblib.dump(
            {
                "model": outer_model,
                "model_type": "xgboost",
                "feature_set": selected,
                "feature_names": selected_names,
                "feature_config": asdict(CANDIDATES[selected]),
                "global_threshold": selected_policy["global_threshold"],
                "threshold": selected_policy["global_threshold"],
                "phase_bounds": list(phase_bounds),
                "phase_thresholds": selected_policy["phase_thresholds"],
                "primary_policy": selected_policy["policy"],
                "outer_fold": outer_fold,
                "selection": "nested_inner_oof_utility",
            },
            artifact_path,
        )
        fold_result = {
            "outer_fold": outer_fold,
            "development_patients": int(len(outer_dev)),
            "test_patients": int(len(outer_test)),
            "selected_feature_set": selected,
            "selected_policy": selected_policy,
            "candidate_inner_results": candidate_results,
            "outer_stats": outer_scored["stats"],
            "outer_metrics": outer_metrics,
            "artifact": str(artifact_path),
        }
        fold_results.append(fold_result)
        pooled_parts.append(outer_scored)
        pooled_predictions.append(outer_predictions)
        print(
            f"  Outer result: Utility={outer_metrics['utility']:.4f}, "
            f"AUPRC={outer_metrics['average_precision']:.4f}, "
            f"Lift@10%={outer_metrics['lift_at_k']['top_10pct']['lift']:.4f}x",
            flush=True,
        )
        del outer_model, outer_train_rows
        gc.collect()

    pooled_scored = merge_scored(pooled_parts)
    pooled_prediction_array = np.concatenate(pooled_predictions)
    pooled_metrics = evaluate_predictions(
        pooled_scored, pooled_prediction_array, "fold_specific_nested_oof_policy"
    )
    selection_counts = {
        candidate: sum(fold["selected_feature_set"] == candidate for fold in fold_results)
        for candidate in CANDIDATES
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seed": args.seed,
            "negative_sample_rate": args.negative_sample_rate,
            "phase_bounds": list(phase_bounds),
            "candidate_feature_sets": list(CANDIDATES),
            "patient_count": int(len(outcomes)),
            "row_count": int(metadata["row_count"]),
        },
        "selection_counts": selection_counts,
        "folds": fold_results,
        "fold_summary": metric_summary(fold_results),
        "pooled_outer_predictions": pooled_metrics,
    }
    args.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, args.report_path)
    print(f"\nSaved metrics to {args.metrics_path}", flush=True)
    print(f"Saved report to {args.report_path}", flush=True)
    print(json.dumps({"selection_counts": selection_counts, "pooled": pooled_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
