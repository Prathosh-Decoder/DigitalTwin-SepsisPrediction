#!/usr/bin/env python3
"""Run leakage-safe literature feature ablations for sepsis early warning."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

CODEX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODEX_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "challenge-2019" / "training"
MODEL_DIR = CODEX_ROOT / "models"
OUTPUT_DIR = CODEX_ROOT / "output"

sys.path.insert(0, str(CODEX_ROOT))
from src.sepsis_features import (  # noqa: E402
    DEFAULT_CONFIG,
    LABEL_COLUMN,
    LAB_TRAJECTORY_CONFIG,
    LITERATURE_CORE_CONFIG,
    LITERATURE_TREND_CONFIG,
    PHYSIOLOGY_TREND_CONFIG,
    FeatureConfig,
    build_feature_frame,
    expected_feature_names,
)
from src.sepsis_policies import DEFAULT_PHASE_BOUNDS, phase_ids  # noqa: E402
from train_sepsis_model import (  # noqa: E402
    compute_prediction_utility_values,
    evaluate,
    load_feature_rows,
    make_model,
    patient_files,
    predict_proba,
    stratified_patient_split,
)


CANDIDATE_CONFIGS: dict[str, FeatureConfig] = {
    "baseline": DEFAULT_CONFIG,
    "physiology_trends": PHYSIOLOGY_TREND_CONFIG,
    "lab_trajectories": LAB_TRAJECTORY_CONFIG,
    "literature_core": LITERATURE_CORE_CONFIG,
    "literature_full": LITERATURE_TREND_CONFIG,
}


@dataclass
class UtilityCache:
    utility_if_zero: np.ndarray
    utility_if_one: np.ndarray
    best_utility: float
    inaction_utility: float

    def normalized(self, predictions: np.ndarray) -> float:
        predictions = np.asarray(predictions, dtype=bool)
        observed = float(np.where(predictions, self.utility_if_one, self.utility_if_zero).sum())
        denominator = self.best_utility - self.inaction_utility
        return (observed - self.inaction_utility) / denominator if denominator else 0.0


def threshold_grid() -> np.ndarray:
    low = np.arange(0.01, 0.301, 0.01)
    middle = np.arange(0.325, 0.501, 0.025)
    high = np.arange(0.55, 0.901, 0.05)
    return np.unique(np.round(np.concatenate([low, middle, high]), 4))


def build_utility_cache(labels_by_patient: list[np.ndarray]) -> UtilityCache:
    zero_parts: list[np.ndarray] = []
    one_parts: list[np.ndarray] = []
    best_total = 0.0
    inaction_total = 0.0
    for labels in labels_by_patient:
        zero_predictions = np.zeros(len(labels), dtype=np.int8)
        one_predictions = np.ones(len(labels), dtype=np.int8)
        utility_if_zero = compute_prediction_utility_values(labels, zero_predictions)
        utility_if_one = compute_prediction_utility_values(labels, one_predictions)
        zero_parts.append(utility_if_zero)
        one_parts.append(utility_if_one)
        inaction_total += float(utility_if_zero.sum())

        best_predictions = np.zeros(len(labels), dtype=np.int8)
        if np.any(labels):
            t_sepsis = int(np.argmax(labels)) + 6
            best_predictions[max(0, t_sepsis - 12) : min(t_sepsis + 4, len(labels))] = 1
        best_total += float(np.where(best_predictions, utility_if_one, utility_if_zero).sum())

    return UtilityCache(
        utility_if_zero=np.concatenate(zero_parts),
        utility_if_one=np.concatenate(one_parts),
        best_utility=best_total,
        inaction_utility=inaction_total,
    )


def _patient_hours(dataframe: pd.DataFrame) -> np.ndarray:
    fallback = pd.Series(np.arange(1, len(dataframe) + 1, dtype=float), index=dataframe.index)
    if "ICULOS" not in dataframe:
        return fallback.to_numpy()
    hours = pd.to_numeric(dataframe["ICULOS"], errors="coerce")
    return hours.where(hours.notna(), fallback).to_numpy(dtype=float)


def score_cohort(
    files: list[Path],
    model: Any,
    config: FeatureConfig,
    feature_names: list[str],
    batch_size: int,
) -> dict[str, Any]:
    labels_by_patient: list[np.ndarray] = []
    scores_by_patient: list[np.ndarray] = []
    hours_by_patient: list[np.ndarray] = []

    for start in range(0, len(files), batch_size):
        batch_files = files[start : start + batch_size]
        frames: list[pd.DataFrame] = []
        batch_labels: list[np.ndarray] = []
        batch_hours: list[np.ndarray] = []
        lengths: list[int] = []
        for path in batch_files:
            dataframe = pd.read_csv(path, sep="|")
            feature_frame = build_feature_frame(dataframe, config=config)
            labels = feature_frame.pop(LABEL_COLUMN).astype(int).to_numpy()
            frames.append(feature_frame.reindex(columns=feature_names))
            batch_labels.append(labels)
            batch_hours.append(_patient_hours(dataframe))
            lengths.append(len(labels))

        batch_X = pd.concat(frames, ignore_index=True)
        batch_scores = predict_proba(model, batch_X)
        split_scores = np.split(batch_scores, np.cumsum(lengths)[:-1])
        labels_by_patient.extend(batch_labels)
        scores_by_patient.extend(split_scores)
        hours_by_patient.extend(batch_hours)

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
            "patients": len(files),
            "rows": int(len(labels)),
            "positive_rows": int(labels.sum()),
            "positive_rate": round(float(labels.mean()), 6) if len(labels) else 0.0,
        },
    }


def evaluate_predictions(scored: dict[str, Any], predictions: np.ndarray, policy: str) -> dict[str, Any]:
    labels = pd.Series(scored["labels"])
    scores = scored["scores"]
    predictions = np.asarray(predictions, dtype=np.int8)
    metrics = evaluate(labels, scores, 0.5)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics.update(
        {
            "threshold": None,
            "policy": policy,
            "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
            "prediction_positive_rate": round(float(predictions.mean()), 6),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "utility": round(float(scored["utility_cache"].normalized(predictions)), 4),
        }
    )
    return metrics


def choose_global_threshold(scored: dict[str, Any]) -> tuple[float, dict[str, Any], list[dict[str, float]]]:
    best_threshold = 0.5
    best_utility = -float("inf")
    scan: list[dict[str, float]] = []
    for threshold in threshold_grid():
        predictions = (scored["scores"] >= threshold).astype(np.int8)
        utility = float(scored["utility_cache"].normalized(predictions))
        scan.append({"threshold": float(threshold), "utility": round(utility, 6)})
        if utility > best_utility:
            best_threshold = float(threshold)
            best_utility = utility
    predictions = (scored["scores"] >= best_threshold).astype(np.int8)
    metrics = evaluate_predictions(scored, predictions, f"global_threshold={best_threshold:.3f}")
    return best_threshold, metrics, scan


def choose_time_phased_thresholds(
    scored: dict[str, Any],
    initial_threshold: float,
    phase_bounds: tuple[int, ...] = DEFAULT_PHASE_BOUNDS,
) -> tuple[list[float], dict[str, Any], list[dict[str, float | int]]]:
    phases = phase_ids(scored["hours"], phase_bounds)
    thresholds = np.repeat(float(initial_threshold), len(phase_bounds) + 1)
    scan: list[dict[str, float | int]] = []

    # Utility is additive by patient-hour, so each phase threshold can be optimized independently.
    for phase in range(len(thresholds)):
        best_threshold = float(thresholds[phase])
        best_utility = -float("inf")
        for candidate in threshold_grid():
            candidate_thresholds = thresholds.copy()
            candidate_thresholds[phase] = candidate
            predictions = (scored["scores"] >= candidate_thresholds[phases]).astype(np.int8)
            utility = float(scored["utility_cache"].normalized(predictions))
            scan.append(
                {"phase": phase, "threshold": float(candidate), "utility": round(utility, 6)}
            )
            if utility > best_utility:
                best_threshold = float(candidate)
                best_utility = utility
        thresholds[phase] = best_threshold

    predictions = (scored["scores"] >= thresholds[phases]).astype(np.int8)
    policy = f"time_phased bounds={list(phase_bounds)} thresholds={thresholds.round(4).tolist()}"
    metrics = evaluate_predictions(scored, predictions, policy)
    return thresholds.round(4).tolist(), metrics, scan


def apply_global_policy(scored: dict[str, Any], threshold: float) -> np.ndarray:
    return (scored["scores"] >= threshold).astype(np.int8)


def apply_phased_policy(
    scored: dict[str, Any], thresholds: list[float], phase_bounds: tuple[int, ...]
) -> np.ndarray:
    phases = phase_ids(scored["hours"], phase_bounds)
    return (scored["scores"] >= np.asarray(thresholds)[phases]).astype(np.int8)


def train_candidate(
    name: str,
    config: FeatureConfig,
    train_files: list[Path],
    validation_files: list[Path],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path]:
    feature_names = expected_feature_names(config)
    print(f"[{name}] loading training rows for {len(feature_names)} features...", flush=True)
    X_train, y_train, train_stats = load_feature_rows(
        train_files,
        negative_sample_rate=args.negative_sample_rate,
        seed=args.seed,
        config=config,
        feature_names=feature_names,
    )
    print(f"[{name}] fitting XGBoost on {len(X_train):,} sampled rows...", flush=True)
    model = make_model("xgboost", args.seed)
    model.fit(X_train, y_train)
    del X_train, y_train
    gc.collect()

    print(f"[{name}] scoring validation cohort...", flush=True)
    validation_scored = score_cohort(
        validation_files, model, config, feature_names, batch_size=args.score_batch_size
    )
    global_threshold, global_metrics, global_scan = choose_global_threshold(validation_scored)
    phase_thresholds, phased_metrics, phased_scan = choose_time_phased_thresholds(
        validation_scored, global_threshold, tuple(args.phase_bounds)
    )

    model_path = MODEL_DIR / f"sepsis_xgboost_{name}_validation_selected.joblib"
    artifact = {
        "model": model,
        "model_type": "xgboost",
        "feature_set": name,
        "feature_names": feature_names,
        "feature_config": asdict(config),
        "global_threshold": global_threshold,
        "threshold": global_threshold,
        "phase_bounds": list(args.phase_bounds),
        "phase_thresholds": phase_thresholds,
        "threshold_selection": "validation_physionet_utility",
        "training_seed": args.seed,
        "negative_sample_rate": args.negative_sample_rate,
        "target": LABEL_COLUMN,
        "prediction_task": "Hourly probability of sepsis within the next 6 hours",
    }
    joblib.dump(artifact, model_path)

    result = {
        "feature_set": name,
        "feature_count": len(feature_names),
        "train_fit_stats": train_stats,
        "validation_stats": validation_scored["stats"],
        "global_threshold": global_threshold,
        "phase_bounds": list(args.phase_bounds),
        "phase_thresholds": phase_thresholds,
        "validation_global": global_metrics,
        "validation_phased": phased_metrics,
        "global_threshold_scan": global_scan,
        "phase_threshold_scan": phased_scan,
        "model_path": str(model_path),
    }
    print(
        f"[{name}] validation Utility global={global_metrics['utility']:.4f}; "
        f"phased={phased_metrics['utility']:.4f}; AUPRC={phased_metrics['average_precision']:.4f}",
        flush=True,
    )
    del validation_scored, model
    gc.collect()
    return result, model_path


def site_name(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent.endswith("a"):
        return "A"
    if parent.endswith("b"):
        return "B"
    return "unknown"


def run_site_holdout(
    feature_set: str,
    config: FeatureConfig,
    development_files: list[Path],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    feature_names = expected_feature_names(config)
    results: list[dict[str, Any]] = []
    for train_site, validation_site in (("A", "B"), ("B", "A")):
        site_train_files = [path for path in development_files if site_name(path) == train_site]
        site_validation_files = [path for path in development_files if site_name(path) == validation_site]
        print(
            f"[site {feature_set}] train {train_site} ({len(site_train_files):,}) -> "
            f"validate {validation_site} ({len(site_validation_files):,})",
            flush=True,
        )
        X_train, y_train, train_stats = load_feature_rows(
            site_train_files,
            negative_sample_rate=args.negative_sample_rate,
            seed=args.seed,
            config=config,
            feature_names=feature_names,
        )
        model = make_model("xgboost", args.seed)
        model.fit(X_train, y_train)
        del X_train, y_train
        gc.collect()

        validation_scored = score_cohort(
            site_validation_files, model, config, feature_names, batch_size=args.score_batch_size
        )
        global_threshold, global_metrics, _ = choose_global_threshold(validation_scored)
        phase_thresholds, phased_metrics, _ = choose_time_phased_thresholds(
            validation_scored, global_threshold, tuple(args.phase_bounds)
        )
        results.append(
            {
                "feature_set": feature_set,
                "train_site": train_site,
                "validation_site": validation_site,
                "train_fit_stats": train_stats,
                "validation_stats": validation_scored["stats"],
                "global_threshold": global_threshold,
                "phase_thresholds": phase_thresholds,
                "global_metrics": global_metrics,
                "phased_metrics": phased_metrics,
            }
        )
        del validation_scored, model
        gc.collect()
    return results


def metric_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {metrics['auroc']} | {metrics['average_precision']} | "
        f"{metrics['precision']} | {metrics['recall']} | {metrics['f1']} | "
        f"{metrics['lift_at_k']['top_10pct']['lift']}x | {metrics['utility']} |"
    )


def write_report(payload: dict[str, Any], path: Path) -> None:
    ablation_rows = []
    for row in payload["validation_ablation"]:
        metrics = row["validation_phased"]
        ablation_rows.append(
            "| {feature_set} | {feature_count} | {global_utility} | {phase_thresholds} | "
            "{phased_utility} | {auprc} | {lift}x |".format(
                feature_set=row["feature_set"],
                feature_count=row["feature_count"],
                global_utility=row["validation_global"]["utility"],
                phase_thresholds=", ".join(f"{value:.2f}" for value in row["phase_thresholds"]),
                phased_utility=metrics["utility"],
                auprc=metrics["average_precision"],
                lift=metrics["lift_at_k"]["top_10pct"]["lift"],
            )
        )

    site_rows = []
    for row in payload["site_holdout"]:
        metrics = row["phased_metrics"]
        site_rows.append(
            f"| {row['feature_set']} | {row['train_site']} | {row['validation_site']} | "
            f"{metrics['average_precision']} | {metrics['lift_at_k']['top_10pct']['lift']}x | "
            f"{metrics['utility']} |"
        )

    final_metrics = payload["final_metrics"]
    final_rows = "\n".join(
        metric_row(split.title(), final_metrics[split]["phased"])
        for split in ("train", "validation", "test")
    )
    test_policy_rows = "\n".join(
        [
            metric_row("Global threshold", final_metrics["test"]["global"]),
            metric_row("Time-phased thresholds", final_metrics["test"]["phased"]),
        ]
    )

    text = f"""# Literature-Derived Sepsis Feature Experiment

## Selection Protocol

- Candidate models were trained on the original patient-level training split.
- Feature groups and all thresholds were selected using validation data only.
- Hospital A/B holdout used only development patients; locked test patients were excluded.
- The internal test split was loaded once after feature and policy selection.
- Primary selection metric: normalized PhysioNet Challenge Utility.

## Validation Ablation

| Feature set | Features | Global Utility | Phase thresholds | Phased Utility | AUPRC | Lift@10% |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
{chr(10).join(ablation_rows)}

Initial random-validation winner: `{payload['initial_selected_feature_set']}`<br>
Final development-selected feature set: `{payload['selected_feature_set']}`

## Hospital Holdout

| Feature set | Train site | Validation site | AUPRC | Lift@10% | Utility |
| --- | --- | --- | ---: | ---: | ---: |
{chr(10).join(site_rows)}

## Final Train / Validation / Test

Time phases use ICU-hour boundaries `{payload['phase_bounds']}` with thresholds `{payload['selected_phase_thresholds']}`.

| Split | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{final_rows}

## Test Policy Comparison

Both policies were fixed on validation before the test split was loaded.

| Policy | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{test_policy_rows}

## Interpretation

The literature-derived features should be retained only when they improve validation Utility without a material hospital-holdout regression. AUROC and AUPRC measure ranking; the time-phased policy changes precision, recall, alarm rate, and Utility but does not change ranking metrics.

This test split is internally locked for this experiment, but it is not equivalent to the original hidden PhysioNet hospital-C test set. Earlier project iterations also consulted this same internal test split, so external generalization remains unproven.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--max-patients", type=int, default=0, help="Use 0 for all patients")
    parser.add_argument("--negative-sample-rate", type=float, default=0.08)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-batch-size", type=int, default=500)
    parser.add_argument("--phase-bounds", type=int, nargs=3, default=list(DEFAULT_PHASE_BOUNDS))
    parser.add_argument(
        "--candidate",
        action="append",
        choices=list(CANDIDATE_CONFIGS),
        help="Candidate feature set; repeat to limit the ablation",
    )
    parser.add_argument("--skip-site-holdout", action="store_true")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=OUTPUT_DIR / "literature_feature_experiment_metrics.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=OUTPUT_DIR / "literature_feature_experiment_report.md",
    )
    parser.add_argument(
        "--selected-model-path",
        type=Path,
        default=MODEL_DIR / "sepsis_xgboost_literature_selected.joblib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_patients = None if args.max_patients <= 0 else args.max_patients
    candidate_names = args.candidate or list(CANDIDATE_CONFIGS)

    files = patient_files(args.data_root)
    train_files, validation_files, test_files = stratified_patient_split(
        files,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
        max_patients=max_patients,
    )
    print(
        f"Patient split: train={len(train_files):,}, validation={len(validation_files):,}, "
        f"locked_test={len(test_files):,}",
        flush=True,
    )

    ablation_results: list[dict[str, Any]] = []
    model_paths: dict[str, Path] = {}
    for name in candidate_names:
        result, model_path = train_candidate(
            name, CANDIDATE_CONFIGS[name], train_files, validation_files, args
        )
        ablation_results.append(result)
        model_paths[name] = model_path

    initial_selected = max(
        ablation_results,
        key=lambda row: (row["validation_phased"]["utility"], row["validation_phased"]["average_precision"]),
    )["feature_set"]
    selected = initial_selected
    print(f"Initial validation-selected feature set: {initial_selected}", flush=True)

    site_results: list[dict[str, Any]] = []
    if not args.skip_site_holdout:
        development_files = train_files + validation_files
        site_feature_sets = list(dict.fromkeys(["baseline", initial_selected]))
        for feature_set in site_feature_sets:
            site_results.extend(
                run_site_holdout(feature_set, CANDIDATE_CONFIGS[feature_set], development_files, args)
            )

        if initial_selected != "baseline":
            mean_site_utility: dict[str, float] = {}
            for feature_set in ("baseline", initial_selected):
                values = [
                    row["phased_metrics"]["utility"]
                    for row in site_results
                    if row["feature_set"] == feature_set
                ]
                mean_site_utility[feature_set] = float(np.mean(values))
            if mean_site_utility[initial_selected] + 0.005 < mean_site_utility["baseline"]:
                selected = "baseline"
                print(
                    f"Falling back to baseline after site holdout: {mean_site_utility}", flush=True
                )

    selected_result = next(row for row in ablation_results if row["feature_set"] == selected)
    selected_artifact = joblib.load(model_paths[selected])
    selected_config = CANDIDATE_CONFIGS[selected]
    feature_names = selected_artifact["feature_names"]
    global_threshold = float(selected_result["global_threshold"])
    phase_thresholds = list(selected_result["phase_thresholds"])
    phase_bounds = tuple(args.phase_bounds)

    print("Selection complete. Loading full train and locked test cohorts once...", flush=True)
    train_scored = score_cohort(
        train_files,
        selected_artifact["model"],
        selected_config,
        feature_names,
        batch_size=args.score_batch_size,
    )
    validation_scored = score_cohort(
        validation_files,
        selected_artifact["model"],
        selected_config,
        feature_names,
        batch_size=args.score_batch_size,
    )
    test_scored = score_cohort(
        test_files,
        selected_artifact["model"],
        selected_config,
        feature_names,
        batch_size=args.score_batch_size,
    )

    final_metrics: dict[str, dict[str, Any]] = {}
    final_stats: dict[str, dict[str, Any]] = {}
    for split_name, scored in (
        ("train", train_scored),
        ("validation", validation_scored),
        ("test", test_scored),
    ):
        global_predictions = apply_global_policy(scored, global_threshold)
        phased_predictions = apply_phased_policy(scored, phase_thresholds, phase_bounds)
        final_metrics[split_name] = {
            "global": evaluate_predictions(
                scored, global_predictions, f"global_threshold={global_threshold:.3f}"
            ),
            "phased": evaluate_predictions(
                scored,
                phased_predictions,
                f"time_phased bounds={list(phase_bounds)} thresholds={phase_thresholds}",
            ),
        }
        final_stats[split_name] = scored["stats"]

    selected_artifact.update(
        {
            "selected_by": "validation_utility_with_development_site_holdout",
            "global_threshold": global_threshold,
            "threshold": global_threshold,
            "phase_bounds": list(phase_bounds),
            "phase_thresholds": phase_thresholds,
            "primary_policy": "time_phased_thresholds",
            "source_model_path": str(model_paths[selected]),
        }
    )
    joblib.dump(selected_artifact, args.selected_model_path)

    payload = {
        "protocol": {
            "seed": args.seed,
            "negative_sample_rate": args.negative_sample_rate,
            "validation_size": args.validation_size,
            "test_size": args.test_size,
            "max_patients": max_patients,
            "test_loaded_after_selection": True,
        },
        "phase_bounds": list(phase_bounds),
        "validation_ablation": ablation_results,
        "initial_selected_feature_set": initial_selected,
        "selected_feature_set": selected,
        "selected_global_threshold": global_threshold,
        "selected_phase_thresholds": phase_thresholds,
        "site_holdout": site_results,
        "final_stats": final_stats,
        "final_metrics": final_metrics,
        "selected_model_path": str(args.selected_model_path),
    }
    args.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, args.report_path)
    print(f"Saved selected model -> {args.selected_model_path}", flush=True)
    print(f"Saved metrics -> {args.metrics_path}", flush=True)
    print(f"Saved report -> {args.report_path}", flush=True)
    print(json.dumps({"selected": selected, "test": final_metrics["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
