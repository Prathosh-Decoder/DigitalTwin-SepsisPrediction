# Five-Fold Nested Cross-Validation

## Protocol

- Outer folds: `5` patient-stratified folds for unbiased evaluation.
- Inner folds: `3` patient-stratified folds inside each outer development cohort.
- Inner selection: baseline versus literature-core features, then global versus ICU-phase Utility thresholds.
- Negative-row sampling (`0.08`) is applied only to model-fitting rows.
- Every held-out inner and outer cohort retains all hourly rows.
- No patient appears in training and evaluation within the same fold.

## Outer-Fold Results

| Fold | Selected features | Policy | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | literature_core | time_phased | 0.8509 | 0.1233 | 0.0767 | 0.6729 | 0.1377 | 5.5333x | 0.4432 |
| 2 | literature_core | time_phased | 0.8487 | 0.1285 | 0.0697 | 0.6746 | 0.1263 | 5.3365x | 0.4207 |
| 3 | literature_core | time_phased | 0.8520 | 0.1411 | 0.0724 | 0.6753 | 0.1308 | 5.5629x | 0.4305 |
| 4 | baseline | time_phased | 0.8507 | 0.1239 | 0.0721 | 0.6717 | 0.1302 | 5.3014x | 0.4308 |
| 5 | baseline | time_phased | 0.8429 | 0.1168 | 0.0667 | 0.6715 | 0.1213 | 5.2640x | 0.4058 |

## Inner-Selected Thresholds

The four phase thresholds correspond to ICU hours `1-12`, `13-24`, `25-48`, and `49+`.

| Outer fold | Selected features | Global threshold | Phase thresholds |
| ---: | --- | ---: | --- |
| 1 | literature_core | 0.21 | 0.23, 0.22, 0.23, 0.19 |
| 2 | literature_core | 0.21 | 0.24, 0.19, 0.23, 0.18 |
| 3 | literature_core | 0.23 | 0.23, 0.21, 0.23, 0.17 |
| 4 | baseline | 0.23 | 0.23, 0.23, 0.22, 0.19 |
| 5 | baseline | 0.22 | 0.21, 0.24, 0.22, 0.19 |

## Mean and Uncertainty

The confidence intervals use the fold mean with a t interval (`df=4`). They describe fold-to-fold variation, not external-hospital uncertainty.

| Metric | Mean | SD | 95% CI |
| --- | ---: | ---: | ---: |
| AUROC | 0.8490 | 0.0036 | [0.8445, 0.8536] |
| AUPRC | 0.1267 | 0.0091 | [0.1155, 0.1380] |
| Precision | 0.0715 | 0.0037 | [0.0669, 0.0761] |
| Recall | 0.6732 | 0.0017 | [0.6711, 0.6753] |
| F1 | 0.1293 | 0.0061 | [0.1217, 0.1368] |
| Lift@10% | 5.3996 | 0.1383 | [5.2279, 5.5714] |
| Utility | 0.4262 | 0.0139 | [0.4089, 0.4435] |

## Pooled Out-of-Fold Predictions

Each patient contributes predictions from exactly one outer model.

| AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8491 | 0.1262 | 0.0714 | 0.6732 | 0.1290 | 5.3976x | 0.4262 |

Feature selection counts: `{'baseline': 2, 'literature_core': 3}`. Literature-core won three folds and baseline won two, so the engineered features are helpful but not uniformly stable.

## Interpretation

This nested estimate is more defensible than a single internal train/validation/test split because feature and threshold decisions are repeated entirely inside each outer training cohort. It still evaluates hospitals A and B only and therefore does not replace external validation on an independent hospital system.
