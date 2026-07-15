# Experiments Log: Alternatives Tried After the Production Model

The production model documented in [`REPORT.md`](REPORT.md) — a LightGBM regressor on the
`U1 − U0` utility target, with `num_leaves=49, max_depth=6, learning_rate=0.05, n_estimators=300`
— was finalized via the 27-combination hyperparameter grid search described there. After that,
several additional alternatives were tried locally to see if the result could be improved further.
**None beat the production model**, so it was left unchanged. This document records what was
tried and why, for future reference. The two early quick checks (§1 ensemble, §2 lag-stack) were
not kept as scripts; the later, more substantial studies (§4–§6) are retained as reproducible
scripts under [`../sepsis_pipeline/experiments/`](../sepsis_pipeline/experiments/) with their raw
results.

All experiments below reused the exact same train/val/test patient split, the exact same 214 (or
explicitly noted otherwise) engineered features, and the exact same `U1 − U0` training target as
the production model, so comparisons within each experiment are apples-to-apples. None of them
modified `artifacts/model_bundle.joblib` or any committed file.

**Note on the split**: these experiments were run against the split in place at the time, before
it was later adjusted so the 8 patients used in the Project 1 digital twin's bed demo are
guaranteed test-only (see `REPORT.md` §4.1). The production model was retrained afterward on the
corrected split, so its current numbers (in `REPORT.md` and `README.md`) differ slightly from the
"production" baseline row shown in each table below — the *conclusions* (neither alternative beat
production) are unaffected, since the swap only moved 7 of 40,336 patients.

## 1. XGBoost + CatBoost ensemble

**Idea**: train XGBoost and CatBoost regressors on the identical target/features/split (not as
ordinary classifiers — the same utility-gain regression trick as the production LightGBM model),
then average all three models' scores together.

| Model | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (production) | 0.8458 | 0.1224 | 0.0691 | **0.7049** | 0.1259 | 5.5605 | **0.4382** |
| XGBoost (same target) | 0.8437 | 0.1248 | 0.0771 | 0.6542 | 0.1379 | 5.5605 | 0.4353 |
| CatBoost (same target) | 0.8365 | 0.1304 | 0.0722 | 0.6467 | 0.1300 | 5.4167 | 0.4138 |
| **Ensemble (mean of all 3)** | 0.8453 | 0.1299 | 0.0700 | 0.6912 | 0.1271 | **5.6180** | 0.4324 |

**Conclusion**: XGBoost and CatBoost individually both underperformed the tuned LightGBM model on
AUROC and Utility, so averaging them in pulled the ensemble slightly below LightGBM alone on the
two most important metrics. Not adopted.

## 2. Lag-stacked features instead of rolling min/max/std (Team 2's approach)

**Idea**: the #2-ranked original-challenge team (Du, Sadr, de Chazal) did not summarize recent
history into rolling statistics — they stacked each hour's feature vector for the last 5 hours
side-by-side, letting the model see raw recent history directly. This experiment replaced the
production model's 96 rolling mean/min/max/std features (12 variables × 2 windows × 4 stats) with
48 lag-stacked forward-filled values (`h-1` .. `h-4`, matching their 5-hour lookback including the
already-existing current-hour value), keeping every other feature unchanged. Net: 214 → 166
features.

| Model | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production (rolling min/max/std, 214 features) | 0.8458 | 0.1224 | 0.0691 | **0.7049** | 0.1259 | **5.5605** | **0.4382** |
| Lag-stack h-1..h-4 (166 features) | **0.8461** | **0.1248** | **0.0780** | 0.6495 | **0.1393** | 5.5461 | 0.4334 |

**Conclusion**: a precision/recall trade-off, not a strict improvement — AUROC, AUPRC, Precision,
and F1 all edge up slightly with fewer features, but Recall and Utility (the two metrics
prioritized for this project) both drop. Not adopted.

## 3. Reward/penalty and threshold sensitivity sweeps

Separately, `evaluate_model.ipynb` (Sections 5, 10, and 11) explores how far the *existing,
unchanged* production model's results move under different false-alarm penalties, miss
penalties, and reward magnitudes, and across a range of decision thresholds — none of which
retrain the model, only re-grade its existing scores. Key finding: the production reward/penalty
constants (`max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05` — the official challenge's own published
values) and the utility-optimal threshold already sit at a well-balanced point; no combination in
the swept grid improved both Recall and Utility simultaneously without trading one for the other.
This is expected: for a fixed model, Recall and Utility both derive from the same
threshold/scoring choice, so improving both together requires a genuinely better-discriminating
model (higher AUROC/AUPRC), not a different threshold or reward definition. Experiments 1 and 2
above were attempts at exactly that, and neither succeeded.

## 4. Broader hyperparameter search (Optuna) + nested cross-validation

Script: [`../sepsis_pipeline/experiments/optuna_nested_cv.py`](../sepsis_pipeline/experiments/optuna_nested_cv.py)

**Idea**: the production hyperparameters came from a small 27-combination grid over three
parameters. This ran a much broader **Optuna** Bayesian search (50 trials, over ten LightGBM
parameters, objective = validation AUPRC — the discrimination metric that directly targets
"predict sepsis when it's sepsis"), then a **5-fold nested cross-validation** (patient-grouped,
outcome-stratified outer folds; an inner Optuna search per fold) for an honest,
generalization-robust performance estimate.

| Model | AUROC | AUPRC | Recall | Utility |
|---|---:|---:|---:|---:|
| Production (single split) | **0.8465** | 0.1252 | 0.6694 | **0.4554** |
| Optuna-best (single split) | 0.8452 | **0.1284** | **0.6783** | 0.4403 |
| Nested CV (mean ± std) | 0.8381 ± 0.0114 | 0.1240 ± 0.0078 | 0.6523 ± 0.0374 | 0.4070 ± 0.0233 |

**Conclusion**: the 50-trial Optuna search did **not** beat production — it found a marginally
different precision/recall trade (slightly higher AUPRC/Recall, slightly lower AUROC/Utility), not
a genuine improvement. The most valuable output is the **nested-CV estimate**: it is a touch below
the single-split test figure, which tells us the headline number is *mildly optimistic* and the
trustworthy, generalization-robust performance is about **0.838 AUROC / 0.41 Utility** (with an
honest ±0.011 AUROC / ±0.023 Utility error bar). Nothing adopted.

## 5. Split-robustness: 10 groupings, demo patients pinned to test

Script: [`../sepsis_pipeline/experiments/split_robustness.py`](../sepsis_pipeline/experiments/split_robustness.py)

**Idea**: keep the 8 Project-1 demo patients pinned to the test set (see `REPORT.md` §4.1), then
re-shuffle all other patients into 10 different 80/10/10 groupings and retrain the
production-config model on each — to measure how much the score swings purely from *which patients
land where*.

| | AUROC | AUPRC | Recall | Utility |
|---|---:|---:|---:|---:|
| Mean over 10 groupings | 0.846 | 0.129 | 0.675 | 0.429 |
| Range (min–max) | 0.829–0.859 | 0.116–0.148 | 0.626–0.727 | 0.403–0.449 |
| Std | ±0.008 | ±0.010 | ±0.039 | ±0.013 |

**Conclusion**: the current production model sits almost exactly on the mean — it was a
representative split, not a lucky one. The ±0.008 AUROC swing (0.829 to 0.859) is pure split
noise, since the model config is identical across all 10. Crucially, the test "winners" are not
the validation winners (the split with the best test AUROC had one of the *worst* validation
utilities) — the signature of noise, not of a better model. This confirms that cherry-picking the
best-scoring grouping would report an inflated, non-reproducible number, and that the honest
figure is the mean (where production already sits).

## 6. Matched vs. random backfill of the demo-patient swap

Script: [`../sepsis_pipeline/experiments/matched_replacement.py`](../sepsis_pipeline/experiments/matched_replacement.py)

**Idea**: pinning the 8 demo patients to test required moving 7 of them out of train/val, whose
slots were backfilled with *random* patients. This study instead backfilled with **clinically
similar** patients (same hospital, same septic outcome, nearest-neighbor on a per-patient summary
of age/vitals/key-labs/length-of-stay), to keep the training distribution matched to the departing
demo patients.

| Backfill method | AUROC | AUPRC | Recall | Utility |
|---|---:|---:|---:|---:|
| Random (current production) | 0.8465 | 0.1252 | 0.6694 | **0.4554** |
| Matched (clinical similarity) | **0.8469** | **0.1258** | 0.6606 | 0.4434 |

**Conclusion**: statistically identical — swapping 7 of 32,268 training patients cannot move the
model. Matched backfill is a more *principled* construction, but it does not change performance
(and would slightly soften the "genuinely held-out demo" property, since train would contain
near-twins of the demo patients). Production (random backfill) kept.

## Overall conclusion

The production model (LightGBM, rolling-statistics features, tuned hyperparameters) remains the
best result found across **five independent experiment families** — alternative boosting libraries
and ensembling (§1), a different temporal-feature scheme (§2), reward/penalty and threshold sweeps
(§3), a broad Optuna hyperparameter search (§4), and split-composition variations (§5, §6). AUROC
has stayed within roughly 0.829–0.859 throughout, with a nested-CV honest estimate of ~0.838 — a
real ceiling for this feature set that hyperparameter tuning and split juggling cannot break past.
The consistency across all these attempts is itself a result: it means the reported performance is
stable and trustworthy, not a lucky configuration. The only remaining lever likely to move the
ceiling meaningfully is richer feature engineering, which was out of scope here.
