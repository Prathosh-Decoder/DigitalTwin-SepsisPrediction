# Experiments Log: Alternatives Tried After the Production Model

The production model documented in [`REPORT.md`](REPORT.md) — a LightGBM regressor on the
`U1 − U0` utility target, with `num_leaves=49, max_depth=6, learning_rate=0.05, n_estimators=300`
— was finalized via the 27-combination hyperparameter grid search described there. After that, a
few additional alternatives were tried locally to see if the result could be improved further.
**None beat the production model**, so it was left unchanged. This document records what was
tried and why, for future reference. The experiment scripts themselves were removed after
producing these results to keep the repository focused on the production pipeline; the numbers
below are the full record.

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

## Overall conclusion

The production model (LightGBM, rolling-statistics features, tuned hyperparameters) remains the
best result found. AUROC has stayed in a narrow 0.836–0.846 band across every alternative tried —
a real ceiling this project has been unable to move past — so all these experiments ultimately
traded metrics against each other around that same ceiling rather than raising it.
