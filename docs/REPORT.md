# Early Sepsis Prediction: Technical Report

**A LightGBM early-warning model trained on the PhysioNet/Computing in Cardiology 2019 Sepsis Challenge dataset**

---

## Abstract

This report documents a machine learning model that predicts, for each hour of an ICU stay,
whether a patient is likely heading toward sepsis approximately six hours before clinical onset.
The model is trained on the PhysioNet/Computing in Cardiology (CinC) 2019 Sepsis Challenge dataset
(40,336 patients across two hospital systems), using an approach adapted from the challenge's
top-ranked submission: a LightGBM regressor trained to directly optimize the competition's own
time-dependent utility function, using 214 causally-constructed features per hourly observation.
On a held-out test set, the model achieves an AUROC of 0.846, an AUPRC of 0.125, and a normalized
utility score of 0.455 — exceeding the original competition winner's officially reported score of
0.360, though under an easier evaluation condition (see §8, Limitations). A prioritization layer on
top of the model (§6) converts its raw score into a 0–100 criticality rank, a calibrated
probability, a tier, a trend, and plain-language drivers for bedside triage. A record of every
alternative tried to improve the model — none of which beat it — is in §7.

This is a single combined document: model report (§1–§5), triage layer (§6), experiment log (§7),
limitations (§8), references (§9). For wiring the model into the Node.js digital twin, see the
separate [`INTEGRATION.md`](INTEGRATION.md).

---

## 1. Problem Description

The task is to predict, using only information available up to and including a given hour of an
ICU stay, whether a patient is heading toward sepsis roughly six hours before it would be
clinically recognized. A prediction is produced fresh every hour from that hour's and all prior
hours' vital signs and laboratory results; no information from later in the stay is ever used,
matching the constraint a real bedside monitoring system would face.

The data is drawn from the PhysioNet/Computing in Cardiology Challenge 2019, a public research
dataset spanning two hospital systems (referred to here as Hospital A and Hospital B). Each
patient's stay is recorded as one row per ICU hour, with 8 vital-sign columns, 26 laboratory
columns, demographic and admission-timing fields, and a binary `SepsisLabel` marking the
pre-onset warning window for patients who go on to develop sepsis.

| | Hospital A | Hospital B | Combined |
|---|---:|---:|---:|
| Patients | 20,336 | 20,000 | 40,336 |
| Ever-septic patients | 1,790 (8.8%) | 1,142 (5.7%) | 2,932 (7.3%) |
| Hourly rows | 790,215 | 761,995 | 1,552,210 |
| Positive (`SepsisLabel=1`) rows | — | — | ~1.8% |

Laboratory values are recorded far less frequently than vitals, since blood draws happen
periodically rather than hourly; this produces heavy, clinically structured (non-random)
missingness that shapes much of the feature engineering in §3.

---

## 2. Related Work

### 2.1 Label and scoring design

The `SepsisLabel` column reflects the Sepsis-3 clinical definition, as operationalized by the
challenge organizers. For each septic patient, an onset time `t_sepsis` is derived from the
earlier of a suspected-infection timestamp (antibiotics ordered near a blood culture) and a
timestamp at which a patient's SOFA organ-dysfunction score rises by 2 or more points within 24
hours. `SepsisLabel` is set to 1 starting **six hours before** `t_sepsis`, not at `t_sepsis`
itself, which is what makes this fundamentally an early-warning task.

Evaluation uses a custom, time-dependent **utility function** rather than plain accuracy or AUROC:
a correct positive prediction is rewarded most when it lands close to six hours before onset, an
overly early flag draws only a small penalty, and a missed case is penalized increasingly severely
the closer it goes unflagged toward actual onset. The full formula is given in §3.3.

### 2.2 The winning approach

The top-ranked entry in the official competition, *"Can I get your signature?"* (Morrill,
Kormilitzin, Nevado-Holgado, Swaminathan, Howison, and Lyons, University of Oxford), scored 0.360
on the official hidden test set [1]. Rather than training a classifier on the raw label, the team
trained a LightGBM regressor to predict the *difference* in utility earned by flagging a given
hour versus not flagging it (`U1 − U0`), directly optimizing the competition's own scoring rule.
A gradient-free optimizer (`nevergrad`) then selected the decision threshold maximizing real
utility on validation data. Their feature set combined hand-crafted clinical scores (Shock Index,
BUN/Creatinine ratio, a Partial SOFA score, and a SOFA-deterioration flag) with **path-signature
features** — a technique from stochastic analysis summarizing an irregularly-sampled time series
as a compact, order-invariant vector of iterated integrals over a lead-lag transformed
representation of the series. Their own ablation study attributed +0.012 utility to the signature
features over hand-crafted features alone (0.418 → 0.430 in 5-fold cross-validation).

This project mimics the utility-regression training target and the hand-crafted clinical features,
but deliberately excludes path-signature features (rationale in §3.4).

### 2.3 Other approaches and field-wide patterns

| Team | Approach | Result |
|---|---|---:|
| Du, Sadr, de Chazal ("Sepsyd") [2] | XGBoost, 30 trees, 40:1 positive-class weighting, missingness masks + rolling variance features | Utility 0.345 |
| Zabihi, Kiranyaz, Gabbouj ("Separatrix") [3] | Ensemble of 5 XGBoost models on random-undersampled splits, 407 engineered features | Utility 0.339 |
| Chang, Rubin, Boverman et al. ("prna", Philips) | RITS recurrent imputation network feeding a temporal convolutional network, trained against a differentiable utility approximation | Utility ≈ 0.41 (internal) |

Across the wider field, gradient-boosted tree ensembles were the most common top-performing model
family for this specific metric, ahead of recurrent/deep architectures. Common preprocessing
patterns — forward-fill imputation, missingness-indicator features, and rolling-window statistics
over 6–24 hour windows — informed the feature design in §3.1.

---

## 3. Methodology

### 3.1 Feature engineering

All 214 features are computed per patient, using only that patient's own rows up to and including
the current hour — verified by a dedicated causality test that recomputes each feature on a
truncated patient history and checks equivalence with the full-history computation. Features fall
into seven groups:

| Category | Description | Count |
|---|---|---:|
| Static demographics | Age, gender, ICU admission time, ICU unit type | 6 |
| Measured-flag + last-known value | Per raw vital/lab: was it recorded this hour, and its forward-filled last value | 68 |
| Hours since last measured | Per lab column: elapsed time since last drawn (capped at 336h) | 26 |
| Overall missingness count | Count of missing core vitals this hour | 1 |
| Rolling window statistics | Mean/min/max/std over 6h and 24h, on raw (pre-fill) values, 12 key variables | 96 |
| Hour-to-hour deltas | Change from the previous hour, same 12 variables | 12 |
| Clinical scores | Shock Index, BUN/Creatinine ratio, Partial SOFA, its 24h delta, and a worsening flag | 5 |

Rolling statistics and deltas are deliberately computed on raw, as-recorded values rather than the
forward-filled series — a forward-filled series exhibits artificially low variance whenever a
single true reading is carried forward across several hours, masking real instability. Missing
values are left as missing rather than zero-filled, since LightGBM handles missingness natively
via learned default split directions. Hospital identity is tracked as metadata but deliberately
excluded from the model's input features, to prevent the model from learning a hospital-specific
shortcut rather than transferable clinical signal — the mechanism behind every top team's score
collapsing on the official challenge's unseen third hospital.

### 3.2 The utility-regression training target

`U1` and `U0` are not predictions but ground-truth quantities computed after the fact from each
patient's already-known outcome: `U1` is the utility the official rule would award for flagging a
given hour, and `U0` the utility for not flagging it. Both are always computable for historical
data, regardless of which split (train/validation/test) a patient's hours fall into. The model
itself never observes `U1` or `U0` — only the 214 input features — and is trained to predict
`U1 − U0` directly, embedding the competition's scoring priorities into the learning objective
rather than relying on a generic classification loss to happen to align with them.

**Worked example** — a hypothetical patient with true sepsis onset (`t_sepsis`) at hour 50:

| Hour | Hours before onset | U1 (if flagged) | U0 (if not flagged) | Target (U1 − U0) |
|---:|---:|---:|---:|---:|
| 42 | 8h | 0.667 | 0.000 | 0.667 |
| 43 | 7h | 0.833 | 0.000 | 0.833 |
| 44 | 6h (optimal) | 1.000 | 0.000 | 1.000 |
| 45 | 5h | 0.889 | −0.222 | 1.111 |
| 46 | 4h | 0.778 | −0.444 | 1.222 |
| 47 | 3h | 0.667 | −0.667 | 1.333 |
| 48 | 2h | 0.556 | −0.889 | 1.444 |
| 49 | 1h | 0.444 | −1.111 | 1.556 |

The reward for flagging (`U1`) peaks at the 6-hour mark and decays thereafter, while the penalty
for silence (`U0`) worsens faster over the same interval, so the net training target keeps
climbing. This does not create an incentive to delay flagging: rewards accumulate independently
at every hour, so a model that begins flagging at hour 44 and continues collects the reward from
every subsequent hour, whereas a model that waits until hour 47 has already accrued the (negative)
`U0` penalty for hours 45–46. For a patient who never develops sepsis, `U1 = -0.05` and `U0 = 0` at
every hour, with no timing component at all.

### 3.3 Reward and penalty formula

Let `dt = (current hour) − t_sepsis`. Six constants, taken from the official challenge
specification:

| Constant | Value | Meaning |
|---|---:|---|
| `dt_early` | −12h | Reward window opens |
| `dt_optimal` | −6h | Point of maximum reward |
| `dt_late` | +3h | Reward/penalty window closes |
| `max_u_tp` | +1.0 | Maximum reward for a correctly timed flag |
| `min_u_fn` | −2.0 | Maximum penalty for a missed case |
| `u_fp` | −0.05 | Flat penalty for any false alarm |

Between `dt_early` and `dt_optimal`, a correct flag's reward ramps linearly from 0 to +1 while
silence earns 0. Between `dt_optimal` and `dt_late`, a flag's reward decays linearly from +1 to 0
while silence's penalty ramps from 0 to −2. Outside this window there is no further timing-based
reward or penalty; a false alarm still costs the flat −0.05, and correctly staying silent earns 0.
Per-row scores sum across all hourly predictions and are normalized cohort-wide so that always
predicting negative scores 0 and a perfectly-timed oracle scores 1.

### 3.4 Deviations from the winning approach

| Mimicked | Independently designed |
|---|---|
| Utility-regression training target (`U1 − U0`) | Rolling window statistics |
| Hand-crafted clinical features (Shock Index, BUN/Cr, Partial SOFA, SOFA-deterioration) | Missingness/hours-since-measured features |
| LightGBM as the model family; published hyperparameters as a starting point | Partial SOFA cutoff implementation, delta features |

Path-signature features — the winning entry's headline technique — were deliberately excluded.
The original ablation attributed only a modest +0.012 utility gain to them; the technique requires
a heavy additional dependency (`iisignature`/`esig`); and signature terms are not clinically
interpretable, which would undermine the SHAP-based explainability goals of this project (§5.4).

---

## 4. Model Architecture

### 4.1 Data split

Patients — not individual rows — are split 80% train / 10% validation / 10% test, so no patient's
hours are divided across splits. The split is stratified by whether a patient is ever septic and
performed independently within each hospital before combining, ensuring proportional
representation of both hospitals and outcome classes in every split. Training data fits the model;
validation data drives early stopping, the hyperparameter search, and threshold selection; the
test set is touched exactly once, to produce the results in §5.

The 8 patients used in the Project 1 digital twin's bed demo (p004880, p001072, p017091, p007057,
p014527, p000295, p010756, p011623) are deliberately constrained to the test split. An initial
80/10/10 assignment placed 6 of them in training and 1 in validation; those 7 were swapped with 7
randomly chosen patients from the test split (preserving split sizes and hospital/class balance)
so that any future live demonstration on these specific patients reflects genuine held-out
performance, not memorized training data. The model was retrained from scratch on the corrected
split, since relabeling a split file after training does not undo what a model has already learned
from those rows.

### 4.2 Model and hyperparameters

LightGBM was selected to mirror the winning team's model family, and for its native handling of
missing values and fast, exact SHAP support. It is trained as a plain squared-error regressor
against the `U1 − U0` target, not as a classifier against `SepsisLabel`.

| Hyperparameter | Value |
|---|---:|
| `num_leaves` | 49 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `n_estimators` | 300 |
| `min_child_samples` | 122 |
| `reg_alpha` | 100 |
| `reg_lambda` | 0 |
| `subsample` | 0.3465 |
| `colsample_bytree` | 0.5494 |
| `early_stopping_rounds` | 20 |

Most values come directly from the winning team's published configuration. `n_estimators` and
`learning_rate` were refined via a 27-combination grid search (`n_estimators ∈ {100,200,300}`,
`num_leaves ∈ {31,49,70}`, `learning_rate ∈ {0.05,0.10,0.15}`) evaluated on the validation split.
The published baseline (100 / 49 / 0.10) ranked 14th of 27 by validation utility; the adopted
configuration (300 / 49 / 0.05) improved both validation utility (0.4285 → 0.4354) and validation
AUROC (0.8420 → 0.8443) over that baseline, while changing only two of the three swept parameters.
Class imbalance (~1.8% positive rows) is handled implicitly through the asymmetric training
target rather than an explicit class-weighting parameter, matching the winning team's approach.

This configuration is the final, adopted model. Several further alternatives were tried afterward
and did not improve on it; see §7 for the full record.

---

## 5. Results

### 5.1 Metric definitions

- **AUROC** — probability the model ranks a random septic hour above a random non-septic hour;
  0.5 = random, 1.0 = perfect; threshold-independent.
- **AUPRC** — the same idea as AUROC but on the precision/recall curve; more informative here
  since only ~1.8% of rows are truly positive, and best read against the no-skill baseline of
  ~0.018 (the positive rate) rather than against 1.0.
- **Precision** — of flagged hours, the fraction genuinely sepsis-warning.
- **Recall** — of genuine sepsis-warning hours, the fraction correctly flagged.
- **F1** — harmonic mean of precision and recall.
- **Lift@10%** — how much more concentrated true cases are in the riskiest 10% of predictions
  versus a random 10%; a practical measure of ranking usefulness independent of any threshold.
- **Utility** — the official challenge's own time-aware scoring rule, and the metric this model
  is directly trained to optimize. It grades each hour on a timing curve rather than flat
  right/wrong (reward peaks at exactly 6h before onset, decays afterward; a miss gets costlier
  the closer it goes unflagged toward onset), then normalizes cohort-wide so that never flagging
  anything scores 0 and a perfectly-timed oracle scores 1. Full formula and a worked example are
  in §3.2–3.3.

### 5.2 Performance

| Split | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.8895 | 0.2095 | 0.0871 | 0.7374 | 0.1558 | 6.304 | 0.5207 |
| Validation | 0.8444 | 0.1388 | 0.0765 | 0.6566 | 0.1371 | 5.418 | 0.4325 |
| **Test** | **0.8465** | **0.1252** | **0.0800** | **0.6694** | **0.1429** | **5.674** | **0.4554** |
| Test — Hospital A | 0.8321 | 0.1230 | 0.0822 | 0.7036 | 0.1472 | 5.659 | 0.4840 |
| Test — Hospital B | 0.8615 | 0.1389 | 0.0762 | 0.6137 | 0.1356 | 5.762 | 0.4073 |

At the test-set operating threshold: 1,883 true positives, 930 false negatives, 21,653 false
positives, 131,853 true negatives. Validation and test scores sit close to each other and
noticeably below training scores, indicating the model is not substantially overfit.

### 5.3 What these values mean in practice

- **AUROC 0.847** — ranks a random septic hour above a random non-septic hour ~85% of the time.
- **AUPRC 0.125** — about 7x the ~0.018 no-skill baseline; low in absolute terms, but that's
  expected given the class imbalance, not a sign of a weak model.
- **Recall 0.669** — catches about 2 in 3 real sepsis-warning hours; the other third (930 hours)
  go unflagged.
- **Precision 0.080** — of everything flagged, ~8% is a genuine warning hour; the rest are false
  alarms, a direct consequence of how rare true positives are (~1.8% of rows).
- **Lift@10% 5.67x** — the riskiest 10% of flagged hours contain ~5.7x more real cases than a
  random 10% would, useful for prioritization even at low raw precision.
- **Utility 0.455** — captures 45.5% of the maximum achievable timing-aware score (0 = never
  flag, 1 = perfect foresight). For reference, the original challenge's winning team scored 0.360
  on the official hidden test set — not fully apples-to-apples, since their test set included a
  genuinely unseen third hospital and this project's does not (§8).

### 5.4 Explainability

SHAP (TreeExplainer) values were computed on a stratified sample of 10,000 test-set rows (2,000
positive, 8,000 negative). Plots are available at `sepsis_pipeline/artifacts/plots/`:
`roc_curve.png`, `pr_curve.png`, `threshold_sweep.png`, `shap_summary_beeswarm.png`, and
`shap_importance_bar.png`. These per-feature attributions also power the plain-language "why"
in the criticality layer (§6).

---

## 6. Criticality & Prioritization Layer

A layer **on top of** the trained model that turns its raw score into a clinician-readable triage
view — answering *"who do I look at first, and why."* It does **not** retrain or modify the model;
it is a separate artifact (`artifacts/criticality_calibrator.joblib`) fitted by
`sepsis_pipeline/07_criticality.py`, leaving the model bundle byte-for-byte unchanged.

### 6.1 The four views

For any patient-hour, the layer produces:

1. **Criticality score (0–100)** — the headline triage number. It is the **percentile rank** of the
   patient's raw model score against a reference distribution of all ICU-hours: a score of **94
   means "riskier than 94% of ICU-hours,"** not "94% chance." It is a **relative risk rank, not a
   probability.** Its only job is to spread patients into a readable 0–100 range so beds are easy
   to order; it is monotonic with the raw score, so it never changes the ranking.
2. **Calibrated probability (%)** — shown underneath the headline. This *is* an honest probability:
   an isotonic-regression calibration (fit on the validation set) of the raw score into
   *"of past hours that scored like this, X% went on to be in the pre-sepsis window."* Because
   sepsis is rare (~1.8% of hours), even a genuinely high-risk hour calibrates to a **small-looking
   number (e.g. 8–15%)** — that is correct, not "safe." This is exactly why the 0–100 criticality,
   not the probability, is the headline.
3. **Tier** — `LOW (<50) / MODERATE (50–75) / HIGH (75–90) / CRITICAL (≥90)`, from the criticality
   score. Display defaults; validated (below) to line up with elevated calibrated risk.
4. **Trend** — `rising / steady / falling`, the change in criticality over the last few hours
   (causal — past hours only). A patient trending up fast is more urgent than one flat-high. Kept
   as a separate arrow, not folded into the headline number.

Plus the **top-3 SHAP drivers** in plain clinical language (e.g. "latest lactate ↑ · mean arterial
pressure falling ↓"), so a flag is explainable rather than a black box.

**Example output card**
```
p017091   CRITICALITY 99.9/100   [CRITICAL]   → steady
          calibrated risk 45.6%   |   why: hours in ICU ↑ · latest temperature ↑ · peak temperature (6h) ↑
```

### 6.2 Validation ("does it make sense?")

`07_criticality.py` checks and saves plots to `artifacts/plots/`:

- **Septic vs non-septic**: mean criticality on pre-sepsis test hours **83.8** vs non-sepsis
  **48.9** — strong, correct separation.
- **Calibration reliability** (`calibration_reliability.png`): predicted vs observed sepsis rate
  tracks the diagonal; **Brier score 0.0169** (low = well-calibrated).
- **Trajectories** (`criticality_trajectories.png`): criticality climbs as septic patients approach
  onset.
- **8-bed triage table**: the demo beds ordered by criticality, each with prob / tier / trend /
  drivers. The four known-septic demo beds rank top by calibrated risk.

### 6.3 Layer-specific caveats

- **Criticality (0–100) is a relative risk rank, not a probability.** "94" = riskier than 94% of
  reference hours, not "94% chance."
- **Calibrated probability is an empirical frequency on this dataset's two hospitals** — not a
  clinically validated or prospectively tested probability. Small numbers are expected for a rare
  event and do **not** mean "safe."
- **Time-in-ICU effect**: the model leans on `ICULOS` (hours in ICU), so a snapshot taken at a
  patient's *last* hour (as the 8-bed demo does) tends to score high across the board — "hours in
  ICU ↑" often shows up as the top driver. This is real model behavior, not a bug; for a live
  system, score the *current* hour and read the trend and the other drivers alongside it.

The layer's code lives in `sepsis_pipeline/criticality.py` (`calibrated_probability`,
`criticality_score`, `tier_from_score`, `criticality_trend`, `top_shap_drivers`) and
`07_criticality.py`; `evaluate_model.ipynb` §12 renders the reliability curve + 8-bed triage cards.
See [`INTEGRATION.md`](INTEGRATION.md) for producing a criticality score for a live patient-hour.

---

## 7. Alternatives Explored

The final model (§4.2) was the best result found across **five independent experiment families**.
None of the alternatives below beat it, so it was left unchanged; this section records what was
tried and why, for future reference. Every experiment reused the exact same train/val/test split,
the same 214 (unless noted) engineered features, and the same `U1 − U0` target, so comparisons
within each experiment are apples-to-apples, and none modified the committed model. The
substantial studies (§7.4–§7.6) are retained as reproducible scripts under
[`../sepsis_pipeline/experiments/`](../sepsis_pipeline/experiments/) with their raw results.

*Note on the split*: the first experiments (§7.1–§7.3) predate the split adjustment that pins the
8 demo patients to test (§4.1), so their "production" baseline rows differ slightly from the
current §5.2 numbers — the *conclusions* are unaffected, since the swap only moved 7 of 40,336
patients.

### 7.1 XGBoost + CatBoost ensemble

Train XGBoost and CatBoost regressors on the identical target/features/split (same utility-gain
trick, not ordinary classifiers), then average all three models' scores.

| Model | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (production) | 0.8458 | 0.1224 | 0.0691 | **0.7049** | 0.1259 | 5.5605 | **0.4382** |
| XGBoost (same target) | 0.8437 | 0.1248 | 0.0771 | 0.6542 | 0.1379 | 5.5605 | 0.4353 |
| CatBoost (same target) | 0.8365 | 0.1304 | 0.0722 | 0.6467 | 0.1300 | 5.4167 | 0.4138 |
| **Ensemble (mean of all 3)** | 0.8453 | 0.1299 | 0.0700 | 0.6912 | 0.1271 | **5.6180** | 0.4324 |

**Conclusion**: XGBoost and CatBoost individually underperformed the tuned LightGBM on AUROC and
Utility, so averaging them pulled the ensemble slightly below LightGBM alone on the two most
important metrics. Not adopted.

### 7.2 Lag-stacked features instead of rolling min/max/std (Team 2's approach)

The #2-ranked team (Du, Sadr, de Chazal) stacked each hour's feature vector for the last 5 hours
side-by-side instead of summarizing with rolling stats. This experiment replaced the 96 rolling
mean/min/max/std features with 48 lag-stacked forward-filled values (`h-1`..`h-4`), keeping
everything else. Net: 214 → 166 features.

| Model | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production (rolling min/max/std, 214 features) | 0.8458 | 0.1224 | 0.0691 | **0.7049** | 0.1259 | **5.5605** | **0.4382** |
| Lag-stack h-1..h-4 (166 features) | **0.8461** | **0.1248** | **0.0780** | 0.6495 | **0.1393** | 5.5461 | 0.4334 |

**Conclusion**: a precision/recall trade-off, not a strict improvement — AUROC/AUPRC/Precision/F1
edge up with fewer features, but Recall and Utility (the two metrics prioritized here) drop. Not
adopted.

### 7.3 Reward/penalty and threshold sensitivity sweeps

`evaluate_model.ipynb` (§5, §10, §11) explores how far the *existing, unchanged* model's results
move under different false-alarm penalties, miss penalties, reward magnitudes, and decision
thresholds — none retrain the model, only re-grade its scores. Key finding: the official published
constants (`max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05`) and the utility-optimal threshold already sit
at a well-balanced point; no combination improved both Recall and Utility simultaneously without
trading one for the other. This is expected — for a fixed model, both derive from the same
threshold/scoring choice, so improving both together requires a genuinely better-discriminating
model, which §7.1–§7.2 and §7.4 attempted without success.

### 7.4 Broader hyperparameter search (Optuna) + nested cross-validation

Script: [`../sepsis_pipeline/experiments/optuna_nested_cv.py`](../sepsis_pipeline/experiments/optuna_nested_cv.py)

A broad **Optuna** Bayesian search (50 trials over ten LightGBM parameters, objective = validation
AUPRC), then a **5-fold nested cross-validation** (patient-grouped, outcome-stratified outer folds;
inner Optuna per fold) for an honest, generalization-robust estimate.

| Model | AUROC | AUPRC | Recall | Utility |
|---|---:|---:|---:|---:|
| Production (single split) | **0.8465** | 0.1252 | 0.6694 | **0.4554** |
| Optuna-best (single split) | 0.8452 | **0.1284** | **0.6783** | 0.4403 |
| Nested CV (mean ± std) | 0.8381 ± 0.0114 | 0.1240 ± 0.0078 | 0.6523 ± 0.0374 | 0.4070 ± 0.0233 |

**Conclusion**: Optuna did not beat production (only a marginally different precision/recall
trade). The most valuable output is the **nested-CV estimate**: slightly below the single-split
test figure, telling us the headline number is *mildly optimistic* and the trustworthy,
generalization-robust performance is about **0.838 AUROC / 0.41 Utility** (±0.011 / ±0.023).
Nothing adopted.

### 7.5 Split-robustness: 10 groupings, demo patients pinned to test

Script: [`../sepsis_pipeline/experiments/split_robustness.py`](../sepsis_pipeline/experiments/split_robustness.py)

Keeping the 8 demo patients pinned to test (§4.1), re-shuffle all other patients into 10 different
80/10/10 groupings and retrain the production-config model on each.

| | AUROC | AUPRC | Recall | Utility |
|---|---:|---:|---:|---:|
| Mean over 10 groupings | 0.846 | 0.129 | 0.675 | 0.429 |
| Range (min–max) | 0.829–0.859 | 0.116–0.148 | 0.626–0.727 | 0.403–0.449 |
| Std | ±0.008 | ±0.010 | ±0.039 | ±0.013 |

**Conclusion**: production sits almost exactly on the mean — a representative split, not a lucky
one. The ±0.008 AUROC swing (0.829–0.859) is pure split noise (identical model config across all
10), and the test "winners" are not the validation winners — the signature of noise, not of a
better model. Cherry-picking the best grouping would report an inflated, non-reproducible number;
the honest figure is the mean, where production already sits.

### 7.6 Matched vs. random backfill of the demo-patient swap

Script: [`../sepsis_pipeline/experiments/matched_replacement.py`](../sepsis_pipeline/experiments/matched_replacement.py)

Pinning the 8 demo patients to test required moving 7 out of train/val, backfilled with *random*
patients. This study instead backfilled with **clinically similar** patients (same hospital, same
outcome, nearest-neighbor on a per-patient summary).

| Backfill method | AUROC | AUPRC | Recall | Utility |
|---|---:|---:|---:|---:|
| Random (current production) | 0.8465 | 0.1252 | 0.6694 | **0.4554** |
| Matched (clinical similarity) | **0.8469** | **0.1258** | 0.6606 | 0.4434 |

**Conclusion**: statistically identical — swapping 7 of 32,268 training patients cannot move the
model. Matched backfill is more *principled* but changes nothing (and would slightly soften the
"genuinely held-out demo" property). Production (random backfill) kept.

### 7.7 Overall

AUROC stayed within roughly 0.829–0.859 across all five families, with a nested-CV honest estimate
of ~0.838 — a real ceiling for this feature set that hyperparameter tuning and split juggling
cannot break past. The consistency is itself a result: the reported performance is stable and
trustworthy, not a lucky configuration. The only remaining lever likely to move the ceiling
meaningfully is richer feature engineering, which was out of scope.

---

## 8. Limitations

- **Retrospective data only.** No prospective or live-deployment validation has been performed.
- **Test set is not a true generalization test.** It is a random 10% holdout from the same two
  training hospitals, not an unseen third hospital. The original challenge's central finding was
  that every top team's utility score collapsed on a genuinely unseen hospital — this model's test
  performance should be read as an upper bound on real-world cross-hospital generalization, not an
  estimate of it. The nested-CV honest estimate (~0.838 AUROC, §7.4) is the more trustworthy figure.
- **Low precision at the current operating point** (~8%): most flags are false alarms, typical for
  this severely imbalanced task. The model is suited to prioritizing clinical attention, not
  standalone diagnosis.
- **Not a clinical device.** A course/research project artifact; no care decision should depend on
  its output.

---

## 9. References

[1] Morrill, J., Kormilitzin, A., Nevado-Holgado, A., Swaminathan, S., Howison, S., Lyons, T.
"The Signature-Based Model for Early Detection of Sepsis From Electronic Health Records in the
Intensive Care Unit." *Computing in Cardiology*, 2019.
PDF: https://www.cinc.org/archives/2019/pdf/CinC2019-014.pdf
Code: https://github.com/jambo6/physionet_sepsis_challenge_2019

[2] Du, J., Sadr, N., de Chazal, P. "Automated Prediction of Sepsis Onset Using Gradient Boosted
Decision Trees." *Computing in Cardiology*, 2019.
https://physionet.org/files/challenge-2019/1.0.0/papers/CinC2019-423.pdf

[3] Zabihi, M., Kiranyaz, S., Gabbouj, M. "Sepsis Prediction in Intensive Care Unit Using Ensemble
of XGBoost Models." *Computing in Cardiology*, 2019.
https://physionet.org/files/challenge-2019/1.0.0/papers/CinC2019-238.pdf

[4] Reyna, M. A., Josef, C., Jeter, R., Shashikumar, S. P., Westover, M. B., Nemati, S.,
Clifford, G. D., Sharma, A. "Early Prediction of Sepsis From Clinical Data: The
PhysioNet/Computing in Cardiology Challenge 2019." *Critical Care Medicine*, 48(2), 210-217, 2020.
https://doi.org/10.1097/CCM.0000000000004145

[5] PhysioNet. Sepsis Challenge 2019 dataset, version 1.0.0.
https://physionet.org/content/challenge-2019/1.0.0/

[6] Official utility-score evaluation code. https://github.com/physionetchallenges/evaluation-2019

[7] Reyna, M. A., Clifford, G. D. "Voting of Predictive Models for Clinical Outcomes: Consensus of
Algorithms for the Early Prediction of Sepsis." arXiv:2012.11013, 2020.
https://arxiv.org/abs/2012.11013
