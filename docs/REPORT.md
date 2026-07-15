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
On a held-out test set, the model achieves an AUROC of 0.846, an AUPRC of 0.122, and a normalized
utility score of 0.438 — exceeding the original competition winner's officially reported score of
0.360, though under an easier evaluation condition (see §6, Limitations).

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
interpretable, which would undermine the SHAP-based explainability goals of this project (§5.5).

---

## 4. Model Architecture

### 4.1 Data split

Patients — not individual rows — are split 80% train / 10% validation / 10% test, so no patient's
hours are divided across splits. The split is stratified by whether a patient is ever septic and
performed independently within each hospital before combining, ensuring proportional
representation of both hospitals and outcome classes in every split. Training data fits the model;
validation data drives early stopping, the hyperparameter search, and threshold selection; the
test set is touched exactly once, to produce the results in §5.

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

This configuration is the final, adopted model. Several further alternatives — an XGBoost/CatBoost
ensemble, and replacing the rolling-statistics features with a lag-stacking approach — were tried
afterward and did not improve on it; see [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full record.

---

## 5. Results

### 5.1 Metric definitions

- **AUROC** (Area Under the ROC Curve) — the probability that, given one randomly chosen
  septic hour and one randomly chosen non-septic hour, the model assigns a higher risk score to
  the septic one. It is computed by sweeping every possible decision threshold and plotting the
  true-positive rate against the false-positive rate at each one; the area under that curve is
  the metric. 0.5 means the model ranks no better than a coin flip; 1.0 means it ranks every
  septic hour above every non-septic hour. Because it is computed over all thresholds, AUROC does
  not depend on which single threshold is eventually chosen for a hard flag/no-flag decision.
- **AUPRC** (Area Under the Precision-Recall Curve) — the same sweeping-over-thresholds idea as
  AUROC, but plotting precision against recall instead of true/false-positive rate. It is more
  informative than AUROC specifically because only ~1.8% of rows in this dataset are truly
  positive: a model can score deceptively well on AUROC just by being good at the easy majority
  (correctly ignoring obviously-healthy hours), while AUPRC exposes how good it really is at
  finding the rare true cases. A random/no-skill model's AUPRC equals the positive rate itself
  (~0.018 here), so AUPRC is best read relative to that baseline, not against 1.0.
- **Precision** — of every hour the model flags as risky (at the chosen decision threshold), the
  fraction that is a genuine sepsis-warning hour. Low precision means many false alarms per real
  catch.
- **Recall** — of every genuine sepsis-warning hour in the data, the fraction the model actually
  flags. Low recall means many real cases go undetected (false negatives).
- **F1** — the harmonic mean of precision and recall, a single number that penalizes models which
  do well on one at the expense of the other.
- **Lift@10%** — take the 10% of hours the model scores as riskiest; Lift@10% is how much more
  concentrated true sepsis-warning hours are in that group compared to picking 10% at random. A
  lift of 5x means those top-10%-riskiest hours contain five times more real cases than a random
  10% would — a practical measure of how useful the ranking is for prioritizing limited clinical
  attention, independent of any specific decision threshold.
- **Utility** — the official PhysioNet/CinC 2019 challenge's own custom, time-aware scoring rule,
  and the metric this model is directly trained to optimize (§3.2). Unlike the metrics above, it
  is not a generic statistical measure — it explicitly rewards catching a case at the right time
  and explicitly punishes catching it too late or missing it, which the other metrics cannot
  express at all. Full calculation detail follows in §5.2.

### 5.2 How Utility is calculated

Utility is computed in three steps.

**Step 1 — score every hour against a timing curve, not a flat right/wrong.** For every hour of
every patient, two hypothetical scores are computed from the *known* outcome (this is only
possible retrospectively, using data whose true future is already known — see §3.2 for why the
model itself never sees these numbers): `U1`, the credit earned if that hour had been flagged as
risky, and `U0`, the credit earned if it had been left unflagged. For a patient who does develop
sepsis at true onset time `t_sepsis`, with `dt = (this hour) − t_sepsis`:

| Timing (`dt`) | Flagging this hour earns... | Staying silent earns... |
|---|---|---|
| More than 12h before onset, or more than 3h after | −0.05 (flat false-alarm cost) | 0 |
| 12h to 6h before onset (ramp-up) | ramps linearly from 0 up to +1.0 | 0 |
| 6h before onset to 3h after (decay) | ramps back down from +1.0 to 0 | ramps down from 0 to −2.0 |

The reward for a correct flag peaks at exactly **6 hours before onset** — matching the horizon
`SepsisLabel` itself is built around — and the penalty for staying silent only starts accruing
*after* that point, worsening the closer the patient gets to actually crashing. For a patient who
never develops sepsis at all, there is no timing curve: flagging always costs the flat −0.05 and
staying silent always earns 0, at every hour of their stay.

**Step 2 — grade the model's actual predictions against that curve.** At every hour, the model
either flagged (predicted 1) or didn't (predicted 0). The score it actually earned for that hour
is `U1` if it flagged, `U0` if it didn't. Summing this across every hour of every patient gives
the cohort's *observed* total utility.

**Step 3 — normalize against two reference points.** Two more totals are computed the same way:
`U_inaction` (what the total would be if the model had predicted "no risk" for every single hour
of every patient — the "do nothing" baseline) and `U_best` (what the total would be if every hour
had received whichever of `U1`/`U0` was larger — an oracle with perfect foresight and perfect
timing). The final reported Utility is:

```
Utility = (U_observed − U_inaction) / (U_best − U_inaction)
```

This normalization is what makes the number interpretable on a fixed scale regardless of dataset
size or class balance: a model that never flags anything scores **0** (it earns exactly the
inaction baseline), and a hypothetical perfect model scores **1**. A negative score means the
model performed *worse* than doing nothing at all — entirely possible if it produces enough
costly, badly-timed false alarms and missed cases.

**Worked example.** For a patient whose true onset is hour 50, here is what a single correctly
timed hour is worth at different points, and why timing so directly shapes the score:

| Hour | Timing | Flag earns (`U1`) | Silent earns (`U0`) |
|---:|---|---:|---:|
| 44 | 6h early (peak reward) | **1.000** | 0.000 |
| 47 | 3h early | 0.667 | −0.667 |
| 50 | at onset | 0.333 | −1.333 |
| 53 | 3h late (window closes) | 0.000 | −2.000 |

Flagging at hour 44 is worth strictly more in isolation than flagging at hour 47 or later — but
because rewards accumulate across *every* hour a model correctly flags (not just one), and the
cost of staying silent keeps worsening the closer a patient gets to onset, the best achievable
strategy is still to start flagging as early as the evidence supports and keep flagging, not to
hold out for a single "best" hour.

### 5.3 Performance

| Split | AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.8816 | 0.1959 | 0.0747 | 0.7696 | 0.1362 | 6.076 | 0.4988 |
| Validation | 0.8443 | 0.1346 | 0.0683 | 0.7097 | 0.1247 | 5.409 | 0.4354 |
| **Test** | **0.8458** | **0.1224** | **0.0691** | **0.7049** | **0.1259** | **5.560** | **0.4382** |
| Test — Hospital A | 0.8308 | 0.1179 | 0.0699 | 0.7344 | 0.1276 | 5.364 | 0.4567 |
| Test — Hospital B | 0.8608 | 0.1427 | 0.0678 | 0.6576 | 0.1229 | 5.622 | 0.4075 |

At the test-set operating threshold: 1,961 true positives, 821 false negatives, 26,420 false
positives, 126,831 true negatives. Validation and test scores sit close to each other and
noticeably below training scores, indicating the model is not substantially overfit.

### 5.4 What these values mean in practice

- **AUROC = 0.846**: given one random septic hour and one random non-septic hour, the model ranks
  the septic one riskier about 85% of the time. Comfortably above the 0.5 (random) baseline, and
  in a range broadly consistent with published results on this same task (§2).
- **AUPRC = 0.122**: roughly **7x** the no-skill baseline for this data (~0.018, the true positive
  rate) — meaningful ranking ability on the rare positive class, even though the absolute number
  looks low next to AUROC. Low absolute AUPRC is expected and typical for a task this imbalanced,
  not a sign of a broken model.
- **Recall = 0.705**: the model catches roughly 7 out of every 10 real sepsis-warning hours in the
  test set; the other 3 in 10 (821 hours) go unflagged.
- **Precision = 0.069**: of everything flagged as risky, only about 7% is a genuine warning hour —
  the other 93% are false alarms. This is a direct consequence of how rare true positives are in
  the data (~1.8% of rows); even a strong ranking model produces many false alarms in absolute
  terms when the positive class this small is the target.
- **Lift@10% = 5.56x**: if a clinical team could only closely monitor the riskiest 10% of
  patient-hours flagged, that group would contain about 5.6x more real sepsis cases than watching
  a random 10% — useful for prioritization even though precision at the binary threshold is low.
- **Utility = 0.438**: on the 0-to-1 scale described in §5.2 (0 = never flag anything, 1 = perfect
  foresight), the model captures **43.8%** of the maximum achievable timing-aware score on the
  test set. For reference, the original 2019 challenge's *winning* team scored 0.360 on the
  official hidden test set — though that comparison isn't fully apples-to-apples, since their test
  set included a genuinely unseen third hospital and this project's test set does not (§6).

### 5.5 Explainability

SHAP (TreeExplainer) values were computed on a stratified sample of 10,000 test-set rows (2,000
positive, 8,000 negative). Plots are available at `sepsis_pipeline/artifacts/plots/`:
`roc_curve.png`, `pr_curve.png`, `threshold_sweep.png`, `shap_summary_beeswarm.png`, and
`shap_importance_bar.png`.

---

## 6. Limitations

- **Retrospective data only.** No prospective or live-deployment validation has been performed.
- **Test set is not a true generalization test.** It is a random 10% holdout from the same two
  training hospitals, not an unseen third hospital. The original challenge's central finding was
  that every top team's utility score collapsed on a genuinely unseen hospital — this model's test
  performance should be read as an upper bound on real-world cross-hospital generalization, not an
  estimate of it.
- **Low precision at the current operating point** (~7%): most flags are false alarms, typical for
  this severely imbalanced task. The model is suited to prioritizing clinical attention, not
  standalone diagnosis.
- **Not a clinical device.** A course/research project artifact; no care decision should depend on
  its output.

---

## References

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
