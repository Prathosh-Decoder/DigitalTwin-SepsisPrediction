# Criticality & Prioritization Layer

A layer **on top of** the trained sepsis model that turns its raw score into a clinician-readable
triage view — answering *"who do I look at first, and why."* It does **not** retrain or modify the
model; it is a separate artifact (`artifacts/criticality_calibrator.joblib`) fitted by
`sepsis_pipeline/07_criticality.py`. The model bundle stays byte-for-byte unchanged.

## The four views

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
   score. Display defaults; validated (see below) to line up with elevated calibrated risk.

4. **Trend** — `rising / steady / falling`, the change in criticality over the last few hours
   (causal — past hours only). A patient trending up fast is more urgent than one flat-high. Kept
   as a separate arrow, not folded into the headline number.

Plus the **top-3 SHAP drivers** in plain clinical language (e.g. "latest lactate ↑ · mean arterial
pressure falling ↓"), so a flag is explainable rather than a black box.

### Example output card
```
p017091   CRITICALITY 99.9/100   [CRITICAL]   → steady
          calibrated risk 45.6%   |   why: hours in ICU ↑ · latest temperature ↑ · peak temperature (6h) ↑
```

## Validation ("does it make sense?")

`07_criticality.py` checks and saves plots to `artifacts/plots/`:

- **Septic vs non-septic**: mean criticality on pre-sepsis test hours **83.8** vs non-sepsis
  **48.9** — strong, correct separation.
- **Calibration reliability** (`calibration_reliability.png`): predicted vs observed sepsis rate
  tracks the diagonal; **Brier score 0.0169** (low = well-calibrated).
- **Trajectories** (`criticality_trajectories.png`): criticality climbs as septic patients approach
  onset.
- **8-bed triage table**: the demo beds ordered by criticality, each with prob / tier / trend /
  drivers.

## Honest caveats (read before showing anyone a number)

- **Criticality (0–100) is a relative risk rank, not a probability.** "94" = riskier than 94% of
  reference hours, not "94% chance."
- **Calibrated probability is an empirical frequency on this dataset's two hospitals** — not a
  clinically validated or prospectively tested probability. Small numbers are expected for a rare
  event and do **not** mean "safe."
- **Time-in-ICU effect**: the model leans on `ICULOS` (hours in ICU), so a snapshot taken at a
  patient's *last* hour (as the 8-bed demo does) tends to score high across the board — "hours in
  ICU ↑" often shows up as the top driver. This is real model behavior, not a bug; for a live
  system, score the *current* hour and read the trend and the other drivers alongside it.
- **Not a clinical device.** A model flag is not "sepsis in exactly 6 hours," and most flags are
  false alarms (precision ≈ 8%). This supports prioritization, not diagnosis. No care decision
  should depend on it.

## Files

- `sepsis_pipeline/criticality.py` — the reusable functions (`calibrated_probability`,
  `criticality_score`, `tier_from_score`, `criticality_trend`, `top_shap_drivers`, plain-name map).
- `sepsis_pipeline/07_criticality.py` — fits the calibrator (validation) + reference (train), saves
  the artifact, runs the validation, saves plots.
- `artifacts/criticality_calibrator.joblib` — `{calibrator, reference_quantiles, tier_bands,
  feature_plain_names}`. Kept separate from `model_bundle.joblib` so the model stays frozen.
- `evaluate_model.ipynb` §12 — renders the reliability curve + 8-bed triage cards.

See [`INTEGRATION.md`](INTEGRATION.md) for producing a criticality score for a live patient-hour.
