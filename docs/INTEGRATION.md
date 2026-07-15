# Integration Guide

This document is for whoever is wiring the trained sepsis-risk model into the rest of the
Connected ICU Ward Digital Twin (the "Project 2" extension the Project 1 report reserved for
later). It explains exactly what artifact to load, what it expects as input, and what it returns
— no need to read the training code to integrate it.

## What to load

```python
from joblib import load

bundle = load("sepsis_pipeline/artifacts/model_bundle.joblib")
model          = bundle["model"]           # trained LightGBM regressor (lightgbm.LGBMRegressor)
feature_names  = bundle["feature_names"]   # ordered list of 214 column names the model expects
threshold      = bundle["threshold"]       # decision threshold (float) chosen on validation data
```

`model_bundle.joblib` is fully self-describing — it carries its own feature order and decision
threshold, so nothing else needs to be hard-coded on the consuming side.

## What the model needs as input

A row of **214 engineered features per patient-hour**, built by `sepsis_pipeline/feature_engineering.py`
(function `build_patient_features`) from the raw hourly vitals/labs. It is **not** safe to feed the
model raw vitals directly — it expects the full derived feature set (rolling stats, forward-fills,
missingness flags, clinical scores, etc.), in the exact column order given by `feature_names`.

Practical integration path for a live/streaming system (e.g. the existing twin backend):

1. Maintain each bed's rolling hourly history exactly as the twin backend already does.
2. On each new hourly reading, run `feature_engineering.build_patient_features()` (or an
   equivalent live-streaming re-implementation of the same logic) over that patient's history up
   to and including the current hour.
3. Select columns in `feature_names` order and call `model.predict(X)` — returns one continuous
   score per row (**not** a 0-1 probability; see caveat below).
4. Compare the score to `threshold`: `score >= threshold` → flag as risky.

## Interpreting the output

The model's raw output is **not a calibrated probability**. It is trained to regress a utility-gain
score (see `docs/REPORT.md` §3.2 for the full mechanics) — realistically ranging from about −0.1 to
+1.5. Treat it only as a ranking/risk score: higher means more model-confidence in near-term sepsis
risk, and the provided `threshold` is the correct cutoff for a binary flag. Do not rescale it and
present it as "X% probability" without a proper calibration step (deliberately out of scope here —
see the Limitations section of the report for why).

## Explainability (SHAP)

For a given prediction, per-feature attributions are available two ways:

- **Static reference set**: `artifacts/shap_values.npy` (array, shape `[n_sampled_rows, 214]`) paired
  row-for-row with `artifacts/shap_sample_features.parquet` — a pre-computed sample from the test
  set, useful for offline analysis or dashboards that don't need live explanations.
- **Live explanations**: load `artifacts/shap_explainer.joblib` (a `shap.TreeExplainer`) and call
  `explainer.shap_values(X)` on any new feature row built the same way as training data, to get a
  live per-feature breakdown for that specific prediction.

## Files in `sepsis_pipeline/artifacts/`

| File | What it is |
|---|---|
| `model_bundle.joblib` | The model + feature order + threshold (see above) |
| `shap_values.npy`, `shap_sample_features.parquet`, `shap_explainer.joblib` | Explainability artifacts (see above) |
| `metrics.json` | Full performance numbers (train/val/test, per-hospital) backing `docs/REPORT.md` |
| `hyperparam_search_results.csv` | The 27-combination hyperparameter grid search results |
| `{train,val,test}_patient_ids.json` | Exact patient IDs in each split, for reproducibility |
| `plots/*.png` | ROC/PR curves, threshold sweep, SHAP summary plots |

## Reproducing or retraining

Run order (from `sepsis_pipeline/`, see `sepsis_pipeline/README.md` for details):

```bash
pip3 install -r requirements.txt
python3 01_build_dataset.py          # parses training_setA/ + training_setB/
python3 02_feature_engineering.py    # builds the 214-feature matrix
python3 03_train_model.py            # trains + saves model_bundle.joblib
python3 04_explain_shap.py           # generates SHAP artifacts
python3 05_sanity_checks.py          # verifies no leakage/causality violations
```

## Important caveats before treating this as production-ready

- **Not a clinical device.** Retrospective research data only; no care decision should depend on
  its output.
- **Test performance is likely optimistic.** The test set is a held-out 10% of the *same two*
  training hospitals, not a genuinely unseen hospital. The original PhysioNet/CinC 2019 challenge's
  key finding was that every top team's score collapsed on a truly unseen third hospital — this
  model has not been validated against that harder bar.
- **Precision is low by design** (~7-8% at the current threshold) — most flags are false alarms.
  This is expected given how rare sepsis is in the data, but means the model is suited to
  *prioritizing attention*, not standalone diagnosis.

Full technical detail, including the exact reward/penalty formulas, model architecture, and
results, is in [`docs/REPORT.md`](REPORT.md).
