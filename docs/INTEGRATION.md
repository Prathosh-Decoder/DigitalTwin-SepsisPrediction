# Integration Guide

This document is for whoever is wiring the trained sepsis-risk model into another system (e.g. a
Node.js/Express-based ICU digital twin backend). It is self-contained — everything needed to
build a valid input and call the model is below; you should not need to open
`feature_engineering.py` or `config.py` to get started, though they're the source of truth if
anything here is unclear.

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

**This model is Python-only** (LightGBM object, pickled via `joblib`). A non-Python caller (e.g.
a Node.js backend) cannot load this file directly — the standard pattern is to wrap this model in
a small Python inference service (e.g. FastAPI/Flask) exposing a `POST /predict`-style endpoint,
and have the other system call that over HTTP. Building that service is not included here; this
document only covers the Python-side contract (input in, prediction out).

## Raw input schema (what you need to collect, per patient, per hour)

Every column below must be present in your raw per-hour data before it can be turned into model
features. Names and units match the PhysioNet Sepsis Challenge 2019 format this model was trained
on (`Temp` in °C, blood pressures in mmHg, etc.). Missing values are expected and fine — use
`NaN`/`null`, never a fabricated value like 0.

| Column | Meaning | Notes |
|---|---|---|
| `patient_id` | Any stable per-patient identifier | Int or string, just needs to be consistent across a patient's rows |
| `hospital` | Any fixed string, e.g. `"LIVE"` | Required by the feature function but excluded from the model's actual features — value doesn't affect predictions |
| `ICULOS` | Hours since ICU admission | Integer, increments by 1 each hour, starting at 1 |
| `SepsisLabel` | Stub with `0` for every row | Required column for the function signature (it's the training label), but never used as a model input — always safe to stub |
| `Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime` | Demographics / admission timing | Static per patient; same convention as the PhysioNet dataset (`Gender`: 0/1, `Unit1`/`Unit2`: 0/1/null for ICU type) |
| `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2` | The 8 core vitals | |
| `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets` | The 26 lab values | Realistically `NaN` most hours — labs aren't drawn every hour, and that's expected/handled |

That's 43 columns total (4 identifiers/label + 5 demographics + 34 vitals/labs) — matches the
original PhysioNet `.psv` schema plus `patient_id`/`hospital`.

## Turning raw rows into model input

**You must pass a patient's full history up to and including the current hour, not just the
latest single reading.** Many features are rolling statistics/deltas over the last 6-24 hours; a
single-row call will not error, but will silently produce degraded features (rolling stats
collapse to the current value, deltas come out `NaN`).

```python
import pandas as pd
from feature_engineering import build_patient_features

# One row per hour so far for this patient, oldest first. Only a few columns shown for brevity --
# every column from the schema table above must be present (NaN where not measured).
patient_history = pd.DataFrame([
    {"patient_id": 1, "hospital": "LIVE", "ICULOS": 1, "SepsisLabel": 0,
     "HR": 82, "O2Sat": 98, "Temp": None, "SBP": 120, "MAP": 88, "DBP": 70, "Resp": 16,
     "Age": 65.0, "Gender": 1, "Unit1": 1, "Unit2": 0, "HospAdmTime": -0.5,
     # ... remaining vitals/labs from the schema table, NaN if not measured this hour
     },
    {"patient_id": 1, "hospital": "LIVE", "ICULOS": 2, "SepsisLabel": 0,
     "HR": 85, "O2Sat": 97, "Temp": 37.1, "SBP": 118, "MAP": 85, "DBP": 68, "Resp": 18,
     "Age": 65.0, "Gender": 1, "Unit1": 1, "Unit2": 0, "HospAdmTime": -0.5,
     },
    # ... one row per hour, up to and including "now"
])

engineered = build_patient_features(patient_history)   # same row count as input, 214+ columns
current_hour_features = engineered.iloc[[-1]]           # the row for "now" -- the most recent hour

X = current_hour_features[feature_names]                # select + order exactly as the model expects
score = model.predict(X)[0]                              # one continuous score, NOT a probability
is_risky = bool(score >= threshold)
```

## Interpreting the output

The model's raw output is **not a calibrated probability**. It is trained to regress a utility-gain
score (see `docs/REPORT.md` §3.2 for the full mechanics) — realistically ranging from about −0.1 to
+1.5. Treat it only as a ranking/risk score: higher means more model-confidence in near-term sepsis
risk, and `threshold` (from the bundle) is the correct cutoff for a binary flag. Do not rescale it
and present it as "X% probability" without a proper calibration step (deliberately out of scope
here — see the Limitations section of the report for why).

## Explainability (SHAP)

For a given prediction, per-feature attributions are available two ways:

- **Static reference set**: `artifacts/shap_values.npy` (array, shape `[n_sampled_rows, 214]`) paired
  row-for-row with `artifacts/shap_sample_features.parquet` — a pre-computed sample from the test
  set, useful for offline analysis or dashboards that don't need live explanations.
- **Live explanations**: load `artifacts/shap_explainer.joblib` (a `shap.TreeExplainer`) and call
  `explainer.shap_values(X)` on any new feature row built the same way as training data (`X` from
  the worked example above works directly), to get a live per-feature breakdown for that specific
  prediction.

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
