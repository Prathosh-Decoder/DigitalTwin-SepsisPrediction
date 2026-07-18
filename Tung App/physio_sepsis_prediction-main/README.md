# PhysioNet Sepsis Probability Predictor

This repository provides a ready-to-use pickle model that accepts one ICU patient's hourly history and returns a calibrated sepsis probability for the PhysioNet/CinC 2019 six-hour early-warning task.

> Research and educational use only. This model is not validated or approved for clinical diagnosis, triage, or treatment decisions.

## Model File

```text
models/sepsis_next_6h_predictor.pkl
```

The pickle contains the XGBoost model, causal feature engineering, sigmoid probability calibrator, ICU-phase decision thresholds, and model metadata. It is approximately 850 KB and does not require Git LFS.

## Download

Clone the repository:

```bash
git clone git@github.com:nttssv/physio_sepsis_prediction.git
cd physio_sepsis_prediction
```

For HTTPS instead of SSH:

```bash
git clone https://github.com/nttssv/physio_sepsis_prediction.git
cd physio_sepsis_prediction
```

To update an existing clone:

```bash
git pull origin main
```

To download only the pickle from this public repository:

```bash
curl -L \
  -o sepsis_next_6h_predictor.pkl \
  https://raw.githubusercontent.com/nttssv/physio_sepsis_prediction/main/models/sepsis_next_6h_predictor.pkl
```

After downloading, compare its SHA-256 value with `artifact_sha256` in `models/sepsis_next_6h_predictor_metadata.json`:

```bash
shasum -a 256 models/sepsis_next_6h_predictor.pkl
```

## Install

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-inference.txt
```

Pickle files can execute code while loading. Only load this artifact from this trusted repository and verify the source before using it.

## Predict From a Patient File

The patient file can be a pipe-delimited PhysioNet `.psv` file or a comma-delimited `.csv` file. Each row represents one chronological ICU hour.

```bash
python examples/use_sepsis_pickle.py /path/to/patient.psv
```

Example output:

```json
{
  "icu_hour": 30.0,
  "sepsis_probability_next_6h": 0.00613,
  "sepsis_risk_percent": 0.613,
  "decision_threshold": 0.028,
  "sepsis_alarm": false,
  "raw_model_score": 0.06426,
  "model_version": "1.0.0"
}
```

## Python API

```python
import pickle
import pandas as pd

with open("models/sepsis_next_6h_predictor.pkl", "rb") as handle:
    predictor = pickle.load(handle)

# Include all observations available up to the current patient hour.
patient_history = pd.read_csv("patient.psv", sep="|")

# Latest calibrated probability as a float between 0 and 1.
probability = predictor.predict_proba(patient_history)
print(f"Sepsis risk: {probability:.2%}")

# Latest probability, threshold, alarm decision, and metadata.
result = predictor.predict_patient(patient_history)
print(result)

# Causal probability and alarm trajectory for every supplied hour.
trajectory = predictor.predict_trajectory(patient_history)
print(trajectory.tail())
```

The predictor can also accept a list of hourly dictionaries:

```python
patient_records = [
    {"HR": 88, "O2Sat": 98, "MAP": 76, "Resp": 18, "ICULOS": 1},
    {"HR": 96, "O2Sat": 95, "MAP": 69, "Resp": 22, "ICULOS": 2},
]

probability = predictor(patient_records)
```

## Input Format

The interface recognizes the 40 PhysioNet Challenge variables:

```text
HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2,
BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN,
Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct,
Glucose, Lactate, Magnesium, Phosphate, Potassium,
Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC,
Fibrinogen, Platelets, Age, Gender, Unit1, Unit2,
HospAdmTime, ICULOS
```

Requirements:

- Rows must be in chronological order.
- Missing measurements may be represented as `NaN` or omitted columns.
- Include as much patient history as is available up to the prediction hour.
- `ICULOS` should contain the ICU hour. If absent, sequential hours are generated.
- `SepsisLabel` is not required and is ignored during inference.

The feature builder is causal: each prediction uses only the current and previous rows supplied for that patient.

## Target Meaning

The returned probability estimates the PhysioNet Challenge hourly `SepsisLabel`. For septic patients, this label begins six hours before the defined clinical onset and remains positive afterward. Before onset, the output functions as a next-six-hour warning probability. It is not a pure incident-onset probability after the patient has already become septic.

## Validation Summary

The final model uses 310 literature-core features and sigmoid calibration learned from nested out-of-fold predictions.

| Metric | Nested out-of-fold result |
| --- | ---: |
| AUROC | 0.8491 |
| AUPRC | 0.1262 |
| Precision | 0.0714 |
| Recall | 0.6732 |
| F1 | 0.1290 |
| Lift@10% | 5.3976x |
| Utility | 0.4262 |

Five-fold Utility SD was `0.0139`, with a fold-based 95% interval of `[0.4089, 0.4435]`. This evaluates public hospital systems A and B; external-hospital performance remains unproven.

Probability calibration reduced cross-fitted Brier score from `0.0442` to `0.0166`. Mean calibrated probability was `1.793%`, compared with an observed positive-hour rate of `1.799%`.

Detailed reports:

- `output/sepsis_next_6h_predictor_report.md`
- `output/nested_cv_5fold_report.md`
- `models/sepsis_next_6h_predictor_metadata.json`

## Data Source

The model was developed from the public training data for the PhysioNet/Computing in Cardiology Challenge 2019:

Reyna MA, et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, version 1.0.0. DOI: [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857).

No raw patient records are stored in this repository.

## Rebuild

Raw PhysioNet patient data and the generated feature cache are intentionally excluded from Git. With the Challenge 2019 training data available locally, the nested evaluation and deployment artifact can be rebuilt using:

```bash
python run_nested_cv.py
python build_deployable_pickle.py
```

Install the additional training dependency first with `pip install -r requirements-training.txt`.

The default data path expected by the training scripts is:

```text
../data/challenge-2019/training/
```
