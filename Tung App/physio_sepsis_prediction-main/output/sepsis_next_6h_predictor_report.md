# Deployable Six-Hour Sepsis Predictor

## Artifact

`models/sepsis_next_6h_predictor.pkl`

The pickle contains one `SepsisRiskPredictor` object. It accepts a patient DataFrame, a PSV/CSV path, a list of hourly records, or a dictionary of columns.

## Probability Calibration

Calibration was learned from patient-level nested out-of-fold scores. The calibration evaluation below is cross-fitted: each fold was calibrated without using that fold's labels.

| Score | AUROC | AUPRC | Brier | Log loss | Mean probability | Positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw XGBoost | 0.8491 | 0.1262 | 0.0442 | 0.1747 | 0.1303 | 0.0180 |
| Calibrated | 0.8490 | 0.1260 | 0.0166 | 0.0734 | 0.0179 | 0.0180 |

## Deployment Policy

- Global calibrated threshold: `0.023`
- ICU phase bounds: `[12, 24, 48]`
- Calibrated phase thresholds: `[0.025, 0.023, 0.028, 0.023]`
- Nested-CV Utility reference: `0.4262` with 95% fold interval `[0.4089, 0.4435]`

The deployment thresholds were tuned on pooled cross-fitted out-of-fold probabilities after the nested evaluation. They are operational settings, not a new unbiased performance estimate.

## Python Usage

```python
import pickle

with open("sepsis_next_6h_predictor.pkl", "rb") as handle:
    predictor = pickle.load(handle)

result = predictor.predict_patient(patient_dataframe)
probability = predictor.predict_proba(patient_dataframe)
trajectory = predictor.predict_trajectory(patient_dataframe)
```

## Target Definition

The probability estimates the PhysioNet Challenge hourly sepsis target, whose label begins six hours before clinical sepsis onset and remains positive afterward. Before onset, this functions as a next-six-hour warning probability; it is not a pure incident-onset label after the patient has already become septic.
