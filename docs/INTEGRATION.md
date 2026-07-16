# Integration Guide — Using the Model in the Digital Twin

This tells you **where the files are, how to load them, what values come out, and how you might
display each one**. It does not assume anything about how your twin is built — however your twin
runs or calls Python code is up to you. For what each value *means*, see §6 of
[`REPORT.md`](REPORT.md).

---

## 1. The two files you need

Both are in `sepsis_pipeline/artifacts/`:

| File | What's inside |
|---|---|
| `model_bundle.joblib` | `{ "model", "feature_names", "threshold" }` — the trained model, the 214 feature-column names it expects (in order), and the risk cutoff |
| `criticality_calibrator.joblib` | `{ "calibrator", "reference_quantiles", "tier_bands", "feature_plain_names" }` — turns the model's raw score into the display values |

They are **Python (joblib) files** — they can only be loaded from Python. Everything below is Python.

---

## 2. Load everything once

```python
from joblib import load

bundle = load("sepsis_pipeline/artifacts/model_bundle.joblib")
model         = bundle["model"]
feature_names = bundle["feature_names"]
threshold     = bundle["threshold"]

crit = load("sepsis_pipeline/artifacts/criticality_calibrator.joblib")
```

---

## 3. Get the values for one bed

Give it that bed's **recent hourly rows** (the last ~24 hours, oldest → newest), each row holding
the raw vitals/labs (§5 lists the columns; use `null`/`NaN` for anything not measured). It returns
everything you need to display.

```python
import pandas as pd
import shap
import config
from feature_engineering import build_patient_features
from criticality import (
    calibrated_probability, criticality_score, tier_from_score,
    criticality_trend, top_shap_drivers,
)

explainer = shap.TreeExplainer(model)   # build once, reuse

def score_bed(recent_hours):
    """recent_hours: list of dicts, one per hour (oldest -> newest)."""
    df = pd.DataFrame(recent_hours)
    df["hospital"] = "LIVE"; df["SepsisLabel"] = 0        # required columns, not used as inputs
    df = df.reindex(columns=config.VITAL_COLS + config.LAB_COLS
                    + config.STATIC_COLS + ["patient_id", "hospital", "SepsisLabel"])

    eng = build_patient_features(df)                      # -> 214 features per hour
    X = eng[feature_names]
    scores = model.predict(X)                             # raw score per hour

    crit_series = criticality_score(scores, crit["reference_quantiles"])
    raw = float(scores[-1])                               # the current (latest) hour
    shap_row = explainer.shap_values(X.iloc[[-1]])[0]

    return {
        "criticality": round(float(crit_series[-1]), 1),                        # 0-100
        "probability": round(float(calibrated_probability(crit["calibrator"], [raw])[0]), 3),  # 0-1
        "tier":        tier_from_score(crit_series[-1]),                        # LOW/MODERATE/HIGH/CRITICAL
        "trend":       criticality_trend(crit_series),                         # rising/steady/falling
        "is_risky":    bool(raw >= threshold),                                 # True/False
        "drivers":     [f"{d['plain_name']} {d['direction']}"                  # e.g. "lactate ↑"
                        for d in top_shap_drivers(shap_row, X.iloc[-1].to_numpy(), feature_names)],
    }
```

---

## 4. The values you get back, and how you could show them

| Value | Example | How you might display it |
|---|---|---|
| `criticality` | `94.0` | The **headline number** on the bed (0–100). Sort/prioritize beds by it. |
| `tier` | `"CRITICAL"` | The bed's **colour / label** — e.g. green (LOW) → amber (HIGH) → red (CRITICAL). |
| `probability` | `0.152` | A small secondary line, shown as a percent (`15%`). |
| `trend` | `"rising"` | An **arrow** next to the number (↑ rising, → steady, ↓ falling). |
| `is_risky` | `true` | A simple on/off **alert dot/badge** if you want a binary flag. |
| `drivers` | `["lactate ↑", "MAP falling ↓", "heart rate ↑"]` | A short **"why" line** under the bed. |

These are just suggestions — display them however fits your twin. Use `criticality` as the main
number and `tier` for colour; `probability`, `trend`, and `drivers` are supporting detail.

---

## 5. What each raw hourly row must contain (input)

One row per hour. Missing measurements → `null`/`NaN`, never a fabricated `0`. Names/units match
the PhysioNet Challenge 2019 data the model was trained on.

- **Timing:** `ICULOS` (hours since ICU admission, starts at 1)
- **Demographics:** `Age`, `Gender` (0/1), `Unit1`, `Unit2` (0/1/null), `HospAdmTime`
- **8 vitals:** `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2`
- **26 labs:** `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`,
  `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`,
  `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`,
  `Fibrinogen`, `Platelets`

Send the **last ~24 hours**, not a single row — some features look back over 6–24 hours.

---

## 6. What the values mean

Full explanation of criticality vs. probability, the tiers, the trend, and the important caveats
(criticality is a *rank*, not a chance; not a clinical device) is in
[`REPORT.md`](REPORT.md#6-criticality--prioritization-layer) **§6. Criticality & Prioritization
Layer**. The model itself — features, training, results — is in §1–§5 of the same report.
