# Integration Guide — Wiring the Sepsis Model into the Digital Twin

This is a step-by-step guide for connecting the trained sepsis model to the Node.js/Express/Socket.IO
ICU digital twin, so each virtual bed shows a live risk read-out (flag + 0–100 criticality +
probability + tier + trend + plain-language reasons). It gives you a **complete, runnable Python
inference service** and the **Node-side code to call it** — not just isolated snippets.

Read top to bottom: §1 the shape of the integration, §2 the data flow, §3–§4 the two pieces of code
you add, §5 what comes back, §6–§8 reference + caveats.

---

## 1. The big picture (and why a service)

The digital twin backend is **Node.js**. The model is **Python** (a LightGBM object plus a Python
feature-engineering step and a Python criticality layer). Node cannot load or run these directly.

The standard, clean way to bridge them is a small **Python inference service** that the twin calls
over HTTP:

```
┌────────────────────────┐        HTTP POST /predict         ┌──────────────────────────────┐
│  Digital Twin (Node.js) │  ──────  {bed history} ─────────▶ │  Python inference service     │
│  Express + Socket.IO    │                                   │  (FastAPI, this guide §3)     │
│  - replays bed data     │ ◀──── {risk, criticality, ... } ──│  - build_patient_features()   │
│  - shows bed cards      │        JSON response              │  - model.predict()            │
└────────────────────────┘                                   │  - criticality layer + SHAP   │
                                                              └──────────────────────────────┘
```

The service owns **everything Python** (features, model, calibration, SHAP). The twin stays pure
Node and just sends each bed's recent hours and displays what comes back. This keeps a single
source of truth and means the model never has to be reimplemented in JavaScript.

---

## 2. The end-to-end flow (what happens each hour, per bed)

1. The twin already holds each bed's hourly readings (it replays them). Keep a **rolling window of
   the last ~24+ hours** per bed (the model's features look back up to 24 h).
2. On each tick, the twin sends that bed's recent hours to the service: `POST /predict` with a
   `history` array (one object per hour, oldest → newest).
3. The service turns the raw hours into the 214 model features, scores the most recent hour,
   converts the score into criticality / probability / tier / trend, and computes the top SHAP
   reasons.
4. The service returns one JSON object (see §5).
5. The twin broadcasts it (e.g. over Socket.IO) and the bed card renders it.

The service is **stateless** — the twin sends the history every call, so nothing needs to be
remembered between requests. (Trend is derived inside the service from the recent hours it
receives.)

---

## 3. Part A — the Python inference service (complete, runnable)

Save as `sepsis_pipeline/serve.py` and run it from inside `sepsis_pipeline/` (so the
`feature_engineering` / `criticality` imports resolve). Requires `pip install fastapi uvicorn`
in addition to the project requirements. It loads everything **once** at startup.

```python
# sepsis_pipeline/serve.py  —  run:  uvicorn serve:app --port 8000
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import shap
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel

import config
from feature_engineering import build_patient_features
from criticality import (
    calibrated_probability, criticality_score, tier_from_score,
    criticality_trend, top_shap_drivers,
)

# --- load everything once ---
bundle = load(config.MODEL_BUNDLE_PATH)
model, feature_names, threshold = bundle["model"], bundle["feature_names"], bundle["threshold"]
crit = load(config.ARTIFACTS_DIR / "criticality_calibrator.joblib")
explainer = shap.TreeExplainer(model)          # built from the model (no cross-version pickle issues)

# the raw columns build_patient_features expects (see §6)
REQUIRED_COLS = (config.VITAL_COLS + config.LAB_COLS + config.STATIC_COLS
                 + ["patient_id", "hospital", "SepsisLabel"])

app = FastAPI()

class PredictRequest(BaseModel):
    history: list[dict]        # one dict per hour, oldest -> newest (raw schema columns; NaN/null OK)

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.history)
    df["hospital"] = "LIVE"          # required by the feature fn, not a model input
    df["SepsisLabel"] = 0            # stub; never used as input
    df = df.reindex(columns=REQUIRED_COLS)     # ensure all columns exist (missing -> NaN)

    eng = build_patient_features(df)           # -> 214 features, one row per input hour
    X = eng[feature_names]
    scores = model.predict(X)                  # raw utility-gain scores (NOT probabilities)

    crit_series = criticality_score(scores, crit["reference_quantiles"])   # 0..100 per hour
    raw_score = float(scores[-1])                                          # the current hour
    criticality = float(crit_series[-1])
    probability = float(calibrated_probability(crit["calibrator"], [raw_score])[0])

    shap_row = explainer.shap_values(X.iloc[[-1]])[0]
    drivers = top_shap_drivers(shap_row, X.iloc[-1].to_numpy(), feature_names)

    return {
        "is_risky":    bool(raw_score >= threshold),
        "criticality": round(criticality, 1),          # 0-100 headline (relative risk rank)
        "probability": round(probability, 4),          # calibrated P(pre-sepsis), 0..1
        "tier":        tier_from_score(criticality),   # LOW / MODERATE / HIGH / CRITICAL
        "trend":       criticality_trend(crit_series), # rising / steady / falling
        "drivers":     [{"reason": d["plain_name"], "direction": d["direction"]} for d in drivers],
        "raw_score":   round(raw_score, 4),
        "threshold":   float(threshold),
    }

@app.get("/health")
def health():
    return {"ok": True}
```

The core of this (`build_patient_features → model.predict → criticality → SHAP`) is the exact
verified flow used in `07_criticality.py`; the service just wraps it in an HTTP endpoint.

---

## 4. Part B — calling it from the Node.js twin

In the twin's per-bed update logic, keep a rolling history and POST it to the service, then
broadcast the result. (Node 18+ has global `fetch`.)

```js
// keep, per bed, an array of the last ~24+ hourly readings (raw fields: HR, O2Sat, Temp, ...)
async function scoreBed(bed) {
  const res = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ history: bed.recentHours }),   // oldest -> newest
  });
  if (!res.ok) return;                    // fail soft: keep last shown value, don't block the tick
  const triage = await res.json();

  bed.triage = triage;                    // { is_risky, criticality, probability, tier, trend, drivers }
  io.to(`bed:${bed.id}`).emit("bed-triage", { bedId: bed.id, ...triage });
}
```

Notes for a robust integration:
- **Call it on the twin's existing tick**, not more often than data changes.
- **Fail soft**: if the service is down or slow, keep the last shown value; never block the tick or
  the live vitals stream on the model call.
- The service is stateless and cheap to scale — run one instance behind the same reverse proxy as
  the twin, or several if you have many beds.

---

## 5. What comes back (the response contract)

| Field | Type | Meaning / how the bed card should use it |
|---|---|---|
| `is_risky` | bool | Binary flag (raw score ≥ the model's chosen threshold). Simple on/off alert. |
| `criticality` | 0–100 | **Headline triage number** — a *relative risk rank* (94 = riskier than 94 % of ICU-hours), **not** a probability. Use it to order/prioritize beds. |
| `probability` | 0–1 | Calibrated likelihood ("of hours like this, X % were pre-sepsis"). Small numbers are normal for a rare event — show as a secondary detail, not the headline. |
| `tier` | string | `LOW / MODERATE / HIGH / CRITICAL` → drive the bed's colour. |
| `trend` | string | `rising / steady / falling` → an arrow; a rising bed is more urgent than a flat-high one. |
| `drivers` | list | Top-3 reasons in plain language (`{reason, direction}`), e.g. "lactate ↑". This is the "why" that makes a flag trustworthy. |
| `raw_score`, `threshold` | float | The underlying model number and cutoff, for debugging/audit. |

Read the criticality section of [`REPORT.md`](REPORT.md#6-criticality--prioritization-layer) (§6)
before displaying any of these to a user — especially that criticality is a rank, not a
probability, and the time-in-ICU caveat.

---

## 6. Reference — the raw input schema (what each `history` row must contain)

One object per hour. Missing measurements should be `null`/`NaN`, **never** a fabricated `0`. Names
and units match the PhysioNet Challenge 2019 format the model was trained on (`Temp` °C, pressures
mmHg). `patient_id`, `hospital`, and `SepsisLabel` are added/stubbed by the service — the twin only
needs to send the clinical columns below (any it doesn't have, omit or set null).

| Group | Columns |
|---|---|
| ICU timing | `ICULOS` (hours since ICU admission, starts at 1, +1/hour) |
| Demographics (static) | `Age`, `Gender` (0/1), `Unit1`, `Unit2` (0/1/null), `HospAdmTime` |
| 8 vitals | `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2` |
| 26 labs | `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets` |

**Send the full recent window, not a single row.** Features include rolling stats and deltas over
6–24 h; a one-row payload won't error but produces degraded features. Aim for the last ~24 h (more
is fine — the whole stay works too).

---

## 7. Reference — artifacts and retraining

Artifacts in `sepsis_pipeline/artifacts/` the service uses:

| File | Role |
|---|---|
| `model_bundle.joblib` | `{model, feature_names, threshold}` — the frozen model |
| `criticality_calibrator.joblib` | `{calibrator, reference_quantiles, tier_bands, feature_plain_names}` — the criticality layer (separate file; the model is untouched) |
| `metrics.json`, `plots/` | performance numbers + curves (context, not needed at serve time) |

To reproduce/retrain from scratch (from `sepsis_pipeline/`):

```bash
pip3 install -r requirements.txt
python3 01_build_dataset.py        # parse training_setA/ + training_setB/
python3 02_feature_engineering.py  # build the 214-feature matrix
python3 03_train_model.py          # train + save model_bundle.joblib
python3 04_explain_shap.py         # SHAP artifacts
python3 05_sanity_checks.py        # leakage/causality checks
python3 07_criticality.py          # fit + save the criticality layer
```

---

## 8. Caveats before treating this as production-ready

- **Not a clinical device.** Retrospective research data; no care decision should depend on it.
- **Test performance is likely optimistic** — the test set is a held-out slice of the *same two*
  hospitals, not a genuinely unseen one (the original challenge's top scores collapsed on an unseen
  hospital).
- **Precision is low by design** (~8 % at the current threshold) — most flags are false alarms. The
  model is for *prioritizing attention*, not diagnosis. Show it as decision support, and lean on
  `criticality` (ranking) + `trend` + `drivers` rather than the raw flag alone.

Full model detail — features, the utility target, results — is in [`REPORT.md`](REPORT.md); the
triage layer's math and caveats are §6 of the same report.
