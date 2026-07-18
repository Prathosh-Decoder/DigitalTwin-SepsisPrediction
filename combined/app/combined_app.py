"""Unified ICU digital-twin API.

The LightGBM model supplies a current-hour active alert. Tung's calibrated
XGBoost model runs in a separate sidecar and supplies a next-six-hour forecast
plus local SHAP contributions. Keeping the libraries in separate processes
avoids the OpenMP crash documented in this repository.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
from joblib import load

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "sepsis_pipeline"
STATIC = Path(__file__).resolve().parent / "static"
TUNG_URL = os.environ.get("TUNG_URL", "http://127.0.0.1:8711")


def data_dir(env_name, fallback):
    configured = os.environ.get(env_name)
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([ROOT / fallback / fallback, ROOT / fallback])
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_A = data_dir("SEPSIS_DATA_A", "training_setA")
DATA_B = data_dir("SEPSIS_DATA_B", "training_setB")

sys.path.insert(0, str(PIPE))
import config  # noqa: E402
from criticality import (  # noqa: E402
    calibrated_probability,
    criticality_score,
    criticality_trend,
    tier_from_score,
)
from feature_engineering import build_patient_features  # noqa: E402
from clinical_narrative import build_narrative  # noqa: E402

bundle = load(PIPE / "artifacts" / "model_bundle.joblib")
ACTIVE_MODEL = bundle["model"]
ACTIVE_FEATURES = bundle["feature_names"]
ACTIVE_THRESHOLD = float(bundle["threshold"])
CRITICALITY = load(PIPE / "artifacts" / "criticality_calibrator.joblib")
ACTIVE_REFERENCE = np.asarray(CRITICALITY["reference_quantiles"], dtype=float)

# A compact, reproducible ICU cohort chosen by the original integration. The
# evaluation categories are intentionally not returned by operational APIs.
DEMO_PATIENTS = [49, 74, 7, 21, 714, 851, 171, 178]
TIER_BANDS = [("LOW", 0, 50), ("MODERATE", 50, 75), ("HIGH", 75, 90), ("CRITICAL", 90, 101)]
VITALS = {
    "HR": ("Heart rate", "bpm", 60, 100),
    "MAP": ("MAP", "mmHg", 65, 110),
    "O2Sat": ("SpO2", "%", 94, 100),
    "Resp": ("Respiration", "/min", 12, 20),
    "Temp": ("Temperature", "C", 36.0, 38.0),
    "Lactate": ("Lactate", "mmol/L", 0.5, 2.0),
}

app = Flask(__name__, static_url_path="", static_folder=str(STATIC))


def clean(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    return value


def J(value, status=200):
    return jsonify(clean(value)), status


def psv_path(patient_id):
    filename = f"p{int(patient_id):06d}.psv"
    path_a = DATA_A / filename
    return path_a if path_a.exists() else DATA_B / filename


def load_patient(patient_id, hour=None):
    path = psv_path(patient_id)
    if not path.exists():
        raise FileNotFoundError(path)
    full = pd.read_csv(path, sep="|")
    requested = len(full) if hour is None else max(1, int(hour))
    frame = full.iloc[: min(requested, len(full))].copy()
    return frame, requested > len(full), len(full)


def engineer(frame, patient_id):
    frame = frame.copy()
    frame["patient_id"] = patient_id
    frame["hospital"] = "LIVE"
    if "SepsisLabel" not in frame:
        frame["SepsisLabel"] = 0
    columns = config.VITAL_COLS + config.LAB_COLS + config.STATIC_COLS + [
        "patient_id", "hospital", "SepsisLabel"
    ]
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return build_patient_features(frame.reindex(columns=columns))


def active_score(patient_id, hour):
    frame, discharged, max_hour = load_patient(patient_id, hour)
    features = engineer(frame, patient_id)
    raw = np.asarray(ACTIVE_MODEL.predict(features[ACTIVE_FEATURES]), dtype=float)
    probability = calibrated_probability(CRITICALITY["calibrator"], raw)
    criticality = criticality_score(raw, ACTIVE_REFERENCE)
    return {
        "model": "Prathosh LightGBM",
        "role": "current active alert",
        "probability": float(probability[-1]),
        "raw_score": float(raw[-1]),
        "threshold": ACTIVE_THRESHOLD,
        "alert": bool(raw[-1] >= ACTIVE_THRESHOLD),
        "criticality": float(criticality[-1]),
        "tier": tier_from_score(criticality[-1], TIER_BANDS),
        "trend": criticality_trend(criticality),
        "is_discharged": discharged,
        "max_hour": max_hour,
    }


def tung_get(path, **params):
    try:
        response = requests.get(f"{TUNG_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def forecast_score(patient_id, hour, drivers=False):
    score = tung_get(
        f"/score/{patient_id}",
        hour=hour if hour is not None else "",
        drivers="1" if drivers else "0",
    )
    if not score:
        return {
            "model": "Tung calibrated XGBoost",
            "role": "sepsis risk in the next 6 hours",
            "available": False,
            "probability": None,
            "threshold": None,
            "alert": False,
            "trend": "unavailable",
            "drivers": [],
        }
    trajectory = (
        tung_get(f"/trajectory/{patient_id}")
        if drivers
        else {"iculos": [], "tung_prob": score.get("recent_probabilities", []), "thresholds": []}
    ) or {"iculos": [], "tung_prob": [], "thresholds": []}
    limit = min(int(hour or len(trajectory["tung_prob"])), len(trajectory["tung_prob"]))
    values = trajectory["tung_prob"][:limit]
    trend = "steady"
    if len(values) >= 2:
        baseline = values[max(0, len(values) - 4)]
        delta = values[-1] - baseline
        trend = "rising" if delta > 0.02 else "falling" if delta < -0.02 else "steady"
    return {
        "model": "Tung calibrated XGBoost",
        "role": "sepsis risk in the next 6 hours",
        "available": True,
        "probability": float(score["tung_prob"]),
        "raw_score": float(score["tung_raw"]),
        "threshold": float(score["threshold"]),
        "alert": bool(score["alarm"]),
        "trend": trend,
        "drivers": score.get("drivers", []),
        "trajectory": {
            "hours": trajectory["iculos"][:limit],
            "probabilities": values,
            "thresholds": trajectory.get("thresholds", [])[:limit],
        },
    }


def latest_measurements(frame):
    measurements = []
    for column, (label, unit, low, high) in VITALS.items():
        if column not in frame:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        value = float(series.iloc[-1]) if not series.empty else None
        previous = float(series.iloc[-2]) if len(series) > 1 else None
        delta = value - previous if value is not None and previous is not None else None
        abnormal = value is not None and not (low <= value <= high)
        measurements.append({
            "key": column,
            "label": label,
            "value": value,
            "unit": unit,
            "delta": delta,
            "abnormal": abnormal,
        })
    return measurements


def patient_state(active, forecast):
    if active["alert"]:
        return "ACTIVE"
    if forecast["alert"]:
        return "FORECAST"
    forecast_watch = (
        forecast["available"]
        and forecast["probability"] >= max(0.05, forecast["threshold"] * 0.5)
    )
    if active["criticality"] >= 75 or forecast_watch:
        return "WATCH"
    return "STABLE"


def patient_payload(patient_id, hour, detail=False):
    frame, discharged, max_hour = load_patient(patient_id, hour)
    active = active_score(patient_id, hour)
    forecast = forecast_score(patient_id, hour, drivers=detail)
    state = patient_state(active, forecast)
    payload = {
        "id": patient_id,
        "bed": f"ICU-{DEMO_PATIENTS.index(patient_id) + 1:02d}",
        "hour": int(frame["ICULOS"].iloc[-1]) if "ICULOS" in frame else len(frame),
        "max_hour": max_hour,
        "is_discharged": discharged,
        "state": state,
        "active_alert": active,
        "forecast": forecast,
        "vitals": latest_measurements(frame),
    }
    if detail:
        payload["narrative"] = build_narrative(payload)
    return payload


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/health")
def health():
    sidecar = tung_get("/health")
    return J({
        "status": "ok" if sidecar else "degraded",
        "active_alert_model": "ready",
        "forecast_model": "ready" if sidecar else "unavailable",
        "narrative": "openai" if os.environ.get("OPENAI_API_KEY") else "rules",
    })


@app.route("/api/twin/beds")
def beds():
    hour = request.args.get("hour", default=24, type=int)
    patients = [patient_payload(patient_id, hour) for patient_id in DEMO_PATIENTS]
    rank = {"ACTIVE": 0, "FORECAST": 1, "WATCH": 2, "STABLE": 3}
    patients.sort(key=lambda patient: (rank[patient["state"]], -patient["active_alert"]["criticality"]))
    counts = {state: sum(patient["state"] == state for patient in patients) for state in rank}
    return J({"hour": hour, "counts": counts, "patients": patients})


@app.route("/api/twin/beds/<int:patient_id>")
def bed(patient_id):
    if patient_id not in DEMO_PATIENTS:
        return J({"error": "unknown demo patient"}, 404)
    hour = request.args.get("hour", default=24, type=int)
    return J(patient_payload(patient_id, hour, detail=True))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8710"))
    print(f"ICU digital twin listening on http://127.0.0.1:{port}", flush=True)
    print(f"Data: {DATA_A} | {DATA_B}; forecast sidecar: {TUNG_URL}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
