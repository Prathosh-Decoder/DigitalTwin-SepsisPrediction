"""Combined sepsis dashboard (Python 3.10).

Serves four tabs over one API:
  - My model   : your LightGBM model + SHAP + criticality (computed locally)
  - Tung model : Tung's XGBoost model (fetched from the sidecar on :8711)
  - Combined   : weighted blend of the two calibrated probabilities
  - Compare    : one patient, both models' per-hour trajectory vs ground truth

This process loads LightGBM + SHAP only. It never imports XGBoost (that lives in the
sidecar) -- LightGBM + XGBoost in one process segfault (OpenMP). Tung data comes over HTTP.

Port 8710 (5000 = macOS AirPlay; 5001 may be the original app).
"""
import json
import math
import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shap
from flask import Flask, jsonify, request
from joblib import load

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "sepsis_pipeline"
DATA_A = ROOT / "training_setA" / "training_setA"
DATA_B = ROOT / "training_setB" / "training_setB"
STATIC = Path(__file__).resolve().parent / "static"
RESULTS = ROOT / "combined" / "eval" / "results"
TUNG_URL = os.environ.get("TUNG_URL", "http://127.0.0.1:8711")

sys.path.insert(0, str(PIPE))
import config  # noqa: E402
from feature_engineering import build_patient_features  # noqa: E402
from criticality import (calibrated_probability, criticality_score,  # noqa: E402
                         criticality_trend, tier_from_score, top_shap_drivers)

# --- artifacts ---
bundle = load(PIPE / "artifacts" / "model_bundle.joblib")
MODEL = bundle["model"]
FEATURES = bundle["feature_names"]
USER_THR = bundle["threshold"]
CRIT = load(PIPE / "artifacts" / "criticality_calibrator.joblib")
EXPLAINER = shap.TreeExplainer(MODEL)
USER_REF = np.asarray(CRIT["reference_quantiles"], dtype=float)

CFG = json.loads((RESULTS / "ensemble_config.json").read_text())
W = float(CFG["weight_user"])                 # weight on YOUR model in the blend
W_TUNED = float(CFG["weight_user_tuned"])
ENS_THR = float(CFG["ensemble_threshold"])
TUNG_THR = float(CFG["tung_threshold"])
TIER_BANDS = [(b[0], b[1], b[2]) for b in CFG["tier_bands"]]
TUNG_REF = np.asarray(CFG["tung_reference_quantiles"], dtype=float)
ENS_REF = np.asarray(CFG["ensemble_reference_quantiles"], dtype=float)

DEMO = {49: "TN", 74: "TN", 7: "FP", 21: "FP", 714: "FN", 851: "FN", 171: "TP", 178: "TP"}
TEST_IDS = json.loads((PIPE / "artifacts" / "test_patient_ids.json").read_text())

app = Flask(__name__, static_url_path="", static_folder=str(STATIC))


def clean(o):
    """Recursively replace NaN/inf floats with None so responses are valid JSON.
    (json.dumps emits the bareword NaN, which the browser's JSON.parse rejects.)"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def J(obj):
    return jsonify(clean(obj))


# ------------------------------------------------------------------ data helpers
def psv_path(pid: int) -> Path:
    fname = f"p{int(pid):06d}.psv"
    a = DATA_A / fname
    return a if a.exists() else DATA_B / fname


def load_patient(pid, up_to_hour=None):
    df = pd.read_csv(psv_path(pid), sep="|")
    onset = None
    if "SepsisLabel" in df and (df["SepsisLabel"] == 1).any():
        onset = int(df.loc[df["SepsisLabel"] == 1, "ICULOS"].iloc[0]) + 6
    is_discharged = False
    full_len = len(df)
    if up_to_hour is not None:
        is_discharged = up_to_hour > full_len
        df = df.iloc[:up_to_hour]
    return df, is_discharged, onset, full_len


def _engineer(df, pid):
    df = df.copy()
    df["patient_id"] = pid
    df["hospital"] = "LIVE"
    if "SepsisLabel" not in df:
        df["SepsisLabel"] = 0
    cols = config.VITAL_COLS + config.LAB_COLS + config.STATIC_COLS + ["patient_id", "hospital", "SepsisLabel"]
    for c in cols:
        if c not in df:
            df[c] = float("nan")
    return build_patient_features(df.reindex(columns=cols))


def tier_of(value, ref):
    c = float(criticality_score([value], ref)[0])
    return c, tier_from_score(c, TIER_BANDS)


def categorize_drivers(drivers):
    """Split a flat driver list into the three display buckets (mirrors app/app.py)."""
    cats = {"vitals_labs": [], "demographics": [], "others": []}
    for d in drivers:
        name = d["label"].lower()
        if any(k in name for k in ("age", "gender", "icu unit", "unit1", "unit2")):
            cats["demographics"].append(d)
        elif any(k in name for k in ("hours in", "time since", "hours since", "hrs since",
                                     "iculos", "icu hour", "lag hours", "admission")):
            cats["others"].append(d)
        else:
            cats["vitals_labs"].append(d)
    return cats


# ------------------------------------------------------------------ per-model scoring
def user_score(pid, hour, want_drivers=False):
    """Local: your LightGBM model, criticality series (for trend), SHAP drivers."""
    df, is_disch, onset, _ = load_patient(pid, hour)
    if len(df) == 0:
        return None
    eng = _engineer(df, pid)
    X = eng[FEATURES]
    raw = MODEL.predict(X)
    prob = calibrated_probability(CRIT["calibrator"], raw)     # fraction 0..1
    crit_series = criticality_score(raw, USER_REF)
    drivers = []
    if want_drivers:
        srow = EXPLAINER.shap_values(X.iloc[[-1]])[0]
        drivers = [{"label": d["plain_name"], "value": round(d["value"], 2), "direction": d["direction"]}
                   for d in top_shap_drivers(srow, X.iloc[-1].to_numpy(), FEATURES, k=5)]
    return {
        "prob_frac": float(prob[-1]),
        "criticality": round(float(crit_series[-1]), 1),
        "tier": tier_from_score(crit_series[-1], TIER_BANDS),
        "trend": criticality_trend(crit_series),
        "is_risky": bool(float(raw[-1]) >= USER_THR),
        "drivers": drivers,
        "is_discharged": is_disch,
        "true_onset_hour": onset,
        "sepsis_now": bool(int(df["SepsisLabel"].iloc[-1]) == 1) if "SepsisLabel" in df else False,
        "iculos": eng["ICULOS"].to_numpy().tolist(),
        "prob_frac_series": [float(p) for p in prob],
        "alarm_series": [bool(float(r) >= USER_THR) for r in raw],
    }


def tung_get(path, **params):
    try:
        r = requests.get(f"{TUNG_URL}{path}", params=params, timeout=15)
        return r.json()
    except Exception:
        return None


def tung_trend(pid, hour):
    """Tung criticality series up to `hour`, for a trend arrow. Returns (iculos, prob_frac_series)."""
    tj = tung_get(f"/trajectory/{pid}")
    if not tj:
        return None, None
    icu = np.asarray(tj["iculos"], dtype=float)
    prob = np.asarray(tj["tung_prob"], dtype=float)
    if hour is not None:
        prob = prob[:hour]
        icu = icu[:hour]
    return icu, prob


def assemble(pid, hour, model, want_drivers=False):
    """Return the display dict for `model` in {user, tung, ensemble}."""
    us = user_score(pid, hour, want_drivers=want_drivers)
    if us is None:
        return {"id": pid, "category": DEMO.get(pid), "is_discharged": True, "empty": True}
    base = {"id": pid, "category": DEMO.get(pid), "is_discharged": us["is_discharged"],
            "true_onset_hour": us["true_onset_hour"], "sepsis_now": us["sepsis_now"]}

    if model == "user":
        base.update({"probability": round(us["prob_frac"] * 100, 1), "criticality": us["criticality"],
                     "tier": us["tier"], "trend": us["trend"], "is_risky": us["is_risky"],
                     "drivers": categorize_drivers(us["drivers"])})
        return base

    # tung / ensemble need Tung's numbers
    ts = tung_get(f"/score/{pid}", hour=hour if hour else "", drivers="1" if want_drivers else "0")
    if not ts:
        base.update({"probability": None, "criticality": None, "tier": "N/A", "trend": "steady",
                     "is_risky": False, "drivers": categorize_drivers([]), "tung_unavailable": True})
        return base
    tung_prob = float(ts["tung_prob"])

    if model == "tung":
        c, tier = tier_of(tung_prob, TUNG_REF)
        icu, series = tung_trend(pid, hour)
        trend = criticality_trend(criticality_score(series, TUNG_REF)) if series is not None and len(series) else "steady"
        base.update({"probability": round(tung_prob * 100, 1), "criticality": round(c, 1),
                     "tier": tier, "trend": trend, "is_risky": bool(ts["alarm"]),
                     "drivers": categorize_drivers(ts.get("drivers", []))})
        return base

    # ensemble: blend fractions; criticality vs ensemble reference; drivers = union
    ens = W * us["prob_frac"] + (1 - W) * tung_prob
    c, tier = tier_of(ens, ENS_REF)
    # trend from per-hour ensemble criticality (align user & tung series by ICULOS)
    trend = "steady"
    icu_t, series_t = tung_trend(pid, hour)
    if series_t is not None and len(series_t):
        um = dict(zip(us["iculos"], us["prob_frac_series"]))
        tm = dict(zip(icu_t.tolist(), series_t.tolist()))
        common = [h for h in um if h in tm]
        if common:
            ens_series = np.array([W * um[h] + (1 - W) * tm[h] for h in common])
            trend = criticality_trend(criticality_score(ens_series, ENS_REF))
    drivers = ([{**d, "source": "You"} for d in us["drivers"][:3]] +
               [{**d, "source": "Tung"} for d in ts.get("drivers", [])[:3]])
    base.update({"probability": round(ens * 100, 1), "criticality": round(c, 1), "tier": tier,
                 "trend": trend, "is_risky": bool(ens >= ENS_THR),
                 "drivers": categorize_drivers(drivers)})
    return base


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/api/config")
def api_config():
    return J({
        "weight_user": W, "weight_user_tuned": W_TUNED, "ensemble_threshold": ENS_THR,
        "tung_threshold": TUNG_THR, "user_raw_threshold": USER_THR,
        "categories": DEMO,
        "disclaimer": "Tung & Combined figures are in-sample: Tung trained on these patients. "
                      "Only your model's numbers are held-out. See eval/results/comparison.md.",
    })


@app.route("/api/beds")
def api_beds():
    model = request.args.get("model", "user")
    hour = request.args.get("hour", type=int)
    out = []
    for pid, cat in DEMO.items():
        d = assemble(pid, hour, model, want_drivers=False)
        out.append({k: d.get(k) for k in ("id", "category", "criticality", "tier", "trend",
                                          "is_risky", "is_discharged", "probability",
                                          "sepsis_now", "tung_unavailable")})
    return J(out)


@app.route("/api/beds/<int:pid>")
def api_bed(pid):
    if pid not in DEMO:
        return jsonify({"error": "unknown patient"}), 404
    model = request.args.get("model", "user")
    hour = request.args.get("hour", type=int)
    return J(assemble(pid, hour, model, want_drivers=True))


def compare_payload(pid, hour):
    us = user_score(pid, None)                 # full user trajectory (raw/prob/alarm per hour)
    tj = tung_get(f"/trajectory/{pid}")
    df_full, _, onset, _ = load_patient(pid, None)
    labels = df_full.get("SepsisLabel", pd.Series([0] * len(df_full)))
    label_by_icu = dict(zip(df_full["ICULOS"].astype(float), labels.astype(int)))
    um = dict(zip(us["iculos"], us["prob_frac_series"])) if us else {}
    ualarm = dict(zip(us["iculos"], us["alarm_series"])) if us else {}
    tm = dict(zip(tj["iculos"], tj["tung_prob"])) if tj else {}
    hours = sorted(set(um) & set(tm))
    traj = {"iculos": hours,
            "user": [round(um[h] * 100, 1) for h in hours],
            "tung": [round(tm[h] * 100, 1) for h in hours],
            "user_alarm": [bool(ualarm.get(h, False)) for h in hours],
            "label": [int(label_by_icu.get(h, 0)) for h in hours]}
    max_hour = len(hours)
    hcur = max(1, min(hour or max_hour, max_hour))
    idx = hcur - 1
    hcur_icu = hours[idx] if hours else None
    ens_p = round((W * um[hcur_icu] + (1 - W) * tm[hcur_icu]) * 100, 1) if hcur_icu is not None else None
    return {
        "id": pid, "category": DEMO.get(pid), "true_onset_hour": onset,
        "hour": hcur, "iculos_at_hour": hcur_icu, "max_hour": max_hour,
        "trajectory": traj, "weight_user": W,
        "ensemble_threshold": round(ENS_THR * 100, 1), "tung_threshold": round(TUNG_THR * 100, 1),
        "user": {"prob": traj["user"][idx] if hours else None,
                 "alarm": bool(ualarm.get(hcur_icu, False))},
        "tung": {"prob": traj["tung"][idx] if hours else None,
                 "alarm": bool(hcur_icu is not None and tm[hcur_icu] >= TUNG_THR)},
        "ensemble": {"prob": ens_p,
                     "alarm": bool(hcur_icu is not None and (W * um[hcur_icu] + (1 - W) * tm[hcur_icu]) >= ENS_THR)},
        "ground_truth": {"sepsis_now": bool(label_by_icu.get(hcur_icu, 0) == 1) if hcur_icu is not None else False,
                         "onset_hour": onset},
    }


@app.route("/api/compare/<int:pid>")
def api_compare(pid):
    hour = request.args.get("hour", type=int)
    return J(compare_payload(pid, hour))


@app.route("/api/compare_random")
def api_compare_random():
    want_septic = request.args.get("septic", "0") == "1"
    pool = list(DEMO.keys()) + TEST_IDS
    random.shuffle(pool)
    chosen = None
    for pid in pool[:60]:
        _, _, onset, _ = load_patient(pid, None)
        if want_septic and onset is None:
            continue
        chosen = pid
        break
    if chosen is None:
        chosen = random.choice(list(DEMO.keys()))
    return J(compare_payload(chosen, None))


if __name__ == "__main__":
    print(f"Combined app: weight_user={W} (tuned={W_TUNED}), ens_thr={ENS_THR}. Sidecar={TUNG_URL}", flush=True)
    print("Listening on http://127.0.0.1:8710", flush=True)
    app.run(host="127.0.0.1", port=8710, debug=False)
