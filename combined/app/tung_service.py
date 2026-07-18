"""Tung-model sidecar for the combined app.

Runs in the dedicated `.venv-tung` (Python 3.12, xgboost + shap, numpy<2.4, pandas<2.3),
because Tung's model needs 3.11+, its booster must be healed, and its inference must be
reconstructed from source (see combined/tung_predictor.py). It must NOT share a process
with LightGBM (OpenMP segfault) -- hence a separate service the main app calls over HTTP.

Endpoints:
  GET /health
  GET /score/<pid>?hour=H&drivers=0|1   latest-hour risk (+ optional SHAP drivers)
  GET /trajectory/<pid>                  full per-hour probability trajectory

Port 8711 (5000 = macOS AirPlay, 5001 may be the original app).
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_A = ROOT / "training_setA" / "training_setA"
DATA_B = ROOT / "training_setB" / "training_setB"
sys.path.insert(0, str(ROOT / "combined"))
from tung_predictor import TungModel  # noqa: E402

app = Flask(__name__)
MODEL = TungModel()

_SUFFIX = {
    "_ffill": "latest {}", "_missing": "{} not measured", "_since_measured": "hrs since {} drawn",
}


def pretty(name: str) -> str:
    """Light cleanup of Tung's literature_core feature names for display."""
    for suf, tmpl in _SUFFIX.items():
        if name.endswith(suf):
            return tmpl.format(name[: -len(suf)].replace("_", " "))
    return name.replace("_", " ")


def psv_path(pid: int) -> Path:
    fname = f"p{int(pid):06d}.psv"
    a = DATA_A / fname
    return a if a.exists() else DATA_B / fname


def load_slice(pid: int, hour):
    """Read a patient's PSV, optionally truncated to the first `hour` ICU rows."""
    df = pd.read_csv(psv_path(pid), sep="|")
    is_discharged = False
    if hour is not None:
        is_discharged = hour > len(df)
        df = df.iloc[:hour]
    return df, is_discharged


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "tung", "n_features": len(MODEL.feature_names)})


@app.route("/score/<int:pid>")
def score(pid):
    hour = request.args.get("hour", type=int)
    want_drivers = request.args.get("drivers", default="0") == "1"
    df, is_discharged = load_slice(pid, hour)
    if len(df) == 0:
        return jsonify({"id": pid, "tung_prob": 0.0, "tung_raw": 0.0, "alarm": False,
                        "iculos": None, "is_discharged": is_discharged, "drivers": []})
    if want_drivers:
        prob, raw, thr, alarm, drivers = MODEL.explain_row(df, k=5)
        drivers = [{"label": pretty(d["feature"]), "value": d["value"], "direction": d["direction"]}
                   for d in drivers]
    else:
        row = MODEL.trajectory(df).iloc[-1]
        prob, raw, alarm, drivers = float(row["tung_prob"]), float(row["tung_raw"]), bool(row["alarm"]), []
    return jsonify({
        "id": pid, "tung_prob": round(prob, 5), "tung_raw": round(raw, 5),
        "alarm": bool(alarm), "iculos": float(df["ICULOS"].iloc[-1]) if "ICULOS" in df else float(len(df)),
        "is_discharged": is_discharged, "drivers": drivers,
    })


@app.route("/trajectory/<int:pid>")
def trajectory(pid):
    df, _ = load_slice(pid, None)
    traj = MODEL.trajectory(df)
    return jsonify({
        "id": pid,
        "iculos": traj["ICULOS"].astype(float).tolist(),
        "tung_prob": [round(float(p), 5) for p in traj["tung_prob"]],
    })


if __name__ == "__main__":
    print(f"Tung sidecar up ({len(MODEL.feature_names)} features). Listening on :8711", flush=True)
    app.run(host="127.0.0.1", port=8711, debug=False)
