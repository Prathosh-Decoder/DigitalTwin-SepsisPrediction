import os
import sys
import pandas as pd
import shap
from joblib import load
from flask import Flask, jsonify, request

# Add the sepsis_pipeline directory to sys.path
pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sepsis_pipeline'))
sys.path.insert(0, pipeline_dir)

import config
from feature_engineering import build_patient_features
from criticality import (
    calibrated_probability, criticality_score, tier_from_score,
    criticality_trend, top_shap_drivers,
)

app = Flask(__name__, static_url_path='', static_folder='static')

# 1. Load model artifacts
bundle_path = os.path.join(pipeline_dir, "artifacts", "model_bundle.joblib")
crit_path = os.path.join(pipeline_dir, "artifacts", "criticality_calibrator.joblib")

bundle = load(bundle_path)
model = bundle["model"]
feature_names = bundle["feature_names"]
threshold = bundle["threshold"]
crit = load(crit_path)

explainer = shap.TreeExplainer(model)

# 2. Hardcoded patient list
# 2 True Negatives, 2 False Positives, 2 False Negatives, 2 True Positives
DEMO_PATIENTS = [
    "p000049", "p000074", "p000007", "p000021", 
    "p000714", "p000851", "p000171", "p000178"
]

PATIENT_CATEGORIES = {
    "p000049": "TN", "p000074": "TN",
    "p000007": "FP", "p000021": "FP",
    "p000714": "FN", "p000851": "FN",
    "p000171": "TP", "p000178": "TP"
}

def load_patient_data(patient_id, up_to_hour=None):
    """Load up to 24 hours of patient data from their PSV file."""
    psv_path_a = os.path.join(os.path.dirname(__file__), '..', 'training_setA', 'training_setA', f"{patient_id}.psv")
    psv_path_b = os.path.join(os.path.dirname(__file__), '..', 'training_setB', 'training_setB', f"{patient_id}.psv")
    
    psv_path = psv_path_a if os.path.exists(psv_path_a) else psv_path_b
    df = pd.read_csv(psv_path, sep='|')
    
    true_onset_hour = None
    if 'SepsisLabel' in df.columns and (df['SepsisLabel'] == 1).any():
        true_onset_hour = int(df.loc[df['SepsisLabel'] == 1, 'ICULOS'].iloc[0]) + 6
    
    is_discharged = False
    if up_to_hour is not None:
        if up_to_hour > len(df):
            is_discharged = True
        df = df.iloc[:up_to_hour]
        
    # Use the last 24 rows
    df = df.tail(24).copy()
    return df, is_discharged, true_onset_hour

def score_bed(patient_id, up_to_hour=None):
    df, is_discharged, true_onset_hour = load_patient_data(patient_id, up_to_hour)
    
    if len(df) == 0:
        return {
            "id": patient_id,
            "criticality": 0.0,
            "probability": 0.0,
            "tier": "LOW",
            "trend": "steady",
            "is_risky": False,
            "drivers": [],
            "is_discharged": is_discharged,
            "true_onset_hour": true_onset_hour
        }

    # Required columns
    df["patient_id"] = patient_id
    df["hospital"] = "LIVE"
    if "SepsisLabel" not in df.columns:
        df["SepsisLabel"] = 0
    
    cols = config.VITAL_COLS + config.LAB_COLS + config.STATIC_COLS + ["patient_id", "hospital", "SepsisLabel"]
    
    # Fill missing columns with NaN
    for c in cols:
        if c not in df.columns:
            df[c] = float('nan')
            
    df = df.reindex(columns=cols)

    eng = build_patient_features(df)
    X = eng[feature_names]
    scores = model.predict(X)

    crit_series = criticality_score(scores, crit["reference_quantiles"])
    raw = float(scores[-1])
    shap_row = explainer.shap_values(X.iloc[[-1]])[0]

    raw_drivers = top_shap_drivers(shap_row, X.iloc[-1].to_numpy(), feature_names, k=5)
    
    raw_drivers = top_shap_drivers(shap_row, X.iloc[-1].to_numpy(), feature_names, k=5)
    
    drivers_categorized = {
        "vitals_labs": [],
        "demographics": [],
        "others": []
    }
    
    for d in raw_drivers:
        name = d['plain_name'].lower()
        formatted_str = f"{d['plain_name']} ({round(d['value'], 2)}) {d['direction']}"
        
        if name in ["age", "gender", "icu unit"]:
            drivers_categorized["demographics"].append(formatted_str)
        elif any(kw in name for kw in ["hours in", "time since", "hours since"]):
            drivers_categorized["others"].append(formatted_str)
        else:
            drivers_categorized["vitals_labs"].append(formatted_str)

    return {
        "id": patient_id,
        "criticality": round(float(crit_series[-1]), 1),
        "probability": round(float(calibrated_probability(crit["calibrator"], [raw])[0]) * 100, 1),
        "tier": tier_from_score(crit_series[-1]),
        "trend": criticality_trend(crit_series),
        "is_risky": bool(raw >= threshold),
        "drivers": drivers_categorized,
        "is_discharged": is_discharged,
        "true_onset_hour": true_onset_hour
    }

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/beds')
def get_all_beds():
    hour_param = request.args.get('hour', type=int)
    results = []
    for pid in DEMO_PATIENTS:
        res = score_bed(pid, up_to_hour=hour_param)
        results.append({
            "id": res["id"],
            "criticality": res["criticality"],
            "tier": res["tier"],
            "trend": res["trend"],
            "is_discharged": res["is_discharged"],
            "category": PATIENT_CATEGORIES[pid],
            "is_risky": res["is_risky"]
        })
    return jsonify(results)

@app.route('/api/beds/<patient_id>')
def get_bed_detail(patient_id):
    hour_param = request.args.get('hour', type=int)
    if patient_id not in DEMO_PATIENTS:
        return jsonify({"error": "Patient not found"}), 404
    return jsonify(score_bed(patient_id, up_to_hour=hour_param))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
