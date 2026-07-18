import os
import sys
import pandas as pd
import json

# Add the sepsis_pipeline directory to sys.path
pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sepsis_pipeline'))
sys.path.insert(0, pipeline_dir)

import config
from feature_engineering import build_patient_features
from joblib import load

bundle_path = os.path.join(pipeline_dir, "artifacts", "model_bundle.joblib")
bundle = load(bundle_path)
model = bundle["model"]
feature_names = bundle["feature_names"]
threshold = bundle["threshold"]

with open(os.path.join(pipeline_dir, "artifacts", "test_patient_ids.json"), "r") as f:
    test_ids = json.load(f)

# Categories
TN = []
FP = []
FN = []
TP = []

print("Scanning patients...")
for pid in test_ids:
    if len(TN) >= 2 and len(FP) >= 2 and len(FN) >= 2 and len(TP) >= 2:
        break

    pid_str = f"p{pid:06d}"
    psv_path_a = os.path.join(os.path.dirname(__file__), '..', 'training_setA', 'training_setA', f"{pid_str}.psv")
    psv_path_b = os.path.join(os.path.dirname(__file__), '..', 'training_setB', 'training_setB', f"{pid_str}.psv")
    
    psv_path = None
    if os.path.exists(psv_path_a):
        psv_path = psv_path_a
    elif os.path.exists(psv_path_b):
        psv_path = psv_path_b
        
    if not psv_path:
        continue

    df = pd.read_csv(psv_path, sep='|')
    
    true_sepsis = False
    if 'SepsisLabel' in df.columns and (df['SepsisLabel'] == 1).any():
        true_sepsis = True
        
    df["patient_id"] = pid
    df["hospital"] = "LIVE"
    if "SepsisLabel" not in df.columns:
        df["SepsisLabel"] = 0
        
    cols = config.VITAL_COLS + config.LAB_COLS + config.STATIC_COLS + ["patient_id", "hospital", "SepsisLabel"]
    for c in cols:
        if c not in df.columns:
            df[c] = float('nan')
    df = df.reindex(columns=cols)

    eng = build_patient_features(df)
    X = eng[feature_names]
    scores = model.predict(X)
    
    pred_sepsis = (scores >= threshold).any()
    
    if not true_sepsis and not pred_sepsis and len(TN) < 2:
        TN.append(pid_str)
    elif not true_sepsis and pred_sepsis and len(FP) < 2:
        FP.append(pid_str)
    elif true_sepsis and not pred_sepsis and len(FN) < 2:
        FN.append(pid_str)
    elif true_sepsis and pred_sepsis and len(TP) < 2:
        TP.append(pid_str)

print(f"TN (True Negative): {TN}")
print(f"FP (False Positive): {FP}")
print(f"FN (False Negative): {FN}")
print(f"TP (True Positive): {TP}")
