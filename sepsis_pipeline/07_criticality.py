"""Fit the criticality layer on top of the frozen model and validate it.

Produces artifacts/criticality_calibrator.joblib = {calibrator, reference_quantiles, tier_bands,
feature_plain_names} and validation plots. Does NOT modify model_bundle.joblib (verified by md5).

Run AFTER 03_train_model.py and 04_explain_shap.py.

Usage:
    python3 07_criticality.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from joblib import dump, load

import config
import criticality as C
import sepsis_utils

CALIBRATOR_PATH = config.ARTIFACTS_DIR / "criticality_calibrator.joblib"
DEMO_PATIENT_IDS = [4880, 1072, 17091, 7057, 14527, 295, 10756, 11623]


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    model_md5_before = md5(config.MODEL_BUNDLE_PATH)

    print(f"Loading model bundle + features ...")
    bundle = load(config.MODEL_BUNDLE_PATH)
    model, feature_names = bundle["model"], bundle["feature_names"]

    features = pd.read_parquet(config.FEATURES_PARQUET)

    train_ids = set(json.loads(config.TRAIN_IDS_PATH.read_text()))
    val_ids = set(json.loads(config.VAL_IDS_PATH.read_text()))
    test_ids = set(json.loads(config.TEST_IDS_PATH.read_text()))

    tr = features[features["patient_id"].isin(train_ids)]
    va = features[features["patient_id"].isin(val_ids)]
    te = features[features["patient_id"].isin(test_ids)].copy()

    train_scores = model.predict(tr[feature_names])
    val_scores = model.predict(va[feature_names])
    test_scores = model.predict(te[feature_names])
    val_y = va[config.LABEL_COL].to_numpy()
    test_y = te[config.LABEL_COL].to_numpy()

    # --- fit calibrator (VALIDATION only) + reference (TRAIN only) ---
    calibrator = C.fit_calibrator(val_scores, val_y)
    reference_quantiles = C.build_reference_quantiles(train_scores)
    feature_plain_names = C.build_plain_names(feature_names)

    dump({"calibrator": calibrator, "reference_quantiles": reference_quantiles,
          "tier_bands": C.DEFAULT_TIER_BANDS, "feature_plain_names": feature_plain_names},
         CALIBRATOR_PATH)
    print(f"Saved {CALIBRATOR_PATH.name}")

    # ------------------------------------------------------------------
    # Validation: does it make sense?
    # ------------------------------------------------------------------
    te["crit"] = C.criticality_score(test_scores, reference_quantiles)
    te["prob"] = C.calibrated_probability(calibrator, test_scores)

    # (1) septic vs non-septic criticality
    crit_pos = te.loc[test_y == 1, "crit"].mean()
    crit_neg = te.loc[test_y == 0, "crit"].mean()
    print(f"\n[1] Mean criticality -- pre-sepsis hours: {crit_pos:.1f} vs non-sepsis: {crit_neg:.1f} "
          f"({'OK, septic higher' if crit_pos > crit_neg else 'WARNING: not higher!'})")

    # (2) reliability curve + Brier
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(test_y, te["prob"], n_bins=10, strategy="quantile")
    brier = brier_score_loss(test_y, te["prob"])
    print(f"[2] Brier score (lower is better): {brier:.4f}")
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="calibrated")
    plt.plot([0, prob_pred.max()], [0, prob_pred.max()], "--", color="gray", label="perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed sepsis rate")
    plt.title(f"Reliability curve (Brier={brier:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "calibration_reliability.png", dpi=150)
    plt.close()

    # (3) trajectories: criticality rising toward onset for a few septic test patients
    septic_pids = (te[te[config.LABEL_COL] == 1]["patient_id"].value_counts().index[:4].tolist())
    plt.figure(figsize=(7, 4.5))
    for pid in septic_pids:
        pdf = te[te["patient_id"] == pid].sort_values("ICULOS")
        onset_hr = pdf.loc[pdf[config.LABEL_COL] == 1, "ICULOS"].min()
        plt.plot(pdf["ICULOS"] - onset_hr, pdf["crit"], marker=".", label=f"p{pid:06d}")
    plt.axvline(0, color="red", ls="--", lw=1, label="label onset")
    plt.xlabel("hours relative to SepsisLabel onset")
    plt.ylabel("criticality (0-100)")
    plt.title("Criticality trajectories (septic test patients)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "criticality_trajectories.png", dpi=150)
    plt.close()
    print("[3] Saved reliability + trajectory plots to artifacts/plots/")

    # (4) 8-bed triage table (all demo patients are test-only)
    from joblib import load as jload
    explainer = jload(config.SHAP_EXPLAINER_PATH)
    print("\n[4] 8-bed triage table (ordered by criticality):")
    rows = []
    for pid in DEMO_PATIENT_IDS:
        pdf = te[te["patient_id"] == pid].sort_values("ICULOS")
        if pdf.empty:
            continue
        recent = pdf.iloc[-1]
        feat_row = recent[feature_names].astype(float)
        trend = C.criticality_trend(pdf["crit"].to_numpy())
        shap_vals = explainer.shap_values(feat_row.to_frame().T)[0]
        drivers = C.top_shap_drivers(shap_vals, feat_row.to_numpy(), feature_names)
        rows.append({"pid": pid, "crit": float(recent["crit"]), "prob": float(recent["prob"]),
                     "tier": C.tier_from_score(recent["crit"]), "trend": trend,
                     "drivers": " · ".join(f"{d['plain_name']} {d['direction']}" for d in drivers)})
    rows.sort(key=lambda r: r["crit"], reverse=True)
    for r in rows:
        print(f"  p{r['pid']:06d}  CRIT {r['crit']:5.1f}/100  {r['tier']:8s}  {r['trend']:7s}  "
              f"risk {r['prob']:.1%}  | {r['drivers']}")

    # ------------------------------------------------------------------
    model_md5_after = md5(config.MODEL_BUNDLE_PATH)
    assert model_md5_before == model_md5_after, "MODEL BUNDLE CHANGED -- must stay frozen!"
    print(f"\nmodel_bundle.joblib md5 unchanged ({model_md5_after[:8]}...). Model untouched.")


if __name__ == "__main__":
    main()
