# Digital Twin — Sepsis Prediction

An early sepsis-warning model for the Connected ICU Ward Digital Twin project, trained on the
PhysioNet/Computing in Cardiology 2019 Sepsis Challenge dataset. Predicts, hour by hour, whether a
patient is heading toward sepsis roughly six hours before clinical onset, using an approach adapted
from the challenge's top-ranked submission (LightGBM trained directly against the official
utility-scoring rule).

**Start here:**
- [`docs/REPORT.md`](docs/REPORT.md) — full technical write-up: problem, related work, methodology, model, results
- [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — how to load and use the trained model artifact in another system
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — alternative models/features tried after finalizing the production model, and why they weren't adopted
- [`sepsis_pipeline/README.md`](sepsis_pipeline/README.md) — how to run/reproduce the training pipeline

## Repository structure

```
DigitalTwins/
├── docs/
│   ├── REPORT.md              # full technical report
│   └── INTEGRATION.md         # integration guide for consuming the trained model
├── training_setA/             # PhysioNet dataset, hospital A (20,336 patients)
├── training_setB/             # PhysioNet dataset, hospital B (20,000 patients)
└── sepsis_pipeline/
    ├── config.py               # paths, constants, hyperparameters
    ├── sepsis_utils.py         # official utility-score formula, threshold search, eval metrics
    ├── feature_engineering.py  # causal per-patient feature construction (214 features)
    ├── 01_build_dataset.py     # parses raw PSV files into one combined dataset
    ├── 02_feature_engineering.py
    ├── 03_train_model.py       # train/val/test split, LightGBM training, model bundle
    ├── 04_explain_shap.py      # SHAP explainability artifacts
    ├── 05_sanity_checks.py     # leakage/causality/split verification suite
    ├── 06_hyperparameter_search.py
    ├── evaluate_model.ipynb    # interactive exploration of the trained model
    ├── Sepsis_Prediction_Report.docx
    └── artifacts/               # model_bundle.joblib, SHAP outputs, metrics.json, plots/
```

## Quick start

```bash
cd sepsis_pipeline
pip3 install -r requirements.txt
python3 01_build_dataset.py && python3 02_feature_engineering.py && python3 03_train_model.py
python3 04_explain_shap.py && python3 05_sanity_checks.py
```

Produces `sepsis_pipeline/artifacts/model_bundle.joblib` (the trained model) and the SHAP
explainability artifacts. See [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for how to consume them.

## Headline results

Held-out test set (patients never seen during training or hyperparameter selection):

| AUROC | AUPRC | Precision | Recall | F1 | Lift@10% | Utility |
|---:|---:|---:|---:|---:|---:|---:|
| 0.846 | 0.122 | 0.069 | 0.705 | 0.126 | 5.56x | 0.438 |

Full results, per-hospital breakdown, and what each metric means: [`docs/REPORT.md`](docs/REPORT.md).

## Important caveat

This is a research/course project, not a validated clinical device. See the Limitations section of
[`docs/REPORT.md`](docs/REPORT.md) before drawing conclusions about real-world readiness.
