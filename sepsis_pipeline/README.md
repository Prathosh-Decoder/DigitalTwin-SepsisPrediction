# Sepsis Prediction Training Pipeline

Standalone pipeline: trains a sepsis-risk model on the PhysioNet/CinC 2019 dataset
(`training_setA` + `training_setB`) and produces a pickled model + SHAP explainability
artifacts. See [`../docs/REPORT.md`](../docs/REPORT.md) for the full technical write-up
(methodology, model, results) and [`../docs/INTEGRATION.md`](../docs/INTEGRATION.md) for
how to consume the trained model artifact.

## Run order

```bash
pip3 install -r requirements.txt

python3 01_build_dataset.py --n-patients 200   # fast smoke test first
python3 02_feature_engineering.py
python3 03_train_model.py
python3 04_explain_shap.py
python3 05_sanity_checks.py

python3 01_build_dataset.py                    # then the full 40,336-patient run
python3 02_feature_engineering.py
python3 03_train_model.py
python3 04_explain_shap.py
python3 05_sanity_checks.py
```

Optional: `python3 06_hyperparameter_search.py` re-runs the 27-combination hyperparameter
grid search that informed `config.py`'s `LGBM_PARAMS`. `evaluate_model.ipynb` loads the
already-trained model and lets you explore thresholds, per-hospital performance, and
reward/penalty sensitivity interactively without retraining anything.

## Deliverables (in `artifacts/`)

- `model_bundle.joblib` -- `{model, feature_names, threshold, impute_medians}`
- `shap_values.npy`, `shap_sample_features.parquet`, `shap_explainer.joblib`
- `metrics.json` -- AUROC/AUPRC/utility on train/val/test + test hospital-A/B slices
- `hyperparam_search_results.csv` -- full grid search results
- `plots/` -- ROC, PR, threshold sweep, SHAP beeswarm + bar charts
