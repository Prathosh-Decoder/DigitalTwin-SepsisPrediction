# Combined — your model + Tung's model

Ensembles your LightGBM sepsis model with Tung's XGBoost model, produces an honest
precision comparison, and serves a four-tab dashboard.

## Results (held-out test set, 156,319 rows)

> **Fairness caveat.** Tung's model trained on **all 40,336 patients, including the
> 4,034 test patients scored here** — so its and the combined numbers are
> *in-sample-optimistic*. Only **your model** is a true held-out estimate. There is
> no leakage-free set (Tung saw everyone). See [eval/results/comparison.md](eval/results/comparison.md).

| Model | AUROC | AUPRC | Precision @ recall 0.67 |
|---|---:|---:|---:|
| Your model (held-out) | 0.846 | 0.121 | 0.076 |
| Tung (in-sample) | 0.899 | 0.234 | 0.132 |
| Combined 50/50 (in-sample) | 0.879 | 0.176 | 0.107 |

The validation-tuned blend weight collapsed to **pure Tung** (Tung is in-sample on the
validation patients too, so the optimizer discards your model). The app therefore serves a
genuine **50/50 blend**, which sits honestly between the two. Full numbers + native
operating points + PhysioNet utility are in `eval/results/comparison.md`.

## Run the app

```bash
bash run.sh
# → http://127.0.0.1:8710
```

`run.sh` creates the Tung venv (first run), runs the eval pipeline if needed, starts the
Tung sidecar on `:8711`, then the main app on `:8710`.

Four tabs: **My model**, **Tung model**, **Combined** (each: bed grid + probability +
criticality + SHAP drivers), and **Compare** (one patient, both models' per-hour risk vs
ground-truth onset).

## Why three Python environments

The two models cannot share a process (**LightGBM + XGBoost = OpenMP segfault**), so they
run as separate services:

| Piece | Interpreter | Why |
|---|---|---|
| `app/combined_app.py` (main app) | your `python3` (3.10) | flask + lightgbm + **shap** + criticality — already works in your `app/` |
| `app/tung_service.py` (sidecar) | `.venv-tung` (3.12) | Tung's pickle needs ≥3.11; **shap** needs numpy<2.4; pandas<2.3 avoids a predict segfault |
| `eval/*.py` (offline) | `.venv-tung` (3.12) | each a separate process; no shap needed |

## Getting Tung's model to run

Tung's packaged pickle needed two fixes, both handled in
[`tung_predictor.py`](tung_predictor.py):

1. **Booster heal** — the pickled XGBoost booster segfaults on `predict` against the
   installed xgboost; re-exporting it via `save_model`→`load_model` (xgboost's own
   cross-version guidance) normalizes it.
2. **Source reconstruction** — the predictor's methods were cloudpickled *by value* and
   their bytecode segfaults under this Python build, so inference is rebuilt from the real
   `src` source modules + the pickle's data attributes, never calling the pickled methods.

## Files

```
eval/
  score_user.py   your model → user_scores.parquet (reproduces artifacts/metrics.json)
  score_tung.py   Tung's model → tung_scores.parquet
  evaluate.py     merge, tune blend on val, evaluate on test → results/*
  results/        comparison.md, metrics.json, ensemble_config.json (+ *.parquet)
app/
  tung_service.py sidecar (:8711): Tung probability + SHAP + trajectory
  combined_app.py main app (:8710): user model + ensemble + 4-tab API + UI
  static/         four-tab dashboard
tung_predictor.py     shared, healed Tung inference (used by eval + sidecar)
requirements-tung.txt  pinned deps for .venv-tung
run.sh                 launch everything
```

## Regenerate the comparison

```bash
cd eval
../.venv-tung/bin/python score_user.py
../.venv-tung/bin/python score_tung.py   # ~3 min
../.venv-tung/bin/python evaluate.py
```
