# ICU digital twin: dual-model sepsis surveillance

This dashboard gives the two models separate operational responsibilities:

| Layer | Model | Output |
|---|---|---|
| Six-hour forecast | Tung calibrated XGBoost | Probability of entering the PhysioNet sepsis warning window in the next 6 hours |
| Active alert | Prathosh LightGBM | Current-hour alert, calibrated probability, criticality percentile, and trend |
| Explanation | SHAP on Tung XGBoost | Patient-local features raising or lowering the six-hour forecast |
| Narrative | OpenAI API or rules fallback | One observation and one review recommendation |

The dashboard does not expose `SepsisLabel` or evaluation categories. Those are appropriate for offline validation, not an operational monitor.

## Run

```bash
cd combined
export SEPSIS_DATA_A=/absolute/path/to/training_setA
export SEPSIS_DATA_B=/absolute/path/to/training_setB
bash run.sh
```

Open <http://127.0.0.1:8710>. The first run creates `.venv-main` and `.venv-tung`.

For LLM narratives, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL`. Without a key, the same API returns a deterministic rules-based observation and recommendation so the full dashboard remains usable.

## API

```text
GET /api/health
GET /api/twin/beds?hour=24
GET /api/twin/beds/171?hour=24
```

The patient-detail endpoint returns both model outputs, measurements, the forecast trajectory, signed local SHAP drivers, and the narrative. Probabilities and SHAP values are model signals, not diagnoses or causal effects.

## Process boundary

LightGBM runs in the main Flask process on `:8710`. XGBoost and SHAP run in a forecast sidecar on `:8711`. This boundary is intentional: the repository previously reproduced an OpenMP crash when both tree libraries were loaded into one Python process.

## Validation caveat

The Tung artifact was trained using all available challenge patients, including patients used by the friend repository's comparison. Its live demo scores are therefore suitable for integration testing, but not an independent estimate of generalization. Use the nested cross-validation report from the Tung model repository for model reporting and reserve an unseen hospital cohort for final external validation.
