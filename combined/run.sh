#!/usr/bin/env bash
# Launch the combined sepsis dashboard: Tung sidecar (dedicated 3.12 venv) + main app (3.10).
# The two models MUST run in separate processes (LightGBM + XGBoost share-process = OpenMP segfault).
set -euo pipefail
cd "$(dirname "$0")"

PY312="${PY312:-/Users/prathosh/prathosh/bin/python3.12}"   # any Python >=3.11 works
PY310="${PY310:-python3}"                                    # your framework 3.10 (flask+shap+lightgbm)
VENV=".venv-tung"

# 1. dedicated Tung venv (Tung needs >=3.11; SHAP needs numpy<2.4; pandas<2.3 to avoid a predict segfault)
if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating $VENV ..."
  "$PY312" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r requirements-tung.txt
fi

# 2. eval artifacts (ensemble weight/threshold/reference quantiles) must exist
if [ ! -f eval/results/ensemble_config.json ]; then
  echo "Running the evaluation pipeline (first run) ..."
  ( cd eval && ../"$VENV/bin/python" score_user.py && ../"$VENV/bin/python" score_tung.py && ../"$VENV/bin/python" evaluate.py )
fi

# 3. Tung sidecar on :8711
echo "Starting Tung sidecar on :8711 ..."
"$VENV/bin/python" app/tung_service.py &
SIDECAR=$!
trap 'kill $SIDECAR 2>/dev/null || true' EXIT
for i in $(seq 1 40); do
  curl -s http://127.0.0.1:8711/health >/dev/null 2>&1 && break
  sleep 0.5
done
echo "Sidecar ready."

# 4. main app on :8710 (foreground)
echo "Open http://127.0.0.1:8710"
"$PY310" app/combined_app.py
