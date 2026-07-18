#!/usr/bin/env bash
# Start the two isolated model services behind the unified ICU dashboard.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_MAIN="${PYTHON_MAIN:-python3.12}"
PYTHON_TUNG="${PYTHON_TUNG:-python3.12}"
MAIN_VENV="${MAIN_VENV:-.venv-main}"
TUNG_VENV="${TUNG_VENV:-.venv-tung}"

create_venv() {
  local python="$1" venv="$2" requirements="$3"
  if [ ! -x "$venv/bin/python" ]; then
    echo "Creating $venv"
    "$python" -m venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    "$venv/bin/pip" install --quiet -r "$requirements"
  fi
}

create_venv "$PYTHON_MAIN" "$MAIN_VENV" requirements-main.txt
create_venv "$PYTHON_TUNG" "$TUNG_VENV" requirements-tung.txt

if [ -z "${SEPSIS_DATA_A:-}" ] && [ ! -d "../training_setA" ]; then
  echo "SEPSIS_DATA_A must point to the directory containing p000001.psv" >&2
  exit 1
fi
if [ -z "${SEPSIS_DATA_B:-}" ] && [ ! -d "../training_setB" ]; then
  echo "SEPSIS_DATA_B must point to the directory containing the set-B PSV files" >&2
  exit 1
fi

cleanup() {
  kill "${SIDECAR_PID:-}" "${MAIN_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting six-hour forecast sidecar on :8711"
"$TUNG_VENV/bin/python" app/tung_service.py &
SIDECAR_PID=$!
for _ in $(seq 1 90); do
  curl -fsS http://127.0.0.1:8711/health >/dev/null 2>&1 && break
  kill -0 "$SIDECAR_PID" 2>/dev/null || { echo "Forecast sidecar failed" >&2; exit 1; }
  sleep 1
done
curl -fsS http://127.0.0.1:8711/health >/dev/null

echo "Starting unified dashboard on :${PORT:-8710}"
"$MAIN_VENV/bin/python" app/combined_app.py &
MAIN_PID=$!
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:${PORT:-8710}/api/health" >/dev/null 2>&1 && break
  kill -0 "$MAIN_PID" 2>/dev/null || { echo "Dashboard service failed" >&2; exit 1; }
  sleep 1
done

echo "Open http://127.0.0.1:${PORT:-8710}"
wait "$MAIN_PID"
