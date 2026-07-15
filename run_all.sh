#!/usr/bin/env bash

# One-command launcher for the SAFiRi ETA project.
# Place this file in the repository root, then run: ./run_all.sh

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Set any flag to 0 to skip that stage, for example:
# INSTALL_DEPS=0 RUN_PIPELINE=0 RUN_TESTS=0 ./run_all.sh
INSTALL_DEPS="${INSTALL_DEPS:-1}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
START_SERVERS="${START_SERVERS:-1}"

cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: ${PYTHON_BIN} was not found. Install Python 3.10 or newer." >&2
    exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" && ! -f "${VENV_DIR}/Scripts/activate" ]]; then
    echo "[1/6] Creating virtual environment: ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "[1/6] Reusing virtual environment: ${VENV_DIR}"
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # Linux and macOS
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
    # Git Bash on Windows
    # shellcheck disable=SC1091
    source "${VENV_DIR}/Scripts/activate"
else
    echo "Error: the virtual environment was not created correctly." >&2
    exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SAFIRI_PROJECT_ROOT="${ROOT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}" "${ROOT_DIR}/logs"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
    echo "[2/6] Installing project dependencies"
    python -m pip install -e ".[dev]"
else
    echo "[2/6] Dependency installation skipped"
fi

if [[ "${RUN_PIPELINE}" == "1" ]]; then
    echo "[3/6] Running data pipeline, training, evaluation, and report generation"
    python -m safiri_eta.cli all
else
    echo "[3/6] Pipeline skipped"
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
    echo "[4/6] Running automated tests"
    python -m pytest
else
    echo "[4/6] Tests skipped"
fi

if [[ "${START_SERVERS}" != "1" ]]; then
    echo "[5/6] API startup skipped"
    echo "[6/6] Dashboard startup skipped"
    echo "Completed successfully."
    exit 0
fi

API_LOG="${ROOT_DIR}/logs/api.log"
DASHBOARD_LOG="${ROOT_DIR}/logs/dashboard.log"

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    echo
    echo "Stopping SAFiRi services..."

    if [[ -n "${API_PID:-}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
        kill "${API_PID}" 2>/dev/null || true
    fi

    if [[ -n "${DASHBOARD_PID:-}" ]] && kill -0 "${DASHBOARD_PID}" 2>/dev/null; then
        kill "${DASHBOARD_PID}" 2>/dev/null || true
    fi

    wait "${API_PID:-}" 2>/dev/null || true
    wait "${DASHBOARD_PID:-}" 2>/dev/null || true
    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

echo "[5/6] Starting FastAPI"
python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    >"${API_LOG}" 2>&1 &
API_PID=$!

echo "[6/6] Starting Streamlit dashboard"
python -m streamlit run dashboard/app.py \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    >"${DASHBOARD_LOG}" 2>&1 &
DASHBOARD_PID=$!

sleep 2

if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "Error: FastAPI failed to start. See ${API_LOG}" >&2
    exit 1
fi

if ! kill -0 "${DASHBOARD_PID}" 2>/dev/null; then
    echo "Error: Streamlit failed to start. See ${DASHBOARD_LOG}" >&2
    exit 1
fi

echo
echo "SAFiRi project is running."
echo "API:       http://localhost:8000"
echo "API docs:  http://localhost:8000/docs"
echo "Dashboard: http://localhost:8501"
echo "API log:   ${API_LOG}"
echo "UI log:    ${DASHBOARD_LOG}"
echo
echo "Press Ctrl+C to stop both services."

wait "${API_PID}" "${DASHBOARD_PID}"
