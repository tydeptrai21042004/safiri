# SAFiRi Milestone-Aware ETA Lab

An end-to-end research prototype for the SAFiRi AI Internship take-home assignment. It predicts shipment ETA and delay risk from point-in-time milestone events, explains how remaining stages contribute to ETA, detects data/operational exceptions, and provides a FastAPI service plus Streamlit demo.

> **Data disclosure:** the default end-to-end dataset is synthetic. The code does not claim production performance or use private SAFiRi data.

## What is included

- Reproducible generator for 300 shipment journeys.
- Raw, ground-truth and observed-event tables.
- Event normalization, validation, deduplication and lineage.
- Leakage-safe point-in-time snapshots.
- Route-stage median baseline.
- Direct remaining-time gradient boosting.
- Stage-aware ETA with remaining-stage decomposition.
- Calibrated delay-risk classifier.
- P10/P50/P90 ETA interval.
- Local model-sensitivity explanations.
- Missing, late, duplicate and low-reliability event exceptions.
- FastAPI endpoints and Streamlit scenario dashboard.
- Chronological evaluation, plots, a three-page report and automated tests.

## Architecture

```text
Synthetic event generator
        |
        v
data/raw/events_raw.csv                 Bronze: immutable source representation
        |
        v
Validation -> normalization -> deduplication
        |
        v
data/processed/events_canonical.csv     Silver: canonical milestone stream
        |
        v
Point-in-time snapshot builder
        |
        v
snapshots.csv + stage_training_rows.csv Gold: model-ready tables
        |
        +--> ETA baseline / direct ETA / stage-aware ETA
        +--> calibrated delay-risk model
        +--> explanations + exception rules
        |
        v
FastAPI + Streamlit dashboard
```

## Quick start

Python 3.10 or newer is required.

### Linux, macOS or Git Bash

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make all
make test
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m safiri_eta.cli all
python -m pytest
```

The repository already contains generated sample data, trained artifacts and measured test results. Running `all` recreates them with seed 42.

## Run the demo

Terminal 1:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open API documentation at `http://localhost:8000/docs`.

Terminal 2:

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

Open the dashboard at `http://localhost:8501`.

Docker is also supported:

```bash
docker compose up --build
```

## Pipeline commands

| Command | Output |
|---|---|
| `python -m safiri_eta.cli generate` | Shipments, raw events and complete ground truth |
| `python -m safiri_eta.cli prepare` | Validated canonical events and point-in-time snapshots |
| `python -m safiri_eta.cli train` | `artifacts/model_bundle.joblib` |
| `python -m safiri_eta.cli evaluate` | Metrics, test predictions and figures |
| `python -m safiri_eta.cli report` | Three-page `reports/technical_report.pdf` |
| `python -m safiri_eta.cli all` | Complete pipeline |

To change sample size:

```bash
python -m safiri_eta.cli all --n-shipments 1000
```

## Point-in-time rule

An event is available to a prediction only when:

```text
event.observed_at <= snapshot.snapshot_time
```

`actual_time` and `observed_at` are deliberately different. The former is the physical event time; the latter is when the platform learned it. Future milestone timestamps are used only to construct training labels.

All snapshots belonging to one shipment are assigned to the same chronological train, validation or test split.

## Models

### Baseline

For every remaining stage, use its historical median duration for the current route. This is difficult to beat accidentally and provides a clear operational reference.

### Direct ETA

Predict total remaining hours directly from the current snapshot.

### Stage-aware ETA

For current milestone index `k`, predict every future transition:

```text
ETA = snapshot_time + sum(predicted duration of stages k+1 ... DELIVERED)
```

This gives customs, gate-out and final-mile contributions instead of one opaque duration.

### Delay risk

The binary target is:

```text
actual_delivery_time > planned_delivery_time + 12 hours
```

The classifier is class weighted and its probability is calibrated on the validation set.

## Actual measured results

The included seed-42 run produced the following chronological test results:

| ETA model | MAE | RMSE | Within +/-12 h |
|---|---:|---:|---:|
| Route-stage median | 10.01 h | 14.98 h | 76.4% |
| Direct ETA | 6.43 h | 8.84 h | 85.1% |
| Stage-aware ETA | 7.68 h | 11.93 h | 80.4% |

Delay-risk performance:

- Precision: 1.000
- Recall: 0.742
- F1: 0.852
- PR-AUC: 0.915
- Brier score: 0.054

The direct model is the accuracy winner in this synthetic run. The stage-aware model remains valuable because it improves on the baseline and provides a stage decomposition. The report presents this as an accuracy-interpretability trade-off, not as a claim that the structured model must always win.

## API examples

Health check:

```bash
curl http://localhost:8000/health
```

Existing shipment prediction:

```bash
curl http://localhost:8000/shipments/SHP-0300/prediction
```

Scenario analysis:

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "shipment_id": "SHP-0300",
    "congestion_level": 0.95,
    "documents_ready": 0,
    "missing_event_count": 2
  }'
```

## Prediction explanation

Each response includes:

- As-of time and model version.
- Planned delivery and ETA P10/P50/P90.
- Delay probability and risk band.
- Predicted hours for every remaining stage.
- Historical route-stage median.
- Top local feature sensitivities in hours.
- Data-quality score and issues.
- Exception rules and recommended actions.

Local perturbation explanations are model diagnostics, not causal claims.

## Repository layout

```text
safiri-eta-lab/
├── api/                    FastAPI application
├── dashboard/              Streamlit demo
├── configs/                Simulation and model settings
├── data/raw/               Synthetic source data and ground truth
├── data/processed/         Canonical events and snapshots
├── artifacts/              Trained model bundle
├── reports/                Metrics, figures and technical report
├── scripts/                Utility scripts
├── src/safiri_eta/         Reusable pipeline and model package
├── tests/                  Unit and integration tests
├── DATA_CARD.md
├── MODEL_CARD.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Suggested five-minute demo

1. Show the event timeline and explain `actual_time` versus `observed_at`.
2. Select a shipment at an intermediate milestone.
3. Present ETA, delay probability and P10-P90 interval.
4. Explain the remaining-stage chart and top drivers.
5. Increase congestion, mark documents not ready and add a missing event.
6. Show the updated ETA, risk, exceptions and recommended actions.
7. Finish with the chronological test table and limitations.

## Limitations

- Synthetic data cannot validate production performance.
- The assignment-sized sample is intentionally small.
- Missing-event robustness is a sensitivity test; full event deletion requires rebuilding snapshots.
- A model explanation is not evidence that a feature caused a delay.
- Production rollout requires real-source schema mapping, shadow evaluation, route calibration, drift monitoring and human review.

See [DATA_CARD.md](DATA_CARD.md), [MODEL_CARD.md](MODEL_CARD.md) and [reports/technical_report.pdf](reports/technical_report.pdf) for details.
