from __future__ import annotations

import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator


ROOT = Path(os.getenv("SAFIRI_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safiri_eta.config import load_config, project_path  # noqa: E402
from safiri_eta.constants import MILESTONES  # noqa: E402
from safiri_eta.models import load_default_bundle  # noqa: E402
from safiri_eta.service import predict_package  # noqa: E402
from safiri_eta.utils import read_json  # noqa: E402


app = FastAPI(
    title="SAFiRi Milestone-Aware ETA API",
    version="1.0.0",
    description="Research API for final ETA, delay risk, stage explanations and exceptions.",
)


class PredictionRequest(BaseModel):
    shipment_id: str = "ad-hoc"
    snapshot_id: str = "ad-hoc"
    snapshot_time: datetime
    planned_delivery_time: datetime
    route: str
    mode: str
    cargo_type: str = "GENERAL"
    service_level: str = "STANDARD"
    current_stage: str
    event_index: int = Field(ge=0, le=6)
    stage_progress: float = Field(ge=0, le=1)
    hours_since_booking: float = Field(ge=0)
    planned_remaining_hours: float = Field(ge=0)
    current_delay_hours: float = 0.0
    event_age_hours: float = Field(default=0.0, ge=0)
    congestion_level: float = Field(default=0.5, ge=0, le=1)
    weather_severity: float = Field(default=0.3, ge=0, le=1)
    documents_ready: int = Field(default=1, ge=0, le=1)
    observed_event_count: int = Field(default=1, ge=1)
    missing_event_count: int = Field(default=0, ge=0)
    late_event_count: int = Field(default=0, ge=0)
    duplicate_event_count: int = Field(default=0, ge=0)
    source_reliability: float = Field(default=0.9, ge=0, le=1)
    booking_month: int = Field(default=1, ge=1, le=12)
    booking_weekday: int = Field(default=0, ge=0, le=6)
    booking_hour: int = Field(default=8, ge=0, le=23)
    is_weekend: int = Field(default=0, ge=0, le=1)

    @field_validator("current_stage")
    @classmethod
    def valid_stage(cls, value: str) -> str:
        if value not in MILESTONES[:-1]:
            raise ValueError(f"current_stage must be one of {MILESTONES[:-1]}")
        return value


class SimulationRequest(BaseModel):
    shipment_id: str
    congestion_level: float | None = Field(default=None, ge=0, le=1)
    weather_severity: float | None = Field(default=None, ge=0, le=1)
    documents_ready: int | None = Field(default=None, ge=0, le=1)
    current_delay_hours: float | None = None
    missing_event_count: int | None = Field(default=None, ge=0)


@lru_cache(maxsize=1)
def resources() -> dict[str, Any]:
    config = load_config(ROOT / "configs" / "default.yaml")
    bundle = load_default_bundle(config)
    processed = project_path(config, "processed_dir")
    raw = project_path(config, "raw_dir")
    snapshots = pd.read_csv(processed / "snapshots.csv")
    events = pd.read_csv(processed / "events_canonical.csv")
    shipments = pd.read_csv(raw / "shipments.csv")
    return {"config": config, "bundle": bundle, "snapshots": snapshots, "events": events, "shipments": shipments}


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            output[key] = None
        elif hasattr(value, "item"):
            output[key] = value.item()
        else:
            output[key] = value
    return output


@app.get("/health")
def health() -> dict[str, Any]:
    data = resources()
    return {
        "status": "ok",
        "model_version": data["bundle"]["version"],
        "shipments": int(len(data["shipments"])),
        "snapshots": int(len(data["snapshots"])),
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    path = project_path(resources()["config"], "reports_dir") / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run evaluation before requesting metrics")
    return read_json(path)


@app.get("/shipments")
def list_shipments(limit: int = Query(default=50, ge=1, le=300)) -> list[dict[str, Any]]:
    columns = ["shipment_id", "route", "mode", "cargo_type", "service_level", "planned_delivery_time", "is_delayed"]
    return [_clean_record(row) for row in resources()["shipments"][columns].head(limit).to_dict(orient="records")]


@app.get("/shipments/{shipment_id}/timeline")
def timeline(shipment_id: str) -> list[dict[str, Any]]:
    frame = resources()["events"]
    rows = frame.loc[frame["shipment_id"] == shipment_id].sort_values("event_index")
    if rows.empty:
        raise HTTPException(status_code=404, detail="Shipment not found")
    columns = [
        "event_type",
        "location_code",
        "planned_time",
        "actual_time",
        "observed_at",
        "source",
        "reporting_lag_hours",
        "out_of_order_report",
    ]
    return [_clean_record(row) for row in rows[columns].to_dict(orient="records")]


@app.get("/shipments/{shipment_id}/prediction")
def shipment_prediction(shipment_id: str) -> dict[str, Any]:
    frame = resources()["snapshots"]
    rows = frame.loc[frame["shipment_id"] == shipment_id].sort_values("snapshot_time")
    if rows.empty:
        raise HTTPException(status_code=404, detail="Shipment not found or already delivered")
    return predict_package(rows.iloc[-1], resources()["bundle"])


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    payload = request.model_dump()
    payload["snapshot_time"] = request.snapshot_time.isoformat()
    payload["planned_delivery_time"] = request.planned_delivery_time.isoformat()
    return predict_package(payload, resources()["bundle"])


@app.post("/simulate")
def simulate(request: SimulationRequest) -> dict[str, Any]:
    frame = resources()["snapshots"]
    rows = frame.loc[frame["shipment_id"] == request.shipment_id].sort_values("snapshot_time")
    if rows.empty:
        raise HTTPException(status_code=404, detail="Shipment not found or already delivered")
    payload = rows.iloc[-1].to_dict()
    overrides = request.model_dump(exclude={"shipment_id"}, exclude_none=True)
    payload.update(overrides)
    result = predict_package(payload, resources()["bundle"])
    result["scenario_overrides"] = overrides
    return result

