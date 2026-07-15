from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import data_quality, detect_exceptions, recommended_actions


EXPLANATION_FEATURES = [
    ("current_delay_hours", "Current milestone delay"),
    ("congestion_level", "Congestion level"),
    ("weather_severity", "Weather severity"),
    ("documents_ready", "Document readiness"),
    ("missing_event_count", "Missing milestone count"),
    ("event_age_hours", "Event reporting age"),
    ("source_reliability", "Source reliability"),
]


def _to_frame(snapshot: dict[str, Any] | pd.Series) -> pd.DataFrame:
    payload = snapshot.to_dict() if isinstance(snapshot, pd.Series) else dict(snapshot)
    return pd.DataFrame([payload])


def _iso(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).round("s").isoformat().replace("+00:00", "Z")


def _risk_level(probability: float) -> str:
    if probability >= 0.80:
        return "CRITICAL"
    if probability >= 0.60:
        return "HIGH"
    if probability >= 0.30:
        return "MEDIUM"
    return "LOW"


def _local_drivers(snapshot: dict[str, Any], bundle: dict[str, Any], base_prediction: float) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    references = bundle["reference_values"]
    stage_model = bundle["stage_eta"]
    for feature, label in EXPLANATION_FEATURES:
        if feature not in snapshot or feature not in references:
            continue
        perturbed = dict(snapshot)
        perturbed[feature] = references[feature]
        reference_prediction = float(stage_model.predict_remaining(_to_frame(perturbed))[0])
        contribution = base_prediction - reference_prediction
        drivers.append(
            {
                "feature": feature,
                "label": label,
                "direction": "LATER" if contribution > 0 else "EARLIER",
                "contribution_hours": round(float(contribution), 2),
                "observed_value": snapshot[feature],
                "reference_value": references[feature],
            }
        )
    drivers.sort(key=lambda item: abs(item["contribution_hours"]), reverse=True)
    return drivers[:5]


def predict_package(snapshot: dict[str, Any] | pd.Series, bundle: dict[str, Any]) -> dict[str, Any]:
    payload = snapshot.to_dict() if isinstance(snapshot, pd.Series) else dict(snapshot)
    frame = _to_frame(payload)
    stage_model = bundle["stage_eta"]
    baseline = bundle["baseline"]
    remaining = float(stage_model.predict_remaining(frame)[0])
    lower, upper = stage_model.interval(np.asarray([remaining]))
    risk_probability = float(bundle["risk"].predict_proba(frame)[0])
    breakdown_frame = stage_model.predict_breakdown(frame)
    breakdown: list[dict[str, Any]] = []
    for row in breakdown_frame.itertuples(index=False):
        breakdown.append(
            {
                "stage": row.target_stage,
                "predicted_hours": round(float(row.predicted_stage_hours), 2),
                "historical_median_hours": round(
                    baseline.stage_duration(str(payload["route"]), str(row.target_stage)), 2
                ),
            }
        )

    snapshot_time = pd.to_datetime(payload["snapshot_time"], utc=True)
    planned_delivery = pd.to_datetime(payload["planned_delivery_time"], utc=True)
    eta_p50 = snapshot_time + timedelta(hours=remaining)
    eta_p10 = snapshot_time + timedelta(hours=float(lower[0]))
    eta_p90 = snapshot_time + timedelta(hours=float(upper[0]))
    predicted_delay = float((eta_p50 - planned_delivery).total_seconds() / 3600.0)
    drivers = _local_drivers(payload, bundle, remaining)
    quality = data_quality(payload)
    exceptions = detect_exceptions(payload)
    actions = recommended_actions(payload, risk_probability)
    main_stage = max(breakdown, key=lambda item: item["predicted_hours"])["stage"] if breakdown else "DELIVERED"
    summary = (
        f"The shipment is at {payload['current_stage']}. The most time-consuming remaining stage is "
        f"{main_stage}. The median ETA implies {predicted_delay:+.1f} hours versus plan, with "
        f"{risk_probability:.0%} probability of exceeding the configured delay threshold. "
        f"Data quality is {quality['level'].lower()}."
    )
    return {
        "shipment_id": str(payload.get("shipment_id", "ad-hoc")),
        "snapshot_id": str(payload.get("snapshot_id", "ad-hoc")),
        "as_of": _iso(snapshot_time),
        "current_stage": str(payload["current_stage"]),
        "eta": {
            "p10": _iso(eta_p10),
            "p50": _iso(eta_p50),
            "p90": _iso(eta_p90),
            "planned_delivery": _iso(planned_delivery),
            "remaining_hours": round(remaining, 2),
            "predicted_delay_hours": round(predicted_delay, 2),
        },
        "risk": {
            "delay_probability": round(risk_probability, 4),
            "risk_level": _risk_level(risk_probability),
            "threshold_hours": bundle["delay_threshold_hours"],
        },
        "remaining_stages": breakdown,
        "drivers": drivers,
        "data_quality": quality,
        "exceptions": exceptions,
        "recommended_actions": actions,
        "analyst_summary": summary,
        "model_version": bundle["version"],
        "explanation_note": "Local perturbation drivers describe model sensitivity, not causal effects.",
    }
