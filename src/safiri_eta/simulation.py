from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_directories, project_path
from .constants import MILESTONES, STATUS_VARIANTS
from .utils import set_seed, write_json


@dataclass(frozen=True)
class RouteProfile:
    origin: str
    destination: str
    mode: str
    planned_hours: tuple[float, ...]
    congestion_bias: float
    weather_bias: float


ROUTES: dict[str, RouteProfile] = {
    "SGSIN-AUSYD": RouteProfile("SGSIN", "AUSYD", "SEA", (8, 72, 10, 8, 14, 7, 16), 0.58, 0.30),
    "CNSHA-DEHAM": RouteProfile("CNSHA", "DEHAM", "SEA", (10, 360, 16, 9, 20, 8, 20), 0.66, 0.42),
    "USLAX-USCHI": RouteProfile("USLAX", "USCHI", "ROAD", (5, 40, 3, 2, 4, 3, 18), 0.50, 0.22),
    "AEDXB-GBLHR": RouteProfile("AEDXB", "GBLHR", "AIR", (4, 9, 3, 2, 6, 2, 7), 0.34, 0.28),
    "SGSIN-USLAX": RouteProfile("SGSIN", "USLAX", "MULTIMODAL", (6, 190, 8, 7, 13, 6, 24), 0.54, 0.36),
    "NLRTM-FRPAR": RouteProfile("NLRTM", "FRPAR", "MULTIMODAL", (5, 22, 7, 5, 8, 5, 12), 0.46, 0.25),
}

SOURCES = {
    "carrier_api": 0.86,
    "port_system": 0.95,
    "customs_feed": 0.93,
    "manual_update": 0.72,
}


def _source_for_stage(stage: str, rng: np.random.Generator) -> str:
    if stage in {"PORT_ARRIVED", "DISCHARGED", "GATE_OUT"}:
        choices = ["port_system", "carrier_api", "manual_update"]
        probs = [0.62, 0.30, 0.08]
    elif stage in {"CUSTOMS_STARTED", "CUSTOMS_CLEARED"}:
        choices = ["customs_feed", "carrier_api", "manual_update"]
        probs = [0.68, 0.25, 0.07]
    else:
        choices = ["carrier_api", "manual_update"]
        probs = [0.90, 0.10]
    return str(rng.choice(choices, p=probs))


def _stage_shock(
    stage_index: int,
    congestion: float,
    weather: float,
    documents_ready: int,
    cumulative_delay: float,
    rng: np.random.Generator,
) -> float:
    transport_stage = stage_index in {0, 1, 6}
    port_stage = stage_index in {1, 2, 5}
    customs_stage = stage_index in {3, 4}
    # Planned durations represent ordinary operating conditions. Effects are
    # therefore centred around typical congestion/weather, so the generator
    # produces both early and delayed shipments instead of adding delay to
    # every stage by construction.
    congestion_effect = (congestion - 0.55) * (10.0 if port_stage else 2.0)
    weather_effect = (weather - 0.35) * (8.0 if transport_stage else 1.5)
    document_effect = (1 - documents_ready) * (14.0 if customs_stage else 0.0)
    interaction = max(congestion - 0.55, 0.0) * (1 - documents_ready) * (8.0 if customs_stage else 0.0)
    random_disruption = float(rng.gamma(1.4, 2.0)) if rng.random() < 0.13 else 0.0
    recovery = min(max(cumulative_delay, 0.0) * float(rng.uniform(0.08, 0.22)), 12.0)
    noise = float(rng.normal(0.0, 1.8 + stage_index * 0.25))
    schedule_buffer = 0.0
    return congestion_effect + weather_effect + document_effect + interaction + random_disruption + noise - recovery - schedule_buffer


def generate_dataset(config: dict[str, Any], n_shipments: int | None = None) -> dict[str, Path]:
    ensure_directories(config)
    settings = config["simulation"]
    seed = int(settings["seed"])
    n = int(n_shipments or settings["n_shipments"])
    rng = set_seed(seed)
    raw_dir = project_path(config, "raw_dir")

    shipment_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    raw_event_rows: list[dict[str, Any]] = []
    route_names = list(ROUTES)
    cargo_types = ["GENERAL", "PERISHABLE", "ELECTRONICS", "FASHION"]
    service_levels = ["STANDARD", "EXPRESS"]
    base_booking = pd.Timestamp(settings["start_date"], tz="UTC")

    for shipment_number in range(n):
        shipment_id = f"SHP-{shipment_number + 1:04d}"
        route_name = route_names[shipment_number % len(route_names)]
        profile = ROUTES[route_name]
        booking_time = base_booking + pd.Timedelta(hours=shipment_number * 10 + int(rng.integers(0, 7)))
        cargo_type = str(rng.choice(cargo_types, p=[0.48, 0.13, 0.21, 0.18]))
        service_level = str(rng.choice(service_levels, p=[0.76, 0.24]))
        express_factor = 0.86 if service_level == "EXPRESS" else 1.0
        congestion = float(np.clip(rng.beta(2.2, 2.4) * 0.65 + profile.congestion_bias * 0.35, 0, 1))
        weather = float(np.clip(rng.beta(1.8, 4.2) * 0.65 + profile.weather_bias * 0.35, 0, 1))
        documents_ready = int(rng.random() > (0.12 + 0.16 * congestion))
        planned_times = [booking_time]
        for duration in profile.planned_hours:
            planned_times.append(planned_times[-1] + pd.Timedelta(hours=duration * express_factor))

        actual_times = [booking_time + pd.Timedelta(minutes=float(rng.uniform(2, 25)))]
        cumulative_delay = float((actual_times[0] - planned_times[0]).total_seconds() / 3600)
        stage_context: list[tuple[float, float, int]] = []
        for stage_index, planned_duration in enumerate(profile.planned_hours):
            local_congestion = float(np.clip(congestion + rng.normal(0, 0.09), 0, 1))
            local_weather = float(np.clip(weather + rng.normal(0, 0.08), 0, 1))
            shock = _stage_shock(
                stage_index,
                local_congestion,
                local_weather,
                documents_ready,
                cumulative_delay,
                rng,
            )
            cargo_multiplier = 1.12 if cargo_type == "PERISHABLE" and stage_index in {3, 4} else 1.0
            actual_duration = max(0.5, planned_duration * express_factor * cargo_multiplier + shock)
            actual_times.append(actual_times[-1] + pd.Timedelta(hours=actual_duration))
            cumulative_delay = float((actual_times[-1] - planned_times[stage_index + 1]).total_seconds() / 3600)
            stage_context.append((local_congestion, local_weather, documents_ready))

        final_delay_hours = float((actual_times[-1] - planned_times[-1]).total_seconds() / 3600)
        shipment_rows.append(
            {
                "shipment_id": shipment_id,
                "route": route_name,
                "origin": profile.origin,
                "destination": profile.destination,
                "mode": profile.mode,
                "cargo_type": cargo_type,
                "service_level": service_level,
                "booking_time": booking_time,
                "planned_delivery_time": planned_times[-1],
                "actual_delivery_time": actual_times[-1],
                "final_delay_hours": final_delay_hours,
                "is_delayed": int(final_delay_hours > float(config["model"]["delay_threshold_hours"])),
                "base_congestion": congestion,
                "base_weather": weather,
                "documents_ready": documents_ready,
            }
        )

        for event_index, event_type in enumerate(MILESTONES):
            if event_index == 0:
                local_congestion, local_weather, local_documents = congestion, weather, documents_ready
            else:
                local_congestion, local_weather, local_documents = stage_context[event_index - 1]
            truth_row = {
                "event_id": f"TRUTH-{shipment_id}-{event_index:02d}",
                "shipment_id": shipment_id,
                "event_type": event_type,
                "event_index": event_index,
                "location_code": profile.origin if event_index <= 1 else profile.destination,
                "planned_time": planned_times[event_index],
                "actual_time": actual_times[event_index],
                "congestion_level": local_congestion,
                "weather_severity": local_weather,
                "documents_ready": local_documents,
            }
            truth_rows.append(truth_row)

            always_observed = event_type in {"BOOKED", "DELIVERED"}
            missing_probability = float(settings["missing_event_probability"])
            if not always_observed and rng.random() < missing_probability:
                continue

            source = _source_for_stage(event_type, rng)
            base_lag = float(rng.gamma(1.4, 1.7))
            if rng.random() < float(settings["late_event_probability"]):
                base_lag += float(rng.uniform(7, 28))
            observed_at = actual_times[event_index] + pd.Timedelta(hours=base_lag)
            raw_event_rows.append(
                {
                    "source_event_id": f"EVT-{shipment_id}-{event_index:02d}-A",
                    "shipment_id": shipment_id,
                    "source_status": str(rng.choice(STATUS_VARIANTS[event_type])),
                    "location_code": truth_row["location_code"],
                    "planned_time": planned_times[event_index],
                    "actual_time": actual_times[event_index],
                    "observed_at": observed_at,
                    "source": source,
                    "source_reliability": SOURCES[source],
                    "congestion_level": local_congestion,
                    "weather_severity": local_weather,
                    "documents_ready": local_documents,
                }
            )
            if rng.random() < float(settings["duplicate_event_probability"]):
                duplicate_source = _source_for_stage(event_type, rng)
                raw_event_rows.append(
                    {
                        **raw_event_rows[-1],
                        "source_event_id": f"EVT-{shipment_id}-{event_index:02d}-B",
                        "observed_at": observed_at + pd.Timedelta(minutes=float(rng.uniform(5, 90))),
                        "source": duplicate_source,
                        "source_reliability": SOURCES[duplicate_source],
                    }
                )

    shipments = pd.DataFrame(shipment_rows).sort_values("booking_time").reset_index(drop=True)
    truth = pd.DataFrame(truth_rows).sort_values(["shipment_id", "event_index"]).reset_index(drop=True)
    raw_events = pd.DataFrame(raw_event_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    shipments_path = raw_dir / "shipments.csv"
    truth_path = raw_dir / "events_ground_truth.csv"
    raw_events_path = raw_dir / "events_raw.csv"
    shipments.to_csv(shipments_path, index=False)
    truth.to_csv(truth_path, index=False)
    raw_events.to_csv(raw_events_path, index=False)

    manifest = {
        "dataset_type": "synthetic",
        "seed": seed,
        "shipments": int(len(shipments)),
        "ground_truth_events": int(len(truth)),
        "raw_observed_events": int(len(raw_events)),
        "routes": int(shipments["route"].nunique()),
        "delayed_rate": float(shipments["is_delayed"].mean()),
        "assumptions": [
            "All timestamps are UTC.",
            "BOOKED and DELIVERED are always observed.",
            "Intermediate milestones may be missing or reported late.",
            "Synthetic data is not presented as real SAFiRi operational data.",
        ],
    }
    write_json(raw_dir / "manifest.json", manifest)
    return {
        "shipments": shipments_path,
        "truth_events": truth_path,
        "raw_events": raw_events_path,
        "manifest": raw_dir / "manifest.json",
    }
