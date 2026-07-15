from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_directories, project_path
from .constants import MILESTONES
from .utils import hours_between, write_json


TIME_COLUMNS = ["booking_time", "planned_delivery_time", "actual_delivery_time"]
EVENT_TIME_COLUMNS = ["planned_time", "actual_time", "observed_at"]


def _read_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = project_path(config, "raw_dir")
    processed_dir = project_path(config, "processed_dir")
    shipments = pd.read_csv(raw_dir / "shipments.csv")
    events = pd.read_csv(processed_dir / "events_canonical.csv")
    truth = pd.read_csv(raw_dir / "events_ground_truth.csv")
    for column in TIME_COLUMNS:
        shipments[column] = pd.to_datetime(shipments[column], utc=True)
    for frame in (events, truth):
        columns = EVENT_TIME_COLUMNS if frame is events else ["planned_time", "actual_time"]
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return shipments, events, truth


def _assign_splits(shipments: pd.DataFrame, train_fraction: float, validation_fraction: float) -> pd.DataFrame:
    ordered = shipments.sort_values(["booking_time", "shipment_id"]).copy()
    n = len(ordered)
    train_end = max(1, int(n * train_fraction))
    validation_end = max(train_end + 1, int(n * (train_fraction + validation_fraction)))
    ordered["split"] = "test"
    ordered.iloc[:train_end, ordered.columns.get_loc("split")] = "train"
    ordered.iloc[train_end:validation_end, ordered.columns.get_loc("split")] = "validation"
    return ordered


def build_snapshots(config: dict[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    processed_dir = project_path(config, "processed_dir")
    shipments, events, truth = _read_inputs(config)
    split_cfg = config["split"]
    shipments = _assign_splits(
        shipments,
        float(split_cfg["train_fraction"]),
        float(split_cfg["validation_fraction"]),
    )
    threshold = float(config["model"]["delay_threshold_hours"])
    shipment_lookup = shipments.set_index("shipment_id")
    truth_lookup = {
        shipment_id: group.sort_values("event_index").set_index("event_index")
        for shipment_id, group in truth.groupby("shipment_id")
    }
    snapshot_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    for shipment_id, group in events.groupby("shipment_id", sort=False):
        if shipment_id not in shipment_lookup.index or shipment_id not in truth_lookup:
            continue
        shipment = shipment_lookup.loc[shipment_id]
        event_group = group.sort_values(["observed_at", "event_index"])
        known_by_index: dict[int, pd.Series] = {}
        truth_group = truth_lookup[shipment_id]

        for snapshot_number, (observed_at, arrivals) in enumerate(event_group.groupby("observed_at", sort=True)):
            for _, event in arrivals.iterrows():
                known_by_index[int(event["event_index"])] = event
            current_index = max(known_by_index)
            if current_index >= len(MILESTONES) - 1:
                continue
            current = known_by_index[current_index]
            snapshot_time = pd.Timestamp(observed_at)
            booking_time = pd.Timestamp(shipment["booking_time"])
            planned_delivery = pd.Timestamp(shipment["planned_delivery_time"])
            actual_delivery = pd.Timestamp(shipment["actual_delivery_time"])
            # Once the physical delivery time has passed, the record is no
            # longer a meaningful pre-delivery ETA snapshot. A late report can
            # otherwise create a zero-target row after delivery.
            if snapshot_time >= actual_delivery:
                continue
            event_age = max(0.0, hours_between(snapshot_time, current["actual_time"]))
            current_delay = hours_between(current["actual_time"], current["planned_time"])
            known_indices = set(known_by_index)
            expected_indices = set(range(current_index + 1))
            missing_count = len(expected_indices - known_indices)
            known_events = pd.DataFrame(list(known_by_index.values()))
            row = {
                "snapshot_id": f"{shipment_id}-SS-{snapshot_number:02d}",
                "shipment_id": shipment_id,
                "snapshot_time": snapshot_time,
                "split": shipment["split"],
                "route": shipment["route"],
                "mode": shipment["mode"],
                "cargo_type": shipment["cargo_type"],
                "service_level": shipment["service_level"],
                "current_stage": MILESTONES[current_index],
                "event_index": current_index,
                "stage_progress": current_index / (len(MILESTONES) - 1),
                "hours_since_booking": max(0.0, hours_between(snapshot_time, booking_time)),
                "planned_remaining_hours": max(0.0, hours_between(planned_delivery, snapshot_time)),
                "current_delay_hours": current_delay,
                "event_age_hours": event_age,
                "congestion_level": float(current["congestion_level"]),
                "weather_severity": float(current["weather_severity"]),
                "documents_ready": int(current["documents_ready"]),
                "observed_event_count": len(known_indices),
                "missing_event_count": missing_count,
                "late_event_count": int(known_events["is_late_report"].sum()),
                "duplicate_event_count": int(known_events["duplicate_count"].sum()),
                "source_reliability": float(known_events["source_reliability"].mean()),
                "booking_month": booking_time.month,
                "booking_weekday": booking_time.weekday(),
                "booking_hour": booking_time.hour,
                "is_weekend": int(booking_time.weekday() >= 5),
                "actual_delivery_time": actual_delivery,
                "planned_delivery_time": planned_delivery,
                "target_remaining_hours": max(0.0, hours_between(actual_delivery, snapshot_time)),
                "final_delay_hours": hours_between(actual_delivery, planned_delivery),
                "is_delayed": int(hours_between(actual_delivery, planned_delivery) > threshold),
                "available_event_types": ";".join(MILESTONES[index] for index in sorted(known_indices)),
            }
            snapshot_rows.append(row)

            for transition_index in range(current_index, len(MILESTONES) - 1):
                target_stage = MILESTONES[transition_index + 1]
                physical_start = pd.Timestamp(truth_group.loc[transition_index, "actual_time"])
                physical_end = pd.Timestamp(truth_group.loc[transition_index + 1, "actual_time"])
                # Intersect each physical stage interval with the time that is
                # still in the future at this snapshot. This handles a report
                # arriving after one or more downstream physical milestones.
                remaining_start = max(physical_start, snapshot_time)
                target_duration = max(0.0, hours_between(physical_end, remaining_start))
                stage_rows.append(
                    {
                        **row,
                        "target_stage": target_stage,
                        "target_stage_index": transition_index + 1,
                        "is_first_remaining_stage": int(transition_index == current_index),
                        "target_stage_duration_hours": target_duration,
                    }
                )

    snapshots = pd.DataFrame(snapshot_rows).sort_values(["snapshot_time", "shipment_id"]).reset_index(drop=True)
    stages = pd.DataFrame(stage_rows).sort_values(
        ["snapshot_time", "shipment_id", "target_stage_index"]
    ).reset_index(drop=True)
    shipments_path = processed_dir / "shipments_split.csv"
    snapshots_path = processed_dir / "snapshots.csv"
    stages_path = processed_dir / "stage_training_rows.csv"
    shipments.to_csv(shipments_path, index=False)
    snapshots.to_csv(snapshots_path, index=False)
    stages.to_csv(stages_path, index=False)

    leakage_violations = 0
    if len(snapshots):
        leakage_violations = int(
            (pd.to_datetime(snapshots["snapshot_time"], utc=True) >= pd.to_datetime(snapshots["actual_delivery_time"], utc=True)).sum()
        )
    report = {
        "snapshots": int(len(snapshots)),
        "stage_training_rows": int(len(stages)),
        "shipments_by_split": shipments["split"].value_counts().to_dict(),
        "snapshots_by_split": snapshots["split"].value_counts().to_dict(),
        "future_snapshot_violations": leakage_violations,
        "shipment_split_overlap": False,
    }
    report_path = processed_dir / "snapshot_report.json"
    write_json(report_path, report)
    return {
        "shipments": shipments_path,
        "snapshots": snapshots_path,
        "stage_rows": stages_path,
        "report": report_path,
    }
