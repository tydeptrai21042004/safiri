from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_directories, project_path
from .constants import MILESTONE_INDEX, STATUS_TO_CANONICAL
from .utils import write_json


REQUIRED_EVENT_COLUMNS = {
    "source_event_id",
    "shipment_id",
    "source_status",
    "location_code",
    "planned_time",
    "actual_time",
    "observed_at",
    "source",
    "source_reliability",
}


def normalize_events(config: dict[str, Any]) -> dict[str, Path]:
    ensure_directories(config)
    raw_dir = project_path(config, "raw_dir")
    processed_dir = project_path(config, "processed_dir")
    raw = pd.read_csv(raw_dir / "events_raw.csv")
    missing_columns = sorted(REQUIRED_EVENT_COLUMNS - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Raw event file is missing required columns: {missing_columns}")

    raw["event_type"] = raw["source_status"].str.strip().str.lower().map(STATUS_TO_CANONICAL)
    for column in ("planned_time", "actual_time", "observed_at"):
        raw[column] = pd.to_datetime(raw[column], utc=True, errors="coerce")

    reasons = pd.Series("", index=raw.index, dtype="object")
    reasons = reasons.mask(raw["shipment_id"].isna() | raw["shipment_id"].eq(""), "missing_shipment_id")
    reasons = reasons.mask(raw["event_type"].isna(), "unknown_event_type")
    reasons = reasons.mask(raw[["planned_time", "actual_time", "observed_at"]].isna().any(axis=1), "invalid_timestamp")
    reasons = reasons.mask(raw["observed_at"] < raw["actual_time"], "observed_before_actual")
    rejected = raw.loc[reasons.ne("")].copy()
    rejected["rejection_reason"] = reasons.loc[reasons.ne("")]
    valid = raw.loc[reasons.eq("")].copy()

    valid["event_index"] = valid["event_type"].map(MILESTONE_INDEX).astype(int)
    valid["reporting_lag_hours"] = (
        valid["observed_at"] - valid["actual_time"]
    ).dt.total_seconds().div(3600).clip(lower=0)
    valid["is_late_report"] = (valid["reporting_lag_hours"] > 6).astype(int)

    key = ["shipment_id", "event_type", "actual_time", "location_code"]
    duplicate_counts = valid.groupby(key)["source_event_id"].transform("size")
    valid["duplicate_count"] = duplicate_counts.sub(1).clip(lower=0).astype(int)
    valid = valid.sort_values(
        key + ["source_reliability", "observed_at"],
        ascending=[True, True, True, True, False, True],
    )
    canonical = valid.drop_duplicates(key, keep="first").copy()
    canonical["event_id"] = canonical["source_event_id"]
    canonical = canonical.sort_values(["shipment_id", "observed_at", "event_index"]).reset_index(drop=True)

    sequence_rows: list[dict[str, Any]] = []
    for shipment_id, group in canonical.groupby("shipment_id", sort=False):
        highest = -1
        for row in group.itertuples(index=False):
            out_of_order = int(row.event_index < highest)
            highest = max(highest, row.event_index)
            sequence_rows.append(
                {
                    "event_id": row.event_id,
                    "out_of_order_report": out_of_order,
                }
            )
    sequence_flags = pd.DataFrame(sequence_rows)
    canonical = canonical.merge(sequence_flags, on="event_id", how="left")

    canonical_path = processed_dir / "events_canonical.csv"
    rejected_path = processed_dir / "events_rejected.csv"
    canonical.to_csv(canonical_path, index=False)
    rejected.to_csv(rejected_path, index=False)
    report = {
        "raw_records": int(len(raw)),
        "valid_records_before_deduplication": int(len(valid)),
        "canonical_records": int(len(canonical)),
        "rejected_records": int(len(rejected)),
        "duplicates_removed": int(len(valid) - len(canonical)),
        "late_report_rate": float(canonical["is_late_report"].mean()) if len(canonical) else 0.0,
        "out_of_order_reports": int(canonical["out_of_order_report"].sum()) if len(canonical) else 0,
    }
    report_path = processed_dir / "validation_report.json"
    write_json(report_path, report)
    return {"canonical_events": canonical_path, "rejected_events": rejected_path, "report": report_path}

