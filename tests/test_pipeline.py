from __future__ import annotations

from copy import deepcopy

import pandas as pd

from safiri_eta.config import load_config
from safiri_eta.constants import CATEGORICAL_FEATURES, MILESTONES, NUMERIC_FEATURES
from safiri_eta.simulation import generate_dataset
from safiri_eta.snapshots import build_snapshots
from safiri_eta.validation import normalize_events


def prepared_config(tmp_path):
    config = deepcopy(load_config())
    config["project_root"] = str(tmp_path)
    config["paths"] = {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "artifacts_dir": "artifacts",
        "reports_dir": "reports",
    }
    generate_dataset(config, n_shipments=42)
    normalize_events(config)
    build_snapshots(config)
    return config


def test_normalization_removes_duplicates_and_preserves_lineage(tmp_path):
    config = prepared_config(tmp_path)
    canonical = pd.read_csv(tmp_path / "data/processed/events_canonical.csv")
    assert canonical["event_type"].isin(MILESTONES).all()
    assert not canonical.duplicated(["shipment_id", "event_type", "actual_time", "location_code"]).any()
    assert (pd.to_datetime(canonical["observed_at"], utc=True) >= pd.to_datetime(canonical["actual_time"], utc=True)).all()
    assert {"source_status", "source", "source_reliability", "duplicate_count"} <= set(canonical.columns)


def test_snapshots_are_point_in_time_and_group_split_is_clean(tmp_path):
    config = prepared_config(tmp_path)
    snapshots = pd.read_csv(tmp_path / "data/processed/snapshots.csv")
    events = pd.read_csv(tmp_path / "data/processed/events_canonical.csv")
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) <= set(snapshots.columns)
    assert (pd.to_datetime(snapshots["snapshot_time"], utc=True) < pd.to_datetime(snapshots["actual_delivery_time"], utc=True)).all()
    split_counts = snapshots.groupby("shipment_id")["split"].nunique()
    assert split_counts.max() == 1

    for snapshot in snapshots.sample(min(25, len(snapshots)), random_state=1).itertuples():
        known = set(snapshot.available_event_types.split(";"))
        visible = events.loc[
            (events["shipment_id"] == snapshot.shipment_id)
            & (pd.to_datetime(events["observed_at"], utc=True) <= pd.Timestamp(snapshot.snapshot_time))
        ]
        assert known == set(visible["event_type"])


def test_stage_targets_sum_to_remaining_time_with_small_numerical_tolerance(tmp_path):
    config = prepared_config(tmp_path)
    snapshots = pd.read_csv(tmp_path / "data/processed/snapshots.csv")
    stages = pd.read_csv(tmp_path / "data/processed/stage_training_rows.csv")
    summed = stages.groupby("snapshot_id")["target_stage_duration_hours"].sum()
    merged = snapshots.set_index("snapshot_id").join(summed.rename("stage_sum"))
    difference = (merged["target_remaining_hours"] - merged["stage_sum"]).abs()
    assert difference.max() < 1e-6

