from __future__ import annotations

from copy import deepcopy

import pandas as pd

from safiri_eta.config import load_config
from safiri_eta.simulation import generate_dataset


def temporary_config(tmp_path):
    config = deepcopy(load_config())
    config["project_root"] = str(tmp_path)
    config["paths"] = {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "artifacts_dir": "artifacts",
        "reports_dir": "reports",
    }
    return config


def test_generator_is_reproducible_and_has_complete_ground_truth(tmp_path):
    config = temporary_config(tmp_path)
    paths = generate_dataset(config, n_shipments=24)
    first = pd.read_csv(paths["shipments"])
    first_truth = pd.read_csv(paths["truth_events"])
    paths = generate_dataset(config, n_shipments=24)
    second = pd.read_csv(paths["shipments"])
    pd.testing.assert_frame_equal(first, second)
    assert first["shipment_id"].nunique() == 24
    assert len(first_truth) == 24 * 8
    assert set(first["mode"]) >= {"SEA", "AIR", "ROAD"}
    assert first["is_delayed"].isin([0, 1]).all()


def test_raw_events_include_reporting_and_operational_context(tmp_path):
    config = temporary_config(tmp_path)
    paths = generate_dataset(config, n_shipments=20)
    raw = pd.read_csv(paths["raw_events"])
    required = {
        "source_event_id",
        "shipment_id",
        "source_status",
        "actual_time",
        "observed_at",
        "congestion_level",
        "weather_severity",
        "documents_ready",
        "source_reliability",
    }
    assert required <= set(raw.columns)
    assert raw["congestion_level"].between(0, 1).all()
    assert raw["weather_severity"].between(0, 1).all()

