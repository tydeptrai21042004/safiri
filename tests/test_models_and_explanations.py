from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safiri_eta.config import load_config, project_path
from safiri_eta.models import load_default_bundle
from safiri_eta.service import predict_package


def test_default_artifacts_exist_and_predictions_are_valid():
    config = load_config()
    bundle_path = project_path(config, "artifacts_dir") / "model_bundle.joblib"
    assert bundle_path.exists(), "Run `python -m safiri_eta.cli all` before the test suite"
    bundle = load_default_bundle(config)
    snapshots = pd.read_csv(project_path(config, "processed_dir") / "snapshots.csv")
    test = snapshots.loc[snapshots["split"] == "test"].head(12)
    remaining = bundle["stage_eta"].predict_remaining(test)
    probability = bundle["risk"].predict_proba(test)
    assert len(remaining) == len(test)
    assert np.isfinite(remaining).all()
    assert (remaining >= 0).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_prediction_package_is_analyst_ready():
    config = load_config()
    bundle = load_default_bundle(config)
    snapshots = pd.read_csv(project_path(config, "processed_dir") / "snapshots.csv")
    snapshot = snapshots.loc[snapshots["split"] == "test"].iloc[12]
    result = predict_package(snapshot, bundle)
    assert result["eta"]["p10"] <= result["eta"]["p50"] <= result["eta"]["p90"]
    assert 0 <= result["risk"]["delay_probability"] <= 1
    assert result["remaining_stages"]
    assert result["drivers"]
    assert result["data_quality"]["level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["recommended_actions"]
    assert "causal" in result["explanation_note"].lower()

