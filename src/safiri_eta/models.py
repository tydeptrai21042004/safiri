from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import project_path
from .constants import (
    CATEGORICAL_FEATURES,
    MILESTONES,
    NUMERIC_FEATURES,
    STAGE_CATEGORICAL_FEATURES,
    STAGE_NUMERIC_FEATURES,
)


def _preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _regression_pipeline(config: dict[str, Any], numeric: list[str], categorical: list[str]) -> Pipeline:
    params = config["model"]
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(numeric, categorical)),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=float(params["learning_rate"]),
                    max_iter=int(params["max_iter"]),
                    max_leaf_nodes=int(params["max_leaf_nodes"]),
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    l2_regularization=float(params["l2_regularization"]),
                    random_state=int(params["random_state"]),
                ),
            ),
        ]
    )


def _classification_pipeline(config: dict[str, Any]) -> Pipeline:
    params = config["model"]
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            (
                "model",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=float(params["learning_rate"]),
                    max_iter=int(params["max_iter"]),
                    max_leaf_nodes=int(params["max_leaf_nodes"]),
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    l2_regularization=float(params["l2_regularization"]),
                    random_state=int(params["random_state"]),
                ),
            ),
        ]
    )


def reference_values(frame: pd.DataFrame) -> dict[str, Any]:
    references: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        references[feature] = float(pd.to_numeric(frame[feature], errors="coerce").median())
    for feature in CATEGORICAL_FEATURES:
        mode = frame[feature].mode(dropna=True)
        references[feature] = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
    return references


@dataclass
class BaselineStageModel:
    route_stage_median: dict[tuple[str, str], float]
    stage_median: dict[str, float]
    global_median: float

    @classmethod
    def fit(cls, stage_rows: pd.DataFrame) -> "BaselineStageModel":
        target = "target_stage_duration_hours"
        route_stage = stage_rows.groupby(["route", "target_stage"])[target].median().to_dict()
        stage = stage_rows.groupby("target_stage")[target].median().to_dict()
        return cls(route_stage, stage, float(stage_rows[target].median()))

    def stage_duration(self, route: str, target_stage: str) -> float:
        return float(
            self.route_stage_median.get(
                (route, target_stage),
                self.stage_median.get(target_stage, self.global_median),
            )
        )

    def predict_remaining(self, snapshots: pd.DataFrame) -> np.ndarray:
        predictions: list[float] = []
        for row in snapshots.itertuples(index=False):
            current_index = int(row.event_index)
            remaining = sum(
                self.stage_duration(str(row.route), MILESTONES[index])
                for index in range(current_index + 1, len(MILESTONES))
            )
            remaining = max(0.0, remaining - float(row.event_age_hours))
            predictions.append(remaining)
        return np.asarray(predictions, dtype=float)


class StageETAModel:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.residual_q10 = 0.0
        self.residual_q90 = 0.0

    def fit(self, stage_rows: pd.DataFrame) -> "StageETAModel":
        features = STAGE_NUMERIC_FEATURES + STAGE_CATEGORICAL_FEATURES
        self.pipeline.fit(stage_rows[features], stage_rows["target_stage_duration_hours"])
        return self

    @staticmethod
    def expand_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row_number, (_, snapshot) in enumerate(snapshots.reset_index(drop=True).iterrows()):
            current_index = int(snapshot["event_index"])
            for target_index in range(current_index + 1, len(MILESTONES)):
                payload = snapshot.to_dict()
                payload["_row_number"] = row_number
                payload["target_stage"] = MILESTONES[target_index]
                payload["target_stage_index"] = target_index
                payload["is_first_remaining_stage"] = int(target_index == current_index + 1)
                rows.append(payload)
        return pd.DataFrame(rows)

    def predict_breakdown(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        expanded = self.expand_snapshots(snapshots)
        if expanded.empty:
            return expanded.assign(predicted_stage_hours=pd.Series(dtype=float))
        features = STAGE_NUMERIC_FEATURES + STAGE_CATEGORICAL_FEATURES
        expanded["predicted_stage_hours"] = np.clip(self.pipeline.predict(expanded[features]), 0.0, None)
        return expanded

    def predict_remaining(self, snapshots: pd.DataFrame) -> np.ndarray:
        expanded = self.predict_breakdown(snapshots)
        if expanded.empty:
            return np.zeros(len(snapshots), dtype=float)
        grouped = expanded.groupby("_row_number")["predicted_stage_hours"].sum()
        return grouped.reindex(range(len(snapshots)), fill_value=0.0).to_numpy(dtype=float)

    def fit_residual_interval(self, validation_snapshots: pd.DataFrame) -> None:
        prediction = self.predict_remaining(validation_snapshots)
        residual = validation_snapshots["target_remaining_hours"].to_numpy(dtype=float) - prediction
        self.residual_q10 = float(np.quantile(residual, 0.10))
        self.residual_q90 = float(np.quantile(residual, 0.90))

    def interval(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.clip(prediction + self.residual_q10, 0.0, None)
        upper = np.clip(prediction + self.residual_q90, lower, None)
        return lower, upper


class DirectETAModel:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def fit(self, snapshots: pd.DataFrame) -> "DirectETAModel":
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.pipeline.fit(snapshots[features], snapshots["target_remaining_hours"])
        return self

    def predict_remaining(self, snapshots: pd.DataFrame) -> np.ndarray:
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        return np.clip(self.pipeline.predict(snapshots[features]), 0.0, None)


class DelayRiskModel:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.calibrator: LogisticRegression | None = None

    def fit(self, snapshots: pd.DataFrame) -> "DelayRiskModel":
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        target = snapshots["is_delayed"].to_numpy(dtype=int)
        positives = max(int(target.sum()), 1)
        negatives = max(int((1 - target).sum()), 1)
        positive_weight = negatives / positives
        weights = np.where(target == 1, positive_weight, 1.0)
        self.pipeline.fit(snapshots[features], target, model__sample_weight=weights)
        return self

    def fit_calibrator(self, validation: pd.DataFrame) -> None:
        y = validation["is_delayed"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            self.calibrator = None
            return
        raw = self.pipeline.predict_proba(validation[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        calibrator = LogisticRegression(random_state=0)
        calibrator.fit(raw.reshape(-1, 1), y)
        self.calibrator = calibrator

    def predict_proba(self, snapshots: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(snapshots[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        if self.calibrator is None:
            return np.clip(raw, 0.0, 1.0)
        return np.clip(self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], 0.0, 1.0)


def train_models(config: dict[str, Any]) -> Path:
    processed_dir = project_path(config, "processed_dir")
    artifacts_dir = project_path(config, "artifacts_dir")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    snapshots = pd.read_csv(processed_dir / "snapshots.csv")
    stages = pd.read_csv(processed_dir / "stage_training_rows.csv")
    train_snapshots = snapshots.loc[snapshots["split"] == "train"].copy()
    validation_snapshots = snapshots.loc[snapshots["split"] == "validation"].copy()
    train_stages = stages.loc[stages["split"] == "train"].copy()

    baseline = BaselineStageModel.fit(train_stages)
    stage_eta = StageETAModel(
        _regression_pipeline(config, STAGE_NUMERIC_FEATURES, STAGE_CATEGORICAL_FEATURES)
    ).fit(train_stages)
    stage_eta.fit_residual_interval(validation_snapshots)
    direct_eta = DirectETAModel(
        _regression_pipeline(config, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    ).fit(train_snapshots)
    risk = DelayRiskModel(_classification_pipeline(config)).fit(train_snapshots)
    risk.fit_calibrator(validation_snapshots)

    bundle = {
        "version": "1.0.0",
        "baseline": baseline,
        "stage_eta": stage_eta,
        "direct_eta": direct_eta,
        "risk": risk,
        "reference_values": reference_values(train_snapshots),
        "delay_threshold_hours": float(config["model"]["delay_threshold_hours"]),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "milestones": MILESTONES,
    }
    output = artifacts_dir / "model_bundle.joblib"
    joblib.dump(bundle, output)
    return output


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(path))


def load_default_bundle(config: dict[str, Any]) -> dict[str, Any]:
    return load_bundle(project_path(config, "artifacts_dir") / "model_bundle.joblib")

