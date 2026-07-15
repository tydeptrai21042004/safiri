from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import project_path
from .models import load_default_bundle
from .utils import write_json


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    absolute = np.abs(actual - predicted)
    return {
        "mae_hours": float(mean_absolute_error(actual, predicted)),
        "rmse_hours": float(np.sqrt(mean_squared_error(actual, predicted))),
        "median_ae_hours": float(np.median(absolute)),
        "p90_ae_hours": float(np.quantile(absolute, 0.90)),
        "within_6_hours": float(np.mean(absolute <= 6)),
        "within_12_hours": float(np.mean(absolute <= 12)),
    }


def _risk_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    predicted = (probability >= 0.50).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)) if len(np.unique(actual)) > 1 else None,
        "brier_score": float(brier_score_loss(actual, probability)),
        "confusion_matrix": matrix.tolist(),
    }


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def _make_plots(metrics: dict[str, Any], test: pd.DataFrame, predictions: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    model_names = ["Baseline", "Direct ETA", "Stage-aware ETA"]
    values = [metrics["eta"][key]["mae_hours"] for key in ("baseline", "direct", "stage_aware")]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    bars = ax.bar(model_names, values, color=["#9CA3AF", "#4F81BD", "#173F5F"])
    ax.set_ylabel("MAE (hours)")
    ax.set_title("Final ETA error on chronological test set", loc="left", fontweight="bold")
    _style_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:.2f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "eta_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    stage_order = test.groupby("current_stage")["event_index"].median().sort_values().index
    dynamic = predictions.assign(current_stage=test["current_stage"].values).groupby("current_stage")["absolute_error"].mean().reindex(stage_order)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(dynamic.index, dynamic.values, marker="o", color="#173F5F", linewidth=2.2)
    ax.set_ylabel("MAE (hours)")
    ax.set_title("ETA error as milestones become available", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=22)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(figures / "dynamic_eta_by_milestone.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    missing_levels = list(metrics["robustness_missing_events"].keys())
    missing_mae = [metrics["robustness_missing_events"][key]["mae_hours"] for key in missing_levels]
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.plot(missing_levels, missing_mae, marker="o", color="#B26A00", linewidth=2.2)
    ax.set_xlabel("Stress scenario")
    ax.set_ylabel("MAE (hours)")
    ax.set_title("Sensitivity to missing-event indicators", loc="left", fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(figures / "missing_event_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = project_path(config, "processed_dir")
    reports_dir = project_path(config, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_default_bundle(config)
    snapshots = pd.read_csv(processed_dir / "snapshots.csv")
    test = snapshots.loc[snapshots["split"] == "test"].reset_index(drop=True)
    actual = test["target_remaining_hours"].to_numpy(dtype=float)
    baseline_prediction = bundle["baseline"].predict_remaining(test)
    direct_prediction = bundle["direct_eta"].predict_remaining(test)
    stage_prediction = bundle["stage_eta"].predict_remaining(test)
    probability = bundle["risk"].predict_proba(test)

    robustness: dict[str, dict[str, float]] = {}
    for label, extra_missing in (("observed", 0), ("missing_10pct", 1), ("missing_20pct", 2), ("missing_30pct", 3)):
        stressed = test.copy()
        stressed["missing_event_count"] = stressed["missing_event_count"] + extra_missing
        stressed["source_reliability"] = np.clip(stressed["source_reliability"] - extra_missing * 0.035, 0, 1)
        robustness[label] = _regression_metrics(actual, bundle["stage_eta"].predict_remaining(stressed))

    metrics = {
        "dataset": {
            "test_snapshots": int(len(test)),
            "test_shipments": int(test["shipment_id"].nunique()),
            "delay_prevalence": float(test["is_delayed"].mean()),
        },
        "eta": {
            "baseline": _regression_metrics(actual, baseline_prediction),
            "direct": _regression_metrics(actual, direct_prediction),
            "stage_aware": _regression_metrics(actual, stage_prediction),
        },
        "delay_risk": _risk_metrics(test["is_delayed"].to_numpy(dtype=int), probability),
        "robustness_missing_events": robustness,
        "interval": {
            "residual_q10_hours": float(bundle["stage_eta"].residual_q10),
            "residual_q90_hours": float(bundle["stage_eta"].residual_q90),
        },
        "evaluation_protocol": "Chronological shipment-level 70/15/15 split; test set untouched during training.",
    }
    predictions = pd.DataFrame(
        {
            "snapshot_id": test["snapshot_id"],
            "shipment_id": test["shipment_id"],
            "actual_remaining_hours": actual,
            "baseline_remaining_hours": baseline_prediction,
            "direct_remaining_hours": direct_prediction,
            "stage_remaining_hours": stage_prediction,
            "delay_probability": probability,
            "is_delayed": test["is_delayed"],
            "absolute_error": np.abs(actual - stage_prediction),
        }
    )
    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    write_json(reports_dir / "metrics.json", metrics)
    _make_plots(metrics, test, predictions, reports_dir / "figures")
    return metrics

