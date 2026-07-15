from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(os.getenv("SAFIRI_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safiri_eta.config import load_config, project_path  # noqa: E402
from safiri_eta.models import load_default_bundle  # noqa: E402
from safiri_eta.service import predict_package  # noqa: E402
from safiri_eta.utils import read_json  # noqa: E402


st.set_page_config(page_title="SAFiRi ETA Lab", page_icon="🚚", layout="wide")


@st.cache_resource
def load_resources():
    config = load_config(ROOT / "configs" / "default.yaml")
    processed = project_path(config, "processed_dir")
    raw = project_path(config, "raw_dir")
    reports = project_path(config, "reports_dir")
    return {
        "config": config,
        "bundle": load_default_bundle(config),
        "snapshots": pd.read_csv(processed / "snapshots.csv"),
        "events": pd.read_csv(processed / "events_canonical.csv"),
        "shipments": pd.read_csv(raw / "shipments.csv"),
        "metrics": read_json(reports / "metrics.json"),
    }


data = load_resources()
shipments = data["shipments"]
snapshots = data["snapshots"]
events = data["events"]
metrics = data["metrics"]

st.title("SAFiRi Milestone-Aware ETA Lab")
st.caption("Synthetic research prototype - dynamic ETA, delay risk, stage explanations and data-quality exceptions")

stage_mae = metrics["eta"]["stage_aware"]["mae_hours"]
direct_mae = metrics["eta"]["direct"]["mae_hours"]
pr_auc = metrics["delay_risk"]["pr_auc"]
columns = st.columns(4)
columns[0].metric("Shipments", f"{len(shipments):,}")
columns[1].metric("Direct ETA MAE", f"{direct_mae:.2f} h")
columns[2].metric("Stage ETA MAE", f"{stage_mae:.2f} h")
columns[3].metric("Delay PR-AUC", f"{pr_auc:.3f}")

st.divider()
left, right = st.columns([1, 2.2])
with left:
    st.subheader("Shipment selection")
    shipment_ids = sorted(snapshots["shipment_id"].unique())
    default_index = max(0, len(shipment_ids) - 8)
    shipment_id = st.selectbox("Shipment", shipment_ids, index=default_index)
    shipment_snapshots = snapshots.loc[snapshots["shipment_id"] == shipment_id].sort_values("snapshot_time")
    snapshot_labels = [f"{row.current_stage} - {row.snapshot_time}" for row in shipment_snapshots.itertuples()]
    selected_label = st.selectbox("Prediction snapshot", snapshot_labels, index=len(snapshot_labels) - 1)
    snapshot = shipment_snapshots.iloc[snapshot_labels.index(selected_label)].copy()

    st.subheader("What-if scenario")
    congestion = st.slider("Congestion", 0.0, 1.0, float(snapshot["congestion_level"]), 0.05)
    weather = st.slider("Weather severity", 0.0, 1.0, float(snapshot["weather_severity"]), 0.05)
    documents = st.toggle("Documents ready", value=bool(snapshot["documents_ready"]))
    current_delay = st.slider("Current delay (hours)", -12.0, 48.0, float(snapshot["current_delay_hours"]), 1.0)
    missing = st.slider("Missing milestones", 0, 5, int(snapshot["missing_event_count"]), 1)
    snapshot["congestion_level"] = congestion
    snapshot["weather_severity"] = weather
    snapshot["documents_ready"] = int(documents)
    snapshot["current_delay_hours"] = current_delay
    snapshot["missing_event_count"] = missing

result = predict_package(snapshot, data["bundle"])

with right:
    st.subheader("Current prediction")
    eta = result["eta"]
    risk = result["risk"]
    cards = st.columns(4)
    cards[0].metric("ETA P50", eta["p50"].replace("T", " ").replace("Z", " UTC"))
    cards[1].metric("Predicted delay", f"{eta['predicted_delay_hours']:+.1f} h")
    cards[2].metric("Delay probability", f"{risk['delay_probability']:.0%}")
    cards[3].metric("Data quality", result["data_quality"]["level"], f"{result['data_quality']['score']:.0%}")
    st.info(result["analyst_summary"])

    stage_frame = pd.DataFrame(result["remaining_stages"])
    if not stage_frame.empty:
        stage_long = stage_frame.melt(
            id_vars="stage",
            value_vars=["predicted_hours", "historical_median_hours"],
            var_name="series",
            value_name="hours",
        )
        fig = px.bar(
            stage_long,
            x="stage",
            y="hours",
            color="series",
            barmode="group",
            title="Remaining-stage duration",
            labels={"stage": "Stage", "hours": "Hours", "series": "Estimate"},
            color_discrete_map={"predicted_hours": "#173F5F", "historical_median_hours": "#9CA3AF"},
        )
        fig.update_layout(legend_orientation="h", legend_y=1.12, margin=dict(t=70, b=30))
        st.plotly_chart(fig, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Timeline", "Explanation", "Exceptions and actions"])
with tab1:
    timeline = events.loc[events["shipment_id"] == shipment_id].sort_values("event_index")
    st.dataframe(
        timeline[["event_type", "planned_time", "actual_time", "observed_at", "source", "reporting_lag_hours"]],
        use_container_width=True,
        hide_index=True,
    )
with tab2:
    driver_frame = pd.DataFrame(result["drivers"])
    if not driver_frame.empty:
        driver_frame["signed_hours"] = driver_frame["contribution_hours"]
        fig = px.bar(
            driver_frame.sort_values("signed_hours"),
            x="signed_hours",
            y="label",
            orientation="h",
            color="direction",
            title="Local model sensitivity",
            labels={"signed_hours": "Contribution to remaining time (hours)", "label": "Feature"},
            color_discrete_map={"LATER": "#C2413B", "EARLIER": "#2E7D32"},
        )
        st.plotly_chart(fig, use_container_width=True)
    st.caption(result["explanation_note"])
with tab3:
    if result["exceptions"]:
        st.dataframe(pd.DataFrame(result["exceptions"]), use_container_width=True, hide_index=True)
    else:
        st.success("No exception rules were triggered.")
    st.markdown("**Recommended actions**")
    for action in result["recommended_actions"]:
        st.markdown(f"- {action}")
    if result["data_quality"]["issues"]:
        st.markdown("**Data-quality issues**")
        for issue in result["data_quality"]["issues"]:
            st.markdown(f"- {issue}")

