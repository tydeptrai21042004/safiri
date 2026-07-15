from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app, resources


client = TestClient(app)


def test_health_and_metrics_endpoints():
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["shipments"] == 300
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "stage_aware" in metrics.json()["eta"]


def test_timeline_prediction_and_scenario_endpoints():
    shipment_id = resources()["snapshots"]["shipment_id"].iloc[-1]
    timeline = client.get(f"/shipments/{shipment_id}/timeline")
    assert timeline.status_code == 200
    assert len(timeline.json()) >= 2
    prediction = client.get(f"/shipments/{shipment_id}/prediction")
    assert prediction.status_code == 200
    assert "analyst_summary" in prediction.json()
    scenario = client.post(
        "/simulate",
        json={"shipment_id": shipment_id, "congestion_level": 0.95, "documents_ready": 0},
    )
    assert scenario.status_code == 200
    assert scenario.json()["scenario_overrides"]["congestion_level"] == 0.95

