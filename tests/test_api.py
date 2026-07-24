import math

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


VALID_PAYLOAD = {
    "date": "2025-10-31",
    "total_kg": 190741,
    "total_nominal_kg": 190631.01,
    "total_brix_units": 1931393,
    "total_hours": 58.93,
    "total_pallets": 0,
    "orders": 4,
    "avg_brix": 10.1411869,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint_returns_valid_response(client):
    response = client.post(
        "/v1/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["date"] == VALID_PAYLOAD["date"]
    assert body["modeling_version"] == "2.0"
    assert body["target"] == "active_energy_kWh"

    assert math.isfinite(body["prediction_kwh"])
    assert math.isfinite(body["post_only_prediction_kwh"])
    assert math.isfinite(body["full_history_prediction_kwh"])
    assert math.isfinite(body["branch_disagreement_kwh"])

    expected_ensemble = (
        0.7 * body["post_only_prediction_kwh"]
        + 0.3 * body["full_history_prediction_kwh"]
    )

    assert body["prediction_kwh"] == pytest.approx(
        expected_ensemble
    )

    assert body["branch_disagreement_status"] in {
        "low",
        "moderate",
        "high",
        "undefined",
    }


def test_prediction_rejects_missing_required_field(client):
    payload = VALID_PAYLOAD.copy()
    payload.pop("total_hours")

    response = client.post(
        "/v1/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_prediction_rejects_extra_field(client):
    payload = VALID_PAYLOAD.copy()
    payload["unexpected_field"] = 123

    response = client.post(
        "/v1/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_prediction_rejects_negative_input(client):
    payload = VALID_PAYLOAD.copy()
    payload["total_kg"] = -1

    response = client.post(
        "/v1/predict",
        json=payload,
    )

    assert response.status_code == 422