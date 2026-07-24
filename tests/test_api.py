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


def test_root_endpoint_returns_service_navigation(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "Damavand Energy Forecasting API",
        "api_version": "1.0.0",
        "documentation_url": "/docs",
        "health_url": "/health",
        "readiness_url": "/ready",
        "model_url": "/v1/model",
    }


def test_readiness_endpoint_confirms_model_is_loaded(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "modeling_version": "2.0",
        "target": "active_energy_kWh",
    }


def test_model_information_endpoint_returns_active_ensemble(
    client,
):
    response = client.get("/v1/model")

    assert response.status_code == 200

    body = response.json()

    assert body["modeling_version"] == "2.0"
    assert body["ensemble_type"] == (
        "constrained_weighted_prediction_average"
    )
    assert body["target"] == "active_energy_kWh"

    assert set(body["components"]) == {
        "post_only",
        "full_history",
    }

    post_only = body["components"]["post_only"]

    assert post_only["estimator_class"] == (
        "ExtraTreesRegressor"
    )
    assert post_only["weight"] == pytest.approx(0.7)
    assert post_only["features"] == [
        "total_kg",
        "total_nominal_kg",
        "total_brix_units",
        "total_hours",
        "total_pallets",
        "orders",
        "avg_brix",
        "yield_ratio_actual_over_nominal",
        "weekday",
        "is_weekend",
        "month",
        "year",
        "week_of_year",
    ]

    full_history = body["components"]["full_history"]

    assert full_history["estimator_class"] == (
        "AdaBoostRegressor"
    )
    assert full_history["weight"] == pytest.approx(0.3)
    assert full_history["features"] == [
        "total_kg",
        "total_hours",
        "total_pallets",
        "orders",
        "avg_brix",
        "yield_ratio_actual_over_nominal",
        "weekday",
        "is_weekend",
        "month",
        "year",
    ]


def test_batch_prediction_returns_ordered_results(client):
    first_record = VALID_PAYLOAD.copy()

    second_record = VALID_PAYLOAD.copy()
    second_record["date"] = "2025-10-30"
    second_record["total_kg"] = 180000
    second_record["total_nominal_kg"] = 181000
    second_record["total_brix_units"] = 1800000

    response = client.post(
        "/v1/predict/batch",
        json={
            "records": [
                first_record,
                second_record,
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 2
    assert len(body["predictions"]) == 2

    assert body["predictions"][0]["date"] == (
        first_record["date"]
    )
    assert body["predictions"][1]["date"] == (
        second_record["date"]
    )

    for prediction in body["predictions"]:
        assert prediction["modeling_version"] == "2.0"
        assert prediction["target"] == "active_energy_kWh"
        assert math.isfinite(prediction["prediction_kwh"])


def test_batch_prediction_rejects_empty_records(client):
    response = client.post(
        "/v1/predict/batch",
        json={"records": []},
    )

    assert response.status_code == 422


def test_batch_prediction_rejects_invalid_nested_record(client):
    valid_record = VALID_PAYLOAD.copy()

    invalid_record = VALID_PAYLOAD.copy()
    invalid_record["total_hours"] = -1

    response = client.post(
        "/v1/predict/batch",
        json={
            "records": [
                valid_record,
                invalid_record,
            ]
        },
    )

    assert response.status_code == 422


def test_batch_prediction_rejects_more_than_500_records(client):
    records = [
        VALID_PAYLOAD.copy()
        for _ in range(501)
    ]

    response = client.post(
        "/v1/predict/batch",
        json={"records": records},
    )

    assert response.status_code == 422