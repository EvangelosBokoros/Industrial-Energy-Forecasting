import json
import math

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import (
    text_string_to_metric_families,
)

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


def _metric_value(
    metrics_text: str,
    sample_name: str,
    labels: dict[str, str],
) -> float:
    """Return one Prometheus sample value or zero when absent."""

    for family in text_string_to_metric_families(
        metrics_text
    ):
        for sample in family.samples:
            if (
                sample.name == sample_name
                and all(
                    sample.labels.get(label_name)
                    == label_value
                    for label_name, label_value
                    in labels.items()
                )
            ):
                return float(sample.value)

    return 0.0


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_exposes_service_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers[
        "content-type"
    ].startswith("text/plain")

    body = response.text

    assert "jmm_http_requests_total" in body
    assert (
        "jmm_http_request_duration_seconds"
        in body
    )
    assert "jmm_prediction_records_total" in body
    assert "jmm_operational_range_total" in body
    assert "jmm_calendar_coverage_total" in body
    assert "jmm_branch_disagreement_total" in body
    assert "jmm_warning_codes_total" in body


def test_prediction_updates_service_metrics(client):
    before = client.get("/metrics").text

    before_prediction_count = _metric_value(
        before,
        "jmm_prediction_records_total",
        {"endpoint": "single"},
    )
    before_request_count = _metric_value(
        before,
        "jmm_http_requests_total",
        {
            "method": "POST",
            "path": "/v1/predict",
            "status_code": "200",
        },
    )
    before_operational_count = _metric_value(
        before,
        "jmm_operational_range_total",
        {
            "status": (
                "inside_typical_development_range"
            )
        },
    )
    before_disagreement_count = _metric_value(
        before,
        "jmm_branch_disagreement_total",
        {"status": "moderate"},
    )
    before_calendar_warning_count = _metric_value(
        before,
        "jmm_warning_codes_total",
        {"code": "UNSEEN_CALENDAR_VALUE"},
    )

    prediction_response = client.post(
        "/v1/predict",
        json=VALID_PAYLOAD,
    )

    assert prediction_response.status_code == 200
    assert (
        prediction_response.json()[
            "branch_disagreement_status"
        ]
        == "moderate"
    )

    after = client.get("/metrics").text

    assert _metric_value(
        after,
        "jmm_prediction_records_total",
        {"endpoint": "single"},
    ) == pytest.approx(
        before_prediction_count + 1.0
    )

    assert _metric_value(
        after,
        "jmm_http_requests_total",
        {
            "method": "POST",
            "path": "/v1/predict",
            "status_code": "200",
        },
    ) == pytest.approx(
        before_request_count + 1.0
    )

    assert _metric_value(
        after,
        "jmm_operational_range_total",
        {
            "status": (
                "inside_typical_development_range"
            )
        },
    ) == pytest.approx(
        before_operational_count + 1.0
    )

    assert _metric_value(
        after,
        "jmm_branch_disagreement_total",
        {"status": "moderate"},
    ) == pytest.approx(
        before_disagreement_count + 1.0
    )

    assert _metric_value(
        after,
        "jmm_warning_codes_total",
        {"code": "UNSEEN_CALENDAR_VALUE"},
    ) == pytest.approx(
        before_calendar_warning_count + 1.0
    )


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
    assert math.isfinite(
        body["post_only_prediction_kwh"]
    )
    assert math.isfinite(
        body["full_history_prediction_kwh"]
    )
    assert math.isfinite(
        body["branch_disagreement_kwh"]
    )

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

    assert (
        body["operational_range_status"]
        == "inside_typical_development_range"
    )

    # The development reference contains ISO weeks 37-43.
    # October 31, 2025 belongs to ISO week 44.
    assert (
        body["calendar_coverage_status"]
        == "contains_unseen_calendar_values"
    )

    assert body["tail_features"] == []
    assert body["outside_range_features"] == []

    assert body["unseen_calendar_features"] == [
        "week_of_year=44"
    ]

    assert (
        "UNSEEN_CALENDAR_VALUE"
        in body["warning_codes"]
    )


def test_prediction_returns_outside_range_diagnostics(
    client,
):
    payload = VALID_PAYLOAD.copy()

    # Pallets was constant at zero during development.
    payload["total_pallets"] = 1

    # January 2026 introduces an unseen month, year, and ISO week.
    payload["date"] = "2026-01-05"

    response = client.post(
        "/v1/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["operational_range_status"]
        == "outside_observed_range"
    )

    assert body["outside_range_features"] == [
        "total_pallets"
    ]

    assert (
        body["calendar_coverage_status"]
        == "contains_unseen_calendar_values"
    )

    assert body["unseen_calendar_features"] == [
        "month=1",
        "week_of_year=2",
        "year=2026",
    ]

    assert (
        "OUTSIDE_OBSERVED_RANGE"
        in body["warning_codes"]
    )

    assert (
        "UNSEEN_CALENDAR_VALUE"
        in body["warning_codes"]
    )

    # Diagnostics warn about support but do not block prediction.
    assert math.isfinite(body["prediction_kwh"])


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
        "service": "Industrial Energy Forecasting API",
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
        assert math.isfinite(
            prediction["prediction_kwh"]
        )

        assert prediction[
            "operational_range_status"
        ] in {
            "inside_typical_development_range",
            "development_distribution_tail",
            "outside_observed_range",
        }

        assert prediction[
            "calendar_coverage_status"
        ] in {
            "represented",
            "contains_unseen_calendar_values",
        }

        assert isinstance(
            prediction["tail_features"],
            list,
        )
        assert isinstance(
            prediction["outside_range_features"],
            list,
        )
        assert isinstance(
            prediction["unseen_calendar_features"],
            list,
        )
        assert isinstance(
            prediction["warning_codes"],
            list,
        )


def test_batch_prediction_rejects_empty_records(client):
    response = client.post(
        "/v1/predict/batch",
        json={"records": []},
    )

    assert response.status_code == 422


def test_batch_prediction_rejects_invalid_nested_record(
    client,
):
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


def test_batch_prediction_rejects_more_than_500_records(
    client,
):
    records = [
        VALID_PAYLOAD.copy()
        for _ in range(501)
    ]

    response = client.post(
        "/v1/predict/batch",
        json={"records": records},
    )

    assert response.status_code == 422


def test_response_contains_generated_request_id(client):
    response = client.get("/health")

    assert response.status_code == 200

    request_id = response.headers.get(
        "X-Request-ID"
    )

    assert request_id is not None
    assert request_id != ""


def test_valid_caller_request_id_is_preserved(client):
    response = client.get(
        "/health",
        headers={
            "X-Request-ID": "portfolio-test-123",
        },
    )

    assert response.status_code == 200

    assert response.headers["X-Request-ID"] == (
        "portfolio-test-123"
    )


def test_request_is_logged_as_structured_json(
    client,
    caplog,
):
    caplog.set_level(
        "INFO",
        logger="jmm.api",
    )

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": "structured-log-test",
        },
    )

    assert response.status_code == 200

    matching_records = [
        record
        for record in caplog.records
        if (
            record.name == "jmm.api"
            and "structured-log-test"
            in record.getMessage()
        )
    ]

    assert matching_records

    log_record = json.loads(
        matching_records[-1].getMessage()
    )

    assert log_record["event"] == (
        "request_completed"
    )
    assert log_record["request_id"] == (
        "structured-log-test"
    )
    assert log_record["method"] == "GET"
    assert log_record["path"] == "/health"
    assert log_record["status_code"] == 200
    assert log_record["duration_ms"] >= 0


def test_unhandled_error_returns_safe_response(
    client,
    monkeypatch,
):
    def raise_internal_error(
        raw_input,
    ):
        raise RuntimeError(
            "Sensitive internal implementation detail"
        )

    monkeypatch.setattr(
        client.app.state.prediction_service,
        "predict",
        raise_internal_error,
    )

    response = client.post(
        "/v1/predict",
        json=VALID_PAYLOAD,
        headers={
            "X-Request-ID": "failure-test-001",
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Internal server error",
        "request_id": "failure-test-001",
    }

    assert response.headers["X-Request-ID"] == (
        "failure-test-001"
    )

    assert (
        "Sensitive internal implementation detail"
        not in response.text
    )
