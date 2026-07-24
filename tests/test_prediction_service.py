import math

import pytest

from src.serving.prediction_service import (
    PredictionService,
    classify_disagreement,
)


def valid_raw_input() -> dict[str, object]:
    return {
        "date": "2025-10-31",
        "total_kg": 185000.0,
        "total_nominal_kg": 190000.0,
        "total_brix_units": 25000.0,
        "total_hours": 20.0,
        "total_pallets": 120.0,
        "orders": 25,
        "avg_brix": 13.5,
    }


def test_prediction_matches_saved_ensemble_weights() -> None:
    service = PredictionService()
    result = service.predict(valid_raw_input())

    expected_prediction = (
        0.7 * result.post_only_prediction_kwh
        + 0.3 * result.full_history_prediction_kwh
    )

    assert result.prediction_kwh == pytest.approx(
        expected_prediction
    )


def test_branch_disagreement_calculation_is_correct() -> None:
    service = PredictionService()
    result = service.predict(valid_raw_input())

    expected_kwh = abs(
        result.post_only_prediction_kwh
        - result.full_history_prediction_kwh
    )

    component_mean = (
        abs(result.post_only_prediction_kwh)
        + abs(result.full_history_prediction_kwh)
    ) / 2.0

    expected_pct = expected_kwh / component_mean * 100.0

    assert result.branch_disagreement_kwh == pytest.approx(
        expected_kwh
    )
    assert result.branch_disagreement_pct == pytest.approx(
        expected_pct
    )
    assert math.isfinite(result.prediction_kwh)


def test_disagreement_thresholds() -> None:
    assert classify_disagreement(9.99) == "low"
    assert classify_disagreement(10.0) == "moderate"
    assert classify_disagreement(19.99) == "moderate"
    assert classify_disagreement(20.0) == "high"
    assert classify_disagreement(None) == "undefined"