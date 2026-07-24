import math

import pytest

from src.serving.prediction_service import (
    PredictionService,
    classify_disagreement,
)


def valid_raw_input() -> dict[str, object]:
    """
    General valid request used for prediction-math testing.

    Some values intentionally sit outside the development reference.
    That does not prevent the model from producing a prediction.
    """

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


def supported_raw_input() -> dict[str, object]:
    """
    Request whose operational and calendar values are represented
    by the saved development reference.
    """

    return {
        "date": "2025-10-10",
        "total_kg": 190741.0,
        "total_nominal_kg": 190631.01,
        "total_brix_units": 1931393.0,
        "total_hours": 58.93,
        "total_pallets": 0.0,
        "orders": 4,
        "avg_brix": 10.1411869,
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

    expected_pct = (
        expected_kwh
        / component_mean
        * 100.0
    )

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


def test_supported_input_receives_support_diagnostics() -> None:
    service = PredictionService()

    result = service.predict(
        supported_raw_input()
    )

    assert (
        result.operational_range_status
        == "inside_typical_development_range"
    )

    assert (
        result.calendar_coverage_status
        == "represented"
    )

    assert result.tail_features == ()
    assert result.outside_range_features == ()
    assert result.unseen_calendar_features == ()

    if result.branch_disagreement_status == "moderate":
        expected_warnings = (
            "MODERATE_BRANCH_DISAGREEMENT",
        )

    elif result.branch_disagreement_status == "high":
        expected_warnings = (
            "HIGH_BRANCH_DISAGREEMENT",
        )

    else:
        expected_warnings = ()

    assert result.warning_codes == expected_warnings


def test_outside_and_unseen_input_receives_warnings() -> None:
    service = PredictionService()

    raw_input = supported_raw_input()

    # total_pallets was always zero during development.
    raw_input["total_pallets"] = 1.0

    # January 2026 was not represented in development.
    raw_input["date"] = "2026-01-05"

    result = service.predict(raw_input)

    assert (
        result.operational_range_status
        == "outside_observed_range"
    )

    assert result.outside_range_features == (
        "total_pallets",
    )

    assert (
        result.calendar_coverage_status
        == "contains_unseen_calendar_values"
    )

    assert result.unseen_calendar_features == (
        "month=1",
        "week_of_year=2",
        "year=2026",
    )

    assert (
        "OUTSIDE_OBSERVED_RANGE"
        in result.warning_codes
    )

    assert (
        "UNSEEN_CALENDAR_VALUE"
        in result.warning_codes
    )