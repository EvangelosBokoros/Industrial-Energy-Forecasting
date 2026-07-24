from datetime import date

import pytest

from src.serving.feature_builder import (
    build_feature_values,
    build_model_feature_frames,
)
from src.serving.model_loader import load_model_bundle


def valid_raw_input() -> dict[str, object]:
    return {
        "date": "2025-10-18",
        "total_kg": 1100.0,
        "total_nominal_kg": 1000.0,
        "total_brix_units": 150.0,
        "total_hours": 10.0,
        "total_pallets": 20.0,
        "orders": 5,
        "avg_brix": 13.5,
    }


def test_build_feature_values_calculates_derived_features() -> None:
    values = build_feature_values(valid_raw_input())

    expected_date = date(2025, 10, 18)

    assert values["yield_ratio_actual_over_nominal"] == pytest.approx(1.1)
    assert values["weekday"] == expected_date.weekday()
    assert values["is_weekend"] == 1
    assert values["month"] == 10
    assert values["year"] == 2025
    assert values["week_of_year"] == expected_date.isocalendar().week
    assert values["orders"] == 5


def test_zero_nominal_weight_is_rejected() -> None:
    raw_input = valid_raw_input()
    raw_input["total_nominal_kg"] = 0

    with pytest.raises(
        ValueError,
        match="total_nominal_kg must be greater than zero",
    ):
        build_feature_values(raw_input)


def test_non_integer_orders_are_rejected() -> None:
    raw_input = valid_raw_input()
    raw_input["orders"] = 5.5

    with pytest.raises(
        ValueError,
        match="orders must be a nonnegative integer count",
    ):
        build_feature_values(raw_input)


def test_missing_required_field_is_rejected() -> None:
    raw_input = valid_raw_input()
    del raw_input["total_hours"]

    with pytest.raises(
        ValueError,
        match="Missing required input fields",
    ):
        build_feature_values(raw_input)


def test_feature_frames_match_saved_model_feature_order() -> None:
    bundle = load_model_bundle()

    frames = build_model_feature_frames(
        raw_input=valid_raw_input(),
        post_only_features=bundle.artifact["post_only_features"],
        full_history_features=bundle.artifact["full_history_features"],
    )

    assert list(frames.post_only.columns) == list(
        bundle.artifact["post_only_features"]
    )
    assert list(frames.full_history.columns) == list(
        bundle.artifact["full_history_features"]
    )

    assert frames.post_only.shape == (1, 13)
    assert frames.full_history.shape == (1, 10)