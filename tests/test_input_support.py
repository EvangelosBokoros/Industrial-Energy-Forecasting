from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.serving.input_support import (
    InputSupportChecker,
    get_value_range_flag,
)


def build_test_reference() -> dict[str, object]:
    """
    Create a small controlled reference profile for unit tests.

    These values are intentionally simple so each test can clearly
    verify typical, tail, outside-range, and calendar behavior.
    """

    return {
        "reference_version": "test",
        "reference_scope": "unit_test_reference",
        "development_start": "2025-09-12",
        "development_end": "2025-10-24",
        "development_rows": 37,
        "typical_lower_quantile": 0.1,
        "typical_upper_quantile": 0.9,
        "numeric_features": {
            "total_kg": {
                "minimum": 50.0,
                "p10": 80.0,
                "p90": 120.0,
                "maximum": 150.0,
            },
            "total_nominal_kg": {
                "minimum": 50.0,
                "p10": 80.0,
                "p90": 120.0,
                "maximum": 150.0,
            },
            "total_brix_units": {
                "minimum": 500.0,
                "p10": 800.0,
                "p90": 1200.0,
                "maximum": 1500.0,
            },
            "total_hours": {
                "minimum": 5.0,
                "p10": 8.0,
                "p90": 12.0,
                "maximum": 15.0,
            },
            "total_pallets": {
                "minimum": 0.0,
                "p10": 0.0,
                "p90": 0.0,
                "maximum": 0.0,
            },
            "orders": {
                "minimum": 1.0,
                "p10": 2.0,
                "p90": 8.0,
                "maximum": 10.0,
            },
            "avg_brix": {
                "minimum": 5.0,
                "p10": 8.0,
                "p90": 12.0,
                "maximum": 15.0,
            },
            "yield_ratio_actual_over_nominal": {
                "minimum": 0.8,
                "p10": 0.9,
                "p90": 1.1,
                "maximum": 1.2,
            },
        },
        "calendar_features": {
            "weekday": [0, 1, 2, 3, 4, 5, 6],
            "is_weekend": [0, 1],
            "month": [9, 10],
            "year": [2025],
            "week_of_year": [37, 38, 39, 40, 41, 42, 43],
        },
    }


def write_test_reference(
    tmp_path: Path,
) -> Path:
    """Write the controlled reference to a temporary JSON file."""

    reference_path = tmp_path / "input_reference.json"

    reference_path.write_text(
        json.dumps(
            build_test_reference(),
            indent=2,
        ),
        encoding="utf-8",
    )

    return reference_path


def build_typical_feature_values() -> dict[str, float | int]:
    """
    Create one already-engineered feature dictionary.

    These values represent what feature_builder.py would return.
    The support checker does not calculate any of them.
    """

    return {
        "total_kg": 100.0,
        "total_nominal_kg": 100.0,
        "total_brix_units": 1000.0,
        "total_hours": 10.0,
        "total_pallets": 0.0,
        "orders": 5,
        "avg_brix": 10.0,
        "yield_ratio_actual_over_nominal": 1.0,
        "weekday": 2,
        "is_weekend": 0,
        "month": 10,
        "year": 2025,
        "week_of_year": 41,
    }


def test_typical_features_have_no_warnings(
    tmp_path: Path,
) -> None:
    checker = InputSupportChecker(
        write_test_reference(tmp_path)
    )

    result = checker.evaluate(
        build_typical_feature_values(),
        branch_disagreement_status="low",
    )

    assert (
        result.operational_range_status
        == "inside_typical_development_range"
    )
    assert result.calendar_coverage_status == "represented"
    assert result.tail_features == ()
    assert result.outside_range_features == ()
    assert result.unseen_calendar_features == ()
    assert result.warning_codes == ()


def test_distribution_tail_is_identified(
    tmp_path: Path,
) -> None:
    checker = InputSupportChecker(
        write_test_reference(tmp_path)
    )

    feature_values = build_typical_feature_values()

    # Seven hours is inside the observed 5–15 range,
    # but below the typical p10 boundary of eight hours.
    feature_values["total_hours"] = 7.0

    result = checker.evaluate(feature_values)

    assert (
        result.operational_range_status
        == "development_distribution_tail"
    )
    assert result.tail_features == ("total_hours",)
    assert result.outside_range_features == ()
    assert result.warning_codes == (
        "DEVELOPMENT_DISTRIBUTION_TAIL",
    )


def test_outside_observed_range_is_identified(
    tmp_path: Path,
) -> None:
    checker = InputSupportChecker(
        write_test_reference(tmp_path)
    )

    feature_values = build_typical_feature_values()

    # Pallets was constant at zero in the test development profile.
    # A live value of one is therefore outside the observed range.
    feature_values["total_pallets"] = 1.0

    result = checker.evaluate(feature_values)

    assert (
        result.operational_range_status
        == "outside_observed_range"
    )
    assert result.tail_features == ()
    assert result.outside_range_features == (
        "total_pallets",
    )
    assert result.warning_codes == (
        "OUTSIDE_OBSERVED_RANGE",
    )


def test_unseen_calendar_values_are_identified(
    tmp_path: Path,
) -> None:
    checker = InputSupportChecker(
        write_test_reference(tmp_path)
    )

    feature_values = build_typical_feature_values()
    feature_values["month"] = 1
    feature_values["year"] = 2026
    feature_values["week_of_year"] = 1

    result = checker.evaluate(
        feature_values,
        branch_disagreement_status="moderate",
    )

    assert (
        result.calendar_coverage_status
        == "contains_unseen_calendar_values"
    )
    assert result.unseen_calendar_features == (
        "month=1",
        "week_of_year=1",
        "year=2026",
    )
    assert result.warning_codes == (
        "UNSEEN_CALENDAR_VALUE",
        "MODERATE_BRANCH_DISAGREEMENT",
    )


def test_high_branch_disagreement_creates_warning(
    tmp_path: Path,
) -> None:
    checker = InputSupportChecker(
        write_test_reference(tmp_path)
    )

    result = checker.evaluate(
        build_typical_feature_values(),
        branch_disagreement_status="high",
    )

    assert result.warning_codes == (
        "HIGH_BRANCH_DISAGREEMENT",
    )


def test_missing_engineered_feature_is_rejected(
    tmp_path: Path,
) -> None:
    checker = InputSupportChecker(
        write_test_reference(tmp_path)
    )

    feature_values = build_typical_feature_values()
    del feature_values["yield_ratio_actual_over_nominal"]

    with pytest.raises(
        ValueError,
        match="missing features",
    ):
        checker.evaluate(feature_values)


def test_range_check_uses_floating_point_tolerance() -> None:
    result = get_value_range_flag(
        value=0.9988601736336944,
        observed_min=0.998860174,
        observed_max=1.10,
        lower_quantile=1.00,
        upper_quantile=1.08,
    )

    assert result != "outside_observed_range"


def test_project_reference_file_loads_successfully() -> None:
    """
    Verify that the real generated project reference is compatible
    with the checker.
    """

    checker = InputSupportChecker()

    feature_values: dict[str, float | int] = {}

    for feature, boundaries in (
        checker.numeric_reference.items()
    ):
        feature_values[feature] = (
            float(boundaries["p10"])
            + float(boundaries["p90"])
        ) / 2.0

    for feature, represented_values in (
        checker.calendar_reference.items()
    ):
        feature_values[feature] = int(
            represented_values[0]
        )

    result = checker.evaluate(feature_values)

    assert (
        result.operational_range_status
        == "inside_typical_development_range"
    )
    assert result.calendar_coverage_status == "represented"
    assert result.outside_range_features == ()
    assert result.unseen_calendar_features == ()
def test_upper_distribution_tail_is_identified(
    tmp_path: Path,
) -> None:
    checker = InputSupportChecker(
        write_test_reference(tmp_path)
    )

    feature_values = build_typical_feature_values()

    # Fourteen hours is inside the observed 5–15 range,
    # but above the typical p90 boundary of twelve hours.
    feature_values["total_hours"] = 14.0

    result = checker.evaluate(feature_values)

    assert (
        result.operational_range_status
        == "development_distribution_tail"
    )
    assert result.tail_features == ("total_hours",)
    assert result.outside_range_features == ()
    assert result.warning_codes == (
        "DEVELOPMENT_DISTRIBUTION_TAIL",
    )