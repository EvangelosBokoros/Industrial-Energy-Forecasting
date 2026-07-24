from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


RAW_NUMERIC_FIELDS = (
    "total_kg",
    "total_nominal_kg",
    "total_brix_units",
    "total_hours",
    "total_pallets",
    "orders",
    "avg_brix",
)

REQUIRED_RAW_FIELDS = (
    "date",
    *RAW_NUMERIC_FIELDS,
)


@dataclass(frozen=True)
class ModelFeatureFrames:
    """Features prepared for both ensemble branches."""

    feature_values: dict[str, float | int]
    post_only: pd.DataFrame
    full_history: pd.DataFrame


def _parse_iso_date(value: Any) -> date:
    """Parse an API date while requiring YYYY-MM-DD for strings."""

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                "date must use the YYYY-MM-DD format"
            ) from error

    raise TypeError(
        "date must be a date, datetime, or YYYY-MM-DD string"
    )


def _coerce_nonnegative_number(
    field_name: str,
    value: Any,
) -> float:
    """Convert a raw operational value to a finite nonnegative float."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not boolean")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{field_name} must be numeric"
        ) from error

    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite")

    if numeric_value < 0:
        raise ValueError(f"{field_name} must not be negative")

    return numeric_value


def build_feature_values(
    raw_input: Mapping[str, Any],
) -> dict[str, float | int]:
    """Create validated model and calendar features from raw API input."""

    missing_fields = [
        field
        for field in REQUIRED_RAW_FIELDS
        if field not in raw_input
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required input fields: {missing_fields}"
        )

    prediction_date = _parse_iso_date(raw_input["date"])

    values = {
        field: _coerce_nonnegative_number(
            field,
            raw_input[field],
        )
        for field in RAW_NUMERIC_FIELDS
    }

    if values["total_nominal_kg"] <= 0:
        raise ValueError(
            "total_nominal_kg must be greater than zero"
        )

    orders = values["orders"]

    if not orders.is_integer():
        raise ValueError(
            "orders must be a nonnegative integer count"
        )

    timestamp = pd.Timestamp(prediction_date)

    feature_values: dict[str, float | int] = {
        **values,
        "orders": int(orders),
        "yield_ratio_actual_over_nominal": (
            values["total_kg"]
            / values["total_nominal_kg"]
        ),
        "weekday": int(timestamp.weekday()),
        "is_weekend": int(timestamp.weekday() in {5, 6}),
        "month": int(timestamp.month),
        "year": int(timestamp.year),
        "week_of_year": int(timestamp.isocalendar().week),
    }

    return feature_values


def build_ordered_feature_frame(
    feature_values: Mapping[str, float | int],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Build a one-row DataFrame in the model's required feature order."""

    missing_features = [
        feature
        for feature in feature_names
        if feature not in feature_values
    ]

    if missing_features:
        raise ValueError(
            f"Cannot construct model features; missing: {missing_features}"
        )

    ordered_values = {
        feature: feature_values[feature]
        for feature in feature_names
    }

    return pd.DataFrame([ordered_values], columns=list(feature_names))


def build_model_feature_frames(
    raw_input: Mapping[str, Any],
    post_only_features: Sequence[str],
    full_history_features: Sequence[str],
) -> ModelFeatureFrames:
    """Prepare correctly ordered feature frames for both models."""

    feature_values = build_feature_values(raw_input)

    return ModelFeatureFrames(
        feature_values=feature_values,
        post_only=build_ordered_feature_frame(
            feature_values,
            post_only_features,
        ),
        full_history=build_ordered_feature_frame(
            feature_values,
            full_history_features,
        ),
    )