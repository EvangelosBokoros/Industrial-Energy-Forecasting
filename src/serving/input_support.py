from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_REFERENCE_PATH = (
    PROJECT_ROOT
    / "config"
    / "serving"
    / "input_reference.json"
)

OPERATIONAL_FEATURES = (
    "total_kg",
    "total_nominal_kg",
    "total_brix_units",
    "total_hours",
    "total_pallets",
    "orders",
    "avg_brix",
    "yield_ratio_actual_over_nominal",
)

CALENDAR_FEATURES = (
    "weekday",
    "is_weekend",
    "month",
    "year",
    "week_of_year",
)

# These are the same floating-point tolerances used by the
# Version 2.1 behavioral stress test.
RANGE_ABSOLUTE_TOLERANCE = 1e-9
RANGE_RELATIVE_TOLERANCE = 1e-9


OperationalRangeStatus = Literal[
    "inside_typical_development_range",
    "development_distribution_tail",
    "outside_observed_range",
]

CalendarCoverageStatus = Literal[
    "represented",
    "contains_unseen_calendar_values",
]


@dataclass(frozen=True)
class InputSupportResult:
    """
    Development-support diagnostics for one prediction request.

    These diagnostics describe how closely the request resembles
    the model-development data. They do not alter the prediction.
    """

    operational_range_status: OperationalRangeStatus
    calendar_coverage_status: CalendarCoverageStatus

    tail_features: tuple[str, ...]
    outside_range_features: tuple[str, ...]
    unseen_calendar_features: tuple[str, ...]

    warning_codes: tuple[str, ...]


def get_value_range_flag(
    value: float,
    observed_min: float,
    observed_max: float,
    lower_quantile: float,
    upper_quantile: float,
) -> str:
    """
    Classify one feature using the Version 2.1 range rules.

    The tolerance prevents tiny floating-point differences from
    incorrectly classifying a boundary value as outside the range.
    """

    tolerance = max(
        RANGE_ABSOLUTE_TOLERANCE,
        abs(value) * RANGE_RELATIVE_TOLERANCE,
        abs(observed_min) * RANGE_RELATIVE_TOLERANCE,
        abs(observed_max) * RANGE_RELATIVE_TOLERANCE,
        abs(lower_quantile) * RANGE_RELATIVE_TOLERANCE,
        abs(upper_quantile) * RANGE_RELATIVE_TOLERANCE,
    )

    if (
        value < observed_min - tolerance
        or value > observed_max + tolerance
    ):
        return "outside_observed_range"

    if abs(observed_max - observed_min) <= tolerance:
        return "constant_in_development"

    if (
        value < lower_quantile - tolerance
        or value > upper_quantile + tolerance
    ):
        return "development_distribution_tail"

    return "inside_typical_development_range"


class InputSupportChecker:
    """
    Compare already-built model features with development evidence.

    This class does not perform feature engineering. The values must
    already have been created by feature_builder.py.
    """

    def __init__(
        self,
        reference_path: Path = DEFAULT_INPUT_REFERENCE_PATH,
    ) -> None:
        self.reference_path = Path(reference_path)
        self.reference = self._load_reference(
            self.reference_path
        )

        self.numeric_reference = self.reference[
            "numeric_features"
        ]
        self.calendar_reference = self.reference[
            "calendar_features"
        ]

    @staticmethod
    def _load_reference(
        reference_path: Path,
    ) -> dict[str, Any]:
        """Load and validate the serving reference JSON."""

        if not reference_path.exists():
            raise FileNotFoundError(
                "Serving input reference was not found: "
                f"{reference_path}"
            )

        with reference_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            reference = json.load(file)

        if not isinstance(reference, dict):
            raise TypeError(
                "Serving input reference must be a JSON object."
            )

        numeric_reference = reference.get(
            "numeric_features"
        )
        calendar_reference = reference.get(
            "calendar_features"
        )

        if not isinstance(numeric_reference, dict):
            raise ValueError(
                "Serving input reference must contain "
                "numeric_features."
            )

        if not isinstance(calendar_reference, dict):
            raise ValueError(
                "Serving input reference must contain "
                "calendar_features."
            )

        missing_numeric = sorted(
            set(OPERATIONAL_FEATURES)
            - set(numeric_reference)
        )

        missing_calendar = sorted(
            set(CALENDAR_FEATURES)
            - set(calendar_reference)
        )

        if missing_numeric:
            raise ValueError(
                "Serving input reference is missing numeric "
                f"features: {missing_numeric}"
            )

        if missing_calendar:
            raise ValueError(
                "Serving input reference is missing calendar "
                f"features: {missing_calendar}"
            )

        required_boundaries = {
            "minimum",
            "p10",
            "p90",
            "maximum",
        }

        for feature in OPERATIONAL_FEATURES:
            boundaries = numeric_reference[feature]

            if not isinstance(boundaries, dict):
                raise ValueError(
                    f"Reference boundaries for {feature} "
                    "must be a JSON object."
                )

            missing_boundaries = sorted(
                required_boundaries - set(boundaries)
            )

            if missing_boundaries:
                raise ValueError(
                    f"Reference boundaries for {feature} "
                    f"are missing: {missing_boundaries}"
                )

            minimum = float(boundaries["minimum"])
            p10 = float(boundaries["p10"])
            p90 = float(boundaries["p90"])
            maximum = float(boundaries["maximum"])

            if not all(
                math.isfinite(value)
                for value in (
                    minimum,
                    p10,
                    p90,
                    maximum,
                )
            ):
                raise ValueError(
                    f"Reference boundaries for {feature} "
                    "must be finite."
                )

            if not minimum <= p10 <= p90 <= maximum:
                raise ValueError(
                    f"Reference boundaries for {feature} "
                    "are not ordered correctly."
                )

        for feature in CALENDAR_FEATURES:
            represented_values = calendar_reference[
                feature
            ]

            if (
                not isinstance(represented_values, list)
                or not represented_values
            ):
                raise ValueError(
                    f"Calendar reference for {feature} must "
                    "be a non-empty list."
                )

        return reference

    @staticmethod
    def _finite_float(
        feature: str,
        value: Any,
    ) -> float:
        """Convert an existing feature to a finite float."""

        if isinstance(value, bool):
            raise TypeError(
                f"Feature {feature} must be numeric, "
                "not boolean."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Feature {feature} must be numeric."
            ) from error

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Feature {feature} must be finite."
            )

        return numeric_value

    def evaluate(
        self,
        feature_values: Mapping[str, float | int],
        branch_disagreement_status: str | None = None,
    ) -> InputSupportResult:
        """
        Evaluate features already created by feature_builder.py.
        """

        required_features = (
            *OPERATIONAL_FEATURES,
            *CALENDAR_FEATURES,
        )

        missing_features = sorted(
            feature
            for feature in required_features
            if feature not in feature_values
        )

        if missing_features:
            raise ValueError(
                "Cannot evaluate input support; missing "
                f"features: {missing_features}"
            )

        tail_features: list[str] = []
        outside_range_features: list[str] = []

        for feature in OPERATIONAL_FEATURES:
            value = self._finite_float(
                feature,
                feature_values[feature],
            )

            boundaries = self.numeric_reference[
                feature
            ]

            range_flag = get_value_range_flag(
                value=value,
                observed_min=float(
                    boundaries["minimum"]
                ),
                observed_max=float(
                    boundaries["maximum"]
                ),
                lower_quantile=float(
                    boundaries["p10"]
                ),
                upper_quantile=float(
                    boundaries["p90"]
                ),
            )

            if range_flag == "outside_observed_range":
                outside_range_features.append(feature)

            elif range_flag == (
                "development_distribution_tail"
            ):
                tail_features.append(feature)

        if outside_range_features:
            operational_status: OperationalRangeStatus = (
                "outside_observed_range"
            )

        elif tail_features:
            operational_status = (
                "development_distribution_tail"
            )

        else:
            operational_status = (
                "inside_typical_development_range"
            )

        unseen_calendar_features: list[str] = []

        for feature in CALENDAR_FEATURES:
            value = int(
                self._finite_float(
                    feature,
                    feature_values[feature],
                )
            )

            represented_values = {
                int(item)
                for item in self.calendar_reference[
                    feature
                ]
            }

            if value not in represented_values:
                unseen_calendar_features.append(
                    f"{feature}={value}"
                )

        if unseen_calendar_features:
            calendar_status: CalendarCoverageStatus = (
                "contains_unseen_calendar_values"
            )
        else:
            calendar_status = "represented"

        warning_codes: list[str] = []

        if operational_status == (
            "outside_observed_range"
        ):
            warning_codes.append(
                "OUTSIDE_OBSERVED_RANGE"
            )

        elif operational_status == (
            "development_distribution_tail"
        ):
            warning_codes.append(
                "DEVELOPMENT_DISTRIBUTION_TAIL"
            )

        if unseen_calendar_features:
            warning_codes.append(
                "UNSEEN_CALENDAR_VALUE"
            )

        if branch_disagreement_status == "moderate":
            warning_codes.append(
                "MODERATE_BRANCH_DISAGREEMENT"
            )

        elif branch_disagreement_status == "high":
            warning_codes.append(
                "HIGH_BRANCH_DISAGREEMENT"
            )

        return InputSupportResult(
            operational_range_status=(
                operational_status
            ),
            calendar_coverage_status=calendar_status,
            tail_features=tuple(
                sorted(tail_features)
            ),
            outside_range_features=tuple(
                sorted(outside_range_features)
            ),
            unseen_calendar_features=tuple(
                sorted(unseen_calendar_features)
            ),
            warning_codes=tuple(warning_codes),
        )