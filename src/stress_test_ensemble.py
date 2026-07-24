from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_tabular_data
from src.feature_sets import get_feature_columns
from src.model_ensemble import (
    build_full_history_champion_model,
    build_post_only_champion_model,
    calculate_weighted_ensemble_predictions,
)
from src.preprocessing import clean_energy_dataset
from src.settings import (
    DATE_COL,
    FULL_HISTORY_VALIDATION_END,
    INTERVENTION_DATE,
    POST_VALIDATION_END,
    TARGET_COL,
)


DATA_PATH = Path("data/processed/damavand.csv")

REPORT_PATH = Path(
    "reports/ensemble_2_1_behavioral_stress_test_report.xlsx"
)
FIGURES_DIR = Path(
    "reports/figures/ensemble_2_1_behavioral_stress_test"
)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "Damavand Energy Forecasting"
MLFLOW_RUN_NAME = (
    "2.1 Behavioral Scenario Evaluation and Synthetic Stress Test"
)

POST_ONLY_MODEL_NAME = "0.5 post_only_extra_trees"
FULL_HISTORY_MODEL_NAME = "1.3 full_history_adaboost"
OFFICIAL_ENSEMBLE_NAME = "2.0 ensemble_70_30_official"
SENSITIVITY_ENSEMBLE_NAME = "2.0 ensemble_60_40_sensitivity"

OPERATIONAL_FEATURES = [
    "total_kg",
    "total_nominal_kg",
    "total_brix_units",
    "total_hours",
    "total_pallets",
    "orders",
    "avg_brix",
    "yield_ratio_actual_over_nominal",
]

CALENDAR_FEATURES = [
    "weekday",
    "is_weekend",
    "month",
    "year",
    "week_of_year",
]

ACTIVITY_FEATURES = [
    "total_kg",
    "total_nominal_kg",
    "total_brix_units",
    "total_hours",
    "total_pallets",
    "orders",
]

LOCAL_PERTURBATION_FEATURES = [
    "total_kg",
    "total_nominal_kg",
    "total_brix_units",
    "total_hours",
]

VALID_SCENARIO_CLASSES = {
    "plausible",
    "boundary",
    "out_of_distribution",
    "adversarial",
}

VALID_SCENARIO_ORIGINS = {
    "historical_reference",
    "local_synthetic_perturbation",
    "synthetic_perturbation",
}

EXPECTED_HISTORICAL_REFERENCE_COUNT = 8
EXPECTED_LOCAL_PERTURBATION_COUNT = 8
EXPECTED_STRESS_PERTURBATION_COUNT = 4
LOCAL_ACTIVITY_INCREASE_SCALE = 1.05
LOCAL_ACTIVITY_DECREASE_SCALE = 0.95
LOCAL_INCREASE_MAX_OUTSIDE_FEATURES = 1

HIGH_DISAGREEMENT_THRESHOLD_PCT = 20.0
MODERATE_DISAGREEMENT_THRESHOLD_PCT = 10.0

DISTRIBUTION_LOWER_QUANTILE = 0.10
DISTRIBUTION_UPPER_QUANTILE = 0.90

# Numerical tolerance prevents tiny floating-point differences from
# incorrectly classifying a value as outside an observed range.
RANGE_ABSOLUTE_TOLERANCE = 1e-9
RANGE_RELATIVE_TOLERANCE = 1e-9


def ensure_output_directories() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_modeling_data() -> pd.DataFrame:
    df = load_tabular_data(DATA_PATH)
    df = clean_energy_dataset(df, DATE_COL, TARGET_COL)
    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        dayfirst=True,
        errors="coerce",
    )

    if df[DATE_COL].isna().any():
        raise ValueError("Some dates could not be parsed.")

    return df.sort_values(DATE_COL).reset_index(drop=True)


def filter_date_range(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if start_date is not None:
        mask = mask & (df[DATE_COL] >= pd.Timestamp(start_date))

    if end_date is not None:
        mask = mask & (df[DATE_COL] <= pd.Timestamp(end_date))

    return df.loc[mask].copy()


def is_integer_like(value: object) -> bool:
    if isinstance(value, bool):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return np.isfinite(numeric_value) and numeric_value.is_integer()


def get_value_range_flag(
    value: float,
    observed_min: float,
    observed_max: float,
    lower_quantile: float,
    upper_quantile: float,
) -> str:
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

    value_range = observed_max - observed_min

    if abs(value_range) <= tolerance:
        return "constant_in_development"

    if (
        value < lower_quantile - tolerance
        or value > upper_quantile + tolerance
    ):
        return "development_distribution_tail"

    return "inside_typical_development_range"


def select_anchor_row(
    reference_df: pd.DataFrame,
    target_values: dict[str, float],
    *,
    candidate_mask: pd.Series | None = None,
) -> pd.Series:
    if candidate_mask is None:
        candidates = reference_df.copy()
    else:
        candidates = reference_df.loc[candidate_mask].copy()

    if candidates.empty:
        raise ValueError("No rows are available for the requested anchor.")

    distance = pd.Series(0.0, index=candidates.index)

    for column, target_value in target_values.items():
        if column not in candidates.columns:
            raise ValueError(f"Anchor feature is missing: {column}")

        scale = float(reference_df[column].quantile(0.75)) - float(
            reference_df[column].quantile(0.25)
        )

        if not np.isfinite(scale) or scale == 0:
            scale = float(reference_df[column].std(ddof=0))

        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        distance = distance + (
            (candidates[column].astype(float) - float(target_value)) / scale
        ) ** 2

    return candidates.loc[distance.idxmin()].copy()


def get_quantile_profile(
    reference_df: pd.DataFrame,
    columns: list[str],
    quantile: float,
) -> dict[str, float]:
    return {
        column: float(reference_df[column].quantile(quantile))
        for column in columns
    }


def row_to_feature_dict(
    row: pd.Series,
    feature_columns: list[str],
) -> dict[str, float]:
    return {
        column: float(row[column])
        for column in feature_columns
    }


def recalculate_confirmed_derived_features(
    row: dict[str, float],
) -> dict[str, float]:
    updated_row = row.copy()

    total_nominal_kg = float(updated_row["total_nominal_kg"])

    if total_nominal_kg <= 0:
        raise ValueError(
            "total_nominal_kg must be positive to calculate "
            "yield_ratio_actual_over_nominal."
        )

    updated_row["yield_ratio_actual_over_nominal"] = (
        float(updated_row["total_kg"]) / total_nominal_kg
    )

    return updated_row


def build_local_activity_updates(
    anchor_row: pd.Series,
    scale_factor: float,
) -> dict[str, float]:
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be positive and finite.")

    updates = {
        feature: float(anchor_row[feature]) * scale_factor
        for feature in LOCAL_PERTURBATION_FEATURES
    }

    if updates["total_nominal_kg"] <= 0:
        raise ValueError(
            "Local perturbation produced nonpositive total_nominal_kg."
        )

    if updates["total_hours"] <= 0:
        raise ValueError(
            "Local perturbation produced nonpositive total_hours."
        )

    return updates


def choose_local_activity_scale(
    anchor_row: pd.Series,
    reference_df: pd.DataFrame,
) -> float:
    proposed_updates = build_local_activity_updates(
        anchor_row,
        LOCAL_ACTIVITY_INCREASE_SCALE,
    )

    outside_feature_count = 0

    for feature, proposed_value in proposed_updates.items():
        observed_max = float(reference_df[feature].max())
        tolerance = max(
            RANGE_ABSOLUTE_TOLERANCE,
            abs(observed_max) * RANGE_RELATIVE_TOLERANCE,
        )

        if proposed_value > observed_max + tolerance:
            outside_feature_count += 1

    if outside_feature_count > LOCAL_INCREASE_MAX_OUTSIDE_FEATURES:
        return LOCAL_ACTIVITY_DECREASE_SCALE

    return LOCAL_ACTIVITY_INCREASE_SCALE


def get_perturbation_type(scale_factor: float) -> str:
    percentage = abs(scale_factor - 1.0) * 100.0
    rounded_percentage = int(round(percentage))

    if scale_factor > 1.0:
        return f"operating_scale_up_{rounded_percentage}pct"

    if scale_factor < 1.0:
        return f"operating_scale_down_{rounded_percentage}pct"

    return "none"


def build_scenario_from_anchor(
    *,
    scenario_name: str,
    scenario_class: str,
    scenario_origin: str,
    parent_scenario_name: str,
    perturbation_type: str,
    perturbation_scale: float,
    description: str,
    construction_method: str,
    anchor_row: pd.Series,
    feature_columns: list[str],
    row_updates: dict[str, float] | None = None,
) -> dict[str, object]:
    if scenario_class not in VALID_SCENARIO_CLASSES:
        raise ValueError(
            f"Unknown scenario class: {scenario_class}. "
            f"Expected one of: {sorted(VALID_SCENARIO_CLASSES)}"
        )

    if scenario_origin not in VALID_SCENARIO_ORIGINS:
        raise ValueError(
            f"Unknown scenario origin: {scenario_origin}. "
            f"Expected one of: {sorted(VALID_SCENARIO_ORIGINS)}"
        )

    if not np.isfinite(perturbation_scale) or perturbation_scale <= 0:
        raise ValueError(
            "perturbation_scale must be positive and finite."
        )

    row = row_to_feature_dict(anchor_row, feature_columns)

    if row_updates:
        row.update(
            {
                column: float(value)
                for column, value in row_updates.items()
            }
        )

    row = recalculate_confirmed_derived_features(row)

    anchor_date = pd.Timestamp(anchor_row[DATE_COL])

    return {
        "scenario_name": scenario_name,
        "scenario_class": scenario_class,
        "scenario_origin": scenario_origin,
        "parent_scenario_name": parent_scenario_name,
        "perturbation_type": perturbation_type,
        "perturbation_scale": float(perturbation_scale),
        "description": description,
        "construction_method": construction_method,
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "scenario_date": anchor_date.strftime("%Y-%m-%d"),
        **row,
    }


def create_behavioral_scenarios(
    reference_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    median_profile = get_quantile_profile(
        reference_df,
        OPERATIONAL_FEATURES,
        0.50,
    )
    low_activity_profile = get_quantile_profile(
        reference_df,
        ACTIVITY_FEATURES,
        0.15,
    )
    high_activity_profile = get_quantile_profile(
        reference_df,
        ACTIVITY_FEATURES,
        0.85,
    )

    weekday_mask = reference_df["is_weekend"].astype(int) == 0
    baseline_anchor = select_anchor_row(
        reference_df,
        median_profile,
        candidate_mask=weekday_mask,
    )
    low_activity_anchor = select_anchor_row(
        reference_df,
        low_activity_profile,
    )
    high_activity_anchor = select_anchor_row(
        reference_df,
        high_activity_profile,
    )

    weekend_mask = reference_df["is_weekend"].astype(int) == 1
    weekend_anchor = select_anchor_row(
        reference_df,
        median_profile,
        candidate_mask=weekend_mask,
    )

    high_orders_mask = (
        reference_df["orders"]
        >= reference_df["orders"].quantile(0.75)
    )
    high_orders_anchor = select_anchor_row(
        reference_df,
        {
            "orders": float(reference_df["orders"].quantile(0.90)),
            "total_kg": float(reference_df["total_kg"].median()),
            "total_hours": float(reference_df["total_hours"].median()),
        },
        candidate_mask=high_orders_mask,
    )

    high_production_mask = (
        reference_df["total_kg"]
        >= reference_df["total_kg"].quantile(0.75)
    )
    low_orders_high_production_anchor = select_anchor_row(
        reference_df,
        {
            "orders": float(reference_df["orders"].quantile(0.10)),
            "total_kg": float(reference_df["total_kg"].quantile(0.90)),
            "total_hours": float(reference_df["total_hours"].quantile(0.75)),
        },
        candidate_mask=high_production_mask,
    )

    high_brix_anchor = select_anchor_row(
        reference_df,
        {
            "avg_brix": float(reference_df["avg_brix"].quantile(0.90)),
            "total_kg": float(reference_df["total_kg"].median()),
        },
    )
    low_brix_anchor = select_anchor_row(
        reference_df,
        {
            "avg_brix": float(reference_df["avg_brix"].quantile(0.10)),
            "total_kg": float(reference_df["total_kg"].median()),
        },
    )

    historical_specs = [
        {
            "scenario_name": "baseline_median_weekday",
            "scenario_class": "plausible",
            "description": (
                "Real development row closest to the median weekday "
                "operational profile."
            ),
            "anchor_row": baseline_anchor,
        },
        {
            "scenario_name": "low_production_low_hours",
            "scenario_class": "plausible",
            "description": (
                "Real development row closest to a low-activity profile."
            ),
            "anchor_row": low_activity_anchor,
        },
        {
            "scenario_name": "high_production_high_hours",
            "scenario_class": "plausible",
            "description": (
                "Real development row closest to a high-activity profile."
            ),
            "anchor_row": high_activity_anchor,
        },
        {
            "scenario_name": "weekend_moderate_production",
            "scenario_class": "plausible",
            "description": (
                "Real weekend development row closest to the median profile."
            ),
            "anchor_row": weekend_anchor,
        },
        {
            "scenario_name": "high_orders_moderate_production",
            "scenario_class": "boundary",
            "description": (
                "Real development row with high orders and moderate activity."
            ),
            "anchor_row": high_orders_anchor,
        },
        {
            "scenario_name": "low_orders_high_production",
            "scenario_class": "boundary",
            "description": (
                "Real high-production development row with relatively "
                "low orders."
            ),
            "anchor_row": low_orders_high_production_anchor,
        },
        {
            "scenario_name": "high_brix_day",
            "scenario_class": "boundary",
            "description": (
                "Real development row close to the upper average-brix range."
            ),
            "anchor_row": high_brix_anchor,
        },
        {
            "scenario_name": "low_brix_day",
            "scenario_class": "boundary",
            "description": (
                "Real development row close to the lower average-brix range."
            ),
            "anchor_row": low_brix_anchor,
        },
    ]

    scenarios: list[dict[str, object]] = []

    for spec in historical_specs:
        parent_name = str(spec["scenario_name"])
        scenario_class = str(spec["scenario_class"])
        anchor_row = spec["anchor_row"]

        scenarios.append(
            build_scenario_from_anchor(
                scenario_name=parent_name,
                scenario_class=scenario_class,
                scenario_origin="historical_reference",
                parent_scenario_name="",
                perturbation_type="none",
                perturbation_scale=1.0,
                description=str(spec["description"]),
                construction_method="historical_anchor_replay",
                anchor_row=anchor_row,
                feature_columns=feature_columns,
            )
        )

        local_scale = choose_local_activity_scale(
            anchor_row,
            reference_df,
        )
        perturbation_type = get_perturbation_type(local_scale)
        local_updates = build_local_activity_updates(
            anchor_row,
            local_scale,
        )
        local_name = f"{parent_name}_{perturbation_type}"

        scenarios.append(
            build_scenario_from_anchor(
                scenario_name=local_name,
                scenario_class=scenario_class,
                scenario_origin="local_synthetic_perturbation",
                parent_scenario_name=parent_name,
                perturbation_type=perturbation_type,
                perturbation_scale=local_scale,
                description=(
                    f"Controlled local operating-scale perturbation around "
                    f"{parent_name}. Continuous production and operating "
                    f"features are scaled by {local_scale:.2f}; orders, "
                    "calendar context, average Brix, and total pallets "
                    "remain unchanged."
                ),
                construction_method=(
                    "historical_anchor_plus_local_operating_scale_perturbation"
                ),
                anchor_row=anchor_row,
                feature_columns=feature_columns,
                row_updates=local_updates,
            )
        )

    low_hours = float(reference_df["total_hours"].quantile(0.10))
    high_hours = float(reference_df["total_hours"].quantile(0.90))

    scenarios.append(
        build_scenario_from_anchor(
            scenario_name="high_intensity_kg_low_hours",
            scenario_class="adversarial",
            scenario_origin="synthetic_perturbation",
            parent_scenario_name="high_production_high_hours",
            perturbation_type="low_hours_adversarial",
            perturbation_scale=(
                low_hours / max(float(high_activity_anchor["total_hours"]), 1e-12)
            ),
            description=(
                "High-activity historical anchor with operating hours reduced "
                "to a low development quantile."
            ),
            construction_method="historical_anchor_plus_adversarial_change",
            anchor_row=high_activity_anchor,
            feature_columns=feature_columns,
            row_updates={"total_hours": low_hours},
        )
    )

    scenarios.append(
        build_scenario_from_anchor(
            scenario_name="low_intensity_kg_high_hours",
            scenario_class="adversarial",
            scenario_origin="synthetic_perturbation",
            parent_scenario_name="low_production_low_hours",
            perturbation_type="high_hours_adversarial",
            perturbation_scale=(
                high_hours / max(float(low_activity_anchor["total_hours"]), 1e-12)
            ),
            description=(
                "Low-activity historical anchor with operating hours increased "
                "to a high development quantile."
            ),
            construction_method="historical_anchor_plus_adversarial_change",
            anchor_row=low_activity_anchor,
            feature_columns=feature_columns,
            row_updates={"total_hours": high_hours},
        )
    )

    outside_high_updates = {
        column: float(reference_df[column].max() * 1.10)
        for column in ACTIVITY_FEATURES
    }

    maximum_observed_orders = float(reference_df["orders"].max())

    if not is_integer_like(maximum_observed_orders):
        raise ValueError(
            "orders must contain integer-like counts before creating "
            "the outside-observed scenario."
        )

    outside_high_updates["orders"] = float(
        int(maximum_observed_orders) + 1
    )

    scenarios.append(
        build_scenario_from_anchor(
            scenario_name="outside_observed_high_activity",
            scenario_class="out_of_distribution",
            scenario_origin="synthetic_perturbation",
            parent_scenario_name="high_production_high_hours",
            perturbation_type="activity_above_observed_maxima",
            perturbation_scale=1.10,
            description=(
                "High-activity anchor with activity features set 10% above "
                "their observed development maxima."
            ),
            construction_method="historical_anchor_plus_ood_change",
            anchor_row=high_activity_anchor,
            feature_columns=feature_columns,
            row_updates=outside_high_updates,
        )
    )

    below_min_updates = {
        "total_kg": max(
            0.0,
            float(reference_df["total_kg"].min() * 0.50),
        ),
        "total_nominal_kg": max(
            1e-6,
            float(reference_df["total_nominal_kg"].min() * 0.50),
        ),
        "total_brix_units": max(
            0.0,
            float(reference_df["total_brix_units"].min() * 0.50),
        ),
        "total_hours": max(
            1e-6,
            float(reference_df["total_hours"].min() * 0.50),
        ),
        "total_pallets": max(
            0.0,
            float(reference_df["total_pallets"].min() * 0.50),
        ),
        "orders": 0,
    }

    scenarios.append(
        build_scenario_from_anchor(
            scenario_name="near_shutdown_low_activity",
            scenario_class="out_of_distribution",
            scenario_origin="synthetic_perturbation",
            parent_scenario_name="low_production_low_hours",
            perturbation_type="activity_below_observed_minima",
            perturbation_scale=0.50,
            description=(
                "Low-activity anchor with activity features set below "
                "their observed development minima."
            ),
            construction_method="historical_anchor_plus_ood_change",
            anchor_row=low_activity_anchor,
            feature_columns=feature_columns,
            row_updates=below_min_updates,
        )
    )

    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df.insert(
        0,
        "scenario_id",
        range(1, len(scenarios_df) + 1),
    )

    return scenarios_df


def add_schema_check(
    records: list[dict[str, object]],
    *,
    scenario_name: str,
    check_name: str,
    passed: bool,
    value: object,
    message: str,
) -> None:
    records.append(
        {
            "scenario_name": scenario_name,
            "check_name": check_name,
            "status": "passed" if passed else "failed",
            "value": value,
            "message": message,
        }
    )


def calculate_schema_checks(
    scenarios_df: pd.DataFrame,
    required_feature_columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    required_columns = [
        "scenario_name",
        "scenario_class",
        "scenario_origin",
        "parent_scenario_name",
        "perturbation_type",
        "perturbation_scale",
        "description",
        "construction_method",
        "anchor_date",
        "scenario_date",
        *required_feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in scenarios_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Scenario table is missing required columns: {missing_columns}"
        )

    for _, scenario in scenarios_df.iterrows():
        scenario_name = str(scenario["scenario_name"])

        scenario_class = str(scenario["scenario_class"])
        add_schema_check(
            records,
            scenario_name=scenario_name,
            check_name="scenario_class_allowed",
            passed=scenario_class in VALID_SCENARIO_CLASSES,
            value=scenario_class,
            message=(
                "Scenario class is recognized."
                if scenario_class in VALID_SCENARIO_CLASSES
                else "Scenario class is not recognized."
            ),
        )

        scenario_origin = str(scenario["scenario_origin"])
        origin_valid = scenario_origin in VALID_SCENARIO_ORIGINS
        add_schema_check(
            records,
            scenario_name=scenario_name,
            check_name="scenario_origin_allowed",
            passed=origin_valid,
            value=scenario_origin,
            message=(
                "Scenario origin is recognized."
                if origin_valid
                else "Scenario origin is not recognized."
            ),
        )

        parent_name = str(scenario["parent_scenario_name"]).strip()
        all_scenario_names = set(
            scenarios_df["scenario_name"].astype(str)
        )

        if scenario_origin == "historical_reference":
            parent_valid = parent_name == ""
        else:
            parent_valid = (
                parent_name in all_scenario_names
                and parent_name != scenario_name
            )

        add_schema_check(
            records,
            scenario_name=scenario_name,
            check_name="parent_scenario_reference_valid",
            passed=parent_valid,
            value=parent_name,
            message=(
                "Parent scenario reference is valid."
                if parent_valid
                else "Parent scenario reference is invalid."
            ),
        )

        perturbation_type = str(scenario["perturbation_type"])
        perturbation_scale = pd.to_numeric(
            pd.Series([scenario["perturbation_scale"]]),
            errors="coerce",
        ).iloc[0]
        scale_valid = (
            not pd.isna(perturbation_scale)
            and np.isfinite(float(perturbation_scale))
            and float(perturbation_scale) > 0
        )

        if scenario_origin == "historical_reference":
            perturbation_metadata_valid = (
                perturbation_type == "none"
                and scale_valid
                and np.isclose(float(perturbation_scale), 1.0)
            )
        else:
            perturbation_metadata_valid = (
                perturbation_type != "none" and scale_valid
            )

        add_schema_check(
            records,
            scenario_name=scenario_name,
            check_name="perturbation_metadata_valid",
            passed=perturbation_metadata_valid,
            value=(
                f"type={perturbation_type}, "
                f"scale={scenario['perturbation_scale']}"
            ),
            message=(
                "Perturbation metadata is valid."
                if perturbation_metadata_valid
                else "Perturbation metadata is invalid."
            ),
        )

        parsed_date = pd.to_datetime(
            scenario["scenario_date"],
            errors="coerce",
        )
        date_valid = not pd.isna(parsed_date)

        add_schema_check(
            records,
            scenario_name=scenario_name,
            check_name="scenario_date_parseable",
            passed=date_valid,
            value=scenario["scenario_date"],
            message=(
                "Scenario date is parseable."
                if date_valid
                else "Scenario date could not be parsed."
            ),
        )

        for column in required_feature_columns:
            value = pd.to_numeric(
                pd.Series([scenario[column]]),
                errors="coerce",
            ).iloc[0]

            finite_numeric = (
                not pd.isna(value)
                and np.isfinite(float(value))
            )

            add_schema_check(
                records,
                scenario_name=scenario_name,
                check_name=f"{column}_is_finite_numeric",
                passed=finite_numeric,
                value=scenario[column],
                message=(
                    f"{column} is a finite numeric value."
                    if finite_numeric
                    else f"{column} must be a finite numeric value."
                ),
            )

        for column in OPERATIONAL_FEATURES:
            value = float(scenario[column])
            nonnegative = value >= 0.0

            add_schema_check(
                records,
                scenario_name=scenario_name,
                check_name=f"{column}_is_nonnegative",
                passed=nonnegative,
                value=value,
                message=(
                    f"{column} is nonnegative."
                    if nonnegative
                    else f"{column} must not be negative."
                ),
            )

        orders = scenario["orders"]
        orders_valid = (
            is_integer_like(orders)
            and int(float(orders)) >= 0
        )

        add_schema_check(
            records,
            scenario_name=scenario_name,
            check_name="orders_is_nonnegative_integer",
            passed=orders_valid,
            value=orders,
            message=(
                "orders is a nonnegative integer count."
                if orders_valid
                else "orders must be a nonnegative integer count."
            ),
        )

        weekday = scenario["weekday"]
        weekday_valid = (
            is_integer_like(weekday)
            and 0 <= int(float(weekday)) <= 6
        )

        is_weekend = scenario["is_weekend"]
        is_weekend_valid = (
            is_integer_like(is_weekend)
            and int(float(is_weekend)) in {0, 1}
        )

        month = scenario["month"]
        month_valid = (
            is_integer_like(month)
            and 1 <= int(float(month)) <= 12
        )

        year = scenario["year"]
        year_valid = (
            is_integer_like(year)
            and int(float(year)) >= 1
        )

        week_of_year = scenario["week_of_year"]
        week_of_year_valid = (
            is_integer_like(week_of_year)
            and 1 <= int(float(week_of_year)) <= 53
        )

        checks = [
            (
                "weekday_allowed",
                weekday_valid,
                weekday,
                "weekday must be an integer from 0 to 6.",
            ),
            (
                "is_weekend_allowed",
                is_weekend_valid,
                is_weekend,
                "is_weekend must be 0 or 1.",
            ),
            (
                "month_allowed",
                month_valid,
                month,
                "month must be an integer from 1 to 12.",
            ),
            (
                "year_allowed",
                year_valid,
                year,
                "year must be a positive integer.",
            ),
            (
                "week_of_year_allowed",
                week_of_year_valid,
                week_of_year,
                "week_of_year must be an integer from 1 to 53.",
            ),
        ]

        for check_name, passed, value, failure_message in checks:
            add_schema_check(
                records,
                scenario_name=scenario_name,
                check_name=check_name,
                passed=passed,
                value=value,
                message=(
                    f"{check_name} passed."
                    if passed
                    else failure_message
                ),
            )

        weekend_consistency_valid = False

        if weekday_valid and is_weekend_valid:
            expected_is_weekend = (
                1
                if int(float(weekday)) in {5, 6}
                else 0
            )
            weekend_consistency_valid = (
                int(float(is_weekend)) == expected_is_weekend
            )

        add_schema_check(
            records,
            scenario_name=scenario_name,
            check_name="weekday_is_weekend_consistent",
            passed=weekend_consistency_valid,
            value=f"weekday={weekday}, is_weekend={is_weekend}",
            message=(
                "weekday and is_weekend are consistent."
                if weekend_consistency_valid
                else "weekday and is_weekend are inconsistent."
            ),
        )

        date_calendar_consistent = False

        if (
            date_valid
            and weekday_valid
            and month_valid
            and year_valid
            and week_of_year_valid
        ):
            iso_calendar = parsed_date.isocalendar()

            date_calendar_consistent = (
                int(parsed_date.weekday()) == int(float(weekday))
                and int(parsed_date.month) == int(float(month))
                and int(parsed_date.year) == int(float(year))
                and int(iso_calendar.week)
                == int(float(week_of_year))
            )

        add_schema_check(
            records,
            scenario_name=scenario_name,
            check_name="scenario_date_calendar_consistent",
            passed=date_calendar_consistent,
            value=scenario["scenario_date"],
            message=(
                "scenario_date matches the calendar features."
                if date_calendar_consistent
                else "scenario_date does not match the calendar features."
            ),
        )

    return pd.DataFrame(records)


def summarize_schema_checks(
    schema_checks_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for scenario_name, group in schema_checks_df.groupby(
        "scenario_name"
    ):
        failed_group = group[group["status"] == "failed"]

        records.append(
            {
                "scenario_name": scenario_name,
                "schema_status": (
                    "valid"
                    if failed_group.empty
                    else "invalid"
                ),
                "failed_schema_check_count": int(
                    len(failed_group)
                ),
                "failed_schema_checks": (
                    ""
                    if failed_group.empty
                    else "; ".join(
                        failed_group["check_name"].astype(str)
                    )
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_operational_range_checks(
    scenarios_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    reference_intensity = (
        reference_df["total_kg"]
        / reference_df["total_hours"].replace(0, np.nan)
    )
    reference_intensity = reference_intensity.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if reference_intensity.empty:
        raise ValueError(
            "Cannot calculate kg_per_hour reference distribution."
        )

    intensity_min = float(reference_intensity.min())
    intensity_max = float(reference_intensity.max())
    intensity_q_lower = float(
        reference_intensity.quantile(DISTRIBUTION_LOWER_QUANTILE)
    )
    intensity_q_upper = float(
        reference_intensity.quantile(DISTRIBUTION_UPPER_QUANTILE)
    )
    intensity_unique_count = int(reference_intensity.nunique(dropna=True))

    for _, scenario in scenarios_df.iterrows():
        scenario_name = str(scenario["scenario_name"])

        for column in OPERATIONAL_FEATURES:
            reference_values = pd.to_numeric(
                reference_df[column],
                errors="coerce",
            ).dropna()

            if reference_values.empty:
                raise ValueError(
                    f"No valid development values are available for {column}."
                )

            value = float(scenario[column])
            observed_min = float(reference_values.min())
            observed_max = float(reference_values.max())
            lower_quantile = float(
                reference_values.quantile(DISTRIBUTION_LOWER_QUANTILE)
            )
            upper_quantile = float(
                reference_values.quantile(DISTRIBUTION_UPPER_QUANTILE)
            )
            unique_count = int(reference_values.nunique(dropna=True))

            range_status = get_value_range_flag(
                value=value,
                observed_min=observed_min,
                observed_max=observed_max,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )

            records.append(
                {
                    "scenario_name": scenario_name,
                    "feature": column,
                    "value": value,
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "development_q10": lower_quantile,
                    "development_q90": upper_quantile,
                    "development_unique_count": unique_count,
                    "range_status": range_status,
                }
            )

        scenario_hours = float(scenario["total_hours"])
        scenario_intensity = (
            float(scenario["total_kg"]) / scenario_hours
            if scenario_hours != 0
            else np.nan
        )

        if (
            pd.isna(scenario_intensity)
            or not np.isfinite(scenario_intensity)
        ):
            intensity_status = "undefined"
        else:
            intensity_status = get_value_range_flag(
                value=float(scenario_intensity),
                observed_min=intensity_min,
                observed_max=intensity_max,
                lower_quantile=intensity_q_lower,
                upper_quantile=intensity_q_upper,
            )

        records.append(
            {
                "scenario_name": scenario_name,
                "feature": "kg_per_hour",
                "value": scenario_intensity,
                "observed_min": intensity_min,
                "observed_max": intensity_max,
                "development_q10": intensity_q_lower,
                "development_q90": intensity_q_upper,
                "development_unique_count": intensity_unique_count,
                "range_status": intensity_status,
            }
        )

    return pd.DataFrame(records)


def summarize_operational_range_checks(
    operational_checks_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for scenario_name, group in operational_checks_df.groupby(
        "scenario_name"
    ):
        statuses = set(group["range_status"])

        if "undefined" in statuses:
            overall_status = "undefined_operational_value"
        elif "outside_observed_range" in statuses:
            overall_status = "outside_observed_range"
        elif "development_distribution_tail" in statuses:
            overall_status = "development_distribution_tail"
        else:
            overall_status = "inside_typical_development_range"

        records.append(
            {
                "scenario_name": scenario_name,
                "operational_range_status": overall_status,
                "outside_operational_feature_count": int(
                    (
                        group["range_status"]
                        == "outside_observed_range"
                    ).sum()
                ),
                "distribution_tail_feature_count": int(
                    (
                        group["range_status"]
                        == "development_distribution_tail"
                    ).sum()
                ),
                "constant_operational_feature_count": int(
                    (
                        group["range_status"]
                        == "constant_in_development"
                    ).sum()
                ),
                "undefined_operational_feature_count": int(
                    (
                        group["range_status"]
                        == "undefined"
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_calendar_coverage_checks(
    scenarios_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    reference_values = {
        feature: set(
            pd.to_numeric(
                reference_df[feature],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )
        for feature in CALENDAR_FEATURES
    }

    maximum_reference_year = max(reference_values["year"])

    for _, scenario in scenarios_df.iterrows():
        scenario_name = str(scenario["scenario_name"])

        for feature in CALENDAR_FEATURES:
            value = int(float(scenario[feature]))
            represented = value in reference_values[feature]

            if represented:
                coverage_status = "represented_in_development"
            elif (
                feature == "year"
                and value > maximum_reference_year
            ):
                coverage_status = "future_year_not_represented"
            else:
                coverage_status = f"unseen_{feature}"

            records.append(
                {
                    "scenario_name": scenario_name,
                    "feature": feature,
                    "value": value,
                    "represented_in_development": represented,
                    "coverage_status": coverage_status,
                    "represented_values": ", ".join(
                        str(item)
                        for item in sorted(
                            reference_values[feature]
                        )
                    ),
                }
            )

    return pd.DataFrame(records)


def summarize_calendar_coverage_checks(
    calendar_checks_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for scenario_name, group in calendar_checks_df.groupby(
        "scenario_name"
    ):
        unseen_group = group[
            group["coverage_status"]
            != "represented_in_development"
        ]

        records.append(
            {
                "scenario_name": scenario_name,
                "calendar_coverage_status": (
                    "represented"
                    if unseen_group.empty
                    else "contains_unseen_calendar_values"
                ),
                "unseen_calendar_feature_count": int(
                    len(unseen_group)
                ),
                "unseen_calendar_features": (
                    ""
                    if unseen_group.empty
                    else "; ".join(
                        (
                            f"{row.feature}={row.value} "
                            f"({row.coverage_status})"
                        )
                        for row in unseen_group.itertuples()
                    )
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_operational_reference_ranges(
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for column in OPERATIONAL_FEATURES:
        reference_values = pd.to_numeric(
            reference_df[column],
            errors="coerce",
        ).dropna()

        if reference_values.empty:
            continue

        minimum = float(reference_values.min())
        maximum = float(reference_values.max())
        lower_quantile = float(
            reference_values.quantile(DISTRIBUTION_LOWER_QUANTILE)
        )
        upper_quantile = float(
            reference_values.quantile(DISTRIBUTION_UPPER_QUANTILE)
        )
        unique_count = int(reference_values.nunique(dropna=True))

        records.append(
            {
                "feature": column,
                "observed_min": minimum,
                "observed_max": maximum,
                "development_q10": lower_quantile,
                "development_q90": upper_quantile,
                "development_unique_count": unique_count,
                "development_feature_status": (
                    "constant_in_development"
                    if unique_count == 1
                    else "variable_in_development"
                ),
            }
        )

    reference_intensity = (
        reference_df["total_kg"]
        / reference_df["total_hours"].replace(0, np.nan)
    )
    reference_intensity = reference_intensity.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if not reference_intensity.empty:
        minimum = float(reference_intensity.min())
        maximum = float(reference_intensity.max())
        lower_quantile = float(
            reference_intensity.quantile(DISTRIBUTION_LOWER_QUANTILE)
        )
        upper_quantile = float(
            reference_intensity.quantile(DISTRIBUTION_UPPER_QUANTILE)
        )
        unique_count = int(reference_intensity.nunique(dropna=True))

        records.append(
            {
                "feature": "kg_per_hour",
                "observed_min": minimum,
                "observed_max": maximum,
                "development_q10": lower_quantile,
                "development_q90": upper_quantile,
                "development_unique_count": unique_count,
                "development_feature_status": (
                    "constant_in_development"
                    if unique_count == 1
                    else "variable_in_development"
                ),
            }
        )

    return pd.DataFrame(records)


def build_predictions_table(
    scenarios_df: pd.DataFrame,
    predictions_by_model: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []

    metadata_columns = [
        "scenario_class",
        "scenario_origin",
        "parent_scenario_name",
        "perturbation_type",
        "perturbation_scale",
        "description",
        "construction_method",
        "anchor_date",
        "scenario_date",
    ]

    scenario_metadata = (
        scenarios_df.set_index("scenario_name")[
            metadata_columns
        ].to_dict(orient="index")
    )

    for model_name, predictions in predictions_by_model.items():
        for scenario_name, prediction in zip(
            scenarios_df["scenario_name"],
            predictions,
        ):
            metadata = scenario_metadata[str(scenario_name)]

            records.append(
                {
                    "scenario_name": scenario_name,
                    **metadata,
                    "model_name": model_name,
                    "predicted_active_energy_kWh": float(
                        prediction
                    ),
                }
            )

    return pd.DataFrame(records)


def calculate_branch_disagreement(
    post_only_predictions: np.ndarray,
    full_history_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    post_predictions = np.asarray(
        post_only_predictions,
        dtype=float,
    )
    full_predictions = np.asarray(
        full_history_predictions,
        dtype=float,
    )

    disagreement_kwh = np.abs(
        post_predictions - full_predictions
    )

    component_mean = (
        np.abs(post_predictions) + np.abs(full_predictions)
    ) / 2.0

    disagreement_pct = np.divide(
        disagreement_kwh,
        component_mean,
        out=np.full_like(disagreement_kwh, np.nan),
        where=component_mean > 0,
    ) * 100.0

    return disagreement_kwh, disagreement_pct


def classify_disagreement(
    disagreement_pct: float,
) -> str:
    if not np.isfinite(disagreement_pct):
        return "undefined"

    if disagreement_pct >= HIGH_DISAGREEMENT_THRESHOLD_PCT:
        return "high"

    if disagreement_pct >= MODERATE_DISAGREEMENT_THRESHOLD_PCT:
        return "moderate"

    return "low"


def build_wide_prediction_summary(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "scenario_name",
        "scenario_class",
        "scenario_origin",
        "parent_scenario_name",
        "perturbation_type",
        "perturbation_scale",
        "description",
        "construction_method",
        "anchor_date",
        "scenario_date",
    ]

    metadata_df = predictions_df[
        metadata_columns
    ].drop_duplicates()

    wide_predictions_df = predictions_df.pivot_table(
        index="scenario_name",
        columns="model_name",
        values="predicted_active_energy_kWh",
        aggfunc="first",
    ).reset_index()

    wide_df = metadata_df.merge(
        wide_predictions_df,
        on="scenario_name",
        how="left",
    )

    disagreement_kwh, disagreement_pct = (
        calculate_branch_disagreement(
            wide_df[POST_ONLY_MODEL_NAME].to_numpy(),
            wide_df[FULL_HISTORY_MODEL_NAME].to_numpy(),
        )
    )

    wide_df["branch_disagreement_kwh"] = disagreement_kwh
    wide_df["branch_disagreement_pct"] = disagreement_pct
    wide_df["branch_disagreement_status"] = [
        classify_disagreement(value)
        for value in disagreement_pct
    ]

    wide_df["sensitivity_minus_official"] = (
        wide_df[SENSITIVITY_ENSEMBLE_NAME]
        - wide_df[OFFICIAL_ENSEMBLE_NAME]
    )

    return wide_df


def build_paired_sensitivity_table(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    indexed = summary_df.set_index("scenario_name")
    local_rows = summary_df[
        summary_df["scenario_origin"]
        == "local_synthetic_perturbation"
    ].copy()

    records: list[dict[str, object]] = []

    for _, perturbed_row in local_rows.iterrows():
        parent_name = str(perturbed_row["parent_scenario_name"])

        if parent_name not in indexed.index:
            raise ValueError(
                f"Missing parent scenario for local perturbation: {parent_name}"
            )

        parent_row = indexed.loc[parent_name]
        original_official = float(parent_row[OFFICIAL_ENSEMBLE_NAME])
        perturbed_official = float(
            perturbed_row[OFFICIAL_ENSEMBLE_NAME]
        )
        official_delta_kwh = perturbed_official - original_official
        official_delta_pct = (
            official_delta_kwh
            / max(abs(original_official), 1e-12)
            * 100.0
        )

        scale = float(perturbed_row["perturbation_scale"])
        expected_direction = (
            "increase" if scale > 1.0 else "decrease"
        )
        flat_tolerance = max(1e-6, abs(original_official) * 1e-9)

        if abs(official_delta_kwh) <= flat_tolerance:
            directional_status = "locally_flat"
        elif (
            expected_direction == "increase"
            and official_delta_kwh > 0
        ) or (
            expected_direction == "decrease"
            and official_delta_kwh < 0
        ):
            directional_status = "passed"
        else:
            directional_status = "diagnostic_warning"

        post_only_delta = float(
            perturbed_row[POST_ONLY_MODEL_NAME]
            - parent_row[POST_ONLY_MODEL_NAME]
        )
        full_history_delta = float(
            perturbed_row[FULL_HISTORY_MODEL_NAME]
            - parent_row[FULL_HISTORY_MODEL_NAME]
        )

        records.append(
            {
                "parent_scenario_name": parent_name,
                "perturbed_scenario_name": str(
                    perturbed_row["scenario_name"]
                ),
                "scenario_class": str(
                    perturbed_row["scenario_class"]
                ),
                "perturbation_type": str(
                    perturbed_row["perturbation_type"]
                ),
                "perturbation_scale": scale,
                "expected_direction": expected_direction,
                "directional_status": directional_status,
                "original_official_prediction_kwh": original_official,
                "perturbed_official_prediction_kwh": perturbed_official,
                "official_delta_kwh": official_delta_kwh,
                "official_delta_pct": official_delta_pct,
                "post_only_delta_kwh": post_only_delta,
                "full_history_delta_kwh": full_history_delta,
                "original_branch_disagreement_pct": float(
                    parent_row["branch_disagreement_pct"]
                ),
                "perturbed_branch_disagreement_pct": float(
                    perturbed_row["branch_disagreement_pct"]
                ),
                "branch_disagreement_delta_pct_points": float(
                    perturbed_row["branch_disagreement_pct"]
                    - parent_row["branch_disagreement_pct"]
                ),
                "original_operational_range_status": str(
                    parent_row["operational_range_status"]
                ),
                "perturbed_operational_range_status": str(
                    perturbed_row["operational_range_status"]
                ),
                "original_warning": str(parent_row["warning"]),
                "perturbed_warning": str(perturbed_row["warning"]),
            }
        )

    return pd.DataFrame(records).sort_values(
        "parent_scenario_name"
    ).reset_index(drop=True)


def add_hard_check(
    records: list[dict[str, object]],
    *,
    check_name: str,
    passed: bool,
    details: str,
) -> None:
    records.append(
        {
            "check_name": check_name,
            "status": "passed" if passed else "failed",
            "details": details,
        }
    )


def run_hard_checks(
    *,
    scenarios_df: pd.DataFrame,
    schema_summary_df: pd.DataFrame,
    operational_summary_df: pd.DataFrame,
    operational_checks_df: pd.DataFrame,
    predictions_by_model: dict[str, np.ndarray],
    post_only_predictions: np.ndarray,
    full_history_predictions: np.ndarray,
    ensemble_70_30_predictions: np.ndarray,
    ensemble_60_40_predictions: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    add_hard_check(
        records,
        check_name="scenario_names_unique",
        passed=scenarios_df["scenario_name"].is_unique,
        details="Each scenario must have a unique name.",
    )

    add_hard_check(
        records,
        check_name="all_schema_checks_passed",
        passed=bool(
            (schema_summary_df["schema_status"] == "valid").all()
        ),
        details="Every scenario must be structurally valid.",
    )

    scenario_index = scenarios_df.set_index("scenario_name")
    baseline_row = scenario_index.loc["baseline_median_weekday"]
    weekend_row = scenario_index.loc["weekend_moderate_production"]

    add_hard_check(
        records,
        check_name="baseline_anchor_is_weekday",
        passed=int(float(baseline_row["is_weekend"])) == 0,
        details=(
            "The baseline_median_weekday scenario must use a "
            "weekday historical anchor."
        ),
    )

    add_hard_check(
        records,
        check_name="weekend_anchor_is_weekend",
        passed=int(float(weekend_row["is_weekend"])) == 1,
        details=(
            "The weekend_moderate_production scenario must use a "
            "weekend historical anchor."
        ),
    )

    add_hard_check(
        records,
        check_name="baseline_and_weekend_anchors_are_distinct",
        passed=(
            str(baseline_row["anchor_date"])
            != str(weekend_row["anchor_date"])
        ),
        details=(
            "The weekday baseline and weekend scenario must use "
            "different historical anchor dates."
        ),
    )

    origin_counts = scenarios_df["scenario_origin"].value_counts()

    add_hard_check(
        records,
        check_name="historical_reference_count_correct",
        passed=int(origin_counts.get("historical_reference", 0))
        == EXPECTED_HISTORICAL_REFERENCE_COUNT,
        details=(
            "The evaluation must contain exactly eight historical "
            "reference replays."
        ),
    )

    add_hard_check(
        records,
        check_name="local_perturbation_count_correct",
        passed=int(
            origin_counts.get("local_synthetic_perturbation", 0)
        ) == EXPECTED_LOCAL_PERTURBATION_COUNT,
        details=(
            "The evaluation must contain exactly eight paired local "
            "synthetic perturbations."
        ),
    )

    add_hard_check(
        records,
        check_name="stress_perturbation_count_correct",
        passed=int(origin_counts.get("synthetic_perturbation", 0))
        == EXPECTED_STRESS_PERTURBATION_COUNT,
        details=(
            "The evaluation must contain exactly four adversarial or "
            "out-of-distribution perturbations."
        ),
    )

    historical_names = set(
        scenarios_df.loc[
            scenarios_df["scenario_origin"] == "historical_reference",
            "scenario_name",
        ].astype(str)
    )
    local_df = scenarios_df[
        scenarios_df["scenario_origin"]
        == "local_synthetic_perturbation"
    ]
    local_parent_counts = (
        local_df["parent_scenario_name"].astype(str).value_counts()
    )

    local_pairing_valid = bool(
        set(local_parent_counts.index) == historical_names
        and (local_parent_counts == 1).all()
    )
    add_hard_check(
        records,
        check_name="local_perturbations_paired_one_to_one",
        passed=local_pairing_valid,
        details=(
            "Every historical reference must have exactly one paired "
            "local synthetic perturbation."
        ),
    )

    local_values_valid = True
    local_context_preserved = True

    for _, local_row in local_df.iterrows():
        parent_name = str(local_row["parent_scenario_name"])

        if parent_name not in scenario_index.index:
            local_values_valid = False
            local_context_preserved = False
            continue

        parent_row = scenario_index.loc[parent_name]
        scale = float(local_row["perturbation_scale"])
        expected_updates = build_local_activity_updates(
            parent_row,
            scale,
        )

        for feature, expected_value in expected_updates.items():
            if not np.isclose(
                float(local_row[feature]),
                expected_value,
            ):
                local_values_valid = False

        for feature in [
            "orders",
            "total_pallets",
            "avg_brix",
            *CALENDAR_FEATURES,
        ]:
            if not np.isclose(
                float(local_row[feature]),
                float(parent_row[feature]),
            ):
                local_context_preserved = False

    add_hard_check(
        records,
        check_name="local_perturbation_values_correct",
        passed=local_values_valid,
        details=(
            "Each paired local perturbation must match its recorded "
            "operating scale and preserve confirmed derived-feature logic."
        ),
    )

    add_hard_check(
        records,
        check_name="local_perturbation_context_preserved",
        passed=local_context_preserved,
        details=(
            "Local perturbations must preserve orders, calendar context, "
            "average Brix, and total pallets from their historical parents."
        ),
    )

    local_orders_preserved = True

    for _, local_row in local_df.iterrows():
        parent_name = str(local_row["parent_scenario_name"])

        if parent_name not in scenario_index.index:
            local_orders_preserved = False
            continue

        parent_row = scenario_index.loc[parent_name]

        if not np.isclose(
            float(local_row["orders"]),
            float(parent_row["orders"]),
        ):
            local_orders_preserved = False

    add_hard_check(
        records,
        check_name="local_perturbation_orders_preserved",
        passed=local_orders_preserved,
        details=(
            "Paired local operating-scale perturbations must preserve "
            "the discrete order count."
        ),
    )

    for model_name, predictions in predictions_by_model.items():
        predictions_array = np.asarray(predictions, dtype=float)

        add_hard_check(
            records,
            check_name=f"{model_name}_predictions_finite",
            passed=bool(np.isfinite(predictions_array).all()),
            details="All predictions must be finite.",
        )

        add_hard_check(
            records,
            check_name=f"{model_name}_predictions_nonnegative",
            passed=bool((predictions_array >= 0).all()),
            details="All energy predictions must be nonnegative.",
        )

    expected_70_30 = (
        0.7 * np.asarray(post_only_predictions, dtype=float)
        + 0.3 * np.asarray(full_history_predictions, dtype=float)
    )
    expected_60_40 = (
        0.6 * np.asarray(post_only_predictions, dtype=float)
        + 0.4 * np.asarray(full_history_predictions, dtype=float)
    )

    add_hard_check(
        records,
        check_name="ensemble_70_30_math_correct",
        passed=bool(
            np.allclose(
                ensemble_70_30_predictions,
                expected_70_30,
            )
        ),
        details=(
            "The official ensemble must equal "
            "0.7 * post-only + 0.3 * full-history."
        ),
    )

    add_hard_check(
        records,
        check_name="ensemble_60_40_math_correct",
        passed=bool(
            np.allclose(
                ensemble_60_40_predictions,
                expected_60_40,
            )
        ),
        details=(
            "The sensitivity ensemble must equal "
            "0.6 * post-only + 0.4 * full-history."
        ),
    )

    expected_yield_ratio = (
        scenarios_df["total_kg"].astype(float)
        / scenarios_df["total_nominal_kg"].astype(float)
    )

    add_hard_check(
        records,
        check_name="yield_ratio_consistent",
        passed=bool(
            np.allclose(
                scenarios_df[
                    "yield_ratio_actual_over_nominal"
                ].astype(float),
                expected_yield_ratio,
            )
        ),
        details=(
            "yield_ratio_actual_over_nominal must equal "
            "total_kg / total_nominal_kg."
        ),
    )

    orders_are_integer_counts = bool(
        scenarios_df["orders"].map(is_integer_like).all()
        and (scenarios_df["orders"].astype(float) >= 0).all()
    )

    add_hard_check(
        records,
        check_name="orders_are_nonnegative_integer_counts",
        passed=orders_are_integer_counts,
        details=(
            "Every scenario must use a nonnegative integer-like "
            "order count."
        ),
    )

    constant_reference_rows = operational_checks_df[
        operational_checks_df["development_unique_count"] == 1
    ]

    constant_statuses_valid = bool(
        constant_reference_rows.empty
        or constant_reference_rows["range_status"].isin(
            {
                "constant_in_development",
                "outside_observed_range",
            }
        ).all()
    )

    add_hard_check(
        records,
        check_name="constant_features_classified_explicitly",
        passed=constant_statuses_valid,
        details=(
            "Constant development features must be labeled "
            "constant_in_development when matched or "
            "outside_observed_range when changed."
        ),
    )

    ood_names = {
        "outside_observed_high_activity",
        "near_shutdown_low_activity",
    }

    ood_summary = operational_summary_df[
        operational_summary_df["scenario_name"].isin(
            ood_names
        )
    ]

    add_hard_check(
        records,
        check_name="ood_scenarios_flagged_outside_range",
        passed=bool(
            (
                ood_summary["operational_range_status"]
                == "outside_observed_range"
            ).all()
            and len(ood_summary) == len(ood_names)
        ),
        details=(
            "Deliberate OOD scenarios must be flagged "
            "outside the observed operational range."
        ),
    )

    ood_order_values = scenarios_df.loc[
        scenarios_df["scenario_name"].isin(ood_names),
        "orders",
    ]

    add_hard_check(
        records,
        check_name="ood_orders_are_integer_counts",
        passed=bool(
            ood_order_values.map(is_integer_like).all()
            and (ood_order_values.astype(float) >= 0).all()
        ),
        details=(
            "Deliberate OOD scenarios must use nonnegative integer-like "
            "order counts."
        ),
    )

    hard_checks_df = pd.DataFrame(records)
    failed_checks = hard_checks_df[
        hard_checks_df["status"] == "failed"
    ]

    if not failed_checks.empty:
        failed_names = ", ".join(
            failed_checks["check_name"].astype(str)
        )
        raise AssertionError(
            f"Hard stress-test checks failed: {failed_names}"
        )

    return hard_checks_df


def add_behavioral_check(
    records: list[dict[str, object]],
    *,
    check_name: str,
    status: str,
    observed_value: object,
    expectation: str,
    message: str,
) -> None:
    records.append(
        {
            "check_name": check_name,
            "status": status,
            "observed_value": observed_value,
            "expectation": expectation,
            "message": message,
        }
    )


def build_behavioral_checks(
    summary_df: pd.DataFrame,
    paired_sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    indexed = summary_df.set_index("scenario_name")
    official_column = OFFICIAL_ENSEMBLE_NAME

    pairwise_expectations = [
        (
            "high_activity_above_baseline",
            "high_production_high_hours",
            "baseline_median_weekday",
            "High-activity scenario should predict more than baseline.",
        ),
        (
            "baseline_above_low_activity",
            "baseline_median_weekday",
            "low_production_low_hours",
            "Baseline should predict more than low activity.",
        ),
        (
            "high_activity_above_low_activity",
            "high_production_high_hours",
            "low_production_low_hours",
            "High activity should predict more than low activity.",
        ),
        (
            "baseline_above_near_shutdown",
            "baseline_median_weekday",
            "near_shutdown_low_activity",
            "Baseline should predict more than near shutdown.",
        ),
    ]

    for (
        check_name,
        higher_scenario,
        lower_scenario,
        expectation,
    ) in pairwise_expectations:
        higher_prediction = float(
            indexed.loc[higher_scenario, official_column]
        )
        lower_prediction = float(
            indexed.loc[lower_scenario, official_column]
        )
        passed = higher_prediction > lower_prediction

        add_behavioral_check(
            records,
            check_name=check_name,
            status="passed" if passed else "diagnostic_warning",
            observed_value=(
                f"{higher_scenario}={higher_prediction:.3f}; "
                f"{lower_scenario}={lower_prediction:.3f}"
            ),
            expectation=expectation,
            message=(
                "Directional expectation was satisfied."
                if passed
                else (
                    "Directional expectation was not satisfied. "
                    "This is a diagnostic finding, not an automatic "
                    "implementation failure."
                )
            ),
        )

    baseline_prediction = float(
        indexed.loc[
            "baseline_median_weekday",
            official_column,
        ]
    )
    high_orders_prediction = float(
        indexed.loc[
            "high_orders_moderate_production",
            official_column,
        ]
    )
    relative_change = (
        abs(high_orders_prediction - baseline_prediction)
        / max(abs(baseline_prediction), 1e-12)
        * 100.0
    )

    add_behavioral_check(
        records,
        check_name="high_orders_vs_baseline_material_change",
        status="information",
        observed_value=f"{relative_change:.3f}%",
        expectation=(
            "Report whether high orders materially change the "
            "prediction when the anchored operating profile is similar."
        ),
        message=(
            "This is descriptive only. No monotonic relationship "
            "is enforced for orders."
        ),
    )

    maximum_absolute_row = summary_df.loc[
        summary_df["branch_disagreement_kwh"].idxmax()
    ]

    add_behavioral_check(
        records,
        check_name="maximum_absolute_branch_disagreement",
        status="information",
        observed_value=(
            f"{maximum_absolute_row['scenario_name']}: "
            f"{maximum_absolute_row['branch_disagreement_kwh']:.3f} kWh, "
            f"{maximum_absolute_row['branch_disagreement_pct']:.3f}%"
        ),
        expectation=(
            "Identify the scenario with the largest absolute "
            "difference between the two modeling branches."
        ),
        message=(
            "Absolute branch disagreement is a model-disagreement "
            "diagnostic, not a confidence interval or error probability."
        ),
    )

    maximum_relative_row = summary_df.loc[
        summary_df["branch_disagreement_pct"].idxmax()
    ]

    add_behavioral_check(
        records,
        check_name="maximum_relative_branch_disagreement",
        status="information",
        observed_value=(
            f"{maximum_relative_row['scenario_name']}: "
            f"{maximum_relative_row['branch_disagreement_kwh']:.3f} kWh, "
            f"{maximum_relative_row['branch_disagreement_pct']:.3f}%"
        ),
        expectation=(
            "Identify the scenario with the largest disagreement "
            "relative to the average component prediction."
        ),
        message=(
            "Relative branch disagreement is a model-disagreement "
            "diagnostic, not a confidence interval or error probability."
        ),
    )

    for _, pair in paired_sensitivity_df.iterrows():
        directional_status = str(pair["directional_status"])

        if directional_status == "passed":
            check_status = "passed"
            message = (
                "The local forecast response followed the expected "
                "operating-scale direction."
            )
        elif directional_status == "locally_flat":
            check_status = "information"
            message = (
                "The local forecast was unchanged within numerical "
                "tolerance, indicating local tree-leaf insensitivity."
            )
        else:
            check_status = "diagnostic_warning"
            message = (
                "The local forecast response moved opposite to the "
                "designed operating-scale direction. This is a diagnostic "
                "finding, not an implementation failure."
            )

        add_behavioral_check(
            records,
            check_name=(
                "local_sensitivity_"
                f"{pair['parent_scenario_name']}"
            ),
            status=check_status,
            observed_value=(
                f"{pair['perturbation_type']}: "
                f"delta={pair['official_delta_kwh']:.3f} kWh, "
                f"{pair['official_delta_pct']:.3f}%"
            ),
            expectation=(
                "A coherent local operating-scale increase should normally "
                "increase the forecast, while a coherent decrease should "
                "normally reduce it."
            ),
            message=message,
        )

    paired_warning_count = int(
        (
            paired_sensitivity_df["directional_status"]
            == "diagnostic_warning"
        ).sum()
    )
    paired_flat_count = int(
        (
            paired_sensitivity_df["directional_status"]
            == "locally_flat"
        ).sum()
    )
    paired_pass_count = int(
        (
            paired_sensitivity_df["directional_status"]
            == "passed"
        ).sum()
    )

    add_behavioral_check(
        records,
        check_name="paired_local_sensitivity_summary",
        status=(
            "information"
            if paired_warning_count == 0
            else "diagnostic_warning"
        ),
        observed_value=(
            f"passed={paired_pass_count}; flat={paired_flat_count}; "
            f"warnings={paired_warning_count}"
        ),
        expectation=(
            "Summarize the eight paired local sensitivity responses."
        ),
        message=(
            "Paired perturbations are behavioral diagnostics and do not "
            "provide predictive-accuracy or causal evidence."
        ),
    )

    return pd.DataFrame(records)


def build_warning_text(row: pd.Series) -> str:
    warnings: list[str] = []

    if row["operational_range_status"] == "outside_observed_range":
        warnings.append(
            "Inputs exceed the observed operational development range."
        )
    elif (
        row["operational_range_status"]
        == "development_distribution_tail"
    ):
        warnings.append(
            "One or more inputs fall in the tails of the "
            "post-installation development distribution."
        )

    if (
        row["calendar_coverage_status"]
        == "contains_unseen_calendar_values"
    ):
        warnings.append(
            "One or more calendar values were not represented "
            "during development."
        )

    if row["branch_disagreement_status"] == "high":
        warnings.append(
            "Post-only and full-history branches disagree strongly."
        )
    elif row["branch_disagreement_status"] == "moderate":
        warnings.append(
            "Post-only and full-history branches show "
            "moderate disagreement."
        )

    if row["scenario_class"] == "adversarial":
        warnings.append(
            "Scenario is deliberately adversarial."
        )

    if row["scenario_class"] == "out_of_distribution":
        warnings.append(
            "Scenario is intentionally out of distribution."
        )

    return " ".join(warnings)


def build_plot_labels(plot_df: pd.DataFrame) -> pd.Series:
    origin_labels = {
        "historical_reference": "historical",
        "local_synthetic_perturbation": "local",
        "synthetic_perturbation": "stress",
    }

    return (
        plot_df["scenario_name"].astype(str)
        + " ["
        + plot_df["scenario_origin"].map(origin_labels).fillna("unknown")
        + "]"
    )


def save_official_prediction_plot(
    summary_df: pd.DataFrame,
) -> None:
    plot_df = summary_df.sort_values(
        OFFICIAL_ENSEMBLE_NAME
    ).copy()
    labels = build_plot_labels(plot_df)
    figure_height = max(8.0, len(plot_df) * 0.42)

    plt.figure(figsize=(14, figure_height))
    plt.barh(labels, plot_df[OFFICIAL_ENSEMBLE_NAME])
    plt.title(
        "Behavioral Scenario Evaluation: Official 70/30 Predictions"
    )
    plt.xlabel("Predicted active energy (kWh)")
    plt.ylabel("Scenario")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR
        / "official_70_30_scenario_predictions.png"
    )
    plt.close()


def save_branch_disagreement_kwh_plot(
    summary_df: pd.DataFrame,
) -> None:
    plot_df = summary_df.sort_values(
        "branch_disagreement_kwh"
    ).copy()
    labels = build_plot_labels(plot_df)
    figure_height = max(8.0, len(plot_df) * 0.42)

    plt.figure(figsize=(14, figure_height))
    plt.barh(labels, plot_df["branch_disagreement_kwh"])
    plt.title(
        "Behavioral Scenario Evaluation: Branch Disagreement"
    )
    plt.xlabel("Absolute disagreement (kWh)")
    plt.ylabel("Scenario")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "branch_disagreement_kwh.png"
    )
    plt.close()


def save_branch_disagreement_pct_plot(
    summary_df: pd.DataFrame,
) -> None:
    plot_df = summary_df.sort_values(
        "branch_disagreement_pct"
    ).copy()
    labels = build_plot_labels(plot_df)
    figure_height = max(8.0, len(plot_df) * 0.42)

    plt.figure(figsize=(14, figure_height))
    plt.barh(labels, plot_df["branch_disagreement_pct"])
    plt.title(
        "Behavioral Scenario Evaluation: Relative Branch Disagreement"
    )
    plt.xlabel("Disagreement relative to component mean (%)")
    plt.ylabel("Scenario")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "branch_disagreement_pct.png"
    )
    plt.close()


def save_paired_sensitivity_plot(
    paired_sensitivity_df: pd.DataFrame,
) -> None:
    plot_df = paired_sensitivity_df.sort_values(
        "official_delta_pct"
    ).copy()

    plt.figure(figsize=(13, 7))
    plt.barh(
        plot_df["parent_scenario_name"],
        plot_df["official_delta_pct"],
    )
    plt.axvline(0.0, linewidth=1.0)
    plt.title(
        "Paired Local Operating-Scale Sensitivity: Official 70/30 Prediction Change"
    )
    plt.xlabel("Prediction change relative to historical reference (%)")
    plt.ylabel("Historical reference scenario")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "paired_local_sensitivity_delta_pct.png"
    )
    plt.close()


def log_behavioral_evaluation_to_mlflow(
    *,
    scenarios_df: pd.DataFrame,
    schema_summary_df: pd.DataFrame,
    operational_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paired_sensitivity_df: pd.DataFrame,
    hard_checks_df: pd.DataFrame,
    behavioral_checks_df: pd.DataFrame,
) -> None:
    """Log the finalized Version 2.1 diagnostic run to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    historical_reference_count = int(
        (
            scenarios_df["scenario_origin"]
            == "historical_reference"
        ).sum()
    )
    local_perturbation_count = int(
        (
            scenarios_df["scenario_origin"]
            == "local_synthetic_perturbation"
        ).sum()
    )
    stress_perturbation_count = int(
        (
            scenarios_df["scenario_origin"]
            == "synthetic_perturbation"
        ).sum()
    )

    hard_check_failure_count = int(
        (hard_checks_df["status"] == "failed").sum()
    )
    hard_check_pass_count = int(
        (hard_checks_df["status"] == "passed").sum()
    )
    behavioral_warning_count = int(
        (
            behavioral_checks_df["status"]
            == "diagnostic_warning"
        ).sum()
    )
    invalid_schema_count = int(
        (schema_summary_df["schema_status"] != "valid").sum()
    )

    outside_range_count = int(
        (
            operational_summary_df["operational_range_status"]
            == "outside_observed_range"
        ).sum()
    )
    distribution_tail_count = int(
        (
            operational_summary_df["operational_range_status"]
            == "development_distribution_tail"
        ).sum()
    )
    typical_range_count = int(
        (
            operational_summary_df["operational_range_status"]
            == "inside_typical_development_range"
        ).sum()
    )

    moderate_disagreement_count = int(
        (
            summary_df["branch_disagreement_status"]
            == "moderate"
        ).sum()
    )
    high_disagreement_count = int(
        (
            summary_df["branch_disagreement_status"]
            == "high"
        ).sum()
    )

    deliberate_ood_mask = (
        summary_df["scenario_class"]
        == "out_of_distribution"
    )
    deliberate_ood_count = int(deliberate_ood_mask.sum())
    deliberate_ood_correctly_flagged_count = int(
        (
            deliberate_ood_mask
            & (
                summary_df["operational_range_status"]
                == "outside_observed_range"
            )
        ).sum()
    )

    local_directional_pass_count = int(
        (
            paired_sensitivity_df["directional_status"]
            == "passed"
        ).sum()
    )
    local_directional_flat_count = int(
        (
            paired_sensitivity_df["directional_status"]
            == "locally_flat"
        ).sum()
    )
    local_directional_warning_count = int(
        (
            paired_sensitivity_df["directional_status"]
            == "diagnostic_warning"
        ).sum()
    )

    max_absolute_index = summary_df[
        "branch_disagreement_kwh"
    ].idxmax()
    max_relative_index = summary_df[
        "branch_disagreement_pct"
    ].idxmax()

    maximum_absolute_disagreement = float(
        summary_df.loc[
            max_absolute_index,
            "branch_disagreement_kwh",
        ]
    )
    maximum_relative_disagreement = float(
        summary_df.loc[
            max_relative_index,
            "branch_disagreement_pct",
        ]
    )

    parameters = {
        "official_post_only_weight": 0.70,
        "official_full_history_weight": 0.30,
        "sensitivity_post_only_weight": 0.60,
        "sensitivity_full_history_weight": 0.40,
        "moderate_disagreement_threshold_pct": (
            MODERATE_DISAGREEMENT_THRESHOLD_PCT
        ),
        "high_disagreement_threshold_pct": (
            HIGH_DISAGREEMENT_THRESHOLD_PCT
        ),
        "distribution_lower_quantile": (
            DISTRIBUTION_LOWER_QUANTILE
        ),
        "distribution_upper_quantile": (
            DISTRIBUTION_UPPER_QUANTILE
        ),
        "local_operating_scale_up": (
            LOCAL_ACTIVITY_INCREASE_SCALE
        ),
        "local_operating_scale_down": (
            LOCAL_ACTIVITY_DECREASE_SCALE
        ),
        "scenario_reference_period": (
            "post_installation_development_window"
        ),
        "scenario_construction_method": (
            "historical_replays_paired_local_operating_scale_"
            "perturbations_and_adversarial_ood_cases"
        ),
        "operational_range_method": (
            "min_max_ood_plus_q10_q90_distribution_tails"
        ),
        "orders_rule": (
            "nonnegative_integer_count_preserved_in_local_pairs"
        ),
        "synthetic_accuracy_claim": "none",
    }

    metrics = {
        "scenario_count": float(len(scenarios_df)),
        "historical_reference_scenario_count": float(
            historical_reference_count
        ),
        "local_synthetic_perturbation_count": float(
            local_perturbation_count
        ),
        "stress_synthetic_perturbation_count": float(
            stress_perturbation_count
        ),
        "hard_check_pass_count": float(hard_check_pass_count),
        "hard_check_failure_count": float(
            hard_check_failure_count
        ),
        "behavioral_warning_count": float(
            behavioral_warning_count
        ),
        "invalid_schema_count": float(invalid_schema_count),
        "inside_typical_range_scenario_count": float(
            typical_range_count
        ),
        "development_distribution_tail_scenario_count": float(
            distribution_tail_count
        ),
        "outside_observed_range_scenario_count": float(
            outside_range_count
        ),
        "moderate_branch_disagreement_scenario_count": float(
            moderate_disagreement_count
        ),
        "high_branch_disagreement_scenario_count": float(
            high_disagreement_count
        ),
        "deliberate_ood_scenario_count": float(
            deliberate_ood_count
        ),
        "deliberate_ood_correctly_flagged_count": float(
            deliberate_ood_correctly_flagged_count
        ),
        "local_directional_pass_count": float(
            local_directional_pass_count
        ),
        "local_directional_flat_count": float(
            local_directional_flat_count
        ),
        "local_directional_warning_count": float(
            local_directional_warning_count
        ),
        "maximum_absolute_branch_disagreement_kwh": (
            maximum_absolute_disagreement
        ),
        "maximum_relative_branch_disagreement_pct": (
            maximum_relative_disagreement
        ),
    }

    tags = {
        "run_type": "behavioral_scenario_evaluation",
        "model_role": "official_champion_diagnostic",
        "model_version": "2.0_ensemble_70_30",
        "stress_test_version": "2.1",
        "status": "official",
        "accuracy_evidence": (
            "none_behavioral_diagnostics_only"
        ),
        "maximum_absolute_disagreement_scenario": str(
            summary_df.loc[
                max_absolute_index,
                "scenario_name",
            ]
        ),
        "maximum_relative_disagreement_scenario": str(
            summary_df.loc[
                max_relative_index,
                "scenario_name",
            ]
        ),
    }

    with mlflow.start_run(
        run_name=MLFLOW_RUN_NAME,
        tags=tags,
    ) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        if REPORT_PATH.exists():
            mlflow.log_artifact(
                str(REPORT_PATH),
                artifact_path="behavioral_evaluation",
            )

        if FIGURES_DIR.exists():
            mlflow.log_artifacts(
                str(FIGURES_DIR),
                artifact_path="behavioral_evaluation/figures",
            )

        source_script_path = Path(
            "src/stress_test_ensemble.py"
        )
        if source_script_path.exists():
            mlflow.log_artifact(
                str(source_script_path),
                artifact_path="source",
            )

        test_script_path = Path(
            "tests/test_stress_test_ensemble.py"
        )
        if test_script_path.exists():
            mlflow.log_artifact(
                str(test_script_path),
                artifact_path="source/tests",
            )

        run_id = run.info.run_id

    print()
    print("MLflow logging completed.")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"MLflow run name: {MLFLOW_RUN_NAME}")
    print(f"MLflow run ID: {run_id}")


def main(log_mlflow: bool = False) -> None:
    ensure_output_directories()

    df = load_modeling_data()

    post_only_features = get_feature_columns("full")
    full_history_features = get_feature_columns(
        "vif_auto_full_history"
    )
    all_model_features = list(
        dict.fromkeys(
            post_only_features + full_history_features
        )
    )

    missing_columns = [
        column
        for column in [
            DATE_COL,
            TARGET_COL,
            *all_model_features,
        ]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    post_development_df = filter_date_range(
        df,
        start_date=INTERVENTION_DATE,
        end_date=POST_VALIDATION_END,
    )
    full_history_development_df = filter_date_range(
        df,
        start_date=None,
        end_date=FULL_HISTORY_VALIDATION_END,
    )

    if post_development_df.empty:
        raise ValueError("Post-development data is empty.")

    if full_history_development_df.empty:
        raise ValueError(
            "Full-history development data is empty."
        )

    scenarios_df = create_behavioral_scenarios(
        reference_df=post_development_df,
        feature_columns=all_model_features,
    )

    schema_checks_df = calculate_schema_checks(
        scenarios_df=scenarios_df,
        required_feature_columns=all_model_features,
    )
    schema_summary_df = summarize_schema_checks(
        schema_checks_df
    )

    operational_checks_df = (
        calculate_operational_range_checks(
            scenarios_df=scenarios_df,
            reference_df=post_development_df,
        )
    )
    operational_summary_df = (
        summarize_operational_range_checks(
            operational_checks_df
        )
    )

    calendar_checks_df = (
        calculate_calendar_coverage_checks(
            scenarios_df=scenarios_df,
            reference_df=post_development_df,
        )
    )
    calendar_summary_df = (
        summarize_calendar_coverage_checks(
            calendar_checks_df
        )
    )

    post_only_model = build_post_only_champion_model()
    full_history_model = (
        build_full_history_champion_model()
    )

    post_only_model.fit(
        post_development_df[post_only_features],
        post_development_df[TARGET_COL],
    )
    full_history_model.fit(
        full_history_development_df[
            full_history_features
        ],
        full_history_development_df[TARGET_COL],
    )

    post_only_predictions = post_only_model.predict(
        scenarios_df[post_only_features]
    )
    full_history_predictions = (
        full_history_model.predict(
            scenarios_df[full_history_features]
        )
    )

    ensemble_70_30_predictions = (
        calculate_weighted_ensemble_predictions(
            post_only_predictions=post_only_predictions,
            full_history_predictions=(
                full_history_predictions
            ),
            post_only_weight=0.7,
        )
    )
    ensemble_60_40_predictions = (
        calculate_weighted_ensemble_predictions(
            post_only_predictions=post_only_predictions,
            full_history_predictions=(
                full_history_predictions
            ),
            post_only_weight=0.6,
        )
    )

    predictions_by_model = {
        POST_ONLY_MODEL_NAME: post_only_predictions,
        FULL_HISTORY_MODEL_NAME: full_history_predictions,
        OFFICIAL_ENSEMBLE_NAME: ensemble_70_30_predictions,
        SENSITIVITY_ENSEMBLE_NAME: ensemble_60_40_predictions,
    }

    predictions_df = build_predictions_table(
        scenarios_df=scenarios_df,
        predictions_by_model=predictions_by_model,
    )
    summary_df = build_wide_prediction_summary(
        predictions_df
    )

    summary_df = (
        summary_df.merge(
            schema_summary_df,
            on="scenario_name",
            how="left",
        )
        .merge(
            operational_summary_df,
            on="scenario_name",
            how="left",
        )
        .merge(
            calendar_summary_df,
            on="scenario_name",
            how="left",
        )
    )

    summary_df["warning"] = summary_df.apply(
        build_warning_text,
        axis=1,
    )

    paired_sensitivity_df = build_paired_sensitivity_table(
        summary_df
    )

    hard_checks_df = run_hard_checks(
        scenarios_df=scenarios_df,
        schema_summary_df=schema_summary_df,
        operational_summary_df=operational_summary_df,
        operational_checks_df=operational_checks_df,
        predictions_by_model=predictions_by_model,
        post_only_predictions=post_only_predictions,
        full_history_predictions=full_history_predictions,
        ensemble_70_30_predictions=(
            ensemble_70_30_predictions
        ),
        ensemble_60_40_predictions=(
            ensemble_60_40_predictions
        ),
    )

    behavioral_checks_df = build_behavioral_checks(
        summary_df,
        paired_sensitivity_df,
    )

    operational_ranges_df = (
        calculate_operational_reference_ranges(
            reference_df=post_development_df
        )
    )

    scenario_classification_df = scenarios_df[
        [
            "scenario_id",
            "scenario_name",
            "scenario_class",
            "scenario_origin",
            "parent_scenario_name",
            "perturbation_type",
            "perturbation_scale",
            "description",
            "construction_method",
            "anchor_date",
            "scenario_date",
        ]
    ].copy()

    metadata_records: list[dict[str, Any]] = [
        {
            "item": "stress_test_version",
            "value": "2.1",
        },
        {
            "item": "stress_test_type",
            "value": (
                "behavioral_scenario_evaluation_and_synthetic_stress_test"
            ),
        },
        {
            "item": "status",
            "value": "official_behavioral_stress_test",
        },
        {
            "item": "scenario_class_note",
            "value": (
                "Scenario class is a manually assigned test-design category "
                "describing why a scenario was included. Operational range "
                "status is computed independently from the post-installation "
                "development distribution."
            ),
        },
        {
            "item": "scenario_origin_note",
            "value": (
                "scenario_origin identifies whether the final input is a "
                "historical reference replay, a paired local synthetic "
                "perturbation, or an adversarial/OOD synthetic perturbation"
            ),
        },
        {
            "item": "local_perturbation_rule",
            "value": (
                "scale total_kg, total_nominal_kg, total_brix_units, "
                "and total_hours by approximately 5%; preserve orders, "
                "calendar context, avg_brix, and total_pallets; recalculate "
                "yield_ratio_actual_over_nominal"
            ),
        },
        {
            "item": "important_note",
            "value": (
                "Historical reference replays and synthetic "
                "perturbations are behavioral diagnostics only. Historical "
                "replays are not out-of-sample because the model branches "
                "were fitted using the same development period. Synthetic "
                "perturbations have no observed target values. Neither group "
                "is evidence of out-of-sample predictive accuracy."
            ),
        },
        {
            "item": "scenario_construction",
            "value": (
                "eight historical reference replays, eight paired local "
                "operating-scale perturbations, and four adversarial or "
                "out-of-distribution perturbations"
            ),
        },
        {
            "item": "operational_familiarity_reference",
            "value": (
                "post-installation development window because the "
                "forecasting service targets the current operating regime"
            ),
        },
        {
            "item": "operational_range_rule",
            "value": (
                "outside min/max = outside_observed_range; inside min/max "
                "but below q10 or above q90 = development_distribution_tail; "
                "q10 to q90 = inside_typical_development_range"
            ),
        },
        {
            "item": "distribution_lower_quantile",
            "value": DISTRIBUTION_LOWER_QUANTILE,
        },
        {
            "item": "distribution_upper_quantile",
            "value": DISTRIBUTION_UPPER_QUANTILE,
        },
        {
            "item": "constant_feature_rule",
            "value": (
                "features with one unique development value are labeled "
                "constant_in_development when matched and outside_observed_range "
                "when changed"
            ),
        },
        {
            "item": "confirmed_derived_feature_rule",
            "value": (
                "yield_ratio_actual_over_nominal = "
                "total_kg / total_nominal_kg"
            ),
        },
        {
            "item": "orders_data_type_rule",
            "value": (
                "orders is treated as a nonnegative integer count; "
                "paired local operating-scale perturbations preserve it, "
                "and OOD scenarios use integer values"
            ),
        },
        {
            "item": "official_model",
            "value": OFFICIAL_ENSEMBLE_NAME,
        },
        {
            "item": "sensitivity_model",
            "value": SENSITIVITY_ENSEMBLE_NAME,
        },
        {
            "item": "high_disagreement_threshold_pct",
            "value": HIGH_DISAGREEMENT_THRESHOLD_PCT,
        },
        {
            "item": "moderate_disagreement_threshold_pct",
            "value": MODERATE_DISAGREEMENT_THRESHOLD_PCT,
        },
        {
            "item": "disagreement_threshold_note",
            "value": (
                "Thresholds are heuristic monitoring aids, "
                "not calibrated uncertainty probabilities."
            ),
        },
        {
            "item": "post_development_rows",
            "value": len(post_development_df),
        },
        {
            "item": "full_history_development_rows",
            "value": len(full_history_development_df),
        },
        {
            "item": "scenario_count",
            "value": len(scenarios_df),
        },
        {
            "item": "historical_reference_scenario_count",
            "value": int(
                (
                    scenarios_df["scenario_origin"]
                    == "historical_reference"
                ).sum()
            ),
        },
        {
            "item": "local_synthetic_perturbation_count",
            "value": int(
                (
                    scenarios_df["scenario_origin"]
                    == "local_synthetic_perturbation"
                ).sum()
            ),
        },
        {
            "item": "stress_synthetic_perturbation_count",
            "value": int(
                (
                    scenarios_df["scenario_origin"]
                    == "synthetic_perturbation"
                ).sum()
            ),
        },
        {
            "item": "paired_directional_warning_count",
            "value": int(
                (
                    paired_sensitivity_df["directional_status"]
                    == "diagnostic_warning"
                ).sum()
            ),
        },
        {
            "item": "paired_locally_flat_count",
            "value": int(
                (
                    paired_sensitivity_df["directional_status"]
                    == "locally_flat"
                ).sum()
            ),
        },
        {
            "item": "hard_check_failure_count",
            "value": int(
                (
                    hard_checks_df["status"] == "failed"
                ).sum()
            ),
        },
        {
            "item": "behavioral_warning_count",
            "value": int(
                (
                    behavioral_checks_df["status"]
                    == "diagnostic_warning"
                ).sum()
            ),
        },
        {
            "item": "maximum_absolute_branch_disagreement_scenario",
            "value": str(
                summary_df.loc[
                    summary_df["branch_disagreement_kwh"].idxmax(),
                    "scenario_name",
                ]
            ),
        },
        {
            "item": "maximum_branch_disagreement_kwh",
            "value": float(
                summary_df["branch_disagreement_kwh"].max()
            ),
        },
        {
            "item": "maximum_relative_branch_disagreement_scenario",
            "value": str(
                summary_df.loc[
                    summary_df["branch_disagreement_pct"].idxmax(),
                    "scenario_name",
                ]
            ),
        },
        {
            "item": "maximum_branch_disagreement_pct",
            "value": float(
                summary_df["branch_disagreement_pct"].max()
            ),
        },
    ]

    scenario_class_counts = (
        scenarios_df["scenario_class"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    for scenario_class, count in (
        scenario_class_counts.items()
    ):
        metadata_records.append(
            {
                "item": (
                    f"scenario_count_{scenario_class}"
                ),
                "value": int(count),
            }
        )

    metadata_df = pd.DataFrame(metadata_records)

    with pd.ExcelWriter(REPORT_PATH) as writer:
        metadata_df.to_excel(
            writer,
            sheet_name="Metadata",
            index=False,
        )
        scenario_classification_df.to_excel(
            writer,
            sheet_name="Scenario Classes",
            index=False,
        )
        scenarios_df.to_excel(
            writer,
            sheet_name="Scenario Inputs",
            index=False,
        )
        schema_checks_df.to_excel(
            writer,
            sheet_name="Schema Checks",
            index=False,
        )
        schema_summary_df.to_excel(
            writer,
            sheet_name="Schema Summary",
            index=False,
        )
        operational_checks_df.to_excel(
            writer,
            sheet_name="Operational Checks",
            index=False,
        )
        operational_summary_df.to_excel(
            writer,
            sheet_name="Operational Summary",
            index=False,
        )
        calendar_checks_df.to_excel(
            writer,
            sheet_name="Calendar Coverage",
            index=False,
        )
        calendar_summary_df.to_excel(
            writer,
            sheet_name="Calendar Summary",
            index=False,
        )
        predictions_df.to_excel(
            writer,
            sheet_name="Predictions Long",
            index=False,
        )
        summary_df.to_excel(
            writer,
            sheet_name="Prediction Summary",
            index=False,
        )
        paired_sensitivity_df.to_excel(
            writer,
            sheet_name="Paired Sensitivity",
            index=False,
        )
        hard_checks_df.to_excel(
            writer,
            sheet_name="Hard Checks",
            index=False,
        )
        behavioral_checks_df.to_excel(
            writer,
            sheet_name="Behavioral Checks",
            index=False,
        )
        operational_ranges_df.to_excel(
            writer,
            sheet_name="Operational Ranges",
            index=False,
        )

    save_official_prediction_plot(summary_df)
    save_branch_disagreement_kwh_plot(summary_df)
    save_branch_disagreement_pct_plot(summary_df)
    save_paired_sensitivity_plot(paired_sensitivity_df)

    if log_mlflow:
        log_behavioral_evaluation_to_mlflow(
            scenarios_df=scenarios_df,
            schema_summary_df=schema_summary_df,
            operational_summary_df=operational_summary_df,
            summary_df=summary_df,
            paired_sensitivity_df=paired_sensitivity_df,
            hard_checks_df=hard_checks_df,
            behavioral_checks_df=behavioral_checks_df,
        )

    print()
    print(
        "Behavioral scenario evaluation and synthetic stress test "
        "completed."
    )
    print()
    print(
        "Historical reference replays and synthetic perturbations are "
        "behavioral diagnostics only. They are not evidence of "
        "out-of-sample predictive accuracy."
    )
    print()

    print("Hard checks:")
    print(hard_checks_df.to_string(index=False))
    print()

    print("Behavioral checks:")
    print(behavioral_checks_df.to_string(index=False))
    print()

    print("Paired local operating-scale sensitivity:")
    print(paired_sensitivity_df.to_string(index=False))
    print()

    print("Scenario summary:")
    print(
        summary_df[
            [
                "scenario_name",
                "scenario_class",
                "scenario_origin",
                "parent_scenario_name",
                "perturbation_type",
                "perturbation_scale",
                "construction_method",
                "schema_status",
                "operational_range_status",
                "distribution_tail_feature_count",
                "constant_operational_feature_count",
                "calendar_coverage_status",
                OFFICIAL_ENSEMBLE_NAME,
                "branch_disagreement_kwh",
                "branch_disagreement_pct",
                "branch_disagreement_status",
                "warning",
            ]
        ].to_string(index=False)
    )
    print()

    print("Saved outputs:")
    print(f"- {REPORT_PATH}")
    print(f"- {FIGURES_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Version 2.1 behavioral scenario evaluation "
            "and synthetic stress test."
        )
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help=(
            "Log the finalized Version 2.1 diagnostic metrics and "
            "artifacts to MLflow."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(log_mlflow=args.log_mlflow)
