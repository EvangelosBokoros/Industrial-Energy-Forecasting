from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.stress_test_ensemble import (
    FULL_HISTORY_MODEL_NAME,
    OFFICIAL_ENSEMBLE_NAME,
    POST_ONLY_MODEL_NAME,
    build_local_activity_updates,
    build_paired_sensitivity_table,
    calculate_branch_disagreement,
    calculate_schema_checks,
    choose_local_activity_scale,
    classify_disagreement,
    get_perturbation_type,
    get_value_range_flag,
    is_integer_like,
    recalculate_confirmed_derived_features,
)


class StressTestUtilityTests(unittest.TestCase):
    def test_value_inside_typical_range(self) -> None:
        result = get_value_range_flag(
            value=50.0,
            observed_min=0.0,
            observed_max=100.0,
            lower_quantile=10.0,
            upper_quantile=90.0,
        )
        self.assertEqual(
            result,
            "inside_typical_development_range",
        )

    def test_value_in_distribution_tail(self) -> None:
        result = get_value_range_flag(
            value=5.0,
            observed_min=0.0,
            observed_max=100.0,
            lower_quantile=10.0,
            upper_quantile=90.0,
        )
        self.assertEqual(
            result,
            "development_distribution_tail",
        )

    def test_value_outside_range(self) -> None:
        result = get_value_range_flag(
            value=110.0,
            observed_min=0.0,
            observed_max=100.0,
            lower_quantile=10.0,
            upper_quantile=90.0,
        )
        self.assertEqual(result, "outside_observed_range")

    def test_constant_feature_matching_value(self) -> None:
        result = get_value_range_flag(
            value=5.0,
            observed_min=5.0,
            observed_max=5.0,
            lower_quantile=5.0,
            upper_quantile=5.0,
        )
        self.assertEqual(result, "constant_in_development")

    def test_constant_feature_changed_value(self) -> None:
        result = get_value_range_flag(
            value=6.0,
            observed_min=5.0,
            observed_max=5.0,
            lower_quantile=5.0,
            upper_quantile=5.0,
        )
        self.assertEqual(result, "outside_observed_range")

    def test_range_check_uses_floating_point_tolerance(self) -> None:
        result = get_value_range_flag(
            value=0.9988601736336944,
            observed_min=0.998860174,
            observed_max=1.10,
            lower_quantile=1.00,
            upper_quantile=1.08,
        )
        self.assertNotEqual(result, "outside_observed_range")

    def test_confirmed_yield_ratio_recalculation(self) -> None:
        row = {
            "total_kg": 90.0,
            "total_nominal_kg": 100.0,
            "yield_ratio_actual_over_nominal": 999.0,
        }

        result = recalculate_confirmed_derived_features(row)

        self.assertAlmostEqual(
            result["yield_ratio_actual_over_nominal"],
            0.9,
        )

    def test_yield_ratio_rejects_nonpositive_nominal(self) -> None:
        row = {
            "total_kg": 90.0,
            "total_nominal_kg": 0.0,
            "yield_ratio_actual_over_nominal": 0.0,
        }

        with self.assertRaises(ValueError):
            recalculate_confirmed_derived_features(row)

    def test_branch_disagreement(self) -> None:
        post_only = np.array([100.0, 200.0])
        full_history = np.array([120.0, 160.0])

        disagreement_kwh, disagreement_pct = (
            calculate_branch_disagreement(
                post_only,
                full_history,
            )
        )

        np.testing.assert_allclose(
            disagreement_kwh,
            np.array([20.0, 40.0]),
        )
        np.testing.assert_allclose(
            disagreement_pct,
            np.array(
                [
                    20.0 / 110.0 * 100.0,
                    40.0 / 180.0 * 100.0,
                ]
            ),
        )

    def test_disagreement_classification(self) -> None:
        self.assertEqual(classify_disagreement(5.0), "low")
        self.assertEqual(classify_disagreement(15.0), "moderate")
        self.assertEqual(classify_disagreement(25.0), "high")
        self.assertEqual(
            classify_disagreement(np.nan),
            "undefined",
        )

    def test_local_updates_scale_only_continuous_features(self) -> None:
        anchor = pd.Series(
            {
                "total_kg": 100.0,
                "total_nominal_kg": 120.0,
                "total_brix_units": 1_500.0,
                "total_hours": 10.0,
                "orders": 7.0,
            }
        )

        updates = build_local_activity_updates(
            anchor,
            1.05,
        )

        self.assertAlmostEqual(updates["total_kg"], 105.0)
        self.assertAlmostEqual(
            updates["total_nominal_kg"],
            126.0,
        )
        self.assertAlmostEqual(
            updates["total_brix_units"],
            1_575.0,
        )
        self.assertAlmostEqual(updates["total_hours"], 10.5)
        self.assertNotIn("orders", updates)

    def test_perturbation_type_uses_operating_scale_name(self) -> None:
        self.assertEqual(
            get_perturbation_type(1.05),
            "operating_scale_up_5pct",
        )
        self.assertEqual(
            get_perturbation_type(0.95),
            "operating_scale_down_5pct",
        )

    def test_choose_local_scale_increases_when_supported(self) -> None:
        anchor = pd.Series(
            {
                "total_kg": 50.0,
                "total_nominal_kg": 60.0,
                "total_brix_units": 700.0,
                "total_hours": 5.0,
            }
        )
        reference = pd.DataFrame(
            {
                "total_kg": [40.0, 100.0],
                "total_nominal_kg": [50.0, 120.0],
                "total_brix_units": [600.0, 1_500.0],
                "total_hours": [4.0, 10.0],
            }
        )

        self.assertAlmostEqual(
            choose_local_activity_scale(anchor, reference),
            1.05,
        )

    def test_choose_local_scale_decreases_near_maximum(self) -> None:
        anchor = pd.Series(
            {
                "total_kg": 100.0,
                "total_nominal_kg": 120.0,
                "total_brix_units": 1_500.0,
                "total_hours": 10.0,
            }
        )
        reference = pd.DataFrame(
            {
                "total_kg": [40.0, 100.0],
                "total_nominal_kg": [50.0, 120.0],
                "total_brix_units": [600.0, 1_500.0],
                "total_hours": [4.0, 10.0],
            }
        )

        self.assertAlmostEqual(
            choose_local_activity_scale(anchor, reference),
            0.95,
        )

    def test_integer_like_orders(self) -> None:
        self.assertTrue(is_integer_like(7))
        self.assertTrue(is_integer_like(7.0))
        self.assertFalse(is_integer_like(7.2))
        self.assertFalse(is_integer_like(True))

    def test_schema_rejects_fractional_orders(self) -> None:
        scenario_date = pd.Timestamp("2025-10-20")
        iso_week = int(scenario_date.isocalendar().week)

        base_row = {
            "scenario_name": "example",
            "scenario_class": "plausible",
            "scenario_origin": "historical_reference",
            "parent_scenario_name": "",
            "perturbation_type": "none",
            "perturbation_scale": 1.0,
            "description": "test",
            "construction_method": "test",
            "anchor_date": "2025-10-20",
            "scenario_date": "2025-10-20",
            "total_kg": 100.0,
            "total_nominal_kg": 110.0,
            "total_brix_units": 1_400.0,
            "total_hours": 8.0,
            "total_pallets": 1.0,
            "orders": 3.5,
            "avg_brix": 14.0,
            "yield_ratio_actual_over_nominal": 100.0 / 110.0,
            "weekday": float(scenario_date.weekday()),
            "is_weekend": 0.0,
            "month": float(scenario_date.month),
            "year": float(scenario_date.year),
            "week_of_year": float(iso_week),
        }

        checks = calculate_schema_checks(
            pd.DataFrame([base_row]),
            required_feature_columns=[
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
            ],
        )

        order_check = checks.loc[
            checks["check_name"]
            == "orders_is_nonnegative_integer",
            "status",
        ].iloc[0]

        self.assertEqual(order_check, "failed")

    def test_paired_sensitivity_calculation(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "scenario_name": "parent",
                    "scenario_origin": "historical_reference",
                    "parent_scenario_name": "",
                    "scenario_class": "plausible",
                    "perturbation_type": "none",
                    "perturbation_scale": 1.0,
                    POST_ONLY_MODEL_NAME: 100.0,
                    FULL_HISTORY_MODEL_NAME: 120.0,
                    OFFICIAL_ENSEMBLE_NAME: 106.0,
                    "branch_disagreement_pct": 18.181818,
                    "operational_range_status": (
                        "inside_typical_development_range"
                    ),
                    "warning": "",
                },
                {
                    "scenario_name": "parent_operating_scale_up_5pct",
                    "scenario_origin": "local_synthetic_perturbation",
                    "parent_scenario_name": "parent",
                    "scenario_class": "plausible",
                    "perturbation_type": "operating_scale_up_5pct",
                    "perturbation_scale": 1.05,
                    POST_ONLY_MODEL_NAME: 110.0,
                    FULL_HISTORY_MODEL_NAME: 120.0,
                    OFFICIAL_ENSEMBLE_NAME: 113.0,
                    "branch_disagreement_pct": 8.695652,
                    "operational_range_status": (
                        "inside_typical_development_range"
                    ),
                    "warning": "",
                },
            ]
        )

        paired = build_paired_sensitivity_table(summary)

        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(
            float(paired.loc[0, "official_delta_kwh"]),
            7.0,
        )
        self.assertEqual(
            paired.loc[0, "directional_status"],
            "passed",
        )


if __name__ == "__main__":
    unittest.main()
