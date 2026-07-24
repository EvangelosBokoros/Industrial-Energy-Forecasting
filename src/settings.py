DATE_COL = "Ημ/νία"

TARGET_COL = "active_energy_kWh"

INTERVENTION_DATE = "2025-09-12"

POST_TRAIN_END = "2025-10-17"
POST_VALIDATION_START = "2025-10-18"
POST_VALIDATION_END = "2025-10-24"
POST_TEST_START = "2025-10-25"

FEATURE_COLUMNS = [
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

# Full-history forecasting split configuration
#
# This split uses all available historical rows up to the end of the
# post-installation training period, while keeping the same validation and
# test windows as the post-only benchmark.
FULL_HISTORY_TRAIN_END = "2025-10-17"
FULL_HISTORY_VALIDATION_START = "2025-10-18"
FULL_HISTORY_VALIDATION_END = "2025-10-24"
FULL_HISTORY_TEST_START = "2025-10-25"