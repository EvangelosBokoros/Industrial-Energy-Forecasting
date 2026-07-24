from src.settings import FEATURE_COLUMNS


FEATURE_SETS = {
    "full": FEATURE_COLUMNS,
    "vif_auto_post_only": [
        "orders",
        "avg_brix",
        "yield_ratio_actual_over_nominal",
        "weekday",
        "is_weekend",
        "week_of_year",
    ],
    "reduced_without_total_kg": [
        "total_hours",
        "orders",
        "avg_brix",
        "yield_ratio_actual_over_nominal",
        "weekday",
        "is_weekend",
        "week_of_year",
    ],
    "domain_reduced_with_total_kg": [
        "total_kg",
        "total_hours",
        "orders",
        "avg_brix",
        "yield_ratio_actual_over_nominal",
        "weekday",
        "is_weekend",
        "week_of_year",
    ],
    "vif_auto_full_history": [
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
    ],
}


def get_available_feature_sets() -> list[str]:
    """
    Return the available named feature sets.
    """
    return sorted(FEATURE_SETS.keys())


def get_feature_columns(feature_set_name: str) -> list[str]:
    """
    Return the feature columns for a named feature set.

    Raises:
        ValueError: If the requested feature set does not exist.
    """
    if feature_set_name not in FEATURE_SETS:
        available_feature_sets = ", ".join(get_available_feature_sets())
        raise ValueError(
            f"Unknown feature set: {feature_set_name}. "
            f"Available feature sets: {available_feature_sets}"
        )

    return FEATURE_SETS[feature_set_name].copy()