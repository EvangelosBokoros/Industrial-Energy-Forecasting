import itertools

import pandas as pd


def validate_post_split_dates(
    intervention_date: str,
    post_train_end: str,
    post_validation_start: str,
    post_validation_end: str,
    post_test_start: str,
) -> None:
    """
    Validate the chronological order of the post-installation split dates.
    """
    intervention_dt = pd.to_datetime(intervention_date)
    post_train_end_dt = pd.to_datetime(post_train_end)
    post_validation_start_dt = pd.to_datetime(post_validation_start)
    post_validation_end_dt = pd.to_datetime(post_validation_end)
    post_test_start_dt = pd.to_datetime(post_test_start)

    if not (
        intervention_dt
        <= post_train_end_dt
        < post_validation_start_dt
        <= post_validation_end_dt
        < post_test_start_dt
    ):
        raise ValueError(
            "Invalid post-installation split order. Expected: "
            "intervention_date <= post_train_end < post_validation_start "
            "<= post_validation_end < post_test_start."
        )


def create_post_split_masks(
    df: pd.DataFrame,
    date_col: str,
    intervention_date: str,
    post_train_end: str,
    post_validation_start: str,
    post_validation_end: str,
    post_test_start: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Create chronological train, validation, and test masks for the
    post-installation forecasting model.

    The post-installation model only uses data from the new operating regime.
    """
    validate_post_split_dates(
        intervention_date=intervention_date,
        post_train_end=post_train_end,
        post_validation_start=post_validation_start,
        post_validation_end=post_validation_end,
        post_test_start=post_test_start,
    )

    dates = pd.to_datetime(df[date_col], errors="raise")

    intervention_dt = pd.to_datetime(intervention_date)
    post_train_end_dt = pd.to_datetime(post_train_end)
    post_validation_start_dt = pd.to_datetime(post_validation_start)
    post_validation_end_dt = pd.to_datetime(post_validation_end)
    post_test_start_dt = pd.to_datetime(post_test_start)

    post_train_mask = (
        (dates >= intervention_dt)
        & (dates <= post_train_end_dt)
    )

    post_validation_mask = (
        (dates >= post_validation_start_dt)
        & (dates <= post_validation_end_dt)
    )

    post_test_mask = dates >= post_test_start_dt

    return post_train_mask, post_validation_mask, post_test_mask


def assert_no_split_overlap(split_masks: dict[str, pd.Series]) -> None:
    """
    Raise an error if any two split masks overlap.

    This protects against leakage between train, validation, and test data.
    """
    for first_name, second_name in itertools.combinations(split_masks.keys(), 2):
        overlap_count = int(
            (split_masks[first_name] & split_masks[second_name]).sum()
        )

        if overlap_count > 0:
            raise ValueError(
                f"Split overlap detected between {first_name} and "
                f"{second_name}: {overlap_count} rows"
            )


def assert_non_empty_splits(split_masks: dict[str, pd.Series]) -> None:
    """
    Raise an error if any split contains zero rows.
    """
    for split_name, split_mask in split_masks.items():
        row_count = int(split_mask.sum())

        if row_count == 0:
            raise ValueError(f"Split '{split_name}' contains zero rows.")


def summarize_split_sizes(split_masks: dict[str, pd.Series]) -> dict[str, int]:
    """
    Return the number of rows in each split.
    """
    return {
        split_name: int(split_mask.sum())
        for split_name, split_mask in split_masks.items()
    }