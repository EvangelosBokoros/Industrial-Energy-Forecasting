import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Check that the dataset contains all required columns.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Parse the date column robustly.

    The Damavand dataset uses European-style dates, so day-first parsing is
    attempted first. A non-day-first parse is also tested, and the version
    with more valid parsed dates is kept.
    """
    result = df.copy()

    parsed_day_first = pd.to_datetime(
        result[date_col],
        errors="coerce",
        dayfirst=True,
    )

    parsed_not_day_first = pd.to_datetime(
        result[date_col],
        errors="coerce",
        dayfirst=False,
    )

    if parsed_day_first.notna().sum() >= parsed_not_day_first.notna().sum():
        result[date_col] = parsed_day_first
    else:
        result[date_col] = parsed_not_day_first

    return result


def clean_energy_dataset(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    remove_zero_target: bool = True,
) -> pd.DataFrame:
    """
    Clean the final Damavand forecasting dataset.

    Steps:
    - parse date column
    - remove rows with missing date or target
    - optionally remove rows where target is zero
    - sort by date
    - reset index
    """
    result = df.copy()

    result = parse_date_column(result, date_col=date_col)

    result = result.dropna(subset=[date_col, target_col])

    if remove_zero_target:
        result = result[result[target_col] != 0]

    result = result.sort_values(date_col).reset_index(drop=True)

    return result