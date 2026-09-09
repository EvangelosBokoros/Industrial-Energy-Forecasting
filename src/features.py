import pandas as pd

from src.settings import DATE_COL, FEATURE_COLUMNS, TARGET_COL


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features from the project date column.

    The current industrial energy dataset already contains these columns, but this
    function keeps the forecasting pipeline reproducible if calendar features
    need to be recreated later.
    """
    result = df.copy()

    result[DATE_COL] = pd.to_datetime(result[DATE_COL])

    result["year"] = result[DATE_COL].dt.year
    result["month"] = result[DATE_COL].dt.month
    result["weekday"] = result[DATE_COL].dt.weekday
    result["is_weekend"] = result["weekday"].isin([5, 6]).astype(int)
    result["week_of_year"] = result[DATE_COL].dt.isocalendar().week.astype(int)

    return result


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model feature matrix.

    The selected feature set is defined in settings.py.
    """
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    return df[FEATURE_COLUMNS].copy()


def build_target_vector(df: pd.DataFrame) -> pd.Series:
    """
    Build the target vector for electricity forecasting.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    return df[TARGET_COL].copy()


def build_model_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X and y for model training or evaluation.
    """
    X = build_feature_matrix(df)
    y = build_target_vector(df)

    return X, y
