from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data import load_tabular_data
from src.preprocessing import clean_energy_dataset, validate_required_columns
from src.settings import (
    DATE_COL,
    FEATURE_COLUMNS,
    INTERVENTION_DATE,
    POST_TEST_START,
    POST_TRAIN_END,
    POST_VALIDATION_END,
    POST_VALIDATION_START,
    TARGET_COL,
)
from src.splits import create_post_split_masks


DEFAULT_DATA_PATH = Path("data/processed/industrial_energy_daily.csv")
DEFAULT_OUTPUT_PATH = Path("reports/multicollinearity_report.xlsx")


def calculate_single_vif(feature_df: pd.DataFrame, feature_name: str) -> float:
    """
    Calculate VIF for one feature.

    VIF = 1 / (1 - RΒ²)

    RΒ² is calculated by predicting one feature from all other features.
    """
    y = feature_df[feature_name]
    X = feature_df.drop(columns=[feature_name])

    if y.nunique(dropna=False) <= 1:
        return float("inf")

    if X.shape[1] == 0:
        return 1.0

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)

    if r2 >= 1.0:
        return float("inf")

    if np.isclose(1.0 - r2, 0.0):
        return float("inf")

    return float(1.0 / (1.0 - r2))


def calculate_vif_table(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> pd.DataFrame:
    """
    Calculate VIF values for a list of features.
    """
    records = []

    for feature_name in feature_columns:
        vif = calculate_single_vif(feature_df[feature_columns], feature_name)

        if np.isinf(vif):
            status = "remove"
        elif vif >= threshold:
            status = "remove"
        else:
            status = "keep"

        records.append(
            {
                "feature": feature_name,
                "vif": vif,
                "status": status,
            }
        )

    vif_table = pd.DataFrame(records)
    vif_table = vif_table.sort_values("vif", ascending=False).reset_index(drop=True)

    return vif_table


def reduce_features_by_vif(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Iteratively remove the feature with the highest VIF until all remaining
    features have VIF below the selected threshold.
    """
    remaining_features = feature_columns.copy()
    removal_steps = []

    step = 1

    while len(remaining_features) > 1:
        vif_table = calculate_vif_table(
            feature_df=feature_df,
            feature_columns=remaining_features,
            threshold=threshold,
        )

        max_vif = vif_table.loc[0, "vif"]
        feature_to_remove = vif_table.loc[0, "feature"]

        if not np.isinf(max_vif) and max_vif < threshold:
            break

        removal_steps.append(
            {
                "step": step,
                "removed_feature": feature_to_remove,
                "removed_vif": max_vif,
                "remaining_feature_count_after_removal": len(remaining_features) - 1,
            }
        )

        remaining_features.remove(feature_to_remove)
        step += 1

    final_vif_table = calculate_vif_table(
        feature_df=feature_df,
        feature_columns=remaining_features,
        threshold=threshold,
    )

    removal_steps_df = pd.DataFrame(removal_steps)

    return final_vif_table, removal_steps_df, remaining_features


def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite values so Excel can display them cleanly.
    """
    return df.replace([np.inf, -np.inf], "inf")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate VIF-based multicollinearity diagnostics."
    )

    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the processed industrial energy dataset.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Maximum acceptable VIF value.",
    )

    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where the Excel report will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run VIF-based multicollinearity analysis.
    """
    args = parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    threshold = args.threshold

    raw_df = load_tabular_data(data_path)

    required_columns = [DATE_COL, TARGET_COL] + FEATURE_COLUMNS
    validate_required_columns(raw_df, required_columns)

    clean_df = clean_energy_dataset(
        raw_df,
        date_col=DATE_COL,
        target_col=TARGET_COL,
    )

    model_df = clean_df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COL],
    ).reset_index(drop=True)

    post_train_mask, post_validation_mask, _ = create_post_split_masks(
        model_df,
        date_col=DATE_COL,
        intervention_date=INTERVENTION_DATE,
        post_train_end=POST_TRAIN_END,
        post_validation_start=POST_VALIDATION_START,
        post_validation_end=POST_VALIDATION_END,
        post_test_start=POST_TEST_START,
    )

    development_mask = post_train_mask | post_validation_mask
    development_df = model_df.loc[development_mask].copy()

    feature_df = development_df[FEATURE_COLUMNS].copy()

    initial_vif_table = calculate_vif_table(
        feature_df=feature_df,
        feature_columns=FEATURE_COLUMNS,
        threshold=threshold,
    )

    final_vif_table, removal_steps_df, remaining_features = reduce_features_by_vif(
        feature_df=feature_df,
        feature_columns=FEATURE_COLUMNS,
        threshold=threshold,
    )

    final_features_df = pd.DataFrame(
        {
            "remaining_feature": remaining_features,
        }
    )

    summary_df = pd.DataFrame(
        [
            {
                "analysis_scope": "post_train_plus_post_validation",
                "row_count": len(development_df),
                "start_date": development_df[DATE_COL].min(),
                "end_date": development_df[DATE_COL].max(),
                "vif_threshold": threshold,
                "initial_feature_count": len(FEATURE_COLUMNS),
                "final_feature_count": len(remaining_features),
            }
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        make_excel_safe(initial_vif_table).to_excel(
            writer,
            sheet_name="Initial VIF",
            index=False,
        )

        make_excel_safe(removal_steps_df).to_excel(
            writer,
            sheet_name="Removal Steps",
            index=False,
        )

        make_excel_safe(final_vif_table).to_excel(
            writer,
            sheet_name="Final VIF",
            index=False,
        )

        final_features_df.to_excel(
            writer,
            sheet_name="Remaining Features",
            index=False,
        )

    print("\nInitial VIF values:")
    print(make_excel_safe(initial_vif_table).to_string(index=False))

    print("\nRecommended removal steps:")
    if removal_steps_df.empty:
        print("No features need to be removed.")
    else:
        print(make_excel_safe(removal_steps_df).to_string(index=False))

    print("\nFinal remaining features:")
    for feature_name in remaining_features:
        print(f"- {feature_name}")

    print(f"\nSaved multicollinearity report: {output_path}")


if __name__ == "__main__":
    main()
