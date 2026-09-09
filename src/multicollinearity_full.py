from pathlib import Path
import argparse

import pandas as pd

from src.data import load_tabular_data
from src.multicollinearity import (
    calculate_vif_table,
    make_excel_safe,
    reduce_features_by_vif,
)
from src.preprocessing import clean_energy_dataset, validate_required_columns
from src.settings import DATE_COL, FEATURE_COLUMNS, TARGET_COL


DEFAULT_DATA_PATH = Path("data/processed/industrial_energy_daily.csv")
DEFAULT_OUTPUT_PATH = Path("reports/multicollinearity_full_dataset_report.xlsx")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate VIF-based multicollinearity diagnostics on the full dataset."
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
    Run VIF-based multicollinearity analysis on the full cleaned dataset.
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

    feature_df = model_df[FEATURE_COLUMNS].copy()

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
                "analysis_scope": "full_clean_dataset",
                "row_count": len(model_df),
                "start_date": model_df[DATE_COL].min(),
                "end_date": model_df[DATE_COL].max(),
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

    print("\nInitial VIF values for full dataset:")
    print(make_excel_safe(initial_vif_table).to_string(index=False))

    print("\nRecommended removal steps for full dataset:")
    if removal_steps_df.empty:
        print("No features need to be removed.")
    else:
        print(make_excel_safe(removal_steps_df).to_string(index=False))

    print("\nFinal remaining features for full dataset:")
    for feature_name in remaining_features:
        print(f"- {feature_name}")

    print(f"\nSaved full-dataset multicollinearity report: {output_path}")


if __name__ == "__main__":
    main()
