from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_tabular_data
from src.evaluate import (
    calculate_bootstrap_metric_intervals,
    calculate_regression_metrics,
)
from src.feature_sets import get_feature_columns
from src.model_ensemble import (
    build_full_history_champion_model,
    build_post_only_champion_model,
    calculate_weighted_ensemble_predictions,
)
from src.preprocessing import clean_energy_dataset
from src.settings import (
    DATE_COL,
    FULL_HISTORY_TEST_START,
    FULL_HISTORY_TRAIN_END,
    FULL_HISTORY_VALIDATION_END,
    FULL_HISTORY_VALIDATION_START,
    INTERVENTION_DATE,
    POST_TEST_START,
    POST_TRAIN_END,
    POST_VALIDATION_END,
    POST_VALIDATION_START,
    TARGET_COL,
)


DATA_PATH = Path("data/processed/industrial_energy_daily.csv")

REPORT_PATH = Path("reports/final_model_comparison.xlsx")
FIGURES_DIR = Path("reports/figures/final_model_comparison")


def ensure_output_directories() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_modeling_data() -> pd.DataFrame:
    df = load_tabular_data(DATA_PATH)
    df = clean_energy_dataset(df, DATE_COL, TARGET_COL)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")

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


def validate_non_empty_split(split_name: str, split_df: pd.DataFrame) -> None:
    if split_df.empty:
        raise ValueError(f"{split_name} is empty. Check the split date settings.")


def calculate_metrics_record(
    model_name: str,
    y_true,
    y_pred,
) -> dict[str, float | str]:
    metrics = calculate_regression_metrics(y_true, y_pred)
    return {"model_name": model_name, **metrics}


def build_prediction_records(
    dates: pd.Series,
    y_true: pd.Series,
    predictions_by_model: dict[str, pd.Series],
    period_name: str,
) -> pd.DataFrame:
    records = []

    for model_name, predictions in predictions_by_model.items():
        for date, actual, predicted in zip(dates, y_true, predictions):
            records.append(
                {
                    "period": period_name,
                    "model_name": model_name,
                    "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "residual": float(actual - predicted),
                }
            )

    return pd.DataFrame(records)


def calculate_test_intervals_table(
    y_test,
    predictions_by_model: dict[str, pd.Series],
) -> pd.DataFrame:
    records = []

    for model_name, predictions in predictions_by_model.items():
        intervals = calculate_bootstrap_metric_intervals(
            y_test,
            predictions,
            n_bootstrap=1000,
            confidence_level=0.95,
            random_state=33,
        )

        record = {"model_name": model_name}

        for interval_name, interval_values in intervals.items():
            lower_bound, upper_bound = interval_values
            clean_name = interval_name.replace("_ci", "")
            record[f"{clean_name}_ci_lower"] = lower_bound
            record[f"{clean_name}_ci_upper"] = upper_bound

        records.append(record)

    return pd.DataFrame(records)


def save_metric_bar_plot(
    metrics_df: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    title: str,
) -> None:
    plot_df = metrics_df.copy()

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["model_name"], plot_df[metric_name])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric_name)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_test_prediction_plot(
    test_predictions_df: pd.DataFrame,
    output_path: Path,
) -> None:
    wide_df = test_predictions_df.pivot_table(
        index="date",
        columns="model_name",
        values="predicted",
        aggfunc="first",
    )

    actual_df = test_predictions_df[["date", "actual"]].drop_duplicates()

    plt.figure(figsize=(11, 6))
    plt.plot(actual_df["date"], actual_df["actual"], marker="o", label="Actual")

    for model_name in wide_df.columns:
        plt.plot(wide_df.index, wide_df[model_name], marker="o", label=model_name)

    plt.title("Final Model Comparison: Test Predictions")
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    ensure_output_directories()

    df = load_modeling_data()

    post_only_features = get_feature_columns("full")
    full_history_features = get_feature_columns("vif_auto_full_history")

    required_columns = list(
        set([DATE_COL, TARGET_COL] + post_only_features + full_history_features)
    )
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    post_train_df = filter_date_range(
        df,
        start_date=INTERVENTION_DATE,
        end_date=POST_TRAIN_END,
    )
    post_validation_df = filter_date_range(
        df,
        start_date=POST_VALIDATION_START,
        end_date=POST_VALIDATION_END,
    )
    post_test_df = filter_date_range(
        df,
        start_date=POST_TEST_START,
        end_date=None,
    )

    full_history_train_df = filter_date_range(
        df,
        start_date=None,
        end_date=FULL_HISTORY_TRAIN_END,
    )
    full_history_validation_df = filter_date_range(
        df,
        start_date=FULL_HISTORY_VALIDATION_START,
        end_date=FULL_HISTORY_VALIDATION_END,
    )
    full_history_test_df = filter_date_range(
        df,
        start_date=FULL_HISTORY_TEST_START,
        end_date=None,
    )

    split_dataframes = {
        "post_train": post_train_df,
        "post_validation": post_validation_df,
        "post_test": post_test_df,
        "full_history_train": full_history_train_df,
        "full_history_validation": full_history_validation_df,
        "full_history_test": full_history_test_df,
    }

    for split_name, split_df in split_dataframes.items():
        validate_non_empty_split(split_name, split_df)

    if not post_validation_df[DATE_COL].reset_index(drop=True).equals(
        full_history_validation_df[DATE_COL].reset_index(drop=True)
    ):
        raise ValueError("Post-only and full-history validation dates do not align.")

    if not post_test_df[DATE_COL].reset_index(drop=True).equals(
        full_history_test_df[DATE_COL].reset_index(drop=True)
    ):
        raise ValueError("Post-only and full-history test dates do not align.")

    X_post_train = post_train_df[post_only_features]
    y_post_train = post_train_df[TARGET_COL]

    X_full_history_train = full_history_train_df[full_history_features]
    y_full_history_train = full_history_train_df[TARGET_COL]

    X_post_validation = post_validation_df[post_only_features]
    X_full_history_validation = full_history_validation_df[full_history_features]
    y_validation = post_validation_df[TARGET_COL]

    X_post_test = post_test_df[post_only_features]
    X_full_history_test = full_history_test_df[full_history_features]
    y_test = post_test_df[TARGET_COL]

    validation_post_only_model = build_post_only_champion_model()
    validation_full_history_model = build_full_history_champion_model()

    validation_post_only_model.fit(X_post_train, y_post_train)
    validation_full_history_model.fit(X_full_history_train, y_full_history_train)

    validation_post_only_predictions = validation_post_only_model.predict(
        X_post_validation
    )
    validation_full_history_predictions = validation_full_history_model.predict(
        X_full_history_validation
    )

    validation_70_30_predictions = calculate_weighted_ensemble_predictions(
        post_only_predictions=validation_post_only_predictions,
        full_history_predictions=validation_full_history_predictions,
        post_only_weight=0.7,
    )
    validation_60_40_predictions = calculate_weighted_ensemble_predictions(
        post_only_predictions=validation_post_only_predictions,
        full_history_predictions=validation_full_history_predictions,
        post_only_weight=0.6,
    )

    validation_predictions_by_model = {
        "0.5 post_only_extra_trees": validation_post_only_predictions,
        "1.3 full_history_adaboost": validation_full_history_predictions,
        "2.0 ensemble_70_30_official": validation_70_30_predictions,
        "2.0 ensemble_60_40_sensitivity": validation_60_40_predictions,
    }

    validation_metrics_df = pd.DataFrame(
        [
            calculate_metrics_record(model_name, y_validation, predictions)
            for model_name, predictions in validation_predictions_by_model.items()
        ]
    ).sort_values("mae")

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

    final_post_only_model = build_post_only_champion_model()
    final_full_history_model = build_full_history_champion_model()

    final_post_only_model.fit(
        post_development_df[post_only_features],
        post_development_df[TARGET_COL],
    )
    final_full_history_model.fit(
        full_history_development_df[full_history_features],
        full_history_development_df[TARGET_COL],
    )

    test_post_only_predictions = final_post_only_model.predict(X_post_test)
    test_full_history_predictions = final_full_history_model.predict(
        X_full_history_test
    )

    test_70_30_predictions = calculate_weighted_ensemble_predictions(
        post_only_predictions=test_post_only_predictions,
        full_history_predictions=test_full_history_predictions,
        post_only_weight=0.7,
    )
    test_60_40_predictions = calculate_weighted_ensemble_predictions(
        post_only_predictions=test_post_only_predictions,
        full_history_predictions=test_full_history_predictions,
        post_only_weight=0.6,
    )

    test_predictions_by_model = {
        "0.5 post_only_extra_trees": test_post_only_predictions,
        "1.3 full_history_adaboost": test_full_history_predictions,
        "2.0 ensemble_70_30_official": test_70_30_predictions,
        "2.0 ensemble_60_40_sensitivity": test_60_40_predictions,
    }

    test_metrics_df = pd.DataFrame(
        [
            calculate_metrics_record(model_name, y_test, predictions)
            for model_name, predictions in test_predictions_by_model.items()
        ]
    ).sort_values("mae")

    test_intervals_df = calculate_test_intervals_table(
        y_test=y_test,
        predictions_by_model=test_predictions_by_model,
    )

    validation_predictions_df = build_prediction_records(
        dates=post_validation_df[DATE_COL],
        y_true=y_validation,
        predictions_by_model=validation_predictions_by_model,
        period_name="validation",
    )

    test_predictions_df = build_prediction_records(
        dates=post_test_df[DATE_COL],
        y_true=y_test,
        predictions_by_model=test_predictions_by_model,
        period_name="test",
    )

    metadata_df = pd.DataFrame(
        [
            {"item": "comparison_type", "value": "offline_champion_challenger"},
            {
                "item": "champion",
                "value": "2.0 ensemble_70_30_official",
            },
            {
                "item": "challengers",
                "value": (
                    "0.5 post_only_extra_trees, "
                    "1.3 full_history_adaboost, "
                    "2.0 ensemble_60_40_sensitivity"
                ),
            },
            {
                "item": "selection_rule",
                "value": "official model selected previously by validation MAE within constrained ensemble search",
            },
            {
                "item": "test_usage",
                "value": "final holdout comparison only; not used to choose official model",
            },
            {"item": "post_train_rows", "value": len(post_train_df)},
            {"item": "full_history_train_rows", "value": len(full_history_train_df)},
            {"item": "validation_rows", "value": len(post_validation_df)},
            {"item": "test_rows", "value": len(post_test_df)},
        ]
    )

    with pd.ExcelWriter(REPORT_PATH) as writer:
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        validation_metrics_df.to_excel(
            writer,
            sheet_name="Validation Metrics",
            index=False,
        )
        test_metrics_df.to_excel(
            writer,
            sheet_name="Test Metrics",
            index=False,
        )
        test_intervals_df.to_excel(
            writer,
            sheet_name="Test Bootstrap CI",
            index=False,
        )
        validation_predictions_df.to_excel(
            writer,
            sheet_name="Validation Predictions",
            index=False,
        )
        test_predictions_df.to_excel(
            writer,
            sheet_name="Test Predictions",
            index=False,
        )

    save_metric_bar_plot(
        metrics_df=validation_metrics_df,
        metric_name="mae",
        output_path=FIGURES_DIR / "validation_mae_comparison.png",
        title="Offline Champion-Challenger: Validation MAE",
    )
    save_metric_bar_plot(
        metrics_df=test_metrics_df,
        metric_name="mae",
        output_path=FIGURES_DIR / "test_mae_comparison.png",
        title="Offline Champion-Challenger: Test MAE",
    )
    save_metric_bar_plot(
        metrics_df=validation_metrics_df,
        metric_name="r2",
        output_path=FIGURES_DIR / "validation_r2_comparison.png",
        title="Offline Champion-Challenger: Validation RΒ²",
    )
    save_metric_bar_plot(
        metrics_df=test_metrics_df,
        metric_name="r2",
        output_path=FIGURES_DIR / "test_r2_comparison.png",
        title="Offline Champion-Challenger: Test RΒ²",
    )
    save_test_prediction_plot(
        test_predictions_df=test_predictions_df,
        output_path=FIGURES_DIR / "test_predictions_comparison.png",
    )

    print()
    print("Offline champion-challenger comparison completed.")
    print()
    print("Validation metrics:")
    print(validation_metrics_df.to_string(index=False))
    print()
    print("Test metrics:")
    print(test_metrics_df.to_string(index=False))
    print()
    print("Saved outputs:")
    print(f"- {REPORT_PATH}")
    print(f"- {FIGURES_DIR}")


if __name__ == "__main__":
    main()
