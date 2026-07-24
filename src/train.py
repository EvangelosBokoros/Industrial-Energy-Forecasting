import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.data import load_tabular_data
from src.evaluate import (
    calculate_bootstrap_metric_intervals,
    calculate_regression_metrics,
)
from src.features import build_model_inputs
from src.model import get_model_builders
from src.plots import (
    calculate_residuals,
    plot_actual_vs_predicted,
    plot_actual_vs_predicted_scatter,
    plot_residual_distribution,
    plot_residuals_over_time,
    plot_residuals_vs_predicted,
)
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
from src.splits import (
    assert_no_split_overlap,
    assert_non_empty_splits,
    create_post_split_masks,
    summarize_split_sizes,
)


DATA_PATH = Path("data/processed/damavand.csv")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
METADATA_DIR = REPORTS_DIR / "metadata"
MODELS_DIR = Path("models")

POST_ONLY_REPORT_PATH = REPORTS_DIR / "post_only_training_report.xlsx"
FEATURE_LIST_PATH = METADATA_DIR / "post_only_features.txt"
SELECTED_MODEL_PATH = MODELS_DIR / "post_only_model.joblib"

MLFLOW_EXPERIMENT_NAME = "Damavand Energy Forecasting"
MODELING_VERSION = "0.1"
RUN_NAME = "0.1 Post-Only Gradient Boosting Revised Split"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the Damavand post-installation forecasting model."
    )

    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log this run to MLflow.",
    )

    return parser.parse_args()


def build_prediction_table(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred,
) -> pd.DataFrame:
    """
    Build a prediction table with actuals, predictions, and residuals.
    """
    residuals = calculate_residuals(y_true, y_pred)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
            "actual": y_true.astype(float).values,
            "predicted": pd.Series(y_pred).astype(float).values,
            "residual": residuals,
        }
    )


def flatten_bootstrap_intervals(
    bootstrap_intervals: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """
    Convert bootstrap confidence intervals into flat metric columns.
    """
    flattened = {}

    for metric_name, interval in bootstrap_intervals.items():
        lower_bound, upper_bound = interval
        base_name = metric_name.replace("_ci", "")

        flattened[f"{base_name}_ci_lower"] = lower_bound
        flattened[f"{base_name}_ci_upper"] = upper_bound

    return flattened


def build_bootstrap_intervals_table(
    bootstrap_intervals: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    Build a readable table of bootstrap confidence intervals.
    """
    records = []

    for metric_name, interval in bootstrap_intervals.items():
        lower_bound, upper_bound = interval

        records.append(
            {
                "metric": metric_name.replace("_ci", ""),
                "ci_lower": lower_bound,
                "ci_upper": upper_bound,
            }
        )

    return pd.DataFrame(records)


def save_feature_list(
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """
    Save the feature list used in the run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Feature columns used by the model:",
        "",
    ]

    for index, feature_name in enumerate(feature_columns, start=1):
        lines.append(f"{index}. {feature_name}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """
    Print metrics in a readable format.
    """
    print(f"\n===== {title} =====")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


def print_bootstrap_intervals(
    title: str,
    bootstrap_intervals: dict[str, tuple[float, float]],
) -> None:
    """
    Print bootstrap confidence intervals in a readable format.
    """
    print(f"\n===== {title} =====")

    for metric_name, interval in bootstrap_intervals.items():
        lower_bound, upper_bound = interval
        clean_metric_name = metric_name.replace("_ci", "")

        print(
            f"{clean_metric_name}: "
            f"[{lower_bound:.4f}, {upper_bound:.4f}]"
        )


def log_metric_if_valid(metric_name: str, metric_value: float) -> None:
    """
    Log a metric to MLflow only if it is a valid finite number.
    """
    if np.isfinite(metric_value):
        mlflow.log_metric(metric_name, float(metric_value))


def log_artifact_if_exists(
    artifact_path: Path,
    mlflow_artifact_path: str,
) -> None:
    """
    Log an artifact to MLflow only if the local path exists.
    """
    if artifact_path.exists():
        mlflow.log_artifact(
            str(artifact_path),
            artifact_path=mlflow_artifact_path,
        )


def log_run_to_mlflow(
    model,
    selected_model_name: str,
    model_builders: dict,
    validation_metrics_df: pd.DataFrame,
    test_metrics_record: dict[str, float],
    split_sizes: dict[str, int],
) -> None:
    """
    Log selected run information to MLflow.

    This function is only called when running:

        python -m src.train --log-mlflow
    """
    selected_validation_metrics = validation_metrics_df.loc[
        validation_metrics_df["model_name"] == selected_model_name
    ].iloc[0]

    print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_param("modeling_version", MODELING_VERSION)
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("selected_model_name", selected_model_name)
        mlflow.log_param("date_column", DATE_COL)
        mlflow.log_param("target_column", TARGET_COL)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("candidate_models", ",".join(model_builders.keys()))

        mlflow.log_param("intervention_date", INTERVENTION_DATE)
        mlflow.log_param("post_train_end", POST_TRAIN_END)
        mlflow.log_param("post_validation_start", POST_VALIDATION_START)
        mlflow.log_param("post_validation_end", POST_VALIDATION_END)
        mlflow.log_param("post_test_start", POST_TEST_START)

        for split_name, row_count in split_sizes.items():
            mlflow.log_param(f"{split_name}_rows", row_count)

        for parameter_name, parameter_value in model.get_params().items():
            mlflow.log_param(f"model__{parameter_name}", parameter_value)

        mlflow.set_tag("project", "damavand_energy_forecasting")
        mlflow.set_tag("objective", "post_installation_forecasting")
        mlflow.set_tag("features", ", ".join(FEATURE_COLUMNS))
        mlflow.set_tag("raw_data_logged", "False")
        mlflow.set_tag(
            "training_info",
            "Version 0.1 post-only forecasting baseline with revised validation split.",
        )

        metric_names = [
            "mae",
            "rmse",
            "r2",
            "mape",
            "total_deviation_pct",
        ]

        for metric_name in metric_names:
            log_metric_if_valid(
                f"validation_{metric_name}",
                float(selected_validation_metrics[metric_name]),
            )

            log_metric_if_valid(
                f"test_{metric_name}",
                float(test_metrics_record[metric_name]),
            )

        for metric_name, metric_value in test_metrics_record.items():
            if metric_name.endswith("_ci_lower") or metric_name.endswith("_ci_upper"):
                log_metric_if_valid(
                    f"test_{metric_name}",
                    float(metric_value),
                )

        log_artifact_if_exists(
            POST_ONLY_REPORT_PATH,
            mlflow_artifact_path="reports",
        )

        log_artifact_if_exists(
            SELECTED_MODEL_PATH,
            mlflow_artifact_path="models",
        )

        log_artifact_if_exists(
            FEATURE_LIST_PATH,
            mlflow_artifact_path="metadata",
        )

        if FIGURES_DIR.exists():
            for figure_path in sorted(FIGURES_DIR.glob("*.png")):
                mlflow.log_artifact(
                    str(figure_path),
                    artifact_path="figures",
                )

        print("\nMLflow run logged.")
        print(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"Run name: {RUN_NAME}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")


def main(log_mlflow: bool = False) -> None:
    """
    Run the first post-installation forecasting experiment.
    """
    raw_df = load_tabular_data(DATA_PATH)

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

    X, y = build_model_inputs(model_df)

    post_train_mask, post_validation_mask, post_test_mask = create_post_split_masks(
        model_df,
        date_col=DATE_COL,
        intervention_date=INTERVENTION_DATE,
        post_train_end=POST_TRAIN_END,
        post_validation_start=POST_VALIDATION_START,
        post_validation_end=POST_VALIDATION_END,
        post_test_start=POST_TEST_START,
    )

    split_masks = {
        "post_train": post_train_mask,
        "post_validation": post_validation_mask,
        "post_test": post_test_mask,
    }

    assert_no_split_overlap(split_masks)
    assert_non_empty_splits(split_masks)

    split_sizes = summarize_split_sizes(split_masks)

    print("\nSplit sizes:")
    for split_name, row_count in split_sizes.items():
        print(f"{split_name}: {row_count}")

    X_train = X.loc[post_train_mask]
    y_train = y.loc[post_train_mask]

    X_validation = X.loc[post_validation_mask]
    y_validation = y.loc[post_validation_mask]

    X_test = X.loc[post_test_mask]
    y_test = y.loc[post_test_mask]

    model_builders = get_model_builders()

    validation_records = []

    for model_name, model_builder in model_builders.items():
        model = model_builder()
        model.fit(X_train, y_train)

        validation_predictions = model.predict(X_validation)
        validation_metrics = calculate_regression_metrics(
            y_validation,
            validation_predictions,
        )

        validation_records.append(
            {
                "model_name": model_name,
                **validation_metrics,
            }
        )

    validation_metrics_df = pd.DataFrame(validation_records)
    validation_metrics_df = validation_metrics_df.sort_values("mae").reset_index(
        drop=True
    )

    selected_model_name = validation_metrics_df.loc[0, "model_name"]

    print("\nValidation model comparison:")
    print(validation_metrics_df.to_string(index=False))

    print(f"\nSelected model by validation MAE: {selected_model_name}")

    final_train_mask = post_train_mask | post_validation_mask

    X_final_train = X.loc[final_train_mask]
    y_final_train = y.loc[final_train_mask]

    selected_model = model_builders[selected_model_name]()
    selected_model.fit(X_final_train, y_final_train)

    test_predictions = selected_model.predict(X_test)

    test_metrics = calculate_regression_metrics(
        y_test,
        test_predictions,
    )

    bootstrap_intervals = calculate_bootstrap_metric_intervals(
        y_test,
        test_predictions,
    )

    test_metrics_record = {
        "selected_model": selected_model_name,
        **test_metrics,
        **flatten_bootstrap_intervals(bootstrap_intervals),
    }

    print_metrics("POST-INSTALLATION TEST METRICS", test_metrics)

    print_bootstrap_intervals(
        "POST-INSTALLATION TEST BOOTSTRAP 95% CONFIDENCE INTERVALS",
        bootstrap_intervals,
    )

    test_predictions_df = build_prediction_table(
        dates=model_df.loc[post_test_mask, DATE_COL],
        y_true=y_test,
        y_pred=test_predictions,
    )

    bootstrap_intervals_df = build_bootstrap_intervals_table(bootstrap_intervals)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_feature_list(FEATURE_COLUMNS, FEATURE_LIST_PATH)

    with pd.ExcelWriter(POST_ONLY_REPORT_PATH, engine="openpyxl") as writer:
        validation_metrics_df.to_excel(
            writer,
            sheet_name="Validation Metrics",
            index=False,
        )

        pd.DataFrame([test_metrics_record]).to_excel(
            writer,
            sheet_name="Test Metrics",
            index=False,
        )

        bootstrap_intervals_df.to_excel(
            writer,
            sheet_name="Bootstrap CI",
            index=False,
        )

        test_predictions_df.to_excel(
            writer,
            sheet_name="Test Predictions",
            index=False,
        )

    joblib.dump(selected_model, SELECTED_MODEL_PATH)

    test_dates = model_df.loc[post_test_mask, DATE_COL]

    plot_actual_vs_predicted(
        dates=test_dates,
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "post_only_actual_vs_predicted.png",
        title="Post-Installation Test: Actual vs Predicted",
    )

    plot_residuals_over_time(
        dates=test_dates,
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "post_only_residuals_over_time.png",
        title="Post-Installation Test: Residuals Over Time",
    )

    plot_residuals_vs_predicted(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "post_only_residuals_vs_predicted.png",
        title="Post-Installation Test: Residuals vs Predicted",
    )

    plot_actual_vs_predicted_scatter(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "post_only_actual_vs_predicted_scatter.png",
        title="Post-Installation Test: Actual vs Predicted Scatter",
    )

    plot_residual_distribution(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "post_only_residual_distribution.png",
        title="Post-Installation Test: Residual Distribution",
    )

    print("\nSaved outputs:")
    print(f"- {POST_ONLY_REPORT_PATH}")
    print(f"- {SELECTED_MODEL_PATH}")
    print(f"- {FEATURE_LIST_PATH}")
    print(f"- {FIGURES_DIR}")

    if log_mlflow:
        log_run_to_mlflow(
            model=selected_model,
            selected_model_name=selected_model_name,
            model_builders=model_builders,
            validation_metrics_df=validation_metrics_df,
            test_metrics_record=test_metrics_record,
            split_sizes=split_sizes,
        )
    else:
        print("\nMLflow logging skipped.")
        print("Run with --log-mlflow to log this experiment.")


if __name__ == "__main__":
    args = parse_args()
    main(log_mlflow=args.log_mlflow)