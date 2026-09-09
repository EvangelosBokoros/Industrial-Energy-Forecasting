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
from src.feature_sets import get_feature_columns
from src.model_full_history import get_full_history_model_builders
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
    FULL_HISTORY_TEST_START,
    FULL_HISTORY_TRAIN_END,
    FULL_HISTORY_VALIDATION_END,
    FULL_HISTORY_VALIDATION_START,
    TARGET_COL,
)


DATA_PATH = Path("data/processed/industrial_energy_daily.csv")

REPORTS_DIR = Path("reports")
FIGURES_BASE_DIR = REPORTS_DIR / "figures"
METADATA_DIR = REPORTS_DIR / "metadata"
MODELS_DIR = Path("models")

FULL_HISTORY_REPORT_PATH = REPORTS_DIR / "full_history_training_report.xlsx"
FULL_HISTORY_FEATURE_LIST_PATH = METADATA_DIR / "full_history_features.txt"
FULL_HISTORY_SELECTED_MODEL_PATH = MODELS_DIR / "full_history_model.joblib"
FULL_HISTORY_FIGURES_DIR = FIGURES_BASE_DIR / "full_history"

FULL_HISTORY_VIF_REPORT_PATH = (
    REPORTS_DIR / "full_history_vif_advanced_boosting_training_report.xlsx"
)
FULL_HISTORY_VIF_FEATURE_LIST_PATH = (
    METADATA_DIR / "full_history_vif_advanced_boosting_features.txt"
)
FULL_HISTORY_VIF_SELECTED_MODEL_PATH = (
    MODELS_DIR / "full_history_vif_advanced_boosting_model.joblib"
)
FULL_HISTORY_VIF_FIGURES_DIR = FIGURES_BASE_DIR / "full_history_vif_advanced_boosting"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "Industrial Energy Forecasting"

RUN_CONFIGS = {
    "full": {
        "modeling_version": "1.1",
        "run_name": "1.1 Full-History Optuna Tuned Candidate Comparison",
        "run_type": "full_history_optuna_tuned_candidate",
        "training_info": (
            "1.1 Full-History Optuna Tuned Candidate Comparison. "
            "Hyperparameters selected using the Optuna tuning script."
        ),
    },
    "vif_auto_full_history": {
        "modeling_version": "1.3",
        "run_name": "1.3 Full-History VIF-Reduced Advanced Boosting Candidate Comparison",
        "run_type": "full_history_vif_advanced_boosting_candidate_comparison",
        "training_info": (
            "1.3 Full-History VIF-Reduced Advanced Boosting Candidate Comparison. "
            "Compares the best known VIF-reduced full-history candidates, including "
            "Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost, and "
            "AdaBoost. AdaBoost hyperparameters were selected from the 2000-trial "
            "Optuna run. The 10000-trial run was treated as a sensitivity diagnostic "
            "because the validation window contains only 7 days."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the industrial full-history forecasting model."
    )

    parser.add_argument(
        "--feature-set",
        default="full",
        choices=sorted(RUN_CONFIGS.keys()),
        help=(
            "Named full-history feature set to use for training. "
            "Defaults to 'full'."
        ),
    )

    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log this run to MLflow.",
    )

    return parser.parse_args()


def build_run_config(feature_set_name: str) -> dict[str, str]:
    """
    Return the run configuration for a full-history feature set.
    """
    if feature_set_name not in RUN_CONFIGS:
        available_feature_sets = ", ".join(sorted(RUN_CONFIGS.keys()))
        raise ValueError(
            f"Unknown full-history feature set: {feature_set_name}. "
            f"Available full-history feature sets: {available_feature_sets}"
        )

    return RUN_CONFIGS[feature_set_name]


def build_output_paths(
    feature_set_name: str,
) -> tuple[Path, Path, Path, Path]:
    """
    Build output paths for reports, feature metadata, model artifact, and figures.

    The default full-feature run keeps the original full-history output paths.
    The VIF-reduced 1.3 run gets separate advanced-boosting paths to avoid
    overwriting earlier VIF diagnostic artifacts.
    """
    if feature_set_name == "full":
        return (
            FULL_HISTORY_REPORT_PATH,
            FULL_HISTORY_FEATURE_LIST_PATH,
            FULL_HISTORY_SELECTED_MODEL_PATH,
            FULL_HISTORY_FIGURES_DIR,
        )

    if feature_set_name == "vif_auto_full_history":
        return (
            FULL_HISTORY_VIF_REPORT_PATH,
            FULL_HISTORY_VIF_FEATURE_LIST_PATH,
            FULL_HISTORY_VIF_SELECTED_MODEL_PATH,
            FULL_HISTORY_VIF_FIGURES_DIR,
        )

    raise ValueError(f"Unknown feature set for output paths: {feature_set_name}")


def create_full_history_split_masks(
    model_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Create full-history train, validation, and test masks.

    The full-history training split uses all available rows up to
    FULL_HISTORY_TRAIN_END. Validation and test use the same post-installation
    validation and test windows as the post-only benchmark.
    """
    train_mask = model_df[DATE_COL] <= pd.to_datetime(FULL_HISTORY_TRAIN_END)

    validation_mask = (
        model_df[DATE_COL] >= pd.to_datetime(FULL_HISTORY_VALIDATION_START)
    ) & (
        model_df[DATE_COL] <= pd.to_datetime(FULL_HISTORY_VALIDATION_END)
    )

    test_mask = model_df[DATE_COL] >= pd.to_datetime(FULL_HISTORY_TEST_START)

    return train_mask, validation_mask, test_mask


def assert_no_split_overlap(split_masks: dict[str, pd.Series]) -> None:
    """
    Ensure that train, validation, and test masks do not overlap.
    """
    split_names = list(split_masks.keys())

    for index, first_split_name in enumerate(split_names):
        for second_split_name in split_names[index + 1:]:
            overlap = split_masks[first_split_name] & split_masks[second_split_name]

            if overlap.any():
                raise ValueError(
                    f"Split overlap detected between "
                    f"{first_split_name} and {second_split_name}."
                )


def assert_non_empty_splits(split_masks: dict[str, pd.Series]) -> None:
    """
    Ensure that no split is empty.
    """
    for split_name, split_mask in split_masks.items():
        if split_mask.sum() == 0:
            raise ValueError(f"{split_name} split is empty.")


def summarize_split_sizes(split_masks: dict[str, pd.Series]) -> dict[str, int]:
    """
    Summarize row counts for each split.
    """
    return {
        split_name: int(split_mask.sum())
        for split_name, split_mask in split_masks.items()
    }


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


def build_run_summary_table(
    run_config: dict[str, str],
    feature_set_name: str,
    selected_feature_columns: list[str],
    split_sizes: dict[str, int],
) -> pd.DataFrame:
    """
    Build a compact run summary table for the Excel report.
    """
    summary = {
        "modeling_version": run_config["modeling_version"],
        "training_scope": "full_history",
        "tuning_method": "optuna",
        "hyperparameter_source": "src/train_full_history_optuna.py",
        "feature_set_name": feature_set_name,
        "feature_count": len(selected_feature_columns),
        "selected_features": ", ".join(selected_feature_columns),
        "full_history_train_end": FULL_HISTORY_TRAIN_END,
        "full_history_validation_start": FULL_HISTORY_VALIDATION_START,
        "full_history_validation_end": FULL_HISTORY_VALIDATION_END,
        "full_history_test_start": FULL_HISTORY_TEST_START,
    }

    for split_name, row_count in split_sizes.items():
        summary[f"{split_name}_rows"] = row_count

    return pd.DataFrame([summary])


def save_feature_list(
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """
    Save the feature list used in the run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Feature columns used by the full-history model:",
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
    training_metrics_df: pd.DataFrame,
    validation_metrics_df: pd.DataFrame,
    test_metrics_record: dict[str, float],
    split_sizes: dict[str, int],
    run_config: dict[str, str],
    feature_set_name: str,
    selected_feature_columns: list[str],
    report_path: Path,
    feature_list_path: Path,
    selected_model_path: Path,
    figures_dir: Path,
) -> None:
    """
    Log selected run information to MLflow.

    This function is only called when running:

        python -m src.train_full_history --log-mlflow
    """
    selected_training_metrics = training_metrics_df.loc[
        training_metrics_df["model_name"] == selected_model_name
    ].iloc[0]

    selected_validation_metrics = validation_metrics_df.loc[
        validation_metrics_df["model_name"] == selected_model_name
    ].iloc[0]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_config["run_name"]) as run:
        mlflow.log_param("modeling_version", run_config["modeling_version"])
        mlflow.log_param("training_scope", "full_history")
        mlflow.log_param("tuning_method", "optuna")
        mlflow.log_param("hyperparameter_source", "src/train_full_history_optuna.py")
        mlflow.log_param("feature_set_name", feature_set_name)
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("selected_model_name", selected_model_name)
        mlflow.log_param("date_column", DATE_COL)
        mlflow.log_param("target_column", TARGET_COL)
        mlflow.log_param("feature_count", len(selected_feature_columns))
        mlflow.log_param("candidate_models", ",".join(model_builders.keys()))

        mlflow.log_param("full_history_train_end", FULL_HISTORY_TRAIN_END)
        mlflow.log_param(
            "full_history_validation_start",
            FULL_HISTORY_VALIDATION_START,
        )
        mlflow.log_param(
            "full_history_validation_end",
            FULL_HISTORY_VALIDATION_END,
        )
        mlflow.log_param("full_history_test_start", FULL_HISTORY_TEST_START)

        for split_name, row_count in split_sizes.items():
            mlflow.log_param(f"{split_name}_rows", row_count)

        for parameter_name, parameter_value in model.get_params().items():
            mlflow.log_param(f"model__{parameter_name}", parameter_value)

        mlflow.set_tag("project", "industrial_energy_forecasting")
        mlflow.set_tag("objective", "full_history_forecasting")
        mlflow.set_tag("run_type", run_config["run_type"])
        mlflow.set_tag("selection_stage", "validation")
        mlflow.set_tag("features", ", ".join(selected_feature_columns))
        mlflow.set_tag("raw_data_logged", "False")
        mlflow.set_tag("hyperparameter_source", "src/train_full_history_optuna.py")
        mlflow.set_tag("training_info", run_config["training_info"])

        metric_names = [
            "mae",
            "rmse",
            "r2",
            "mape",
            "total_deviation_pct",
        ]

        for metric_name in metric_names:
            log_metric_if_valid(
                f"training_{metric_name}",
                float(selected_training_metrics[metric_name]),
            )

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
            report_path,
            mlflow_artifact_path="reports",
        )

        log_artifact_if_exists(
            selected_model_path,
            mlflow_artifact_path="models",
        )

        log_artifact_if_exists(
            feature_list_path,
            mlflow_artifact_path="metadata",
        )

        if figures_dir.exists():
            for figure_path in sorted(figures_dir.glob("*.png")):
                mlflow.log_artifact(
                    str(figure_path),
                    artifact_path="figures",
                )

        print("\nMLflow run logged.")
        print(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"Run name: {run_config['run_name']}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")


def main(
    log_mlflow: bool = False,
    feature_set_name: str = "full",
) -> None:
    """
    Run a full-history forecasting experiment.
    """
    run_config = build_run_config(feature_set_name)
    selected_feature_columns = get_feature_columns(feature_set_name)

    (
        report_path,
        feature_list_path,
        selected_model_path,
        figures_dir,
    ) = build_output_paths(feature_set_name)

    print("\nTraining scope: full_history")
    print(f"Feature set: {feature_set_name}")
    print(f"Feature count: {len(selected_feature_columns)}")

    raw_df = load_tabular_data(DATA_PATH)

    required_columns = [DATE_COL, TARGET_COL] + selected_feature_columns
    validate_required_columns(raw_df, required_columns)

    clean_df = clean_energy_dataset(
        raw_df,
        date_col=DATE_COL,
        target_col=TARGET_COL,
    )

    model_df = clean_df.dropna(
        subset=selected_feature_columns + [TARGET_COL],
    ).reset_index(drop=True)

    X = model_df[selected_feature_columns].copy()
    y = model_df[TARGET_COL].copy()

    train_mask, validation_mask, test_mask = create_full_history_split_masks(model_df)

    split_masks = {
        "full_history_train": train_mask,
        "full_history_validation": validation_mask,
        "full_history_test": test_mask,
    }

    assert_no_split_overlap(split_masks)
    assert_non_empty_splits(split_masks)

    split_sizes = summarize_split_sizes(split_masks)

    print("\nSplit sizes:")
    for split_name, row_count in split_sizes.items():
        print(f"{split_name}: {row_count}")

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    model_builders = get_full_history_model_builders()

    training_records = []
    validation_records = []

    for model_name, model_builder in model_builders.items():
        model = model_builder()
        model.fit(X_train, y_train)

        training_predictions = model.predict(X_train)
        training_metrics = calculate_regression_metrics(
            y_train,
            training_predictions,
        )

        validation_predictions = model.predict(X_validation)
        validation_metrics = calculate_regression_metrics(
            y_validation,
            validation_predictions,
        )

        training_records.append(
            {
                "model_name": model_name,
                **training_metrics,
            }
        )

        validation_records.append(
            {
                "model_name": model_name,
                **validation_metrics,
            }
        )

    training_metrics_df = pd.DataFrame(training_records)
    validation_metrics_df = pd.DataFrame(validation_records)

    validation_metrics_df = validation_metrics_df.sort_values("mae").reset_index(
        drop=True
    )

    model_order = validation_metrics_df["model_name"].tolist()

    training_metrics_df = (
        training_metrics_df.set_index("model_name")
        .loc[model_order]
        .reset_index()
    )

    selected_model_name = validation_metrics_df.loc[0, "model_name"]

    print("\nValidation model comparison:")
    print(validation_metrics_df.to_string(index=False))

    print("\nTraining diagnostics, not used for selection:")
    print(
        training_metrics_df[
            [
                "model_name",
                "mae",
                "r2",
                "total_deviation_pct",
            ]
        ].to_string(index=False)
    )

    print(f"\nSelected model by validation MAE: {selected_model_name}")

    final_train_mask = train_mask | validation_mask

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
        "training_scope": "full_history",
        "tuning_method": "optuna",
        "feature_set_name": feature_set_name,
        "selected_model": selected_model_name,
        **test_metrics,
        **flatten_bootstrap_intervals(bootstrap_intervals),
    }

    print_metrics("FULL-HISTORY TEST METRICS", test_metrics)

    print_bootstrap_intervals(
        "FULL-HISTORY TEST BOOTSTRAP 95% CONFIDENCE INTERVALS",
        bootstrap_intervals,
    )

    test_predictions_df = build_prediction_table(
        dates=model_df.loc[test_mask, DATE_COL],
        y_true=y_test,
        y_pred=test_predictions,
    )

    bootstrap_intervals_df = build_bootstrap_intervals_table(bootstrap_intervals)

    run_summary_df = build_run_summary_table(
        run_config=run_config,
        feature_set_name=feature_set_name,
        selected_feature_columns=selected_feature_columns,
        split_sizes=split_sizes,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_feature_list(selected_feature_columns, feature_list_path)

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        run_summary_df.to_excel(
            writer,
            sheet_name="Run Summary",
            index=False,
        )

        training_metrics_df.to_excel(
            writer,
            sheet_name="Training Metrics",
            index=False,
        )

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

    joblib.dump(selected_model, selected_model_path)

    test_dates = model_df.loc[test_mask, DATE_COL]

    plot_actual_vs_predicted(
        dates=test_dates,
        y_true=y_test,
        y_pred=test_predictions,
        output_path=figures_dir / "full_history_actual_vs_predicted.png",
        title="Full-History Test: Actual vs Predicted",
    )

    plot_residuals_over_time(
        dates=test_dates,
        y_true=y_test,
        y_pred=test_predictions,
        output_path=figures_dir / "full_history_residuals_over_time.png",
        title="Full-History Test: Residuals Over Time",
    )

    plot_residuals_vs_predicted(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=figures_dir / "full_history_residuals_vs_predicted.png",
        title="Full-History Test: Residuals vs Predicted",
    )

    plot_actual_vs_predicted_scatter(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=figures_dir / "full_history_actual_vs_predicted_scatter.png",
        title="Full-History Test: Actual vs Predicted Scatter",
    )

    plot_residual_distribution(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=figures_dir / "full_history_residual_distribution.png",
        title="Full-History Test: Residual Distribution",
    )

    print("\nSaved outputs:")
    print(f"- {report_path}")
    print(f"- {selected_model_path}")
    print(f"- {feature_list_path}")
    print(f"- {figures_dir}")

    if log_mlflow:
        log_run_to_mlflow(
            model=selected_model,
            selected_model_name=selected_model_name,
            model_builders=model_builders,
            training_metrics_df=training_metrics_df,
            validation_metrics_df=validation_metrics_df,
            test_metrics_record=test_metrics_record,
            split_sizes=split_sizes,
            run_config=run_config,
            feature_set_name=feature_set_name,
            selected_feature_columns=selected_feature_columns,
            report_path=report_path,
            feature_list_path=feature_list_path,
            selected_model_path=selected_model_path,
            figures_dir=figures_dir,
        )
    else:
        print("\nMLflow logging skipped.")
        print("Run with --log-mlflow to log this experiment.")


if __name__ == "__main__":
    args = parse_args()
    main(
        log_mlflow=args.log_mlflow,
        feature_set_name=args.feature_set,
    )
