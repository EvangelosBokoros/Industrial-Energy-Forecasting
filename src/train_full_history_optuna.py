import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from src.data import load_tabular_data
from src.evaluate import (
    calculate_bootstrap_metric_intervals,
    calculate_regression_metrics,
)
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
    FULL_HISTORY_TEST_START,
    FULL_HISTORY_TRAIN_END,
    FULL_HISTORY_VALIDATION_END,
    FULL_HISTORY_VALIDATION_START,
    TARGET_COL,
)


DATA_PATH = Path("data/processed/damavand.csv")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures" / "full_history_optuna"
METADATA_DIR = REPORTS_DIR / "metadata"
MODELS_DIR = Path("models")

OPTUNA_REPORT_PATH = REPORTS_DIR / "full_history_optuna_training_report.xlsx"
FEATURE_LIST_PATH = METADATA_DIR / "full_history_optuna_features.txt"
SELECTED_MODEL_PATH = MODELS_DIR / "full_history_optuna_model.joblib"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "Damavand Energy Forecasting"
MODELING_VERSION = "1.1"
RUN_NAME = "1.1 Full-History Optuna Tuned Candidate Comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune full-history forecasting models with Optuna."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials to run.",
    )

    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log this run to MLflow.",
    )

    return parser.parse_args()


def create_full_history_split_masks(
    model_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_mask = model_df[DATE_COL] <= pd.to_datetime(FULL_HISTORY_TRAIN_END)

    validation_mask = (
        model_df[DATE_COL] >= pd.to_datetime(FULL_HISTORY_VALIDATION_START)
    ) & (
        model_df[DATE_COL] <= pd.to_datetime(FULL_HISTORY_VALIDATION_END)
    )

    test_mask = model_df[DATE_COL] >= pd.to_datetime(FULL_HISTORY_TEST_START)

    return train_mask, validation_mask, test_mask


def assert_no_split_overlap(split_masks: dict[str, pd.Series]) -> None:
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
    for split_name, split_mask in split_masks.items():
        if split_mask.sum() == 0:
            raise ValueError(f"{split_name} split is empty.")


def summarize_split_sizes(split_masks: dict[str, pd.Series]) -> dict[str, int]:
    return {
        split_name: int(split_mask.sum())
        for split_name, split_mask in split_masks.items()
    }


def build_model_from_trial(trial: optuna.Trial):
    model_type = trial.suggest_categorical(
        "model_type",
        ["random_forest", "extra_trees", "gradient_boosting"],
    )

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int(
                "rf_n_estimators",
                300,
                900,
                step=100,
            ),
            max_depth=trial.suggest_int("rf_max_depth", 3, 10),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 2, 15),
            max_features=trial.suggest_float(
                "rf_max_features",
                0.5,
                1.0,
                step=0.1,
            ),
            random_state=33,
            n_jobs=-1,
        )

    if model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int(
                "et_n_estimators",
                300,
                900,
                step=100,
            ),
            max_depth=trial.suggest_int("et_max_depth", 3, 10),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 2, 15),
            max_features=trial.suggest_float(
                "et_max_features",
                0.5,
                1.0,
                step=0.1,
            ),
            random_state=33,
            n_jobs=-1,
        )

    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int(
                "gb_n_estimators",
                50,
                400,
                step=50,
            ),
            learning_rate=trial.suggest_categorical(
                "gb_learning_rate",
                [0.01, 0.03, 0.05, 0.08, 0.1],
            ),
            max_depth=trial.suggest_int("gb_max_depth", 1, 4),
            min_samples_leaf=trial.suggest_int("gb_min_samples_leaf", 2, 20),
            subsample=trial.suggest_float(
                "gb_subsample",
                0.6,
                1.0,
                step=0.1,
            ),
            random_state=33,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def build_model_from_best_params(best_params: dict):
    model_type = best_params["model_type"]

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=best_params["rf_n_estimators"],
            max_depth=best_params["rf_max_depth"],
            min_samples_leaf=best_params["rf_min_samples_leaf"],
            max_features=best_params["rf_max_features"],
            random_state=33,
            n_jobs=-1,
        )

    if model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=best_params["et_n_estimators"],
            max_depth=best_params["et_max_depth"],
            min_samples_leaf=best_params["et_min_samples_leaf"],
            max_features=best_params["et_max_features"],
            random_state=33,
            n_jobs=-1,
        )

    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=best_params["gb_n_estimators"],
            learning_rate=best_params["gb_learning_rate"],
            max_depth=best_params["gb_max_depth"],
            min_samples_leaf=best_params["gb_min_samples_leaf"],
            subsample=best_params["gb_subsample"],
            random_state=33,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def build_best_trials_by_model_table(study: optuna.study.Study) -> pd.DataFrame:
    """
    Build a table with the best Optuna trial for each model family.

    The best trial is selected by validation MAE within each model family.
    """
    best_trials_by_model = {}

    for trial in study.trials:
        if trial.value is None:
            continue

        model_type = trial.params.get("model_type")

        if model_type is None:
            continue

        current_best_trial = best_trials_by_model.get(model_type)

        if current_best_trial is None or trial.value < current_best_trial.value:
            best_trials_by_model[model_type] = trial

    records = []

    for model_type, trial in best_trials_by_model.items():
        record = {
            "model_type": model_type,
            "trial_number": trial.number,
            "validation_mae": trial.value,
            "validation_rmse": trial.user_attrs.get("validation_rmse"),
            "validation_r2": trial.user_attrs.get("validation_r2"),
            "validation_mape": trial.user_attrs.get("validation_mape"),
            "validation_total_deviation_pct": trial.user_attrs.get(
                "validation_total_deviation_pct"
            ),
        }

        for parameter_name, parameter_value in trial.params.items():
            record[f"param_{parameter_name}"] = parameter_value

        records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values("validation_mae", ascending=True)
        .reset_index(drop=True)
    )


def build_prediction_table(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred,
) -> pd.DataFrame:
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
    split_sizes: dict[str, int],
    best_params: dict,
    best_value: float,
    n_trials: int,
) -> pd.DataFrame:
    summary = {
        "modeling_version": MODELING_VERSION,
        "training_scope": "full_history",
        "tuning_method": "optuna",
        "feature_set_name": "full",
        "feature_count": len(FEATURE_COLUMNS),
        "selected_features": ", ".join(FEATURE_COLUMNS),
        "optimization_metric": "validation_mae",
        "best_validation_mae": best_value,
        "n_trials": n_trials,
        "full_history_train_end": FULL_HISTORY_TRAIN_END,
        "full_history_validation_start": FULL_HISTORY_VALIDATION_START,
        "full_history_validation_end": FULL_HISTORY_VALIDATION_END,
        "full_history_test_start": FULL_HISTORY_TEST_START,
    }

    for split_name, row_count in split_sizes.items():
        summary[f"{split_name}_rows"] = row_count

    for parameter_name, parameter_value in best_params.items():
        summary[f"best_param_{parameter_name}"] = parameter_value

    return pd.DataFrame([summary])


def save_feature_list(
    feature_columns: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Feature columns used by the full-history Optuna model:",
        "",
    ]

    for index, feature_name in enumerate(feature_columns, start=1):
        lines.append(f"{index}. {feature_name}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n===== {title} =====")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


def print_bootstrap_intervals(
    title: str,
    bootstrap_intervals: dict[str, tuple[float, float]],
) -> None:
    print(f"\n===== {title} =====")

    for metric_name, interval in bootstrap_intervals.items():
        lower_bound, upper_bound = interval
        clean_metric_name = metric_name.replace("_ci", "")

        print(
            f"{clean_metric_name}: "
            f"[{lower_bound:.4f}, {upper_bound:.4f}]"
        )


def log_metric_if_valid(metric_name: str, metric_value: float) -> None:
    if np.isfinite(metric_value):
        mlflow.log_metric(metric_name, float(metric_value))


def log_artifact_if_exists(
    artifact_path: Path,
    mlflow_artifact_path: str,
) -> None:
    if artifact_path.exists():
        mlflow.log_artifact(
            str(artifact_path),
            artifact_path=mlflow_artifact_path,
        )


def log_run_to_mlflow(
    model,
    best_params: dict,
    validation_metrics: dict[str, float],
    test_metrics_record: dict[str, float],
    split_sizes: dict[str, int],
    n_trials: int,
) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_param("modeling_version", MODELING_VERSION)
        mlflow.log_param("training_scope", "full_history")
        mlflow.log_param("tuning_method", "optuna")
        mlflow.log_param("optimization_metric", "validation_mae")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("selected_model_name", best_params["model_type"])
        mlflow.log_param("date_column", DATE_COL)
        mlflow.log_param("target_column", TARGET_COL)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

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

        for parameter_name, parameter_value in best_params.items():
            mlflow.log_param(f"best__{parameter_name}", parameter_value)

        for parameter_name, parameter_value in model.get_params().items():
            mlflow.log_param(f"model__{parameter_name}", parameter_value)

        mlflow.set_tag("project", "damavand_energy_forecasting")
        mlflow.set_tag("objective", "full_history_forecasting")
        mlflow.set_tag("run_type", "full_history_optuna_tuning")
        mlflow.set_tag("selection_stage", "validation")
        mlflow.set_tag("features", ", ".join(FEATURE_COLUMNS))
        mlflow.set_tag("raw_data_logged", "False")
        mlflow.set_tag(
            "training_info",
            "1.1 Full-History Optuna Tuned Candidate Comparison.",
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
                float(validation_metrics[metric_name]),
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
            OPTUNA_REPORT_PATH,
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


def main(log_mlflow: bool = False, n_trials: int = 30) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\nTraining scope: full_history")
    print("Tuning method: optuna")
    print("Feature set: full")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Optuna trials: {n_trials}")
    print("Optimization metric: validation MAE")

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

    X = model_df[FEATURE_COLUMNS].copy()
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

    def objective(trial: optuna.Trial) -> float:
        model = build_model_from_trial(trial)

        model.fit(X_train, y_train)

        validation_predictions = model.predict(X_validation)
        validation_metrics = calculate_regression_metrics(
            y_validation,
            validation_predictions,
        )

        trial.set_user_attr("validation_rmse", validation_metrics["rmse"])
        trial.set_user_attr("validation_r2", validation_metrics["r2"])
        trial.set_user_attr("validation_mape", validation_metrics["mape"])
        trial.set_user_attr(
            "validation_total_deviation_pct",
            validation_metrics["total_deviation_pct"],
        )

        return validation_metrics["mae"]

    sampler = optuna.samplers.TPESampler(seed=33)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="full_history_optuna_tuning",
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_validation_mae = study.best_value

    print("\nBest Optuna trial overall:")
    print(f"validation_mae: {best_validation_mae:.4f}")

    print("\nBest overall parameters:")
    for parameter_name, parameter_value in best_params.items():
        print(f"{parameter_name}: {parameter_value}")

    best_trials_by_model_df = build_best_trials_by_model_table(study)

    print("\nBest Optuna trial by model family:")
    print(best_trials_by_model_df.to_string(index=False))

    best_model_for_validation = build_model_from_best_params(best_params)
    best_model_for_validation.fit(X_train, y_train)

    validation_predictions = best_model_for_validation.predict(X_validation)

    validation_metrics = calculate_regression_metrics(
        y_validation,
        validation_predictions,
    )

    print_metrics("FULL-HISTORY OPTUNA VALIDATION METRICS", validation_metrics)

    final_train_mask = train_mask | validation_mask

    X_final_train = X.loc[final_train_mask]
    y_final_train = y.loc[final_train_mask]

    selected_model = build_model_from_best_params(best_params)
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
        "feature_set_name": "full",
        "selected_model": best_params["model_type"],
        **test_metrics,
        **flatten_bootstrap_intervals(bootstrap_intervals),
    }

    print_metrics("FULL-HISTORY OPTUNA TEST METRICS", test_metrics)

    print_bootstrap_intervals(
        "FULL-HISTORY OPTUNA TEST BOOTSTRAP 95% CONFIDENCE INTERVALS",
        bootstrap_intervals,
    )

    test_predictions_df = build_prediction_table(
        dates=model_df.loc[test_mask, DATE_COL],
        y_true=y_test,
        y_pred=test_predictions,
    )

    bootstrap_intervals_df = build_bootstrap_intervals_table(bootstrap_intervals)

    run_summary_df = build_run_summary_table(
        split_sizes=split_sizes,
        best_params=best_params,
        best_value=best_validation_mae,
        n_trials=n_trials,
    )

    trials_df = study.trials_dataframe()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_feature_list(FEATURE_COLUMNS, FEATURE_LIST_PATH)

    with pd.ExcelWriter(OPTUNA_REPORT_PATH, engine="openpyxl") as writer:
        run_summary_df.to_excel(
            writer,
            sheet_name="Run Summary",
            index=False,
        )

        pd.DataFrame([validation_metrics]).to_excel(
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

        best_trials_by_model_df.to_excel(
            writer,
            sheet_name="Best By Model",
            index=False,
        )

        trials_df.to_excel(
            writer,
            sheet_name="Optuna Trials",
            index=False,
        )

    joblib.dump(selected_model, SELECTED_MODEL_PATH)

    test_dates = model_df.loc[test_mask, DATE_COL]

    plot_actual_vs_predicted(
        dates=test_dates,
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "full_history_optuna_actual_vs_predicted.png",
        title="Full-History Optuna Test: Actual vs Predicted",
    )

    plot_residuals_over_time(
        dates=test_dates,
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "full_history_optuna_residuals_over_time.png",
        title="Full-History Optuna Test: Residuals Over Time",
    )

    plot_residuals_vs_predicted(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "full_history_optuna_residuals_vs_predicted.png",
        title="Full-History Optuna Test: Residuals vs Predicted",
    )

    plot_actual_vs_predicted_scatter(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "full_history_optuna_actual_vs_predicted_scatter.png",
        title="Full-History Optuna Test: Actual vs Predicted Scatter",
    )

    plot_residual_distribution(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=FIGURES_DIR / "full_history_optuna_residual_distribution.png",
        title="Full-History Optuna Test: Residual Distribution",
    )

    print("\nSaved outputs:")
    print(f"- {OPTUNA_REPORT_PATH}")
    print(f"- {SELECTED_MODEL_PATH}")
    print(f"- {FEATURE_LIST_PATH}")
    print(f"- {FIGURES_DIR}")

    if log_mlflow:
        log_run_to_mlflow(
            model=selected_model,
            best_params=best_params,
            validation_metrics=validation_metrics,
            test_metrics_record=test_metrics_record,
            split_sizes=split_sizes,
            n_trials=n_trials,
        )
    else:
        print("\nMLflow logging skipped.")
        print("Run with --log-mlflow to log this experiment.")


if __name__ == "__main__":
    args = parse_args()
    main(
        log_mlflow=args.log_mlflow,
        n_trials=args.trials,
    )