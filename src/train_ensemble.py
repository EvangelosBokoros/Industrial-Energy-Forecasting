from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_tabular_data
from src.evaluate import (
    calculate_bootstrap_metric_intervals,
    calculate_regression_metrics,
)
from src.feature_sets import get_feature_columns
from src.model_ensemble import (
    ENSEMBLE_VERSION,
    FULL_HISTORY_MODEL_NAME,
    POST_ONLY_MODEL_NAME,
    build_full_history_champion_model,
    build_post_only_champion_model,
    calculate_weighted_ensemble_predictions,
    get_candidate_ensemble_weights,
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

REPORT_PATH = Path("reports/ensemble_2_0_training_report.xlsx")
MODEL_PATH = Path("models/ensemble_2_0_model.joblib")
FEATURE_LIST_PATH = Path("reports/metadata/ensemble_2_0_features.txt")
FIGURES_DIR = Path("reports/figures/ensemble_2_0")

RUN_NAME = "2.0 Post-Only and Full-History Ensemble Diagnostic"
RUN_TYPE = "post_only_full_history_ensemble_diagnostic"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "Industrial Energy Forecasting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the 2.0 post-only/full-history ensemble."
    )

    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log the ensemble experiment to MLflow.",
    )

    return parser.parse_args()


def ensure_output_directories() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_modeling_data() -> pd.DataFrame:
    df = load_tabular_data(DATA_PATH)
    df = clean_energy_dataset(df, DATE_COL, TARGET_COL)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")

    if df[DATE_COL].isna().any():
        raise ValueError("Some dates could not be parsed.")

    return df.sort_values(DATE_COL).reset_index(drop=True)


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_non_empty_split(split_name: str, split_df: pd.DataFrame) -> None:
    if split_df.empty:
        raise ValueError(f"{split_name} is empty. Check the split date settings.")


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


def calculate_metrics_record(
    model_name: str,
    y_true,
    y_pred,
) -> dict[str, float | str]:
    metrics = calculate_regression_metrics(y_true, y_pred)
    return {"model_name": model_name, **metrics}


def evaluate_candidate_weights(
    y_true,
    post_only_predictions,
    full_history_predictions,
) -> pd.DataFrame:
    records = []

    for post_only_weight in get_candidate_ensemble_weights():
        ensemble_predictions = calculate_weighted_ensemble_predictions(
            post_only_predictions=post_only_predictions,
            full_history_predictions=full_history_predictions,
            post_only_weight=post_only_weight,
        )

        metrics = calculate_regression_metrics(y_true, ensemble_predictions)

        records.append(
            {
                "post_only_weight": post_only_weight,
                "full_history_weight": round(1 - post_only_weight, 1),
                **metrics,
            }
        )

    return pd.DataFrame(records).sort_values("mae").reset_index(drop=True)


def save_prediction_plot(
    predictions_df: pd.DataFrame,
    prediction_column: str,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(
        predictions_df[DATE_COL],
        predictions_df[TARGET_COL],
        marker="o",
        label="Actual",
    )
    plt.plot(
        predictions_df[DATE_COL],
        predictions_df[prediction_column],
        marker="o",
        label="Predicted",
    )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_residual_plot(
    predictions_df: pd.DataFrame,
    residual_column: str,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.axhline(0, linestyle="--")
    plt.bar(predictions_df[DATE_COL].astype(str), predictions_df[residual_column])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_component_comparison_plot(
    predictions_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(
        predictions_df[DATE_COL],
        predictions_df[TARGET_COL],
        marker="o",
        label="Actual",
    )
    plt.plot(
        predictions_df[DATE_COL],
        predictions_df["post_only_prediction"],
        marker="o",
        label="Post-only",
    )
    plt.plot(
        predictions_df[DATE_COL],
        predictions_df["full_history_prediction"],
        marker="o",
        label="Full-history",
    )
    plt.plot(
        predictions_df[DATE_COL],
        predictions_df["selected_ensemble_prediction"],
        marker="o",
        label="Selected ensemble",
    )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_weight_search_plot(
    validation_weight_results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(
        validation_weight_results_df["post_only_weight"],
        validation_weight_results_df["mae"],
        marker="o",
    )
    plt.title("2.0 Validation MAE by Ensemble Weight")
    plt.xlabel("Post-only model weight")
    plt.ylabel("Validation MAE")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_list(
    post_only_features: list[str],
    full_history_features: list[str],
    selected_weight: float,
) -> None:
    with FEATURE_LIST_PATH.open("w", encoding="utf-8") as file:
        file.write(f"Ensemble version: {ENSEMBLE_VERSION}\n")
        file.write(f"Run name: {RUN_NAME}\n\n")

        file.write("Post-only component:\n")
        file.write(f"- model: {POST_ONLY_MODEL_NAME}\n")
        file.write("- training scope: post-only\n")
        file.write("- source version: 0.5\n")
        file.write("- feature set: full\n")
        file.write("- features:\n")
        for feature in post_only_features:
            file.write(f"  - {feature}\n")

        file.write("\nFull-history component:\n")
        file.write(f"- model: {FULL_HISTORY_MODEL_NAME}\n")
        file.write("- training scope: full-history\n")
        file.write("- source version: 1.3\n")
        file.write("- feature set: vif_auto_full_history\n")
        file.write("- features:\n")
        for feature in full_history_features:
            file.write(f"  - {feature}\n")

        file.write("\nSelected ensemble:\n")
        file.write("- ensemble type: constrained weighted prediction average\n")
        file.write(f"- post-only weight: {selected_weight:.1f}\n")
        file.write(f"- full-history weight: {1 - selected_weight:.1f}\n")


def save_report(
    training_metrics_df: pd.DataFrame,
    validation_weight_results_df: pd.DataFrame,
    validation_component_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    validation_predictions_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(REPORT_PATH) as writer:
        training_metrics_df.to_excel(
            writer,
            sheet_name="Training Diagnostics",
            index=False,
        )
        validation_weight_results_df.to_excel(
            writer,
            sheet_name="Validation Weight Search",
            index=False,
        )
        validation_component_metrics_df.to_excel(
            writer,
            sheet_name="Validation Components",
            index=False,
        )
        test_metrics_df.to_excel(
            writer,
            sheet_name="Test Metrics",
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
        metadata_df.to_excel(
            writer,
            sheet_name="Metadata",
            index=False,
        )


def log_run_to_mlflow(
    selected_weight: float,
    validation_best_metrics: dict,
    selected_test_metrics: dict,
    test_intervals: dict,
    training_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tags(
            {
                "project": "industrial_energy_forecasting",
                "modeling_version": ENSEMBLE_VERSION,
                "run_type": RUN_TYPE,
                "objective": "ensemble_forecasting",
                "selection_stage": "validation",
                "source": "src/train_ensemble.py",
            }
        )

        mlflow.log_params(
            {
                "ensemble_type": "constrained_weighted_prediction_average",
                "candidate_post_only_weights": ", ".join(
                    str(weight) for weight in get_candidate_ensemble_weights()
                ),
                "post_only_model": POST_ONLY_MODEL_NAME,
                "post_only_source_version": "0.5",
                "post_only_feature_set": "full",
                "full_history_model": FULL_HISTORY_MODEL_NAME,
                "full_history_source_version": "1.3",
                "full_history_feature_set": "vif_auto_full_history",
                "selected_post_only_weight": selected_weight,
                "selected_full_history_weight": 1 - selected_weight,
            }
        )

        for metric_name in ["mae", "rmse", "r2", "mape", "total_deviation_pct"]:
            mlflow.log_metric(
                f"validation_{metric_name}",
                float(validation_best_metrics[metric_name]),
            )
            mlflow.log_metric(
                f"test_{metric_name}",
                float(selected_test_metrics[metric_name]),
            )

        for _, row in training_metrics_df.iterrows():
            model_name = row["model_name"]

            for metric_name in ["mae", "rmse", "r2", "mape", "total_deviation_pct"]:
                mlflow.log_metric(
                    f"{model_name}_training_{metric_name}",
                    float(row[metric_name]),
                )

        for _, row in test_metrics_df.iterrows():
            model_name = row["model_name"]

            for metric_name in ["mae", "rmse", "r2", "mape", "total_deviation_pct"]:
                mlflow.log_metric(
                    f"{model_name}_test_{metric_name}",
                    float(row[metric_name]),
                )

        for interval_name, interval_values in test_intervals.items():
            lower_bound, upper_bound = interval_values
            mlflow.log_metric(f"test_{interval_name}_lower", lower_bound)
            mlflow.log_metric(f"test_{interval_name}_upper", upper_bound)

        mlflow.log_artifact(str(REPORT_PATH))
        mlflow.log_artifact(str(FEATURE_LIST_PATH))
        mlflow.log_artifact(str(MODEL_PATH))

        for figure_path in FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(figure_path))


def main() -> None:
    args = parse_args()
    ensure_output_directories()

    df = load_modeling_data()

    post_only_features = get_feature_columns("full")
    full_history_features = get_feature_columns("vif_auto_full_history")

    required_columns = list(
        set([DATE_COL, TARGET_COL] + post_only_features + full_history_features)
    )
    validate_columns(df, required_columns)

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

    post_only_model = build_post_only_champion_model()
    full_history_model = build_full_history_champion_model()

    post_only_model.fit(X_post_train, y_post_train)
    full_history_model.fit(X_full_history_train, y_full_history_train)

    post_train_predictions = post_only_model.predict(X_post_train)
    full_history_train_predictions = full_history_model.predict(X_full_history_train)

    post_validation_predictions = post_only_model.predict(X_post_validation)
    full_history_validation_predictions = full_history_model.predict(
        X_full_history_validation
    )

    validation_weight_results_df = evaluate_candidate_weights(
        y_true=y_validation,
        post_only_predictions=post_validation_predictions,
        full_history_predictions=full_history_validation_predictions,
    )

    best_weight_row = validation_weight_results_df.iloc[0]
    selected_post_only_weight = float(best_weight_row["post_only_weight"])

    selected_validation_predictions = calculate_weighted_ensemble_predictions(
        post_only_predictions=post_validation_predictions,
        full_history_predictions=full_history_validation_predictions,
        post_only_weight=selected_post_only_weight,
    )

    post_train_full_history_predictions = full_history_model.predict(
        post_train_df[full_history_features]
    )

    selected_training_ensemble_predictions = calculate_weighted_ensemble_predictions(
        post_only_predictions=post_train_predictions,
        full_history_predictions=post_train_full_history_predictions,
        post_only_weight=selected_post_only_weight,
    )

    training_metrics_df = pd.DataFrame(
        [
            calculate_metrics_record(
                f"{POST_ONLY_MODEL_NAME}_on_post_train",
                y_post_train,
                post_train_predictions,
            ),
            calculate_metrics_record(
                f"{FULL_HISTORY_MODEL_NAME}_on_full_history_train",
                y_full_history_train,
                full_history_train_predictions,
            ),
            calculate_metrics_record(
                "selected_weighted_ensemble_on_post_train",
                y_post_train,
                selected_training_ensemble_predictions,
            ),
        ]
    ).sort_values("mae")

    validation_component_metrics_df = pd.DataFrame(
        [
            calculate_metrics_record(
                POST_ONLY_MODEL_NAME,
                y_validation,
                post_validation_predictions,
            ),
            calculate_metrics_record(
                FULL_HISTORY_MODEL_NAME,
                y_validation,
                full_history_validation_predictions,
            ),
            calculate_metrics_record(
                "selected_weighted_ensemble",
                y_validation,
                selected_validation_predictions,
            ),
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

    post_test_predictions = final_post_only_model.predict(X_post_test)
    full_history_test_predictions = final_full_history_model.predict(X_full_history_test)

    selected_test_predictions = calculate_weighted_ensemble_predictions(
        post_only_predictions=post_test_predictions,
        full_history_predictions=full_history_test_predictions,
        post_only_weight=selected_post_only_weight,
    )

    test_metrics_df = pd.DataFrame(
        [
            calculate_metrics_record(
                POST_ONLY_MODEL_NAME,
                y_test,
                post_test_predictions,
            ),
            calculate_metrics_record(
                FULL_HISTORY_MODEL_NAME,
                y_test,
                full_history_test_predictions,
            ),
            calculate_metrics_record(
                "selected_weighted_ensemble",
                y_test,
                selected_test_predictions,
            ),
        ]
    ).sort_values("mae")

    selected_test_metrics = calculate_regression_metrics(
        y_test,
        selected_test_predictions,
    )

    test_intervals = calculate_bootstrap_metric_intervals(
        y_test,
        selected_test_predictions,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=33,
    )

    validation_predictions_df = post_validation_df[[DATE_COL, TARGET_COL]].copy()
    validation_predictions_df["post_only_prediction"] = post_validation_predictions
    validation_predictions_df["full_history_prediction"] = (
        full_history_validation_predictions
    )
    validation_predictions_df["selected_ensemble_prediction"] = (
        selected_validation_predictions
    )
    validation_predictions_df["selected_ensemble_residual"] = (
        validation_predictions_df[TARGET_COL]
        - validation_predictions_df["selected_ensemble_prediction"]
    )

    test_predictions_df = post_test_df[[DATE_COL, TARGET_COL]].copy()
    test_predictions_df["post_only_prediction"] = post_test_predictions
    test_predictions_df["full_history_prediction"] = full_history_test_predictions
    test_predictions_df["selected_ensemble_prediction"] = selected_test_predictions
    test_predictions_df["selected_ensemble_residual"] = (
        test_predictions_df[TARGET_COL]
        - test_predictions_df["selected_ensemble_prediction"]
    )

    metadata_df = pd.DataFrame(
        [
            {"item": "modeling_version", "value": ENSEMBLE_VERSION},
            {"item": "run_name", "value": RUN_NAME},
            {"item": "run_type", "value": RUN_TYPE},
            {
                "item": "ensemble_type",
                "value": "constrained_weighted_prediction_average",
            },
            {
                "item": "candidate_post_only_weights",
                "value": ", ".join(str(weight) for weight in get_candidate_ensemble_weights()),
            },
            {"item": "post_only_model", "value": POST_ONLY_MODEL_NAME},
            {"item": "post_only_source_version", "value": "0.5"},
            {"item": "post_only_feature_set", "value": "full"},
            {"item": "full_history_model", "value": FULL_HISTORY_MODEL_NAME},
            {"item": "full_history_source_version", "value": "1.3"},
            {"item": "full_history_feature_set", "value": "vif_auto_full_history"},
            {"item": "selected_post_only_weight", "value": selected_post_only_weight},
            {
                "item": "selected_full_history_weight",
                "value": 1 - selected_post_only_weight,
            },
            {"item": "selection_metric", "value": "validation_mae"},
            {"item": "post_train_rows", "value": len(post_train_df)},
            {"item": "full_history_train_rows", "value": len(full_history_train_df)},
            {"item": "validation_rows", "value": len(post_validation_df)},
            {"item": "test_rows", "value": len(post_test_df)},
        ]
    )

    save_prediction_plot(
        predictions_df=validation_predictions_df,
        prediction_column="selected_ensemble_prediction",
        output_path=FIGURES_DIR / "selected_ensemble_validation_predictions.png",
        title="2.0 Selected Ensemble Validation Predictions",
    )

    save_residual_plot(
        predictions_df=validation_predictions_df,
        residual_column="selected_ensemble_residual",
        output_path=FIGURES_DIR / "selected_ensemble_validation_residuals.png",
        title="2.0 Selected Ensemble Validation Residuals",
    )

    save_prediction_plot(
        predictions_df=test_predictions_df,
        prediction_column="selected_ensemble_prediction",
        output_path=FIGURES_DIR / "selected_ensemble_test_predictions.png",
        title="2.0 Selected Ensemble Test Predictions",
    )

    save_residual_plot(
        predictions_df=test_predictions_df,
        residual_column="selected_ensemble_residual",
        output_path=FIGURES_DIR / "selected_ensemble_test_residuals.png",
        title="2.0 Selected Ensemble Test Residuals",
    )

    save_component_comparison_plot(
        predictions_df=test_predictions_df,
        output_path=FIGURES_DIR / "test_component_comparison.png",
        title="2.0 Test Component Comparison",
    )

    save_weight_search_plot(
        validation_weight_results_df=validation_weight_results_df,
        output_path=FIGURES_DIR / "validation_weight_search_mae.png",
    )

    save_report(
        training_metrics_df=training_metrics_df,
        validation_weight_results_df=validation_weight_results_df,
        validation_component_metrics_df=validation_component_metrics_df,
        test_metrics_df=test_metrics_df,
        validation_predictions_df=validation_predictions_df,
        test_predictions_df=test_predictions_df,
        metadata_df=metadata_df,
    )

    save_feature_list(
        post_only_features=post_only_features,
        full_history_features=full_history_features,
        selected_weight=selected_post_only_weight,
    )

    joblib.dump(
        {
            "modeling_version": ENSEMBLE_VERSION,
            "run_name": RUN_NAME,
            "ensemble_type": "constrained_weighted_prediction_average",
            "post_only_model": final_post_only_model,
            "full_history_model": final_full_history_model,
            "post_only_weight": selected_post_only_weight,
            "full_history_weight": 1 - selected_post_only_weight,
            "post_only_features": post_only_features,
            "full_history_features": full_history_features,
            "target_column": TARGET_COL,
        },
        MODEL_PATH,
    )

    print()
    print("Training scope: ensemble")
    print(f"Modeling version: {ENSEMBLE_VERSION}")
    print(f"Run name: {RUN_NAME}")
    print("Ensemble type: constrained weighted prediction average")
    print()
    print("Component models:")
    print(f"- {POST_ONLY_MODEL_NAME}: post-only 0.5 Extra Trees, full feature set")
    print(
        f"- {FULL_HISTORY_MODEL_NAME}: full-history 1.3 AdaBoost, "
        "vif_auto_full_history feature set"
    )
    print()
    print("Candidate post-only weights:")
    print(get_candidate_ensemble_weights())
    print()
    print("Split sizes:")
    print(f"post_train: {len(post_train_df)}")
    print(f"full_history_train: {len(full_history_train_df)}")
    print(f"validation: {len(post_validation_df)}")
    print(f"test: {len(post_test_df)}")
    print()
    print("Training diagnostics, not used for selection:")
    print(training_metrics_df.to_string(index=False))
    print()
    print("Validation weight search:")
    print(validation_weight_results_df.to_string(index=False))
    print()
    print("Validation component comparison:")
    print(validation_component_metrics_df.to_string(index=False))
    print()
    print(
        "Selected ensemble weight by validation MAE within constrained range: "
        f"post_only_weight={selected_post_only_weight:.1f}, "
        f"full_history_weight={1 - selected_post_only_weight:.1f}"
    )
    print()
    print("Test comparison:")
    print(test_metrics_df.to_string(index=False))
    print()
    print("Selected ensemble test bootstrap 95% confidence intervals:")
    for metric_name, interval_values in test_intervals.items():
        lower_bound, upper_bound = interval_values
        print(f"{metric_name}: [{lower_bound:.4f}, {upper_bound:.4f}]")

    print()
    print("Saved outputs:")
    print(f"- {REPORT_PATH}")
    print(f"- {MODEL_PATH}")
    print(f"- {FEATURE_LIST_PATH}")
    print(f"- {FIGURES_DIR}")

    if args.log_mlflow:
        log_run_to_mlflow(
            selected_weight=selected_post_only_weight,
            validation_best_metrics=best_weight_row.to_dict(),
            selected_test_metrics=selected_test_metrics,
            test_intervals=test_intervals,
            training_metrics_df=training_metrics_df,
            test_metrics_df=test_metrics_df,
        )
        print()
        print("MLflow logging completed.")
    else:
        print()
        print("MLflow logging skipped.")
        print("Run with --log-mlflow to log this experiment.")


if __name__ == "__main__":
    main()
