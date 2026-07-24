from collections.abc import Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true, y_pred) -> float:
    """
    Calculate Mean Absolute Error.
    """
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true, y_pred) -> float:
    """
    Calculate Root Mean Squared Error.
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def calculate_r2(y_true, y_pred) -> float:
    """
    Calculate R² score.
    """
    return float(r2_score(y_true, y_pred))


def calculate_mape(y_true, y_pred) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Returns the result as a percentage.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if np.any(actual == 0):
        raise ValueError("MAPE cannot be calculated when actual values contain zero.")

    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def calculate_total_deviation_pct(y_true, y_pred) -> float:
    """
    Calculate total deviation percentage.

    Formula:
    (sum(predicted) - sum(actual)) / sum(actual) * 100

    Interpretation:
    - positive value: model overpredicts total energy
    - negative value: model underpredicts total energy
    """
    actual_total = np.sum(y_true)
    predicted_total = np.sum(y_pred)

    if actual_total == 0:
        raise ValueError("Actual total is zero; cannot calculate deviation percentage.")

    return float((predicted_total - actual_total) / actual_total * 100)


def calculate_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Calculate the main forecasting regression metrics.
    """
    return {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
        "total_deviation_pct": calculate_total_deviation_pct(y_true, y_pred),
    }


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 33,
) -> tuple[float, float]:
    """
    Calculate a bootstrap confidence interval for one metric.

    The function resamples the evaluation rows with replacement and recalculates
    the metric many times. The lower and upper percentiles form the confidence
    interval.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.shape[0] != predicted.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if actual.shape[0] < 2:
        raise ValueError("At least two observations are required for bootstrapping.")

    rng = np.random.default_rng(random_state)
    n_rows = actual.shape[0]

    bootstrap_scores = []

    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, n_rows, size=n_rows)

        sample_actual = actual[sample_indices]
        sample_predicted = predicted[sample_indices]

        bootstrap_scores.append(metric_fn(sample_actual, sample_predicted))

    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return float(lower_bound), float(upper_bound)


def calculate_bootstrap_metric_intervals(
    y_true,
    y_pred,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 33,
) -> dict[str, tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for the main forecasting metrics.
    """
    return {
        "mae_ci": bootstrap_metric_ci(
            y_true,
            y_pred,
            metric_fn=calculate_mae,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        ),
        "rmse_ci": bootstrap_metric_ci(
            y_true,
            y_pred,
            metric_fn=calculate_rmse,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        ),
        "mape_ci": bootstrap_metric_ci(
            y_true,
            y_pred,
            metric_fn=calculate_mape,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        ),
        "total_deviation_pct_ci": bootstrap_metric_ci(
            y_true,
            y_pred,
            metric_fn=calculate_total_deviation_pct,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        ),
    }