from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_residuals(y_true, y_pred) -> np.ndarray:
    """
    Calculate residuals as actual minus predicted.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.shape[0] != predicted.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    return actual - predicted


def prepare_output_path(output_path: str | Path) -> Path:
    """
    Create the parent folder for a plot output path if needed.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def plot_actual_vs_predicted(
    dates,
    y_true,
    y_pred,
    output_path: str | Path,
    title: str = "Actual vs Predicted Energy",
) -> Path:
    """
    Plot actual and predicted energy consumption over time.
    """
    path = prepare_output_path(output_path)

    dates = pd.to_datetime(dates)
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(dates, actual, label="Actual")
    ax.plot(dates, predicted, label="Predicted")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy consumption (kWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_residuals_over_time(
    dates,
    y_true,
    y_pred,
    output_path: str | Path,
    title: str = "Residuals Over Time",
) -> Path:
    """
    Plot residuals over time.

    Residual = actual - predicted.
    """
    path = prepare_output_path(output_path)

    dates = pd.to_datetime(dates)
    residuals = calculate_residuals(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(dates, residuals, label="Residual")
    ax.axhline(0, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (kWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_residuals_vs_predicted(
    y_true,
    y_pred,
    output_path: str | Path,
    title: str = "Residuals vs Predicted Energy",
) -> Path:
    """
    Plot residuals against predicted values.
    """
    path = prepare_output_path(output_path)

    predicted = np.asarray(y_pred, dtype=float)
    residuals = calculate_residuals(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(predicted, residuals)
    ax.axhline(0, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Predicted energy consumption (kWh)")
    ax.set_ylabel("Residual (kWh)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_actual_vs_predicted_scatter(
    y_true,
    y_pred,
    output_path: str | Path,
    title: str = "Actual vs Predicted Scatter",
) -> Path:
    """
    Plot actual values against predicted values.

    A stronger model should place points close to the diagonal reference line.
    """
    path = prepare_output_path(output_path)

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(actual, predicted)
    ax.plot([min_value, max_value], [min_value, max_value], linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Actual energy consumption (kWh)")
    ax.set_ylabel("Predicted energy consumption (kWh)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_residual_distribution(
    y_true,
    y_pred,
    output_path: str | Path,
    title: str = "Residual Distribution",
) -> Path:
    """
    Plot the distribution of residuals.
    """
    path = prepare_output_path(output_path)

    residuals = calculate_residuals(y_true, y_pred)

    bin_count = min(12, max(5, len(residuals) // 2))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(residuals, bins=bin_count)
    ax.axvline(0, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Residual (kWh)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path