from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


def build_dummy_model() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def build_linear_regression_model() -> LinearRegression:
    return LinearRegression()


def build_ridge_regression_model(alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha)


def build_random_forest_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        random_state=33,
        n_jobs=-1,
    )


def build_gradient_boosting_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
        random_state=33,
    )


def get_full_history_model_builders() -> dict:
    return {
        "dummy_mean": build_dummy_model,
        "linear_regression": build_linear_regression_model,
        "ridge_regression": build_ridge_regression_model,
        "random_forest": build_random_forest_model,
        "gradient_boosting": build_gradient_boosting_model,
    }