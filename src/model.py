from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


def build_dummy_model() -> DummyRegressor:
    """
    Build a simple mean baseline model.

    This model predicts the average training target value.
    It is used as a minimum benchmark.
    """
    return DummyRegressor(strategy="mean")


def build_linear_regression_model() -> LinearRegression:
    """
    Build a Linear Regression baseline model.
    """
    return LinearRegression()


def build_ridge_regression_model(alpha: float = 1.0) -> Ridge:
    """
    Build a Ridge Regression model.

    Ridge is useful when the dataset is small or when features are correlated.
    """
    return Ridge(alpha=alpha)


def build_random_forest_model(
    n_estimators: int = 300,
    max_depth: int | None = 4,
    min_samples_leaf: int = 2,
    random_state: int = 33,
) -> RandomForestRegressor:
    """
    Build a controlled Random Forest model.

    The defaults are intentionally conservative because the post-installation
    dataset is small.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def build_gradient_boosting_model(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    min_samples_leaf: int = 3,
    random_state: int = 33,
) -> GradientBoostingRegressor:
    """
    Build a conservative Gradient Boosting model.

    The defaults use shallow trees and moderate learning rate to reduce
    overfitting risk on the small post-installation dataset.
    """
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


def get_model_builders() -> dict[str, callable]:
    """
    Return the available model builders for the first post-installation
    forecasting comparison.
    """
    return {
        "dummy_mean": build_dummy_model,
        "linear_regression": build_linear_regression_model,
        "ridge_regression": build_ridge_regression_model,
        "random_forest": build_random_forest_model,
        "gradient_boosting": build_gradient_boosting_model,
    }