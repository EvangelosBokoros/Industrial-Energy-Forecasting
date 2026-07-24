from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


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


def build_catboost_model(
    iterations: int = 200,
    learning_rate: float = 0.03,
    depth: int = 3,
    l2_leaf_reg: float = 10.0,
    random_seed: int = 33,
):
    """
    Build a regularized CatBoost regression model.

    This model is added as an additional candidate only.
    It does not replace or modify the existing baseline models.
    """
    if CatBoostRegressor is None:
        raise ImportError(
            "CatBoost is not installed. Install it with: python -m pip install catboost"
        )

    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=2,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=random_seed,
        allow_writing_files=False,
        verbose=False,
    )


def get_model_builders() -> dict[str, callable]:
    """
    Return the available model builders for the first post-installation
    forecasting comparison.
    """
    model_builders = {
        "dummy_mean": build_dummy_model,
        "linear_regression": build_linear_regression_model,
        "ridge_regression": build_ridge_regression_model,
        "random_forest": build_random_forest_model,
        "gradient_boosting": build_gradient_boosting_model,
    }

    if CatBoostRegressor is not None:
        model_builders["catboost_regularized"] = build_catboost_model

    return model_builders