from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


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

) -> RandomForestRegressor:
    """
    Build a controlled Random Forest model.

    The defaults are intentionally conservative because the post-installation
    dataset is small.
    """
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        random_state=33,
        n_jobs=-1,
    )


def build_gradient_boosting_model(

) -> GradientBoostingRegressor:
    """
    Build a conservative Gradient Boosting model.

    The defaults use shallow trees and moderate learning rate to reduce
    overfitting risk on the small post-installation dataset.
    """
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
        random_state=33,
    )


def build_catboost_model():
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
        iterations=100,
        learning_rate=0.05,
        depth=2,
        random_seed=33,
        allow_writing_files=False,
        verbose=False,
    )


def build_xgboost_model():
    """
    Build a regularized XGBoost regression model.

    This model is added as an additional candidate only.
    It does not replace or modify the existing baseline models.
    """
    if XGBRegressor is None:
        raise ImportError(
            "XGBoost is not installed. Install it with: python -m pip install xgboost"
        )

    return XGBRegressor(
        objective="reg:tweedie",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=33,
        n_jobs=-1,
    )


def build_huber_regression_model() -> Pipeline:
    """
    Build a robust linear regression model with feature scaling.

    Huber regression is useful for small datasets where a few unusual days
    may strongly affect ordinary least squares.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000)),
        ]
    )


def build_bayesian_ridge_model() -> Pipeline:
    """
    Build a Bayesian Ridge regression model with feature scaling.

    Bayesian Ridge can be useful on small datasets because it regularizes
    coefficient estimates.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", BayesianRidge()),
        ]
    )


def build_elastic_net_model() -> Pipeline:
    """
    Build an ElasticNet regression model with feature scaling.

    ElasticNet combines L1 and L2 regularization and can help when predictors
    are correlated.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.3,
                    max_iter=10000,
                    random_state=33,
                ),
            ),
        ]
    )


def build_svr_rbf_model() -> Pipeline:
    """
    Build an RBF-kernel SVR model with feature scaling.

    SVR can sometimes perform well on very small nonlinear tabular datasets.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        ]
    )


def build_extra_trees_model() -> ExtraTreesRegressor:
    """
    Build an Extra Trees regression model.

    Extra Trees is tested as another tree-based ensemble candidate for
    small-data comparison.
    """
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        random_state=33,
        n_jobs=-1,
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
        "huber_regression": build_huber_regression_model,
        "bayesian_ridge": build_bayesian_ridge_model,
        "elastic_net": build_elastic_net_model,
        "svr_rbf_scaled": build_svr_rbf_model,
        "extra_trees": build_extra_trees_model,
    }

    if CatBoostRegressor is not None:
        model_builders["catboost_regularized"] = build_catboost_model

    if XGBRegressor is not None:
        model_builders["xgboost_regularized"] = build_xgboost_model

    return model_builders