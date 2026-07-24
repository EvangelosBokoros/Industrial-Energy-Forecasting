from __future__ import annotations

import inspect

from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def build_dummy_model() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def build_linear_regression_model() -> LinearRegression:
    return LinearRegression()


def build_ridge_regression_model(alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha)


def build_random_forest_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=700,
        max_depth=4,
        min_samples_leaf=2,
        max_features=0.9,
        random_state=33,
        n_jobs=-1,
    )


def build_extra_trees_model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=900,
        max_depth=6,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=33,
        n_jobs=-1,
    )


def build_gradient_boosting_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=33,
    )


def build_xgboost_regularized_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=15,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=50.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=33,
        n_jobs=-1,
        verbosity=0,
    )


def build_catboost_regularized_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=250,
        learning_rate=0.1,
        depth=3,
        l2_leaf_reg=20.0,
        random_strength=1.0,
        loss_function="RMSE",
        random_seed=33,
        verbose=False,
        allow_writing_files=False,
    )


def build_adaboost_regularized_model() -> AdaBoostRegressor:
    base_estimator = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=3,
        random_state=33,
    )

    parameters = {
        "n_estimators": 50,
        "learning_rate": 0.03,
        "loss": "linear",
        "random_state": 33,
    }

    signature = inspect.signature(AdaBoostRegressor)

    if "estimator" in signature.parameters:
        parameters["estimator"] = base_estimator
    else:
        parameters["base_estimator"] = base_estimator

    return AdaBoostRegressor(**parameters)


def get_full_history_model_builders() -> dict:
    return {
        "dummy_mean": build_dummy_model,
        "linear_regression": build_linear_regression_model,
        "ridge_regression": build_ridge_regression_model,
        "random_forest": build_random_forest_model,
        "extra_trees": build_extra_trees_model,
        "gradient_boosting": build_gradient_boosting_model,
        "xgboost_regularized": build_xgboost_regularized_model,
        "catboost_regularized": build_catboost_regularized_model,
        "adaboost_regularized": build_adaboost_regularized_model,
    }