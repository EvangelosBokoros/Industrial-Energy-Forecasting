from __future__ import annotations

import inspect

import numpy as np
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor


ENSEMBLE_VERSION = "2.0"

POST_ONLY_MODEL_NAME = "post_only_extra_trees"
FULL_HISTORY_MODEL_NAME = "full_history_adaboost"


def build_post_only_champion_model() -> ExtraTreesRegressor:
    """
    Build the selected post-only champion model.

    This is the 0.5 post-only Extra Trees model.
    """
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        random_state=33,
        n_jobs=-1,
    )


def build_full_history_champion_model() -> AdaBoostRegressor:
    """
    Build the selected full-history champion model.

    This is the 1.3 full-history VIF-reduced AdaBoost model.
    """
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


def calculate_weighted_ensemble_predictions(
    post_only_predictions,
    full_history_predictions,
    post_only_weight: float,
) -> np.ndarray:
    """
    Combine post-only and full-history predictions using a weighted average.

    Formula:
        ensemble_prediction =
            post_only_weight * post_only_prediction
            + (1 - post_only_weight) * full_history_prediction
    """
    if not 0 <= post_only_weight <= 1:
        raise ValueError("post_only_weight must be between 0.0 and 1.0.")

    post_only_predictions = np.asarray(post_only_predictions, dtype=float)
    full_history_predictions = np.asarray(full_history_predictions, dtype=float)

    if post_only_predictions.shape != full_history_predictions.shape:
        raise ValueError(
            "post_only_predictions and full_history_predictions must have the same shape."
        )

    full_history_weight = 1 - post_only_weight

    return (
        post_only_weight * post_only_predictions
        + full_history_weight * full_history_predictions
    )


def get_candidate_ensemble_weights() -> list[float]:
    """
    Return candidate post-only weights for validation-based ensemble selection.

    The range is intentionally restricted to true post-only-dominant ensembles.

    A weight of 0.9 means:
        90% post-only model
        10% full-history model

    A weight of 0.6 means:
        60% post-only model
        40% full-history model
    """
    return [0.6 , 0.7]