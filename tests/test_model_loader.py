from copy import deepcopy

import pytest

from src.serving.model_loader import (
    load_model_bundle,
    validate_model_bundle,
)


def test_real_model_bundle_loads_successfully() -> None:
    bundle = load_model_bundle()

    assert bundle.metadata["modeling_version"] == "2.0"
    assert bundle.metadata["target_column"] == "active_energy_kWh"

    assert (
        type(bundle.artifact["post_only_model"]).__name__
        == "ExtraTreesRegressor"
    )
    assert (
        type(bundle.artifact["full_history_model"]).__name__
        == "AdaBoostRegressor"
    )


def test_target_mismatch_is_rejected() -> None:
    bundle = load_model_bundle()
    incorrect_metadata = deepcopy(bundle.metadata)

    incorrect_metadata["target_column"] = "incorrect_target"

    with pytest.raises(
        ValueError,
        match="Target column does not match metadata",
    ):
        validate_model_bundle(
            incorrect_metadata,
            bundle.artifact,
        )


def test_feature_order_mismatch_is_rejected() -> None:
    bundle = load_model_bundle()
    incorrect_metadata = deepcopy(bundle.metadata)

    features = incorrect_metadata["components"]["post_only"]["features"]
    incorrect_metadata["components"]["post_only"]["features"] = list(
        reversed(features)
    )

    with pytest.raises(
        ValueError,
        match="feature order does not match",
    ):
        validate_model_bundle(
            incorrect_metadata,
            bundle.artifact,
        )