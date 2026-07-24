import json
from copy import deepcopy
from pathlib import Path

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
        bundle.metadata["artifact_sha256"]
        == "a4a7945ca5e77387bab3854597f380f8445b35849ef3b640b340507b26112282"
    )

    assert (
        type(bundle.artifact["post_only_model"]).__name__
        == "ExtraTreesRegressor"
    )
    assert (
        type(bundle.artifact["full_history_model"]).__name__
        == "AdaBoostRegressor"
    )


def test_artifact_checksum_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = load_model_bundle()
    incorrect_metadata = deepcopy(bundle.metadata)

    altered_artifact_path = tmp_path / "altered_model.joblib"
    altered_artifact_path.write_bytes(
        b"This is not the approved model artifact."
    )

    incorrect_metadata["artifact_path"] = str(
        altered_artifact_path
    )
    incorrect_metadata["artifact_sha256"] = "0" * 64

    temporary_metadata_path = (
        tmp_path / "model_metadata.json"
    )
    temporary_metadata_path.write_text(
        json.dumps(incorrect_metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Model artifact checksum mismatch",
    ):
        load_model_bundle(
            metadata_path=temporary_metadata_path,
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