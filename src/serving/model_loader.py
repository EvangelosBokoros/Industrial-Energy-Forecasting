from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METADATA_PATH = (
    PROJECT_ROOT / "config" / "serving" / "model_metadata.json"
)


@dataclass(frozen=True)
class ModelBundle:
    """Validated serving metadata and trained model artifact."""

    metadata: dict[str, Any]
    artifact: dict[str, Any]


def _resolve_project_path(path_value: str) -> Path:
    """Resolve a metadata path relative to the repository root."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_artifact_checksum(
    *,
    metadata: dict[str, Any],
    artifact_path: Path,
) -> None:
    """Verify the model artifact before deserializing it."""

    expected_checksum = metadata.get("artifact_sha256")

    if not isinstance(expected_checksum, str):
        raise ValueError(
            "Model metadata must contain artifact_sha256"
        )

    expected_checksum = expected_checksum.strip().lower()

    if (
        len(expected_checksum) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_checksum
        )
    ):
        raise ValueError(
            "artifact_sha256 must be a valid SHA-256 checksum"
        )

    actual_checksum = _calculate_sha256(artifact_path)

    if not hmac.compare_digest(
        actual_checksum,
        expected_checksum,
    ):
        raise ValueError(
            "Model artifact checksum mismatch: "
            f"expected={expected_checksum}, "
            f"actual={actual_checksum}"
        )


def _validate_component(
    *,
    component_name: str,
    component_metadata: dict[str, Any],
    artifact: dict[str, Any],
    model_key: str,
    weight_key: str,
    features_key: str,
) -> None:
    """Validate one ensemble branch against the saved artifact."""

    model = artifact[model_key]

    expected_class = component_metadata["estimator_class"]
    actual_class = type(model).__name__

    if actual_class != expected_class:
        raise ValueError(
            f"{component_name} estimator mismatch: "
            f"metadata={expected_class}, artifact={actual_class}"
        )

    expected_weight = float(component_metadata["weight"])
    actual_weight = float(artifact[weight_key])

    if not math.isclose(
        actual_weight,
        expected_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{component_name} weight mismatch: "
            f"metadata={expected_weight}, artifact={actual_weight}"
        )

    expected_features = list(component_metadata["features"])
    actual_features = list(artifact[features_key])

    if actual_features != expected_features:
        raise ValueError(
            f"{component_name} feature order does not match the artifact"
        )


def validate_model_bundle(
    metadata: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    """Validate serving metadata against the trained artifact."""

    required_artifact_keys = {
        "modeling_version",
        "ensemble_type",
        "target_column",
        "post_only_model",
        "full_history_model",
        "post_only_weight",
        "full_history_weight",
        "post_only_features",
        "full_history_features",
    }

    missing_keys = required_artifact_keys.difference(artifact)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Artifact is missing required keys: {missing}")

    if artifact["modeling_version"] != metadata["modeling_version"]:
        raise ValueError("Modeling version does not match metadata")

    if artifact["ensemble_type"] != metadata["ensemble_type"]:
        raise ValueError("Ensemble type does not match metadata")

    if artifact["target_column"] != metadata["target_column"]:
        raise ValueError("Target column does not match metadata")

    components = metadata["components"]

    _validate_component(
        component_name="post_only",
        component_metadata=components["post_only"],
        artifact=artifact,
        model_key="post_only_model",
        weight_key="post_only_weight",
        features_key="post_only_features",
    )

    _validate_component(
        component_name="full_history",
        component_metadata=components["full_history"],
        artifact=artifact,
        model_key="full_history_model",
        weight_key="full_history_weight",
        features_key="full_history_features",
    )

    weight_total = (
        float(artifact["post_only_weight"])
        + float(artifact["full_history_weight"])
    )

    if not math.isclose(weight_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Ensemble weights must sum to 1.0, received {weight_total}"
        )


def load_model_bundle(
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> ModelBundle:
    """Load and validate the complete serving model bundle."""

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata file was not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    artifact_path = _resolve_project_path(metadata["artifact_path"])

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact was not found: {artifact_path}"
        )

    _validate_artifact_checksum(
        metadata=metadata,
        artifact_path=artifact_path,
    )

    artifact = joblib.load(artifact_path)

    if not isinstance(artifact, dict):
        raise TypeError(
            "Expected the model artifact to be a dictionary, "
            f"received {type(artifact).__name__}"
        )

    validate_model_bundle(metadata, artifact)

    return ModelBundle(
        metadata=metadata,
        artifact=artifact,
    )


if __name__ == "__main__":
    bundle = load_model_bundle()

    print("Model bundle loaded and validated successfully")
    print("Modeling version:", bundle.metadata["modeling_version"])
    print("Target:", bundle.metadata["target_column"])
    print(
        "Post-only estimator:",
        type(bundle.artifact["post_only_model"]).__name__,
    )
    print(
        "Full-history estimator:",
        type(bundle.artifact["full_history_model"]).__name__,
    )