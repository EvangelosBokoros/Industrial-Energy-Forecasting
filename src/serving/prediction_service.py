from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.serving.feature_builder import build_model_feature_frames
from src.serving.model_loader import ModelBundle, load_model_bundle


MODERATE_DISAGREEMENT_THRESHOLD_PCT = 10.0
HIGH_DISAGREEMENT_THRESHOLD_PCT = 20.0


@dataclass(frozen=True)
class PredictionResult:
    """Prediction and diagnostic results for one operational record."""

    prediction_kwh: float
    post_only_prediction_kwh: float
    full_history_prediction_kwh: float
    branch_disagreement_kwh: float
    branch_disagreement_pct: float | None
    branch_disagreement_status: str


def classify_disagreement(
    disagreement_pct: float | None,
) -> str:
    """Classify component-model disagreement."""

    if disagreement_pct is None or not math.isfinite(disagreement_pct):
        return "undefined"

    if disagreement_pct >= HIGH_DISAGREEMENT_THRESHOLD_PCT:
        return "high"

    if disagreement_pct >= MODERATE_DISAGREEMENT_THRESHOLD_PCT:
        return "moderate"

    return "low"


class PredictionService:
    """Run the validated two-branch forecasting ensemble."""

    def __init__(
        self,
        model_bundle: ModelBundle | None = None,
    ) -> None:
        self.model_bundle = model_bundle or load_model_bundle()

    def predict(
        self,
        raw_input: Mapping[str, Any],
    ) -> PredictionResult:
        """Generate one ensemble prediction."""

        artifact = self.model_bundle.artifact

        feature_frames = build_model_feature_frames(
            raw_input=raw_input,
            post_only_features=artifact["post_only_features"],
            full_history_features=artifact["full_history_features"],
        )

        post_only_prediction = float(
            artifact["post_only_model"].predict(
                feature_frames.post_only
            )[0]
        )

        full_history_prediction = float(
            artifact["full_history_model"].predict(
                feature_frames.full_history
            )[0]
        )

        post_only_weight = float(artifact["post_only_weight"])
        full_history_weight = float(
            artifact["full_history_weight"]
        )

        prediction = (
            post_only_weight * post_only_prediction
            + full_history_weight * full_history_prediction
        )

        disagreement_kwh = abs(
            post_only_prediction - full_history_prediction
        )

        component_mean = (
            abs(post_only_prediction)
            + abs(full_history_prediction)
        ) / 2.0

        disagreement_pct = (
            disagreement_kwh / component_mean * 100.0
            if component_mean > 0
            else None
        )

        return PredictionResult(
            prediction_kwh=prediction,
            post_only_prediction_kwh=post_only_prediction,
            full_history_prediction_kwh=full_history_prediction,
            branch_disagreement_kwh=disagreement_kwh,
            branch_disagreement_pct=disagreement_pct,
            branch_disagreement_status=classify_disagreement(
                disagreement_pct
            ),
        )