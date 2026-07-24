from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.serving.feature_builder import (
    build_model_feature_frames,
)
from src.serving.input_support import InputSupportChecker
from src.serving.model_loader import (
    ModelBundle,
    load_model_bundle,
)


MODERATE_DISAGREEMENT_THRESHOLD_PCT = 10.0
HIGH_DISAGREEMENT_THRESHOLD_PCT = 20.0


@dataclass(frozen=True)
class PredictionResult:
    """
    Prediction and diagnostic results for one operational record.
    """

    prediction_kwh: float
    post_only_prediction_kwh: float
    full_history_prediction_kwh: float

    branch_disagreement_kwh: float
    branch_disagreement_pct: float | None
    branch_disagreement_status: str

    operational_range_status: str
    calendar_coverage_status: str

    tail_features: tuple[str, ...]
    outside_range_features: tuple[str, ...]
    unseen_calendar_features: tuple[str, ...]

    warning_codes: tuple[str, ...]


def classify_disagreement(
    disagreement_pct: float | None,
) -> str:
    """Classify component-model disagreement."""

    if (
        disagreement_pct is None
        or not math.isfinite(disagreement_pct)
    ):
        return "undefined"

    if disagreement_pct >= HIGH_DISAGREEMENT_THRESHOLD_PCT:
        return "high"

    if disagreement_pct >= MODERATE_DISAGREEMENT_THRESHOLD_PCT:
        return "moderate"

    return "low"


class PredictionService:
    """
    Run the validated two-branch forecasting ensemble.

    The model bundle and development-support reference are loaded
    once when the service is created.
    """

    def __init__(
        self,
        model_bundle: ModelBundle | None = None,
        input_support_checker: InputSupportChecker | None = None,
    ) -> None:
        self.model_bundle = (
            model_bundle or load_model_bundle()
        )

        self.input_support_checker = (
            input_support_checker
            or InputSupportChecker()
        )

    def predict(
        self,
        raw_input: Mapping[str, Any],
    ) -> PredictionResult:
        """Generate one ensemble prediction and its diagnostics."""

        artifact = self.model_bundle.artifact

        feature_frames = build_model_feature_frames(
            raw_input=raw_input,
            post_only_features=artifact[
                "post_only_features"
            ],
            full_history_features=artifact[
                "full_history_features"
            ],
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

        post_only_weight = float(
            artifact["post_only_weight"]
        )
        full_history_weight = float(
            artifact["full_history_weight"]
        )

        prediction = (
            post_only_weight * post_only_prediction
            + full_history_weight
            * full_history_prediction
        )

        disagreement_kwh = abs(
            post_only_prediction
            - full_history_prediction
        )

        component_mean = (
            abs(post_only_prediction)
            + abs(full_history_prediction)
        ) / 2.0

        disagreement_pct = (
            disagreement_kwh
            / component_mean
            * 100.0
            if component_mean > 0
            else None
        )

        disagreement_status = classify_disagreement(
            disagreement_pct
        )

        input_support = (
            self.input_support_checker.evaluate(
                feature_values=(
                    feature_frames.feature_values
                ),
                branch_disagreement_status=(
                    disagreement_status
                ),
            )
        )

        return PredictionResult(
            prediction_kwh=prediction,
            post_only_prediction_kwh=(
                post_only_prediction
            ),
            full_history_prediction_kwh=(
                full_history_prediction
            ),
            branch_disagreement_kwh=(
                disagreement_kwh
            ),
            branch_disagreement_pct=(
                disagreement_pct
            ),
            branch_disagreement_status=(
                disagreement_status
            ),
            operational_range_status=(
                input_support.operational_range_status
            ),
            calendar_coverage_status=(
                input_support.calendar_coverage_status
            ),
            tail_features=input_support.tail_features,
            outside_range_features=(
                input_support.outside_range_features
            ),
            unseen_calendar_features=(
                input_support.unseen_calendar_features
            ),
            warning_codes=input_support.warning_codes,
        )