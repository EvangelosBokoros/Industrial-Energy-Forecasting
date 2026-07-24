from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


OperationalRangeStatus = Literal[
    "inside_typical_development_range",
    "development_distribution_tail",
    "outside_observed_range",
]

CalendarCoverageStatus = Literal[
    "represented",
    "contains_unseen_calendar_values",
]

WarningCode = Literal[
    "DEVELOPMENT_DISTRIBUTION_TAIL",
    "OUTSIDE_OBSERVED_RANGE",
    "UNSEEN_CALENDAR_VALUE",
    "MODERATE_BRANCH_DISAGREEMENT",
    "HIGH_BRANCH_DISAGREEMENT",
]


class PredictionRequest(BaseModel):
    """Raw operational inputs accepted by the prediction API."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    date: date

    total_kg: float = Field(ge=0)
    total_nominal_kg: float = Field(gt=0)
    total_brix_units: float = Field(ge=0)
    total_hours: float = Field(ge=0)
    total_pallets: float = Field(ge=0)
    orders: int = Field(ge=0)
    avg_brix: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """
    Forecast, model disagreement, and development-support diagnostics.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    date: date
    modeling_version: str
    target: str

    prediction_kwh: float
    post_only_prediction_kwh: float
    full_history_prediction_kwh: float

    branch_disagreement_kwh: float
    branch_disagreement_pct: float | None
    branch_disagreement_status: Literal[
        "low",
        "moderate",
        "high",
        "undefined",
    ]

    operational_range_status: OperationalRangeStatus
    calendar_coverage_status: CalendarCoverageStatus

    tail_features: list[str]
    outside_range_features: list[str]
    unseen_calendar_features: list[str]

    warning_codes: list[WarningCode]


class RootResponse(BaseModel):
    """Basic information and navigation for the API."""

    model_config = ConfigDict(extra="forbid")

    service: str
    api_version: str
    documentation_url: str
    health_url: str
    readiness_url: str
    model_url: str


class ReadinessResponse(BaseModel):
    """Confirmation that the validated model is ready for predictions."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]
    modeling_version: str
    target: str


class ModelComponentResponse(BaseModel):
    """Public serving details for one ensemble component."""

    model_config = ConfigDict(extra="forbid")

    estimator_class: str
    weight: float
    features: list[str]


class ModelInfoResponse(BaseModel):
    """Public metadata describing the active forecasting model."""

    model_config = ConfigDict(extra="forbid")

    modeling_version: str
    ensemble_type: str
    target: str
    components: dict[str, ModelComponentResponse]


class BatchPredictionRequest(BaseModel):
    """Collection of daily records submitted for batch prediction."""

    model_config = ConfigDict(extra="forbid")

    records: list[PredictionRequest] = Field(
        min_length=1,
        max_length=500,
    )


class BatchPredictionResponse(BaseModel):
    """Ordered predictions returned for a batch request."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1, le=500)
    predictions: list[PredictionResponse]