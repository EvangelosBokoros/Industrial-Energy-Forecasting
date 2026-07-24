from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    """Forecast and model-disagreement information returned by the API."""

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