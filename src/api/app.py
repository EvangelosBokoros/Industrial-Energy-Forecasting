from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request

from src.api.schemas import PredictionRequest, PredictionResponse
from src.serving.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load and validate the model once when the API starts.

    The same loaded model is then reused for every prediction request.
    """
    app.state.prediction_service = PredictionService()

    yield


app = FastAPI(
    title="Damavand Energy Forecasting API",
    description=(
        "Daily active-energy forecasting using the official "
        "70% post-only Extra Trees and 30% full-history AdaBoost ensemble."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm that the web application is running."""

    return {"status": "ok"}


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Generate one daily active-energy forecast."""

    service: PredictionService = request.app.state.prediction_service

    result = service.predict(
        payload.model_dump(mode="python")
    )

    metadata = service.model_bundle.metadata

    return PredictionResponse(
        date=payload.date,
        modeling_version=metadata["modeling_version"],
        target=metadata["target_column"],
        prediction_kwh=result.prediction_kwh,
        post_only_prediction_kwh=(
            result.post_only_prediction_kwh
        ),
        full_history_prediction_kwh=(
            result.full_history_prediction_kwh
        ),
        branch_disagreement_kwh=(
            result.branch_disagreement_kwh
        ),
        branch_disagreement_pct=(
            result.branch_disagreement_pct
        ),
        branch_disagreement_status=(
            result.branch_disagreement_status
        ),
    )