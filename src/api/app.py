from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelComponentResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
    RootResponse,
)
from src.serving.prediction_service import (
    PredictionResult,
    PredictionService,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load and validate the model once when the API starts.

    The same loaded model is reused for every prediction request.
    """
    app.state.prediction_service = PredictionService()

    yield


app = FastAPI(
    title="Damavand Energy Forecasting API",
    description=(
        "Daily active-energy forecasting using the official "
        "70% post-only Extra Trees and 30% full-history "
        "AdaBoost ensemble."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _build_prediction_response(
    payload: PredictionRequest,
    result: PredictionResult,
    metadata: dict[str, object],
) -> PredictionResponse:
    """Build the public API response for one prediction."""

    return PredictionResponse(
        date=payload.date,
        modeling_version=str(metadata["modeling_version"]),
        target=str(metadata["target_column"]),
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


@app.get(
    "/",
    response_model=RootResponse,
)
def root(request: Request) -> RootResponse:
    """Return basic API information and endpoint navigation."""

    return RootResponse(
        service=request.app.title,
        api_version=request.app.version,
        documentation_url="/docs",
        health_url="/health",
        readiness_url="/ready",
        model_url="/v1/model",
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm that the web application is running."""

    return {"status": "ok"}


@app.get(
    "/ready",
    response_model=ReadinessResponse,
)
def readiness(request: Request) -> ReadinessResponse:
    """Confirm that the validated model is loaded and ready."""

    service: PredictionService = (
        request.app.state.prediction_service
    )
    metadata = service.model_bundle.metadata

    return ReadinessResponse(
        status="ready",
        modeling_version=metadata["modeling_version"],
        target=metadata["target_column"],
    )


@app.get(
    "/v1/model",
    response_model=ModelInfoResponse,
)
def model_information(
    request: Request,
) -> ModelInfoResponse:
    """Return public metadata for the active forecasting model."""

    service: PredictionService = (
        request.app.state.prediction_service
    )
    metadata = service.model_bundle.metadata

    components = {
        component_name: ModelComponentResponse(
            estimator_class=component["estimator_class"],
            weight=round(float(component["weight"]), 12),
            features=list(component["features"]),
        )
        for component_name, component in metadata[
            "components"
        ].items()
    }

    return ModelInfoResponse(
        modeling_version=metadata["modeling_version"],
        ensemble_type=metadata["ensemble_type"],
        target=metadata["target_column"],
        components=components,
    )


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Generate one daily active-energy forecast."""

    service: PredictionService = (
        request.app.state.prediction_service
    )

    result = service.predict(
        payload.model_dump(mode="python")
    )

    return _build_prediction_response(
        payload=payload,
        result=result,
        metadata=service.model_bundle.metadata,
    )


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """
    Generate ordered forecasts for between 1 and 500 daily records.
    """

    service: PredictionService = (
        request.app.state.prediction_service
    )
    metadata = service.model_bundle.metadata

    predictions: list[PredictionResponse] = []

    for record in payload.records:
        result = service.predict(
            record.model_dump(mode="python")
        )

        predictions.append(
            _build_prediction_response(
                payload=record,
                result=result,
                metadata=metadata,
            )
        )

    return BatchPredictionResponse(
        count=len(predictions),
        predictions=predictions,
    )