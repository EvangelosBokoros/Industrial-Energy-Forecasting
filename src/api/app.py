from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
)
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from src.api.metrics import ApiMetrics
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


REQUEST_ID_HEADER = "X-Request-ID"

REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._-]{1,128}$"
)

logger = logging.getLogger("jmm.api")


def _resolve_request_id(
    header_value: str | None,
) -> str:
    """
    Reuse a safe caller-provided request ID or generate a new UUID.

    Restricting the allowed characters prevents malformed values
    from entering response headers and structured logs.
    """

    if header_value is not None:
        candidate = header_value.strip()

        if REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate

    return str(uuid4())


def _resolve_metrics_path(request: Request) -> str:
    """Return a bounded route label for HTTP metrics."""

    route = request.scope.get("route")
    route_path = getattr(route, "path", None)

    if isinstance(route_path, str):
        return route_path

    return "__unmatched__"


def _serialize_request_log(
    *,
    event: str,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    error_type: str | None = None,
) -> str:
    """Create one machine-readable JSON log record."""

    record: dict[str, object] = {
        "event": event,
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
    }

    if error_type is not None:
        record["error_type"] = error_type

    return json.dumps(
        record,
        sort_keys=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load and validate the model and serving references once.

    The same loaded prediction service is reused for every request.
    """

    app.state.prediction_service = PredictionService()
    app.state.metrics = ApiMetrics()

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


@app.middleware("http")
async def request_observability(
    request: Request,
    call_next: Callable[
        [Request],
        Awaitable[Response],
    ],
) -> Response:
    """
    Add request correlation, timing, metrics, logs, and safe errors.
    """

    request_id = _resolve_request_id(
        request.headers.get(REQUEST_ID_HEADER)
    )

    request.state.request_id = request_id

    started_at = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception as error:
        duration_seconds = (
            time.perf_counter()
            - started_at
        )
        duration_ms = round(
            duration_seconds * 1000.0,
            3,
        )

        metrics: ApiMetrics = request.app.state.metrics
        metrics.observe_http_request(
            method=request.method,
            path=_resolve_metrics_path(request),
            status_code=500,
            duration_seconds=duration_seconds,
        )

        logger.exception(
            _serialize_request_log(
                event="request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=duration_ms,
                error_type=type(error).__name__,
            )
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
            headers={
                REQUEST_ID_HEADER: request_id,
            },
        )

    duration_seconds = (
        time.perf_counter()
        - started_at
    )
    duration_ms = round(
        duration_seconds * 1000.0,
        3,
    )

    response.headers[
        REQUEST_ID_HEADER
    ] = request_id

    metrics = request.app.state.metrics
    metrics.observe_http_request(
        method=request.method,
        path=_resolve_metrics_path(request),
        status_code=response.status_code,
        duration_seconds=duration_seconds,
    )

    logger.info(
        _serialize_request_log(
            event="request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
    )

    return response


def _build_prediction_response(
    payload: PredictionRequest,
    result: PredictionResult,
    metadata: dict[str, object],
) -> PredictionResponse:
    """Build the public API response for one prediction."""

    return PredictionResponse(
        date=payload.date,
        modeling_version=str(
            metadata["modeling_version"]
        ),
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
        operational_range_status=(
            result.operational_range_status
        ),
        calendar_coverage_status=(
            result.calendar_coverage_status
        ),
        tail_features=list(
            result.tail_features
        ),
        outside_range_features=list(
            result.outside_range_features
        ),
        unseen_calendar_features=list(
            result.unseen_calendar_features
        ),
        warning_codes=list(
            result.warning_codes
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
        modeling_version=str(
            metadata["modeling_version"]
        ),
        target=str(metadata["target_column"]),
    )


@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics(request: Request) -> Response:
    """Expose Prometheus-compatible service metrics."""

    api_metrics: ApiMetrics = request.app.state.metrics

    return Response(
        content=generate_latest(
            api_metrics.registry
        ),
        media_type=CONTENT_TYPE_LATEST,
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
            estimator_class=str(
                component["estimator_class"]
            ),
            weight=round(
                float(component["weight"]),
                12,
            ),
            features=list(component["features"]),
        )
        for component_name, component in metadata[
            "components"
        ].items()
    }

    return ModelInfoResponse(
        modeling_version=str(
            metadata["modeling_version"]
        ),
        ensemble_type=str(
            metadata["ensemble_type"]
        ),
        target=str(metadata["target_column"]),
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

    response = _build_prediction_response(
        payload=payload,
        result=result,
        metadata=service.model_bundle.metadata,
    )

    metrics: ApiMetrics = request.app.state.metrics
    metrics.observe_prediction(
        result=result,
        endpoint="single",
    )

    return response


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

    prediction_results: list[PredictionResult] = []
    predictions: list[PredictionResponse] = []

    for record in payload.records:
        result = service.predict(
            record.model_dump(mode="python")
        )

        prediction_results.append(result)

        predictions.append(
            _build_prediction_response(
                payload=record,
                result=result,
                metadata=metadata,
            )
        )

    metrics: ApiMetrics = request.app.state.metrics

    for result in prediction_results:
        metrics.observe_prediction(
            result=result,
            endpoint="batch",
        )

    return BatchPredictionResponse(
        count=len(predictions),
        predictions=predictions,
    )
