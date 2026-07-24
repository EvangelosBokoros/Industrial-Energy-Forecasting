from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
)

if TYPE_CHECKING:
    from src.serving.prediction_service import PredictionResult


PredictionEndpoint = Literal[
    "single",
    "batch",
]


class ApiMetrics:
    """Prometheus metrics for API and model-serving behavior."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        self.http_requests_total = Counter(
            "jmm_http_requests_total",
            "Total HTTP requests processed by the API.",
            (
                "method",
                "path",
                "status_code",
            ),
            registry=self.registry,
        )

        self.http_request_duration_seconds = Histogram(
            "jmm_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            (
                "method",
                "path",
            ),
            buckets=(
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
            ),
            registry=self.registry,
        )

        self.prediction_records_total = Counter(
            "jmm_prediction_records_total",
            "Prediction records returned successfully.",
            ("endpoint",),
            registry=self.registry,
        )

        self.operational_range_total = Counter(
            "jmm_operational_range_total",
            "Predictions by operational input-support status.",
            ("status",),
            registry=self.registry,
        )

        self.calendar_coverage_total = Counter(
            "jmm_calendar_coverage_total",
            "Predictions by calendar-coverage status.",
            ("status",),
            registry=self.registry,
        )

        self.branch_disagreement_total = Counter(
            "jmm_branch_disagreement_total",
            "Predictions by branch-disagreement status.",
            ("status",),
            registry=self.registry,
        )

        self.warning_codes_total = Counter(
            "jmm_warning_codes_total",
            "Prediction warning codes returned by the API.",
            ("code",),
            registry=self.registry,
        )

    def observe_http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record one completed HTTP request."""

        self.http_requests_total.labels(
            method=method,
            path=path,
            status_code=str(status_code),
        ).inc()

        self.http_request_duration_seconds.labels(
            method=method,
            path=path,
        ).observe(duration_seconds)

    def observe_prediction(
        self,
        *,
        result: PredictionResult,
        endpoint: PredictionEndpoint,
    ) -> None:
        """Record one successfully returned prediction."""

        self.prediction_records_total.labels(
            endpoint=endpoint,
        ).inc()

        self.operational_range_total.labels(
            status=result.operational_range_status,
        ).inc()

        self.calendar_coverage_total.labels(
            status=result.calendar_coverage_status,
        ).inc()

        self.branch_disagreement_total.labels(
            status=result.branch_disagreement_status,
        ).inc()

        for warning_code in result.warning_codes:
            self.warning_codes_total.labels(
                code=warning_code,
            ).inc()
