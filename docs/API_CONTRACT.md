# Damavand Energy Forecasting API Contract

## Contract identification

**API route version:** `v1`
**Application version:** `1.0.0`
**Schema version:** `1.0`  
**Forecasting model:** `2.0_ensemble_70_30`  
**Behavioral evaluation:** `2.1`  
**Target:** `active_energy_kWh`

## Purpose

The API provides daily active-energy forecasts for the Damavand industrial process.

The service uses the official weighted ensemble:

```text
70% post-only Extra Trees
30% full-history AdaBoost
```

Every prediction also includes component-model disagreement, operational development-support, calendar-coverage, and warning-code diagnostics.

Diagnostics do not block prediction generation. They describe how closely the request resembles the model-development data and whether the two model branches disagree.

## Base URL

The application does not impose a deployment hostname.

For local development:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

## Content type

Prediction requests and responses use JSON:

```text
Content-Type: application/json
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service information and endpoint navigation |
| `GET` | `/health` | Confirm that the web application is running |
| `GET` | `/ready` | Confirm that the model and serving reference loaded successfully |
| `GET` | `/v1/model` | Return public metadata for the active ensemble |
| `POST` | `/v1/predict` | Generate one daily prediction |
| `POST` | `/v1/predict/batch` | Generate between 1 and 500 ordered predictions |

## Request identification

Every HTTP response contains:

```text
X-Request-ID
```

A caller may supply:

```text
X-Request-ID: portfolio-test-123
```

A caller-provided request ID is preserved when it contains 1–128 characters and only letters, numbers, `.`, `_`, or `-`.

Accepted pattern:

```text
^[A-Za-z0-9._-]{1,128}$
```

When the value is absent or invalid, the API generates a UUID. An invalid request ID does not reject the request.

---

## `GET /`

Returns basic application information.

### Successful response

**Status:** `200 OK`

```json
{
  "service": "Damavand Energy Forecasting API",
  "api_version": "1.0.0",
  "documentation_url": "/docs",
  "health_url": "/health",
  "readiness_url": "/ready",
  "model_url": "/v1/model"
}
```

---

## `GET /health`

Confirms that the FastAPI application is running.

This endpoint does not by itself confirm that the forecasting artifact loaded successfully. Use `/ready` for model readiness.

### Successful response

**Status:** `200 OK`

```json
{
  "status": "ok"
}
```

---

## `GET /ready`

Confirms that the validated model artifact and serving reference loaded during application startup.

### Successful response

**Status:** `200 OK`

```json
{
  "status": "ready",
  "modeling_version": "2.0",
  "target": "active_energy_kWh"
}
```

If model or serving-reference initialization fails, application startup fails rather than reporting a false ready state.

---

## `GET /v1/model`

Returns public information about the active ensemble.

### Successful response

**Status:** `200 OK`

```json
{
  "modeling_version": "2.0",
  "ensemble_type": "constrained_weighted_prediction_average",
  "target": "active_energy_kWh",
  "components": {
    "post_only": {
      "estimator_class": "ExtraTreesRegressor",
      "weight": 0.7,
      "features": [
        "total_kg",
        "total_nominal_kg",
        "total_brix_units",
        "total_hours",
        "total_pallets",
        "orders",
        "avg_brix",
        "yield_ratio_actual_over_nominal",
        "weekday",
        "is_weekend",
        "month",
        "year",
        "week_of_year"
      ]
    },
    "full_history": {
      "estimator_class": "AdaBoostRegressor",
      "weight": 0.3,
      "features": [
        "total_kg",
        "total_hours",
        "total_pallets",
        "orders",
        "avg_brix",
        "yield_ratio_actual_over_nominal",
        "weekday",
        "is_weekend",
        "month",
        "year"
      ]
    }
  }
}
```

---

## `POST /v1/predict`

Generates one daily active-energy forecast.

### Request body

```json
{
  "date": "2025-10-10",
  "total_kg": 190741.0,
  "total_nominal_kg": 190631.01,
  "total_brix_units": 1931393.0,
  "total_hours": 58.93,
  "total_pallets": 0.0,
  "orders": 4,
  "avg_brix": 10.1411869
}
```

### Request fields

| Field | Type | Validation | Meaning |
|---|---|---|---|
| `date` | ISO date | Required | Operational date in `YYYY-MM-DD` format |
| `total_kg` | Number | `>= 0` | Actual total production mass |
| `total_nominal_kg` | Number | `> 0` | Nominal total production mass |
| `total_brix_units` | Number | `>= 0` | Total Brix units |
| `total_hours` | Number | `>= 0` | Total operating hours |
| `total_pallets` | Number | `>= 0` | Total pallet count or quantity |
| `orders` | Integer | `>= 0` | Number of orders |
| `avg_brix` | Number | `>= 0` | Average Brix value |

All fields are required.

Unknown fields, NaN values, and infinite values are rejected.

### Derived features

The caller does not provide derived features.

The serving feature builder calculates:

```text
yield_ratio_actual_over_nominal
weekday
is_weekend
month
year
week_of_year
```

Yield ratio:

```text
total_kg / total_nominal_kg
```

Calendar values are derived from `date`.

The same engineered values are reused for the post-only model, the full-history model, and the development-support checker.

### Successful response

**Status:** `200 OK`

The numerical values below are illustrative.

```json
{
  "date": "2025-10-10",
  "modeling_version": "2.0",
  "target": "active_energy_kWh",
  "prediction_kwh": 123456.78,
  "post_only_prediction_kwh": 121000.0,
  "full_history_prediction_kwh": 129189.27,
  "branch_disagreement_kwh": 8189.27,
  "branch_disagreement_pct": 6.55,
  "branch_disagreement_status": "low",
  "operational_range_status": "inside_typical_development_range",
  "calendar_coverage_status": "represented",
  "tail_features": [],
  "outside_range_features": [],
  "unseen_calendar_features": [],
  "warning_codes": []
}
```

### Response fields

| Field | Type | Meaning |
|---|---|---|
| `date` | Date | Date supplied in the request |
| `modeling_version` | String | Active model version |
| `target` | String | Forecast target |
| `prediction_kwh` | Number | Official weighted ensemble prediction |
| `post_only_prediction_kwh` | Number | Post-only Extra Trees prediction |
| `full_history_prediction_kwh` | Number | Full-history AdaBoost prediction |
| `branch_disagreement_kwh` | Number | Absolute difference between component predictions |
| `branch_disagreement_pct` | Number or `null` | Difference relative to the mean absolute component prediction |
| `branch_disagreement_status` | String | `low`, `moderate`, `high`, or `undefined` |
| `operational_range_status` | String | Overall operational support classification |
| `calendar_coverage_status` | String | Whether calendar values appeared during development |
| `tail_features` | Array of strings | Operational features inside min–max but outside p10–p90 |
| `outside_range_features` | Array of strings | Operational features outside their observed min–max |
| `unseen_calendar_features` | Array of strings | Calendar feature/value pairs absent during development |
| `warning_codes` | Array of strings | Machine-readable diagnostic warnings |

## Ensemble calculation

```text
prediction_kwh =
    0.70 × post_only_prediction_kwh
    +
    0.30 × full_history_prediction_kwh
```

The component weights are loaded from the validated model artifact.

## Branch disagreement

Absolute disagreement:

```text
branch_disagreement_kwh =
    absolute(
        post_only_prediction_kwh
        -
        full_history_prediction_kwh
    )
```

Percentage disagreement:

```text
component_mean =
    (
        absolute(post_only_prediction_kwh)
        +
        absolute(full_history_prediction_kwh)
    )
    / 2
```

```text
branch_disagreement_pct =
    branch_disagreement_kwh
    / component_mean
    × 100
```

When `component_mean` is zero, the percentage is `null` and the status is `undefined`.

### Classification thresholds

| Percentage | Status |
|---:|---|
| Less than `10%` | `low` |
| At least `10%` but less than `20%` | `moderate` |
| At least `20%` | `high` |
| Cannot be calculated | `undefined` |

Branch disagreement is not a confidence interval, error probability, or guarantee of forecast accuracy.

## Operational development support

The API compares engineered operational values with:

```text
config/serving/input_reference.json
```

For each operational feature, the reference contains:

```text
minimum
p10
p90
maximum
```

### Per-feature classification

```text
below minimum or above maximum
→ outside_observed_range
```

```text
between minimum and p10
or
between p90 and maximum
→ development_distribution_tail
```

```text
between p10 and p90
→ inside_typical_development_range
```

Boundary comparisons use a small floating-point tolerance.

For a feature that was constant during development, the observed value is accepted and any different value is outside the observed range.

### Overall operational status

Possible values:

```text
inside_typical_development_range
development_distribution_tail
outside_observed_range
```

Priority:

```text
outside_observed_range
    over
development_distribution_tail
    over
inside_typical_development_range
```

## Calendar coverage

The API checks:

```text
weekday
is_weekend
month
year
week_of_year
```

Possible values:

```text
represented
contains_unseen_calendar_values
```

An unseen calendar value is a support diagnostic. It does not mean the date is invalid.

## Warning codes

| Code | Meaning |
|---|---|
| `DEVELOPMENT_DISTRIBUTION_TAIL` | At least one operational feature is inside min–max but outside p10–p90 |
| `OUTSIDE_OBSERVED_RANGE` | At least one operational feature is outside its observed range |
| `UNSEEN_CALENDAR_VALUE` | At least one derived calendar value was absent during development |
| `MODERATE_BRANCH_DISAGREEMENT` | Component disagreement is at least 10% but below 20% |
| `HIGH_BRANCH_DISAGREEMENT` | Component disagreement is at least 20% |

Multiple codes may be returned. An empty array means none of the defined warning conditions were present.

Warnings do not block prediction generation.

## Outside-range example

### Request

```json
{
  "date": "2026-01-05",
  "total_kg": 190741.0,
  "total_nominal_kg": 190631.01,
  "total_brix_units": 1931393.0,
  "total_hours": 58.93,
  "total_pallets": 1.0,
  "orders": 4,
  "avg_brix": 10.1411869
}
```

In the current serving reference, `total_pallets` was constant at zero.

Illustrative diagnostic section:

```json
{
  "operational_range_status": "outside_observed_range",
  "calendar_coverage_status": "contains_unseen_calendar_values",
  "tail_features": [],
  "outside_range_features": [
    "total_pallets"
  ],
  "unseen_calendar_features": [
    "month=1",
    "week_of_year=2",
    "year=2026"
  ],
  "warning_codes": [
    "OUTSIDE_OBSERVED_RANGE",
    "UNSEEN_CALENDAR_VALUE"
  ]
}
```

Additional disagreement warning codes may also be present.

---

## `POST /v1/predict/batch`

Generates predictions for an ordered collection of records.

### Batch size

```text
1 to 500 records
```

### Request body

```json
{
  "records": [
    {
      "date": "2025-10-10",
      "total_kg": 190741.0,
      "total_nominal_kg": 190631.01,
      "total_brix_units": 1931393.0,
      "total_hours": 58.93,
      "total_pallets": 0.0,
      "orders": 4,
      "avg_brix": 10.1411869
    },
    {
      "date": "2025-10-11",
      "total_kg": 180000.0,
      "total_nominal_kg": 181000.0,
      "total_brix_units": 1800000.0,
      "total_hours": 54.0,
      "total_pallets": 0.0,
      "orders": 3,
      "avg_brix": 10.0
    }
  ]
}
```

### Successful response

**Status:** `200 OK`

Each item in `predictions` has the same schema as the single-prediction response.

```json
{
  "count": 2,
  "predictions": [
    {
      "date": "2025-10-10",
      "modeling_version": "2.0",
      "target": "active_energy_kWh",
      "prediction_kwh": 123456.78,
      "post_only_prediction_kwh": 121000.0,
      "full_history_prediction_kwh": 129189.27,
      "branch_disagreement_kwh": 8189.27,
      "branch_disagreement_pct": 6.55,
      "branch_disagreement_status": "low",
      "operational_range_status": "inside_typical_development_range",
      "calendar_coverage_status": "represented",
      "tail_features": [],
      "outside_range_features": [],
      "unseen_calendar_features": [],
      "warning_codes": []
    }
  ]
}
```

Prediction values are illustrative.

### Ordering guarantee

```text
records[0] → predictions[0]
records[1] → predictions[1]
```

The complete batch is schema-validated before prediction begins.

When any nested record is invalid, the whole request returns a validation error. Schema-invalid batches do not return partial results.

Unexpected internal errors return a safe `500` response rather than a partial batch.

## Validation errors

Invalid request bodies return:

```text
422 Unprocessable Entity
```

Examples include missing fields, unknown fields, invalid dates, negative values, `total_nominal_kg <= 0`, non-integer `orders`, NaN or infinity, empty batches, more than 500 records, and invalid nested records.

The response follows FastAPI and Pydantic's validation-error structure.

Example:

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": [
        "body",
        "total_kg"
      ],
      "msg": "Input should be greater than or equal to 0",
      "input": -1,
      "ctx": {
        "ge": 0
      }
    }
  ]
}
```

Exact wording may vary with the installed Pydantic version.

Validation-error responses also include `X-Request-ID`.

## Internal errors

Unexpected exceptions return:

```text
500 Internal Server Error
```

Response:

```json
{
  "detail": "Internal server error",
  "request_id": "failure-test-001"
}
```

The same request ID is returned in the `X-Request-ID` header.

Internal exception details are not returned to the caller. The real exception and exception type are written to the server log.

## Structured request logging

Completed request:

```json
{
  "event": "request_completed",
  "request_id": "portfolio-test-123",
  "method": "POST",
  "path": "/v1/predict",
  "status_code": 200,
  "duration_ms": 18.42
}
```

Unexpected failure:

```json
{
  "event": "request_failed",
  "request_id": "failure-test-001",
  "method": "POST",
  "path": "/v1/predict",
  "status_code": 500,
  "duration_ms": 12.7,
  "error_type": "RuntimeError"
}
```

The middleware does not log request payloads or production feature values.

## Operational guarantees

The current implementation guarantees that:

- the model artifact is loaded once during startup;
- the serving reference is loaded once during startup;
- the loaded prediction service is reused across requests;
- input fields are validated before prediction;
- unknown request fields are rejected;
- model feature order comes from the validated artifact;
- ensemble weights come from the validated artifact;
- one feature-building path serves both model branches;
- the same engineered values are reused for support diagnostics;
- batch output preserves request order;
- every response includes `X-Request-ID`;
- unexpected internal errors are hidden from callers;
- prediction request payloads are not logged;
- outside-range diagnostics do not silently block prediction.

## Interpretation and limitations

The API returns a point forecast.

The response is not a causal estimate, guaranteed future energy value, confidence interval, prediction interval, error probability, or automatic operational decision.

Development-support diagnostics describe similarity to the saved development reference. They do not directly measure forecast error.

An input inside the typical range can still have prediction error.

An input outside the observed range still receives a numerical prediction, but that prediction should be reviewed with additional care.

Branch disagreement measures divergence between the two ensemble components. It does not prove that either component is correct.

The current application does not define authentication or authorization. Those controls must be added at the deployment or infrastructure layer when required.

## Contract acceptance criteria

The contract is accepted when:

1. `/health` returns `200` and `{"status": "ok"}`.
2. `/ready` confirms model version `2.0`.
3. `/v1/model` reports the two components and correct weights.
4. `/v1/predict` returns the weighted ensemble result.
5. `/v1/predict` returns support and disagreement diagnostics.
6. `/v1/predict/batch` accepts 1–500 records.
7. Batch output preserves input order.
8. Invalid input returns `422`.
9. Every response includes `X-Request-ID`.
10. Unexpected internal errors return a safe `500`.
11. The automated test suite passes.
12. Representative requests work through Swagger UI.
