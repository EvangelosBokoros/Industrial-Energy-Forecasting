# Independent Clean-Machine Acceptance Test

## Status

**Passed**

## Test date

2026-07-21

## Purpose

This test verifies that the committed Git repository contains everything
required to build and run the forecasting API on independent hardware.

The test was performed without copying the original project directory,
virtual environment, datasets, MLflow workspace, reports, caches, or
other ignored development files.

## Source revision

- Branch: `feature/release-hardening`
- Commit: `dc9b9e1`
- Source transfer method: complete Git bundle
- Bundle: `JMM-release-hardening.bundle`

The commit listed above is the exact repository revision used during the
independent test. The acceptance-test documentation was committed later.

## Independent environment

The repository was cloned on a separate Windows computer that had not
previously been used to develop this project.

Installed prerequisites:

- Git for Windows: `2.55.0.windows.3`
- WSL: `2.7.10.0`
- Default WSL version: `2`
- Docker client and server: `29.6.2`

Python and a project virtual environment were not installed on the host.
The forecasting service was built and executed entirely through Docker.

## Repository verification

The bundle was cloned into a new directory.

Verified state:

- Branch: `feature/release-hardening`
- Working tree: clean
- Latest tested commit: `dc9b9e1`
- Complete Git history available

The following required serving files were present:

- `models/ensemble_2_0_model.joblib`
- `config/serving/model_metadata.json`
- `config/serving/input_reference.json`
- `Dockerfile`
- `requirements-serving.txt`

## Model artifact integrity

The official model artifact produced the following SHA-256 checksum:

```text
A4A7945CA5E77387BAB3854597F380F8445B35849EF3B640B340507B26112282
```

This matched the approved checksum stored in the serving metadata.

## Docker build

The image was built from the clean clone using:

```powershell
docker build --no-cache --tag jmm-energy-api:clean-test .
```

Result:

- Build completed successfully
- Image: `jmm-energy-api:clean-test`
- Image ID observed during the test: `f20b5066a7f6`

The image ID is recorded as test evidence and is not intended to be a
permanent release identifier.

## Container startup and readiness

The container was started using:

```powershell
docker run --detach `
  --name jmm-energy-api-clean-test `
  --publish 8000:8000 `
  jmm-energy-api:clean-test
```

The container reached Docker health status:

```text
healthy
```

API checks returned:

```text
GET /health -> 200 OK
GET /ready  -> 200 OK
```

Readiness response:

```json
{
  "status": "ready",
  "modeling_version": "2.0",
  "target": "active_energy_kWh"
}
```

## Reference single prediction

The following request was submitted to:

```text
POST /v1/predict
```

Input:

```json
{
  "date": "2025-10-31",
  "total_kg": 190741,
  "total_nominal_kg": 190631.01,
  "total_brix_units": 1931393,
  "total_hours": 58.93,
  "total_pallets": 0,
  "orders": 4,
  "avg_brix": 10.1411869
}
```

Observed response values:

```text
prediction_kwh:               18424.13407399847
post_only_prediction_kwh:     18971.65182261905
full_history_prediction_kwh:  17146.592660550457
branch_disagreement_kwh:       1825.0591620685918
branch_disagreement_pct:         10.106023635334983
branch_disagreement_status:      moderate
```

The operational inputs were classified as:

```text
inside_typical_development_range
```

The service correctly reported:

```text
week_of_year=44
```

as an unseen calendar value.

The warning codes were:

```text
UNSEEN_CALENDAR_VALUE
MODERATE_BRANCH_DISAGREEMENT
```

The clean-machine prediction matched the known reference prediction from
the original development computer.

## Batch prediction

A two-record request was submitted to:

```text
POST /v1/predict/batch
```

Results:

- HTTP response: `200 OK`
- Returned count: `2`
- Input record order was preserved
- Each record received its own support diagnostics
- The first batch prediction matched the equivalent single prediction

The absolute difference between the first batch prediction and the
single prediction was:

```text
0.00000000000 kWh
```

This confirms deterministic consistency between the single and batch
serving paths for the tested record.

## Runtime log evidence

Docker runtime logs showed successful requests for:

```text
GET  /health
GET  /ready
POST /v1/predict
POST /v1/predict/batch
```

All acceptance-test requests returned `200 OK`.

A local copy of the runtime log was retained as:

```text
JMM-clean-machine-docker.log
```

The raw runtime log is supporting local evidence and is not required for
normal repository operation.

## Acceptance conclusion

The test demonstrates that the committed repository alone is sufficient
to:

1. reconstruct the project on independent hardware;
2. reproduce the approved serving model artifact;
3. verify model-artifact integrity;
4. build the Docker image without using the original development
   environment;
5. start the forecasting service successfully;
6. load modeling version `2.0`;
7. produce the expected single forecast;
8. produce consistent ordered batch forecasts;
9. expose working health and readiness endpoints.

This test provides reproducibility, integration, packaging, and serving
evidence.

It is not additional model-accuracy evidence and does not replace the
chronological validation and test results documented in the modeling
records.