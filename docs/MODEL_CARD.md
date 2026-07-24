# Damavand Energy Forecasting Model Card

## Model identification

**Model name:** Damavand Daily Active Energy Forecasting Ensemble
**Official forecasting version:** 2.0
**Official post-selection evaluation:** 2.1
**Status:** Validated release candidate
**Model type:** Weighted regression ensemble
**Forecast horizon:** Daily
**Target:** `active_energy_kWh`
**Serving status:** FastAPI and Docker implementation complete
**External CI status:** Workflow configured; first GitHub-hosted run pending publication

The official forecasting system combines two independently developed tree-based regression branches:

```text
70% post-only Extra Trees
30% full-history AdaBoost
```

Version 2.0 defines the forecasting model and its selected ensemble weights. Version 2.1 defines the official behavioral scenario evaluation, historical replay analysis, local sensitivity checks, branch-disagreement analysis, and operational-range safeguards. Version 2.1 does not replace or retrain the Version 2.0 forecasting model.

## Model card at a glance

| Area | Evidence |
|---|---|
| Regime-aware design | Separate post-intervention and full-history branches |
| Official model | 70% post-only Extra Trees, 30% full-history AdaBoost |
| Selection rule | Predefined chronological validation-first weight comparison |
| Validation | MAE 861.033 kWh, MAPE 4.668%, R² 0.809 |
| Final holdout | MAE 1,022.456 kWh, MAPE 6.214%, R² 0.755 |
| Behavioral evaluation | 20 scenarios, 17 tests, 27 runtime hard checks |
| Serving | Typed FastAPI single and batch inference |
| Safeguards | Input support, calendar coverage, branch disagreement, warning codes |
| Artifact governance | Metadata validation and SHA-256 verification before deserialization |
| Automated evidence | 60 tests, Docker smoke test, 100-record batch integration |
| Reproducibility | Exact model hash and prediction reproduced on an independent clean machine |
| Observability | Prometheus-compatible aggregate service metrics |

## Model summary

The model forecasts daily active energy consumption for the Damavand industrial process using production, operational, and calendar variables.

The ensemble was designed to combine two complementary signals:

- the post-only Extra Trees branch emphasizes the current post-installation operating regime;
- the full-history AdaBoost branch contributes broader historical production-energy structure.

The 70/30 weight was selected by validation MAE within a predefined post-only-dominant search containing the 70/30 and 60/40 candidate blends. The final test period was not used to choose the official weight.

## Development ownership and project context

Senerqon provided the professional project context for the Damavand forecasting use case.

Within the technical scope represented by this repository, the portfolio author independently designed and implemented the modeling workflow, validation policy, experiment tracking, ensemble governance, behavioral evaluation, inference service, safeguards, testing, containerization, observability, reproducibility checks, and release hardening.

This model card describes the technical system and its evidence boundaries. It is not an official Damavand or Senerqon product statement.

## Intended use

The model is intended to support:

- daily operational energy forecasting;
- production and energy planning;
- expected-consumption baselining;
- batch forecasting for planned operating schedules;
- comparison of forecast energy against later observed energy;
- operational monitoring through input-range and branch-disagreement warnings.

The model should be used as a decision-support tool. Forecasts should be interpreted together with the reported input-support status and component-disagreement diagnostics.

## Intended users

The expected users are:

- operations and production-planning teams;
- energy-management teams;
- data and machine-learning engineers;
- analysts responsible for forecast monitoring and retraining decisions.

## Unsupported uses

The model is not designed for:

- real-time equipment control;
- safety-critical automation without human oversight;
- direct causal estimation of individual production variables;
- forecasting from incomplete, invalid, or inconsistent feature inputs;
- treating branch disagreement as a calibrated confidence interval;
- unrestricted extrapolation beyond observed operating conditions;
- permanent use without revalidation as new post-installation data accumulates.

Forecasts for outside-range inputs should be treated as flagged estimates requiring review rather than routine forecasts.

## Prediction target

```text
active_energy_kWh
```

The target represents daily active electrical energy consumption measured in kilowatt-hours.

## Input features

### Post-only Extra Trees branch

The post-only branch uses the full 13-feature set:

```text
total_kg
total_nominal_kg
total_brix_units
total_hours
total_pallets
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
month
year
week_of_year
```

### Full-history AdaBoost branch

The full-history branch uses the VIF-reduced feature set:

```text
total_kg
total_hours
total_pallets
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
month
year
```

### Derived-feature requirement

The yield ratio must remain internally consistent:

```text
yield_ratio_actual_over_nominal = total_kg / total_nominal_kg
```

The `orders` feature is treated as a nonnegative integer-like count.

## Model architecture

### Post-only branch

```text
Model: ExtraTreesRegressor
Version: 0.5
Feature set: full
Training scope: post-installation only
```

Selected configuration:

```python
ExtraTreesRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    random_state=33,
    n_jobs=-1,
)
```

### Full-history branch

```text
Model: AdaBoostRegressor
Version: 1.3
Feature set: vif_auto_full_history
Training scope: all available history through the training cutoff
```

Selected configuration:

```python
AdaBoostRegressor(
    estimator=DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=3,
        random_state=33,
    ),
    n_estimators=50,
    learning_rate=0.03,
    loss="linear",
    random_state=33,
)
```

### Ensemble calculation

```text
official_prediction =
    0.70 * post_only_prediction
    + 0.30 * full_history_prediction
```

The retained sensitivity blend is:

```text
sensitivity_prediction =
    0.60 * post_only_prediction
    + 0.40 * full_history_prediction
```

The 60/40 blend is not the official model because its strongest advantage appeared on the final test period rather than the validation selection period.

## Data regimes

The project contains a known intervention or installation event on:

```text
2025-09-12
```

This creates two operating regimes:

- pre-installation history, which provides more observations but may reflect an earlier process relationship;
- post-installation history, which is more representative of the current operating regime.

The ensemble deliberately combines one branch from each regime strategy.

## Chronological split design

### Post-only split

```text
post_train:      30 rows, 2025-09-12 to 2025-10-17
post_validation:  7 rows, 2025-10-18 to 2025-10-24
post_test:        7 rows, 2025-10-25 onward
```

### Full-history split

```text
full_history_train:      452 rows through 2025-10-17
full_history_validation:   7 rows, 2025-10-18 to 2025-10-24
full_history_test:         7 rows, 2025-10-25 onward
```

The validation and test windows are shared across model branches so that comparisons are made on the same post-installation periods.

## Model-selection policy

The project uses chronological validation-first selection.

- Validation MAE is the primary selection metric.
- The final test period is not used for tuning or weight selection.
- Candidate models that improve test results while weakening validation are not promoted on that basis.
- The official 70/30 ensemble was selected only from the predefined 70/30 and 60/40 blends.
- The 60/40 blend remains a documented sensitivity result.

This policy is intended to reduce test-set chasing under the short seven-day evaluation windows.

## Evidence hierarchy

The project deliberately separates different kinds of evidence:

1. **Chronological validation and final holdout testing** provide the primary generalization evidence.
2. **Bootstrap intervals** describe uncertainty in the seven-day holdout metrics.
3. **Historical replay** provides representative-profile fit evidence after refitting.
4. **Synthetic local and adversarial scenarios** provide behavioral and safeguard evidence.
5. **The 100-record synthetic batch exercise** provides API, ordering, warning-system, and container-integration evidence.
6. **Docker and clean-machine tests** provide serving and reproducibility evidence.
7. **Prometheus metrics** demonstrate observability capability, not live accuracy history.

Synthetic scenarios are not treated as observed ground truth, and no synthetic result is reported as a model-accuracy improvement.

## Official performance

### Validation performance

| Metric | Value |
|---|---:|
| MAE | 861.033368 kWh |
| RMSE | 1,036.787105 kWh |
| R² | 0.809018 |
| MAPE | 4.668177% |
| Total deviation | 0.633766% |

### Final holdout test performance

| Metric | Value |
|---|---:|
| MAE | 1,022.455700 kWh |
| RMSE | 1,282.703944 kWh |
| R² | 0.754668 |
| MAPE | 6.213821% |
| Total deviation | 1.751245% |

### Bootstrap 95% confidence intervals on the final test period

| Metric | 95% confidence interval |
|---|---:|
| MAE | [507.5856, 1,542.1063] kWh |
| RMSE | [745.0319, 1,689.8226] kWh |
| MAPE | [2.7120%, 10.1456%] |
| Total deviation | [-3.3546%, 7.7958%] |

The intervals are wide because the final holdout window contains seven days.

## Champion-challenger comparison

| Candidate | Validation MAE | Validation R² | Test MAE | Test R² | Test total deviation |
|---|---:|---:|---:|---:|---:|
| Post-only Extra Trees | 735.7531 | 0.8429 | 1,492.3472 | 0.4246 | 4.3923% |
| Full-history AdaBoost | 1,329.8433 | 0.0274 | 1,473.4256 | 0.5238 | -4.4112% |
| Official 70/30 ensemble | 861.0334 | 0.8090 | 1,022.4557 | 0.7547 | 1.7512% |
| 60/40 sensitivity ensemble | 902.7935 | 0.7576 | 931.1506 | 0.8075 | 0.8709% |

The post-only branch remains the strongest single model on validation MAE. The official ensemble is retained because it preserves strong validation performance while materially improving final holdout behavior relative to both component branches.

## Historical replay evidence

Version 2.1 includes eight representative real post-installation development profiles selected through a deterministic archetype-matching procedure.

The selection procedure defines operating archetypes such as baseline, low activity, high activity, high orders, low orders with high production, high Brix, low Brix, and weekend operation. Candidate real rows are compared with target quantile profiles using standardized multivariable distance, and the closest complete historical row is selected.

This prevents manual selection based on forecast error and provides reproducible representative-profile coverage.

### Replay summary

```text
profiles:                       8
replay MAE:               695.559 kWh
replay RMSE:              847.874 kWh
replay MAPE:                3.110%
mean signed deviation:    +463.970 kWh
aggregate deviation:    +3,711.760 kWh
aggregate deviation percentage: +1.560%
```

The weekend-moderate and selected high-Brix profiles produced the largest percentage deviations among the eight replay examples. They are monitoring candidates because they identify operating conditions that were less precisely represented than the other selected archetypes.

Replay results are supplementary representative-profile evidence after final refitting. Chronological validation and final holdout testing remain the primary measures of generalization.

## Behavioral evaluation

Version 2.1 evaluates 20 scenarios:

```text
8 historical reference replays
8 paired local synthetic operating-scale perturbations
4 adversarial or out-of-distribution perturbations
```

The paired synthetic scenarios apply a coherent 5% increase to:

```text
total_kg
total_nominal_kg
total_brix_units
total_hours
```

They preserve:

```text
orders
total_pallets
avg_brix
calendar context
```

The yield ratio is recalculated consistently.

### Verification results

```text
unit tests passed:       17
runtime hard checks:     27 passed, 0 failed
invalid schemas:          0
```

### Input-support classification

```text
inside typical development range: 9
development-distribution tail:    7
outside observed range:           4
```

Classification logic:

- `inside_typical_development_range`: all variable operational features are within their q10-q90 intervals;
- `development_distribution_tail`: all features remain within observed min/max, but at least one variable feature is outside q10-q90;
- `outside_observed_range`: at least one operational feature is outside the observed development min/max.

The classification is a marginal feature-support check. Scenario design labels such as `boundary`, `adversarial`, and `out_of_distribution` describe why a complete profile was included and are distinct from the calculated range status.

### Directional sensitivity results

```text
paired directional checks passed: 8
locally flat responses:            0
diagnostic warnings:               0
```

The strongest local response occurred for:

```text
high_orders_moderate_production
```

with:

```text
official ensemble change:     +2,030.734 kWh, +10.099%
post-only Extra Trees change: +1,263.583 kWh
full-history AdaBoost change: +3,820.752 kWh
```

This profile is the primary local sensitivity hotspot for future monitoring.

## Production-intensity diagnostic

The adversarial evaluation also uses:

```text
kg_per_hour = total_kg / total_hours
```

This diagnostic represents production processed per operating hour.

It is not an additional model input. It is used to identify unusual relationships between production volume and operating time that may not be obvious when `total_kg` and `total_hours` are checked separately.

## Component disagreement

Absolute disagreement is the difference between the two branch predictions in kWh.

Relative disagreement expresses that difference as a percentage of the mean magnitude of the component predictions.

### Predefined thresholds

```text
low:      below 10%
moderate: 10% to below 20%
high:     20% or greater
```

### Version 2.1 results

```text
low-disagreement scenarios:      15
moderate-disagreement scenarios:  5
high-disagreement scenarios:      0
```

Maximum absolute disagreement:

```text
scenario: low_orders_high_production
absolute disagreement: 5,190.830 kWh
relative disagreement: 7.825%
```

Maximum relative disagreement:

```text
scenario: high_orders_moderate_production_operating_scale_up_5pct
absolute disagreement: 3,624.241 kWh
relative disagreement: 15.852%
```

Branch disagreement is an operational monitoring signal. It is not a calibrated prediction interval.

## Implemented warning behavior

The prediction service currently returns:

- official 70/30 prediction;
- post-only branch prediction;
- full-history branch prediction;
- absolute branch disagreement in kWh;
- relative branch disagreement percentage;
- disagreement category;
- operational-range status;
- calendar-coverage status;
- distribution-tail feature names;
- outside-range feature names;
- unseen calendar feature/value pairs;
- machine-readable warning codes.

Implemented warning codes are:

```text
DEVELOPMENT_DISTRIBUTION_TAIL
OUTSIDE_OBSERVED_RANGE
UNSEEN_CALENDAR_VALUE
MODERATE_BRANCH_DISAGREEMENT
HIGH_BRANCH_DISAGREEMENT
```

Input-support status and branch disagreement are reported independently. Low branch disagreement does not remove an outside-range warning.

Warnings do not block a numerical forecast. They identify conditions requiring additional operational review.

## Serving implementation and operational controls

The official ensemble is served through a FastAPI application with:

```text
GET  /
GET  /health
GET  /ready
GET  /metrics
GET  /v1/model
POST /v1/predict
POST /v1/predict/batch
```

The implementation includes:

- strict Pydantic validation and rejection of unknown fields;
- one shared feature-building path for both model branches and support diagnostics;
- deterministic model feature ordering;
- single-record and ordered 1–500 record batch inference;
- request correlation IDs and structured JSON request logs;
- safe internal-error responses;
- startup loading of one validated prediction service;
- bounded-label Prometheus metrics;
- a non-root Docker runtime and health check;
- automated container acceptance testing.

The approved artifact is:

```text
models/ensemble_2_0_model.joblib
```

Approved SHA-256:

```text
A4A7945CA5E77387BAB3854597F380F8445B35849EF3B640B340507B26112282
```

The checksum is verified before Joblib deserialization. Startup also validates the model version, target, component classes, feature names, feature order, weights, and weight sum.

The automated test suite currently reports:

```text
60 passed
```

The independent clean-machine procedure reproduced the same artifact hash and approved reference prediction:

```text
18424.13407399847 kWh
```

These controls establish serving integrity and reproducibility. They do not create additional model-accuracy evidence.

## Known limitations

The primary limitations are:

- only 30 post-installation training rows were available for the post-only branch;
- validation and final test windows contain seven days each;
- the intervention may have changed the production-energy relationship;
- the two branches use tree-based models with piecewise-constant response behavior;
- marginal range checks do not measure the probability of a complete multivariable profile;
- long-term seasonal stability has not yet been established;
- the model has not yet been evaluated through continuing live production traffic and delayed actual-value monitoring;
- operational counters are process-local until scraped by an external monitoring system;
- no authentication or authorization layer is defined by the application itself;
- branch disagreement is not calibrated probabilistic uncertainty.

These limitations define the current evidence boundary and the required monitoring plan.

## Monitoring status and recommendations

### Implemented operational monitoring

The API currently exposes aggregate Prometheus-compatible metrics for:

- HTTP requests by method, bounded route, and status;
- request-duration histograms;
- successful prediction records by single or batch request type;
- operational-range classifications;
- calendar-coverage classifications;
- branch-disagreement classifications;
- warning-code totals.

The metrics do not contain raw production inputs, individual forecasts, observed energy values, or personal data.

The counters are held in the running process and reset on restart. Persistent history requires an external Prometheus-compatible scraper.

### Input monitoring

Continue tracking:

- missing or invalid fields;
- feature type violations;
- non-integer or negative `orders` values;
- yield-ratio consistency;
- operational-range status;
- frequency of distribution-tail and outside-range requests;
- calendar values not represented in development data;
- univariate feature drift;
- multivariate drift when sufficient live data becomes available.

### Prediction monitoring

Continue tracking:

- official prediction distribution;
- component prediction distributions;
- absolute and relative branch disagreement;
- frequency of moderate and high disagreement;
- forecasts near observed prediction extrema;
- repeated forecasts for unsupported operating conditions.

### Future actual-based accuracy monitoring

When delayed actual energy becomes available, track:

- rolling MAE;
- rolling RMSE;
- rolling MAPE;
- rolling signed bias;
- rolling aggregate total deviation;
- errors grouped by weekday and weekend;
- errors grouped by operational-range status;
- errors for high-Brix, high-order, low-production, and weekend profiles;
- component and ensemble performance separately.

Live actual-value ingestion and rolling accuracy computation are not implemented because no continuing operational data feed is connected to this portfolio release.

## Retraining and revalidation triggers

Retraining or weight revalidation should be considered when one or more of the following occur:

- a meaningful volume of new post-installation data becomes available;
- rolling MAE or MAPE degrades materially relative to the official baseline;
- signed bias remains persistently positive or negative;
- outside-range or distribution-tail requests become frequent;
- moderate or high branch disagreement becomes materially more common;
- input distributions change because of process, product, equipment, or scheduling changes;
- a new intervention or installation changes the operating regime;
- weekend or high-Brix errors remain systematically higher after more observations are collected;
- the 60/40 sensitivity blend or another challenger consistently outperforms the official model on new chronological data.

Any new model or ensemble weight should be selected using a fresh chronological validation period and then confirmed on a separate holdout period.

## Serving and release status

### Implemented for the portfolio release candidate

```text
FastAPI single and batch prediction
strict request validation
deterministic feature construction and ordering
model metadata validation
SHA-256 artifact verification
input-support and branch-disagreement diagnostics
request IDs and structured logs
Prometheus operational metrics
Docker packaging
non-root container runtime
Docker health check
60 automated tests
automated Docker smoke testing
100-record synthetic batch integration
independent clean-machine acceptance
GitHub Actions workflow configuration
```

### Pending final publication and release actions

```text
first GitHub-hosted CI execution
merge of the approved release branch into main
final release provenance manifest
release notes and rollback instructions
versioned Docker image tag
v1.0.0 Git tag and GitHub release
```

### Required only for a real continuing production deployment

```text
deployment-specific authentication and authorization
TLS and network controls
persistent Prometheus-compatible monitoring storage
alert routing and operational ownership
delayed actual-value ingestion
rolling accuracy and drift monitoring
documented incident response
retraining approval workflow
```

The separate 100-record synthetic dataset is used only for batch inference, API, Docker, calendar-coverage, warning-system, and integration testing. It remains separate from real chronological performance evaluation.

## Reproducibility and artifacts

### Approved serving package

```text
models/ensemble_2_0_model.joblib
config/serving/model_metadata.json
config/serving/input_reference.json
```

Approved model SHA-256:

```text
A4A7945CA5E77387BAB3854597F380F8445B35849EF3B640B340507B26112282
```

### Serving implementation

```text
src/api/
src/serving/
Dockerfile
.dockerignore
requirements-serving.txt
scripts/smoke_test_container.ps1
```

### Automated validation

```text
tests/
pyproject.toml
.github/workflows/ci.yml
```

Current local result:

```text
60 passed
```

The GitHub Actions workflow runs the Python 3.12 test suite and then the no-cache Docker acceptance test. The workflow is committed; hosted execution remains pending until publication.

### Independent acceptance evidence

```text
docs/validation/clean_machine_acceptance_test.md
```

The clean-machine exercise confirmed repository reconstruction, exact model checksum, no-cache Docker build, healthy startup, readiness, deterministic reference prediction, and single/batch consistency.

### Version 2.0 model-development evidence

```text
reports/ensemble_2_0_training_report.xlsx
reports/metadata/ensemble_2_0_features.txt
```

### Version 2.1 behavioral evidence

```text
reports/ensemble_2_1_behavioral_stress_test_report.xlsx
reports/figures/ensemble_2_1_behavioral_stress_test/
src/stress_test_ensemble.py
tests/test_stress_test_ensemble.py
```

Expected behavioral figures:

```text
branch_disagreement_kwh.png
branch_disagreement_pct.png
official_70_30_scenario_predictions.png
paired_local_sensitivity_delta_pct.png
```

### MLflow tracking

```text
Experiment: Damavand Energy Forecasting
Version 2.1 run name: 2.1 Behavioral Scenario Evaluation and Synthetic Stress Test
Run ID: c07bfa3a352f49e3b85bb838e62bdee7
Tracking URI: sqlite:///mlflow.db
```

The local MLflow database and artifact store are excluded from the repository. The run identifier is retained as provenance and is not presented as a publicly hosted experiment.

### Data and privacy boundary

The repository excludes:

```text
raw industrial data
processed private datasets
credentials and environment secrets
personal data
internal communications
MLflow databases and artifact stores
private runtime logs
```

Operational metrics contain aggregate classifications and counts rather than raw request values or individual predictions.

## Governance decision

The official forecasting model is:

```text
Version 2.0
70% post-only Extra Trees
30% full-history AdaBoost
```

The official post-selection evaluation is:

```text
Version 2.1
Behavioral Scenario Evaluation and Synthetic Stress Test
```

The model is approved as a validated and defensible industrial energy-forecasting release candidate under the currently available post-intervention evidence.

Approval is conditional on the following interpretation:

- chronological validation and holdout testing remain the primary accuracy evidence;
- behavioral scenarios and historical replay are supplementary diagnostics;
- support and disagreement warnings must accompany forecasts;
- outside-range forecasts require additional operational review;
- model performance, weights, and support boundaries must be revalidated as real post-intervention observations accumulate;
- a live deployment requires persistent monitoring, actual-value ingestion, security controls, and operational ownership beyond this portfolio release.

The current repository demonstrates end-to-end model governance, serving integrity, reproducibility, testing, and observability without claiming long-running production operation or enterprise-scale infrastructure.
