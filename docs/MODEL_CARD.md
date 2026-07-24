# Damavand Energy Forecasting Model Card

## Model identification

**Model name:** Damavand Daily Active Energy Forecasting Ensemble  
**Official forecasting version:** 2.0  
**Official post-selection evaluation:** 2.1  
**Status:** Official forecasting candidate  
**Model type:** Weighted regression ensemble  
**Forecast horizon:** Daily  
**Target:** `active_energy_kWh`

The official forecasting system combines two independently developed tree-based regression branches:

```text
70% post-only Extra Trees
30% full-history AdaBoost
```

Version 2.0 defines the forecasting model and its selected ensemble weights. Version 2.1 defines the official behavioral scenario evaluation, historical replay analysis, local sensitivity checks, branch-disagreement analysis, and operational-range safeguards. Version 2.1 does not replace or retrain the Version 2.0 forecasting model.

## Model summary

The model forecasts daily active energy consumption for the Damavand industrial process using production, operational, and calendar variables.

The ensemble was designed to combine two complementary signals:

- the post-only Extra Trees branch emphasizes the current post-installation operating regime;
- the full-history AdaBoost branch contributes broader historical production-energy structure.

The 70/30 weight was selected by validation MAE within a predefined post-only-dominant search containing the 70/30 and 60/40 candidate blends. The final test period was not used to choose the official weight.

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
total_nominal_kg
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

## Warning behavior

A future prediction service should report at least:

- official 70/30 prediction;
- post-only branch prediction;
- full-history branch prediction;
- absolute branch disagreement in kWh;
- relative branch disagreement percentage;
- disagreement category;
- operational-range status;
- calendar-coverage status;
- schema-validation result.

Input-range status and branch disagreement must be reported independently. Low branch disagreement does not remove an outside-range warning.

## Known limitations

The primary limitations are:

- only 30 post-installation training rows were available for the post-only branch;
- validation and final test windows contain seven days each;
- the intervention may have changed the production-energy relationship;
- the two branches use tree-based models with piecewise-constant response behavior;
- marginal range checks do not measure the probability of a complete multivariable profile;
- long-term seasonal stability has not yet been established;
- the model has not yet been evaluated through live production monitoring;
- branch disagreement is not calibrated probabilistic uncertainty.

These limitations define the current evidence boundary and the required monitoring plan.

## Monitoring recommendations

### Input monitoring

Track:

- missing or invalid fields;
- feature type violations;
- non-integer or negative `orders` values;
- yield-ratio inconsistency;
- operational-range status;
- frequency of distribution-tail and outside-range requests;
- calendar values not represented in development data;
- univariate feature drift;
- multivariate drift when sufficient live data becomes available.

### Prediction monitoring

Track:

- official prediction distribution;
- component prediction distributions;
- absolute and relative branch disagreement;
- frequency of moderate and high disagreement;
- forecasts near observed minimum or maximum predictions;
- repeated forecasts for unsupported operating conditions.

### Accuracy monitoring

When actual energy becomes available, track:

- rolling MAE;
- rolling RMSE;
- rolling MAPE;
- rolling signed bias;
- rolling aggregate total deviation;
- errors grouped by weekday/weekend;
- errors grouped by operational-range status;
- errors for high-Brix, high-order, low-production, and weekend profiles;
- component and ensemble performance separately.

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

## Deployment requirements

Before production deployment, the project should include:

- FastAPI single-row and batch prediction endpoints;
- strict input-schema validation;
- deterministic feature ordering;
- model and feature-metadata loading checks;
- independent operational-range and disagreement warnings;
- structured prediction logging;
- delayed actual-value ingestion for accuracy monitoring;
- Docker packaging;
- automated unit, API, integration, and model-loading tests;
- GitHub Actions continuous integration;
- documented rollback and model-version procedures.

The separate 100-day synthetic dataset may be used for batch-inference, API, Docker, calendar-coverage, warning-system, and integration testing. It should remain separate from the real chronological performance evaluation.

## Reproducibility and artifacts

### Official forecasting model

```text
models/ensemble_2_0_model.joblib
```

### Version 2.0 report

```text
reports/ensemble_2_0_training_report.xlsx
```

### Feature metadata

```text
reports/metadata/ensemble_2_0_features.txt
```

### Version 2.1 behavioral report

```text
reports/ensemble_2_1_behavioral_stress_test_report.xlsx
```

### Version 2.1 figures

```text
reports/figures/ensemble_2_1_behavioral_stress_test/
```

Expected figures:

```text
branch_disagreement_kwh.png
branch_disagreement_pct.png
official_70_30_scenario_predictions.png
paired_local_sensitivity_delta_pct.png
```

### Version 2.1 implementation and tests

```text
src/stress_test_ensemble.py
tests/test_stress_test_ensemble.py
```

### MLflow tracking

```text
Experiment: Damavand Energy Forecasting
Version 2.1 run name: 2.1 Behavioral Scenario Evaluation and Synthetic Stress Test
Run ID: c07bfa3a352f49e3b85bb838e62bdee7
Tracking URI: sqlite:///mlflow.db
```

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

The model is approved as a validated and defensible industrial energy-forecasting candidate under the currently available post-installation evidence, subject to input-support warnings, ongoing performance monitoring, and revalidation as additional real data becomes available.
