# Modeling Notes

## Project objective

This project builds a production-style machine learning pipeline for Damavand post-installation energy forecasting.

The current modeling objective is to predict post-installation daily active energy consumption using production, operational, and calendar features. The project is structured as a realistic ML engineering workflow rather than a single notebook experiment.

The pipeline includes:

* data loading from a processed Damavand dataset
* validation of required columns
* date parsing and cleaning
* chronological post-installation train, validation, and test splits
* multiple candidate regression models
* model selection using validation metrics
* final holdout test reporting
* bootstrap confidence intervals
* diagnostic plots
* Excel report generation
* model artifact export
* MLflow experiment tracking
* feature-set sensitivity checks

The current baseline is not yet treated as the final production model. It is the first tracked post-installation forecasting baseline before adding stronger model families such as CatBoost.

## Data and split strategy

The processed dataset is loaded from:

```text
data/processed/damavand.csv
```

The target variable is:

```text
active_energy_kWh
```

The current post-installation split is:

```text
post_train:      2025-09-12 to 2025-10-17
post_validation: 2025-10-18 to 2025-10-24
post_test:       2025-10-25 onward
```

The resulting split sizes are:

```text
post_train: 30 rows
post_validation: 7 rows
post_test: 7 rows
```

The validation period is used for model and feature-set decisions. The test period is used as a final holdout check after a candidate has been selected.

Because the validation and test windows are short, final conclusions are interpreted carefully. Bootstrap confidence intervals are reported for test metrics to reflect uncertainty from the small test window.

## Modeling Version 0.1 — Full Feature Baseline

Version 0.1 is the first tracked post-installation forecasting baseline.

The default command:

```powershell
python -m src.train
```

uses the full original feature set.

The full feature set contains 13 features:

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

Candidate models evaluated:

```text
dummy_mean
linear_regression
ridge_regression
random_forest
gradient_boosting
```

Model selection is based on validation MAE.

### Validation results

Gradient Boosting was selected by validation MAE.

```text
selected model: gradient_boosting
validation MAE: 1693.194496
validation RMSE: 1907.001901
validation R²: 0.353876
validation MAPE: 9.724538%
validation total deviation: 8.622433%
```

### Test results

```text
test MAE: 1641.4317
test RMSE: 2140.8388
test R²: 0.3166
test MAPE: 10.2851%
test total deviation: 4.4430%
```

### Bootstrap 95% confidence intervals

```text
test MAE CI: [745.6186, 2733.9019]
test RMSE CI: [904.1184, 3126.4487]
test MAPE CI: [3.7573, 18.4719]
test total deviation CI: [-3.1223, 14.9665]
```

### Version 0.1 interpretation

The full-feature baseline is retained as the current selected baseline because it provides the strongest balance of validation performance, domain credibility, and final holdout behavior among the checked feature sets.

The final holdout test result is positive but should be interpreted with caution because the test set contains only 7 rows.

## Multicollinearity diagnostic

A VIF-based multicollinearity diagnostic was added for the post-installation development period.

The diagnostic uses:

```text
post_train + post_validation
```

It does not use the final test period to calculate VIF.

This is important because the test period should remain a final holdout period and should not influence feature diagnostics or model-selection decisions.

The VIF diagnostic confirmed strong redundancy among several production-related variables. This was expected because production weight, nominal production weight, brix-related quantities, pallet count, operating hours, and calendar variables are structurally related.

The automatic VIF reduction used a threshold of:

```text
VIF < 5
```

The automatic post-only VIF feature set retained:

```text
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

The diagnostic is useful, but VIF is not treated as an automatic feature-selection rule. VIF is a linear redundancy diagnostic, while the selected candidate model is tree-based Gradient Boosting. Tree-based models can often tolerate correlated predictors better than linear models.

Therefore, VIF results were used to design feature-set sensitivity checks rather than to automatically replace the full feature set.

## Feature-set experiment system

A named feature-set registry was added in:

```text
src/feature_sets.py
```

The training script now supports:

```powershell
python -m src.train --feature-set <feature_set_name>
```

If no feature set is provided, the default is:

```text
full
```

Therefore, the standard command still uses the original full feature set:

```powershell
python -m src.train
```

Named feature sets currently available:

```text
full
vif_auto_post_only
reduced_without_total_kg
domain_reduced_with_total_kg
```

This makes feature-set diagnostics reproducible and prevents hidden manual edits to `settings.py`.

For MLflow logging, each run records:

* feature set name
* feature count
* selected feature list
* selected model
* validation metrics
* test metrics
* bootstrap intervals
* report artifacts
* model artifacts
* diagnostic plots

## Feature-set diagnostic runs

Feature-set diagnostic runs were logged to MLflow under the experiment:

```text
Damavand Energy Forecasting
```

The logged runs are:

```text
0.1   Full Feature Baseline
0.1.1 VIF Auto Feature Set Diagnostic
0.1.2 Reduced Feature Challenger Without Total KG
0.1.3 Domain Reduced Feature Set With Total KG
```

These feature-set runs are diagnostic experiments. They are used to evaluate whether reduced feature sets provide a stronger validation-stage model than the full feature set.

The final model selection is not based on test metrics alone. Test metrics are retained as final holdout diagnostics.

## Run 0.1.1 — VIF Auto Feature Set Diagnostic

Feature set:

```text
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

Feature count:

```text
6
```

### Validation results

```text
selected model: gradient_boosting
validation MAE: 7265.365491
validation RMSE: 9016.756211
validation R²: -13.444889
validation MAPE: 41.150888%
validation total deviation: 38.684591%
```

### Test results

```text
test MAE: 4401.3620
test RMSE: 4755.0347
test R²: -2.3714
test MAPE: 25.6888%
test total deviation: 6.6977%
```

### Interpretation

The automatic VIF-reduced feature set performs poorly. It removes too much production information and does not provide a credible replacement for the full feature baseline.

This run is useful because it shows that blindly following VIF reduction is not appropriate for this dataset and modeling objective.

## Run 0.1.2 — Reduced Feature Challenger Without Total KG

Feature set:

```text
total_hours
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

Feature count:

```text
7
```

### Validation results

```text
selected model: gradient_boosting
validation MAE: 1659.338997
validation RMSE: 1862.765469
validation R²: 0.383505
validation MAPE: 9.189292%
validation total deviation: 8.769039%
```

### Test results

```text
test MAE: 2742.0671
test RMSE: 3533.1256
test R²: -0.8613
test MAPE: 17.0850%
test total deviation: 8.1745%
```

### Interpretation

This reduced feature set is the strongest reduced challenger on validation MAE and validation R². However, it removes `total_kg`, which is a core production-volume feature.

The validation improvement over the full baseline is small:

```text
full validation MAE: 1693.194496
reduced validation MAE: 1659.338997
difference: 33.855499 kWh
```

Because the improvement is small and the feature set removes a key production-volume driver, this reduced model is not selected as the current baseline.

The weaker final holdout test behavior supports the concern that this reduced set is less stable, but test performance is not used as the sole selection criterion.

## Run 0.1.3 — Domain Reduced Feature Set With Total KG

Feature set:

```text
total_kg
total_hours
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

Feature count:

```text
8
```

### Validation results

```text
selected model: gradient_boosting
validation MAE: 1877.156970
validation RMSE: 2491.238160
validation R²: -0.102666
validation MAPE: 10.095593%
validation total deviation: 10.459972%
```

### Test results

```text
test MAE: 2381.5436
test RMSE: 2959.5078
test R²: -0.3060
test MAPE: 14.7015%
test total deviation: 8.7337%
```

### Interpretation

This reduced feature set is more domain-credible than the reduced set without `total_kg` because it preserves production volume and operating hours.

However, its validation performance is weaker than the full-feature baseline.

Therefore, it is not selected.

## Feature-set decision summary

The feature-set experiments show:

```text
VIF-auto feature set:
- rejected
- validation performance is much worse
- removes too much production information

Reduced set without total_kg:
- strongest reduced validation result
- small validation improvement over full baseline
- removes key production-volume feature
- not selected due to weak domain credibility and weaker stability

Domain reduced with total_kg:
- more domain-credible
- validation performance worse than full baseline
- not selected

Full feature baseline:
- strongest overall balance
- preserves important production and calendar drivers
- selected as the current baseline
```

The final decision is:

```text
Retain the full feature set for the current post-installation baseline.
```

This conclusion is based on validation-stage behavior, feature credibility, and the diagnostic evidence from reduced-feature checks.

## MLflow tracking

MLflow is used for intentional experiment tracking.

The local tracking backend is:

```text
sqlite:///mlflow.db
```

The project does not log raw data to MLflow.

The current MLflow experiment is:

```text
Damavand Energy Forecasting
```

The training script logs:

* model version
* feature set name
* feature count
* selected feature list
* model type
* selected model name
* candidate model names
* split dates
* split row counts
* validation metrics
* test metrics
* bootstrap confidence intervals
* Excel report artifact
* model artifact
* feature-list artifact
* diagnostic plot artifacts

Normal runs do not log to MLflow:

```powershell
python -m src.train
```

Intentional runs are logged with:

```powershell
python -m src.train --log-mlflow
```

or with a selected feature set:

```powershell
python -m src.train --feature-set vif_auto_post_only --log-mlflow
```

## Current conclusion

The current selected baseline is:

```text
0.1 Full Feature Baseline
```

Selected model:

```text
GradientBoostingRegressor
```

Selected feature set:

```text
full
```

The full-feature baseline remains selected because it provides the best overall balance among the tested feature sets.

The feature-set diagnostic runs are retained as evidence that multicollinearity was investigated and that reduced feature sets were evaluated rather than assumed.

## Next modeling step

The next modeling step is to add CatBoost as a stronger tabular model candidate.

CatBoost should be evaluated as a new model-family candidate after the current baseline and feature-set diagnostics.

The next version should likely be:

```text
0.2 CatBoost Candidate
```

The CatBoost run should be compared against the current `0.1 Full Feature Baseline` using the same post-installation split and the same validation-first selection logic.
