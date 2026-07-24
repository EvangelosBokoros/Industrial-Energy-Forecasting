# Modeling Notes

## Project objective

This project builds a production-style machine learning pipeline for Damavand post-installation energy forecasting.

The current objective is to predict daily post-installation active energy consumption using production, operational, and calendar features.

This is a portfolio-facing engineering project based on a real industrial energy dataset and a previously delivered Damavand energy analytics project. The goal of this repository is not only to produce a model, but to demonstrate a professional machine learning workflow:

* reproducible data loading
* validated preprocessing
* time-based train, validation, and test splits
* feature-set management
* model comparison
* validation-first model selection
* final holdout reporting
* bootstrap uncertainty intervals
* diagnostic plots
* Excel report generation
* model artifact export
* MLflow experiment tracking
* documented modeling decisions

The project is intentionally structured as a production-style ML workflow rather than a one-off notebook.

## Current production-safe selected model

The current selected production-safe baseline is:

```text
Feature set: full
Selected model: gradient_boosting
```

The selected baseline uses all original production, operational, and calendar features.

This model is selected because it provides the best balance of:

* validation performance
* domain credibility
* full production-volume representation
* stable holdout behavior
* reproducible MLflow tracking
* interpretability of the modeling decision

The reduced feature set without `total_kg` achieved slightly better validation MAE, but it is not selected as the production-safe baseline because it removes a core production-volume driver.

## Data and split strategy

The processed dataset is loaded from:

```text
data/processed/damavand.csv
```

The target variable is:

```text
active_energy_kWh
```

The post-installation split is:

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

The validation set is used for model and feature-set decisions.

The test set is used only as a final holdout check after a candidate is selected.

Because both validation and test windows are short, results are interpreted cautiously. Test bootstrap confidence intervals are reported to communicate uncertainty.

## Full feature set

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

The full feature set is retained as the production-safe feature set because it preserves production volume, nominal production, operating intensity, product characteristics, order activity, and calendar structure.

## Candidate models

The current model comparison includes:

```text
dummy_mean
linear_regression
ridge_regression
random_forest
gradient_boosting
catboost_regularized
```

Candidate model selection is based on validation MAE.

The selected model is then evaluated on the held-out post-installation test period.

## Modeling Version 0.1 — Full Feature Baseline

Version 0.1 is the first tracked post-installation forecasting baseline.

Run name:

```text
0.1 Full Feature Baseline
```

Feature set:

```text
full
```

Selected model:

```text
gradient_boosting
```

### Validation results

```text
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

### Version 0.1 decision

The full-feature Gradient Boosting model is retained as the initial production-safe baseline.

It is not treated as the final model for all future work, but it is the current best real-data baseline before synthetic simulation and XGBoost extension work.

## Multicollinearity diagnostic

A VIF-based multicollinearity diagnostic was added for the post-installation development period.

The diagnostic uses:

```text
post_train + post_validation
```

It does not use the final test period.

This avoids contaminating the final holdout test set.

The diagnostic confirmed strong redundancy among several production-related variables. This was expected because production weight, nominal production weight, brix-related quantities, pallet count, operating hours, and calendar variables are structurally related.

The automatic VIF reduction used a threshold of:

```text
VIF < 5
```

The automatic post-installation VIF feature set retained:

```text
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

The VIF diagnostic is useful, but VIF is not treated as an automatic feature-selection rule.

VIF is a linear redundancy diagnostic. The selected candidate model is tree-based Gradient Boosting. Tree-based models can tolerate correlated predictors better than linear models.

Therefore, VIF results were used to design feature-set sensitivity checks rather than to automatically replace the full feature set.

## Feature-set experiment system

A named feature-set registry was added in:

```text
src/feature_sets.py
```

The training script supports:

```powershell
python -m src.train --feature-set <feature_set_name>
```

If no feature set is provided, the default is:

```text
full
```

Available feature sets:

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

## Modeling Version 0.1 feature-set diagnostics

After the initial full-feature baseline, three feature-set diagnostics were logged.

The purpose was to test whether reduced feature sets improved validation performance or stability.

### Run 0.1.1 — VIF Auto Feature Set Diagnostic

Feature set:

```text
vif_auto_post_only
```

Feature count:

```text
6
```

Retained features:

```text
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

Selected model:

```text
gradient_boosting
```

Validation results:

```text
validation MAE: 7265.365491
validation RMSE: 9016.756211
validation R²: -13.444889
validation MAPE: 41.150888%
validation total deviation: 38.684591%
```

Test results:

```text
test MAE: 4401.3620
test RMSE: 4755.0347
test R²: -2.3714
test MAPE: 25.6888%
test total deviation: 6.6977%
```

Decision:

```text
Rejected
```

Reason:

The automatic VIF feature set removed too much production information and performed much worse than the full feature set.

### Run 0.1.2 — Reduced Feature Challenger Without Total KG

Feature set:

```text
reduced_without_total_kg
```

Feature count:

```text
7
```

Features:

```text
total_hours
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

Selected model:

```text
gradient_boosting
```

Validation results:

```text
validation MAE: 1659.338997
validation RMSE: 1862.765469
validation R²: 0.383505
validation MAPE: 9.189292%
validation total deviation: 8.769039%
```

Test results:

```text
test MAE: 2742.0671
test RMSE: 3533.1256
test R²: -0.8613
test MAPE: 17.0850%
test total deviation: 8.1745%
```

Decision:

```text
Rejected as production-safe baseline
Retained as validation challenger
```

Reason:

This feature set produced the best validation MAE among the reduced feature sets, slightly better than the full feature set. However, it removes `total_kg`, which is a core production-volume feature.

The validation improvement over the full feature set is small:

```text
full validation MAE: 1693.194496
reduced_without_total_kg validation MAE: 1659.338997
difference: 33.855499 kWh
```

Because the improvement is small and the feature set removes a core production driver, it is not selected as the production-safe baseline.

### Run 0.1.3 — Domain Reduced Feature Set With Total KG

Feature set:

```text
domain_reduced_with_total_kg
```

Feature count:

```text
8
```

Features:

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

Selected model:

```text
gradient_boosting
```

Validation results:

```text
validation MAE: 1877.156970
validation RMSE: 2491.238160
validation R²: -0.102666
validation MAPE: 10.095593%
validation total deviation: 10.459972%
```

Test results:

```text
test MAE: 2381.5436
test RMSE: 2959.5078
test R²: -0.3060
test MAPE: 14.7015%
test total deviation: 8.7337%
```

Decision:

```text
Rejected
```

Reason:

This feature set is more domain-credible than the reduced set without `total_kg`, but its validation performance is weaker than the full feature baseline.

## Feature-set decision after Version 0.1

The feature-set diagnostics showed:

```text
VIF-auto feature set:
- rejected
- validation performance much worse
- removed too much production information

Reduced without total_kg:
- slightly better validation MAE
- weak production-domain credibility
- removes core production-volume feature
- retained as a challenger only

Domain reduced with total_kg:
- more domain-credible
- weaker validation performance
- rejected

Full feature set:
- strongest production-safe balance
- preserves production-volume information
- selected as current baseline feature set
```

The production-safe feature decision is:

```text
Retain the full feature set.
```

## Modeling Version 0.2 — Initial CatBoost Candidate Added

Version 0.2 added CatBoost as an additional model candidate.

Run name:

```text
0.2 Full Feature Baseline
```

The purpose was to test whether CatBoost could improve over the existing Gradient Boosting baseline.

CatBoost was added as a candidate model only. It did not replace the existing models.

The selected model remained:

```text
gradient_boosting
```

Initial CatBoost performance on the full feature set was poor.

Validation results for initial CatBoost:

```text
catboost_regularized validation MAE: 7193.397588
catboost_regularized validation RMSE: 7749.902140
catboost_regularized validation R²: -9.671021
catboost_regularized validation MAPE: 40.808120%
catboost_regularized validation total deviation: 40.083347%
```

Decision:

```text
CatBoost not selected
```

Reason:

The first CatBoost configuration substantially underperformed the existing Gradient Boosting model.

## Modeling Version 0.3 — Tuned CatBoost Candidate Comparison

Version 0.3 tested a tuned CatBoost candidate.

Run name:

```text
0.3 Full Feature Baseline With Tuned CatBoost Candidate
```

The tuned CatBoost configuration improved substantially compared with the first CatBoost attempt.

The tuned CatBoost candidate used:

```text
loss_function: RMSE
iterations: 100
learning_rate: 0.05
depth: 2
random_seed: 33
allow_writing_files: False
verbose: False
```

### Full feature set results

Feature set:

```text
full
```

Selected model:

```text
gradient_boosting
```

Validation comparison:

```text
gradient_boosting validation MAE: 1693.194496
catboost_regularized validation MAE: 3109.107828
```

CatBoost validation results:

```text
catboost_regularized validation MAE: 3109.107828
catboost_regularized validation RMSE: 3738.813973
catboost_regularized validation R²: -1.483598
catboost_regularized validation MAPE: 17.340896%
catboost_regularized validation total deviation: 16.485174%
```

Test results for the selected Gradient Boosting model:

```text
test MAE: 1641.4317
test RMSE: 2140.8388
test R²: 0.3166
test MAPE: 10.2851%
test total deviation: 4.4430%
```

Decision:

```text
CatBoost rejected on full feature set
Gradient Boosting retained
```

Reason:

Tuned CatBoost improved compared with the initial CatBoost attempt, but it remained substantially worse than Gradient Boosting on validation MAE.

## Modeling Version 0.3 feature-set diagnostics

Because CatBoost was tuned in version 0.3, the feature-set diagnostics were rerun under the updated candidate pool.

The purpose was to check whether CatBoost performed better under reduced feature sets.

### Run 0.3.1 — VIF Auto Feature Set With Tuned CatBoost Candidate

Feature set:

```text
vif_auto_post_only
```

Selected model:

```text
gradient_boosting
```

Validation comparison:

```text
gradient_boosting validation MAE: 7265.365491
catboost_regularized validation MAE: 13498.759186
```

CatBoost validation results:

```text
catboost_regularized validation MAE: 13498.759186
catboost_regularized validation RMSE: 14144.757601
catboost_regularized validation R²: -34.547132
catboost_regularized validation MAPE: 75.295003%
catboost_regularized validation total deviation: 75.218344%
```

Decision:

```text
CatBoost rejected
VIF-auto feature set rejected
```

Reason:

CatBoost performed worse than Gradient Boosting, and the VIF-auto feature set remained weak.

### Run 0.3.2 — Reduced Feature Set Without Total KG With Tuned CatBoost Candidate

Feature set:

```text
reduced_without_total_kg
```

Selected model:

```text
gradient_boosting
```

Validation comparison:

```text
gradient_boosting validation MAE: 1659.338997
catboost_regularized validation MAE: 7745.187633
```

CatBoost validation results:

```text
catboost_regularized validation MAE: 7745.187633
catboost_regularized validation RMSE: 9506.626018
catboost_regularized validation R²: -15.057073
catboost_regularized validation MAPE: 42.315445%
catboost_regularized validation total deviation: 43.158055%
```

Decision:

```text
CatBoost rejected
Reduced feature set retained only as a challenger
```

Reason:

CatBoost performed much worse than Gradient Boosting. The reduced feature set without `total_kg` still achieved the best validation MAE overall, but it remains less production-credible because it removes a core production-volume feature.

### Run 0.3.3 — Domain Reduced Feature Set With Total KG With Tuned CatBoost Candidate

Feature set:

```text
domain_reduced_with_total_kg
```

Selected model:

```text
gradient_boosting
```

Validation comparison:

```text
gradient_boosting validation MAE: 1877.156970
catboost_regularized validation MAE: 8768.076565
```

CatBoost validation results:

```text
catboost_regularized validation MAE: 8768.076565
catboost_regularized validation RMSE: 10140.714555
catboost_regularized validation R²: -17.270510
catboost_regularized validation MAPE: 47.390656%
catboost_regularized validation total deviation: 48.857839%
```

Decision:

```text
CatBoost rejected
Domain reduced feature set rejected
```

Reason:

CatBoost did not improve under the domain-reduced feature set. Gradient Boosting remained stronger.

## CatBoost decision summary

CatBoost was evaluated in two stages:

```text
0.2 Initial CatBoost candidate
0.3 Tuned CatBoost candidate
```

It was also evaluated across all available feature sets:

```text
full
vif_auto_post_only
reduced_without_total_kg
domain_reduced_with_total_kg
```

CatBoost did not outperform Gradient Boosting in any feature set.

Best CatBoost validation MAE observed:

```text
3109.107828
```

Best Gradient Boosting validation MAE under the production-safe full feature set:

```text
1693.194496
```

Best Gradient Boosting validation MAE under any tested feature set:

```text
1659.338997
```

However, the feature set with MAE `1659.338997` removes `total_kg`, so it is not selected as the production-safe final baseline.

The CatBoost decision is:

```text
Reject CatBoost for the current real-data post-installation forecasting task.
```

Reason:

```text
CatBoost was tested, tuned, and checked across feature sets, but validation evidence did not support selecting it.
```

## Final current model decision

The selected production-safe model remains:

```text
Full feature set + Gradient Boosting
```

The selected run family is:

```text
0.3 Full Feature Baseline With Tuned CatBoost Candidate
```

The selected model is:

```text
gradient_boosting
```

The selected feature set is:

```text
full
```

The full feature set is selected over the reduced feature challenger because:

* it preserves `total_kg`, a core production-volume feature
* it retains the full production and calendar signal set
* its validation MAE is very close to the reduced challenger
* it has stronger domain credibility
* it has better final holdout behavior than the reduced challenger
* it is more defensible for a production-style model

The reduced feature set without `total_kg` is documented as a validation challenger, not selected as the production-safe model.

## MLflow tracking

MLflow is used for intentional experiment tracking.

The local tracking backend is:

```text
sqlite:///mlflow.db
```

The experiment name is:

```text
Damavand Energy Forecasting
```

The logged run families are:

```text
0.1   Full Feature Baseline
0.1.1 VIF Auto Feature Set Diagnostic
0.1.2 Reduced Feature Challenger Without Total KG
0.1.3 Domain Reduced Feature Set With Total KG

0.2   Full Feature Baseline

0.3   Full Feature Baseline With Tuned CatBoost Candidate
0.3.1 VIF Auto Feature Set With Tuned CatBoost Candidate
0.3.2 Reduced Feature Set Without Total KG With Tuned CatBoost Candidate
0.3.3 Domain Reduced Feature Set With Total KG With Tuned CatBoost Candidate
```

The training script logs:

* run name
* feature set name
* feature count
* selected feature list
* selected model
* candidate model list
* split sizes
* validation metrics
* test metrics
* bootstrap confidence intervals
* model artifact
* Excel report artifact
* feature-list artifact
* diagnostic plot artifacts

Normal development runs do not log to MLflow:

```powershell
python -m src.train
```

Intentional tracked runs use:

```powershell
python -m src.train --log-mlflow
```

Feature-set runs use:

```powershell
python -m src.train --feature-set <feature_set_name> --log-mlflow
```

## Synthetic future-data extension

A synthetic 100-day future-data extension may be added later for portfolio demonstration.

This synthetic data should not be presented as real operational history.

It should be clearly labeled as synthetic and placed separately, for example:

```text
data/sample/damavand_synthetic_future_100d.csv
```

Synthetic data may be used for:

* pipeline demonstration
* API testing
* retraining workflow simulation
* dashboard demonstration
* larger-dataset stress testing

Synthetic data should not be used to claim improved real-world model accuracy.

The real model-selection benchmark remains based on the real post-installation train, validation, and test periods.

## Next modeling step

The next real model-family candidate is XGBoost.

The next model version should be:

```text
0.4 XGBoost Candidate Comparison
```

The XGBoost run should be compared against the current selected production-safe baseline:

```text
Full feature set + Gradient Boosting
```

The same validation-first discipline should be maintained:

```text
Use validation metrics for model selection.
Use test metrics only as final holdout diagnostics.
```

If XGBoost does not improve validation performance in a domain-credible way, the project should retain the current full-feature Gradient Boosting model.

## Current final conclusion

The current project conclusion is:

```text
The full-feature Gradient Boosting model is the selected production-safe baseline for the real post-installation forecasting task.
```

CatBoost conclusion:

```text
CatBoost was evaluated and tuned but rejected because validation performance was consistently weaker than Gradient Boosting across all tested feature sets.
```

Feature-set conclusion:

```text
The full feature set is retained for production-safe modeling because it preserves key production-volume information and provides the strongest domain-credible balance.
```

The next experimental extension is:

```text
XGBoost candidate comparison
```

A separate later portfolio extension may include:

```text
Synthetic future-data simulation
```
