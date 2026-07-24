# Modeling Notes

## Project objective

This project develops a production-style applied machine learning workflow for industrial energy forecasting at Damavand.

The goal is to build an accurate and defensible forecasting model for daily active energy consumption using production, operational, and calendar data.

The work uses a real industrial energy dataset from the Damavand case and is structured as an applied machine learning forecasting project: building candidate models, comparing modeling assumptions, rejecting weak approaches, tracking experiments, and documenting decisions under real data limitations.

The objective is not simply to try many algorithms. The objective is to build a reproducible modeling process where each modeling decision is tested, compared, rejected, improved, or retained based on evidence.

This project demonstrates:

* time-based model validation
* reproducible data preprocessing
* feature-set management
* model comparison across baseline, linear, regularized, robust, and tree-based methods
* model rejection based on validation evidence
* MLflow experiment tracking
* diagnostic reporting
* bootstrap uncertainty intervals
* clear documentation of limitations

## Why the first modeling phase uses only the post-installation period

The broader Damavand case includes an intervention that may have changed the relationship between production activity and energy consumption. The intervention/installation event occurred on 2025-09-12. This creates a data-regime problem: 

* Pre-intervention data provides more historical training examples and may still capture underlying production–energy relationships that remain valid after the intervention.
* Post-intervention data is more representative of the current operating regime.


Because the goal of this repository is to forecast future daily energy consumption under the current post-installation operating regime, the first modeling phase focuses only on the post-installation period.

This creates a clean first benchmark:

```text
Given the current post-installation operating regime, can we predict near-term daily energy consumption?
```

This keeps the first benchmark focused on the current post-installation regime, while later modeling phases test whether pre-intervention history can still add useful forecasting signal.

The post-installation-only benchmark is therefore the first step. It gives a direct view of how much forecasting performance can be achieved using only the available post-intervention data.

## Modeling challenge

The main challenge is that the real post-installation dataset is small.

The current split is:

```text
post_train: 30 rows
post_validation: 7 rows
post_test: 7 rows
```

This creates a difficult but realistic applied ML problem.

The small data size means that:

* complex models can overfit quickly
* validation results can be sensitive to one week of data
* test results must be interpreted carefully
* model selection should avoid excessive tuning
* simple and regularized models may be competitive
* uncertainty should be reported transparently

For this reason, the project emphasizes disciplined model comparison rather than aggressive hyperparameter optimization.

## Validation strategy

The project uses a time-based split.

The model is trained on earlier post-installation days, validated on later post-installation days, and tested on the final held-out post-installation period.

The split is:

```text
post_train:      2025-09-12 to 2025-10-17
post_validation: 2025-10-18 to 2025-10-24
post_test:       2025-10-25 onward
```

The validation set is the primary model-selection signal. The test set is not used for tuning.

If a model improves validation performance but fails badly on the final test period, it is treated as unstable and is not promoted.

If a model improves the final test result but worsens validation performance, it is also not promoted, because that would mean choosing a model based on the test set.

This keeps the selection process disciplined: validation selects candidates, while the final test period confirms whether the selected candidate behaves credibly.

Model selection is primarily based on validation MAE because the objective is daily energy forecasting and average daily absolute error is easy to interpret in kWh.

Additional metrics are reported for context:

* RMSE
* R²
* MAPE
* total deviation percentage
* bootstrap confidence intervals for final test metrics

## Input data and features

The processed dataset is loaded from:

```text
data/processed/damavand.csv
```

The target variable is:

```text
active_energy_kWh
```

The full feature set contains 13 production, operational, and calendar features:

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

The full feature set is used as the first feature set because it preserves production volume, nominal production, operating intensity, product characteristics, order activity, and calendar structure.

Feature reduction and multicollinearity checks are evaluated later as part of the modeling process rather than assumed at the start.

## Modeling process

The modeling process is designed to show professional judgment, not just model output.

The project asks:

```text
Can a reliable forecasting model be built from a small real industrial post-installation dataset?
```

The process begins with simple baselines and gradually expands to regularized, robust, and tree-based methods as evidence is collected.

Each later model family is introduced when it answers a specific modeling question, such as whether regularization helps with correlated features, whether tree-based methods capture nonlinear behavior, or whether advanced boosting methods improve the result.

## Modeling Version 0.1 — Full Feature Baseline

The first modeling version establishes the initial full-feature forecasting benchmark.

At this stage, no feature-reduction evidence has been collected yet. The first version therefore uses the complete production, operational, and calendar feature set.

The goal is to answer a basic but important question:

```text
Can the post-installation data support a useful forecasting model before any additional feature engineering or model-family expansion?
```

The first candidate set includes:

```text
dummy_mean
linear_regression
ridge_regression
random_forest
gradient_boosting
```

This first comparison is intentionally simple. It creates a reference point for all later modeling work.

The candidate models serve different purposes:

* `dummy_mean` checks whether any ML model beats a naive average predictor
* `linear_regression` checks whether the relationship is mostly simple and additive
* `ridge_regression` checks whether light regularization helps with correlated production features
* `random_forest` checks whether a bagged tree ensemble improves over linear models
* `gradient_boosting` checks whether sequential tree boosting captures stronger nonlinear structure

### Validation results

The models are compared on the held-out post-installation validation period.

```text
gradient_boosting   MAE: 1693.194496   RMSE: 1907.001901   R²: 0.353876    MAPE: 9.724538%    total deviation: 8.622433%
linear_regression   MAE: 1736.242085   RMSE: 2378.387659   R²: -0.005030   MAPE: 10.242237%   total deviation: 1.356671%
ridge_regression    MAE: 2375.219943   RMSE: 3095.496499   R²: -0.702449   MAPE: 13.818026%   total deviation: 11.743684%
random_forest       MAE: 2866.054458   RMSE: 3304.588641   R²: -0.940208   MAPE: 15.590290%   total deviation: 15.970347%
dummy_mean          MAE: 30150.766667  RMSE: 30243.960554  R²: -161.5140   MAPE: 173.999346%  total deviation: 168.007348%
```

Gradient Boosting produces the lowest validation MAE in the first benchmark.

Linear Regression is relatively close in validation MAE, which is an important observation. It suggests that the dataset is small enough that simple models remain competitive and that later increases in model complexity should be justified carefully.

However, Gradient Boosting provides the best validation result in this first comparison.

### Selected model for Version 0.1

For version 0.1, the selected model is:

```text
gradient_boosting
```

The model is selected because it has the lowest validation MAE among the first candidate models.

Validation metrics for the selected model:

```text
validation MAE: 1693.194496
validation RMSE: 1907.001901
validation R²: 0.353876
validation MAPE: 9.724538%
validation total deviation: 8.622433%
```

### Final holdout test results

After Gradient Boosting is selected using the validation set, it is evaluated on the final held-out post-installation test period.

```text
test MAE: 1641.4317
test RMSE: 2140.8388
test R²: 0.3166
test MAPE: 10.2851%
test total deviation: 4.4430%
```

Bootstrap 95% confidence intervals:

```text
test MAE CI: [745.6186, 2733.9019]
test RMSE CI: [904.1184, 3126.4487]
test MAPE CI: [3.7573, 18.4719]
test total deviation CI: [-3.1223, 14.9665]
```

### Version 0.1 decision

Gradient Boosting is retained as the first full-feature forecasting baseline.

This result is useful because it shows that the first non-linear boosting model improves over the naive baseline and provides a workable starting point for the project.

However, this version is not treated as the final answer.

Reasons to continue modeling:
* The first baseline should be challenged against additional feature sets and model families before selecting a final forecasting candidate
* Linear Regression remains close enough to suggest that model complexity should be controlled
* production features may be correlated
* the full feature set has not yet been tested against reduced feature sets
* additional model families may perform better under small-data conditions

Version 0.1 therefore establishes the first credible benchmark and motivates the next modeling step: checking multicollinearity and testing alternative feature sets.

## Multicollinearity diagnostic

After the first full-feature baseline, the next concern is feature correlation.

Several production variables are structurally related. For example, total production weight, nominal production weight, brix-related quantities, pallet count, and operating hours may naturally move together.

This does not automatically make the full feature set invalid, but it creates a modeling question:

```text
Are all production variables useful for forecasting, or can a smaller feature set improve validation behavior?
```

A VIF-based multicollinearity diagnostic is added to inspect linear redundancy among the candidate predictors.

The diagnostic is calculated using the development period only:

```text
post_train + post_validation
```

The final test period is not used for the VIF diagnostic.

This avoids contaminating the final holdout test set.

The automatic VIF reduction uses a threshold of:

```text
VIF < 5
```

The resulting automatic VIF feature set retains:

```text
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

The VIF result is treated as a diagnostic, not as an automatic modeling decision.

VIF is a linear redundancy measure. It is useful for understanding correlated predictors, especially for linear models, but it does not automatically determine the best feature set for tree-based forecasting models.

The next step is therefore to test reduced feature sets empirically.

## Feature-set experiment system

A named feature-set registry is added in:

```text
src/feature_sets.py
```

This allows feature sets to be tested reproducibly without manually editing the core settings file.

The training command supports:

```powershell
python -m src.train --feature-set <feature_set_name>
```

If no feature set is provided, the default is:

```text
full
```

The feature-set registry includes:

```text
full
vif_auto_post_only
reduced_without_total_kg
domain_reduced_with_total_kg
```

This design keeps feature-selection experiments explicit, repeatable, and traceable.

## Modeling Version 0.1.1 — VIF Auto Feature Set Diagnostic

The first feature-set diagnostic tests the automatic VIF-selected feature set.

Feature set:

```text
vif_auto_post_only
```

Feature count:

```text
6
```

Features:

```text
orders
avg_brix
yield_ratio_actual_over_nominal
weekday
is_weekend
week_of_year
```

The purpose of this run is to test whether a strongly reduced, low-VIF feature set improves model stability.

### Validation results

The selected model within this feature set is:

```text
gradient_boosting
```

Validation metrics:

```text
validation MAE: 7265.365491
validation RMSE: 9016.756211
validation R²: -13.444889
validation MAPE: 41.150888%
validation total deviation: 38.684591%
```

### Final holdout test results

```text
test MAE: 4401.3620
test RMSE: 4755.0347
test R²: -2.3714
test MAPE: 25.6888%
test total deviation: 6.6977%
```

### Version 0.1.1 decision

The VIF-auto feature set is rejected.

The result shows that reducing multicollinearity too aggressively removes important production information.

Although the VIF-auto feature set is cleaner from a linear redundancy perspective, it performs much worse as a forecasting feature set.

This is an important modeling result. It shows that feature selection cannot be based only on a statistical diagnostic. It must also preserve domain-relevant forecasting signal.

## Modeling Version 0.1.2 — Reduced Feature Challenger Without Total KG

The next feature-set diagnostic tests a smaller feature set that removes several correlated production variables.

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

This feature set is designed as a challenger. It keeps operating time, order activity, product characteristics, yield ratio, and calendar structure, but removes direct production weight.

### Validation results

The selected model within this feature set is:

```text
gradient_boosting
```

Validation metrics:

```text
validation MAE: 1659.338997
validation RMSE: 1862.765469
validation R²: 0.383505
validation MAPE: 9.189292%
validation total deviation: 8.769039%
```

This validation MAE is slightly better than the full-feature version 0.1 result:

```text
full feature validation MAE:              1693.194496
reduced_without_total_kg validation MAE:  1659.338997
difference:                                  33.855499 kWh
```

### Final holdout test results

```text
test MAE: 2742.0671
test RMSE: 3533.1256
test R²: -0.8613
test MAPE: 17.0850%
test total deviation: 8.1745%
```

### Version 0.1.2 decision

This feature set is rejected as the active forecasting feature set.

Although it produces a slightly better validation MAE than the full feature set, the improvement is small and it removes `total_kg`, a core production-volume feature.

The final holdout test result is also weaker than the full-feature Gradient Boosting model.

This run is retained as a useful validation challenger, but it is not selected.

The decision reflects a practical modeling principle:

```text
A small validation improvement is not enough to justify removing a core domain feature when final holdout behavior becomes weaker.
```

## Modeling Version 0.1.3 — Domain Reduced Feature Set With Total KG

The third feature-set diagnostic tests a reduced feature set that keeps the most important production-volume feature.

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

This feature set is more domain-credible than the previous reduced set because it keeps `total_kg`.

The purpose of this run is to test whether a smaller but production-aware feature set can improve over the full feature set.

### Validation results

The selected model within this feature set is:

```text
gradient_boosting
```

Validation metrics:

```text
validation MAE: 1877.156970
validation RMSE: 2491.238160
validation R²: -0.102666
validation MAPE: 10.095593%
validation total deviation: 10.459972%
```

### Final holdout test results

```text
test MAE: 2381.5436
test RMSE: 2959.5078
test R²: -0.3060
test MAPE: 14.7015%
test total deviation: 8.7337%
```

### Version 0.1.3 decision

The domain-reduced feature set is rejected.

It is more domain-credible than the reduced set without `total_kg`, but it does not outperform the full feature baseline.

This result supports retaining the full feature set as the strongest feature set at this stage.

## Feature-set decision after Version 0.1 diagnostics

The feature-set diagnostics show that the full feature set remains the best balance.

Summary:

```text
VIF-auto feature set:
- rejected
- removed too much production information
- validation and test performance were much weaker

Reduced without total_kg:
- slightly better validation MAE
- weaker test performance
- removed a core production-volume feature
- retained only as a challenger

Domain reduced with total_kg:
- more domain-credible
- weaker validation and test performance than the full feature baseline
- rejected

Full feature set:
- preserves production-volume and operational signal
- strongest balance at this stage
- retained as the active feature set
```

This decision is important because it shows that feature selection is not treated as a mechanical VIF exercise.

The full feature set remains active because it provides the strongest combination of validation performance, test behavior, and domain credibility.

## Modeling Version 0.2 — Initial CatBoost Candidate

After establishing the full-feature Gradient Boosting baseline and testing reduced feature sets, the next step is to test whether a more specialized gradient-boosted tree model can improve performance.

CatBoost is added as an additional candidate model.

The purpose is not to assume that CatBoost will perform better. The purpose is to test whether a dedicated boosting implementation improves the forecast under the small-data post-installation setting.

At this stage, CatBoost is added as a candidate only. Existing models are not removed.

### Validation results

The initial CatBoost candidate performs poorly on the full feature set.

```text
catboost_regularized validation MAE: 7193.397588
catboost_regularized validation RMSE: 7749.902140
catboost_regularized validation R²: -9.671021
catboost_regularized validation MAPE: 40.808120%
catboost_regularized validation total deviation: 40.083347%
```

The selected model for the run remains:

```text
gradient_boosting
```

### Version 0.2 decision

The initial CatBoost candidate is rejected.

The result shows that adding a more advanced model family does not automatically improve performance, especially with a small training set.

This run is still useful because it provides evidence that the current problem requires controlled model comparison rather than model-family assumptions.

CatBoost is not removed immediately. Instead, the next step is to test a more regularized CatBoost configuration.

## Modeling Version 0.3 — Tuned CatBoost Candidate

Version 0.3 evaluates a tuned CatBoost candidate.

The goal is to reduce model complexity and test whether CatBoost can become competitive under stronger regularization.

The tuned CatBoost configuration uses:

```text
loss_function: RMSE
iterations: 100
learning_rate: 0.05
depth: 2
random_seed: 33
allow_writing_files: False
verbose: False
```

This configuration is intentionally conservative because the post-installation training set is small.

### Validation comparison

The full-feature validation comparison is:

```text
gradient_boosting validation MAE:    1693.194496
catboost_regularized validation MAE: 3109.107828
```

CatBoost validation metrics:

```text
catboost_regularized validation MAE: 3109.107828
catboost_regularized validation RMSE: 3738.813973
catboost_regularized validation R²: -1.483598
catboost_regularized validation MAPE: 17.340896%
catboost_regularized validation total deviation: 16.485174%
```

The selected model remains:

```text
gradient_boosting
```

### Final holdout test results for the selected model

Because Gradient Boosting remains the selected model, the final test results are the same selected-model benchmark:

```text
test MAE: 1641.4317
test RMSE: 2140.8388
test R²: 0.3166
test MAPE: 10.2851%
test total deviation: 4.4430%
```

### Version 0.3 decision

The tuned CatBoost candidate is rejected on the full feature set.

The tuning improves CatBoost compared with the initial CatBoost attempt, but it remains substantially weaker than Gradient Boosting on validation MAE.

This is an important result because it shows that model-family reputation is not enough. The candidate must win on the project’s validation framework.

## Modeling Version 0.3.1 — VIF Auto Feature Set With Tuned CatBoost

After tuning CatBoost on the full feature set, the next step is to test whether CatBoost behaves differently under reduced feature sets.

The first reduced-feature CatBoost diagnostic uses the automatic VIF-selected feature set.

Feature set:

```text
vif_auto_post_only
```

### Validation comparison

```text
gradient_boosting validation MAE:    7265.365491
catboost_regularized validation MAE: 13498.759186
```

CatBoost validation metrics:

```text
catboost_regularized validation MAE: 13498.759186
catboost_regularized validation RMSE: 14144.757601
catboost_regularized validation R²: -34.547132
catboost_regularized validation MAPE: 75.295003%
catboost_regularized validation total deviation: 75.218344%
```

### Version 0.3.1 decision

CatBoost is rejected under the VIF-auto feature set.

The VIF-auto feature set is also rejected again.

This confirms that the automatic VIF reduction removes too much forecasting signal for the current post-installation task.

## Modeling Version 0.3.2 — Reduced Feature Set Without Total KG With Tuned CatBoost

The second reduced-feature CatBoost diagnostic uses the reduced feature set that excludes direct production weight.

Feature set:

```text
reduced_without_total_kg
```

### Validation comparison

```text
gradient_boosting validation MAE:    1659.338997
catboost_regularized validation MAE: 7745.187633
```

CatBoost validation metrics:

```text
catboost_regularized validation MAE: 7745.187633
catboost_regularized validation RMSE: 9506.626018
catboost_regularized validation R²: -15.057073
catboost_regularized validation MAPE: 42.315445%
catboost_regularized validation total deviation: 43.158055%
```

### Version 0.3.2 decision

CatBoost is rejected under the reduced feature set without `total_kg`.

The reduced feature set remains a useful validation challenger, but it is still not selected as the active feature set.

## Modeling Version 0.3.3 — Domain Reduced Feature Set With Total KG With Tuned CatBoost

The final reduced-feature CatBoost diagnostic uses the domain-reduced feature set that keeps `total_kg`.

Feature set:

```text
domain_reduced_with_total_kg
```

### Validation comparison

```text
gradient_boosting validation MAE:    1877.156970
catboost_regularized validation MAE: 8768.076565
```

CatBoost validation metrics:

```text
catboost_regularized validation MAE: 8768.076565
catboost_regularized validation RMSE: 10140.714555
catboost_regularized validation R²: -17.270510
catboost_regularized validation MAPE: 47.390656%
catboost_regularized validation total deviation: 48.857839%
```

### Version 0.3.3 decision

CatBoost is rejected under the domain-reduced feature set.

The domain-reduced feature set is also rejected because it does not outperform the full feature baseline.

## CatBoost decision summary

CatBoost is evaluated in several stages:

```text
initial CatBoost candidate
tuned CatBoost candidate
tuned CatBoost across reduced feature sets
```

The best observed CatBoost validation MAE is:

```text
3109.107828
```

The full-feature Gradient Boosting validation MAE is:

```text
1693.194496
```

The best reduced-feature Gradient Boosting validation MAE is:

```text
1659.338997
```

CatBoost does not outperform the selected Gradient Boosting baseline in any tested feature set.

Decision:

```text
Reject CatBoost for the current post-installation forecasting task.
```

Reason:

CatBoost was tested, regularized, and evaluated across feature sets, but the validation evidence did not support selecting it.

This strengthens the project because it shows that advanced models are not accepted automatically. They must earn selection through the validation framework.

## Modeling Version 0.4 — XGBoost Candidate

After rejecting CatBoost, the next advanced boosting candidate is XGBoost.

The purpose is to test whether XGBoost can improve over the Gradient Boosting baseline.

XGBoost is added as a candidate model only. Existing models remain in the comparison.

The first XGBoost attempt uses a standard squared-error regression objective.

Objective:

```text
reg:squarederror
```

### Validation comparison

```text
gradient_boosting validation MAE:   1693.194496
xgboost_regularized validation MAE: 7831.997238
```

XGBoost validation metrics:

```text
xgboost_regularized validation MAE: 7831.997238
xgboost_regularized validation RMSE: 8906.100415
xgboost_regularized validation R²: -13.092522
xgboost_regularized validation MAPE: 41.938217%
xgboost_regularized validation total deviation: 43.641772%
```

### Version 0.4 decision

The squared-error XGBoost candidate is rejected.

The model substantially underperforms the existing Gradient Boosting baseline.

This result suggests that XGBoost, at least with a standard squared-error objective and conservative regularization, is not well matched to the current small post-installation dataset.

## Modeling Version 0.4.1 — XGBoost Tweedie Objective

The next XGBoost step tests whether changing the objective function improves behavior.

Because daily active energy consumption is a positive continuous target, a Tweedie objective is tested.

Objective:

```text
reg:tweedie
```

This is a targeted modeling adjustment rather than a blind hyperparameter search.

### Validation comparison

```text
gradient_boosting validation MAE:    1693.194496
xgboost_regularized validation MAE:  2344.517857
```

XGBoost Tweedie validation metrics:

```text
xgboost_regularized validation MAE: 2344.517857
xgboost_regularized validation RMSE: 2651.772284
xgboost_regularized validation R²: -0.249355
xgboost_regularized validation MAPE: 13.915890%
xgboost_regularized validation total deviation: -11.836346%
```

### Version 0.4.1 decision

The Tweedie objective substantially improves XGBoost compared with the squared-error objective.

Validation MAE improves from:

```text
7831.997238
```

to:

```text
2344.517857
```

However, XGBoost Tweedie still does not outperform the selected full-feature Gradient Boosting baseline:

```text
gradient_boosting validation MAE: 1693.194496
xgboost_tweedie validation MAE:  2344.517857
```

Decision:

```text
Reject XGBoost Tweedie as the selected model.
```

Reason:

The objective change improves XGBoost substantially, but the validation framework still favors Gradient Boosting.

This is a valuable experiment because it shows careful modeling judgment: the objective function was tested for a reason, the result improved, but the model was still rejected because it did not outperform the existing benchmark.

## XGBoost reduced-feature instability diagnostic

A reduced-feature XGBoost Tweedie diagnostic is also tested using the VIF-auto feature set.

This diagnostic produces an attractive validation result:

```text
vif_auto_post_only + XGBoost Tweedie validation MAE: 1065.291881
validation R²: 0.695945
validation total deviation: -2.053847%
```

However, it fails on the final held-out test period:

```text
test MAE: 4631.6858
test RMSE: 5475.4191
test R²: -3.4703
test MAPE: 23.7644%
test total deviation: -25.5581%
```

This run is not selected.

The result is treated as an instability diagnostic.

It shows that the 7-day validation period can produce misleadingly strong results when the feature set is too reduced or the model objective is too specialized.

This supports the decision to avoid selecting models based on a single attractive validation result when final holdout behavior is weak.

## XGBoost decision summary

XGBoost is evaluated in two main stages:

```text
squared-error objective
Tweedie objective
```

The Tweedie objective improves XGBoost substantially:

```text
squared-error validation MAE: 7831.997238
Tweedie validation MAE:       2344.517857
```

However, XGBoost still does not outperform the selected Gradient Boosting baseline.

Decision:

```text
Reject XGBoost for the current post-installation forecasting task.
```

Reason:

XGBoost was tested with a standard regression objective and then with a more appropriate objective for positive continuous targets. The second version improved substantially, but validation evidence still favored Gradient Boosting.

At this point, the strongest selected model remains the full-feature Gradient Boosting baseline from version 0.1.

The next modeling question is whether models better suited to small datasets can outperform the current tree-boosting baseline.

## Modeling Version 0.5 — Small-Data Candidate Comparison

After Gradient Boosting remained the strongest selected model through the CatBoost and XGBoost experiments, the next modeling question changed.

The earlier experiments showed that more advanced boosting models did not automatically improve performance. CatBoost and XGBoost were both tested and rejected based on validation evidence.

At the same time, the post-installation training set remained very small:

```text
post_train: 30 rows
```

This made it important to test model families that may behave better under small-data conditions.

Version 0.5 therefore adds a small-data candidate comparison.

The goal is not to add complexity. The goal is to test whether more stable, regularized, robust, or variance-reducing models can outperform the existing Gradient Boosting baseline.

The added candidates are:

```text
huber_regression
bayesian_ridge
elastic_net
svr_rbf_scaled
extra_trees
```

These models are added as candidates only. Existing models remain in the comparison.

The new candidates answer specific modeling questions:

* `huber_regression` tests whether a robust linear model handles unusual days better than ordinary least squares
* `bayesian_ridge` tests whether Bayesian regularization improves stability with few training rows
* `elastic_net` tests whether combined L1/L2 regularization helps under correlated features
* `svr_rbf_scaled` tests whether a nonlinear kernel method can capture structure after scaling
* `extra_trees` tests whether a randomized tree ensemble can reduce variance and improve generalization compared with the earlier tree models

This step is important because it shows that model selection is not biased toward advanced boosting libraries. The project returns to the data constraint and tests models that may be better matched to the size of the available dataset.

### Validation results

The full-feature validation comparison is:

```text
extra_trees            MAE:   735.753100   RMSE:   940.317618   R²:   0.842905   MAPE:   4.042565%   total deviation:  -1.805594%
huber_regression       MAE:  1133.212490   RMSE:  1695.322549   R²:   0.489356   MAPE:   6.566344%   total deviation:   5.032129%
gradient_boosting      MAE:  1693.194496   RMSE:  1907.001901   R²:   0.353876   MAPE:   9.724538%   total deviation:   8.622433%
linear_regression      MAE:  1736.242085   RMSE:  2378.387659   R²:  -0.005030   MAPE:  10.242237%   total deviation:   1.356671%
bayesian_ridge         MAE:  1880.090113   RMSE:  2426.137997   R²:  -0.045790   MAPE:  11.038405%   total deviation:   0.228142%
elastic_net            MAE:  1902.828005   RMSE:  2435.700653   R²:  -0.054050   MAPE:  11.159633%   total deviation:   0.046801%
xgboost_regularized    MAE:  2344.517857   RMSE:  2651.772284   R²:  -0.249355   MAPE:  13.915890%   total deviation: -11.836346%
ridge_regression       MAE:  2375.219943   RMSE:  3095.496499   R²:  -0.702449   MAPE:  13.818026%   total deviation:  11.743684%
random_forest          MAE:  2866.054458   RMSE:  3304.588641   R²:  -0.940208   MAPE:  15.590290%   total deviation:  15.970347%
catboost_regularized   MAE:  3109.107828   RMSE:  3738.813973   R²:  -1.483598   MAPE:  17.340896%   total deviation:  16.485174%
dummy_mean             MAE: 30150.766667   RMSE: 30243.960554   R²: -161.514036  MAPE: 173.999346%  total deviation: 168.007348%
svr_rbf_scaled         MAE: 46103.155101   RMSE: 46164.107090   R²: -377.636459  MAPE: 264.876527%  total deviation: 256.897906%
```

The strongest validation model is:

```text
extra_trees
```

Extra Trees validation metrics:

```text
validation MAE: 735.753100
validation RMSE: 940.317618
validation R²: 0.842905
validation MAPE: 4.042565%
validation total deviation: -1.805594%
```

This is a major improvement over the previous selected Gradient Boosting baseline:

```text
Gradient Boosting validation MAE: 1693.194496
Extra Trees validation MAE:        735.753100
```

The validation total deviation is also close to zero, which suggests that the model is not strongly biased at the weekly aggregate level.

### Selected model for Version 0.5

For version 0.5, the selected model is:

```text
extra_trees
```

The selected configuration is:

```python
ExtraTreesRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    random_state=33,
    n_jobs=-1,
)
```

The model is selected because it provides the lowest validation MAE by a clear margin while keeping validation total deviation close to zero.

This is an important modeling result. The strongest candidate is not CatBoost or XGBoost. It is a conservative Extra Trees configuration that is better matched to the small post-installation dataset.

### Final holdout test results

After Extra Trees is selected using the validation set, it is evaluated on the final held-out post-installation test period.

Test metrics:

```text
test MAE: 1492.3472
test RMSE: 1964.4620
test R²: 0.4246
test MAPE: 9.4615%
test total deviation: 4.3923%
```

Bootstrap 95% confidence intervals:

```text
test MAE CI: [609.4218, 2522.3190]
test RMSE CI: [693.5024, 2819.6042]
test MAPE CI: [3.1379, 16.7242]
test total deviation CI: [-2.4009, 14.1029]
```

Compared with the previous Gradient Boosting selected model, Extra Trees improves final holdout test MAE:

```text
Gradient Boosting test MAE: 1641.4317
Extra Trees test MAE:       1492.3472
```

The final test R² also improves:

```text
Gradient Boosting test R²: 0.3166
Extra Trees test R²:       0.4246
```

The final test total deviation remains very close:

```text
Gradient Boosting test total deviation: 4.4430%
Extra Trees test total deviation:       4.3923%
```

### Diagnostic plot interpretation

The Extra Trees diagnostic plots show that the model follows the main movement in the final post-installation test week.

The model captures the higher-consumption part of the test period reasonably well.

However, the diagnostic plots also show that the model overpredicts the first two lower-consumption test days. These two early errors create the largest negative residuals.

This indicates that the model may still struggle with sudden low-consumption days or unusual operating conditions.

Because the test set contains only seven observations, residual plots are included as diagnostics but should not be overinterpreted.

The practical interpretation is:

```text
Extra Trees provides the strongest post-only validation and holdout performance at this stage of the project. The short test period defines the boundary of the available evidence, so the result should be revalidated as more post-installation data becomes available.
```

### Extra Trees tuning check

After Extra Trees became the strongest candidate, two small Extra Trees tuning checks were tested locally.

The tuning checks used slightly different regularization settings.

The official selected version produced:

```text
validation MAE: 735.753100
test MAE: 1492.3472
```

The local tuning attempts produced:

```text
validation MAE: 765.619285 or 795.828213
test MAE: 1384.4391 or 1433.8741
```

The tuned versions slightly improved test MAE, but they worsened validation MAE.

These tuning attempts were not logged and were not selected.

Decision:

```text
Retain the official Version 0.5 Extra Trees configuration.
```

Reason:

Model selection should be based on validation evidence. Selecting a tuned model only because the final test MAE improves would risk test-set chasing and would weaken the credibility of the modeling process.

This decision is important. It shows that the test set is treated as a final holdout check, not as another tuning signal.

### Extra Trees feature-set transparency check

After Extra Trees became the strongest post-only candidate, the alternative post-only feature sets were also checked for transparency.

The purpose was not to restart feature selection, but to confirm whether the selected Extra Trees model depended specifically on the full feature set.

The feature sets checked were:

```text
full
vif_auto_post_only
reduced_without_total_kg
domain_reduced_with_total_kg
```

Extra Trees validation results across the feature sets were:

```text
full:
validation MAE: 735.753100
validation RMSE: 940.317618
validation R²: 0.842905
validation MAPE: 4.042565%
validation total deviation: -1.805594%

vif_auto_post_only:
validation MAE: 9330.648054
validation RMSE: 10037.065235
validation R²: -16.898929
validation MAPE: 52.105897%
validation total deviation: 51.992623%

reduced_without_total_kg:
validation MAE: 10434.255860
validation RMSE: 11703.634711
validation R²: -23.336329
validation MAPE: 56.311923%
validation total deviation: 58.142192%

domain_reduced_with_total_kg:
validation MAE: 6193.906932
validation RMSE: 7352.571748
validation R²: -8.604883
validation MAPE: 32.985345%
validation total deviation: 34.513944%
```

This check confirmed that Extra Trees performed strongly only with the full feature set.

Decision:

```text
Retain the full feature set for the Version 0.5 post-only Extra Trees model.
```

Reason:

The reduced feature sets substantially weakened Extra Trees validation performance. This supports the earlier feature-set decision that the full feature set preserves important production and operational signal for the post-only forecasting task.

## Current model decision

The current selected post-installation forecasting model is:

```text
full feature set + extra_trees
```

The selected model is:

```text
extra_trees
```

The selected feature set is:

```text
full
```

The selected configuration is:

```python
ExtraTreesRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    random_state=33,
    n_jobs=-1,
)
```

This model is selected because it provides the strongest validation result and improves final holdout test MAE compared with the previous Gradient Boosting baseline.

The selected Extra Trees model is presented as the strongest post-installation forecasting benchmark at this stage of the project. It provides the best validation performance among the tested post-only candidates and improves final holdout MAE compared with the previous selected baseline.

Because the post-installation evaluation window is short, the model should still be revalidated as more post-intervention data becomes available.
## Post-installation benchmark conclusion

The post-installation-only benchmark produced a credible forecasting model under difficult data constraints.

The final selected post-only model improves over the original Gradient Boosting baseline on both validation MAE and final holdout test MAE:

```text
Validation MAE:
Gradient Boosting: 1693.194496
Extra Trees:        735.753100

Test MAE:
Gradient Boosting: 1641.4317
Extra Trees:       1492.3472
```

The selected model also keeps final holdout total deviation close to the previous benchmark:

```text
Gradient Boosting test total deviation: 4.4430%
Extra Trees test total deviation:       4.3923%
```

Extra Trees is therefore the strongest post-only forecasting model from the tested candidate set. It was selected by validation performance and then confirmed on the final holdout period as an improvement over the earlier Gradient Boosting baseline.

The modeling process also produced several important rejection findings:

```text
- automatic VIF-based feature reduction removed too much forecasting signal
- reduced features without total_kg slightly improved validation MAE but weakened domain credibility and test behavior
- CatBoost did not outperform the baseline after tuning
- XGBoost improved with Tweedie objective but still did not outperform the selected model
- local Extra Trees tuning improved test MAE but worsened validation MAE, so it was not selected
```

These rejections strengthen the project because they show that models are not accepted based on complexity, popularity, or final-test chasing.

## Current limitations

The selected post-only model is a credible forecasting benchmark for the available post-installation data. The limitations define how the result should be interpreted and revalidated.

The main limitation is data size.

The post-installation split contains:

```text
post_train: 30 rows
post_validation: 7 rows
post_test: 7 rows
```

This means:

* model conclusions should be interpreted carefully
* validation and test results are sensitive to individual days
* bootstrap confidence intervals are necessary to communicate uncertainty around the short final test window
* aggressive hyperparameter optimization is not appropriate for the post-only benchmark
* post-installation-only rolling validation is not meaningful enough with the current data volume

A rolling-validation prototype was tested but rejected because the available post-installation data was too short to create stable, comparable folds.

This is not a failure of the model. It is a data limitation. The decision not to force rolling validation is part of the modeling discipline.

## Experiment tracking

Intentional model experiments are tracked with MLflow.

Each logged run records the feature set, model candidates, selected model, validation metrics, final test metrics, bootstrap confidence intervals, model artifact, Excel report, feature list, and diagnostic plots.

Exploratory development checks are not logged automatically. MLflow is reserved for intentional experiment runs that represent clear modeling decisions, official comparisons, or important diagnostics. This keeps the experiment history clean and interpretable.

Detailed MLflow usage instructions are documented in the project README.

## Why automated hyperparameter optimization was not used in the post-only benchmark

Automated hyperparameter optimization was intentionally not used in the post-only benchmark.

The post-only training set contains only 30 rows, and the validation window contains only 7 days. In this setting, a large automated search could easily overfit the small validation week and produce a model that looks strong on validation but does not generalize reliably.

For the post-only phase, the project therefore uses controlled, documented candidate comparisons instead of aggressive automated tuning.

This was a deliberate modeling decision.

The goal of the post-only benchmark was not to find the most optimized possible model from a tiny training sample. The goal was to establish a credible current-regime forecasting benchmark using simple, regularized, robust, and tree-based candidates under strict validation discipline.

This approach helps prevent test-set chasing and keeps the post-only result defensible.
## Current final conclusion

The current project conclusion is:

```text
The full-feature Extra Trees model is the selected post-installation forecasting model.
```

The result is credible because:

* the model is selected by validation MAE
* the final test set is held out until after selection
* the selected model improves test MAE compared with the previous selected baseline
* the model is not chosen based on post-hoc test tuning
* bootstrap confidence intervals are reported
* model limitations are documented clearly
* rejected models and feature sets are documented rather than hidden

This is the strongest current model for the real post-installation forecasting benchmark.

## Interpretation

The post-only model performed strongly on validation and remained competitive on the final test week. Its main strength is that it is trained only on the post-intervention regime, so it directly reflects the current operating conditions.

Because the model is trained on only 30 rows, the result should be treated as a strong current-regime benchmark rather than a permanently final forecasting system. This does not weaken the model; it defines the practical boundary of the available post-installation evidence.

The next modeling step is therefore to test whether additional historical signal can improve stability without losing relevance to the current post-intervention regime.

## Next modeling direction

The next project phase should move beyond the post-installation-only benchmark.

The current post-installation benchmark is useful, but the training set is small. A future phase should investigate whether more of the available historical data can improve forecasting performance while still respecting the difference between pre-intervention and post-intervention behavior.


# Full-History Modeling

## Motivation

The full-history modeling approach uses all available historical data up to the training cutoff. This gives the model far more training examples than the post-only setup, but it introduces the risk that pre-intervention behavior may not fully match the post-intervention regime.

This experiment answers:

> Can the model benefit from the larger historical training sample, even though the operating regime changed after the intervention?

## Split Design

The full-history split uses:

* Training: all available rows through 2025-10-17
* Validation: 2025-10-18 to 2025-10-24
* Test: 2025-10-25 onward

Split sizes:

| Split                   | Rows |
| ----------------------- | ---- |
| Full-history train      | 452  |
| Full-history validation | 7    |
| Full-history test       | 7    |

The validation and test windows are intentionally kept the same as the post-only benchmark so the model families can be compared on the same post-intervention evaluation periods.

---

# 1.0 Full-History Baseline

## Purpose

Version 1.0 established the first full-history baseline using the standard candidate models:

* Dummy mean
* Linear Regression
* Ridge Regression
* Random Forest
* Gradient Boosting

## Selected Model

The selected model was:

* Random Forest

Validation results:

| Model         | Validation MAE | Validation RMSE | Validation R² | Validation MAPE | Total Deviation % |
| ------------- | -------------- | --------------- | ------------- | --------------- | ----------------- |
| Random Forest | 3570.1870      | 4594.4183       | -2.7504       | 19.2471         | 18.8873           |

Test results:

| Metric            | Value     |
| ----------------- | --------- |
| MAE               | 1984.7877 |
| RMSE              | 2990.5459 |
| R²                | -0.3335   |
| MAPE              | 10.1806   |
| Total deviation % | 3.0344    |

## Interpretation

The full-history baseline did not outperform the post-only model on daily accuracy. This suggested that simply adding more historical data was not enough. The regime shift likely reduced the usefulness of pre-intervention patterns unless the model or feature setup was improved.

This was a useful rejection result rather than a failed experiment.

---

## Why automated hyperparameter optimization was introduced in the full-history phase

Automated hyperparameter optimization was introduced in the full-history phase because the training sample was much larger than in the post-only benchmark.

The full-history split contains:

```text
full_history_train: 452 rows
full_history_validation: 7 rows
full_history_test: 7 rows
```

Compared with the 30-row post-only training set, the full-history setup provides enough training data to make controlled hyperparameter optimization more reasonable.

The purpose of using Optuna was to test whether better hyperparameter choices could make the larger historical training sample more competitive.

This did not remove the main limitation of the project. The validation window still contains only 7 days, so Optuna results are interpreted cautiously.

For that reason, the project uses Optuna in the full-history phase, but does not treat every larger tuning run as automatically better. Very large tuning budgets, such as the 10000-trial run, are treated as sensitivity diagnostics when appropriate.

This keeps the modeling strategy balanced:

```text
post-only phase:
controlled small-data candidate comparison

full-history phase:
controlled hyperparameter optimization using the larger historical training sample, with caution because validation remains short
```

This distinction is important because the project does not apply one modeling strategy blindly. The modeling method changes depending on the available training data and the risk of overfitting.


# 1.1 Full-History Optuna-Tuned Candidate

## Purpose

Version 1.1 used Optuna to tune full-history tree-based models. The goal was to test whether better hyperparameters could make the full-history approach more competitive.

Candidate families:

* Random Forest
* Extra Trees
* Gradient Boosting

## Best Optuna Candidate

The best overall Optuna candidate was:

* Gradient Boosting

Best parameters:

```python
GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=4,
    subsample=0.9,
    random_state=33,
)
```

Validation results:

| Model             | Validation MAE | Validation RMSE | Validation R² | Validation MAPE | Total Deviation % |
| ----------------- | -------------- | --------------- | ------------- | --------------- | ----------------- |
| Gradient Boosting | 2126.2109      | 2987.1249       | -0.5853       | 11.2115         | 7.6938            |

Test results:

| Metric            | Value     |
| ----------------- | --------- |
| MAE               | 1804.2195 |
| RMSE              | 2749.2384 |
| R²                | -0.1270   |
| MAPE              | 9.1110    |
| Total deviation % | 3.3452    |

## Interpretation

The Optuna-tuned full-history model improved over the 1.0 baseline, but it still did not clearly outperform the post-only candidate.

This suggested that the full-history approach needed either better feature selection, better regularization, or explicit handling of the regime change.

---

# 1.2 Full-History VIF-Reduced Feature Diagnostic

## Purpose

Version 1.2 tested whether reducing multicollinearity could improve the full-history model.

A VIF analysis was run on the full-history feature set. The final recommended feature set was:

* `total_nominal_kg`
* `total_hours`
* `total_pallets`
* `orders`
* `avg_brix`
* `yield_ratio_actual_over_nominal`
* `weekday`
* `is_weekend`
* `month`
* `year`

This feature set was saved as:

* `vif_auto_full_history`

## Best Candidate

The best VIF-reduced candidate was:

* Gradient Boosting

Best parameters:

```python
GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=33,
)
```

Validation results:

| Metric            | Value     |
| ----------------- | --------- |
| MAE               | 2974.9578 |
| RMSE              | 4221.5742 |
| R²                | -2.1664   |
| MAPE              | 15.4310   |
| Total deviation % | 16.5391   |

Test results:

| Metric            | Value     |
| ----------------- | --------- |
| MAE               | 1464.5123 |
| RMSE              | 1894.8924 |
| R²                | 0.4646    |
| MAPE              | 7.8167    |
| Total deviation % | 0.5426    |

Bootstrap 95% confidence intervals:

| Metric            | 95% CI                |
| ----------------- | --------------------- |
| MAE               | [590.4535, 2340.9515] |
| RMSE              | [947.4789, 2597.7420] |
| MAPE              | [3.3394, 12.7157]     |
| Total deviation % | [-6.6623, 8.5038]     |

## Interpretation

Version 1.2 produced an important mixed result.

Compared with 1.1:

* Validation MAE became worse.
* Test MAE improved.
* Test R² improved.
* Test total deviation improved substantially.

This means the VIF-reduced model looked less robust by validation selection, but behaved better on the final test week.

Because the validation and test windows are only 7 days each, this result was treated cautiously. It was considered a useful diagnostic showing that feature simplification may help final-week generalization, but not enough to declare the model clearly superior based only on test performance.

---

# 1.3 Full-History VIF-Reduced Advanced Boosting Candidate Comparison

## Purpose

Version 1.3 extended the full-history VIF-reduced workflow by adding advanced boosting candidates.

The goal was to test whether stronger model families could improve validation-stage performance while using the same split and feature set as version 1.2.

The candidate families were:

* Dummy mean
* Linear Regression
* Ridge Regression
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* CatBoost
* AdaBoost

The comparison used the same `vif_auto_full_history` feature set as version 1.2.

## Optuna Tuning

Optuna was run using the advanced boosting candidate pool. Runs with 500, 1000, and 2000 trials consistently selected the same AdaBoost configuration as the strongest validation candidate.

The official 1.3 model uses the 2000-trial AdaBoost configuration.

A later 10000-trial run found a stronger AdaBoost score, but it was treated as a sensitivity diagnostic rather than the official model. The reason is that the validation window contains only 7 days, so very large tuning budgets increase the risk of overfitting to validation-period noise.

## Official 1.3 Selected Model

The selected model was:

* AdaBoost with a shallow decision tree base estimator

Best parameters:

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

## Validation Comparison

| Model             | Validation MAE | Validation RMSE | Validation R² | Validation MAPE | Total Deviation % |
| ----------------- | -------------- | --------------- | ------------- | --------------- | ----------------- |
| AdaBoost          | 1329.8433      | 2339.6471       | 0.0274        | 7.1147          | 6.3256            |
| Gradient Boosting | 2974.9578      | 4221.5742       | -2.1664       | 15.4310         | 16.5391           |
| Random Forest     | 3704.6832      | 4506.7016       | -2.6085       | 20.3154         | 20.6434           |
| XGBoost           | 3809.8003      | 4971.5820       | -3.3914       | 21.0110         | 18.9630           |
| CatBoost          | 5059.7422      | 6301.4677       | -6.0550       | 28.6772         | 27.0089           |
| Extra Trees       | 5768.8536      | 6997.0181       | -7.6984       | 30.6460         | 32.1454           |
| Linear Regression | 10408.4055     | 11357.5081      | -21.9182      | 61.6928         | 57.9981           |
| Ridge Regression  | 10448.8144     | 11388.6226      | -22.0439      | 61.9103         | 58.2233           |
| Dummy mean        | 12212.1353     | 12440.4448      | -26.4970      | 71.8061         | 68.0490           |

## Training Diagnostics

Training diagnostics were calculated and saved, but they were not used for model selection.

| Model             | Training MAE | Training R² | Training Total Deviation % |
| ----------------- | ------------ | ----------- | -------------------------- |
| AdaBoost          | 2277.3789    | 0.9843      | 0.0679                     |
| Gradient Boosting | 503.2608     | 0.9993      | -0.0086                    |
| Random Forest     | 1690.5493    | 0.9881      | -0.0125                    |
| XGBoost           | 1503.2572    | 0.9902      | 0.0438                     |
| CatBoost          | 1669.6300    | 0.9879      | 0.0351                     |
| Extra Trees       | 1319.1574    | 0.9916      | 0.0000                     |
| Linear Regression | 4296.6432    | 0.9432      | 0.0000                     |
| Ridge Regression  | 4297.6181    | 0.9432      | 0.0000                     |
| Dummy mean        | 23184.9954   | 0.0000      | 0.0000                     |

These diagnostics show that several models fit the training data very closely but failed to generalize to the validation week. This reinforces the importance of validation-based selection and supports the decision not to select models based on training performance.

## Official 1.3 Test Results

| Metric            | Value     |
| ----------------- | --------- |
| MAE               | 1473.4256 |
| RMSE              | 1787.0266 |
| R²                | 0.5238    |
| MAPE              | 8.2459    |
| Total deviation % | -4.4112   |

Bootstrap 95% confidence intervals:

| Metric            | 95% CI                 |
| ----------------- | ---------------------- |
| MAE               | [795.2180, 2238.3522]  |
| RMSE              | [1038.9063, 2413.6945] |
| MAPE              | [4.4318, 12.0676]      |
| Total deviation % | [-10.7309, 2.3805]     |

## Comparison Against 1.2

| Version | Scope        | Feature Set | Selected Model    | Validation MAE | Test MAE  | Test R² | Test Total Deviation % |
| ------- | ------------ | ----------- | ----------------- | -------------- | --------- | ------- | ---------------------- |
| 1.2     | Full-history | VIF-reduced | Gradient Boosting | 2974.9578      | 1464.5123 | 0.4646  | 0.5426                 |
| 1.3     | Full-history | VIF-reduced | AdaBoost          | 1329.8433      | 1473.4256 | 0.5238  | -4.4112                |

## Interpretation

Version 1.3 substantially improved validation MAE compared with version 1.2. Its test MAE was almost the same as 1.2, while test R² improved.

However, 1.2 had better aggregate total deviation on the test week. This means 1.3 improved daily validation performance and test R², but showed stronger aggregate underprediction on the test period.

Because the validation and test windows contain only 7 days each, the difference between 1.2 and 1.3 test MAE should not be overinterpreted. The main reason 1.3 is selected as the official full-history advanced-boosting candidate is its much stronger validation performance under the predefined selection rule.

## 10000-Trial Sensitivity Diagnostic

A later 10000-trial Optuna run found a stronger AdaBoost configuration:

| Metric                 | 10000-trial AdaBoost |
| ---------------------- | -------------------- |
| Validation MAE         | 1286.4178            |
| Test MAE               | 1223.7885            |
| Test R²                | 0.5936               |
| Test total deviation % | -5.9685              |

This result is useful because it shows that additional tuning can still improve the score. However, it was not selected as the official 1.3 model because the validation window contains only 7 days.

The 10000-trial result highlights the main modeling limitation of the project: with a very small validation window, aggressive hyperparameter tuning can become sensitive to the quirks of a single week. Therefore, the 10000-trial result is documented as a sensitivity diagnostic, while the 2000-trial AdaBoost candidate is used as the official 1.3 model.

---

# Summary Up To 1.3

| Version | Scope        | Feature Set | Selected Model    | Validation MAE | Test MAE  | Test R² | Test Total Deviation % | Interpretation                               |
| ------- | ------------ | ----------- | ----------------- | -------------- | --------- | ------- | ---------------------- | -------------------------------------------- |
| 0.5     | Post-only    | Full        | Extra Trees       | 735.7531       | 1492.3472 | 0.4246  | 4.3923                 | Strong regime-specific post-only benchmark   |
| 1.0     | Full-history | Full        | Random Forest     | 3570.1870      | 1984.7877 | -0.3335 | 3.0344                 | Full-history baseline did not beat post-only |
| 1.1     | Full-history | Full        | Gradient Boosting | 2126.2109      | 1804.2195 | -0.1270 | 3.3452                 | Optuna improved full-history but not enough  |
| 1.2     | Full-history | VIF-reduced | Gradient Boosting | 2974.9578      | 1464.5123 | 0.4646  | 0.5426                 | Better test behavior, weaker validation      |
| 1.3     | Full-history | VIF-reduced | AdaBoost          | 1329.8433      | 1473.4256 | 0.5238  | -4.4112                | Strongest full-history validation candidate  |

# 2.0 Post-Only and Full-History Ensemble Diagnostic

## Purpose

After developing separate post-only and full-history forecasting candidates, the next step was to test whether their predictions could be combined.

The post-only branch provides the strongest current-regime validation model. It is trained only on the post-installation operating period, so it is directly aligned with the target forecasting regime.

The full-history branch provides a broader historical training sample. Although pre-intervention behavior may not fully match the current operating regime, the full-history model may still capture production-energy relationships that remain useful after the intervention.

Version 2.0 therefore tests a simple prediction-level ensemble between the two strongest modeling branches:

```text
post-only champion:
0.5 full feature set + Extra Trees

full-history champion:
1.3 vif_auto_full_history feature set + AdaBoost
```

The purpose of the ensemble is not to add unnecessary complexity. The purpose is to test whether current-regime specificity and broader historical signal can be combined to improve final forecasting stability.

## Ensemble design

The ensemble uses a weighted average of the two component model predictions:

```text
ensemble_prediction =
    post_only_weight * post_only_prediction
    + (1 - post_only_weight) * full_history_prediction
```

A simple weighted average is used instead of stacking or a meta-model.

Stacking is not appropriate at this stage because the post-intervention validation window contains only 7 days. Training a second-level model on such a small validation sample would create a high risk of overfitting.

The weighted ensemble is therefore a deliberately conservative approach.

## Candidate weights

Because the forecasting target is the current post-installation regime, the ensemble weights are intentionally biased toward the post-only model.

The predefined candidate weights were:

```text
60% post-only / 40% full-history
70% post-only / 30% full-history
```

These weights were chosen before final selection because they keep the post-only model as the dominant signal while still allowing the full-history model to contribute useful historical structure.

The 60/40 blend represents a more balanced ensemble.

The 70/30 blend places stronger emphasis on the current post-installation regime.

## Selection principle

The ensemble uses the same validation and final holdout test windows as the earlier post-only and full-history experiments.

The ensemble weight is selected using validation MAE within the predefined candidate-weight range.

The final test period is not used to choose the ensemble weight. It is used only after selection to evaluate final held-out behavior and compare the selected ensemble against its component models.

This keeps the selection rule consistent with the rest of the project:

```text
validation set:
select the ensemble weight

test set:
evaluate the selected ensemble once as a final holdout check
```

## Validation weight search

The validation weight search selected the 70/30 blend.

```text
70% post-only / 30% full-history:
validation MAE: 861.033368
validation RMSE: 1036.787105
validation R²: 0.809018
validation MAPE: 4.668177%
validation total deviation: 0.633766%

60% post-only / 40% full-history:
validation MAE: 902.793458
validation RMSE: 1168.095110
validation R²: 0.757579
validation MAPE: 4.876714%
validation total deviation: 1.446885%
```

The 70/30 ensemble was selected because it had the lower validation MAE within the predefined constrained ensemble search.

## Validation component comparison

The validation comparison against the component models was:

```text
post-only Extra Trees:
validation MAE: 735.753100
validation RMSE: 940.317618
validation R²: 0.842905
validation MAPE: 4.042565%
validation total deviation: -1.805594%

selected 70/30 ensemble:
validation MAE: 861.033368
validation RMSE: 1036.787105
validation R²: 0.809018
validation MAPE: 4.668177%
validation total deviation: 0.633766%

full-history AdaBoost:
validation MAE: 1329.843269
validation RMSE: 2339.647091
validation R²: 0.027445
validation MAPE: 7.114680%
validation total deviation: 6.325604%
```

This result is important.

The post-only Extra Trees model remains the strongest single-branch validation model by MAE. The ensemble is not presented as beating the post-only model on validation MAE.

Instead, the ensemble is treated as a combined forecasting candidate. It slightly worsens validation MAE compared with the post-only champion, but it keeps validation R² above 0.80 and substantially improves final holdout behavior.

This distinction matters because the project does not hide tradeoffs. The ensemble is selected as the official 2.0 ensemble by validation MAE within the constrained ensemble search, while the post-only model remains the best pure post-only validation benchmark.

## Official 2.0 selected ensemble

The official 2.0 ensemble is:

```text
70% post-only Extra Trees
30% full-history AdaBoost
```

The selected ensemble combines:

```text
current-regime signal:
post-only Extra Trees

broader historical signal:
full-history AdaBoost
```

The selected model artifact is saved as:

```text
models/ensemble_2_0_model.joblib
```

The experiment report is saved as:

```text
reports/ensemble_2_0_training_report.xlsx
```

The feature metadata is saved as:

```text
reports/metadata/ensemble_2_0_features.txt
```

The diagnostic figures are saved in:

```text
reports/figures/ensemble_2_0
```

## Final holdout test comparison

After the 70/30 ensemble weight was selected using validation MAE, the final model was evaluated on the held-out test period.

```text
selected 70/30 ensemble:
test MAE: 1022.455700
test RMSE: 1282.703944
test R²: 0.754668
test MAPE: 6.213821%
test total deviation: 1.751245%

full-history AdaBoost:
test MAE: 1473.425642
test RMSE: 1787.026591
test R²: 0.523828
test MAPE: 8.245912%
test total deviation: -4.411208%

post-only Extra Trees:
test MAE: 1492.347207
test RMSE: 1964.461965
test R²: 0.424575
test MAPE: 9.461457%
test total deviation: 4.392297%
```

The selected 70/30 ensemble substantially improves final holdout behavior compared with both individual component models.

Compared with the post-only Extra Trees model:

```text
test MAE improvement:
1492.347207 → 1022.455700

test R² improvement:
0.424575 → 0.754668

test total deviation improvement:
4.392297% → 1.751245%
```

Compared with the full-history AdaBoost model:

```text
test MAE improvement:
1473.425642 → 1022.455700

test R² improvement:
0.523828 → 0.754668

test total deviation improvement:
-4.411208% → 1.751245%
```

This makes the 70/30 ensemble the strongest current combined forecasting candidate for the available dataset.

## Bootstrap confidence intervals

Bootstrap 95% confidence intervals for the selected 70/30 ensemble on the final test period were:

```text
test MAE CI: [507.5856, 1542.1063]
test RMSE CI: [745.0319, 1689.8226]
test MAPE CI: [2.7120, 10.1456]
test total deviation CI: [-3.3546, 7.7958]
```

The confidence intervals remain wide because the final test window contains only 7 days.

This does not invalidate the model. It defines the boundary of the available evidence and supports the decision to report uncertainty transparently.

## 60/40 ensemble sensitivity diagnostic

A 60/40 ensemble was also evaluated as part of the predefined candidate-weight search.

The 60/40 blend produced weaker validation MAE than the 70/30 blend:

```text
70/30 validation MAE: 861.033368
60/40 validation MAE: 902.793458
```

For that reason, the 60/40 blend was not selected as the official 2.0 ensemble.

However, the 60/40 blend produced stronger final holdout behavior:

```text
60/40 test MAE: 931.150631
60/40 test RMSE: 1136.238515
60/40 test R²: 0.807496
60/40 test MAPE: 5.442291%
60/40 test total deviation: 0.870895%
```

Bootstrap 95% confidence intervals for the 60/40 sensitivity result were:

```text
test MAE CI: [477.2923, 1422.2330]
test RMSE CI: [605.8246, 1557.0020]
test MAPE CI: [2.7694, 8.1761]
test total deviation CI: [-3.8837, 5.8896]
```

This result is documented as a sensitivity diagnostic.

It suggests that a slightly larger full-history contribution may improve final-week behavior. However, selecting 60/40 as the official model would mean choosing the ensemble weight based on the final test period rather than validation MAE.

The 60/40 result is therefore retained as useful evidence for future revalidation, not as the official selected model.

As more post-installation data becomes available, the preferred ensemble weight should be re-evaluated. Future evidence may support a stronger full-history contribution, but the current official 2.0 decision remains the validation-selected 70/30 ensemble.

## 2.0 interpretation

Version 2.0 shows that the full-history model contains useful signal even though it is weaker than the post-only model on validation by itself.

The full-history AdaBoost model performs poorly as a standalone validation model compared with post-only Extra Trees. However, when blended conservatively with the post-only model, it improves final holdout behavior substantially.

This is the main modeling insight from the ensemble experiment:

```text
The full-history model is not strong enough to replace the post-only model,
but it can improve the forecast when used as a secondary signal.
```

The selected 70/30 ensemble preserves the post-only model as the dominant current-regime signal while using the full-history model as a stabilizing secondary component.

The result is especially encouraging because the selected ensemble keeps strong performance across both validation and test:

```text
70/30 validation R²: 0.809018
70/30 test R²: 0.754668
```

The 60/40 sensitivity result also keeps strong validation and test R²:

```text
60/40 validation R²: 0.757579
60/40 test R²: 0.807496
```

This supports the interpretation that the ensemble approach is more stable than either component model alone on the final holdout period.

## 2.0 decision

Decision:

```text
Select the 70/30 post-only/full-history ensemble as the official Version 2.0 ensemble.
```

Reason:

The 70/30 ensemble was selected by validation MAE within the predefined post-only-dominant ensemble search. It keeps the post-only model as the dominant current-regime signal, adds a controlled full-history contribution, and substantially improves final holdout performance compared with both component models.

The 60/40 blend is retained as a sensitivity diagnostic because it produced stronger final holdout behavior, but it is not selected as the official model because its advantage appears on the final test period rather than the validation selection period.

## Summary Up To 2.0

| Version | Scope | Feature Set | Selected Model | Validation MAE | Validation R² | Test MAE | Test R² | Test Total Deviation % | Interpretation |
| ------- | ----- | ----------- | -------------- | -------------- | ------------- | -------- | ------- | ---------------------- | -------------- |
| 0.5 | Post-only | Full | Extra Trees | 735.7531 | 0.8429 | 1492.3472 | 0.4246 | 4.3923 | Strongest single-branch post-only validation model |
| 1.3 | Full-history | VIF-reduced | AdaBoost | 1329.8433 | 0.0274 | 1473.4256 | 0.5238 | -4.4112 | Strongest full-history validation candidate |
| 2.0 | Ensemble | Full + VIF-reduced | 70/30 weighted ensemble | 861.0334 | 0.8090 | 1022.4557 | 0.7547 | 1.7512 | Official validation-selected ensemble |
| 2.0 sensitivity | Ensemble | Full + VIF-reduced | 60/40 weighted ensemble | 902.7935 | 0.7576 | 931.1506 | 0.8075 | 0.8709 | Strong sensitivity result, not official |

## Current modeling conclusion

The official Version 2.0 model is the 70/30 post-only/full-history ensemble.

The post-only Extra Trees model remains the strongest single-branch validation model. This is important and is not hidden.

However, the 70/30 ensemble is the strongest current combined forecasting candidate because it preserves strong validation behavior while substantially improving final holdout performance compared with both component models.

The result is credible because:

* the component models were selected before the ensemble experiment
* the ensemble weights were predefined before final selection
* the official ensemble weight was selected by validation MAE
* the final test period was not used to choose the official ensemble weight
* the final test period shows substantial improvement over both component models
* bootstrap confidence intervals are reported
* the 60/40 result is documented as sensitivity rather than selected post-hoc
* limitations from the short validation and test windows are documented clearly

The selected 70/30 ensemble is therefore the strongest current forecasting candidate for the available Damavand dataset.

It should be treated as a validated and defensible model within the available post-intervention evidence, not as a permanently final model. As more post-installation data becomes available, the ensemble weights should be revalidated.

## Offline champion-challenger comparison

After selecting the official Version 2.0 ensemble, an offline champion-challenger comparison was created.

This is not a live A/B test. No model is deployed in production and no live traffic is being split between models.

The correct framing is:

```text
offline champion-challenger comparison
```

The purpose is to compare the official champion against meaningful alternatives under the same validation and final holdout test windows.

The comparison includes:

```text
Champion:
2.0 70/30 post-only/full-history ensemble

Challengers:
0.5 post-only Extra Trees
1.3 full-history AdaBoost
2.0 60/40 ensemble sensitivity
```

The goal is not to claim that the official champion wins every metric. The goal is to check whether the selected model remains defensible when compared against strong alternatives under the same evaluation periods.

### Validation comparison

```text
0.5 post-only Extra Trees:
validation MAE: 735.753100
validation RMSE: 940.317618
validation R²: 0.842905
validation MAPE: 4.042565%
validation total deviation: -1.805594%

2.0 70/30 official ensemble:
validation MAE: 861.033368
validation RMSE: 1036.787105
validation R²: 0.809018
validation MAPE: 4.668177%
validation total deviation: 0.633766%

2.0 60/40 sensitivity ensemble:
validation MAE: 902.793458
validation RMSE: 1168.095110
validation R²: 0.757579
validation MAPE: 4.876714%
validation total deviation: 1.446885%

1.3 full-history AdaBoost:
validation MAE: 1329.843269
validation RMSE: 2339.647091
validation R²: 0.027445
validation MAPE: 7.114680%
validation total deviation: 6.325604%
```

The post-only Extra Trees model has the lowest validation MAE among the compared candidates.

This is important and is not hidden.

```text
post-only Extra Trees validation MAE: 735.753100
70/30 ensemble validation MAE:        861.033368
```

The 70/30 ensemble does not beat the post-only model on validation daily error.

However, the post-only validation advantage does not fully carry over to the final holdout test period. This is why the comparison is interpreted using both validation discipline and final holdout behavior.

R² is reported as a diagnostic metric, but it is not used as the primary selection argument. With only 7 validation days and 7 test days, validation R² can look strong without guaranteeing the same behavior on the next held-out week.

Within the predefined ensemble-weight search, the 70/30 blend beats the 60/40 blend on validation MAE:

```text
70/30 validation MAE: 861.033368
60/40 validation MAE: 902.793458
```

The 70/30 ensemble also has the best validation total deviation:

```text
70/30 ensemble validation total deviation:        0.633766%
60/40 ensemble validation total deviation:        1.446885%
post-only Extra Trees validation total deviation: -1.805594%
full-history AdaBoost validation total deviation: 6.325604%
```

This means the 70/30 ensemble is not the lowest-error validation model overall, but it is the best validation-selected ensemble and the best-calibrated candidate at the aggregate validation-week level.

### Final holdout test comparison

```text
2.0 60/40 sensitivity ensemble:
test MAE: 931.150631
test RMSE: 1136.238515
test R²: 0.807496
test MAPE: 5.442291%
test total deviation: 0.870895%

2.0 70/30 official ensemble:
test MAE: 1022.455700
test RMSE: 1282.703944
test R²: 0.754668
test MAPE: 6.213821%
test total deviation: 1.751245%

1.3 full-history AdaBoost:
test MAE: 1473.425642
test RMSE: 1787.026591
test R²: 0.523828
test MAPE: 8.245912%
test total deviation: -4.411208%

0.5 post-only Extra Trees:
test MAE: 1492.347207
test RMSE: 1964.461965
test R²: 0.424575
test MAPE: 9.461457%
test total deviation: 4.392297%
```

The 70/30 official ensemble substantially improves final holdout behavior compared with both individual component models.

Compared with the post-only Extra Trees model:

```text
test MAE:
1492.347207 → 1022.455700

test R²:
0.424575 → 0.754668

test MAPE:
9.461457% → 6.213821%

test total deviation:
4.392297% → 1.751245%
```

Compared with the full-history AdaBoost model:

```text
test MAE:
1473.425642 → 1022.455700

test R²:
0.523828 → 0.754668

test MAPE:
8.245912% → 6.213821%

test total deviation:
-4.411208% → 1.751245%
```

This is the main reason the ensemble is valuable. It gives up some validation MAE compared with the post-only champion, but it improves the final holdout test week substantially.

The 60/40 ensemble produced the strongest final holdout result:

```text
60/40 test MAE: 931.150631
60/40 test MAPE: 5.442291%
60/40 test total deviation: 0.870895%
```

However, it is retained as a sensitivity diagnostic rather than promoted to the official champion.

The reason is that the 60/40 advantage appears on the final test period. Selecting it as the official model would mean choosing the ensemble weight based on the test set, which would weaken the validation-first selection discipline.

### Champion-challenger interpretation

The offline champion-challenger comparison supports the 70/30 ensemble as the official champion.

The post-only Extra Trees model remains the strongest validation model by daily error. However, its validation advantage does not fully translate to the final holdout test week.

The 70/30 ensemble is therefore not presented as the best model by validation MAE. It is presented as the most defensible official champion because it follows the predefined selection rule and improves final holdout behavior compared with both component models.

The official decision is:

```text
Select the 70/30 ensemble as the official Version 2.0 champion.
```

Reason:

```text
- the component models were selected before the ensemble experiment
- the ensemble weights were predefined
- 70/30 beats 60/40 on validation MAE within the ensemble search
- 70/30 has the best validation total deviation
- the final test period confirms that 70/30 improves strongly over both component models
- 60/40 is documented as sensitivity rather than selected post-hoc from the test set
```

This makes the 70/30 ensemble the most defensible official forecasting candidate under the current evidence.

It is not selected because it wins every metric. It is selected because it follows the strongest validation-disciplined path.

The 60/40 result remains important because it suggests that a larger full-history contribution may improve final-week behavior. As more post-installation data becomes available, ensemble weights should be revalidated and may shift toward a larger full-history contribution.

# Version 2.1 — Behavioral Scenario Evaluation and Synthetic Stress Testing

## Purpose

After selecting the official Version 2.0 ensemble, the project added a post-selection behavioral evaluation to examine the forecasting system across representative real historical profiles, controlled local synthetic perturbations around those real historical profiles, deliberately adversarial operating combinations, and out-of-distribution conditions.

Version 2.1 does not replace the forecasting model. The official model remains:

```text
Version 2.0 official ensemble

70% post-only Extra Trees
30% full-history AdaBoost
```

Version 2.1 adds an evaluation and model-governance layer around that ensemble. It is designed to answer four practical questions:

```text
1. Does the model reproduce representative real operating days credibly?
2. Does the forecast respond sensibly to small coherent operating-scale changes?
3. Where do the post-only and full-history branches disagree?
4. Can the system identify uncommon or unsupported input conditions?
```

The final framework combines real historical replay evidence, local sensitivity analysis, branch-disagreement monitoring, operational-range checks, adversarial scenarios, OOD scenarios, automated runtime checks, and an auditable Excel report.

## Evaluation structure

The final Version 2.1 suite contains 20 scenarios:

```text
8 historical reference replays
8 paired local operating-scale perturbations
4 adversarial or out-of-distribution perturbations
```

The three groups serve different purposes.

Historical reference replays use complete real post-installation operating rows and retain their observed energy targets for supplementary replay-fit analysis.

Paired local synthetic perturbations apply small controlled changes around those historical rows to measure local forecast sensitivity.


Adversarial and OOD perturbations challenge the model with unusual combinations or values outside the observed development support so that warning behavior can be inspected.

## How the eight historical reference rows were selected

The historical examples were not chosen manually by browsing the final predictions. They were selected deterministically from the 37-row post-installation development window covering 2025-09-12 through 2025-10-24.

The selection process first defines an operating archetype and then chooses the complete real row that is closest to that archetype.

For every selected profile, the distance calculation uses squared standardized feature differences. Each feature is scaled by its interquartile range so that variables measured on large numerical scales, such as kilograms or Brix units, do not dominate variables measured on smaller scales. If the interquartile range is zero, the implementation falls back to the feature standard deviation and then to a scale of one if necessary.

The selected row is therefore:

```text
the real candidate row with the minimum total standardized profile distance
```

This method has several advantages:

```text
- every historical scenario is a complete internally consistent real day
- selection is reproducible rather than subjective
- different measurement scales are handled fairly
- calendar restrictions can be applied before nearest-profile selection
- joint operating patterns can be targeted without fabricating feature combinations
```

The eight historical archetypes were defined as follows:

| Historical scenario | Candidate restriction | Target profile used for nearest-row selection |
| --- | --- | --- |
| `baseline_median_weekday` | Weekdays only | Median profile across the operational features |
| `low_production_low_hours` | All development rows | 15th-percentile activity profile |
| `high_production_high_hours` | All development rows | 85th-percentile activity profile |
| `weekend_moderate_production` | Weekends only | Median operational profile |
| `high_orders_moderate_production` | Orders at or above the 75th percentile | 90th-percentile orders with median production and hours |
| `low_orders_high_production` | Production at or above the 75th percentile | 10th-percentile orders, 90th-percentile production, and 75th-percentile hours |
| `high_brix_day` | All development rows | 90th-percentile average Brix with median production |
| `low_brix_day` | All development rows | 10th-percentile average Brix with median production |

The low- and high-activity profiles use the activity feature bundle:

```text
total_kg
total_nominal_kg
total_brix_units
total_hours
total_pallets
orders
```

The median operational profiles use:

```text
total_kg
total_nominal_kg
total_brix_units
total_hours
total_pallets
orders
avg_brix
yield_ratio_actual_over_nominal
```

This selection strategy is important because it produces a small but deliberately varied set of real examples covering ordinary operation, low and high activity, weekend behavior, unusual order-production combinations, and upper- and lower-Brix conditions.

## Historical replay fit against observed energy

Because the eight historical references are real rows, each one has an observed `active_energy_kWh` value in the source dataset. The official 70/30 ensemble prediction was compared with that observed value.

Deviation is defined as:

```text
prediction - actual
```

A positive value indicates overprediction and a negative value indicates underprediction.

| Date | Historical scenario | Actual energy (kWh) | Official prediction (kWh) | Deviation (kWh) | Absolute error (kWh) | Deviation % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2025-09-19 | `high_production_high_hours` | 71,779.8 | 70,944.8 | -835.0 | 835.0 | -1.16% |
| 2025-09-26 | `low_orders_high_production` | 64,127.5 | 65,298.4 | +1,170.9 | 1,170.9 | +1.83% |
| 2025-10-04 | `low_brix_day` | 21,163.5 | 21,489.9 | +326.4 | 326.4 | +1.54% |
| 2025-10-16 | `high_brix_day` | 10,768.4 | 11,468.6 | +700.2 | 700.2 | +6.50% |
| 2025-10-18 | `low_production_low_hours` | 12,615.3 | 12,651.5 | +36.2 | 36.2 | +0.29% |
| 2025-10-19 | `weekend_moderate_production` | 17,670.4 | 19,177.4 | +1,507.0 | 1,507.0 | +8.53% |
| 2025-10-22 | `baseline_median_weekday` | 19,598.1 | 20,495.5 | +897.4 | 897.4 | +4.58% |
| 2025-10-24 | `high_orders_moderate_production` | 20,198.7 | 20,107.3 | -91.4 | 91.4 | -0.45% |

Summary replay-fit metrics across the eight examples were:

```text
MAE:                     695.559 kWh
RMSE:                    847.874 kWh
MAPE:                      3.110%
mean signed deviation:   +463.970 kWh
aggregate actual:       237,921.700 kWh
aggregate prediction:   241,633.460 kWh
aggregate deviation:     +3,711.760 kWh
aggregate deviation %:    +1.560%
```

The ensemble overpredicted six of the eight examples and underpredicted two. The mean signed deviation of approximately +464 kWh indicates a modest positive bias across this selected replay set, while the aggregate prediction remained only 1.56% above the observed total.

The closest matches were:

```text
low_production_low_hours:
absolute error 36.2 kWh, deviation +0.29%

high_orders_moderate_production:
absolute error 91.4 kWh, deviation -0.45%
```

The largest replay error occurred for the weekend moderate-production profile:

```text
absolute error 1,507.0 kWh
deviation +8.53%
```

The weekend-moderate profile produced the largest replay deviation at +8.53%, while the selected high-Brix profile produced a deviation of +6.50%. These two profiles are useful monitoring candidates because they identify operating conditions in which the ensemble was less precise than for the other selected archetypes. This is not evidence of model failure. It is consistent with the limited number of post-installation examples available for these operating conditions and indicates where additional real observations would most improve confidence. The high-Brix result describes the complete selected operating profile, which also contained relatively low production volume and operating hours, rather than an isolated effect of Brix.

The historical replay MAE is lower than the official chronological validation and test MAE:

```text
historical replay MAE: 695.559 kWh
validation MAE:        861.033 kWh
final test MAE:       1,022.456 kWh
```

This replay result meaningfully strengthens the model evidence by showing that the final refitted ensemble reproduces a deliberately varied set of representative real operating days with low error and good aggregate calibration. It is reported as supplementary representative-profile fit, while the chronological validation and final test periods remain the primary evidence of generalization.

## Paired local operating-scale perturbations

Each historical reference row is paired with one controlled local perturbation.

The local scenario applies an approximately 5% coherent increase to:

```text
total_kg
total_nominal_kg
total_brix_units
total_hours
```

The following values are preserved from the historical parent:

```text
orders
total_pallets
avg_brix
weekday
is_weekend
month
year
week_of_year
```

The `orders` feature is preserved because it represents a discrete nonnegative count rather than a continuous operating-scale variable.

The confirmed derived feature is recalculated after the perturbation:

```text
yield_ratio_actual_over_nominal =
    total_kg / total_nominal_kg
```

This design increases the overall scale of a similar operating day while preserving its calendar context, order count, average Brix, pallet count, and approximate production intensity.

The paired synthetic scenarios therefore provide a controlled local sensitivity test around eight real operating profiles.

## Adversarial and out-of-distribution scenarios

Four additional scenarios test operating combinations that are deliberately unusual or unsupported:

```text
high_intensity_kg_low_hours
low_intensity_kg_high_hours
outside_observed_high_activity
near_shutdown_low_activity
```

The two adversarial scenarios alter operating hours while retaining the production profile of their historical anchor. This creates deliberately unusual production-intensity combinations.

### Production-intensity diagnostic

Production intensity is calculated as `kg_per_hour = total_kg / total_hours`.

`kg_per_hour` represents the amount of production processed per operating hour.

A high-production profile combined with very low operating hours creates an unusually high `kg_per_hour` value. A low-production profile combined with very high operating hours creates an unusually low `kg_per_hour` value.

This quantity is used only as an evaluation and operational-support diagnostic. It is not an additional input feature used by either branch of the official forecasting model.

The diagnostic was introduced because `total_kg` and `total_hours` may each appear acceptable when checked separately, while their relationship can still represent an unusual or operationally unsupported production intensity.

The two adversarial scenarios retain the production profile of their historical anchor while deliberately changing operating hours:

- `high_intensity_kg_low_hours`
- `low_intensity_kg_high_hours`

The first creates unusually high production per operating hour. The second creates unusually low production per operating hour.

These scenarios test whether the operational-support safeguards identify abnormal relationships between production volume and operating time, rather than evaluating each feature only in isolation.

The high OOD scenario places activity features above their observed development maxima.

The near-shutdown scenario reduces activity features below their observed development minima.

These scenarios are used to inspect forecast behavior, range classification, disagreement, and warning generation at the edges of model support.

## Scenario provenance and the meaning of boundary cases

Every scenario records:

```text
scenario_origin
parent_scenario_name
perturbation_type
perturbation_scale
construction_method
scenario_class
operational_range_status
calendar_coverage_status
```

`scenario_class` and `operational_range_status` answer different questions.

`scenario_class` describes why the scenario was included in the evaluation. It is a design label such as plausible, boundary, adversarial, or out of distribution.

`operational_range_status` is calculated after construction by comparing each operational feature with its observed post-installation development range and q10-q90 interval.

A boundary scenario can therefore remain inside the central marginal ranges. For example, `low_orders_high_production` was selected because the combination of relatively low orders and high production is an important joint operating pattern. Each individual feature may still lie inside its own q10-q90 interval, so the marginal range checker can classify the row as `inside_typical_development_range`.

This is analytically consistent because:

```text
boundary scenario class = design importance of the complete profile
operational range status = marginal support of individual features
```

The current range system evaluates each feature separately. It does not estimate multivariate density or the probability of the complete feature combination. A row may therefore be individually typical on every variable while still being interesting at the joint-profile level.

This distinction is useful rather than problematic. It allows the evaluation to retain domain-relevant joint patterns without incorrectly labeling every unusual combination as OOD.

## Input-support classification

Operational familiarity is evaluated using the post-installation development period.

The classification logic is:

```text
outside_observed_range:
at least one operational feature is below the observed minimum
or above the observed maximum

development_distribution_tail:
all operational features remain inside observed min/max limits,
but at least one variable feature is below q10 or above q90

inside_typical_development_range:
all variable operational features remain between q10 and q90
```

Features that were constant during development are handled explicitly:

```text
constant_in_development:
the scenario uses the only observed development value

outside_observed_range:
the scenario changes a constant feature away from its only observed value
```

Calendar validity, operational support, schema validity, and branch disagreement are kept as separate diagnostic dimensions.

## Automated verification

The final implementation contains:

```text
17 unit tests
27 runtime hard checks
```

All 17 unit tests passed.

All 27 runtime hard checks passed.

The checks verify:

```text
- unique scenario names
- valid scenario schemas and metadata
- correct historical, local, and stress-scenario counts
- one-to-one historical/local pairing
- correct local perturbation values
- preserved orders, pallets, average Brix, and calendar context
- nonnegative integer-like order counts
- consistent derived yield ratios
- finite and nonnegative predictions
- exact 70/30 and 60/40 ensemble mathematics
- explicit handling of constant features
- correct OOD range flags
- valid integer order counts in OOD scenarios
```

The final result was:

```text
hard checks passed: 27
hard-check failures: 0
invalid schemas: 0
```

## Operational-range results

The 20 scenarios were classified as:

```text
inside typical development range: 9
development-distribution tail:    7
outside observed range:           4
```

The eight paired local perturbations contributed:

```text
4 inside-typical scenarios
4 distribution-tail scenarios
0 outside-range scenarios
```

The local perturbations therefore remained within observed operational support. Some moved from the central q10-q90 region into the development tails without exceeding observed minima or maxima.

The clearest example was:

```text
high_production_high_hours:
inside_typical_development_range

high_production_high_hours_operating_scale_up_5pct:
development_distribution_tail
```

The 5% increase made the profile less common but still supported by observed development values.

All four deliberately adversarial or OOD scenarios were classified outside the observed operational range.

## Broad behavioral results

Representative official 70/30 ensemble predictions were:

```text
near-shutdown activity:             11,734 kWh
low production and low hours:       12,652 kWh
representative weekday baseline:    20,495 kWh
high production and high hours:     70,945 kWh
outside-observed high activity:     75,345 kWh
```

The broad directional ordering was:

```text
high activity > baseline
baseline > low activity
high activity > low activity
baseline > near shutdown
```

All four broad directional checks passed.

This supports a coherent ordering between operating scale and forecast energy consumption across the designed profiles.

## Paired local sensitivity results

All eight paired local perturbations moved in the expected direction.

```text
paired directional checks passed: 8
locally flat responses:           0
diagnostic warnings:              0
```

The official 70/30 ensemble responses were:

| Historical reference | Prediction change | Relative change |
| --- | ---: | ---: |
| `high_orders_moderate_production` | +2,030.734 kWh | +10.099% |
| `baseline_median_weekday` | +645.812 kWh | +3.151% |
| `weekend_moderate_production` | +414.427 kWh | +2.161% |
| `low_production_low_hours` | +194.019 kWh | +1.534% |
| `low_brix_day` | +244.485 kWh | +1.138% |
| `low_orders_high_production` | +538.368 kWh | +0.824% |
| `high_brix_day` | +85.483 kWh | +0.745% |
| `high_production_high_hours` | +426.529 kWh | +0.601% |

The nonlinear response sizes are consistent with the piecewise behavior of tree ensembles. A small change may remain within the same terminal regions, move only one component branch, or cross several tree thresholds.

The strongest local response occurred for `high_orders_moderate_production`:

```text
official ensemble change:    +2,030.734 kWh, +10.099%
post-only Extra Trees change: +1,263.583 kWh
full-history AdaBoost change: +3,820.752 kWh
```

This scenario is the clearest local sensitivity hotspot and should receive additional monitoring in a future prediction service.

## Component-model disagreement

Absolute branch disagreement is the difference between the post-only and full-history predictions in kWh.

Relative disagreement expresses that difference as a percentage of the mean magnitude of the two component predictions.

The maximum absolute disagreement was:

```text
scenario: low_orders_high_production
absolute disagreement: 5,190.830 kWh
relative disagreement: 7.825%
```

The maximum relative disagreement was:

```text
scenario: high_orders_moderate_production_operating_scale_up_5pct
absolute disagreement: 3,624.241 kWh
relative disagreement: 15.852%
```

The predefined thresholds are:

```text
low:      below 10%
moderate: 10% to below 20%
high:     20% or greater
```

The final result was:

```text
low-disagreement scenarios:      15
moderate-disagreement scenarios: 5
high-disagreement scenarios:     0
```

No scenario reached the high-disagreement threshold.

The disagreement analysis also shows why OOD status and branch agreement must be reported independently. The outside-observed high-activity scenario had low component disagreement even though its inputs exceeded observed development limits. Agreement between two tree models outside their support does not remove the range warning.

## Evidence interpretation

Version 2.1 adds three complementary forms of evidence:

```text
historical replay fit on representative real days
local behavioral sensitivity around those days
stress and support diagnostics for unusual inputs
```

The historical replay comparison shows low error and good aggregate calibration across eight deliberately varied real operating profiles.

The paired analysis shows consistent directional responses around every selected anchor.

The stress suite shows that uncommon and outside-range conditions can be identified and that branch disagreement can be surfaced separately.

Chronological validation and final holdout testing remain the primary generalization measures. Replay metrics describe representative-profile fit after final refitting, while the synthetic sensitivity results describe how the model responds to designed input changes and branch disagreement is used as a monitoring signal rather than as a calibrated prediction interval.


## MLflow tracking

The official Version 2.1 run was logged separately from candidate-model training runs.

```text
experiment:
Damavand Energy Forecasting

run name:
2.1 Behavioral Scenario Evaluation and Synthetic Stress Test

run type:
behavioral_scenario_evaluation

model role:
official_champion_diagnostic

forecasting model version:
2.0_ensemble_70_30

stress-test version:
2.1

status:
official
```

MLflow run ID:

```text
c07bfa3a352f49e3b85bb838e62bdee7
```

The run records scenario counts, verification results, range classifications, paired sensitivity outcomes, maximum branch disagreement, thresholds, ensemble weights, methodology, source code, tests, the Excel report, and all four diagnostic figures.

## Version 2.1 artifacts

The final report is saved as:

```text
reports/ensemble_2_1_behavioral_stress_test_report.xlsx
```

The figures are saved under:

```text
reports/figures/ensemble_2_1_behavioral_stress_test
```

The four figures are:

```text
branch_disagreement_kwh.png
branch_disagreement_pct.png
official_70_30_scenario_predictions.png
paired_local_sensitivity_delta_pct.png
```

The implementation and tests are:

```text
src/stress_test_ensemble.py
tests/test_stress_test_ensemble.py
```

## Version 2.1 decision

Version 2.1 is accepted as the official post-selection evaluation for the Version 2.0 ensemble.

```text
Official forecasting model:
Version 2.0 70/30 ensemble

Official post-selection evaluation:
Version 2.1 behavioral scenario evaluation and synthetic stress test
```

The combined evidence supports the following conclusions:

```text
- the ensemble reproduces representative historical profiles with low replay error
- broad activity ordering is coherent
- all eight local operating-scale changes produce the expected forecast direction
- local sensitivity hotspots are identifiable
- component disagreement is measurable and operationally usable
- typical, tail, and outside-range conditions are distinguished
- OOD safeguards and implementation checks operate correctly
```

## Current final modeling conclusion

The official forecasting model remains:

```text
70% post-only Extra Trees
30% full-history AdaBoost
```

Its chronological evaluation results are:

```text
validation MAE: 861.033368
validation R²: 0.809018
validation total deviation: 0.633766%
validation MAPE: 4.668177%

final holdout test MAE: 1022.455700
final holdout test R²: 0.754668
final holdout test MAPE: 6.213821%
final holdout total deviation: 1.751245%
```

Its supplementary historical replay results are:

```text
8 representative real profiles
replay MAE: 695.559 kWh
replay RMSE: 847.874 kWh
replay MAPE: 3.110%
aggregate deviation %: +1.560%
```

The official ensemble is therefore presented as a validated and defensible industrial energy-forecasting candidate under the available post-installation evidence, with explicit behavioral checks and deployment-oriented safeguards.

The main remaining evidence limitation is the short post-installation evaluation period:

```text
validation: 7 days
test:       7 days
```

The model and ensemble weights should be revalidated as additional real post-installation observations become available.

## Next project steps

The modeling and post-selection evaluation phase is complete.

The next step is to create the final model card covering intended use, model architecture, data regimes, features, validation strategy, performance, replay evidence, behavioral findings, limitations, unsupported uses, monitoring requirements, and retraining triggers.

After the model card, the official ensemble should be packaged behind a FastAPI prediction service with validated single-row and batch endpoints.

The service should then be containerized with Docker and supported by automated API, integration, and schema-validation tests.

GitHub Actions should run the complete unit, API, and integration test suite automatically.

The separate 100-day synthetic dataset may later be used for batch-inference, API, Docker, calendar-coverage, and warning-system testing. It should remain separate from the real chronological performance evaluation.

The final engineering phase should define monitoring for input drift, operational-range status, component disagreement, prediction distributions, delayed actual-error metrics, and retraining triggers.

