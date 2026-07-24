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

## Current Conclusion

The post-only model remains a strong benchmark because it is trained only on the post-intervention regime. The full-history approach initially struggled, but improved through tuning, VIF-reduced features, and advanced boosting.

The 1.3 AdaBoost model is the strongest full-history validation candidate so far. It substantially improves full-history validation MAE and achieves competitive test performance, but still shows aggregate underprediction on the test week.

The main project limitation is the small post-intervention evaluation window. With only 7 validation days and 7 test days in the official post-intervention split, model comparison is sensitive to short-term variation. This is why the project documents not only model scores, but also uncertainty, validation discipline, and diagnostic experiments.

## Next Planned Experiments

The next modeling steps are:

1. Post-only and full-history ensemble
   Test whether combining the post-only champion and full-history champion improves stability.

2. Regime-aware full-history diagnostic
   Add intervention-aware features such as `is_post_installation` and possibly `days_since_intervention` to test whether full-history data becomes more useful when the model is explicitly told that the operating regime changed.

3. Pre-only transfer diagnostic
   Train models only on pre-intervention data and evaluate transfer performance on the full post-intervention period. This will test whether the old production-energy relationship generalizes to the new regime.

4. Final model comparison
   Compare post-only, full-history, ensemble, regime-aware, and transfer-diagnostic candidates under clearly stated selection rules.

5. Deployment packaging
   Package the final selected model with an API, Docker, tests, and CI.

6. Scenario stress testing
   Use synthetic production scenarios only as stress tests and sensitivity checks, not as evidence of real-world predictive accuracy.


