# Damavand Professional M&V Case Study

## AI-based baseline modeling and verified industrial energy savings

This case study documents the professional Measurement & Verification (M&V) work I completed during my time at Senerqon for the Damavand industrial energy project.

The objective was to estimate the electrical energy consumption that would have occurred without the energy-saving intervention and use that counterfactual baseline to verify the achieved savings.

This professional M&V project is separate from the independent forecasting and MLOps project presented in the rest of this repository. The two projects share the same industrial setting and curated daily data foundation, but they answer different questions and use different modeling and validation approaches.

## At a glance

| Area | Professional M&V project |
| --- | --- |
| Objective | Estimate counterfactual daily electricity consumption and verify achieved savings |
| My role | End-to-end technical ownership of the AI-based M&V analysis |
| Data challenge | Production/order and batch-level operational records had to be aligned with daily active-energy measurements |
| Data engineering | Daily aggregation, temporal alignment, operational feature engineering, and feature-governance decisions |
| Selected model | CatBoost counterfactual baseline |
| Independent validation | 2024-11-28 to 2024-12-05, excluded from training |
| Reporting period | 2025-09-12 to 2025-10-31 |
| Verified savings | **11.89%** |
| Guaranteed target | **10.10%** |
| Review process | Technical presentation followed by two rounds of technical review; all raised issues were addressed through documented responses and additional validation or robustness checks where required |
| Outcome | Technical acceptance of the M&V analysis, supporting a critical milestone in formally closing the wider professional engagement |

## My technical ownership

I independently completed the AI-based M&V work represented in this case study.

My responsibility included:

- transforming the client’s raw production and operational records into a model-ready daily analytical dataset;
- defining the feature representation used to align factory activity with daily active-energy consumption;
- evaluating alternative modeling approaches and selecting the CatBoost baseline;
- defining the training and independent-validation structure;
- developing and validating the counterfactual energy model;
- preparing the technical model report;
- presenting and explaining the methodology;
- answering two rounds of technical reviewer questions;
- performing additional validation and robustness checks requested during review;
- supporting the final savings-verification conclusion.

The work was therefore not limited to fitting a model. It covered the complete analytical path from raw operational data to a technically reviewed M&V result.

## 1. Industrial data problem

The source data did not arrive at the same analytical scale as the electricity measurements.

The client’s operational data was provided at production-order, product, and batch level, with multiple records potentially occurring on the same day. Active electrical energy, however, was measured at daily resolution.

A direct row-level relationship between individual production records and daily electricity consumption was therefore not meaningful.

I resolved this by building a unified daily analytical dataframe in which each operating day represented one modeling observation.

For each date, the production records were aggregated and transformed into physical and operational indicators describing:

- production volume;
- nominal production;
- operating duration;
- order complexity;
- process intensity;
- production efficiency;
- packaging activity;
- calendar context.

The resulting daily representation was aligned directly with daily active-energy consumption.

### Engineered daily features

The final model-ready data included variables such as:

- `total_kg`
- `total_nominal_kg`
- `yield_ratio_actual_over_nominal`
- `total_hours`
- `total_pallets`
- `orders`
- `total_brix_units`
- `avg_brix`
- `weekday`
- `is_weekend`
- `month`
- `year`
- `week_of_year`

The weighted `avg_brix` representation was used to describe the average process characteristic of a day without allowing small batches to distort the daily value.

### Feature-governance decisions

More granular identifiers and categorical fields were evaluated but not retained as direct daily model inputs, including product codes, product descriptions, order codes, detailed packaging categories, and individual production phases.

The main reasons were that these fields were not uniquely defined at daily level, could create high-dimensional noise, increased the risk that the model would learn identifiers rather than physical operating behavior, and were more difficult to justify as stable daily drivers during technical review.

Their operational information was instead represented through aggregated physical quantities such as production mass, operating time, order count, and BRIX-related features.

Real operating outliers were retained because they represented genuine production and energy conditions. Only zero-energy records were excluded.

The data-synthesis step was a major part of the technical work: it converted raw production records into a daily operational representation suitable for defensible model-based M&V.

## 2. Counterfactual M&V methodology

The professional model was designed to answer a counterfactual question:

> What would the facility’s electricity consumption have been under the observed operating conditions if the energy-saving intervention had not been implemented?

The baseline model was trained on pre-intervention data.

During the reporting period, the model received the real production, operational, and calendar conditions and estimated the corresponding no-intervention electricity consumption.

Savings were then evaluated from the difference between:

```text
counterfactual baseline consumption
minus
actual measured post-intervention consumption
```

This is fundamentally different from forecasting future actual consumption. The M&V model estimates the no-intervention baseline against which the post-intervention energy result is evaluated.

## 3. Model selection

Alternative regression and machine-learning approaches were examined during development.

The technical model-selection review documents the evaluation of approaches including MLP and XGBoost before CatBoost was retained for the professional baseline.

The selection objective was not simply to maximize training accuracy. The model needed to provide a stable, reproducible, and technically defensible reconstruction of pre-intervention energy behavior.

CatBoost was selected because it provided a strong combination of baseline fit, stability, nonlinear modeling capability, and suitability for the correlated operational variables present in the industrial dataset.

The final CatBoost configuration used controlled model complexity and a fixed random seed for reproducibility.

## 4. Validation design

The original M&V implementation used an independent validation window inside the pre-intervention baseline period:

```text
2024-11-28 to 2024-12-05
```

This period was explicitly excluded from model training.

A reviewer-requested overlap check confirmed:

```text
training / validation overlap: 0 rows
```

Keeping validation inside the baseline period was intentional. Post-intervention observations already contain the effect being measured, so using them to tune the no-intervention baseline would contaminate the baseline model.

### Baseline training performance

| Metric | Result |
| --- | ---: |
| MAE | 544.67 kWh |
| Total deviation | -0.46% |
| R² | 0.994 |

### Independent validation performance

| Metric | Result |
| --- | ---: |
| Period | 2024-11-28 to 2024-12-05 |
| MAE | 3,035.28 kWh |
| Total deviation | -2.24% |

The validation period was not used during training.

The small negative aggregate deviation was also conservative with respect to savings because the model was not systematically inflating the counterfactual baseline.

## 5. Reporting-period evaluation and verified savings

The reporting period began on:

```text
2025-09-12
```

The trained baseline model was applied to the real post-intervention operating conditions to estimate the electricity consumption that would have been expected without the energy-saving measures.

The analysis produced:

| Result | Value |
| --- | ---: |
| Reporting-period R² | 0.9252 |
| Verified electrical energy savings | **11.89%** |
| Guaranteed project target | **10.10%** |

The achieved saving therefore exceeded the guaranteed target.

The Damavand reference letter independently confirms the use of an AI-based predictive energy model, describes its application to pre- and post-installation conditions, and states that the verified total energy saving across the facility reached **11.89%**, compared with a guaranteed performance target of **10.10%**.

## 6. Technical review and defense

The professional submission went through a technical presentation and two rounds of reviewer questions.

### First review round — model selection and methodology

The first technical response addressed topics including:

- model-family selection;
- CatBoost hyperparameters;
- multicollinearity and VIF results;
- treatment of correlated operational variables;
- baseline performance;
- validation behavior;
- stability of the selected methodology.

### Second review round — validation and robustness

The second response addressed a deeper set of questions including:

- high training R² and overfitting;
- explicit separation of training and validation data;
- validation-window design;
- interpretation of reporting-period MAE in a counterfactual M&V setting;
- uncertainty analysis;
- multicollinearity;
- hyperparameter and random-seed sensitivity;
- robustness of the savings conclusion.

Additional reviewer-requested checks included stricter validation, structural checks, bootstrap analysis, and materially different model parameterizations.

Every technical issue raised across the two review rounds was addressed with a documented methodological explanation, supporting evidence, or an additional quantitative check where appropriate. The supplementary validation, structural checks, bootstrap analysis, sensitivity tests, and materially different model parameterizations continued to support a savings conclusion in approximately the same 11%–12% range. The review process therefore reinforced rather than overturned the original technical conclusion, and the M&V analysis was subsequently accepted.

## 7. Professional outcome

The accepted M&V analysis verified:

> **11.89% electrical energy savings against a guaranteed 10.10% target.**

Technical acceptance of the submission was a critical milestone in formally closing the wider professional engagement.

For me, the project demonstrated end-to-end ownership across industrial data engineering, applied machine learning, counterfactual modeling, validation design, technical reporting, presentation, reviewer response, and business-facing model acceptance.

## 8. Relationship to the independent forecasting project

The professional M&V project and the forecasting project in this repository are **two separate technical projects**.

| | Professional M&V project | Independent forecasting project |
| --- | --- | --- |
| Primary question | What would energy consumption have been without the intervention? | What daily energy consumption should be expected next? |
| Purpose | Counterfactual baseline and savings verification | Forward operational forecasting |
| Main model | CatBoost | 70% post-only Extra Trees + 30% full-history AdaBoost |
| Validation logic | Pre-intervention baseline validation for M&V | Intervention-aware chronological forecasting validation |
| Delivery context | Professional project completed during my time at Senerqon | Independently developed after leaving Senerqon |
| Shared element | Curated daily analytical data foundation | Curated daily analytical data foundation |

The independent forecasting system reuses the daily analytical data foundation because the underlying industrial source data is the same, but it has its own model-development, validation, serving, testing, observability, and governance lifecycle.

For the forecasting project itself, see:

- [README.md](README.md)
- [Model Card](docs/MODEL_CARD.md)
- [API Contract](docs/API_CONTRACT.md)
- [Detailed Modeling Notes](docs/Modeling_notes.md)

## 9. Supporting professional evidence

The public repository includes the following redacted portfolio evidence for the professional M&V project:

1. [Original model report — public redacted English portfolio version](docs/case_study/damavand-original-model-report.pdf)  
   Documents the CatBoost baseline methodology, training and validation results, savings result, model code, execution outputs, and dataset-synthesis methodology.

2. [Model-selection and methodology Q&A](docs/case_study/damavand-model-selection-methodology-qa.pdf)  
   Documents the first technical review round, including model selection, hyperparameters, multicollinearity, VIF analysis, and baseline-model performance.

3. [Validation and reviewer-response Q&A](docs/case_study/damavand-validation-review-qa.pdf)  
   Documents the second technical review round, including overfitting, validation leakage, reporting-period interpretation, uncertainty, sensitivity checks, and robustness of the savings conclusion.

4. [Damavand reference letter — public redacted version](docs/case_study/damavand-reference-letter-redacted.pdf)  
   Provides client-side confirmation of the AI-based predictive methodology and the verified **11.89%** total energy saving against the **10.10%** guaranteed target.

## Evidence and privacy note

The public case-study documents are redacted portfolio versions. Detailed daily operational energy values, private communications, personal information, and other commercially sensitive material are intentionally excluded.

The case-study evidence documents the professional M&V work. The separate forecasting and MLOps system in this repository was independently developed after my time at Senerqon.
