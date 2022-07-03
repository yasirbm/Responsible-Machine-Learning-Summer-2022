# Responsible Machine Learning - Summer 2022

The authors spent 7 weeks developing interpretable machine learning models as part of a 'Responsible Machine Learning' class at George Washington University, taught by Professor Patrick Hall. The objective of this project was to predict the probability of applicants being charged more for mortgages than others, using the Home Mortgage Disclosure Act's historic mortgage reporting data. We strived to keep our models as interpretable and explainable as possible, to avoid the confusion and adverse bias that can often be caused through black-box machine learning models.

## Basic Details

**Authors:**
* Abid Shafiullah (abidshafi@gwu.edu)
* Somendar Chaudhary (somender@gwu.edu)
* Yasir Mohammad (yasir@gwu.edu)

**Basic Details:**
* Model version: 1.0
* License: Apache License
* Model date: June 2022
* Software Used: Python 3.6+, InterpretML v0.2.5.
* Model Type: Explainable Boosting Machine

**Intended Use:**
 * In short, this model's primary goal is to reduce discrimination and bias in the issuance of mortgage rates to applicants by providing transparency with interpretable and explainable machine learning models.
 * The goal of this project is to predict whether or not the annual percentage rate (APR) charged for a mortgage is 150 basis points (1.5%) or more above a survey-based estimate of similar mortgages. (High-priced mortgages are legal, but somewhat punitive to borrowers. High-priced mortgages often fall on the shoulders of minority home owners, and are one of many issues that perpetuates a massive disparity in overall wealth between different demographic groups in the US.) The intended use is also to provide an interpretable machine learning model, which is transparent and helps prevent biases.
* **Primary intended users**: Students and academics interested in interpretable machine learning models.
* **Out-of-scope use cases**: Any use beyond an educational example is out-of-scope.

## Training Data
  * **Source of training data:** Home Mortgage Disclosure Act (HMDA) data in the class repository https://github.com/jphall663/GWU_rml/tree/master/assignments/data
  * **Training & Validation Data split**: 70% training data, 30% validation data 
  * Training data rows = 112253, columns = 23
  * Validation data rows = 48085, columns = 23

**Data dictionary:**

| Name | Modeling Role | Measurement Level| Description|
| ---- | ------------- | ---------------- | ---------- |
|**high_priced**| target | int | whether (1) or not (0) the annual percentage rate (APR) charged for a mortgage is 150 basis points (1.5%) or more above a survey-based estimate of similar mortgages |
| **conforming** | input | int | whether the mortgage conforms to normal standards (1), or whether the loan is different (0) |
| **debt_to_income_ratio_std** | input | int | standardized debt-to-income ratio for mortgage applicants | 
| **debt_to_income_ratio_missing** | demographic information | int | missing marker (1) for debt to income ratio std |
| **income_std** | input | int | standardized income for mortgage applicants |
| **loan_amount_std	** | input | int | standardized amount of the mortgage for applicants |
| **intro_rate_period_std	** | input | int | standardized introductory rate period for mortgage applicants |
| **loan_to_value_ratio_std** | input | int | ratio of the mortgage size to the value of the property for mortgage applicants |
| **no_intro_rate_period_std** | input | int | whether (1) or not (0) a mortgage does not include an introductory rate period |
| **property_value_std** | input | int | value of the mortgaged property |
| **term_360**| input | int | whether the mortgage is a standard 360 month mortgage (1) or a different type of mortgage (0) |
| **male**| demographic information | int | whether a person identifies as male (1) or not male (0) |
| **female**| demographic information | int | whether a person identifies as female (1) or not female (0) |
| **black**| demographic information | int | whether a person identifies as black (1) or not black (0) |
| **asian**| demographic information | int | whether a person identifies as asian (1) or not asian (0) |
| **white**| demographic information | int | whether a person identifies as white (1) or not white (0) |
| **amind**| demographic information | int | whether a person identifies as amind (1) or not amind (0) |
| **hipac**| demographic information | int | whether a person identifies as hipac (1) or not hipac (0) |
| **hispanic**| demographic information | int | whether a person identifies as hispanic (1) or not hispanic (0) |
| **non_hispanic**| demographic information | int | whether a person identifies as non_hispanic (1) or not non_hispanic (0) |
| **agegte62**| demographic information | int | 	whether a person is over the age of 62 (1) or not over the age of 62 (0) |
| **agelt62E**| demographic information | int | whether a person is below the age of 62 (1) or not below the age of 62 (0) |
| **row_id**| ID | int | unique row indentifier |

  **Evaluation Data:**
  * Soure of evaluation data: Home Mortgage Disclosure Act (HMDA) data in the class repository https://github.com/jphall663/GWU_rml/tree/master/assignments/data
  * 19,830 rows of data
  * There is no difference in the columns between training and evaluation (or test) data, besides the high_priced column (target variable) not existing in the test data
  
## Model Details
  * **Columns used in our final model:**'property_value_std', 'no_intro_rate_period_std', 'loan_amount_std', 'income_std', 'conforming', 'intro_rate_period_std', 'debt_to_income_ratio_std', 'term_360'
  * **Target column in our final model:**'high_priced
  * **Final model type:** Explainable Boosting Machine
  * **Final model software used:** Python 3.6+, InterpretML v0.2.5.
  * **Final model hyperparameter settings:** {'max_bins': [128, 256, 512], 'max_interaction_bins': [16, 32, 64], 'interactions': [5, 10, 15], 'outer_bags': [4, 8, 12], 'inner_bags': [0, 4], 'learning_rate': [0.001, 0.01, 0.05], 'validation_size': [0.1, 0.25, 0.5], 'min_samples_leaf': [1, 2, 5, 10], 'max_leaves': [1, 3, 5]}, 'early_stopping_rounds': 100.0, 'n_jobs': 4, 'random_state': 12345}
 
## Quantitative Analysis

| Partition | AUC |
| :--------:|:---:|
| Valid | 0.7912 |

| Compare v. Control  | AIR |
| :-----------------: | :---: |
| Asian people vs. White people | 1.129 | 
|Black people vs. White people| 0.819 |
|Females vs. Males | 0.966 |

* This analysis showed that even with a selective cutoff of 0.17, less discriminatory models are available. The new set of features and hyperparameters leads to a ~13% increase in AIR with a ~5% decrease in AUC.

* **AUC of Other models tried**: Monotonic XGBoost - Validation AUC 0.7920 and Elastic Net - Validation AUC 0.7538

**Plots:**

 
 Correlation Heatmap - used to gauge an idea for which features may be more 'important':
 
 ![image](https://user-images.githubusercontent.com/89049995/174693537-dbb8b160-8dd8-45b0-b0f0-af9aa87965de.png)

  
 
  Comparison of local feature importance across models:
  
  ![image](https://user-images.githubusercontent.com/89049995/174693795-29cf173c-673c-4e3e-9534-8dd3ed3f2bf5.png)

  
  Utility Function for partial dependence:
  
  ![image](https://user-images.githubusercontent.com/89049995/174693858-b0ec807b-f55f-4d38-ae47-c4f367fc3058.png)


  Plot of partial dependence from each model for Debt-to-Income ratio:
  
  ![image](https://user-images.githubusercontent.com/89049995/174693927-d70ea3b8-943a-4752-8b43-d45bcd95addc.png)


  Display of grid search results as plot of AIR vs. AUC for EBMs:
  
  ![image](https://user-images.githubusercontent.com/89049995/174693974-9ac27786-f93a-4bc9-acce-22d0dddbb510.png)


Stolen decision tree model view- this can be used as a sandbox for subsequent attacks:

![image](https://user-images.githubusercontent.com/89049995/174694055-e80f5e1e-35ac-4279-8f0c-b942c19115b4.png)


Variable Importance for Stolen Model:

![image](https://user-images.githubusercontent.com/89049995/174694068-1840af4d-4115-4ef7-9e82-1ee7740f3bd2.png)


Residual Analysis - used to detect outliers to improve the model:

![image](https://user-images.githubusercontent.com/89418186/176975247-b48fdf74-cdf1-4131-95f6-0edd485fa94b.png)


## Ethical considerations
  * The model can improve accuracy but if there is bias in the data collection then it would lead to conflicting results.
  * Possible risks of disparate impact due to relatively low Hispanic to White and Black to White AIR's (even though they are both above 0.8)
  * The data in this model was predominantly those of white consumers. Whilst this may be proportionate to the current demographic landscape of the USA, this is projected to decrease in the coming years. A better model might account for such projections when choosing the original dataset. 
  * This sample is very specific to the USA. If this model was used outside the USA, there could be many undesired consequences as other countries would have compeltely different consumer behaviour, demographics etc.
  * The model will become less useful if the economic conditions change dramatically. For example, if the model is built using pre-pandemic time period data, it would become less useful if we are trying to use the model during the pandemic. 
  * Although this model was tested and remediated for bias, there is much more to bias than models and data, and this model should be monitored for bias issues moving forward.
