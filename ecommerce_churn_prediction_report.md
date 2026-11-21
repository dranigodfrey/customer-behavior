# Technical Report

## E-commerce Customer Churn Prediction Using Machine Learning

| | |
|---|---|
| **Dataset** | E-commerce Customer Behavior Dataset |
| **Dataset Source** | Kaggle – `srisyra02/ecommerce-customer-behavior-dataset` (file: `eCommerce-dataset.csv`, version 2) |
| **Project Type** | Supervised Machine Learning – Binary Classification |
| **Programming Language** | Python |
| **Development Environment** | Google Colab / Jupyter Notebook |
| **Status** | Methodology implemented and experiments executed; results reported in Chapter Four |

---

## Table of Contents

- [Abstract](#abstract)
- [Chapter One: Introduction](#chapter-one-introduction)
- [Chapter Two: Dataset Description](#chapter-two-dataset-description)
- [Chapter Three: Methodology](#chapter-three-methodology)
- [Chapter Four: Experimental Results and Analysis](#chapter-four-experimental-results-and-analysis)
- [Chapter Five: Model Implementation and Deployment](#chapter-five-model-implementation-and-deployment)
- [Chapter Six: Limitations and Ethical Considerations](#chapter-six-limitations-and-ethical-considerations)
- [Chapter Seven: Conclusion and Recommendations](#chapter-seven-conclusion-and-recommendations)
- [References](#references)
- [Appendix A: Project Technologies](#appendix-a-project-technologies)
- [Appendix B: Final Model Configuration](#appendix-b-final-model-configuration)

---

## Abstract

Customer churn is a significant challenge for e-commerce and subscription-based businesses because losing existing customers can negatively affect revenue, profitability, and long-term customer relationships. Identifying customers who are likely to discontinue using a service enables businesses to implement targeted customer retention strategies.

This project develops a machine learning-based customer churn prediction system using the E-commerce Customer Behavior Dataset obtained from Kaggle. The dataset contains 50,000 customer records and 25 features representing demographic characteristics, customer engagement, purchasing behavior, and financial information. Of these customers, 28.9% had churned.

The project employs supervised machine learning to classify customers as churned or retained. The target variable is `Churned`, where `1` represents a churned customer and `0` represents a retained customer.

The methodology comprises data acquisition, exploratory data analysis, data inspection, feature preprocessing, model development, hyperparameter optimization, model evaluation, and model serialization. Logistic Regression, Decision Tree, and Random Forest classifiers were compared on a stratified 20% hold-out test set of 10,000 customers.

Random Forest performed best. After randomized hyperparameter search, the tuned Random Forest achieved an accuracy of 0.9094, precision of 0.8520, recall of 0.8308, F1-score of 0.8413, and ROC-AUC of 0.9254 on the test set. The most influential predictors were `Customer_Service_Calls`, `Lifetime_Value`, `Cart_Abandonment_Rate`, and `Age`. The trained pipeline was saved with Joblib and successfully reloaded to generate predictions and churn probabilities for a new customer record.

These results are promising but should be interpreted cautiously, because of potential data leakage, unresolved data-quality anomalies, and the use of the test set for model selection (see Chapter Six).

**Keywords:** Machine Learning, Customer Churn, E-commerce, Classification, Logistic Regression, Decision Tree, Random Forest, Predictive Analytics.

---

## Chapter One: Introduction

### 1.1 Background

The growth of e-commerce and subscription-based platforms has increased the importance of understanding customer behavior. Businesses collect substantial amounts of customer information, including purchasing patterns, login frequency, website engagement, customer service interactions, and transaction histories.

One of the major challenges facing these businesses is customer churn. Customer churn occurs when an existing customer stops purchasing products or using a company's services.

Traditional customer retention approaches often rely on historical reports and reactive strategies. However, machine learning provides opportunities to identify behavioral patterns associated with customer churn and predict which customers are at risk.

Machine learning algorithms can analyze historical customer data, learn relationships between customer characteristics and churn outcomes, and generate predictions for previously unseen customers.

This project applies machine learning techniques to an e-commerce customer behavior dataset to develop a customer churn prediction system.

### 1.2 Problem Statement

E-commerce businesses may experience customer losses because they cannot accurately identify customers who are likely to discontinue their engagement with the platform.

Without predictive analytical tools, businesses may rely on generalized marketing campaigns and reactive customer retention strategies.

Such approaches can result in inefficient resource allocation, missed retention opportunities, and reduced customer lifetime value.

Therefore, there is a need for a predictive system that analyzes customer demographic, transactional, and engagement characteristics to identify customers with a higher probability of churning.

### 1.3 Project Aim

To develop and evaluate a supervised machine learning model capable of predicting customer churn using customer demographic, engagement, and transactional information.

### 1.4 Specific Objectives

The project aims to:

1. Acquire and understand the e-commerce customer behavior dataset.
2. Perform exploratory data analysis to identify patterns associated with customer churn.
3. Clean and preprocess the dataset to make it suitable for machine learning.
4. Identify relevant features for customer churn prediction.
5. Develop multiple supervised classification models.
6. Compare model performance using appropriate evaluation metrics.
7. Optimize the selected model through hyperparameter tuning.
8. Save the trained model and use it for predictions on new customer records.
9. Demonstrate how the saved model can be used to predict churn for a new customer.

### 1.5 Research Questions

The project seeks to answer the following questions:

1. Which customer characteristics are associated with churn?
2. Can customer churn be predicted using demographic and behavioral data?
3. Which of the selected classification algorithms provides the best predictive performance?
4. Which evaluation metrics are most appropriate for assessing churn prediction?
5. How can the trained model be used to support customer retention decisions?

Answers to these questions are summarized in [Section 4.9](#49-answers-to-the-research-questions).

### 1.6 Scope of the Project

The project focuses on predicting customer churn using structured tabular data.

It covers:

- Customer demographic characteristics.
- Customer engagement patterns.
- Purchasing behavior.
- Customer service interactions.
- Financial and transactional indicators.
- Supervised classification algorithms.
- Predictive model evaluation and model persistence.

The project does not include real-time integration with an e-commerce platform or automated marketing campaign execution. A Streamlit user interface was proposed as a future deployment step but is not part of the executed notebook.

---

## Chapter Two: Dataset Description

### 2.1 Dataset Source

The dataset was obtained from Kaggle using the KaggleHub Python library.

**Dataset URL:** <https://www.kaggle.com/datasets/srisyra02/ecommerce-customer-behavior-dataset>

The downloaded dataset (version 2) contains a single CSV file, `eCommerce-dataset.csv`. Loading it produced a DataFrame of **50,000 rows and 25 columns**, matching the dataset description. The earlier discrepancy between the stated number of features and the number of named columns in the supplied sample is therefore resolved: the full file contains all 25 variables, including the `Signup_Quarter` column.

### 2.2 Dataset Features

The dataset contains the following variables:

| Feature | Type | Description |
|---|---|---|
| `Age` | Numerical | Customer age |
| `Gender` | Categorical | Customer gender (3 unique values) |
| `Country` | Categorical | Customer country (8 unique values) |
| `City` | Categorical | Customer city (40 unique values) |
| `Membership_Years` | Numerical | Duration of membership |
| `Login_Frequency` | Numerical | Frequency of customer logins |
| `Session_Duration_Avg` | Numerical | Average session duration |
| `Pages_Per_Session` | Numerical | Average pages viewed per session |
| `Cart_Abandonment_Rate` | Numerical | Rate of abandoned shopping carts |
| `Wishlist_Items` | Numerical | Number of wishlist items |
| `Total_Purchases` | Numerical | Number of purchases |
| `Average_Order_Value` | Numerical | Average order value |
| `Days_Since_Last_Purchase` | Numerical | Days since the previous purchase |
| `Discount_Usage_Rate` | Numerical | Rate of discount usage |
| `Returns_Rate` | Numerical | Rate of product returns |
| `Email_Open_Rate` | Numerical | Email engagement rate |
| `Customer_Service_Calls` | Numerical | Number of customer service calls |
| `Product_Reviews_Written` | Numerical | Number of reviews written |
| `Social_Media_Engagement_Score` | Numerical | Social media engagement score |
| `Mobile_App_Usage` | Numerical | Mobile application usage |
| `Payment_Method_Diversity` | Numerical | Number or diversity of payment methods |
| `Lifetime_Value` | Numerical | Customer lifetime value |
| `Credit_Balance` | Numerical | Customer credit balance |
| `Churned` | Target | Customer churn status |
| `Signup_Quarter` | Categorical | Customer signup quarter (Q1–Q4) |

In total there are **20 numerical features**, **4 categorical features** (`Gender`, `Country`, `City`, `Signup_Quarter`), and the target variable.

> **Note:** The precise meaning, measurement units, and permissible values of each variable should be confirmed against the dataset documentation. Several variables contain values outside plausible ranges (see [Section 2.6](#26-data-quality-observations)).

### 2.3 Target Variable

The target variable is `Churned`.

| Value | Meaning | Customers | Share |
|---|---|---|---|
| `0` | Customer retained | 35,550 | 71.1% |
| `1` | Customer churned | 14,450 | 28.9% |

This makes the project a binary classification problem with moderate class imbalance (roughly 2.5 retained customers for every churned customer).

### 2.4 Feature Categories

The dataset features can be organized into four broad categories:

**Demographic features**

- `Age`
- `Gender`
- `Country`
- `City`

**Engagement features**

- `Login_Frequency`
- `Session_Duration_Avg`
- `Pages_Per_Session`
- `Email_Open_Rate`
- `Social_Media_Engagement_Score`
- `Mobile_App_Usage`

**Transactional features**

- `Total_Purchases`
- `Average_Order_Value`
- `Cart_Abandonment_Rate`
- `Wishlist_Items`
- `Returns_Rate`
- `Days_Since_Last_Purchase`
- `Discount_Usage_Rate`

**Financial and relationship features**

- `Membership_Years`
- `Lifetime_Value`
- `Credit_Balance`
- `Customer_Service_Calls`
- `Product_Reviews_Written`
- `Payment_Method_Diversity`
- `Signup_Quarter`

> These groupings are intended to support interpretation and analysis rather than represent formally verified causal categories.

### 2.5 Missing Values and Duplicates

No duplicate records were found. Fourteen features contain missing values, while the target and the remaining eleven features are complete.

| Feature | Missing Count | Missing (%) |
|---|---:|---:|
| `Social_Media_Engagement_Score` | 6,000 | 12.00 |
| `Credit_Balance` | 5,500 | 11.00 |
| `Mobile_App_Usage` | 5,000 | 10.00 |
| `Returns_Rate` | 4,491 | 8.98 |
| `Wishlist_Items` | 4,000 | 8.00 |
| `Product_Reviews_Written` | 3,500 | 7.00 |
| `Discount_Usage_Rate` | 3,500 | 7.00 |
| `Session_Duration_Avg` | 3,399 | 6.80 |
| `Pages_Per_Session` | 3,000 | 6.00 |
| `Days_Since_Last_Purchase` | 3,000 | 6.00 |
| `Email_Open_Rate` | 2,528 | 5.06 |
| `Payment_Method_Diversity` | 2,500 | 5.00 |
| `Age` | 2,495 | 4.99 |
| `Customer_Service_Calls` | 168 | 0.34 |

### 2.6 Data Quality Observations

The statistical summary revealed values that appear implausible for the described variables:

| Feature | Observed Range | Concern |
|---|---|---|
| `Age` | 5 to 200 | Ages of 200 are not realistic; very young ages are questionable for account holders. |
| `Cart_Abandonment_Rate` | 0 to ≈143.7 | A rate would normally not exceed 100%. |
| `Discount_Usage_Rate` | 0.24 to ≈116.6 | A rate would normally not exceed 100%. |
| `Total_Purchases` | −13 to ≈128.7 | Negative and non-integer purchase counts are not meaningful. |
| `Average_Order_Value` | 26.38 to ≈9,666 | Extreme outliers relative to a median of about 113. |

These values were **not corrected or removed** in the executed workflow. Tree-based models are relatively robust to such outliers, but they may still affect Logistic Regression and the interpretation of results. Cleaning these records is recommended as future work.

---

## Chapter Three: Methodology

### 3.1 Research Design

The project adopts a quantitative, experimental machine learning approach.

Historical customer information is used to train classification algorithms, and their predictive performance is assessed using a separate testing dataset.

The methodology follows a structured machine learning workflow:

1. Data acquisition.
2. Data inspection.
3. Exploratory data analysis.
4. Data preparation.
5. Feature preprocessing.
6. Dataset partitioning.
7. Model development.
8. Hyperparameter optimization.
9. Model evaluation.
10. Model serialization and reuse.

### 3.2 Data Acquisition

The dataset was downloaded using KaggleHub.

```python
import kagglehub

path = kagglehub.dataset_download(
    "srisyra02/ecommerce-customer-behavior-dataset"
)

print("Dataset location:", path)
```

The first CSV file in the downloaded directory (`eCommerce-dataset.csv`) was loaded into a Pandas DataFrame.

### 3.3 Data Inspection and Preparation

Initial inspection examined:

- Dataset dimensions (50,000 × 25).
- Column names and data types.
- Missing values per feature.
- Duplicate observations (none found).
- Descriptive statistics.
- Target variable distribution.

Records with a missing target would have been removed, and the target was verified to contain only the values 0 and 1. No records were dropped because the target was complete.

Missing feature values were handled inside the modeling pipeline rather than by dropping rows:

- **Numerical variables:** median imputation, which is less sensitive to extreme observations than mean imputation.
- **Categorical variables:** most-frequent-category imputation.

### 3.4 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer characteristics and investigate possible relationships between features and churn. The following were produced:

- Churn distribution bar chart and pie chart.
- Histograms of all numerical features.
- A correlation heatmap of numerical variables.
- Boxplots comparing churned and retained customers for ten selected features: `Age`, `Membership_Years`, `Login_Frequency`, `Session_Duration_Avg`, `Cart_Abandonment_Rate`, `Total_Purchases`, `Average_Order_Value`, `Days_Since_Last_Purchase`, `Discount_Usage_Rate`, and `Lifetime_Value`.

Findings are presented in [Section 4.3](#43-exploratory-data-analysis-findings).

### 3.5 Data Preprocessing

The features were divided into numerical and categorical variables.

#### Numerical preprocessing

Numerical features undergo:

- Median imputation for missing values.
- Standardization using `StandardScaler`.

#### Categorical preprocessing

Categorical features undergo:

- Most-frequent-value imputation.
- One-Hot Encoding (`handle_unknown="ignore"`), so categories unseen during training do not cause errors at prediction time.

A Scikit-learn `ColumnTransformer` combines these operations. Because the preprocessing is embedded in each model `Pipeline`, it is fitted on training data only, which prevents information leakage from the test set during preprocessing.

### 3.6 Training and Testing Data

The dataset was divided using a stratified split with a fixed random seed (`random_state=42`):

| Partition | Records |
|---|---:|
| Training | 40,000 (80%) |
| Testing | 10,000 (20%) |

Stratification preserved the class distribution. The test set contains 7,110 retained and 2,890 churned customers (28.9% churn).

### 3.7 Machine Learning Algorithms

Three classification algorithms were compared. Each used `class_weight="balanced"` to compensate for class imbalance.

| Algorithm | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| Decision Tree | `max_depth=12`, `min_samples_split=10`, `class_weight="balanced"` |
| Random Forest | `n_estimators=200`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight="balanced"` |

#### 3.7.1 Logistic Regression

Logistic Regression estimates the probability that a customer belongs to a particular class.

It was selected because:

- It provides a relatively simple classification baseline.
- It produces probability estimates.
- Its coefficients can support interpretation of feature relationships.
- It is computationally efficient.

#### 3.7.2 Decision Tree Classifier

Decision Tree classification uses a sequence of feature-based decisions to classify customers.

It was selected because:

- It can model nonlinear relationships.
- It can capture feature interactions.
- Its decision structure can be interpreted.
- It requires relatively limited feature scaling.

#### 3.7.3 Random Forest Classifier

Random Forest combines predictions from multiple decision trees.

It was selected because:

- It can capture complex relationships.
- It handles nonlinear patterns.
- It supports feature importance analysis.
- It often performs effectively on structured tabular data.

### 3.8 Model Training

Each algorithm was integrated into a Scikit-learn `Pipeline` containing the preprocessing and classification stages.

The models were fitted on the training dataset. The trained pipelines then generated class predictions and churn probability estimates for the testing dataset.

### 3.9 Hyperparameter Optimization

`RandomizedSearchCV` was used to optimize the Random Forest classifier.

| Setting | Value |
|---|---|
| Search iterations (`n_iter`) | 8 random combinations |
| Cross-validation | 3-fold (fits: 24) |
| Scoring metric | F1-score |
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 10, 20, 30, None |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

The F1-score was selected as the primary optimization metric to balance precision and recall. The search was fitted on the training data only.

### 3.10 Model Evaluation

The following evaluation metrics were used:

| Metric | Description |
|---|---|
| **Accuracy** | Measures the proportion of correctly classified observations. |
| **Precision** | Measures the proportion of predicted churners who actually churned. |
| **Recall** | Measures the proportion of actual churners correctly identified. |
| **F1-score** | Calculates the harmonic mean of precision and recall. |
| **ROC-AUC** | Measures the model's ability to distinguish between churned and retained customers across classification thresholds. |
| **Confusion Matrix** | Provides the number of true positives, true negatives, false positives, and false negatives. |

Recall is particularly important because failing to identify a customer who will churn may result in a missed retention opportunity.

However, precision is also important because false-positive predictions may lead to unnecessary retention expenditure.

### 3.11 Feature Importance

Feature importance analysis was performed using the impurity-based importance scores of the tuned Random Forest model.

The analysis identifies which variables contribute most to the model's predictive decisions. Importance scores are interpreted as indicators of predictive usefulness rather than evidence of causation.

### 3.12 Model Serialization

The tuned pipeline was saved using Joblib as `customer_churn_model.pkl`, and the list of expected input columns was saved as `customer_features.pkl`.

The saved artifact contains the preprocessing and classification pipeline, allowing the model to be loaded and used to predict churn for new customer records without repeating training.

---

## Chapter Four: Experimental Results and Analysis

### 4.1 Experimental Environment

The project was implemented using Python and the Scikit-learn machine learning framework.

The main libraries include:

- **Pandas** for data manipulation.
- **NumPy** for numerical computation.
- **Matplotlib** for visualization.
- **Seaborn** for statistical visualization.
- **Scikit-learn** for preprocessing, training, and evaluation.
- **Joblib** for model serialization.
- **KaggleHub** for dataset acquisition.

### 4.2 Dataset Analysis Results

| Item | Result |
|---|---|
| Records | 50,000 |
| Features | 24 (20 numerical, 4 categorical) plus the target |
| Duplicate records | 0 |
| Features with missing values | 14 (up to 12% missing per feature) |
| Missing target labels | 0 |
| Churned customers | 14,450 (28.9%) |
| Retained customers | 35,550 (71.1%) |

### 4.3 Exploratory Data Analysis Findings

Comparing churned and retained customers using boxplots suggests the following approximate patterns. These are visual comparisons of medians and spreads, not statistical tests, and they do not establish causation.

| Feature | Observed Pattern for Churned Customers |
|---|---|
| `Login_Frequency` | Lower (median roughly 8 logins vs. about 12 for retained customers). |
| `Session_Duration_Avg` | Shorter sessions (median roughly 22 vs. about 28). |
| `Cart_Abandonment_Rate` | Higher (median roughly 67 vs. about 55). |
| `Days_Since_Last_Purchase` | Longer time since the last purchase (median roughly 26 vs. about 19). |
| `Total_Purchases` | Slightly lower median. |
| `Discount_Usage_Rate` | Slightly lower median. |
| `Lifetime_Value` | Lower median, with a wider spread. |
| `Age` | Similar distributions, apart from extreme outliers (values of 150 and 200). |
| `Membership_Years` | Nearly identical distributions. |
| `Average_Order_Value` | Dominated by a few extreme outliers; no clear difference. |

In short, churned customers appear to log in less often, spend less time per session, abandon more carts, and have gone longer since their last purchase. Membership length and age showed little visible separation.

### 4.4 Model Performance Comparison

All models were evaluated on the same 10,000-customer test set.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7149 | 0.5047 | 0.7308 | 0.5970 | 0.7899 |
| Decision Tree | 0.8743 | 0.7807 | 0.7858 | 0.7832 | 0.8616 |
| Random Forest | 0.9057 | 0.8351 | 0.8394 | 0.8373 | 0.9243 |
| **Tuned Random Forest** | **0.9094** | **0.8520** | 0.8308 | **0.8413** | **0.9254** |

Key observations:

- **Logistic Regression** performed worst. Its high recall (0.7308) came at the cost of low precision (0.5047), meaning roughly half of the customers it flagged as churners were actually retained. This suggests a linear decision boundary is not well suited to this problem.
- **Decision Tree** performed substantially better than Logistic Regression, indicating that nonlinear relationships and feature interactions matter.
- **Random Forest** was the best of the three untuned models on every metric and was therefore selected for tuning.
- **Tuning** gave a modest improvement: F1 rose from 0.8373 to 0.8413, precision from 0.8351 to 0.8520, and ROC-AUC from 0.9243 to 0.9254. Recall decreased slightly, from 0.8394 to 0.8308. The gains are small, so the untuned Random Forest was already close to the achievable performance of this model family.

The best parameters found were `n_estimators=300`, `max_depth=None`, `min_samples_split=2`, and `min_samples_leaf=1`. The best mean cross-validated F1-score was **0.8365**, which is close to the test-set F1 of 0.8413 and indicates stable behavior across data folds.

### 4.5 Classification Report (Tuned Random Forest)

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Retained Customer | 0.93 | 0.94 | 0.94 | 7,110 |
| Churned Customer | 0.85 | 0.83 | 0.84 | 2,890 |
| **Accuracy** | | | **0.91** | 10,000 |
| Macro average | 0.89 | 0.89 | 0.89 | 10,000 |
| Weighted average | 0.91 | 0.91 | 0.91 | 10,000 |

The model is stronger on the majority (retained) class, as expected, but still identifies churners well.

### 4.6 Confusion Matrix Analysis

The confusion-matrix counts below were reconstructed from the reported precision, recall, and class support (default 0.5 threshold) and are consistent with the reported accuracy of 0.9094.

| | Predicted Retained | Predicted Churned |
|---|---:|---:|
| **Actual Retained** | 6,693 (true negatives) | 417 (false positives) |
| **Actual Churned** | 489 (false negatives) | 2,401 (true positives) |

- **False negatives (489):** churners the model missed, about 16.9% of all churners. These represent lost retention opportunities.
- **False positives (417):** retained customers incorrectly flagged, who might receive unnecessary retention offers.
- The two error types are of similar size at the default threshold.

### 4.7 ROC Analysis

The tuned Random Forest achieved a ROC-AUC of **0.9254**, indicating strong ability to rank churners above non-churners across thresholds. As a general guideline, an AUC close to 1.0 indicates excellent discrimination and an AUC of 0.5 indicates random ranking. AUC values are not guaranteed performance thresholds in production.

### 4.8 Feature Importance Analysis

The top features by Random Forest importance were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Customer_Service_Calls` | 0.1157 |
| 2 | `Lifetime_Value` | 0.1046 |
| 3 | `Cart_Abandonment_Rate` | 0.0834 |
| 4 | `Age` | 0.0694 |
| 5 | `Discount_Usage_Rate` | 0.0523 |
| 6 | `Average_Order_Value` | 0.0485 |
| 7 | `Days_Since_Last_Purchase` | 0.0481 |
| 8 | `Total_Purchases` | 0.0447 |
| 9 | `Email_Open_Rate` | 0.0431 |
| 10 | `Pages_Per_Session` | 0.0397 |
| 11 | `Session_Duration_Avg` | 0.0393 |
| 12 | `Mobile_App_Usage` | 0.0341 |
| 13 | `Returns_Rate` | 0.0293 |
| 14 | `Login_Frequency` | 0.0289 |
| 15 | `Social_Media_Engagement_Score` | 0.0276 |
| 16 | `Wishlist_Items` | 0.0264 |
| 17 | `Credit_Balance` | 0.0263 |
| 18 | `Membership_Years` | 0.0229 |
| 19 | `Product_Reviews_Written` | 0.0190 |
| 20 | `Payment_Method_Diversity` | 0.0098 |

Observations:

- Customer service contact, customer value, cart abandonment, and age carry the most predictive weight.
- Engagement and purchase-recency features (`Days_Since_Last_Purchase`, `Email_Open_Rate`, `Session_Duration_Avg`, `Login_Frequency`) contribute moderately.
- The top 20 features are all numerical; none of the one-hot encoded categorical variables (gender, country, city, signup quarter) ranked in the top 20.
- `Age` ranks fourth despite similar visual distributions across classes in the boxplots. Importance reflects usefulness to the model, possibly through nonlinear or interaction effects, and should not be read as a simple directional effect.
- Importance scores indicate predictive usefulness, **not causation**. A high rank for `Lifetime_Value` should also be checked for leakage (see Section 6.2).

### 4.9 Answers to the Research Questions

1. **Which customer characteristics are associated with churn?** Customer service calls, lifetime value, cart abandonment, age, discount usage, order value, and purchase recency were the most influential. Exploratory analysis also showed lower login frequency, shorter sessions, higher cart abandonment, and longer time since the last purchase among churners.
2. **Can churn be predicted from demographic and behavioral data?** Yes. The tuned Random Forest reached an F1-score of 0.8413 and ROC-AUC of 0.9254 on held-out data.
3. **Which algorithm performed best?** Random Forest, especially after tuning, clearly outperformed the Decision Tree and Logistic Regression.
4. **Which metrics are most appropriate?** Because 28.9% of customers churn, accuracy alone is insufficient. Recall, precision, F1-score, and ROC-AUC, together with the confusion matrix, give a fuller picture.
5. **How can the model support retention decisions?** It outputs a churn probability for each customer, which can be used to rank customers by risk and prioritize retention actions, with the threshold chosen according to business costs (Section 4.10).

### 4.10 Effect of the Prediction Threshold

The default classification threshold is 0.5. An illustrative alternative of **0.40** was tested:

| Threshold | Precision (Churned) | Recall (Churned) | F1 (Churned) | Accuracy |
|---|---:|---:|---:|---:|
| 0.50 (default) | 0.85 | 0.83 | 0.84 | 0.91 |
| 0.40 | 0.76 | 0.87 | 0.81 | 0.88 |

Lowering the threshold increased recall from 0.83 to 0.87, so more churners are identified, but precision fell from 0.85 to 0.76, so more retention offers would go to customers who would have stayed. The 0.40 value was illustrative. In practice the threshold should be selected on validation data using the actual cost of a missed churner relative to the cost of an unnecessary retention offer.

### 4.11 Interpretation of Results

The tuned Random Forest was selected as the final model on the basis of F1-score, recall, and ROC-AUC. It offers a good balance between catching churners and avoiding false alarms at the default threshold, and the threshold can be adjusted when recall is more valuable.

A high accuracy score alone is not sufficient evidence of a successful churn prediction system, given the class imbalance. The results should also be read alongside the caveats in Chapter Six, particularly the potential for data leakage and the fact that the test set influenced model selection.

---

## Chapter Five: Model Implementation and Reuse

### 5.1 Model Serialization

The tuned pipeline was saved with Joblib as `customer_churn_model.pkl`, and the expected input column names as `customer_features.pkl`.

The saved model includes:

- Missing-value imputation.
- Numerical standardization.
- Categorical encoding.
- The tuned Random Forest classifier.

This ensures consistent preprocessing when the model is reused.

### 5.2 Prediction on New Customer Records

New customer information must be supplied with the same 24 input columns used during training, in the saved column order. The model returns:

- A predicted churn class (0 or 1).
- An estimated churn probability.

### 5.3 Demonstration on a New Customer

To demonstrate reuse, the saved model and feature list were reloaded. A test customer was created from the first dataset record with these modified values:

| Feature | Value |
|---|---:|
| `Age` | 28 |
| `Membership_Years` | 0.5 |
| `Login_Frequency` | 2 |
| `Days_Since_Last_Purchase` | 90 |
| `Cart_Abandonment_Rate` | 80 |

The model predicted **churn (class 1)** with an estimated churn probability of **53.67%**. This customer profile (new, rarely logging in, long time since the last purchase, high cart abandonment) is consistent with the churn patterns observed in the exploratory analysis. However, the probability is close to the 0.5 threshold, so this prediction is borderline and would flip at a higher threshold.

### 5.4 Application Interface

A Streamlit application that lets users upload customer records and download predictions was proposed, but it was **not implemented** in the executed notebook. It remains a recommended next step. A suitable interface would display:

- The total number of customers analyzed.
- The number of customers predicted to churn.
- Individual churn predictions and probabilities.
- Downloadable prediction results.

### 5.5 Potential Business Applications

The system can support several business activities:

- **Customer retention:** Identify customers who may require targeted engagement.
- **Marketing optimization:** Prioritize retention campaigns for customers with elevated predicted churn probabilities.
- **Customer relationship management:** Support customer segmentation and engagement planning.
- **Business intelligence:** Provide additional insights into patterns associated with churn.
- **Revenue protection:** Help prioritize retention resources based on predicted risk.

> Predictions should support business decisions rather than automatically determine customer treatment.

---

## Chapter Six: Limitations and Ethical Considerations

### 6.1 Dataset Limitations

The dataset is a structured, historical dataset and may not fully represent all real-world customer behaviors. Performance in production may differ from performance during experimental testing.

The data also contain implausible values (ages up to 200, abandonment and discount rates above 100%, negative and fractional purchase counts, and extreme order values) that were not cleaned in this study. Missing values affect up to 12% of some features and were imputed with simple statistics, which may not reflect the true underlying values.

### 6.2 Potential Data Leakage

Some variables, particularly `Lifetime_Value` (the second most important feature) and `Credit_Balance`, may contain information that is not available at the intended prediction time. Lifetime value, for example, may partly be computed from the customer's complete purchase history, including activity that occurs close to or after the churn event.

The project should establish a clear prediction date and verify the temporal availability of all features. Where future information is included in a feature, that feature should be excluded or reconstructed using only information available before the prediction date. Retraining without these features would show how much of the model's performance depends on them.

### 6.3 Class Imbalance

Churned customers make up 28.9% of the dataset. Class weighting was used to reduce bias toward the majority class, and evaluation went beyond accuracy to include precision, recall, F1-score, and ROC-AUC. Threshold adjustment (Section 4.10) offers a further way to trade precision against recall.

### 6.4 Model Selection and Evaluation Design

The best model was selected by comparing test-set F1-scores, and the illustrative threshold was also examined on the test set. This means the test set influenced model choices, so the reported metrics may be slightly optimistic. A cleaner design would use a separate validation set for model and threshold selection and reserve the test set for a single final evaluation. In addition, only an 8-iteration random search with 3-fold cross-validation was used, so a broader search might find different settings.

### 6.5 Model Interpretability

Random Forest can capture complex patterns but is less directly interpretable than Logistic Regression or a simple Decision Tree. Impurity-based feature importance can also be biased toward features with many distinct values. Feature importance should be supplemented with methods such as permutation importance or SHAP when deeper explanations are required.

### 6.6 Fairness and Privacy

Customer demographic information (age, gender, country, city) should be handled responsibly.

The model should be assessed for potential differences in prediction performance across relevant customer groups, which was not done in this study. Personal information should be protected, and predictions should not be used to unfairly discriminate against customers.

---

## Chapter Seven: Conclusion and Recommendations

### 7.1 Conclusion

This project developed and evaluated a machine learning framework for predicting customer churn in an e-commerce environment using a 50,000-customer dataset in which 28.9% of customers churned.

Three classification algorithms (Logistic Regression, Decision Tree, and Random Forest) were compared. Random Forest was clearly best, and after randomized hyperparameter tuning it reached an accuracy of 0.9094, precision of 0.8520, recall of 0.8308, F1-score of 0.8413, and ROC-AUC of 0.9254 on held-out data. The Decision Tree followed (F1 0.7832), and Logistic Regression performed worst (F1 0.5970), indicating that churn in this dataset depends on nonlinear patterns.

Customer service calls, lifetime value, cart abandonment, age, discount usage, order value, and purchase recency were the most influential predictors. Exploratory analysis showed that churned customers tend to log in less, have shorter sessions, abandon more carts, and have gone longer since their last purchase.

The final pipeline was saved with Joblib and successfully reloaded to produce a churn class and probability for a new customer record, showing that the model can be reused without retraining.

The results show that supervised machine learning can effectively support customer retention strategies. However, they should be treated with care: possible leakage from variables such as `Lifetime_Value`, uncorrected data-quality anomalies, and the use of the test set for model selection could all make the performance estimates optimistic.

### 7.2 Recommendations

The following improvements are recommended:

1. Investigate and, where necessary, remove or reconstruct potentially leaking features such as `Lifetime_Value` and `Credit_Balance`, then compare performance.
2. Clean implausible values (impossible ages, rates above 100%, negative purchase counts, extreme order values).
3. Use a separate validation set (or nested cross-validation) for model and threshold selection, keeping the test set untouched until the end.
4. Conduct temporal validation to assess how well the model predicts future customer churn.
5. Investigate additional algorithms, including gradient boosting, XGBoost, and LightGBM.
6. Apply explainability methods such as permutation importance and SHAP to understand individual predictions.
7. Optimize the classification threshold according to the actual cost of missed churners versus unnecessary retention offers.
8. Evaluate prediction performance across customer groups (for example by gender and country) to check for fairness.
9. Build the proposed Streamlit interface and integrate the prediction system with a customer relationship management platform.
10. Monitor model performance for changes in customer behavior and data distribution, and test the impact of retention interventions through controlled experiments.

---

## References

1. Srisyra02. *E-commerce Customer Behavior Dataset*. Kaggle. <https://www.kaggle.com/datasets/srisyra02/ecommerce-customer-behavior-dataset>
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. <https://jmlr.org/papers/v12/pedregosa11a.html>
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. <https://doi.org/10.1023/A:1010933404324>
4. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
5. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189–1232. <https://doi.org/10.1214/aos/1013203451>
6. Scikit-learn Developers. *Scikit-learn Documentation*. <https://scikit-learn.org/stable/>

---

## Appendix A: Project Technologies

| Component | Technology |
|---|---|
| Programming language | Python |
| Data acquisition | KaggleHub |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | Scikit-learn (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`) |
| Classification | Scikit-learn |
| Hyperparameter optimization | `RandomizedSearchCV` |
| Model storage | Joblib |
| Development environment | Google Colab / Jupyter Notebook |
---

## Appendix B: Final Model Configuration

| Component | Setting |
|---|---|
| Random seed | 42 |
| Train / test split | 80% / 20%, stratified (40,000 / 10,000) |
| Numerical preprocessing | Median imputation → `StandardScaler` |
| Categorical preprocessing | Most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")` |
| Classifier | `RandomForestClassifier` |
| `n_estimators` | 300 |
| `max_depth` | None |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `class_weight` | `"balanced"` |
| Best CV F1-score | 0.8365 |
| Test F1-score | 0.8413 |
| Test ROC-AUC | 0.9254 |
| Saved artifacts | `customer_churn_model.pkl`, `customer_features.pkl` |
---


## Appendix C: Project Presentation
Power Point Presentation | <https://docs.google.com/presentation/d/1uy_fZuC98oxo4RTpKXs8VIBSPPrMmHWT/edit?usp=sharing&ouid=115931841186463797251&rtpof=true&sd=true>
