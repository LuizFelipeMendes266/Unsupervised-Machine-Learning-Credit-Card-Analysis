# Unsupervised Machine Learning : Credit Card Analysis
## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Project Description

We will analyze a credit card database. The goal is to separate customers based on their characteristics, i.e., clusters that best describe financial and behavioral patterns.

## Data

Summary of Features:

| Field            | Description                                                                                  |
|------------------|----------------------------------------------------------------------------------------------|
| customerID       | Unique identifier of the customer                                                            |
| gender           | Whether the customer is a male or a female                                                   |
| SeniorCitizen    | Whether the customer is a senior citizen or not (1, 0)                                        |
| Partner          | Whether the customer has a partner or not (Yes, No)                                           |
| Dependents       | Whether the customer has dependents or not (Yes, No)                                          |
| tenure           | Number of months the customer has stayed with the company                                     |
| PhoneService     | Whether the customer has a phone service or not (Yes, No)                                     |
| MultipleLines    | Whether the customer has multiple lines or not (Yes, No, No phone service)                     |
| InternetService  | Customer’s internet service provider (DSL, Fiber optic, No)                                   |
| OnlineSecurity   | Whether the customer has online security or not (Yes, No, No internet service)                |
| OnlineBackup     | Whether the customer has online backup or not (Yes, No, No internet service)                  |
| DeviceProtection | Whether the customer has device protection or not (Yes, No, No internet service)              |
| TechSupport      | Whether the customer has tech support or not (Yes, No, No internet service)                   |
| StreamingTV      | Whether the customer has streaming TV or not (Yes, No, No internet service)                   |
| StreamingMovies  | Whether the customer has streaming movies or not (Yes, No, No internet service)               |
| Contract         | The contract term of the customer (Month-to-month, One year, Two year)                         |
| PaperlessBilling | Whether the customer has paperless billing or not (Yes, No)                                   |
| PaymentMethod    | The customer’s payment method (Electronic check, Mailed check, Bank transfer, Credit card)    |
| MonthlyCharges   | The amount charged to the customer monthly                                                   |
| TotalCharges     | The total amount charged to the customer                                                      |
| Churn            | Whether the customer churned or not (Yes or No)                                               |


## Methodology

Pandas, Numpy, and Sklearn were the libraries used. The exploratory and descriptive analysis was all produced by the pandas_profile library, and considerations for the features are in the notebook itself. Regarding classifiers, the data was tested on SVM, Logistic Regression, and Random Forest. To evaluate the classifier's performance, the main metrics such as Recall, F1 Score, Accuracy, and ROC AUC curve were used.
I chose to use the BayerSearchCV to optimize the hyperparameters.

## Results

| Classifier         | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| LogisticRegression | 82.18    | 67.74     | 60.87  | 76.13    | 0.87    |
| RandomForest       | 82.18    | 68.33     | 59.42  | 75.89    | 0.87    |
| SVM                | 81.61    | 67.37     | 57.61  | 74.98    | 0.86    |


***In this case, we have very similar metrics. However, logistic regression may be the best choice due to slightly higher recall, as we have an imbalanced dataset.***

## Usage

### Libraries Used
- pandas: data manipulation library used to read, clean and manipulate the dataset.
- numpy: numerical computing library used to perform mathematical operations on the data.
- sklearn: machine learning library used to train and evaluate models.
- bayesSearchCv: hyperparameter optimization library used to optimize the models.
- feature_engine: feature engineering library used to encode categorical variables.
- pandas_profiling: exploratory data analysis library used to generate a report of the dataset.

### Installation

To install the required libraries, you can run the following command:

pip install pandas numpy scikit-learn bayesian-optimization feature-engine pandas-profiling

## References

https://www.kaggle.com/datasets/blastchar/telco-customer-churn
