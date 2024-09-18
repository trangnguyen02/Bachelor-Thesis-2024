# Bachelor-Thesis-2024

## 1. Project Overview
Bachelor Thesis titled "Application of Behavior Scoring Techniques in Trade Credit Limit Adjustment".

This project focuses on building a model predicting customers' risk of turning delinquent within the next year, allowing credit risk analysts to make appropriate adjusments to their credit line. The project is executed in cooperation with Royal FrieslandCampina's Account Receivables Team.

## 2. Project Impact
The model provides critical insights into what makes a borrower profile risky. With its strong performance in distinguishing between high-risk and low-risk customers, the model **reduces the throughput time of a credit review by 25%** and **enhances the consistency** of credit decision-making.
## 3. Technical Skills Demonstrated
- **Programming**: Python (Pandas, NumPy, Scikit-learn)
- **Machine Learning**: Logistic Regression, Random Forest
- **Evaluation**: ROC-AUC Score, G-mean, Brier Score
  
## 4. Data
- **Source**: Confidential data provided by Royal FrieslandCampina
- **Size**: 631 customers, 22 features.

## 5. Project Workflow
### 5.1 Data Collection
- Identify and gather data from SAP ERP and a REST API
### 5.2 Data Understanding
- Imbalanced dataset
### 5.3 Data Cleaning and Preprocessing
### 5.4 Feature Engineering
- Variables for trend analysis: Standardized variations in revenues, net profit, working capital, and equity over a 3-year period
- Key financial ratios: Current ratio, DSO ratio, Operating Profit Margin & ...
- Interaction variables
  ### 5.5 Modelling
- Four Models:
  1. Logistic Regession using tree-based feature selection 
  2. Logistic Regression using Recursive Feature Elimination (RFE)
  3. SMOTE + Random Forest Classifier
  4. SMOTE-ENN + Random Forest Classifier

## 6. Results
- Random Forest Classifier performs worse than the traditional Logistic Regression when features are selected using RFE techniques.
- Best model Logistic Regression using RFE has a ROC-AUC Score of 0.89 with a standard deviation of 0.04 using 200 Bootstrap samples.
