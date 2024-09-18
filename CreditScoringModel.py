#!/usr/bin/env python
# coding: utf-8

### 1. Importing libraries

import os
import glob
import openpyxl
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import plotly.express as px
from scipy.stats import pointbiserialr
from scipy.stats import anderson, normaltest, shapiro
from scipy.stats import levene
from scipy.stats import mannwhitneyu

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline 

pip install pandas scikit-learn imbalanced-learn # type: ignore
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN

from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score 



### 2. Import datasets

### 3. Data Understanding

## 3.1 Descriptive statistics
## 3.2 Missing values
## 3.3 Setting appropriate data types
## 3.4 Outliers
## 3.5 Correlation of data
## 3.6 Comparison of good and bad customers' characteristics

### 4. Feature Engineering

## 4.1 Creating variables for trend analysis
# Standardized variations in revenues, net profit, working capital, and equity over a 3-year period

## 4.2 Creating variables representing key financial ratios
# Current ratio
# Quick ratio
# Debt ratio
# Cash ratio
# DSO ratio
# Operating profit margin

## 4.3 Filling missing values with KNN Imputer

## 4.4 Encoding categorical variables

## 4.5 Feature selection applying Filtering methods (independent of algorithms chosen for the classification task)
# Statistical tests: Selecting variables with the greatest differentiating power
# Correlation analysis between the outcome (good versus bad customers) and predictors

### 5. Credit Scoring Model with imbalanced dataset


def_predictors = ['Credit group', 'Country Risk',
       'Key_Cust', 'Large_Cust', 'Medium_Cust', 'Age', 'c_revenue', 'c_equity', 'c_workCap',
       'r_DSO', 'r_cash', 'r_quick', 'r_solven',
       'r_profit', 'Avg_arrears_t-1', 'Weight']
def_output = ["Default"]



# ## 5.1 SMOTE + Random Forest Classifier

# In[753]:


def_smote = ['Credit group', 'Country Risk', 'Key_Cust', 'Large_Cust', 'Medium_Cust', 'Age', 'c_revenue', 'c_equity', 'c_workCap', 'r_DSO', 'r_cash', 'r_quick', 'r_solven', 'r_profit', 'Avg_arrears_t-1', 'customerSegment']
X_train, X_test, y_train, y_test = train_test_split(sample[def_smote], sample[def_output], stratify=sample[def_output], test_size = 0.3, random_state = 42)
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()


# Step 1: Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy=1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Adjust sample weights after SMOTE

population_weights = []

# Calculate the sample distribution
samp = X_train_res['customerSegment'].value_counts(normalize=True).to_dict()

# Calculate weights
X_train_res['Weight'] = X_train_res['customerSegment'].apply(lambda x: pop[x] / samp[x])
weights = X_train_res['Weight']
# X_test["Weight"] = 0

X_train_features = X_train_res.iloc[:, :-1]

# Step 2: Apply StandardScaler
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define the RandomForestClassifier with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [2, 4, 6, 8, 10, 12, None],
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2', None]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(RandomForestClassifier(random_state=13), param_grid, scoring='roc_auc', cv=kf, n_jobs=-1)
grid_search.fit(X_train_res_scaled, y_train_res, sample_weight=weights)

# Get the best model
best_model = grid_search.best_estimator_

# Print best hyperparameters
print("Best hyperparameters found:", grid_search.best_params_)

# Evaluate on test set
y_rfc = best_model.predict(X_test_scaled)
y_rfc_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_rfc, target_names=['Non-Default', 'Default']))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_rfc_proba)}')

# Define the number of bootstrap samples
n_bootstrap_samples = 200

# Initialize lists to store the bootstrap metrics
roc_auc_scores = []
g_mean_scores = []
brier_scores = []

# Function to calculate G-Mean
def g_mean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    return np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

# Perform bootstrapping
for _ in range(n_bootstrap_samples):
    # Generate a bootstrap sample from the test set
    indices = np.random.choice(len(X_test_scaled), size=len(X_test_scaled), replace=True)
    X_test_bootstrap = X_test_scaled[indices]
    y_test_bootstrap = y_test[indices]

    # Check if both classes are present in the bootstrap sample
    if len(np.unique(y_test_bootstrap)) < 2:
        continue

    # Get predictions for the bootstrap sample
    y_bootstrap_pred = best_model.predict(X_test_bootstrap)
    y_bootstrap_proba = best_model.predict_proba(X_test_bootstrap)[:, 1]

    # Calculate metrics
    roc_auc_scores.append(roc_auc_score(y_test_bootstrap, y_bootstrap_proba))
    g_mean_scores.append(g_mean_score(y_test_bootstrap, y_bootstrap_pred))
    brier_scores.append(brier_score_loss(y_test_bootstrap, y_bootstrap_proba))

# Create a DataFrame with the results
results = pd.DataFrame({
    'Metric': ['ROC-AUC', 'G-Mean', 'Brier Score'],
    'Mean': [np.mean(roc_auc_scores), np.mean(g_mean_scores), np.mean(brier_scores)],
    'Standard Deviation': [np.std(roc_auc_scores), np.std(g_mean_scores), np.std(brier_scores)]
})

print(results)

train_scores = grid_poly.cv_results_['mean_test_score']
print(np.mean(train_scores))
print(np.std(train_scores))
grid_poly.cv_results_


## 5.2: SMOTE ENN link + Random Forest Classifier

# Step 1: Apply SMOTEENN
smote = SMOTEENN(random_state=42, sampling_strategy = 1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Adjust sample weights after SMOTE

popuplation_weights= {}

# Calculate the sample distribution
samp = X_train_res['customerSegment'].value_counts(normalize=True).to_dict()

# Calculate weights
X_train_res['Weight'] = X_train_res['customerSegment'].apply(lambda x: pop[x] / samp[x])
weights = X_train_res['Weight']
# X_test["Weight"] = 0

X_train_features = X_train_res.iloc[:, :-1]

# Step 2: Apply StandardScaler
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define the RandomForestClassifier with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [2, 4, 6, 8, 10, 12, None],
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2', None]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_enn = GridSearchCV(RandomForestClassifier(random_state=13), param_grid, scoring='roc_auc', cv=kf, n_jobs=-1)
grid_enn.fit(X_train_res_scaled, y_train_res, sample_weight=weights)

# Get the best model
best_enn = grid_search.best_estimator_

# Print best hyperparameters
print("Best hyperparameters found:", grid_search.best_params_)

# Evaluate on test set
y_pred_enn = best_enn.predict(X_test_scaled)
y_proba_enn = best_enn.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred_enn, target_names=['Non-Default', 'Default']))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_proba_enn)}')

# Calculate ROC curve and ROC-AUC score: SMOTEENN + RFC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
fpr_enn, tpr_enn, _ = roc_curve(y_test, y_proba_enn)
roc_auc_enn = auc(fpr_enn, tpr_enn)

# Define the number of bootstrap samples
n_bootstrap_samples = 200

# Initialize lists to store the bootstrap metrics
roc_auc_scores = []
g_mean_scores = []
brier_scores = []

# Function to calculate G-Mean
def g_mean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 1:  # handle cases where only one class is present in y_true
        return 0
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    return np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

# Perform bootstrapping
for _ in range(n_bootstrap_samples):
    # Generate a bootstrap sample from the test set
    indices = np.random.choice(len(X_test_scaled), size=len(X_test_scaled), replace=True)
    X_test_bootstrap = X_test_scaled[indices]
    y_test_bootstrap = y_test[indices]

    # Check if both classes are present in the bootstrap sample
    if len(np.unique(y_test_bootstrap)) < 2:
        continue

    # Get predictions for the bootstrap sample
    y_bootstrap_pred = best_enn.predict(X_test_bootstrap)
    y_bootstrap_proba = best_enn.predict_proba(X_test_bootstrap)[:, 1]

    # Calculate metrics
    roc_auc_scores.append(roc_auc_score(y_test_bootstrap, y_bootstrap_proba))
    g_mean_scores.append(g_mean_score(y_test_bootstrap, y_bootstrap_pred))
    brier_scores.append(brier_score_loss(y_test_bootstrap, y_bootstrap_proba))

# Create a DataFrame with the results
results = pd.DataFrame({
    'Metric': ['ROC-AUC', 'G-Mean', 'Brier Score'],
    'Mean': [np.mean(roc_auc_scores), np.mean(g_mean_scores), np.mean(brier_scores)],
    'Standard Deviation': [np.std(roc_auc_scores), np.std(g_mean_scores), np.std(brier_scores)]
})

print(results)

## 5.3 Base Line Model: Using class_weight + Logistic Regression

base_preds = ['Credit group', 'Key_Cust', 'Large_Cust', 'Medium_Cust', 'Age', 'Country Risk', 'Avg_arrears_t-1',
              'Weight']

# Stratified splitting, ensuring that both train & test sets have the same distribution of classes
X_train, X_test, y_train, y_test = train_test_split(sample[base_preds], sample[def_output], stratify=sample[def_output], test_size = 0.3, random_state = 42)
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

# Step 2: Scale the Data
scaler = StandardScaler()
X_train_lin = scaler.fit_transform(X_train.iloc[:, :-1])
X_test_lin = scaler.transform(X_test.iloc[:, :-1])

# Step 3: Perform Grid Search with Logistic Regression
log_base = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter = 10000)

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}

# Perform Grid Search with Cross-Validation
grid_base = GridSearchCV(log_base, param_grid, cv=5, scoring='roc_auc')
grid_base.fit(X_train_lin, y_train, sample_weight=X_train["Weight"]) 
best_base = grid_base.best_estimator_
y_pred_base = best_base.predict(X_test_lin)
y_proba_base = best_base.predict_proba(X_test_lin)[:, :-1]

print(classification_report(y_test, y_pred_base))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba_base))

coefficients = best_base.coef_[0]

# Get feature names
feature_names = X_train.iloc[:, :-1].columns

# Create a DataFrame for visualization
coef_base = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort by the absolute value of coefficients
coef_base['Abs_Coefficient'] = coef_base['Coefficient'].abs()
coef_base = coef_base.sort_values(by='Abs_Coefficient', ascending=False)

# Visualize the coefficients
plt.figure(figsize=(10, 8))
plt.barh(coef_base['Feature'], coef_base['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('')
plt.title('Baseline Model')
plt.show()


# In[47]:


# Calculate ROC curve and ROC-AUC score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_base)
roc_auc_base = auc(fpr_base, tpr_base)


## 5.4: Logistric Regression + Tree-base feature selection
X_train_features = X_train.iloc[:, :-1]
sample_weights = X_train.iloc[:, -1]

# Step 1: Generate Polynomial Features (interaction only)
poly = PolynomialFeatures(interaction_only = True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_features)
X_test_poly = poly.transform(X_test.iloc[:, :-1])

# Step 2: Scale the Data
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Step 3: Feature Selection with ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(X_train_poly_scaled, y_train)
selector = SelectFromModel(estimator=clf, prefit=True)
X_train_poly_selected = selector.transform(X_train_poly_scaled)
X_test_poly_selected = selector.transform(X_test_poly_scaled) 

# Step 4: Perform Grid Search with Logistic Regression
log_poly = LogisticRegression(solver='liblinear', class_weight='balanced')

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}

# Perform Grid Search with Cross-Validation
grid_poly = GridSearchCV(log_poly, param_grid, cv=5, scoring='roc_auc')
grid_poly.fit(X_train_poly_selected, y_train, sample_weight=sample_weights)
best_poly = grid_poly.best_estimator_

# Predict on the test set
y_pred_poly = best_poly.predict(X_test_poly_selected)
y_pred_proba_poly = best_poly.predict_proba(X_test_poly_selected)[:, 1]

# Evaluate the performance
print(classification_report(y_test, y_pred_poly, target_names=['Non-Default', 'Default']))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_poly)}')

# Get the coefficients of the best model
coefficients = best_poly.coef_[0]

# Get feature names for the polynomial features
selected_features = selector.get_support(indices=True)
feature_names = poly.get_feature_names_out(X_train_features.columns)
selected_feature_names = [feature_names[i] for i in selected_features]

# Create a DataFrame for visualization
coef_df = pd.DataFrame({'Feature': selected_feature_names, 'Coefficient': coefficients})

# Sort by the absolute value of coefficients
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False).head(10)

# Visualize the coefficients
plt.figure(figsize=(10, 8))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Tree-based feature selection')
plt.show()


# In[836]:

# Define the number of bootstrap samples
n_bootstrap_samples = 200

# Initialize lists to store the bootstrap metrics
roc_auc_scores = []
g_mean_scores = []
brier_scores = []

# Function to calculate G-Mean
def g_mean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 1:  # handle cases where only one class is present in y_true
        return 0
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    return np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

# Perform bootstrapping
for _ in range(n_bootstrap_samples):
    # Generate a bootstrap sample from the test set
    indices = np.random.choice(len(X_test_poly_selected), size=len(X_test_poly_selected), replace=True)
    X_test_bootstrap = X_test_poly_selected[indices]
    y_test_bootstrap = y_test[indices]

    # Check if both classes are present in the bootstrap sample
    if len(np.unique(y_test_bootstrap)) < 2:
        continue

    # Get predictions for the bootstrap sample
    y_bootstrap_pred = best_poly.predict(X_test_bootstrap)
    y_bootstrap_proba = best_poly.predict_proba(X_test_bootstrap)[:, 1]

    # Calculate metrics
    roc_auc_scores.append(roc_auc_score(y_test_bootstrap, y_bootstrap_proba))
    g_mean_scores.append(g_mean_score(y_test_bootstrap, y_bootstrap_pred))
    brier_scores.append(brier_score_loss(y_test_bootstrap, y_bootstrap_proba))

# Create a DataFrame with the results
results = pd.DataFrame({
    'Metric': ['ROC-AUC', 'G-Mean', 'Brier Score'],
    'Mean': [np.mean(roc_auc_scores), np.mean(g_mean_scores), np.mean(brier_scores)],
    'Standard Deviation': [np.std(roc_auc_scores), np.std(g_mean_scores), np.std(brier_scores)]
})

print(results)


# In[55]:


fpr_tree, tpr_tree, _ = roc_curve(y_test, y_pred_proba_poly)
roc_auc_tree = auc(fpr_tree, tpr_tree)


## 5.4 Logistic Regression + RFE Feature Selection
# In[789]:

X_train, X_test, y_train, y_test = train_test_split(sample[def_chosen], sample[def_output], stratify=sample[def_output], test_size = 0.3, random_state = 42)
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()
X_train_features = X_train.iloc[:, 2:-1]
sample_weights = X_train["Weight"]
X_test_features = X_test.iloc[:, 2:-1]

# Step 1: Generate Polynomial Features (interaction only)
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_features)
X_test_poly = poly.transform(X_test_features)

# Step 2: Scale the Data
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Step 3: Feature Selection with RFE
log_reg = LogisticRegression(solver='liblinear', class_weight='balanced')
selector = RFE(estimator=log_reg, n_features_to_select=17, step=1)
selector = selector.fit(X_train_poly_scaled, y_train)
X_train_poly_selected = selector.transform(X_train_poly_scaled)
X_test_poly_selected = selector.transform(X_test_poly_scaled)

# Step 4: Perform Grid Search with Logistic Regression
log_rfe = LogisticRegression(solver='liblinear', class_weight='balanced')

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}

# Perform Grid Search with Cross-Validation
grid_rfe = GridSearchCV(log_rfe, param_grid, cv=5, scoring='roc_auc')
grid_rfe.fit(X_train_poly_selected, y_train, sample_weight=sample_weights)
best_rfe = grid_rfe.best_estimator_

#Get label for train set
X_train["Default_proba"] = best_rfe.predict_proba(X_train_poly_selected)[:, 1]

# Predict on the test set
y_pred_rfe = best_rfe.predict(X_test_poly_selected)
y_proba_rfe = best_rfe.predict_proba(X_test_poly_selected)[:, 1]
X_test["Default_proba"] = y_proba_rfe

# Evaluate the performance
print(classification_report(y_test, y_pred_rfe, target_names=['Non-Default', 'Default']))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_proba_rfe)}')

# Get the coefficients of the best model
coefficients = best_rfe.coef_[0]

# Get feature names for the polynomial features
selected_features = selector.get_support(indices=True)
feature_names = poly.get_feature_names_out(X_train_features.columns)
selected_feature_names = [feature_names[i] for i in selected_features]

# Create a DataFrame for visualization
coef_rfe = pd.DataFrame({'Feature': selected_feature_names, 'Coefficient': coefficients})

# Sort by the absolute value of coefficients
coef_rfe['Abs_Coefficient'] = coef_rfe['Coefficient'].abs()
coef_rfe = coef_rfe.sort_values(by='Abs_Coefficient', ascending=False)

# Visualize the coefficients
plt.figure(figsize=(10, 8))
plt.barh(coef_rfe['Feature'], coef_rfe['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('')
plt.title('LR with RFE feature selection')
plt.show()


# In[790]:


# Define the number of bootstrap samples
n_bootstrap_samples = 200

# Initialize lists to store the bootstrap metrics
roc_auc_scores = []
g_mean_scores = []
brier_scores = []

# Function to calculate G-Mean
def g_mean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 1:  # handle cases where only one class is present in y_true
        return 0
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    return np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

# Perform bootstrapping
for _ in range(n_bootstrap_samples):
    # Generate a bootstrap sample from the test set
    indices = np.random.choice(len(X_test_poly_selected), size=len(X_test_poly_selected), replace=True)
    X_test_bootstrap = X_test_poly_selected[indices]
    y_test_bootstrap = y_test[indices]

    # Check if both classes are present in the bootstrap sample
    if len(np.unique(y_test_bootstrap)) < 2:
        continue

    # Get predictions for the bootstrap sample
    y_bootstrap_pred = best_rfe.predict(X_test_bootstrap)
    y_bootstrap_proba = best_rfe.predict_proba(X_test_bootstrap)[:, 1]

    # Calculate metrics
    roc_auc_scores.append(roc_auc_score(y_test_bootstrap, y_bootstrap_proba))
    g_mean_scores.append(g_mean_score(y_test_bootstrap, y_bootstrap_pred))
    brier_scores.append(brier_score_loss(y_test_bootstrap, y_bootstrap_proba))

# Create a DataFrame with the results
results = pd.DataFrame({
    'Metric': ['ROC-AUC', 'G-Mean', 'Brier Score'],
    'Mean': [np.mean(roc_auc_scores), np.mean(g_mean_scores), np.mean(brier_scores)],
    'Standard Deviation': [np.std(roc_auc_scores), np.std(g_mean_scores), np.std(brier_scores)]
})

print(results)


## Comparing results 
# In[743]:


plt.figure(figsize=(10, 8))
plt.plot(fpr_base, tpr_base, lw=2, alpha=0.8, label=f'Baseline Model')
plt.plot(fpr_rfc, tpr_rfc, lw=2, alpha=0.8, label=f'SMOTE + RFC')
plt.plot(fpr_enn, tpr_enn, lw=2, alpha=0.8, label=f'SMOTE-ENN + RFC')
plt.plot(fpr_rfe, tpr_rfe, lw=2, alpha=0.8, label=f'RFE + LogReg')
plt.plot(fpr_tree, tpr_tree, lw=2, alpha=0.8, label=f'Tree-based + LogReg')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


#  

# Add default probability to the sample using the best model

# In[67]:


def_chosen= ['Customer', 'Credit rep.group', 'Country Risk', 'Key_Cust', 'Large_Cust', 'Medium_Cust', 'Age', 'c_revenue', 'c_equity', 'c_workCap', 'r_DSO', 'r_cash', 'r_quick', 'r_solven', 'r_profit', 'Avg_arrears_t-1', 'Weight']

from sklearn.feature_selection import RFE
X_train, X_test, y_train, y_test = train_test_split(sample[def_chosen], sample[def_output], stratify=sample[def_output], test_size = 0.3, random_state = 42)
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()
X_train_features = X_train.iloc[:, 2:-1]
sample_weights = X_train["Weight"]
X_test_features = X_test.iloc[:, 2:-1]

# Step 1: Generate Polynomial Features (interaction only)
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_features)
X_test_poly = poly.transform(X_test_features)

# Step 2: Scale the Data
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Step 3: Feature Selection with RFE
log_reg = LogisticRegression(solver='liblinear', class_weight='balanced')
selector = RFE(estimator=log_reg, n_features_to_select=17, step=1)
selector = selector.fit(X_train_poly_scaled, y_train)
X_train_poly_selected = selector.transform(X_train_poly_scaled)
X_test_poly_selected = selector.transform(X_test_poly_scaled)

# Step 4: Perform Grid Search with Logistic Regression
log_rfe = LogisticRegression(solver='liblinear', class_weight='balanced')

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}

# Perform Grid Search with Cross-Validation
grid_rfe = GridSearchCV(log_rfe, param_grid, cv=5, scoring='roc_auc')
grid_rfe.fit(X_train_poly_selected, y_train, sample_weight=sample_weights)
best_rfe = grid_rfe.best_estimator_

#Get label for train set
X_train["Default_proba"] = best_rfe.predict_proba(X_train_poly_selected)[:, 1]

# Predict on the test set
y_pred_rfe = best_rfe.predict(X_test_poly_selected)
y_proba_rfe = best_rfe.predict_proba(X_test_poly_selected)[:, 1]
X_test["Default_proba"] = y_proba_rfe
