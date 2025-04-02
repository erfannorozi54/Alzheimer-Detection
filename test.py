import numpy as np
import pandas as pd
import logging
import re # Keep re in case needed elsewhere, though not in the provided snippet
from collections import Counter, defaultdict # Keep in case needed elsewhere

# Scikit-learn imports
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler # Keep StandardScaler in case needed later
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Keep RF for potential future use
from sklearn.decomposition import PCA # Keep PCA for potential future use

# CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_INSTALLED = True
except ImportError:
    CATBOOST_INSTALLED = False
    print("Warning: CatBoost not installed. CatBoost model will be skipped.")

# xgboost - Keep import in case needed later
try:
    from xgboost import XGBClassifier
    XGBOOST_INSTALLED = True
except ImportError:
    XGBOOST_INSTALLED = False
    # print("Warning: XGBoost not installed.") # Optional warning

# For fetching the Alzheimer’s dataset from UCIML repo
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42

# Configure logging (optional, can be removed if not needed for this version)
# logging.basicConfig(filename='model_selection.log', level=logging.INFO,
#                     format='%(asctime)s - %(message)s', filemode='w')

# =====================================================================================
# === DATA LOADING AND PREPROCESSING ===
# =====================================================================================
print("--- Starting Data Loading and Preprocessing ---")

# 1. Load Data
print("Loading data from UCIML repository...")
darwin = fetch_ucirepo(id=732)
# Load features and target
X_features = darwin.data.features
y_series = darwin.data.targets.iloc[:, 0] # Assuming target is the first column in targets

# Explicitly drop the 'ID' column if it exists and store remaining feature names
if 'ID' in X_features.columns:
    X_df = X_features.drop('ID', axis=1)
    print("Dropped 'ID' column.")
else:
    # If no 'ID' column, check if the first column should be dropped based on previous attempts
    # This assumes the first column is the one causing the shape mismatch if 'ID' isn't the name
    print("No 'ID' column found, attempting to drop the first column based on index.")
    if X_features.shape[1] > 450: # Check if there's an extra column
         X_df = X_features.iloc[:, 1:]
         print("Dropped the first column based on index.")
    else:
         X_df = X_features # Assume all columns are features
         print("Assuming all loaded columns are features.")

feature_names = X_df.columns.tolist() # Store the correct feature names
print(f"Dataset loaded with {X_df.shape[0]} samples and {X_df.shape[1]} features.")

# Ensure data is numeric where possible, handle potential errors
for col in feature_names: # Iterate using the stored feature names
    X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

# Handle NaNs introduced by coerce using SimpleImputer with median strategy
print("Imputing missing values (NaN) using median strategy...")
imputer = SimpleImputer(strategy='median')
# Fit on the feature DataFrame X_df and transform it
X_imputed = imputer.fit_transform(X_df)
# Convert back to DataFrame using the stored feature names
X_df = pd.DataFrame(X_imputed, columns=feature_names) # Use stored feature_names
print("Missing value imputation complete.")

# 2. Encode Target Variable
print("Encoding target variable...")
if y_series.dtype == "object":
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    print(f"Target encoded. Classes: {label_encoder.classes_}")
else:
    y_encoded = y_series.to_numpy() # Ensure y is numpy array
    print("Target is already numeric.")

# Convert y_encoded to pandas Series for consistency if needed later, though numpy is fine for models
y = pd.Series(y_encoded, name=y_series.name if hasattr(y_series, 'name') else 'target')
X = X_df # Use the original DataFrame (or numpy array if preferred)

# Note: The snippet uses X and y directly for GridSearch and LOOCV,
# implying no scaling or PCA is applied before these steps in this specific workflow.
# Scaling is often recommended *before* SVM and Logistic Regression.
# If scaling is desired *within* LOOCV, a Pipeline should be used.
# For simplicity here, we follow the snippet's direct use of X, y.

# Train-test split (as per snippet, though not used in LOOCV evaluation)
# Using 0.1 test size as specified in the snippet
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
)
print(f"Split for reference: Training samples: {X_train_split.shape[0]}, Test samples: {X_test_split.shape[0]}")
print(f"Number of features: {X_train_split.shape[1]}")

print("--- Data Loading and Preprocessing Complete --- \n")


# =====================================================================================
# === MODEL DEFINITIONS AND GRID SEARCH ===
# =====================================================================================
print("--- Starting Grid Search for Hyperparameter Tuning ---")

models_to_tune = {}
param_grids = {}

# CatBoost
if CATBOOST_INSTALLED:
    # Check if GPU is available for CatBoost, otherwise use CPU
    try:
        # A simple check, might need refinement based on actual CatBoost/GPU setup
        import torch
        task_type = 'GPU' if torch.cuda.is_available() else 'CPU'
    except ImportError:
        task_type = 'CPU'
    print(f"Setting CatBoost task_type to: {task_type}")

    models_to_tune['CatBoost'] = CatBoostClassifier(
        random_state=RANDOM_STATE, verbose=0, task_type=task_type
    )
    param_grids['CatBoost'] = {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'iterations': [100, 200, 300]
    }
else:
    print("Skipping CatBoost tuning as it's not installed.")

# SVM
models_to_tune['SVM'] = SVC(probability=True, random_state=RANDOM_STATE)
param_grids['SVM'] = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Logistic Regression
models_to_tune['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
param_grids['Logistic Regression'] = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'] # 'l1' requires solver='liblinear' or 'saga'
    # 'solver': ['liblinear'] # Add if using 'l1' penalty
}

best_estimators = {}

for model_name, model in models_to_tune.items():
    print(f"Performing Grid Search for {model_name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=3, # 3-fold CV as specified
        scoring='roc_auc',
        n_jobs=-1 # Use all available CPU cores
    )
    grid_search.fit(X, y) # Fit on the entire dataset as per snippet
    best_estimators[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best ROC AUC score from Grid Search for {model_name}: {grid_search.best_score_:.4f}\n")

print("--- Grid Search Complete --- \n")


# =====================================================================================
# === LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV) EVALUATION ===
# =====================================================================================
print("--- Starting LOOCV Evaluation with Best Estimators ---")

loocv = LeaveOneOut()

# Modified evaluation function to include Accuracy
def evaluate_model_loocv(model, model_name, X_data, y_data):
    y_true_list = []
    y_pred_proba_list = []
    y_pred_class_list = [] # To store class predictions for accuracy

    print(f"Evaluating {model_name} using LOOCV...")
    for train_index, test_index in loocv.split(X_data):
        # Ensure correct indexing for pandas DataFrames/Series
        X_train_loocv, X_test_loocv = X_data.iloc[train_index], X_data.iloc[test_index]
        y_train_loocv, y_test_loocv = y_data.iloc[train_index], y_data.iloc[test_index]

        # Clone the best estimator to ensure fresh state for each fold
        current_model = clone(model)
        current_model.fit(X_train_loocv, y_train_loocv)

        # Predict probabilities for the left-out sample
        y_pred_proba = current_model.predict_proba(X_test_loocv)[:, 1]
        # Predict class (using 0.5 threshold by default for binary)
        y_pred_class = current_model.predict(X_test_loocv)

        y_true_list.append(y_test_loocv.values[0])
        y_pred_proba_list.append(y_pred_proba[0])
        y_pred_class_list.append(y_pred_class[0])

    # Calculate metrics
    auc_score = roc_auc_score(y_true_list, y_pred_proba_list)
    accuracy = accuracy_score(y_true_list, y_pred_class_list) # Calculate accuracy

    print(f"{model_name} LOOCV ROC AUC Score: {auc_score:.4f}")
    print(f"{model_name} LOOCV Accuracy: {accuracy:.4f}") # Print accuracy

    fpr, tpr, _ = roc_curve(y_true_list, y_pred_proba_list)
    return fpr, tpr, auc_score, accuracy # Return accuracy as well

roc_curves = {}
roc_auc_scores = {}
accuracies = {} # Dictionary to store accuracies

for model_name, best_estimator in best_estimators.items():
    fpr, tpr, auc_score, accuracy = evaluate_model_loocv(best_estimator, model_name, X, y)
    roc_curves[model_name] = (fpr, tpr)
    roc_auc_scores[model_name] = auc_score
    accuracies[model_name] = accuracy # Store accuracy

print("\n--- LOOCV Evaluation Summary ---")
for model_name in best_estimators.keys():
    print(f"{model_name}: ROC AUC = {roc_auc_scores[model_name]:.4f}, Accuracy = {accuracies[model_name]:.4f}")

# Note: Plotting code for ROC curves would typically follow here, using the 'roc_curves' dictionary.
# Example (requires matplotlib):
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# for model_name, (fpr, tpr) in roc_curves.items():
#     plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_scores[model_name]:.2f})')
# plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves from LOOCV')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.show()

print("\n--- Script Finished ---")
