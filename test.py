import numpy as np
from collections import Counter
import logging

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.base import clone

# scikit-learn models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# xgboost
from xgboost import XGBClassifier

# For fetching the Alzheimer’s dataset from UCIML repo
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42

# =====================================================================================
# === DATA LOADING AND PREPROCESSING ===
# =====================================================================================
print("--- Starting Data Loading and Preprocessing ---")

# 1. Load Data
print("Loading data from UCIML repository...")
darwin = fetch_ucirepo(id=732)
X_df = darwin.data.features.iloc[:, 1:]  # skipping the first column, if needed
y_series = darwin.data.targets.iloc[:, 0]
print(f"Initial dataset loaded: {X_df.shape[0]} samples, {X_df.shape[1]} features")

# 2. Encode Target Variable
print("Encoding target variable...")
if y_series.dtype == "object":
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    print(f"Target encoded. Classes: {label_encoder.classes_}")
else:
    y_encoded = y_series.to_numpy() # Ensure y is numpy array
    print("Target is already numeric.")

# 3. Train-Test Split
print("Performing train-test split...")
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y_encoded, test_size=0.1, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"Train set size: {X_train_df.shape[0]}, Test set size: {X_test_df.shape[0]}")

# 4. Feature Scaling (StandardScaler)
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
# Convert DataFrames to numpy for scaling just before fitting/transforming
X_train_scaled = scaler.fit_transform(X_train_df.to_numpy(dtype=float))
X_test_scaled = scaler.transform(X_test_df.to_numpy(dtype=float))
print("Features scaled.")

print("--- Data Loading and Preprocessing Complete --- \n")


# =====================================================================================
# === BASE CLASSIFIER DEFINITIONS ===
# =====================================================================================
base_classifiers_ = {
    "RF": RandomForestClassifier(
        n_estimators=100, max_features='sqrt', random_state=RANDOM_STATE
    ),
    "LR": LogisticRegression(
        max_iter=5000, C=1.0, solver='saga', random_state=RANDOM_STATE
    ),
    "XGB": XGBClassifier(
        n_estimators=1000, random_state=RANDOM_STATE
    ),
    "DT": DecisionTreeClassifier(
        max_depth=12, min_samples_split=5, random_state=RANDOM_STATE
    ),
    "SVM": SVC(
        random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5, weights='distance'
    ),
    "NB": GaussianNB(),
    "GB": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
    )
}

# =====================================================================================
# STEP 4: Repeated K-fold cross-validation with different seeds on the SCALED training data
# =====================================================================================
SEEDS = [42, 123, 2023]  # Adjust or add more if desired
n_train = len(X_train_scaled)

# We'll collect "best-model" picks across seeds
best_models_across_seeds = [[] for _ in range(n_train)]

for seed in SEEDS:
    print(f"Performing cross-validation with seed={seed}")
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
    
    # For this seed, store best model picks per sample
    best_model_for_seed = np.empty(n_train, dtype=object)
    
    for train_index, val_index in skf.split(X_train_scaled, y_train):
        X_tr_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_tr_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Train each base model on the fold’s training portion
        trained_models = {}
        fold_model_accuracies = {}
        for model_name, clf in base_classifiers_.items():
            cloned_model = clone(clf)
            cloned_model.fit(X_tr_fold, y_tr_fold)
            trained_models[model_name] = cloned_model
            
            # Accuracy on the fold’s training portion
            train_preds = cloned_model.predict(X_tr_fold)
            fold_model_accuracies[model_name] = accuracy_score(y_tr_fold, train_preds)
        
        # Predict the validation portion with each model
        val_preds = {}
        for model_name, trained_model in trained_models.items():
            val_preds[model_name] = trained_model.predict(X_val_fold)
        
        # Determine the best model for each validation sample
        for local_i, global_idx in enumerate(val_index):
            true_label = y_val_fold[local_i]
            correct_models = [
                m for m in trained_models.keys()
                if val_preds[m][local_i] == true_label
            ]
            
            if not correct_models:
                # If no model got it right, pick the model with highest training accuracy
                best_pick = max(fold_model_accuracies, key=fold_model_accuracies.get)
                best_model_for_seed[global_idx] = best_pick
            elif len(correct_models) == 1:
                best_model_for_seed[global_idx] = correct_models[0]
            else:
                # Multiple correct models; pick the one with the highest training accuracy
                best_model = max(correct_models, key=lambda m: fold_model_accuracies[m])
                best_model_for_seed[global_idx] = best_model
    
    # Accumulate the best-model picks for each sample for this seed
    for i in range(n_train):
        best_models_across_seeds[i].append(best_model_for_seed[i])

# =====================================================================================
# STEP 5: Combine best-model choices across seeds (majority vote per sample)
# =====================================================================================
combined_best_models = []
for i in range(n_train):
    picks = best_models_across_seeds[i]
    # majority vote
    vote_count = Counter(picks)
    best_model, _ = vote_count.most_common(1)[0]
    combined_best_models.append(best_model)

# Configure logging
logging.basicConfig(filename='meta_learning_labels.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', filemode='w') # 'w' to overwrite each run

# Convert these model labels into numeric form for training the gating classifier
logging.info("--- Inspecting Derived Labels for Meta-Learner ---")
logging.info(f"Original best model choices (all): {combined_best_models}") # Log all choices
model_label_encoder = LabelEncoder()
gating_labels = model_label_encoder.fit_transform(combined_best_models)
logging.info(f"Encoded gating labels (all): {gating_labels.tolist()}") # Log all encoded labels as list
label_mapping = dict(zip(model_label_encoder.classes_, model_label_encoder.transform(model_label_encoder.classes_)))
logging.info(f"Model name to numeric label mapping: {label_mapping}")
logging.info("--------------------------------------------------")

# =====================================================================================
# STEP 6: Train the gating classifier on the ENTIRE (scaled) training set
# =====================================================================================
gating_classifier = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
gating_classifier.fit(X_train_scaled, gating_labels)

# -------------------------------------------------------------------------------------
# Log how accurately the gating classifier picks the "best model" on the training set
# -------------------------------------------------------------------------------------
train_gating_preds = gating_classifier.predict(X_train_scaled)
gating_acc_train = accuracy_score(gating_labels, train_gating_preds)
print(f"Meta-learning (gating) classifier accuracy on train data: {gating_acc_train:.4f}")

# =====================================================================================
# STEP 7: Train final base models on the ENTIRE (scaled) training set
# =====================================================================================
final_base_models = {}
for model_name, clf in base_classifiers_.items():
    cloned_model = clone(clf)
    cloned_model.fit(X_train_scaled, y_train)
    final_base_models[model_name] = cloned_model

# Log performance of each base model trained on the full training set
logging.info("\n--- Base Classifier Performance (Trained on Full Train Set) ---")
print("\n--- Base Classifier Performance (Trained on Full Train Set) ---") # Also print to console
for model_name, model in final_base_models.items():
    # Train accuracy
    train_preds = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    log_msg_train = f"Model: {model_name} | Train Accuracy: {train_acc:.4f}"
    logging.info(log_msg_train)
    print(log_msg_train)

    # Test accuracy
    test_preds = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    log_msg_test = f"Model: {model_name} | Test Accuracy: {test_acc:.4f}"
    logging.info(log_msg_test)
    print(log_msg_test)
logging.info("-----------------------------------------------------------")
print("-----------------------------------------------------------")


# =====================================================================================
# STEP 8: Test-time inference using the gating classifier to select a model for each sample
# =====================================================================================
gating_test_preds = gating_classifier.predict(X_test_scaled)
chosen_model_names = model_label_encoder.inverse_transform(gating_test_preds)

final_predictions = []
for i, chosen_model in enumerate(chosen_model_names):
    # Predict with the chosen base model
    # We slice X_test_scaled[i:i+1] to keep a 2D shape for .predict()
    pred = final_base_models[chosen_model].predict(X_test_scaled[i:i+1])
    final_predictions.append(pred[0])

final_predictions = np.array(final_predictions)
test_accuracy = accuracy_score(y_test, final_predictions)
print(f"Gating approach Test Accuracy: {test_accuracy:.4f}")

# -------------------------------------------------------------------------------------
# (Optional) see how many times each model was chosen on TEST
# -------------------------------------------------------------------------------------
test_model_counts = Counter(chosen_model_names)
print("\n--- Gating decisions on the TEST set ---")
for model_name, count in test_model_counts.items():
    print(f"Model '{model_name}' chosen {count} times")

# =====================================================================================
# STEP 9: Compare with a single-model XGBoost baseline
# =====================================================================================
xgb_baseline = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE)
xgb_baseline.fit(X_train_scaled, y_train)
baseline_preds = xgb_baseline.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"Single-model XGBoost Test Accuracy: {baseline_acc:.4f}")
