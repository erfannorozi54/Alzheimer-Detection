from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# Import custom modules
from utils import feature_selection_comparison, feature_selector, model_selector_classifier

# Setup environment and logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
logging.basicConfig(filename='model_selection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================================
# STEP 1: Load and preprocess the dataset ONCE at the beginning
# =====================================================================================
print("Loading and preprocessing data...")
darwin = fetch_ucirepo(id=732)

# Extract features and targets
X = darwin.data.features.iloc[:, 1:]  # Use X directly, matching main.py
y = darwin.data.targets.iloc[:, 0]
y = LabelEncoder().fit_transform(y) if y.dtype == "object" else y

# Outlier Resolution using IQR Method
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")

# =====================================================================================
# STEP 2: Compare feature selection methods
# =====================================================================================
print("\nComparing feature selection methods...")

# Initialize comparison
comparison = feature_selection_comparison.FeatureSelectionComparison(X, y, n_features=10)

# Run comparison
results = comparison.run_comparison()

print("\n" + "=" * 80)
print("Feature Selection Methods Comparison Results")
print("=" * 80)

# Print overall summary
print("\n1. Overall Accuracy Summary:")
print("-" * 40)
summary = results.groupby(["Method", "Model"])["test_accuracy"].mean().round(3)
print(summary)

# Get best methods for each metric-model pair
best_methods = comparison.get_best_methods()

# Print best methods summary
print("\n2. Best Feature Selection Methods by Model and Metric:")
print("-" * 60)

for model in best_methods["Model"].unique():
    print(f"\nModel: {model}")
    print("-" * 20)
    model_results = best_methods[best_methods["Model"] == model]

    # Create a formatted table for each model
    table = pd.DataFrame({
        "Metric": model_results["Metric"],
        "Best Method": model_results["Best_Method"],
        "Score": model_results["Score"].round(4),
        "Improvement %": model_results["Improvement_%"].round(2),
        "Time (s)": model_results["Selection_Time"].round(3),
    })
    print(table.to_string(index=False))

# Perform statistical analysis
analyses = comparison.statistical_analysis()
print("\n3. Statistical Analyses:")
print("-" * 40)
for name, analysis in analyses.items():
    print(f"\n{name}:")
    print(analysis)

# Create visualizations
print("\n4. Creating visualizations...")
comparison.plot_results()

# Additional visualization for best methods
plt.figure(figsize=(12, 6))
best_methods_pivot = best_methods.pivot(
    index="Metric", columns="Model", values="Best_Method"
)
sns.heatmap(
    best_methods_pivot.notna(),
    annot=best_methods_pivot,
    fmt="",
    cmap="YlOrRd",
    cbar=False,
)
plt.title("Best Feature Selection Methods by Model and Metric")
plt.tight_layout()
plt.savefig("results/best_methods_heatmap.png")

# Save results
results.to_csv("feature_selection_comparison_results.csv", index=False)
best_methods.to_csv("best_feature_selection_methods.csv", index=False)

# Print selected features summary
print("\n5. Selected Features Summary:")
print("-" * 40)
for model in best_methods["Model"].unique():
    print(f"\nModel: {model}")
    model_results = best_methods[best_methods["Model"] == model]
    best_accuracy_method = model_results[
        model_results["Metric"] == "test_accuracy"
    ]["Best_Method"].iloc[0]
    best_features = model_results[model_results["Metric"] == "test_accuracy"][
        "Selected_Features"
    ].iloc[0]
    print(f"Best Method (by accuracy): {best_accuracy_method}")
    print(f"Selected Features: {best_features}")

print("\nAnalysis complete! Results have been saved to CSV files.")

print("\n6. Overall Feature Selection Methods Performance:")
print("-" * 60)

# Define the metrics to include
metrics_to_analyze = ["test_accuracy", "test_precision", "test_recall", "test_f1"]

# Create a summary dataframe for each method
method_summaries = []
for method in results["Method"].unique():
    method_data = results[results["Method"] == method]

    # Get mean scores for each metric
    metric_means = {
        metric: method_data[metric].mean() for metric in metrics_to_analyze
    }

    # Calculate overall mean across all metrics and models
    overall_mean = np.mean(list(metric_means.values()))

    # Add to summaries
    summary = {
        "Method": method,
        "Overall_Mean": overall_mean,
        **metric_means,
        "Selection_Time": method_data["Selection_Time"].mean(),
    }
    method_summaries.append(summary)

# Convert to DataFrame and sort by overall mean
summary_df = pd.DataFrame(method_summaries)
summary_df = summary_df.sort_values("Overall_Mean", ascending=False)

# Round the values for display
display_df = summary_df.round(4)

# Print the summary
print("\nOverall Performance Summary (sorted by mean of all metrics):")
print("-" * 80)
print(display_df.to_string(index=False))

# Create a bar plot of overall means
plt.figure(figsize=(12, 6))
plt.bar(summary_df["Method"], summary_df["Overall_Mean"])
plt.title("Overall Performance of Feature Selection Methods")
plt.xlabel("Method")
plt.ylabel("Mean Score (across all metrics and models)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/overall_performance_barplot.png")

# Show ranking for each metric
print("\nRankings by Different Metrics:")
print("-" * 40)
ranking_df = summary_df.copy()
for column in metrics_to_analyze + ["Overall_Mean"]:
    ranking_df[f"{column}_rank"] = ranking_df[column].rank(ascending=False)

ranking_summary = ranking_df[
    ["Method"] + [f"{col}_rank" for col in metrics_to_analyze + ["Overall_Mean"]]
]
print(ranking_summary.round(2).to_string(index=False))

# =====================================================================================
# STEP 3: Compare classification models using 3 different feature selection methods
# =====================================================================================
print("\n\nComparing classification models with different feature selection methods...")

# Split the data - using the SAME preprocessed data from Step 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the feature selector with training data
fs = feature_selector.FeatureSelector(X_train, y_train)

# Define feature selection methods
feature_selection_methods = {
    "Ensemble": lambda: fs.ensemble_feature_importance_selection(
        models=[
            RandomForestClassifier(random_state=42),
            LogisticRegression(random_state=42),
            SVC(kernel="linear", random_state=42),
        ],
        top_k=20,
    ),
    "RandomForest": lambda: fs.random_forest_selection(top_k=20),
    "Boruta": lambda: fs.boruta_selection(n_estimators=100, max_iter=100, top_k=20),
    "SI": lambda: fs.separation_index_selection(top_k=20),
    "weighted_SI": lambda: fs.weighted_separation_index(top_k=20),
}

# Define classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "ModelSelector": model_selector_classifier.ModelSelectorClassifier(n_folds=5,cv_repeats= 8, meta_learner="lr", n_clusters=6, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "Ridge": RidgeClassifier(random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
}

# Dictionary to store results
results = {}

# Perform feature selection and classification
print("\nPerforming feature selection and classification...")
for fs_name, fs_method in feature_selection_methods.items():
    print(f"\nFeature Selection Method: {fs_name}")

    # Get selected features
    selected_features = fs_method()

    # Select features from data
    if isinstance(selected_features, dict):
        selected_features = list(selected_features.keys())
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print(f"Number of selected features: {len(selected_features)}")
    print("Selected features:", selected_features)

    # Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name}...")

        # Create and fit pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

        # Fit the pipeline
        pipeline.fit(X_train_selected, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test_selected)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Store results
        if fs_name not in results:
            results[fs_name] = {}
        results[fs_name][clf_name] = accuracy

        print(f"{clf_name} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Log ModelSelectorClassifier accuracy
        if clf_name == "ModelSelector":
            logging.info(f"ModelSelectorClassifier Accuracy ({fs_name}): {accuracy:.4f}")

# Create comparison DataFrame
comparison_df = pd.DataFrame(results)
print("\nComparison of all methods:")
print(comparison_df)

# Calculate average performance for each classifier
avg_classifier_performance = comparison_df.mean(axis=1)
print("\nAverage classifier performance across all feature selection methods:")
print(avg_classifier_performance)

# Calculate average performance for each feature selection method
avg_fs_performance = comparison_df.mean()
print("\nAverage feature selection method performance across all classifiers:")
print(avg_fs_performance)

# Find best combination
best_accuracy = 0
best_combination = None

for fs_name in results:
    for clf_name in results[fs_name]:
        accuracy = results[fs_name][clf_name]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_combination = (fs_name, clf_name)

print(f"\nBest combination: {best_combination[0]} feature selection with {best_combination[1]}")
print(f"Best accuracy: {best_accuracy:.4f}")