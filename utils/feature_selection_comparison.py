from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
    KFold,
    train_test_split,
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from scipy import stats
import time

# Import custom classes
from utils import feature_selector


class FeatureSelectionComparison:
    """
    A class to compare different feature selection methods using multiple metrics
    and statistical tests.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_features: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize the comparison framework.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target labels
        n_features : int
            Number of features to select
        random_state : int
            Random state for reproducibility
        """
        self.X = X
        self.y = y
        self.n_features = n_features
        self.random_state = random_state
        self.results = None

        # Initialize models with proper parameters
        self.models = {
            "RF": RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            "LR": LogisticRegression(
                max_iter=2000, tol=1e-4, random_state=random_state
            ),
            "SVM": SVC(probability=True, random_state=random_state),
        }

    def preprocess_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Preprocess the data: scale features and split into train/test sets.
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index,
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            self.y,
            test_size=0.3,
            stratify=self.y,
            random_state=self.random_state,
        )

        return X_train, X_test, y_train, y_test

    def evaluate_model(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate a model using multiple metrics and cross-validation.
        """
        # Cross-validation setup
        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )

        # Compute cross-validation scores
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy"
        )

        # Fit model and make predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate metrics
        metrics = {
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(
                y_test, y_pred, average="weighted"
            ),
            "test_recall": recall_score(y_test, y_pred, average="weighted"),
            "test_f1": f1_score(y_test, y_pred, average="weighted"),
        }

        # Add ROC AUC if probability predictions are available
        if y_pred_proba is not None:
            metrics["test_roc_auc"] = roc_auc_score(y_test, y_pred_proba)

        return metrics

    def run_comparison(self) -> pd.DataFrame:
        """
        Run the comparison of different feature selection methods.
        """
        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data()

        # Initialize feature selector
        selector = feature_selector.FeatureSelector(X_train, y_train)

        # Define selection methods
        selection_methods = {
            "SI": lambda: selector.separation_index_selection(
                top_k=self.n_features
            ),
            "ANOVA": lambda: selector.anova_selection(top_k=self.n_features),
            "MI": lambda: selector.mutual_information_selection(
                top_k=self.n_features
            ),
            "Chi2": lambda: selector.chi_square_selection(
                top_k=self.n_features
            ),
            "RF": lambda: selector.random_forest_selection(
                top_k=self.n_features
            ),
            "RFE": lambda: selector.recursive_feature_elimination(
                n_features_to_select=self.n_features
            ),
            "L1": lambda: selector.l1_based_selection(),
            "PCA": lambda: selector.pca_selection(top_k=self.n_features),
            "SparsePCA": lambda: selector.sparse_pca_selection(
                X_train, top_k=self.n_features
            ),
            "Boruta": lambda: selector.boruta_selection(top_k=self.n_features),
            "ReliefF": lambda: selector.reliefF_selection(
                top_k=self.n_features
            ),
            "Ensemble": lambda: selector.ensemble_feature_importance_selection(
                models=[RandomForestClassifier(), LogisticRegression()],
                top_k=self.n_features,
            ),
        }

        results = []

        # Compare methods
        for method_name, select_features in selection_methods.items():
            try:
                print(f"Evaluating {method_name}...")

                # Time the feature selection
                start_time = time.time()
                selected_features = select_features()
                selection_time = time.time() - start_time

                # Ensure correct number of features
                if isinstance(selected_features, dict):
                    selected_features = list(selected_features.keys())[
                        : self.n_features
                    ]
                elif len(selected_features) > self.n_features:
                    selected_features = selected_features[: self.n_features]

                # Select features from data
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]

                # Evaluate with different models
                for model_name, model in self.models.items():
                    metrics = self.evaluate_model(
                        model,
                        X_train_selected,
                        X_test_selected,
                        y_train,
                        y_test,
                    )

                    results.append(
                        {
                            "Method": method_name,
                            "Model": model_name,
                            "Selected_Features": ", ".join(selected_features),
                            "Selection_Time": selection_time,
                            **metrics,
                        }
                    )

            except Exception as e:
                print(f"Error with {method_name}: {str(e)}")
                continue

        self.results = pd.DataFrame(results)
        return self.results

    def get_best_methods(self) -> pd.DataFrame:
        """
        Find the best feature selection method for each combination of metric and model.

        Returns:
        --------
        pd.DataFrame
            DataFrame showing the best method and its score for each metric-model pair
        """
        if self.results is None:
            raise ValueError("Run comparison first using run_comparison()")

        # Define metrics to analyze
        metrics = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "cv_accuracy_mean",
        ]

        best_methods = []

        # For each model and metric combination
        for model in self.results["Model"].unique():
            model_data = self.results[self.results["Model"] == model]

            for metric in metrics:
                if metric in model_data.columns:
                    # Find the method with the highest score
                    best_idx = model_data[metric].idxmax()
                    best_method = model_data.loc[best_idx]

                    # Add to results
                    best_methods.append(
                        {
                            "Model": model,
                            "Metric": metric,
                            "Best_Method": best_method["Method"],
                            "Score": best_method[metric],
                            "Selected_Features": best_method[
                                "Selected_Features"
                            ],
                            "Selection_Time": best_method["Selection_Time"],
                        }
                    )

        # Convert to DataFrame and sort
        best_df = pd.DataFrame(best_methods)
        best_df = best_df.sort_values(["Model", "Metric"])

        # Format score values
        best_df["Score"] = best_df["Score"].round(4)
        best_df["Selection_Time"] = best_df["Selection_Time"].round(3)

        # Add relative improvement over mean
        for model in best_df["Model"].unique():
            for metric in best_df["Metric"].unique():
                mask = (best_df["Model"] == model) & (
                    best_df["Metric"] == metric
                )
                if mask.any():
                    avg_score = self.results[(self.results["Model"] == model)][
                        metric
                    ].mean()
                    best_score = best_df.loc[mask, "Score"].iloc[0]
                    improvement = ((best_score - avg_score) / avg_score) * 100
                    best_df.loc[mask, "Improvement_%"] = improvement.round(2)

        return best_df

    def statistical_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Perform statistical analysis on the results.
        """
        if self.results is None:
            raise ValueError("Run comparison first using run_comparison()")

        analyses = {}

        # Friedman test for each metric
        metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
        for metric in metrics:
            # Reshape data for Friedman test
            pivot = self.results.pivot(
                index="Model", columns="Method", values=metric
            )

            # Perform Friedman test
            statistic, p_value = stats.friedmanchisquare(
                *[pivot[col] for col in pivot.columns]
            )

            analyses[f"friedman_{metric}"] = pd.DataFrame(
                {"statistic": [statistic], "p_value": [p_value]}
            )

        return analyses

    def plot_results(self) -> None:
        """
        Create visualizations of the results.
        """
        if self.results is None:
            raise ValueError("Run comparison first using run_comparison()")

        # Set up the style
        plt.style.use("default")

        # Plot 1: Performance comparison
        fig, ax = plt.subplots(figsize=(15, 6))
        data = pd.melt(
            self.results,
            id_vars=["Method", "Model"],
            value_vars=["test_accuracy"],
            var_name="Metric",
            value_name="Score",
        )

        methods = data["Method"].unique()
        models = data["Model"].unique()
        x = np.arange(len(methods))
        width = 0.25

        for i, model in enumerate(models):
            mask = data["Model"] == model
            ax.bar(x + i * width, data[mask]["Score"], width, label=model)

        ax.set_ylabel("Accuracy Score")
        ax.set_title("Performance Comparison Across Methods and Models")
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Plot 2: Selection time comparison
        plt.figure(figsize=(10, 5))
        time_data = self.results.groupby("Method")["Selection_Time"].mean()
        plt.bar(time_data.index, time_data.values)
        plt.title("Feature Selection Time Comparison")
        plt.xlabel("Method")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        # Plot 3: Heatmap of metrics
        plt.figure(figsize=(12, 8))
        metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
        pivot = self.results.pivot_table(
            index="Method", columns="Model", values=metrics, aggfunc="mean"
        )

        im = plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar(im)

        # Add text annotations
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                text = plt.text(
                    j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center"
                )

        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title("Performance Metrics Heatmap")
        plt.tight_layout()
        plt.show()
