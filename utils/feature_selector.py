from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

import pandas as pd
from typing import List, Union, Dict
from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
    chi2,
    RFE,
    SelectFromModel,
)
from boruta import BorutaPy


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
import warnings
warnings.filterwarnings("ignore")


class FeatureSelector:
    """
    A comprehensive feature selection toolkit supporting multiple methods:
    - Separation Index (SI): Measures class separation using nearest neighbors
    - ANOVA F-test: Tests for differences in class means
    - Mutual Information: Measures feature-target dependencies
    - Chi-Square: Tests feature-target independence
    - Random Forest Importance: Tree-based feature importance
    - Recursive Feature Elimination (RFE): Iterative feature removal
    - L1-based Selection (Lasso): Regularization-based selection
    - Correlation-based Selection: Removes redundant features
    - PCA: Selects top features based on principal components
    - SparsePCA: Selects features using sparse principal components
    - Boruta: Robust feature selection using random forest
    - ReliefF: Feature selection based on distance metrics and neighbors

    Methods Summary:
    ---------------
    sparse_pca_selection:
        Selects features using Sparse Principal Component Analysis.
        Good for: High-dimensional data with sparse structure, feature correlation

    separation_index_selection:
        Selects features based on how well they separate classes using nearest neighbors.
        Good for: Detecting local class separation patterns, non-linear relationships

    anova_selection:
        Uses F-test to select features based on class mean differences.
        Good for: Linear relationships, normally distributed features

    mutual_information_selection:
        Measures information gain between features and target.
        Good for: Both linear and non-linear relationships

    chi_square_selection:
        Tests statistical independence between features and target.
        Good for: Categorical features, classification tasks

    random_forest_selection:
        Uses tree-based importance scores.
        Good for: Complex interactions, non-linear relationships

    recursive_feature_elimination:
        Iteratively removes features using a model's weights.
        Good for: Feature interactions, considering model-specific importance

    l1_based_selection:
        Uses L1 regularization to select features.
        Good for: High-dimensional data, sparse feature selection

    correlation_based_selection:
        Removes highly correlated features to reduce redundancy.
        Good for: Removing multicollinearity, feature redundancy

    pca_selection:
        Selects features based on the principal components capturing the most variance.

    boruta_selection:
        Uses Boruta to select important features based on random forest.

    reliefF_selection:
        Selects features based on nearest neighbors and distance metrics.
    """

    def __init__(self, X: pd.DataFrame, y: np.ndarray):
        """
        Initialize the feature selector.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target labels
        """
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()

    def anova_selection(
        self,
        top_k: Union[int, None] = None,
        threshold: Union[float, None] = None,
    ) -> Dict[str, float]:
        """
        Perform ANOVA F-test for feature selection.

        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            p-value threshold for feature selection

        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their F-scores
        """
        F_scores, p_values = f_classif(self.X, self.y)
        feature_scores = dict(zip(self.feature_names, F_scores))

        if threshold is not None:
            selected_features = {
                feat: score
                for feat, score, p_val in zip(
                    self.feature_names, F_scores, p_values
                )
                if p_val < threshold
            }
        else:
            sorted_features = dict(
                sorted(
                    feature_scores.items(), key=lambda x: x[1], reverse=True
                )
            )
            if top_k:
                selected_features = dict(list(sorted_features.items())[:top_k])
            else:
                selected_features = sorted_features

        return selected_features

    def mutual_information_selection(
        self,
        top_k: Union[int, None] = None,
        threshold: Union[float, None] = None,
        continuous: bool = False,
    ) -> Dict[str, float]:
        """
        Perform Mutual Information feature selection.

        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum MI score for feature selection
        continuous : bool, optional
            Whether to use mutual_info_regression for continuous targets
        """
        mi_scores = mutual_info_classif(self.X, self.y)
        feature_scores = dict(zip(self.feature_names, mi_scores))

        sorted_features = dict(
            sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        )

        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score
                for feat, score in sorted_features.items()
                if score > threshold
            }
        else:
            selected_features = sorted_features

        return selected_features

    def sparse_pca_selection(
        self, X_train: pd.DataFrame, top_k: int = 5, random_state: int = 42
    ) -> List[str]:
        """
        Perform feature selection using Sparse PCA.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training data
        top_k : int
            Number of features to select

        Returns:
        --------
        List[str]
            Selected feature names
        """
        # Initialize and fit SparsePCA
        sparse_pca = SparsePCA(
            n_components=min(20, X_train.shape[1]), random_state=random_state
        )
        sparse_pca.fit(X_train)

        # Get feature importance scores based on component loadings
        feature_importance = np.abs(sparse_pca.components_).sum(axis=0)

        # Get indices of top features
        top_indices = np.argsort(feature_importance)[-top_k:]

        # Return selected feature names
        return list(X_train.columns[top_indices])

    def chi_square_selection(
        self,
        top_k: Union[int, None] = None,
        threshold: Union[float, None] = None,
    ) -> Dict[str, float]:
        """
        Perform Chi-Square feature selection.
        Note: Features must be non-negative.
        """
        # Scale features to be non-negative for chi-square
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X)

        chi_scores, p_values = chi2(X_scaled, self.y)
        feature_scores = dict(zip(self.feature_names, chi_scores))

        if threshold is not None:
            selected_features = {
                feat: score
                for feat, score, p_val in zip(
                    self.feature_names, chi_scores, p_values
                )
                if p_val < threshold
            }
        else:
            sorted_features = dict(
                sorted(
                    feature_scores.items(), key=lambda x: x[1], reverse=True
                )
            )
            if top_k:
                selected_features = dict(list(sorted_features.items())[:top_k])
            else:
                selected_features = sorted_features

        return selected_features

    def random_forest_selection(
        self,
        top_k: Union[int, None] = None,
        threshold: Union[float, None] = None,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Perform Random Forest feature importance selection.
        """
        rf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        rf.fit(self.X, self.y)

        feature_scores = dict(zip(self.feature_names, rf.feature_importances_))

        sorted_features = dict(
            sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        )

        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score
                for feat, score in sorted_features.items()
                if score > threshold
            }
        else:
            selected_features = sorted_features

        return selected_features

    def recursive_feature_elimination(
        self, n_features_to_select: int, step: int = 1, estimator: str = "rf"
    ) -> List[str]:
        """
        Perform Recursive Feature Elimination.

        Parameters:
        -----------
        n_features_to_select : int
            Number of features to select
        step : int
            Number of features to remove at each iteration
        estimator : str
            Estimator to use ('rf', 'lr', or 'svm')
        """
        if estimator == "rf":
            est = RandomForestClassifier(n_estimators=100, random_state=42)
        elif estimator == "lr":
            est = LogisticRegression(random_state=42)
        elif estimator == "svm":
            est = LinearSVC(random_state=42)
        else:
            raise ValueError("Estimator must be 'rf', 'lr', or 'svm'")

        rfe = RFE(
            estimator=est, n_features_to_select=n_features_to_select, step=step
        )
        rfe.fit(self.X, self.y)

        selected_features = [
            feat
            for feat, selected in zip(self.feature_names, rfe.support_)
            if selected
        ]

        return selected_features

    def l1_based_selection(
        self, C: float = 1.0, penalty: str = "l1", solver: str = "liblinear"
    ) -> List[str]:
        """
        Perform L1-based feature selection using Logistic Regression.
        """
        l1_selector = SelectFromModel(
            LogisticRegression(
                C=C, penalty=penalty, solver=solver, random_state=42
            )
        )
        l1_selector.fit(self.X, self.y)

        selected_features = [
            feat
            for feat, selected in zip(
                self.feature_names, l1_selector.get_support()
            )
            if selected
        ]

        return selected_features

    def correlation_based_selection(self, threshold: float = 0.7) -> List[str]:
        """
        Select features based on correlation with target and between features.
        """
        # Calculate correlations
        corr_matrix = self.X.corr().abs()

        # Find pairs of features with correlation above threshold
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Get features to drop
        to_drop = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column] > threshold)
        ]

        # Keep remaining features
        selected_features = [
            feat for feat in self.feature_names if feat not in to_drop
        ]

        return selected_features

    def pca_selection(self, top_k: int = 5) -> List[str]:
        """
        Perform PCA and select the top-k features that explain the most variance.

        Parameters:
        -----------
        top_k : int, optional
            Number of top principal components to select

        Returns:
        --------
        List[str]
            List of selected feature names
        """
        pca = PCA(n_components=top_k)
        pca.fit(self.X)

        # Get feature importance based on component loadings
        loadings = np.abs(pca.components_)
        feature_importance = loadings.sum(axis=0)

        # Select top features based on importance
        selected_indices = np.argsort(feature_importance)[-top_k:]
        selected_features = [self.feature_names[i] for i in selected_indices]

        return selected_features

    def boruta_selection(
        self,
        n_estimators: int = 100,
        max_iter: int = 100,
        top_k: Union[int, None] = None,
    ) -> List[str]:
        """
        Perform Boruta feature selection to identify important features.

        Parameters:
        -----------
        n_estimators : int, optional
            Number of trees in the random forest
        max_iter : int, optional
            Maximum number of iterations
        top_k : int, optional
            Maximum number of top features to return. If None, returns all selected features.
            If top_k is greater than the number of selected features, returns all selected features.

        Returns:
        --------
        List[str]
            List of selected feature names
        """
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        boruta = BorutaPy(
            rf, n_estimators="auto", max_iter=max_iter, random_state=42
        )

        # Scale features for Boruta
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X)

        boruta.fit(X_scaled, self.y)

        # Get ranking scores
        ranking_scores = boruta.ranking_

        # Create a list of (feature, rank) tuples
        feature_ranks = list(zip(self.feature_names, ranking_scores))

        # Sort features by rank (lower rank is better)
        sorted_features = sorted(feature_ranks, key=lambda x: x[1])

        # Get initially selected features
        selected_features = [
            feat
            for feat, selected in zip(self.feature_names, boruta.support_)
            if selected
        ]

        # If top_k is specified, limit the number of features
        if top_k is not None:
            if len(selected_features) < top_k:
                # If we have fewer selected features than top_k,
                # add tentative features based on ranking until we reach top_k
                tentative_features = [
                    feat
                    for feat, rank in sorted_features
                    if feat not in selected_features
                ]
                additional_needed = top_k - len(selected_features)
                selected_features.extend(
                    tentative_features[:additional_needed]
                )
            else:
                # If we have more selected features than top_k,
                # keep only the top_k features based on ranking
                feature_ranking = {
                    feat: rank for feat, rank in sorted_features
                }
                selected_features = sorted(
                    selected_features, key=lambda x: feature_ranking[x]
                )[:top_k]

        return selected_features

    def reliefF_selection(
        self, k_neighbors: int = 10, top_k: Union[int, None] = None
    ) -> List[str]:
        """
        Perform ReliefF feature selection.

        Parameters:
        -----------
        k_neighbors : int, optional
            Number of neighbors to consider
        top_k : int, optional
            Number of top features to select

        Returns:
        --------
        List[str]
            List of selected feature names
        """
        nn = NearestNeighbors(
            n_neighbors=k_neighbors + 1
        )  # +1 to exclude self
        nn.fit(self.X)
        feature_importances = np.zeros(self.X.shape[1])

        for i in range(self.X.shape[0]):
            # Find k nearest neighbors
            distances, indices = nn.kneighbors(
                self.X.iloc[i].values.reshape(1, -1)
            )
            indices = indices[0][1:]  # Exclude self

            # Split neighbors by class
            same_class = indices[self.y[indices] == self.y[i]]
            diff_class = indices[self.y[indices] != self.y[i]]

            if len(same_class) == 0 or len(diff_class) == 0:
                continue

            # Calculate feature importance
            for j in range(self.X.shape[1]):
                same_class_diff = np.mean(
                    np.abs(self.X.iloc[same_class, j] - self.X.iloc[i, j])
                )
                diff_class_diff = np.mean(
                    np.abs(self.X.iloc[diff_class, j] - self.X.iloc[i, j])
                )
                feature_importances[j] += diff_class_diff - same_class_diff

        # Sort features by importance
        sorted_indices = np.argsort(feature_importances)[::-1]
        if top_k:
            sorted_indices = sorted_indices[:top_k]

        selected_features = [self.feature_names[i] for i in sorted_indices]
        return selected_features

    def ensemble_feature_importance_selection(
        self,
        models: List[
            Union[RandomForestClassifier, LogisticRegression, LinearSVC]
        ],
        top_k: Union[int, None] = None,
        threshold: Union[float, None] = None,
    ) -> List[str]:
        """
        Select features based on ensemble of multiple models' feature importance.

        Parameters:
        -----------
        models : list
            List of scikit-learn models with feature_importances_ or coef_ attribute
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum importance score for selection

        Returns:
        --------
        List[str]
            List of selected feature names
        """
        feature_importances = np.zeros(self.X.shape[1])

        for model in models:
            model.fit(self.X, self.y)
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_.ravel())
            else:
                raise ValueError(
                    f"Model {type(model).__name__} has no feature importance attribute"
                )
            feature_importances += importance / np.sum(importance)  # Normalize

        feature_importances /= len(models)
        feature_scores = dict(zip(self.feature_names, feature_importances))

        # Sort features by importance
        sorted_features = dict(
            sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        )

        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score
                for feat, score in sorted_features.items()
                if score > threshold
            }
        else:
            selected_features = sorted_features

        return list(selected_features.keys())

    def separation_index_selection(
        self,
        top_k: Union[int, None] = None,
        threshold: Union[float, None] = None,
        n_neighbors: int = 5,
    ) -> Dict[str, float]:
        """
        Perform feature selection using Separation Index (SI).

        The SI measures how well each feature separates different classes by analyzing
        the class labels of nearest neighbors in the feature space.

        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum SI score for feature selection (between 0 and 1)
        n_neighbors : int, default=5
            Number of neighbors to consider for SI calculation

        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their SI scores
        """
        if n_neighbors >= len(self.X):
            raise ValueError(
                "n_neighbors must be less than the number of samples"
            )

        si_scores = {}
        for feature in self.feature_names:
            # Reshape feature values for KNN
            feature_values = self.X[feature].values.reshape(-1, 1)

            # Fit KNN on single feature, adding 1 to exclude self from neighbors
            knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            knn.fit(feature_values)

            # Get indices of nearest neighbors
            _, indices = knn.kneighbors(feature_values)

            # Calculate SI for each sample (excluding self)
            si_values = []
            for i, idx in enumerate(indices):
                # Exclude the first neighbor (self) by using idx[1:]
                neighbors_same_class = np.mean(self.y[idx[1:]] == self.y[i])
                si_values.append(neighbors_same_class)

            # Store average SI for this feature
            si_scores[feature] = np.mean(si_values)

        # Sort features by SI scores
        sorted_features = dict(
            sorted(si_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # Select features based on criteria
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score
                for feat, score in sorted_features.items()
                if score > threshold
            }
        else:
            selected_features = sorted_features

        return selected_features
