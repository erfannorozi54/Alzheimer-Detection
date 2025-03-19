from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    StratifiedKFold,
)
import scipy.spatial.distance as distance
from collections import defaultdict
from scipy.stats import entropy
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
from typing import List, Union, Dict, Optional
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
    
    separation_index_selection:
        Selects features based on how well they separate classes using nearest neighbors.
        Good for: Detecting local class separation patterns, non-linear relationships
        
    weighted_separation_index:
        Weight neighbor contributions by their distance, giving closer points more influence.
        Good for: Accounting for varying density in feature space
        
    multi_resolution_separation:
        Evaluates separation at multiple neighborhood sizes to find optimal scale.
        Good for: Features with varying scales of separation patterns
        
    feature_subset_separation:
        Assesses separation power of feature combinations, not just individual features.
        Good for: Capturing feature interactions that enhance class separation
        
    adversarial_separation:
        Measures feature robustness by testing separation under simulated perturbations.
        Good for: Selecting features resistant to noise and outliers
        
    manifold_separation:
        Calculates separation along local manifold structure rather than Euclidean space.
        Good for: High-dimensional data with intrinsic lower-dimensional structure
        
    information_theoretic_separation:
        Uses entropy and mutual information to quantify class separation.
        Good for: Capturing complex dependencies between features and classes
        
    ensemble_separation:
        Combines multiple separation metrics for more robust feature ranking.
        Good for: Datasets where any single metric might be misleading
        
    fisher_discriminant_ratio:
        Classical parametric measure of class separation based on means and variances.
        Good for: Features with approximately normal distributions
        
    interquartile_overlap:
        Robust measure of class separation based on quartile ranges.
        Good for: Datasets with outliers where mean/variance measures might be misleading
        
    maximum_mean_discrepancy:
        Kernel-based measure of distribution differences.
        Good for: Capturing complex, non-linear separation patterns
        
    decision_stump_selection:
        Uses simple decision tree stumps to measure separability.
        Good for: Features with clear threshold-based separation
        
    kolmogorov_smirnov_selection:
        Non-parametric measure based on maximum distance between class CDFs.
        Good for: Any distribution shape without making parametric assumptions
        
    nearest_hits_misses:
        Enhanced nearest neighbor approach comparing same-class vs different-class distances.
        Good for: Datasets where local neighborhood structure is informative
        
    imbalanced_separation:
        Weighted separation measures for imbalanced datasets.
        Good for: Datasets with significant class imbalance
    
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
        
    def weighted_separation_index(
        self, 
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        n_neighbors: int = 5,
        distance_metric: str = 'euclidean'
    ) -> Dict[str, float]:
        """
        Calculate separation index with distance-weighted neighbor contributions.
        
        Closer neighbors have higher influence on the separation index,
        reflecting the intuition that points closer in feature space
        should be more relevant for separation assessment.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        n_neighbors : int, default=5
            Number of neighbors to consider
        distance_metric : str, default='euclidean'
            Distance metric for neighbor weighting
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their weighted SI scores
        """
        if n_neighbors >= len(self.X):
            raise ValueError("n_neighbors must be less than the number of samples")
            
        weighted_si_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values.reshape(-1, 1)
            
            # Fit KNN
            knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=distance_metric)
            knn.fit(feature_values)
            
            # Get neighbors and distances
            distances, indices = knn.kneighbors(feature_values)
            
            # Calculate weighted SI
            weighted_si_values = []
            for i, (idx, dist) in enumerate(zip(indices, distances)):
                # Skip self (first neighbor)
                idx, dist = idx[1:], dist[1:]
                
                # Avoid division by zero with small epsilon
                weights = 1 / (dist + 1e-10)
                weights = weights / np.sum(weights)  # Normalize weights
                
                # Weight by inverse distance
                same_class_indicator = (self.y[idx] == self.y[i]).astype(float)
                weighted_same_class = np.sum(weights * same_class_indicator)
                weighted_si_values.append(weighted_same_class)
            
            weighted_si_scores[feature] = np.mean(weighted_si_values)
        
        # Sort and filter features
        sorted_features = dict(
            sorted(weighted_si_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def multi_resolution_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        k_values: List[int] = [3, 5, 7, 10, 15],
        aggregation: str = 'max'
    ) -> Dict[str, float]:
        """
        Evaluate feature separation at multiple neighborhood sizes to identify
        features that separate classes at different scales.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        k_values : List[int], default=[3, 5, 7, 10, 15]
            Different neighborhood sizes to evaluate
        aggregation : str, default='max'
            Method to aggregate scores across scales ('max', 'mean', 'median')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their multi-resolution scores
        """
        if any(k >= len(self.X) for k in k_values):
            raise ValueError("All k values must be less than the number of samples")
            
        # Store SI scores for each feature at each resolution
        feature_scores = defaultdict(list)
        
        # Calculate SI at each resolution
        for k in k_values:
            si_scores = {}
            for feature in self.feature_names:
                feature_values = self.X[feature].values.reshape(-1, 1)
                
                # Fit KNN
                knn = NearestNeighbors(n_neighbors=k + 1)
                knn.fit(feature_values)
                
                # Get indices of nearest neighbors
                _, indices = knn.kneighbors(feature_values)
                
                # Calculate SI
                si_values = []
                for i, idx in enumerate(indices):
                    neighbors_same_class = np.mean(self.y[idx[1:]] == self.y[i])
                    si_values.append(neighbors_same_class)
                
                si_scores[feature] = np.mean(si_values)
                feature_scores[feature].append(si_scores[feature])
        
        # Aggregate scores across resolutions
        multi_res_scores = {}
        for feature, scores in feature_scores.items():
            if aggregation == 'max':
                multi_res_scores[feature] = np.max(scores)
            elif aggregation == 'mean':
                multi_res_scores[feature] = np.mean(scores)
            elif aggregation == 'median':
                multi_res_scores[feature] = np.median(scores)
            else:
                raise ValueError("Aggregation must be 'max', 'mean', or 'median'")
        
        # Sort and filter features
        sorted_features = dict(
            sorted(multi_res_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def feature_subset_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        max_subset_size: int = 2,
        n_neighbors: int = 5,
        max_features_to_consider: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate separation power of feature combinations rather than individual features.
        
        This method detects synergistic feature interactions that enhance class separation
        beyond what individual features achieve alone.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top feature subsets to return
        threshold : float, optional
            Minimum score for subset selection
        max_subset_size : int, default=2
            Maximum number of features in a subset
        n_neighbors : int, default=5
            Number of neighbors for SI calculation
        max_features_to_consider : int, default=10
            Limit initial feature pool to control computational complexity
            
        Returns:
        --------
        Dict[str, float] or List[str]
            If top_k or threshold is specified, returns a list of individual feature names.
            Otherwise, returns a dictionary of feature subset names and their SI scores.
        """
        if n_neighbors >= len(self.X):
            raise ValueError("n_neighbors must be less than the number of samples")
            
        # Limit initial feature pool 
        if len(self.feature_names) > max_features_to_consider:
            # Preselect features using standard SI
            initial_scores = self.separation_index_selection(top_k=max_features_to_consider)
            features_to_consider = list(initial_scores.keys())
        else:
            features_to_consider = self.feature_names
            
        # Verify that all features to consider are actually in the dataframe
        features_to_consider = [f for f in features_to_consider if f in self.X.columns]
        
        if not features_to_consider:
            raise ValueError("No valid features to consider")
            
        subset_scores = {}
        
        # Evaluate individual features first
        for feature in features_to_consider:
            try:
                feature_values = self.X[feature].values.reshape(-1, 1)
                knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
                knn.fit(feature_values)
                _, indices = knn.kneighbors(feature_values)
                
                si_values = []
                for sample_idx, idx in enumerate(indices):
                    neighbors_same_class = np.mean(self.y[idx[1:]] == self.y[sample_idx])
                    si_values.append(neighbors_same_class)
                    
                subset_name = f"{feature}"
                subset_scores[subset_name] = np.mean(si_values)
            except Exception as e:
                warnings.warn(f"Error processing feature {feature}: {str(e)}")
                continue
        
        # Evaluate feature pairs if max_subset_size >= 2
        if max_subset_size >= 2:
            from itertools import combinations
            
            for size in range(2, min(max_subset_size + 1, len(features_to_consider) + 1)):
                for feature_combo in combinations(features_to_consider, size):
                    try:
                        # Create a numpy array with all features in the combination
                        features_data = []
                        for feat in feature_combo:
                            # Check if feature exists
                            if feat not in self.X.columns:
                                raise ValueError(f"Feature {feat} not found in dataframe")
                            features_data.append(self.X[feat].values)
                        
                        # Stack columns
                        combo_values = np.column_stack(features_data)
                        
                        # Calculate separation index
                        knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
                        knn.fit(combo_values)
                        _, indices = knn.kneighbors(combo_values)
                        
                        si_values = []
                        for sample_idx, idx in enumerate(indices):
                            neighbors_same_class = np.mean(self.y[idx[1:]] == self.y[sample_idx])
                            si_values.append(neighbors_same_class)
                        
                        # Join feature names with '+' for the key
                        subset_name = "+".join(feature_combo)
                        subset_scores[subset_name] = np.mean(si_values)
                    except Exception as e:
                        warnings.warn(f"Error processing feature combination {'+'.join(feature_combo)}: {str(e)}")
                        continue
                    
        # Sort feature subsets by score
        sorted_subsets = dict(sorted(subset_scores.items(), key=lambda x: x[1], reverse=True))
        
        # Filter based on criteria if specified
        if top_k:
            selected_subsets = dict(list(sorted_subsets.items())[:top_k])
        elif threshold:
            selected_subsets = {
                subset: score for subset, score in sorted_subsets.items()
                if score > threshold
            }
        else:
            # If no filtering criteria, return all scored subsets
            return sorted_subsets
            
        # Extract individual features from the selected subsets
        individual_features = set()
        # Track which features came from high-scoring subsets
        feature_importance = {}
        
        for subset, score in selected_subsets.items():
            features = subset.split('+')
            for feature in features:
                individual_features.add(feature)
                # Keep track of the highest score this feature appeared in
                if feature not in feature_importance or score > feature_importance[feature]:
                    feature_importance[feature] = score
        
        # Sort individual features by their importance scores
        sorted_features = sorted(individual_features, 
                                key=lambda x: feature_importance.get(x, 0), 
                                reverse=True)
        
        return sorted_features
    
    def adversarial_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        n_neighbors: int = 5,
        perturbation_strength: float = 0.05,
        n_perturbations: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate feature robustness by measuring separation under simulated perturbations.
        
        This innovative approach identifies features that maintain good class separation
        even when data is slightly perturbed - a key property for robust feature selection.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        n_neighbors : int, default=5
            Number of neighbors for SI calculation
        perturbation_strength : float, default=0.05
            Magnitude of random perturbations as fraction of feature range
        n_perturbations : int, default=10
            Number of perturbation trials to run
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their adversarial robustness scores
        """
        if n_neighbors >= len(self.X):
            raise ValueError("n_neighbors must be less than the number of samples")
            
        robust_scores = {}
        
        for feature in self.feature_names:
            # Get original feature values
            feature_values = self.X[feature].values.reshape(-1, 1)
            feature_range = np.max(feature_values) - np.min(feature_values)
            
            # Calculate base separation index
            knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            knn.fit(feature_values)
            _, indices = knn.kneighbors(feature_values)
            
            base_si_values = []
            for i, idx in enumerate(indices):
                neighbors_same_class = np.mean(self.y[idx[1:]] == self.y[i])
                base_si_values.append(neighbors_same_class)
            base_score = np.mean(base_si_values)
            
            # Test separation under perturbations
            perturbed_scores = []
            for _ in range(n_perturbations):
                # Add random noise
                noise = np.random.normal(0, perturbation_strength * feature_range, size=feature_values.shape)
                perturbed_values = feature_values + noise
                
                # Calculate perturbed separation index
                knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
                knn.fit(perturbed_values)
                _, indices = knn.kneighbors(perturbed_values)
                
                si_values = []
                for i, idx in enumerate(indices):
                    neighbors_same_class = np.mean(self.y[idx[1:]] == self.y[i])
                    si_values.append(neighbors_same_class)
                perturbed_scores.append(np.mean(si_values))
            
            # Robust score = average perturbed score / base score
            # (higher means more robust - maintains separation under perturbation)
            robust_scores[feature] = np.mean(perturbed_scores) / (base_score + 1e-10)
        
        # Sort and filter features
        sorted_features = dict(
            sorted(robust_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def manifold_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        n_neighbors: int = 5,
        distance_metric: str = 'euclidean'
    ) -> Dict[str, float]:
        """
        Calculate separation along local manifold structure rather than in raw feature space.
        
        This method approximates the manifold by constructing a k-nearest neighbor graph
        and measures separation based on local neighborhood topology.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        n_neighbors : int, default=5
            Number of neighbors for manifold approximation
        distance_metric : str, default='euclidean'
            Distance metric for neighbor graph construction
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their manifold separation scores
        """
        if n_neighbors >= len(self.X):
            raise ValueError("n_neighbors must be less than the number of samples")
            
        manifold_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values.reshape(-1, 1)
            
            # Construct neighbor graph
            knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=distance_metric)
            knn.fit(feature_values)
            
            # Get neighbor graph
            graph = knn.kneighbors_graph(feature_values, mode='distance')
            
            # For each sample, look at "friend of friend" patterns
            # Higher-order neighborhood analysis helps capture manifold structure
            sample_scores = []
            for i in range(len(self.X)):
                # Get direct neighbors
                direct_neighbors = graph[i].nonzero()[1]
                direct_neighbors = direct_neighbors[direct_neighbors != i]  # Exclude self
                
                # Check direct neighbor class agreement
                direct_agreement = np.mean(self.y[direct_neighbors] == self.y[i])
                
                # Get second-order neighbors (neighbors of neighbors)
                second_neighbors = set()
                for neighbor in direct_neighbors:
                    neighbor_neighbors = graph[neighbor].nonzero()[1]
                    for nn in neighbor_neighbors:
                        if nn != i and nn not in direct_neighbors:
                            second_neighbors.add(nn)
                
                second_neighbors = list(second_neighbors)
                if second_neighbors:
                    # Check second-order neighbor class agreement
                    second_agreement = np.mean(self.y[second_neighbors] == self.y[i])
                    
                    # Final score: direct agreement - decay factor * second agreement difference
                    # This rewards features where manifold locality preserves class
                    decay = 0.5
                    sample_score = direct_agreement - decay * abs(direct_agreement - second_agreement)
                else:
                    sample_score = direct_agreement
                
                sample_scores.append(sample_score)
            
            # Average score across samples
            manifold_scores[feature] = np.mean(sample_scores)
        
        # Sort and filter features
        sorted_features = dict(
            sorted(manifold_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def information_theoretic_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        bins: int = 10,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Use information theory to quantify class separation through entropy and mutual information.
        
        This method estimates the mutual information between features and class labels,
        capturing complex dependencies beyond what linear correlation can detect.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        bins : int, default=10
            Number of bins for discretization
        normalize : bool, default=True
            Whether to normalize mutual information scores
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their information scores
        """
        info_scores = {}
        
        # Calculate class entropy H(Y)
        class_counts = np.bincount(self.y)
        class_probs = class_counts / np.sum(class_counts)
        class_entropy = entropy(class_probs, base=2)
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values
            
            # Discretize feature into bins
            hist, bin_edges = np.histogram(feature_values, bins=bins)
            digitized = np.digitize(feature_values, bin_edges[:-1])
            
            # Calculate joint probability matrix
            joint_counts = np.zeros((bins, len(np.unique(self.y))))
            for i in range(len(self.X)):
                bin_idx = min(digitized[i] - 1, bins - 1)  # Ensure valid bin index
                joint_counts[bin_idx, self.y[i]] += 1
            
            # Normalize to get joint probability
            joint_probs = joint_counts / np.sum(joint_counts)
            
            # Calculate feature entropy H(X)
            feature_probs = np.sum(joint_probs, axis=1)
            feature_entropy = entropy(feature_probs[feature_probs > 0], base=2)
            
            # Calculate conditional entropy H(Y|X)
            cond_entropy = 0
            for i in range(bins):
                if feature_probs[i] > 0:
                    cond_probs = joint_probs[i, :] / feature_probs[i]
                    cond_probs = cond_probs[cond_probs > 0]
                    cond_entropy += feature_probs[i] * entropy(cond_probs, base=2)
            
            # Mutual information I(X;Y) = H(Y) - H(Y|X)
            mutual_info = class_entropy - cond_entropy
            
            # Normalized mutual information (ranges 0-1)
            if normalize and feature_entropy > 0:
                norm_mutual_info = mutual_info / min(feature_entropy, class_entropy)
            else:
                norm_mutual_info = mutual_info
            
            info_scores[feature] = norm_mutual_info
        
        # Sort and filter features
        sorted_features = dict(
            sorted(info_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def ensemble_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        n_neighbors: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Combine multiple separation metrics for more robust feature ranking.
        
        This method aggregates scores from different separation measures to produce
        a consensus ranking that's more reliable than any single metric.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        n_neighbors : int, default=5
            Number of neighbors for SI calculation
        weights : Dict[str, float], optional
            Weighting of different metrics (keys: 'standard', 'weighted', 'information')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their ensemble scores
        """
        # Default weights if not specified
        if weights is None:
            weights = {
                'standard': 1.0,
                'weighted': 1.0,
                'information': 1.0
            }
        
        # Calculate different separation metrics
        standard_scores = self.separation_index_selection(n_neighbors=n_neighbors)
        weighted_scores = self.weighted_separation_index(n_neighbors=n_neighbors)
        info_scores = self.information_theoretic_separation()
        
        # Collect all metrics into a single structure
        all_metrics = {
            'standard': standard_scores,
            'weighted': weighted_scores,
            'information': info_scores
        }
        
        # Compute ensemble score as weighted average
        ensemble_scores = {}
        for feature in self.feature_names:
            score_sum = 0
            weight_sum = 0
            
            for metric_name, metric_scores in all_metrics.items():
                if feature in metric_scores and metric_name in weights:
                    score_sum += metric_scores[feature] * weights[metric_name]
                    weight_sum += weights[metric_name]
            
            if weight_sum > 0:
                ensemble_scores[feature] = score_sum / weight_sum
            else:
                ensemble_scores[feature] = 0
        
        # Sort and filter features
        sorted_features = dict(
            sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def local_cluster_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        n_clusters: int = 5,
        cluster_method: str = 'kmeans'
    ) -> Dict[str, float]:
        """
        Evaluate separation based on local clusters in feature space.
        
        This novel approach first identifies local clusters in each feature dimension,
        then measures how well these clusters preserve class boundaries.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        n_clusters : int, default=5
            Number of clusters to form in each feature dimension
        cluster_method : str, default='kmeans'
            Clustering algorithm to use ('kmeans' or 'quantile')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their local cluster separation scores
        """
        from sklearn.cluster import KMeans
        
        cluster_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values.reshape(-1, 1)
            
            # Form clusters
            if cluster_method == 'kmeans':
                # Use KMeans clustering
                kmeans = KMeans(n_clusters=min(n_clusters, len(self.X)), random_state=42)
                cluster_labels = kmeans.fit_predict(feature_values)
                
            elif cluster_method == 'quantile':
                # Use quantile-based clustering
                quantiles = np.linspace(0, 1, n_clusters + 1)[1:-1]
                thresholds = np.quantile(feature_values, quantiles)
                cluster_labels = np.zeros(len(feature_values), dtype=int)
                
                for i, threshold in enumerate(thresholds, 1):
                    cluster_labels[feature_values.flatten() >= threshold] = i
            else:
                raise ValueError("cluster_method must be 'kmeans' or 'quantile'")
            
            # Measure class purity within each cluster
            purities = []
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    # Calculate class distribution within this cluster
                    cluster_classes = self.y[cluster_mask]
                    class_counts = np.bincount(cluster_classes, minlength=len(np.unique(self.y)))
                    class_probs = class_counts / np.sum(class_counts)
                    
                    # Purity = probability of most common class
                    purity = np.max(class_probs)
                    purities.append(purity)
            
            # Final score is average cluster purity weighted by cluster size
            cluster_sizes = np.bincount(cluster_labels)
            weighted_purities = np.array(purities) * cluster_sizes[np.unique(cluster_labels)] / len(self.X)
            cluster_scores[feature] = np.sum(weighted_purities)
        
        # Sort and filter features
        sorted_features = dict(
            sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def boundary_sensitivity_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        n_slices: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate feature separation power by analyzing decision boundary sensitivity.
        
        This novel approach identifies features with clean decision boundaries by
        measuring how class distributions change as we move along feature dimensions.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        n_slices : int, default=10
            Number of slices to divide feature range into
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their boundary sensitivity scores
        """
        boundary_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values
            
            # Create slices (bins) along feature dimension
            bin_edges = np.linspace(
                np.min(feature_values),
                np.max(feature_values),
                n_slices + 1
            )
            
            # Digitize samples into bins
            bin_indices = np.digitize(feature_values, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_slices - 1)  # Handle edge cases
            
            # Calculate class distribution in each bin
            class_distributions = []
            for bin_idx in range(n_slices):
                bin_mask = bin_indices == bin_idx
                if np.sum(bin_mask) > 0:
                    bin_classes = self.y[bin_mask]
                    if len(np.unique(self.y)) == 2:  # Binary classification
                        # For binary, just store proportion of positive class
                        class_distr = np.mean(bin_classes)
                    else:
                        # For multi-class, store normalized class counts
                        class_counts = np.bincount(bin_classes, minlength=len(np.unique(self.y)))
                        class_distr = class_counts / np.sum(class_counts)
                    
                    class_distributions.append(class_distr)
                else:
                    # Empty bin, just use overall class distribution
                    if len(np.unique(self.y)) == 2:
                        class_distr = np.mean(self.y)
                    else:
                        class_counts = np.bincount(self.y, minlength=len(np.unique(self.y)))
                        class_distr = class_counts / np.sum(class_counts)
                    
                    class_distributions.append(class_distr)
            
            # Calculate changes in distribution between adjacent bins
            # Features with clear boundaries will have abrupt changes in class distribution
            if len(np.unique(self.y)) == 2:  # Binary case
                distribution_changes = np.abs(np.diff(class_distributions))
            else:
                # For multi-class, use Jensen-Shannon distance between distributions
                distribution_changes = []
                for i in range(len(class_distributions) - 1):
                    js_distance = distance.jensenshannon(
                        class_distributions[i],
                        class_distributions[i+1]
                    )
                    distribution_changes.append(js_distance)
            
            # Score based on maximum change (sharpest boundary)
            if len(distribution_changes) > 0:
                boundary_scores[feature] = np.max(distribution_changes)
            else:
                boundary_scores[feature] = 0
        
        # Sort and filter features
        sorted_features = dict(
            sorted(boundary_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def fisher_discriminant_ratio(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Classical parametric measure of class separation based on means and variances.
        
        Fisher's discriminant ratio measures the squared difference between class means
        normalized by the sum of their variances. Higher values indicate better separation.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their Fisher scores
        """
        # Verify binary classification
        unique_classes = np.unique(self.y)
        if len(unique_classes) != 2:
            warnings.warn("Fisher's ratio is designed for binary classification. Using first two classes.")
            classes_to_use = unique_classes[:2]
        else:
            classes_to_use = unique_classes
        
        fisher_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values
            
            # Get feature values for each class
            class1_values = feature_values[self.y == classes_to_use[0]]
            class2_values = feature_values[self.y == classes_to_use[1]]
            
            # Calculate means and variances
            mean1 = np.mean(class1_values)
            mean2 = np.mean(class2_values)
            var1 = np.var(class1_values)
            var2 = np.var(class2_values)
            
            # Avoid division by zero
            if var1 + var2 == 0:
                fisher_scores[feature] = 0
            else:
                # Calculate Fisher's ratio
                fisher_scores[feature] = ((mean1 - mean2) ** 2) / (var1 + var2)
        
        # Sort and filter features
        sorted_features = dict(
            sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def interquartile_overlap(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Robust measure of class separation based on quartile ranges.
        
        This method measures class separation using the overlap between the interquartile
        ranges (IQR) of the two classes. Lower overlap means better separation.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their IQR separation scores (1 - overlap ratio)
        """
        # Verify binary classification
        unique_classes = np.unique(self.y)
        if len(unique_classes) != 2:
            warnings.warn("IQR overlap is designed for binary classification. Using first two classes.")
            classes_to_use = unique_classes[:2]
        else:
            classes_to_use = unique_classes
        
        overlap_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values
            
            # Get feature values for each class
            class1_values = feature_values[self.y == classes_to_use[0]]
            class2_values = feature_values[self.y == classes_to_use[1]]
            
            # Calculate quartiles
            q1_1, q3_1 = np.percentile(class1_values, [25, 75])
            q1_2, q3_2 = np.percentile(class2_values, [25, 75])
            
            # Calculate IQR ranges
            iqr1_range = [q1_1, q3_1]
            iqr2_range = [q1_2, q3_2]
            
            # Calculate overlap
            overlap_start = max(iqr1_range[0], iqr2_range[0])
            overlap_end = min(iqr1_range[1], iqr2_range[1])
            
            if overlap_end <= overlap_start:
                # No overlap
                overlap_ratio = 0
            else:
                overlap_length = overlap_end - overlap_start
                union_length = max(iqr1_range[1], iqr2_range[1]) - min(iqr1_range[0], iqr2_range[0])
                overlap_ratio = overlap_length / union_length if union_length > 0 else 0
            
            # Convert to separation score (1 - overlap ratio)
            overlap_scores[feature] = 1 - overlap_ratio
        
        # Sort and filter features
        sorted_features = dict(
            sorted(overlap_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def maximum_mean_discrepancy(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        kernel: str = 'rbf',
        gamma: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Kernel-based measure of distribution differences using Maximum Mean Discrepancy.
        
        MMD measures the distance between distributions in a Reproducing Kernel Hilbert Space.
        Higher MMD indicates greater separation between class distributions.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        kernel : str, default='rbf'
            Kernel type ('rbf', 'linear', 'poly')
        gamma : float, optional
            Kernel coefficient for 'rbf' kernel. If None, uses 1/n_features
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their MMD scores
        """
        from sklearn.metrics.pairwise import pairwise_kernels
        
        # Verify binary classification
        unique_classes = np.unique(self.y)
        if len(unique_classes) != 2:
            warnings.warn("MMD is designed for binary classification. Using first two classes.")
            classes_to_use = unique_classes[:2]
        else:
            classes_to_use = unique_classes
        
        mmd_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values.reshape(-1, 1)
            
            # Get feature values for each class
            class1_indices = self.y == classes_to_use[0]
            class2_indices = self.y == classes_to_use[1]
            
            class1_values = feature_values[class1_indices]
            class2_values = feature_values[class2_indices]
            
            n1 = len(class1_values)
            n2 = len(class2_values)
            
            if n1 == 0 or n2 == 0:
                mmd_scores[feature] = 0
                continue
            
            # Calculate kernel matrices
            kernel_params = {'gamma': gamma} if gamma is not None else {}
            K_11 = pairwise_kernels(class1_values, metric=kernel, **kernel_params)
            K_22 = pairwise_kernels(class2_values, metric=kernel, **kernel_params)
            K_12 = pairwise_kernels(class1_values, class2_values, metric=kernel, **kernel_params)
            
            # Calculate MMD^2
            term1 = np.sum(K_11) / (n1 * n1)
            term2 = np.sum(K_22) / (n2 * n2)
            term3 = 2 * np.sum(K_12) / (n1 * n2)
            
            mmd_squared = term1 + term2 - term3
            mmd_scores[feature] = max(0, mmd_squared)  # Ensure non-negative
        
        # Sort and filter features
        sorted_features = dict(
            sorted(mmd_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def decision_stump_selection(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        criterion: str = 'gini',
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Uses simple decision tree stumps to measure feature separability.
        
        For each feature, trains a decision stump (depth-1 tree) and measures its
        performance. Higher scores indicate better class separation.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        criterion : str, default='gini'
            Split criterion for the decision tree ('gini' or 'entropy')
        metric : str, default='accuracy'
            Performance metric ('accuracy' or 'auc')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their decision stump performance scores
        """
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import cross_val_score
        
        stump_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values.reshape(-1, 1)
            
            # Create and train decision stump
            stump = DecisionTreeClassifier(max_depth=1, criterion=criterion)
            
            # Use cross-validation to get more robust performance estimate
            if metric == 'accuracy':
                scores = cross_val_score(stump, feature_values, self.y, cv=5, scoring='accuracy')
            elif metric == 'auc':
                # AUC requires probability estimates and only works for binary classification
                if len(np.unique(self.y)) != 2:
                    warnings.warn("AUC metric only works for binary classification. Using accuracy instead.")
                    scores = cross_val_score(stump, feature_values, self.y, cv=5, scoring='accuracy')
                else:
                    scores = cross_val_score(stump, feature_values, self.y, cv=5, scoring='roc_auc')
            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'accuracy' or 'auc'.")
            
            # Use mean cross-validation score
            stump_scores[feature] = np.mean(scores)
        
        # Sort and filter features
        sorted_features = dict(
            sorted(stump_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def kolmogorov_smirnov_selection(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Non-parametric measure based on maximum distance between class CDFs.
        
        Uses the Kolmogorov-Smirnov statistic to measure the maximum difference
        between the empirical cumulative distribution functions of the two classes.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their KS statistic scores
        """
        from scipy.stats import ks_2samp
        
        # Verify binary classification
        unique_classes = np.unique(self.y)
        if len(unique_classes) != 2:
            warnings.warn("KS statistic is designed for binary classification. Using first two classes.")
            classes_to_use = unique_classes[:2]
        else:
            classes_to_use = unique_classes
        
        ks_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values
            
            # Get feature values for each class
            class1_values = feature_values[self.y == classes_to_use[0]]
            class2_values = feature_values[self.y == classes_to_use[1]]
            
            if len(class1_values) == 0 or len(class2_values) == 0:
                ks_scores[feature] = 0
                continue
            
            # Calculate KS statistic
            ks_stat, p_value = ks_2samp(class1_values, class2_values)
            ks_scores[feature] = ks_stat
        
        # Sort and filter features
        sorted_features = dict(
            sorted(ks_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def nearest_hits_misses(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        n_neighbors: int = 5,
        metric: str = 'euclidean'
    ) -> Dict[str, float]:
        """
        Enhanced nearest neighbor approach comparing same-class vs different-class distances.
        
        For each sample, calculates the average distance to nearest neighbors of the same class
        (hits) and different class (misses). Features with larger difference (misses - hits)
        are better at separating classes.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        n_neighbors : int, default=5
            Number of neighbors to consider
        metric : str, default='euclidean'
            Distance metric to use
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their hit/miss distance scores
        """
        from sklearn.neighbors import NearestNeighbors
        
        if n_neighbors >= len(self.X):
            raise ValueError("n_neighbors must be less than the number of samples")
        
        hitmiss_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values.reshape(-1, 1)
            
            # Fit nearest neighbors model
            nn = NearestNeighbors(n_neighbors=len(self.X)-1, metric=metric)
            nn.fit(feature_values)
            
            # Get distances and indices of all other points (excluding self)
            distances, indices = nn.kneighbors(feature_values)
            
            # Calculate hit/miss statistics for each sample
            hit_distances = []
            miss_distances = []
            
            for i in range(len(self.X)):
                # Current sample's class
                sample_class = self.y[i]
                
                # Find indices of neighbors with same class (hits) and different class (misses)
                neighbor_classes = self.y[indices[i]]
                hit_mask = neighbor_classes == sample_class
                miss_mask = ~hit_mask
                
                # Get distances to hits and misses
                sample_hit_distances = distances[i][hit_mask]
                sample_miss_distances = distances[i][miss_mask]
                
                # Take only the n_neighbors closest hits and misses
                if len(sample_hit_distances) > 0:
                    hit_distances.append(np.mean(sample_hit_distances[:min(n_neighbors, len(sample_hit_distances))]))
                if len(sample_miss_distances) > 0:
                    miss_distances.append(np.mean(sample_miss_distances[:min(n_neighbors, len(sample_miss_distances))]))
            
            # Calculate average hit and miss distances across all samples
            avg_hit_distance = np.mean(hit_distances) if hit_distances else 0
            avg_miss_distance = np.mean(miss_distances) if miss_distances else 0
            
            # Score is the difference: larger values mean better separation
            # Normalize by the sum to get a relative measure
            denominator = avg_hit_distance + avg_miss_distance
            if denominator > 0:
                hitmiss_scores[feature] = (avg_miss_distance - avg_hit_distance) / denominator
            else:
                hitmiss_scores[feature] = 0
        
        # Sort and filter features
        sorted_features = dict(
            sorted(hitmiss_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features
    
    def imbalanced_separation(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        n_neighbors: int = 5,
        weighting: str = 'inverse_frequency'
    ) -> Dict[str, float]:
        """
        Weighted separation measures for imbalanced datasets.
        
        Adapts the separation index to prioritize correctly separating minority class samples
        by applying class-specific weights to each sample's contribution.
        
        Parameters:
        -----------
        top_k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum score threshold for feature selection
        n_neighbors : int, default=5
            Number of neighbors for SI calculation
        weighting : str, default='inverse_frequency'
            Weighting strategy ('inverse_frequency', 'balanced', 'f1_oriented')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their imbalanced separation scores
        """
        if n_neighbors >= len(self.X):
            raise ValueError("n_neighbors must be less than the number of samples")
        
        # Calculate class weights
        class_counts = np.bincount(self.y)
        n_samples = len(self.y)
        
        if weighting == 'inverse_frequency':
            # Inverse of class frequency
            class_weights = n_samples / (len(class_counts) * class_counts)
        elif weighting == 'balanced':
            # Equal weight to each class
            class_weights = np.ones_like(class_counts) * (n_samples / len(class_counts)) / class_counts
        elif weighting == 'f1_oriented':
            # Weights that emphasize both precision and recall for minority class
            # Higher weight for minority class (roughly sqrt of inverse frequency)
            class_weights = np.sqrt(n_samples / (len(class_counts) * class_counts))
        else:
            raise ValueError(f"Unknown weighting strategy: {weighting}")
        
        # Assign weight to each sample based on its class
        sample_weights = np.array([class_weights[c] for c in self.y])
        
        imbalanced_scores = {}
        
        for feature in self.feature_names:
            feature_values = self.X[feature].values.reshape(-1, 1)
            
            # Fit KNN
            knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            knn.fit(feature_values)
            
            # Get indices of nearest neighbors
            _, indices = knn.kneighbors(feature_values)
            
            # Calculate weighted SI for each sample
            weighted_si_values = []
            for i, idx in enumerate(indices):
                # Exclude the first neighbor (self)
                neighbor_idx = idx[1:]
                
                # Count neighbors of same class
                same_class = np.mean(self.y[neighbor_idx] == self.y[i])
                
                # Weight by sample importance
                weighted_si_values.append(same_class * sample_weights[i])
            
            # Store average weighted SI for this feature
            # Normalize by average weight to get comparable scores
            imbalanced_scores[feature] = np.sum(weighted_si_values) / np.sum(sample_weights)
        
        # Sort and filter features
        sorted_features = dict(
            sorted(imbalanced_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            selected_features = dict(list(sorted_features.items())[:top_k])
        elif threshold:
            selected_features = {
                feat: score for feat, score in sorted_features.items() 
                if score > threshold
            }
        else:
            selected_features = sorted_features
            
        return selected_features