import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import json
import os
import warnings
import copy
import logging # Added for logging

# Check for optional base classifiers
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. XGBoost base classifier will not be available.")


class ModelSelectorClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that trains multiple base models and a selector model to determine
    which base model is best suited for each new data point based on its features
    and the cluster it belongs to.
    """
    
    def __init__(self, base_models=None, n_clusters=5, selector_model=None, 
                 n_cv_splits=5, n_seeds=3, random_state=42, 
                 do_gridsearch=True, grid_search_cache_path='model_selector_gs_cache.json',
                 verbose=True):
        """
        Initialize the ModelSelectorClassifier.
        
        Parameters:
        -----------
        base_models : dict, optional
            Dictionary of base models to use. If None, default models will be used.
        n_clusters : int, default=5
            Number of clusters to use in KMeans clustering.
        selector_model : object, default=None
            Model to use as the selector. If None, RandomForestClassifier will be used.
        n_cv_splits : int, default=5
            Number of cross-validation splits for generating selector training data.
        n_seeds : int, default=3
            Number of different random seeds to use for cross-validation repeats.
        random_state : int, default=42
            Random state for reproducibility.
        do_gridsearch : bool, default=True
            Whether to perform grid search for hyperparameter optimization.
        grid_search_cache_path : str, default='model_selector_gs_cache.json'
            Path to store/load grid search results.
        verbose : bool, default=True
            Whether to print progress messages.
        """
        # Initialize grid search parameters
        self.do_gridsearch = do_gridsearch
        self.grid_search_cache_path = grid_search_cache_path
        self.verbose = verbose
        
        # Define default base models
        self.base_models = base_models if base_models is not None else {
            'RF': RandomForestClassifier(random_state=random_state),
            'SVM': SVC(probability=True, random_state=random_state),
            'KNN': KNeighborsClassifier(),
            'LR': LogisticRegression(random_state=random_state, max_iter=5000),
            'DT': DecisionTreeClassifier(random_state=random_state),
            'NB': GaussianNB(),
            'GB': GradientBoostingClassifier(random_state=random_state)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.base_models['XGB'] = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
            
        # Define parameter grids for grid search
        self.base_param_grids = {
            'RF': {
                'n_estimators': [50, 100],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'LR': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['saga']  # Saga supports l1/l2
            },
            'DT': {
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf']  # Keep kernel='rbf' as default
            },
            'KNN': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            'GB': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4]
            },
            'NB': {}  # NB typically doesn't need grid search
        }
        
        # Add XGBoost params if available
        if XGBOOST_AVAILABLE:
            self.base_param_grids['XGB'] = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        self.n_clusters = n_clusters
        self.selector_model = selector_model if selector_model is not None else (
            RandomForestClassifier(n_estimators=100, random_state=random_state))
        self.n_cv_splits = n_cv_splits
        self.n_seeds = n_seeds
        self.random_state = random_state
        
        # These will be set during fit
        self.clusterer = None
        self.scaler = None
        self.trained_base_models = {}
        self.base_model_accuracies = {}
        self.target_dtype_ = None 
        self.most_frequent_class_ = None 
        self.classes_ = None # Store classes seen during fit
    
    def fit(self, X, y):
        """
        Fit the model selector classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Store target dtype, classes and find most frequent class
        self.target_dtype_ = y.dtype
        self.classes_ = np.unique(y) # Store unique classes
        from collections import Counter
        self.most_frequent_class_ = Counter(y).most_common(1)[0][0]

        # Initialize components
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster the data
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        clusters = self.clusterer.fit_predict(X_scaled)
        
        # Store for later use
        self.trained_base_models = {}
        self.base_model_accuracies = {}
        
        # --- Grid Search Logic for Base Models ---
        base_model_params = {}
        if self.do_gridsearch:
            if self.verbose: 
                print("--- Base Model GridSearchCV Enabled ---")
            
            # Try loading from cache
            if os.path.exists(self.grid_search_cache_path):
                try:
                    with open(self.grid_search_cache_path, 'r') as f:
                        base_model_params = json.load(f)
                    if self.verbose: 
                        print(f"Loaded base model parameters from cache: {self.grid_search_cache_path}")
                except (json.JSONDecodeError, IOError) as e:
                    if self.verbose: 
                        print(f"Cache file found but failed to load ({e}). Running GridSearchCV.")
                    base_model_params = {}  # Reset if loading failed
            
            # If cache miss or load failed, run GridSearchCV
            if not base_model_params:
                if self.verbose: 
                    print("Performing GridSearchCV for base models...")
                
                for name, model in self.base_models.items():
                    if name in self.base_param_grids and self.base_param_grids[name]:
                        if self.verbose: 
                            print(f"  GridSearching for {name}...")
                        
                        # Use a fresh instance for GS to avoid state issues
                        clf_instance = copy.deepcopy(model)
                        
                        # Ensure SVM gets probability=True if needed
                        if name == "SVM":
                            clf_instance.probability = True
                        
                        gs = GridSearchCV(clf_instance, self.base_param_grids[name], 
                                         cv=3, scoring='accuracy', n_jobs=-1)
                        try:
                            gs.fit(X_scaled, y)
                            base_model_params[name] = gs.best_params_
                            if self.verbose: 
                                print(f"    Best params for {name}: {gs.best_params_}")
                        except Exception as e:
                            print(f"    GridSearchCV failed for {name}: {e}. Using defaults.")
                            # Keep original model parameters
                            base_model_params[name] = {}
                
                # Save results to cache
                try:
                    # Convert numpy types to standard types for JSON serialization
                    serializable_params = {}
                    for name, params in base_model_params.items():
                        serializable_params[name] = {k: (v.item() if hasattr(v, 'item') else v) 
                                                    for k, v in params.items()}
                    
                    with open(self.grid_search_cache_path, 'w') as f:
                        json.dump(serializable_params, f, indent=4)
                    if self.verbose: 
                        print(f"Saved base model parameters to cache: {self.grid_search_cache_path}")
                except (IOError, TypeError) as e:
                    if self.verbose: 
                        print(f"Failed to save cache file: {e}")
        
        # Update models with optimized parameters
        for name, model in self.base_models.items():
            if name in base_model_params and base_model_params[name]:
                # Create a copy of the model with optimized parameters
                optimized_model = copy.deepcopy(model)
                try:
                    optimized_model.set_params(**base_model_params[name])
                    self.base_models[name] = optimized_model
                except Exception as e:
                    print(f"Error setting parameters for {name}: {e}. Using default parameters.")
        
        # Generate training data for the selector model
        selector_X, selector_y = self._generate_selector_training_data(X_scaled, y, clusters)
        
        # Train the final base models on all data
        for name, model in self.base_models.items():
            try:
                self.trained_base_models[name] = model.fit(X_scaled, y)
                if self.verbose:
                    print(f"Trained base model: {name}")
            except Exception as e:
                print(f"Error training {name}: {e}. Model will be unavailable.")
        
        # Train the selector model
        # Add cluster information to the features
        selector_features = np.column_stack([selector_X, self._get_cluster_features(selector_X)])
        self.selector_model.fit(selector_features, selector_y)
        
        return self
    
    def _generate_selector_training_data(self, X, y, clusters):
        """
        Generate training data for the selector model through cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Scaled training data.
        y : array-like
            Target values.
        clusters : array-like
            Cluster assignments for each sample.
            
        Returns:
        --------
        selector_X : array-like
            Features for the selector model (original features + cluster info).
        selector_y : array-like
            Target labels for the selector model (which base model to use).
        """
        # Initialize arrays to store votes for each model on each sample
        n_samples = X.shape[0]
        model_votes = {name: np.zeros(n_samples) for name in self.base_models.keys()}
        
        # Repeat CV process with different seeds
        for seed in range(self.n_seeds):
            # Create stratified k-fold for cross-validation
            cv = StratifiedKFold(n_splits=self.n_cv_splits, shuffle=True, 
                                random_state=self.random_state + seed)
            
            # For each fold
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train each base model on training portion
                fold_models = {}
                fold_accuracies = {}
                
                for name, model in self.base_models.items():
                    # Clone the model to avoid affecting the original
                    fold_model = clone(model)
                    fold_model.fit(X_train, y_train)
                    
                    # Predict on validation set and calculate accuracy
                    y_pred = fold_model.predict(X_val)
                    fold_accuracies[name] = accuracy_score(y_val, y_pred)
                    
                    # Store the model
                    fold_models[name] = fold_model
                
                # Determine which model is best for each validation sample
                for i, idx in enumerate(val_idx):
                    correct_models = []
                    
                    # Check which models correctly classify this sample
                    for name, model in fold_models.items():
                        if model.predict([X_val[i]])[0] == y_val[i]:
                            correct_models.append(name)
                    
                    # If at least one model is correct, vote for the one with highest accuracy
                    if correct_models:
                        best_model = max(correct_models, key=lambda name: fold_accuracies[name])
                        model_votes[best_model][idx] += 1
        
        # For each sample, determine the model with the most votes
        selector_y = np.array([max(model_votes.keys(), 
                                  key=lambda name: model_votes[name][i]) 
                              for i in range(n_samples)])
        
        # Calculate final accuracy for each base model on all data
        for name, model in self.base_models.items():
            temp_model = clone(model)
            temp_model.fit(X, y)
            self.base_model_accuracies[name] = accuracy_score(y, temp_model.predict(X))
        
        # Include cluster information in the features
        selector_X = X
        
        return selector_X, selector_y
    
    def _get_cluster_features(self, X):
        """
        Get cluster assignments and distances for samples.
        
        Parameters:
        -----------
        X : array-like
            Data samples.
            
        Returns:
        --------
        cluster_features : array-like
            Array containing cluster assignments and distances to cluster centers.
        """
        clusters = self.clusterer.predict(X)
        distances = self.clusterer.transform(X)  # Distance to each cluster center
        
        # Create one-hot encoding of cluster assignments
        cluster_one_hot = np.zeros((X.shape[0], self.n_clusters))
        for i, cluster in enumerate(clusters):
            cluster_one_hot[i, cluster] = 1
            
        # Combine cluster assignments and distances
        return np.hstack([cluster_one_hot, distances])

    def predict(self, X, y_true=None, logger=None):
        """
        Predict class labels for samples in X and optionally log details.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        y_true : array-like of shape (n_samples,), optional
            True labels for logging comparison.
        logger : logging.Logger, optional
            Logger instance for detailed logging.
            
        Returns:
        --------
        y_pred : array-like
            Predicted class labels.
        """
        if logger and y_true is None:
            logger.warning("y_true not provided to predict method. Cannot log prediction correctness.")
            
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get cluster features
        cluster_features = self._get_cluster_features(X_scaled)
        
        # Combine original features with cluster features for selector
        selector_features = np.column_stack([X_scaled, cluster_features])
        
        # Use selector model to predict which base model to use for each sample
        selected_models = self.selector_model.predict(selector_features)
        
        # Use the selected model for each sample
        y_pred = np.empty(X.shape[0], dtype=self.target_dtype_)
        
        for i, selected_model_name in enumerate(selected_models):
            final_prediction = None
            log_entry = {}
            
            if logger:
                log_entry['sample_index'] = i
                log_entry['selected_model'] = selected_model_name
                log_entry['base_model_predictions'] = {}

            # Get predictions from all base models for logging
            if logger:
                for name, model in self.trained_base_models.items():
                    try:
                        pred = model.predict([X_scaled[i]])[0]
                        log_entry['base_model_predictions'][name] = pred
                    except Exception as e:
                        log_entry['base_model_predictions'][name] = f"Error: {e}"

            # Make the final prediction using the selected model
            try:
                if selected_model_name in self.trained_base_models:
                    final_prediction = self.trained_base_models[selected_model_name].predict([X_scaled[i]])[0]
                else:
                    # Fallback strategy
                    if self.verbose:
                        print(f"Warning: Selected model '{selected_model_name}' not available for sample {i}. Using fallback.")
                    if logger:
                        log_entry['fallback_reason'] = f"Selected model '{selected_model_name}' not available."
                        
                    available_models = list(self.trained_base_models.keys())
                    if available_models:
                        best_available = max(available_models, key=lambda m: self.base_model_accuracies.get(m, 0))
                        final_prediction = self.trained_base_models[best_available].predict([X_scaled[i]])[0]
                        if logger:
                            log_entry['fallback_model_used'] = best_available
                    else:
                        if self.verbose:
                            print(f"Warning: No models available for sample {i}. Predicting most frequent class.")
                        if logger:
                            log_entry['fallback_reason'] = "No base models available."
                        final_prediction = self.most_frequent_class_
                        
            except Exception as e:
                print(f"Error predicting sample {i} with model {selected_model_name}: {e}")
                if logger:
                    log_entry['error'] = f"Prediction error with {selected_model_name}: {e}"
                # Fallback to most frequent class on error
                final_prediction = self.most_frequent_class_

            y_pred[i] = final_prediction

            # Log details if logger is provided
            if logger:
                log_entry['final_prediction'] = final_prediction
                if y_true is not None:
                    log_entry['true_target'] = y_true[i]
                    log_entry['is_correct'] = (final_prediction == y_true[i])
                
                # Convert numpy types for JSON compatibility if logging as JSON
                log_entry_serializable = json.loads(json.dumps(log_entry, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o)))
                logger.info(json.dumps(log_entry_serializable)) # Log as JSON string

        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns:
        --------
        y_proba : array-like
            Predicted class probabilities.
        """
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get cluster features
        cluster_features = self._get_cluster_features(X_scaled)
        
        # Combine original features with cluster features for selector
        selector_features = np.column_stack([X_scaled, cluster_features])
        
        # Use selector model to predict which base model to use for each sample
        selected_models = self.selector_model.predict(selector_features)
        
        # Ensure classes_ attribute is set
        if self.classes_ is None:
             raise RuntimeError("Model has not been fitted yet.")
        n_classes = len(self.classes_)

        # Use the selected model for each sample
        y_proba = np.zeros((X.shape[0], n_classes))
        
        for i, model_name in enumerate(selected_models):
            # Handle cases where the selected model might not be available
            if model_name not in self.trained_base_models:
                 # Fallback: Use the best available model or uniform probability
                available_models = list(self.trained_base_models.keys())
                if available_models:
                    best_available = max(available_models, key=lambda m: self.base_model_accuracies.get(m, 0))
                    model_to_use = self.trained_base_models[best_available]
                    if self.verbose:
                        print(f"Warning: Selected model '{model_name}' not available for proba sample {i}. Using fallback '{best_available}'.")
                else:
                    # No models available, return uniform probabilities
                    if self.verbose:
                        print(f"Warning: No models available for proba sample {i}. Returning uniform probabilities.")
                    y_proba[i] = np.ones(n_classes) / n_classes
                    continue # Skip to next sample
            else:
                 model_to_use = self.trained_base_models[model_name]

            # Predict probabilities using the chosen model
            if hasattr(model_to_use, 'predict_proba'):
                try:
                    proba = model_to_use.predict_proba([X_scaled[i]])[0]
                    # Ensure proba aligns with self.classes_
                    aligned_proba = np.zeros(n_classes)
                    model_classes = model_to_use.classes_
                    for j, cls in enumerate(model_classes):
                        target_idx = np.where(self.classes_ == cls)[0]
                        if len(target_idx) > 0:
                            aligned_proba[target_idx[0]] = proba[j]
                    y_proba[i] = aligned_proba
                except Exception as e:
                    print(f"Error getting proba for sample {i} with model {model_name}: {e}. Returning uniform.")
                    y_proba[i] = np.ones(n_classes) / n_classes
            else:
                # If model doesn't have predict_proba, create one-hot based on prediction
                try:
                    pred = model_to_use.predict([X_scaled[i]])[0]
                    prob = np.zeros(n_classes)
                    class_idx = np.where(self.classes_ == pred)[0]
                    if len(class_idx) > 0:
                         prob[class_idx[0]] = 1.0
                    y_proba[i] = prob
                except Exception as e:
                    print(f"Error predicting for proba fallback for sample {i} with model {model_name}: {e}. Returning uniform.")
                    y_proba[i] = np.ones(n_classes) / n_classes
                    
        return y_proba
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model.
        """
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model.
            
        Returns:
        --------
        model : ModelSelectorClassifier
            The loaded model.
        """
        return joblib.load(filepath)
