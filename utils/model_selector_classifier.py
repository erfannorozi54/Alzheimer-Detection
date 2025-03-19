import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging
logging.basicConfig(filename='model_selection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ModelSelectorClassifier(BaseEstimator, ClassifierMixin):
    """
    An advanced model selector classifier that:
    1. Focuses on correct model selection by training specifically on samples where 
       at least one model makes a correct prediction
    2. Uses a dual-stage meta-learning approach: first predict if a model will be correct,
       then select among correct models
    3. Uses feature space segmentation for regional specialization
    4. Employs confidence calibration and ensemble techniques
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        meta_learner='rf',  # 'rf' or 'lr'
        verbose: bool = True,
        metric: str = 'accuracy',  # 'accuracy', 'f1', 'auc', 'log_loss'
        n_clusters: int = 10,  # Number of feature space clusters
        use_feature_weights: bool = True,  # Weight features based on importance
        use_correctness_classifier: bool = True,  # Use two-stage approach
        confidence_threshold: float = 0.6  # Threshold for considering a model "confident"
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = meta_learner
        self.verbose = verbose
        self.metric = metric
        self.n_clusters = n_clusters
        self.use_feature_weights = use_feature_weights
        self.use_correctness_classifier = use_correctness_classifier
        self.confidence_threshold = confidence_threshold
        
    def _initialize_classifiers(self):
        """Initialize base classifiers with proper parameters."""
        self.base_classifiers_ = {
            "RF": RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=self.random_state),
            "LR": LogisticRegression(max_iter=5000, C=1.0, solver='saga', random_state=self.random_state), 
            "XGB": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=self.random_state),
            "DT": DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=self.random_state),
            "SVM": SVC(C=1.0, kernel='rbf', probability=True, random_state=self.random_state),
            "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
            "NB": GaussianNB(),
            "GB": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=self.random_state)
        }
        # Store classifier names in a list to ensure consistent ordering
        self.classifier_names_ = list(self.base_classifiers_.keys())
    
    def _create_meta_learner(self, regression=True):
        """Create a meta-learner based on the specified type."""
        if regression:
            if self.meta_learner == 'rf':
                return RandomForestRegressor(
                    n_estimators=500,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=self.random_state
                )
            else:
                return Ridge(
                    alpha=1.0,
                    random_state=self.random_state
                )
        else:
            if self.meta_learner == 'rf':
                return RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=self.random_state
                )
            else:
                return LogisticRegression(
                    max_iter=5000,
                    C=1.0,
                    solver='saga',
                    random_state=self.random_state
                )

    def _calculate_confidence_features(self, probas):
        """Calculate confidence metrics for a probability array."""
        # Sort probabilities in descending order
        sorted_probas = np.sort(probas, axis=1)[:, ::-1]
        
        # Margin: difference between highest and second highest probability
        margin = sorted_probas[:, 0] - sorted_probas[:, 1]
        
        # Entropy: measure of uncertainty (-sum(p*log(p)))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probas * np.log2(probas + epsilon), axis=1)
        
        # Confidence: highest probability
        confidence = sorted_probas[:, 0]
        
        # Dispersion: standard deviation of probabilities
        dispersion = np.std(probas, axis=1)
        
        return np.column_stack([margin, entropy, confidence, dispersion])

    def _evaluate_classifier(self, y_true, y_pred, y_proba=None):
        """Evaluate a classifier using the specified metric."""
        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted')
        elif self.metric == 'auc':
            if y_proba is None:
                raise ValueError("Probability estimates required for AUC calculation")
            # For multi-class, use One-vs-Rest AUC
            return roc_auc_score(y_true, y_proba, multi_class='ovr')
        elif self.metric == 'log_loss':
            if y_proba is None:
                raise ValueError("Probability estimates required for log loss calculation")
            return -log_loss(y_true, y_proba)  # Negative so higher is better
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _segment_feature_space(self, X):
        """Segment the feature space into regions using K-means clustering."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return kmeans.fit(X)

    def fit(self, X, y):
        """
        Fit the classifier using a more sophisticated meta-learning approach.
        """
        # Input validation
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize components
        self.scaler_ = StandardScaler()
        self._initialize_classifiers()
        
        # Scale features
        X_scaled = self.scaler_.fit_transform(X)
        
        # Segment the feature space (for regional specialization)
        self.kmeans_ = self._segment_feature_space(X_scaled)
        clusters = self.kmeans_.predict(X_scaled)
        
        # Use stratified K-fold to maintain class distribution
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Initialize storage for meta-learner training data
        n_samples = X.shape[0]
        
        # Each base classifier contributes:
        # - Class probabilities
        # - Confidence metrics (margin, entropy, confidence, dispersion)
        base_meta_features = len(self.classifier_names_) * (n_classes + 4)
        
        # Add cluster indicators and distances to centroids
        extra_features = self.n_clusters + 1  # One-hot cluster + distance to assigned centroid
        
        # Total number of meta-features
        n_meta_features = base_meta_features + extra_features
        
        X_meta_train = np.zeros((n_samples, n_meta_features))
        
        # For tracking individual model performance
        base_performances = {name: 0.0 for name in self.classifier_names_}
        cluster_performances = {name: {i: [] for i in range(self.n_clusters)} for name in self.classifier_names_}
        
        # Store all predictions, probabilities, and correctness
        all_val_preds = {}
        all_val_probas = {}
        all_val_correctness = {}
        for name in self.classifier_names_:
            all_val_preds[name] = []
            all_val_probas[name] = []
            all_val_correctness[name] = []
        all_val_labels = []
        all_val_clusters = []
        
        # For each fold in cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            if self.verbose:
                print(f"\nFold {fold_idx+1}/{self.n_folds}")
                print("-" * 40)
            
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_val_fold = y[val_idx]
            
            # Get clusters for validation samples
            val_clusters = clusters[val_idx]
            all_val_clusters.append(val_clusters)
            
            # Train each base classifier and collect meta-features
            for i, name in enumerate(self.classifier_names_):
                # Get the classifier
                clf = self.base_classifiers_[name]
                
                # Train on the fold's training data
                clf.fit(X_train_fold, y_train_fold)
                
                # Make predictions on validation data
                y_val_pred = clf.predict(X_val_fold)
                y_val_proba = clf.predict_proba(X_val_fold)
                
                # Store predictions and probabilities
                all_val_preds[name].append(y_val_pred)
                all_val_probas[name].append(y_val_proba)
                
                # Check correctness
                correctness = (y_val_pred == y_val_fold).astype(int)
                all_val_correctness[name].append(correctness)
                
                # Calculate performance for this fold
                performance = self._evaluate_classifier(y_val_fold, y_val_pred, y_val_proba)
                base_performances[name] += performance / self.n_folds
                
                # Calculate performance by cluster
                for cluster_id in range(self.n_clusters):
                    cluster_mask = (val_clusters == cluster_id)
                    if np.sum(cluster_mask) > 0:
                        cluster_perf = self._evaluate_classifier(
                            y_val_fold[cluster_mask], 
                            y_val_pred[cluster_mask],
                            y_val_proba[cluster_mask] if cluster_mask.any() else None
                        )
                        cluster_performances[name][cluster_id].append(cluster_perf)
                
                if self.verbose:
                    print(f"{name} {self.metric}: {performance:.4f}")
                
                # Calculate starting index for this classifier's meta-features
                start_idx = i * (n_classes + 4)
                
                # Store probabilities as meta-features
                X_meta_train[val_idx, start_idx:start_idx+n_classes] = y_val_proba
                
                # Calculate and store confidence features
                confidence_features = self._calculate_confidence_features(y_val_proba)
                X_meta_train[val_idx, start_idx+n_classes:start_idx+n_classes+4] = confidence_features
            
            # Store validation labels
            all_val_labels.append(y_val_fold)
            
            # Add cluster indicators and distance to centroid
            for i, idx in enumerate(val_idx):
                # Add one-hot encoding of cluster
                cluster_id = val_clusters[i]
                cluster_feature_start = base_meta_features
                X_meta_train[idx, cluster_feature_start + cluster_id] = 1
                
                # Add distance to assigned centroid
                centroid = self.kmeans_.cluster_centers_[cluster_id]
                distance = np.linalg.norm(X_scaled[idx] - centroid)
                X_meta_train[idx, base_meta_features + self.n_clusters] = distance
        
        # Concatenate all validation predictions, probabilities, and labels
        for name in self.classifier_names_:
            all_val_preds[name] = np.concatenate(all_val_preds[name])
            all_val_probas[name] = np.vstack(all_val_probas[name])
            all_val_correctness[name] = np.concatenate(all_val_correctness[name])
        all_val_labels = np.concatenate(all_val_labels)
        all_val_clusters = np.concatenate(all_val_clusters)
        
        # Print cluster-specific performance for each model
        if self.verbose:
            print("\n=== Cluster-Specific Performance ===")
            for cluster_id in range(self.n_clusters):
                print(f"\nCluster {cluster_id} Performance:")
                print("-" * 40)
                for name in self.classifier_names_:
                    perfs = cluster_performances[name][cluster_id]
                    if perfs:
                        avg_perf = np.mean(perfs)
                        print(f"{name:<15} | {avg_perf:.4f}")
        
        # Create correctness classifiers (predict if a model will be correct)
        if self.use_correctness_classifier:
            self.correctness_classifiers_ = {}
            
            for name in self.classifier_names_:
                # Create binary classifier to predict if this model will be correct
                correctness_clf = self._create_meta_learner(regression=False)
                
                # Target is the correctness of this model's predictions
                y_correct = all_val_correctness[name]
                
                # Train the classifier
                correctness_clf.fit(X_meta_train, y_correct)
                
                self.correctness_classifiers_[name] = correctness_clf
        
        # ===== CRITICAL CHANGE HERE =====
        # Only use samples where at least one model is correct for performance prediction
        correct_samples_mask = np.zeros(n_samples, dtype=bool)
        for name in self.classifier_names_:
            correct_samples_mask = correct_samples_mask | (all_val_correctness[name] == 1)
        
        # Count how many samples have at least one correct model
        n_correct_samples = correct_samples_mask.sum()
        
        if self.verbose:
            print(f"\nSamples with at least one correct model: {n_correct_samples}/{n_samples} ({n_correct_samples/n_samples*100:.1f}%)")
        
        # Filter meta-features and labels to only use these samples
        X_meta_correct = X_meta_train[correct_samples_mask]
        correct_sample_indices = np.where(correct_samples_mask)[0]
        
        # Create performance predictors (predict how well a model will perform)
        self.performance_predictors_ = {}
        
        for name in self.classifier_names_:
            # Create a regressor to predict this model's performance
            performance_predictor = self._create_meta_learner(regression=True)
            
            # Target is the correctness (0 or 1) weighted by confidence
            y_perf = np.zeros(n_correct_samples)
            
            for i, idx in enumerate(correct_sample_indices):
                true_label = all_val_labels[idx]
                pred = all_val_preds[name][idx]
                proba = all_val_probas[name][idx]
                
                # If prediction is correct, score is the confidence
                # If prediction is wrong, score is negative confidence
                confidence = proba[np.where(self.classes_ == pred)[0][0]]
                
                if pred == true_label:
                    y_perf[i] = confidence
                else:
                    y_perf[i] = -confidence
            
            # Train the regressor only on samples with at least one correct model
            performance_predictor.fit(X_meta_correct, y_perf)
            
            self.performance_predictors_[name] = performance_predictor
        
        # Calculate model selection accuracy
        predicted_best_models = []
        for i in range(n_samples):
            # Skip examples where no model is correct
            if not correct_samples_mask[i]:
                predicted_best_models.append(None)
                continue
                
            # Step 1: Identify models predicted to be correct
            correct_models = []
            if self.use_correctness_classifier:
                for name in self.classifier_names_:
                    # Predict if this model will be correct
                    correctness_prob = self.correctness_classifiers_[name].predict_proba(X_meta_train[i:i+1])[0][1]
                    if correctness_prob >= self.confidence_threshold:
                        correct_models.append(name)
            
            # If no models are predicted to be correct or we're not using correctness classifier,
            # consider all models
            if not correct_models or not self.use_correctness_classifier:
                correct_models = self.classifier_names_
            
            # Step 2: Among potentially correct models, choose the one with highest predicted performance
            model_perfs = {}
            for name in correct_models:
                perf = self.performance_predictors_[name].predict(X_meta_train[i:i+1])[0]
                model_perfs[name] = perf
            
            best_model = max(model_perfs, key=model_perfs.get) if model_perfs else self.classifier_names_[0]
            predicted_best_models.append(best_model)
        
        # Determine actual best models for comparison
        actual_best_models = []
        for i in range(n_samples):
            # Get models that correctly classify this sample
            correct_models = []
            model_scores = {}
            
            for name in self.classifier_names_:
                if all_val_correctness[name][i] == 1:
                    correct_models.append(name)
                    # Score is the prediction confidence
                    pred = all_val_preds[name][i]
                    proba = all_val_probas[name][i]
                    confidence = proba[np.where(self.classes_ == pred)[0][0]]
                    model_scores[name] = confidence
            
            if correct_models:
                # If multiple models are correct, choose the one with highest confidence
                best_model = max(model_scores, key=model_scores.get)
            else:
                # If no model is correct, there's no "best" model
                best_model = None
            
            actual_best_models.append(best_model)
        
        # Calculate model selection accuracy (only for samples where at least one model is correct)
        correct_comparisons = 0
        total_comparisons = 0
        
        for i in range(n_samples):
            if actual_best_models[i] is not None:
                total_comparisons += 1
                if predicted_best_models[i] == actual_best_models[i]:
                    correct_comparisons += 1
        
        model_selection_accuracy = correct_comparisons / total_comparisons if total_comparisons > 0 else 0
        
        logging.info(f"Model Selection Accuracy (on samples with at least one correct model): {model_selection_accuracy:.4f}")
        
        # Print overall model performance
        if self.verbose:
            print("\n=== Base Classifier Performance ===")
            print("-" * 40)
            for name in self.classifier_names_:
                print(f"{name:<15} | {base_performances[name]:.4f}")
            print("-" * 40)
            print(f"Average {self.metric}: {np.mean(list(base_performances.values())):.4f}")
            print(f"\nModel Selection Accuracy: {model_selection_accuracy:.4f}")
        
        # Train the base classifiers on the full dataset for final predictions
        for name in self.classifier_names_:
            clf = self.base_classifiers_[name]
            clf.fit(X_scaled, y)
        
        # Store dimensions for validation
        self.n_features_in_ = X.shape[1]
        self.n_meta_features_ = n_meta_features
        self.n_classes_ = n_classes
        
        return self
    
    def _create_meta_features(self, X_scaled):
        """Create meta-features for prediction time."""
        n_samples = X_scaled.shape[0]
        meta_features = np.zeros((n_samples, self.n_meta_features_))
        
        # Get cluster assignments
        clusters = self.kmeans_.predict(X_scaled)
        
        for i, name in enumerate(self.classifier_names_):
            # Get the classifier
            clf = self.base_classifiers_[name]
            
            # Make predictions
            proba = clf.predict_proba(X_scaled)
            
            # Calculate starting index for this classifier's meta-features
            start_idx = i * (self.n_classes_ + 4)
            
            # Store probabilities as meta-features
            meta_features[:, start_idx:start_idx+self.n_classes_] = proba
            
            # Calculate and store confidence features
            confidence_features = self._calculate_confidence_features(proba)
            meta_features[:, start_idx+self.n_classes_:start_idx+self.n_classes_+4] = confidence_features
        
        # Add cluster indicators and distance to centroid
        base_meta_features = len(self.classifier_names_) * (self.n_classes_ + 4)
        
        for i in range(n_samples):
            # Add one-hot encoding of cluster
            cluster_id = clusters[i]
            cluster_feature_start = base_meta_features
            meta_features[i, cluster_feature_start + cluster_id] = 1
            
            # Add distance to assigned centroid
            centroid = self.kmeans_.cluster_centers_[cluster_id]
            distance = np.linalg.norm(X_scaled[i] - centroid)
            meta_features[i, base_meta_features + self.n_clusters] = distance
        
        return meta_features
    
    def predict(self, X):
        """Predict using the advanced meta-model approach."""
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Validate dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but AdvancedModelSelectorClassifier "
                           f"was trained with {self.n_features_in_} features.")
        
        # Scale features
        X_scaled = self.scaler_.transform(X)
        
        # Create meta-features
        X_meta = self._create_meta_features(X_scaled)
        
        # For each instance, predict using the best model
        predictions = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            # Step 1: Identify models predicted to be correct
            correct_models = []
            if self.use_correctness_classifier:
                for name in self.classifier_names_:
                    # Predict if this model will be correct
                    correctness_prob = self.correctness_classifiers_[name].predict_proba(X_meta[i:i+1])[0][1]
                    if correctness_prob >= self.confidence_threshold:
                        correct_models.append(name)
            
            # If no models are predicted to be correct or we're not using correctness classifier,
            # consider all models
            if not correct_models or not self.use_correctness_classifier:
                correct_models = self.classifier_names_
            
            # Step 2: Among potentially correct models, choose the one with highest predicted performance
            model_perfs = {}
            model_preds = {}
            
            for name in correct_models:
                # Get predicted performance
                perf = self.performance_predictors_[name].predict(X_meta[i:i+1])[0]
                model_perfs[name] = perf
                
                # Get the model's prediction
                clf = self.base_classifiers_[name]
                model_preds[name] = clf.predict(X_scaled[i:i+1])[0]
            
            # Use the model with highest predicted performance
            best_model = max(model_perfs, key=model_perfs.get) if model_perfs else self.classifier_names_[0]
            predictions[i] = model_preds[best_model]
        
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Scale features
        X_scaled = self.scaler_.transform(X)
        
        # Create meta-features
        X_meta = self._create_meta_features(X_scaled)
        
        # Initialize probability matrix
        proba = np.zeros((X.shape[0], len(self.classes_)))
        
        # For each instance, get probabilities from the best model
        for i in range(X.shape[0]):
            # Step 1: Identify models predicted to be correct
            correct_models = []
            if self.use_correctness_classifier:
                for name in self.classifier_names_:
                    # Predict if this model will be correct
                    correctness_prob = self.correctness_classifiers_[name].predict_proba(X_meta[i:i+1])[0][1]
                    if correctness_prob >= self.confidence_threshold:
                        correct_models.append(name)
            
            # If no models are predicted to be correct or we're not using correctness classifier,
            # consider all models
            if not correct_models or not self.use_correctness_classifier:
                correct_models = self.classifier_names_
            
            # Step 2: Among potentially correct models, choose the one with highest predicted performance
            model_perfs = {}
            model_class_probs = {}
            
            for name in correct_models:
                # Get predicted performance
                perf = self.performance_predictors_[name].predict(X_meta[i:i+1])[0]
                model_perfs[name] = perf
                
                # Get class probabilities from this model
                clf = self.base_classifiers_[name]
                model_class_probs[name] = clf.predict_proba(X_scaled[i:i+1])[0]
            
            # Use the model with highest predicted performance
            best_model = max(model_perfs, key=model_perfs.get) if model_perfs else self.classifier_names_[0]
            proba[i] = model_class_probs[best_model]
        
        return proba
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "meta_learner": self.meta_learner,
            "verbose": self.verbose,
            "metric": self.metric,
            "n_clusters": self.n_clusters,
            "use_feature_weights": self.use_feature_weights,
            "use_correctness_classifier": self.use_correctness_classifier,
            "confidence_threshold": self.confidence_threshold
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self, X, y):
        """Returns the performance on the given test data and labels."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        return self._evaluate_classifier(y, y_pred, y_proba)