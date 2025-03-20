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
    An enhanced model selector with improved meta-learning capabilities:
    1. Uses more robust meta-features
    2. Implements repeated cross-validation for more meta-training data
    3. Enables synthetic data generation to increase training samples
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        cv_repeats: int = 5,  # Number of times to repeat cross-validation
        random_state: int = 42,
        meta_learner='rf',  # 'rf' or 'lr'
        verbose: bool = True,
        metric: str = 'accuracy',  # 'accuracy', 'f1', 'auc', 'log_loss'
        n_clusters: int = 5,  # Number of feature space clusters
        use_correctness_classifier: bool = True,  # Use two-stage approach
        confidence_threshold: float = 0.6,  # Threshold for considering a model "confident"
        use_synthetic_data: bool = False,  # Whether to use synthetic data
        synthetic_multiplier: float = 0.5  # Proportion of synthetic data to generate
    ):
        self.n_folds = n_folds
        self.cv_repeats = cv_repeats
        self.random_state = random_state
        self.meta_learner = meta_learner
        self.verbose = verbose
        self.metric = metric
        self.n_clusters = n_clusters
        self.use_correctness_classifier = use_correctness_classifier
        self.confidence_threshold = confidence_threshold
        self.use_synthetic_data = use_synthetic_data
        self.synthetic_multiplier = synthetic_multiplier
        
    def _initialize_classifiers(self):
        """Initialize base classifiers with proper parameters."""
        self.base_classifiers_ = {
            "RF": RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=self.random_state),
            "LR": LogisticRegression(max_iter=5000, C=1.0, solver='saga', random_state=self.random_state), 
            "XGB": XGBClassifier(n_estimators=100, random_state=self.random_state),
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
                    max_depth=15,
                    min_samples_leaf=3,
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
                    max_depth=15,
                    min_samples_leaf=3,
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
        
        # Gini impurity: sum(p_i * (1 - p_i))
        gini = np.sum(probas * (1 - probas), axis=1)
        
        return np.column_stack([margin, entropy, confidence, dispersion, gini])

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
    
    def _generate_synthetic_samples(self, X, y):
        """Generate synthetic samples by adding noise to existing samples."""
        n_synthetic = int(X.shape[0] * self.synthetic_multiplier)
        
        # Sample indices with replacement
        indices = np.random.RandomState(self.random_state).choice(
            X.shape[0], size=n_synthetic, replace=True
        )
        
        # Create synthetic samples by adding small Gaussian noise
        X_synthetic = X[indices].copy()
        y_synthetic = y[indices].copy()
        
        # Add noise to features (scaled by feature standard deviation)
        feature_stds = np.std(X, axis=0)
        noise = np.random.RandomState(self.random_state).normal(
            0, 0.1, size=X_synthetic.shape
        ) * feature_stds
        
        X_synthetic += noise
        
        return X_synthetic, y_synthetic

    def fit(self, X, y):
        """
        Fit the classifier using an enhanced meta-learning approach with more training data.
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
        
        # Generate synthetic data if enabled
        if self.use_synthetic_data:
            X_synthetic, y_synthetic = self._generate_synthetic_samples(X_scaled, y)
            X_aug = np.vstack([X_scaled, X_synthetic])
            y_aug = np.concatenate([y, y_synthetic])
            
            if self.verbose:
                print(f"Generated {X_synthetic.shape[0]} synthetic samples")
                print(f"Total training samples: {X_aug.shape[0]}")
        else:
            X_aug = X_scaled
            y_aug = y
        
        # Segment the feature space (for regional specialization)
        self.kmeans_ = self._segment_feature_space(X_aug)
        
        # Initialize storage for meta-learning
        all_meta_features = []
        all_meta_correctness = {name: [] for name in self.classifier_names_}
        all_meta_perf_scores = {name: [] for name in self.classifier_names_}
        
        # Each base classifier contributes:
        # - Class probabilities
        # - Confidence metrics (margin, entropy, confidence, dispersion, gini)
        base_meta_features = len(self.classifier_names_) * (n_classes + 5)
        
        # Add cluster indicators
        extra_features = self.n_clusters
        
        # Total number of meta-features
        n_meta_features = base_meta_features + extra_features
        
        # Repeat cross-validation multiple times to generate more meta-learning data
        for repeat in range(self.cv_repeats):
            if self.verbose:
                print(f"\nRepeat {repeat+1}/{self.cv_repeats}")
                print("=" * 40)
            
            # Create stratified K-fold with different random seed for each repeat
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                               random_state=self.random_state + repeat)
            
            # For each fold in cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_aug, y_aug)):
                if self.verbose:
                    print(f"\nFold {fold_idx+1}/{self.n_folds}")
                    print("-" * 40)
                
                X_train_fold = X_aug[train_idx]
                y_train_fold = y_aug[train_idx]
                X_val_fold = X_aug[val_idx]
                y_val_fold = y_aug[val_idx]
                
                # Get cluster assignments for validation samples
                val_clusters = self.kmeans_.predict(X_val_fold)
                
                # Create meta-features for this fold
                X_meta_fold = np.zeros((len(val_idx), n_meta_features))
                
                # For tracking model performance on this fold
                fold_correctness = {name: [] for name in self.classifier_names_}
                fold_perf_scores = {name: [] for name in self.classifier_names_}
                
                # Train each base classifier and collect meta-features
                for i, name in enumerate(self.classifier_names_):
                    # Get the classifier
                    clf = self.base_classifiers_[name]
                    
                    # Train on the fold's training data
                    clf.fit(X_train_fold, y_train_fold)
                    
                    # Make predictions on validation data
                    y_val_pred = clf.predict(X_val_fold)
                    y_val_proba = clf.predict_proba(X_val_fold)
                    
                    # Check correctness
                    correctness = (y_val_pred == y_val_fold).astype(int)
                    fold_correctness[name] = correctness
                    
                    # Calculate performance scores (confidence-weighted correctness)
                    perf_scores = np.zeros(len(val_idx))
                    for j in range(len(val_idx)):
                        pred = y_val_pred[j]
                        true = y_val_fold[j]
                        proba = y_val_proba[j]
                        confidence = proba[np.where(self.classes_ == pred)[0][0]]
                        
                        if pred == true:
                            perf_scores[j] = confidence
                        else:
                            perf_scores[j] = -confidence
                    
                    fold_perf_scores[name] = perf_scores
                    
                    # Calculate overall performance for this fold
                    performance = self._evaluate_classifier(y_val_fold, y_val_pred, y_val_proba)
                    
                    if self.verbose:
                        print(f"{name} {self.metric}: {performance:.4f}")
                    
                    # Calculate starting index for this classifier's meta-features
                    start_idx = i * (n_classes + 5)
                    
                    # Store probabilities as meta-features
                    X_meta_fold[:, start_idx:start_idx+n_classes] = y_val_proba
                    
                    # Calculate and store confidence features
                    confidence_features = self._calculate_confidence_features(y_val_proba)
                    X_meta_fold[:, start_idx+n_classes:start_idx+n_classes+5] = confidence_features
                
                # Add cluster indicators
                cluster_feature_start = base_meta_features
                for j in range(len(val_idx)):
                    cluster_id = val_clusters[j]
                    X_meta_fold[j, cluster_feature_start + cluster_id] = 1
                
                # Store meta-features and targets for meta-learning
                all_meta_features.append(X_meta_fold)
                for name in self.classifier_names_:
                    all_meta_correctness[name].append(fold_correctness[name])
                    all_meta_perf_scores[name].append(fold_perf_scores[name])
        
        # Combine all meta-features and targets
        X_meta_all = np.vstack(all_meta_features)
        for name in self.classifier_names_:
            all_meta_correctness[name] = np.concatenate(all_meta_correctness[name])
            all_meta_perf_scores[name] = np.concatenate(all_meta_perf_scores[name])
        
        # Only use samples where at least one model is correct
        correct_samples_mask = np.zeros(X_meta_all.shape[0], dtype=bool)
        for name in self.classifier_names_:
            correct_samples_mask = correct_samples_mask | (all_meta_correctness[name] == 1)
        
        # Filter meta-features to only use samples with at least one correct model
        X_meta_correct = X_meta_all[correct_samples_mask]
        
        # Filter correctness and performance scores
        correctness_filtered = {}
        perf_scores_filtered = {}
        for name in self.classifier_names_:
            correctness_filtered[name] = all_meta_correctness[name][correct_samples_mask]
            perf_scores_filtered[name] = all_meta_perf_scores[name][correct_samples_mask]
        
        # Count how many samples have at least one correct model
        n_correct_samples = correct_samples_mask.sum()
        
        if self.verbose:
            print(f"\nSamples with at least one correct model: {n_correct_samples}/{X_meta_all.shape[0]} ({n_correct_samples/X_meta_all.shape[0]*100:.1f}%)")
        
        # Create correctness classifiers (predict if a model will be correct)
        if self.use_correctness_classifier:
            self.correctness_classifiers_ = {}
            
            for name in self.classifier_names_:
                # Create binary classifier to predict if this model will be correct
                correctness_clf = self._create_meta_learner(regression=False)
                
                # Train the classifier on all meta-features
                correctness_clf.fit(X_meta_all, all_meta_correctness[name])
                
                self.correctness_classifiers_[name] = correctness_clf
                
                if self.verbose:
                    # Evaluate classifier on training data
                    train_acc = correctness_clf.score(X_meta_all, all_meta_correctness[name])
                    print(f"Correctness classifier for {name}: {train_acc:.4f} accuracy")
        
        # Create performance predictors (predict how well a model will perform)
        self.performance_predictors_ = {}
        
        for name in self.classifier_names_:
            # Create a regressor to predict this model's performance
            performance_predictor = self._create_meta_learner(regression=True)
            
            # Train the regressor only on samples with at least one correct model
            performance_predictor.fit(X_meta_correct, perf_scores_filtered[name])
            
            self.performance_predictors_[name] = performance_predictor
            
            if self.verbose:
                # Evaluate regressor on training data
                train_r2 = performance_predictor.score(X_meta_correct, perf_scores_filtered[name])
                print(f"Performance predictor for {name}: {train_r2:.4f} R²")
        
        # Calculate model selection accuracy on meta-learning data
        predicted_best_models = []
        actual_best_models = []
        
        for i in range(X_meta_correct.shape[0]):
            # Step 1: Identify models predicted to be correct
            correct_models = []
            for name in self.classifier_names_:
                if correctness_filtered[name][i] == 1:
                    correct_models.append(name)
                    
            # Step 2: Find the model with highest performance score
            model_scores = {}
            for name in correct_models:
                model_scores[name] = perf_scores_filtered[name][i]
            
            actual_best_model = max(model_scores, key=model_scores.get)
            actual_best_models.append(actual_best_model)
            
            # Predict using our meta-learners
            # Step 1: Identify models predicted to be correct
            pred_correct_models = []
            if self.use_correctness_classifier:
                for name in self.classifier_names_:
                    # Predict if this model will be correct
                    correctness_prob = self.correctness_classifiers_[name].predict_proba(X_meta_correct[i:i+1])[0][1]
                    if correctness_prob >= self.confidence_threshold:
                        pred_correct_models.append(name)
            
            # If no models are predicted to be correct or we're not using correctness classifier,
            # consider all models
            if not pred_correct_models or not self.use_correctness_classifier:
                pred_correct_models = self.classifier_names_
            
            # Step 2: Among potentially correct models, choose one with highest predicted performance
            model_perfs = {}
            for name in pred_correct_models:
                perf = self.performance_predictors_[name].predict(X_meta_correct[i:i+1])[0]
                model_perfs[name] = perf
            
            best_model = max(model_perfs, key=model_perfs.get) if model_perfs else self.classifier_names_[0]
            predicted_best_models.append(best_model)
        
        # Calculate model selection accuracy
        model_selection_accuracy = np.mean(np.array(predicted_best_models) == np.array(actual_best_models))
        
        if self.verbose:
            print(f"\nModel Selection Accuracy: {model_selection_accuracy:.4f}")
            
        logging.info(f"Model Selection Accuracy: {model_selection_accuracy:.4f}")
        
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
            start_idx = i * (self.n_classes_ + 5)
            
            # Store probabilities as meta-features
            meta_features[:, start_idx:start_idx+self.n_classes_] = proba
            
            # Calculate and store confidence features
            confidence_features = self._calculate_confidence_features(proba)
            meta_features[:, start_idx+self.n_classes_:start_idx+self.n_classes_+5] = confidence_features
        
        # Add cluster indicators
        base_meta_features = len(self.classifier_names_) * (self.n_classes_ + 5)
        cluster_feature_start = base_meta_features
        
        for i in range(n_samples):
            # Add one-hot encoding of cluster
            cluster_id = clusters[i]
            meta_features[i, cluster_feature_start + cluster_id] = 1
        
        return meta_features
    
    def predict(self, X):
        """Predict using the advanced meta-model approach."""
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Validate dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but EnhancedModelSelectorClassifier "
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
            "cv_repeats": self.cv_repeats,
            "random_state": self.random_state,
            "meta_learner": self.meta_learner,
            "verbose": self.verbose,
            "metric": self.metric,
            "n_clusters": self.n_clusters,
            "use_correctness_classifier": self.use_correctness_classifier,
            "confidence_threshold": self.confidence_threshold,
            "use_synthetic_data": self.use_synthetic_data,
            "synthetic_multiplier": self.synthetic_multiplier
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