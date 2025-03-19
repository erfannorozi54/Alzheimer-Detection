import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging
logging.basicConfig(filename='model_selection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
warnings.filterwarnings('ignore')

class ModelSelectorClassifier(BaseEstimator, ClassifierMixin):
    """
    An improved model selector classifier that:
    1. Uses multiple metrics for determining the best model
    2. Employs a better meta-learning approach
    3. Incorporates model confidence as features
    4. Uses a more robust cross-validation strategy
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        meta_learner='rf',  # 'rf' or 'lr'
        verbose: bool = True,
        metric: str = 'f1'  # 'accuracy', 'f1', 'auc', 'log_loss'
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = meta_learner
        self.verbose = verbose
        self.metric = metric
        
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
    
    def _create_meta_learner(self):
        """Create a meta-learner based on the specified type."""
        if self.meta_learner == 'rf':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=self.random_state
            )
        elif self.meta_learner == 'lr':
            return LogisticRegression(
                max_iter=5000,
                C=1.0,
                solver='saga',
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown meta_learner: {self.meta_learner}")

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
        
        return np.column_stack([margin, entropy, confidence])

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

    def fit(self, X, y):
        """
        Fit the classifier using an improved meta-learning approach.
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
        
        # Use stratified K-fold to maintain class distribution
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Initialize storage for meta-learner training data
        n_samples = X.shape[0]
        
        # Each base classifier contributes:
        # - Class probabilities
        # - Confidence metrics (margin, entropy, confidence)
        n_meta_features = len(self.classifier_names_) * (n_classes + 3)
        X_meta_train = np.zeros((n_samples, n_meta_features))
        
        # Store arrays for computing regional performance
        region_performance = []
        
        # For tracking individual model performance
        base_performances = {name: 0.0 for name in self.classifier_names_}
        fold_performances = []
        
        # Store all predictions and probabilities
        all_val_preds = {}
        all_val_probas = {}
        all_val_regions = {}
        for name in self.classifier_names_:
            all_val_preds[name] = []
            all_val_probas[name] = []
        all_val_labels = []
        
        # For each fold in cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            if self.verbose:
                print(f"\nFold {fold_idx+1}/{self.n_folds}")
                print("-" * 40)
            
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_val_fold = y[val_idx]
            
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
                
                # Calculate performance for this fold
                performance = self._evaluate_classifier(y_val_fold, y_val_pred, y_val_proba)
                base_performances[name] += performance / self.n_folds
                
                if self.verbose:
                    print(f"{name} {self.metric}: {performance:.4f}")
                
                # Calculate starting index for this classifier's meta-features
                start_idx = i * (n_classes + 3)
                
                # Store probabilities as meta-features
                X_meta_train[val_idx, start_idx:start_idx+n_classes] = y_val_proba
                
                # Calculate and store confidence features
                confidence_features = self._calculate_confidence_features(y_val_proba)
                X_meta_train[val_idx, start_idx+n_classes:start_idx+n_classes+3] = confidence_features
            
            # Store validation labels
            all_val_labels.append(y_val_fold)
            
            # Define "regions" in the feature space (for regional performance analysis)
            # Here we use a simple clustering approach based on validation fold
            all_val_regions[fold_idx] = val_idx
            
            # Calculate fold performance
            fold_performances.append(np.mean([base_performances[name] for name in self.classifier_names_]))
        
        # Concatenate all validation predictions and labels
        for name in self.classifier_names_:
            all_val_preds[name] = np.concatenate(all_val_preds[name])
            all_val_probas[name] = np.vstack(all_val_probas[name])
        all_val_labels = np.concatenate(all_val_labels)
        
        # Determine the actually best model for each sample using a more robust approach
        # Instead of binary 0/1 accuracy, we'll use prediction confidence when correct,
        # and negative confidence when incorrect
        actual_best_models = []
        best_model_scores = []
        
        for i in range(n_samples):
            true_label = all_val_labels[i]
            model_scores = {}
            
            for name in self.classifier_names_:
                pred = all_val_preds[name][i]
                proba = all_val_probas[name][i]
                confidence = proba[np.where(self.classes_ == pred)[0][0]]
                
                # If prediction is correct, score is positive confidence
                # If prediction is wrong, score is negative confidence
                if pred == true_label:
                    score = confidence
                else:
                    score = -confidence
                
                model_scores[name] = score
            
            # The best model has the highest score
            best_model = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model]
            
            actual_best_models.append(best_model)
            best_model_scores.append(best_score)
        
        # Now, we define the target for our meta-learner:
        # Instead of trying to predict the best model directly,
        # we'll predict the performance score for each model and select the highest
        y_meta = np.zeros((n_samples, len(self.classifier_names_)))
        
        for i, name in enumerate(self.classifier_names_):
            for j in range(n_samples):
                true_label = all_val_labels[j]
                pred = all_val_preds[name][j]
                proba = all_val_probas[name][j]
                confidence = proba[np.where(self.classes_ == pred)[0][0]]
                
                # Same scoring as above
                if pred == true_label:
                    score = confidence
                else:
                    score = -confidence
                
                y_meta[j, i] = score
        
        # Now, let's create regression-based meta-learners instead of classifiers
        # This avoids issues with binary classification when a model is never the best
        self.meta_regressors_ = {}
        for i, name in enumerate(self.classifier_names_):
            if self.meta_learner == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                regressor = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state
                )
            else:
                from sklearn.linear_model import Ridge
                regressor = Ridge(
                    alpha=1.0,
                    random_state=self.random_state
                )
            
            # Get performance scores for this model
            y_scores = np.zeros(n_samples)
            for j in range(n_samples):
                true_label = all_val_labels[j]
                pred = all_val_preds[name][j]
                proba = all_val_probas[name][j]
                confidence = proba[np.where(self.classes_ == pred)[0][0]]
                
                # Same scoring as above - positive when correct, negative when wrong
                if pred == true_label:
                    score = confidence
                else:
                    score = -confidence
                
                y_scores[j] = score
            
            # Train a regressor to predict the performance score
            regressor.fit(X_meta_train, y_scores)
            self.meta_regressors_[name] = regressor
        
        # Calculate model selection accuracy
        predicted_best_models = []
        for i in range(n_samples):
            model_scores = {}
            for name in self.classifier_names_:
                # Get predicted performance score
                score = self.meta_regressors_[name].predict(X_meta_train[i:i+1])[0]
                model_scores[name] = score
            
            best_model = max(model_scores, key=model_scores.get)
            predicted_best_models.append(best_model)
        
        model_selection_accuracy = accuracy_score(actual_best_models, predicted_best_models)
        logging.info(f"Actual Best Models: {actual_best_models[:20]}...")  # Log first 20 examples
        logging.info(f"Predicted Best Models: {predicted_best_models[:20]}...")
        logging.info(f"Model Selection Accuracy: {model_selection_accuracy:.4f}")
        
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
        meta_features = np.zeros((X_scaled.shape[0], self.n_meta_features_))
        
        for i, name in enumerate(self.classifier_names_):
            # Get the classifier
            clf = self.base_classifiers_[name]
            
            # Make predictions
            proba = clf.predict_proba(X_scaled)
            
            # Calculate starting index for this classifier's meta-features
            start_idx = i * (self.n_classes_ + 3)
            
            # Store probabilities as meta-features
            meta_features[:, start_idx:start_idx+self.n_classes_] = proba
            
            # Calculate and store confidence features
            confidence_features = self._calculate_confidence_features(proba)
            meta_features[:, start_idx+self.n_classes_:start_idx+self.n_classes_+3] = confidence_features
        
        return meta_features
    
    def predict(self, X):
        """Predict using the meta-model approach."""
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Validate dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but ImprovedModelSelectorClassifier "
                           f"was trained with {self.n_features_in_} features.")
        
        # Scale features
        X_scaled = self.scaler_.transform(X)
        
        # Create meta-features
        X_meta = self._create_meta_features(X_scaled)
        
        # For each instance, predict using the model with highest predicted performance
        predictions = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            model_scores = {}
            model_preds = {}
            
            # Get the predicted performance score for each model
            for name in self.classifier_names_:
                score = self.meta_regressors_[name].predict(X_meta[i:i+1])[0]
                model_scores[name] = score
                
                # Also get each model's prediction
                clf = self.base_classifiers_[name]
                model_preds[name] = clf.predict(X_scaled[i:i+1])[0]
            
            # Use the model with highest predicted performance score
            best_model = max(model_scores, key=model_scores.get)
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
            model_scores = {}
            model_class_probs = {}
            
            # Get predicted performance score for each model
            for name in self.classifier_names_:
                score = self.meta_regressors_[name].predict(X_meta[i:i+1])[0]
                model_scores[name] = score
                
                # Get class probabilities from each base model
                clf = self.base_classifiers_[name]
                model_class_probs[name] = clf.predict_proba(X_scaled[i:i+1])[0]
            
            # Use the model with highest predicted performance score
            best_model = max(model_scores, key=model_scores.get)
            proba[i] = model_class_probs[best_model]
        
        return proba
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "meta_learner": self.meta_learner,
            "verbose": self.verbose,
            "metric": self.metric
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