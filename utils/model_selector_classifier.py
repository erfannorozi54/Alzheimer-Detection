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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
warnings.filterwarnings('ignore')

class ModelSelectorClassifier(BaseEstimator, ClassifierMixin):
    """
    A completely redesigned classifier that uses a simpler and more effective approach:
    1. Uses a Random Forest as the meta-learner instead of a neural network
    2. Uses the probabilities from base classifiers as meta-features
    3. Implements proper cross-validation without information leakage
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        meta_learner='rf',  # 'rf' or 'lr'
        verbose: bool = True,
        use_proba: bool = True
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = meta_learner
        self.verbose = verbose
        self.use_proba = use_proba
        
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

    def fit(self, X, y):
        """
        Fit the classifier using a stack-based approach.
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

        if self.use_proba:
            # For probability-based stacking: store probabilities for each class
            n_meta_features = len(self.classifier_names_) * n_classes
            X_meta_train = np.zeros((n_samples, n_meta_features))
        else:
            # For prediction-based stacking: store just the predictions
            n_meta_features = len(self.classifier_names_)
            X_meta_train = np.zeros((n_samples, n_meta_features))

        # For tracking individual model performance
        base_accuracies = {name: 0.0 for name in self.classifier_names_}
        fold_accuracies = []

        # Store all base classifier predictions and true labels for each fold
        all_val_preds = []
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

            # Store predictions for this fold
            fold_preds = {}

            # Train each base classifier and collect meta-features
            for i, name in enumerate(self.classifier_names_):
                # Get the classifier
                clf = self.base_classifiers_[name]

                # Train on the fold's training data
                clf.fit(X_train_fold, y_train_fold)

                # Make predictions on validation data
                y_val_pred = clf.predict(X_val_fold)
                fold_preds[name] = y_val_pred

                # Calculate accuracy for this fold
                acc = accuracy_score(y_val_fold, y_val_pred)
                base_accuracies[name] += acc / self.n_folds

                if self.verbose:
                    print(f"{name} Accuracy: {acc:.4f}")

                if self.use_proba:
                    # Get prediction probabilities
                    proba = clf.predict_proba(X_val_fold)

                    # Store probabilities as meta-features
                    for j in range(n_classes):
                        col_idx = i * n_classes + j
                        X_meta_train[val_idx, col_idx] = proba[:, j]
                else:
                    # Store predictions as meta-features (one-hot encoded)
                    X_meta_train[val_idx, i] = y_val_pred

            # Calculate average accuracy for this fold
            fold_accuracies.append(np.mean([base_accuracies[name] for name in self.classifier_names_]))
            all_val_preds.append(fold_preds)
            all_val_labels.append(y_val_fold)

        # --- Model Selection Accuracy Calculation ---
        all_val_preds_combined = {}
        for name in self.classifier_names_:
            all_val_preds_combined[name] = np.concatenate([fold_preds[name] for fold_preds in all_val_preds])
        all_val_labels_combined = np.concatenate(all_val_labels)

        # Determine the actually best model for each sample
        actual_best_models = []
        for i in range(n_samples):
            sample_accuracies = {}
            for name in self.classifier_names_:
                sample_accuracies[name] = accuracy_score(all_val_labels_combined[i:i+1], [all_val_preds_combined[name][i]])
            best_model = max(sample_accuracies, key=sample_accuracies.get)
            actual_best_models.append(best_model)

        # Determine predicted best model for each sample
        if self.use_proba:
            predicted_best_model_indices = np.argmax(X_meta_train, axis=1) // n_classes
        else:
            predicted_best_model_indices = np.argmax(X_meta_train, axis=1)
        predicted_best_models = [self.classifier_names_[i] for i in predicted_best_model_indices]

        # Calculate model selection accuracy
        model_selection_accuracy = accuracy_score(actual_best_models, predicted_best_models)
        logging.info(f"Actual Best Models: {actual_best_models}")
        logging.info(f"Predicted Best Models: {predicted_best_models}")
        logging.info(f"Model Selection Accuracy: {model_selection_accuracy:.4f}")

        # Print overall model performance
        if self.verbose:
            print("\n=== Base Classifier Performance ===")
            print("-" * 40)
            for name in self.classifier_names_:
                print(f"{name:<15} | {base_accuracies[name]:.4f}")
            print("-" * 40)
            print(f"Average Accuracy: {np.mean(list(base_accuracies.values())):.4f}")

        # Create and train the meta-learner
        self.meta_classifier_ = self._create_meta_learner()
        self.meta_classifier_.fit(X_meta_train, y)

        if self.verbose:
            # Evaluate meta-learner on training data
            meta_pred = self.meta_classifier_.predict(X_meta_train)
            meta_acc = accuracy_score(y, meta_pred)
            print(f"\nMeta-learner accuracy on training data: {meta_acc:.4f}")

            # Find the best base classifier for comparison
            best_clf = max(base_accuracies.items(), key=lambda x: x[1])
            print(f"Best individual classifier ({best_clf[0]}): {best_clf[1]:.4f}")

            # Visualize feature importances if meta-learner is Random Forest
            if self.meta_learner == 'rf':
                importances = self.meta_classifier_.feature_importances_
                indices = np.argsort(importances)[::-1]

                print("\nTop 10 most important meta-features:")
                for i in range(min(10, len(importances))):
                    idx = indices[i]
                    if self.use_proba:
                        clf_idx = idx // n_classes
                        class_idx = idx % n_classes
                        feature_name = f"{self.classifier_names_[clf_idx]} (Class {self.classes_[class_idx]})"
                    else:
                        feature_name = self.classifier_names_[idx]
                    print(f"{feature_name:<20} | {importances[idx]:.4f}")

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
        if self.use_proba:
            # For probability-based meta-features
            meta_features = np.zeros((X_scaled.shape[0], self.n_meta_features_))
            
            for i, name in enumerate(self.classifier_names_):
                clf = self.base_classifiers_[name]
                proba = clf.predict_proba(X_scaled)
                
                # Store probabilities as meta-features
                for j in range(self.n_classes_):
                    col_idx = i * self.n_classes_ + j
                    meta_features[:, col_idx] = proba[:, j]
        else:
            # For prediction-based meta-features
            meta_features = np.zeros((X_scaled.shape[0], len(self.classifier_names_)))
            
            for i, name in enumerate(self.classifier_names_):
                clf = self.base_classifiers_[name]
                meta_features[:, i] = clf.predict(X_scaled)
        
        return meta_features

    def predict(self, X):
        """Predict using the stacked classifier."""
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Validate dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but ModelSelectorClassifier "
                           f"was trained with {self.n_features_in_} features.")
        
        # Scale features
        X_scaled = self.scaler_.transform(X)
        
        # Create meta-features
        X_meta = self._create_meta_features(X_scaled)
        
        # Make final predictions using the meta-learner
        return self.meta_classifier_.predict(X_meta)
    
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
        
        # Make probability predictions using the meta-learner
        return self.meta_classifier_.predict_proba(X_meta)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "meta_learner": self.meta_learner,
            "verbose": self.verbose,
            "use_proba": self.use_proba
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        return accuracy_score(y, self.predict(X))
