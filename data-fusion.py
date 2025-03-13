import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from ucimlrepo import fetch_ucirepo
# from tensorflow.keras.layers import LSTM, Lambda, Layer, Dropout


# Step 1: Fetch the DARWIN dataset
darwin = fetch_ucirepo(id=732)

# Access data as pandas DataFrames
X = darwin.data.features
y = darwin.data.targets

# Print metadata and variable information for reference
print(darwin.metadata)
print(darwin.variables)

# Step 2: Preprocess the dataset
# Assuming the target variable needs binary classification
y = y.apply(lambda x: 1 if x == 'P' else 0)  # Adjust if 'P' represents Patients and 'H' for Healthy

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define feature columns for DeepFM
# Assuming you identify sparse and dense features based on dataset characteristics
sparse_features = [col for col in X.columns if X[col].dtype == 'object']
dense_features = [col for col in X.columns if col not in sparse_features]

# Create feature columns
feature_columns = [SparseFeat(feat, vocabulary_size=X[feat].nunique()) for feat in sparse_features] + \
                  [DenseFeat(feat, 1) for feat in dense_features]

# Prepare model input
train_model_input = {name: X_train_scaled[:, i] for i, name in enumerate(get_feature_names(feature_columns))}
test_model_input = {name: X_test_scaled[:, i] for i, name in enumerate(get_feature_names(feature_columns))}

# Step 6: Define and compile the DeepFM model
model = DeepFM(feature_columns, feature_columns, task='binary')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_model_input, y_train, batch_size=256, epochs=10, verbose=2, validation_split=0.2)

# Step 7: Evaluate the model
pred_ans = model.predict(test_model_input)
predictions = (pred_ans > 0.5).astype(int)
accuracy = accuracy_score(y_test, predictions)
print("DeepFM Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# Step 8: Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_predictions))
