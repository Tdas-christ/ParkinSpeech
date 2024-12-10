import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
from utils.feature_extraction import extract_features
import pickle
from xgboost import XGBClassifier  # For XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def classification_model(healthy_dir, parkinson_dir, output_model_path):
    """
    Train a classification model using extracted features and save it as a pickle file.

    Parameters:
        healthy_dir (str): Path to the directory containing healthy patients' audio files.
        parkinson_dir (str): Path to the directory containing Parkinson's patients' audio files.
        output_model_path (str): Path to save the trained model as a pickle file.
    """
    feature_matrix = []
    labels = []

    # Extract features for healthy patients
    print("Extracting features for healthy patients...")
    for file_name in os.listdir(healthy_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(healthy_dir, file_name)
            features, _, _ = extract_features(file_path)
            feature_matrix.append(features)
            labels.append(0)  # Label for Healthy

    # Extract features for Parkinson's patients
    print("Extracting features for Parkinson's patients...")
    for file_name in os.listdir(parkinson_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(parkinson_dir, file_name)
            features, _, _ = extract_features(file_path)
            feature_matrix.append(features)
            labels.append(1)  # Label for Parkinson's

    # Convert to NumPy arrays
    X = np.array(feature_matrix)
    y = np.array(labels)

    # Split dataset into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    print("Training XGBoost Classifier...")
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the trained model as a pickle file
    print(f"Saving the model to {output_model_path}...")
    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)

    print("Model training and saving completed successfully!")

# Set paths for healthy and Parkinson's datasets
healthy_dir = 'HC_AH'
parkinson_dir = 'PD_AH'
output_model_path = './models/classification_model.pkl'

# Train and save the classification model
classification_model(healthy_dir, parkinson_dir, output_model_path)
