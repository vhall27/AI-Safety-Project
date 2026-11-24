# 
# Vanesska Hall and Jaspreet Sahota
# CS 120
# 
# This file generates distribution shifts and evaluates models.
#
# --------------------------------------------------------------

import numpy as np
from preprocess_data import X_val, y_val, scaler
from train_models import mlp_models, random_forest_models
from model_pairs import mlp_pairs, forest_pairs
from sklearn.metrics import accuracy_score

def gaussian_noise(X, severity, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(
        loc=0,
        scale=severity * scaler.scale_,
        size=X.shape
    )
    return X + noise

def count_underspecified_pairs(pairs, accuracies, threshold=0.05):
    num_pairs = 0
    for i, j in pairs:
        gap = abs(accuracies[i] - accuracies[j])
        if gap >= threshold:
            num_pairs += 1
    return num_pairs

X_shifted = gaussian_noise(X_val, severity=0.5)
X_shifted_scaled = scaler.transform(X_shifted)

# Parameter that controls the magnitude of Gaussian noise introduced into the 
# input features to simulate a distribution shift.
severities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# Varying severity to see how the models perform under different distribution shifts
for severity in severities:
    X_shifted = gaussian_noise(X_val, severity)
    X_shifted_scaled = scaler.transform(X_shifted)

    accuracies = []

    for model in mlp_models:
        predictions = model.predict(X_shifted_scaled)
        accuracy = accuracy_score(y_val, predictions)
        accuracies.append(accuracy)

    forest_accuracies = []

    for model in random_forest_models:
        predictions = model.predict(X_shifted)
        accuracy = accuracy_score(y_val, predictions)
        forest_accuracies.append(accuracy)


    print(f"Severity: {severity}")
    mlp_count = count_underspecified_pairs(mlp_pairs, accuracies)
    rf_count = count_underspecified_pairs(forest_pairs, forest_accuracies)

    print(f"Underspecified MLP pairs: {mlp_count}/{len(mlp_pairs)}")
    print(f"Underspecified RF pairs: {rf_count}/{len(forest_pairs)}")

    print(f"MLP percentage: {mlp_count / len(mlp_pairs)}")
    print(f"RF percentage: {rf_count / len(forest_pairs)}")