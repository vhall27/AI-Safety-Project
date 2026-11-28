# 
# Vanesska Hall and Jaspreet Sahota
# CS 120
# 
# This file implements three evaluator access arrangements.
#
# ----------------------------------------------------------

from train_models import mlp_models, mlp_accuracies
from train_models import random_forest_models, random_forest_accuracies
from model_pairs import mlp_pairs, forest_pairs
from preprocess_data import X_val, y_val, scaler
from stress_tests import gaussian_noise, accuracy_score


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report


# This first Set of evaluators is for the MLP Models 

# Ground truth
X_shifted = gaussian_noise(X_val, severity=2.0)
X_shifted_scaled = scaler.transform(X_shifted)

shifted_accuracies = []

for model in mlp_models:
    predictions = model.predict(X_shifted_scaled)
    shifted_accuracies.append(accuracy_score(y_val, predictions))

# IID accuracies
X_val_scaled = scaler.transform(X_val)

iid_accuracies = []

for model in mlp_models:
    predictions = model.predict(X_val_scaled)
    iid_accuracies.append(accuracy_score(y_val, predictions))

# Creating the dataset for the evaluator
X_eval = []
y_eval = []

for i, j in mlp_pairs:
    acc1 = iid_accuracies[i]
    acc2 = iid_accuracies[j]

    X_eval.append([
        min(acc1, acc2),
        max(acc1, acc2),
        abs(acc1 - acc2)
    ])

    shifted_gap = abs(
        shifted_accuracies[i] - shifted_accuracies[j]
    )

    y_eval.append(int(shifted_gap >= 0.05))

# Splitting the results into test and train sets for the evaluator
train_indices, test_indices = train_test_split(
    np.arange(len(mlp_models)),
    test_size=0.30,
    random_state=42
)

X_eval_train = []
y_eval_train = []
X_eval_test = []
y_eval_test = []

for k, (i, j) in enumerate(mlp_pairs):

    if i in train_indices and j in train_indices:
        X_eval_train.append(X_eval[k])
        y_eval_train.append(y_eval[k])

    elif i in test_indices and j in test_indices:
        X_eval_test.append(X_eval[k])
        y_eval_test.append(y_eval[k])

evaluator1 = LogisticRegression(class_weight="balanced")

evaluator1.fit(X_eval_train, y_eval_train)

predictions = evaluator1.predict(X_eval_test)

probabilities1 = evaluator1.predict_proba(X_eval_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_eval_test, probabilities1))
print(classification_report(y_eval_test, predictions, zero_division=0))

print("Evaluator 1 accuracy:", accuracy_score(y_eval_test, predictions))

print("Predicted probabilities:", evaluator1.predict_proba(X_eval_test)[:5])

iid_probabilities = []

for model in mlp_models:
    probabilities = model.predict_proba(X_val_scaled)[:, 1]
    iid_probabilities.append(probabilities)

X_eval2 = []

for k, (i, j) in enumerate(mlp_pairs):
    probability_gap = np.mean(
        np.abs(iid_probabilities[i] - iid_probabilities[j])
    )

    X_eval2.append(X_eval[k] + [probability_gap])

X_eval2_train = []
X_eval2_test = []

for k, (i, j) in enumerate(mlp_pairs):
    if i in train_indices and j in train_indices:
        X_eval2_train.append(X_eval2[k])

    elif i in test_indices and j in test_indices:
        X_eval2_test.append(X_eval2[k])

eval_scaler = StandardScaler()

X_eval2_train_scaled = eval_scaler.fit_transform(X_eval2_train)
X_eval2_test_scaled = eval_scaler.transform(X_eval2_test)

from sklearn.metrics import roc_auc_score, classification_report

evaluator2 = LogisticRegression(class_weight="balanced")
evaluator2.fit(X_eval2_train_scaled, y_eval_train)

predictions2 = evaluator2.predict(X_eval2_test_scaled)
probabilities2 = evaluator2.predict_proba(X_eval2_test_scaled)[:, 1]

print("Evaluator 2 accuracy:", accuracy_score(y_eval_test, predictions2))
print("ROC-AUC:", roc_auc_score(y_eval_test, probabilities2))
print("Predicted probabilities:", probabilities2[:10])
print(classification_report(y_eval_test, predictions2, zero_division=0))


X_stress = gaussian_noise(X_val, severity=1.0, seed=43)
X_stress_scaled = scaler.transform(X_stress)

stress_probabilities = []

for model in mlp_models:
    probabilities = model.predict_proba(X_stress_scaled)[:, 1]
    stress_probabilities.append(probabilities)

X_eval3 = []

for k, (i, j) in enumerate(mlp_pairs):
    stress_gap = np.mean(
        np.abs(stress_probabilities[i] - stress_probabilities[j])
    )

    X_eval3.append(X_eval2[k] + [stress_gap])

df3 = pd.DataFrame(X_eval3, columns=[
    "Lower IID Accuracy",
    "Higher IID Accuracy",
    "IID Gap",
    "Probability Gap",
    "Stress Probability Gap"
])

df3["Underspecified"] = y_eval

print(df3.head(50).to_string(index=False))

X_eval3_train = []
X_eval3_test = []

for k, (i, j) in enumerate(mlp_pairs):
    if i in train_indices and j in train_indices:
        X_eval3_train.append(X_eval3[k])

    elif i in test_indices and j in test_indices:
        X_eval3_test.append(X_eval3[k])

eval3_scaler = StandardScaler()

X_eval3_train_scaled = eval3_scaler.fit_transform(X_eval3_train)
X_eval3_test_scaled = eval3_scaler.transform(X_eval3_test)

evaluator3 = LogisticRegression(class_weight="balanced")

evaluator3.fit(X_eval3_train_scaled, y_eval_train)

predictions3 = evaluator3.predict(X_eval3_test_scaled)
probabilities3 = evaluator3.predict_proba(X_eval3_test_scaled)[:, 1]

print("Evaluator 3 accuracy:", accuracy_score(y_eval_test, predictions3))
print("ROC-AUC:", roc_auc_score(y_eval_test, probabilities3))
print(classification_report(y_eval_test, predictions3, zero_division=0))


# This second set of evaluators is for the Random Forest Models

# Establishing ground truth
rf_shifted_accuracies = []

for model in random_forest_models:
    predictions = model.predict(X_shifted)
    rf_shifted_accuracies.append(
        accuracy_score(y_val, predictions)
    )

rf_X_eval = []
rf_y_eval = []

for i, j in forest_pairs:
    acc1 = random_forest_accuracies[i]
    acc2 = random_forest_accuracies[j]

    rf_X_eval.append([
        min(acc1, acc2),
        max(acc1, acc2),
        abs(acc1 - acc2)
    ])

    shifted_gap = abs(
        rf_shifted_accuracies[i] - rf_shifted_accuracies[j]
    )

    rf_y_eval.append(int(shifted_gap >= 0.05))

print("Total RF pairs:", len(rf_y_eval))
print("Underspecified RF pairs:", sum(rf_y_eval))
print("Non-underspecified RF pairs:",
      len(rf_y_eval) - sum(rf_y_eval))