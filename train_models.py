# 
# Vanesska Hall and Jaspreet Sahota
# CS 120
# 
# This file trains MLP and Random Forest models across random seeds.
#
# ------------------------------------------------------------------

from preprocess_data import X_train_scaled, X_val_scaled
from preprocess_data import y_train, y_val

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Training one baseline model for now
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

predictions = mlp.predict(X_val_scaled)

accuracy = accuracy_score(y_val, predictions)

print("MLP validation accuracy:", accuracy)