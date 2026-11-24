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
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from preprocess_data import X_train, X_val


# Now training 40 MLP models
mlp_models = []
mlp_accuracies = []

for random_seed in range(40):
    cur_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver='adam',
        max_iter=1000,
        random_state=random_seed
    )

    cur_mlp.fit(X_train_scaled, y_train)

    predictions = cur_mlp.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, predictions)

    mlp_models.append(cur_mlp)
    mlp_accuracies.append(accuracy)

    print(f"Seed {random_seed + 1} has an accuracy of {accuracy}")

# Now training 40 Random Forest models
random_forest_models = []
random_forest_accuracies = []

for random_seed in range(40):
    cur_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=random_seed
    )

    cur_forest.fit(X_train, y_train)

    predictions = cur_forest.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    random_forest_models.append(cur_forest)
    random_forest_accuracies.append(accuracy)

    print(f"Seed {random_seed + 1} has an accuracy of {accuracy}")