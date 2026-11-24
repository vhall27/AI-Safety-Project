# 
# Vanesska Hall and Jaspreet Sahota
# CS 120
# 
# Identify models with similar IID accuracy and establishes ground-truth labels.
#
# ------------------------------------------------------------------------------

from train_models import mlp_models, mlp_accuracies
from train_models import random_forest_models, random_forest_accuracies
from itertools import combinations

THRESHOLD = 0.01

# Finding pairs that have similar IID validation accuracy
def find_pairs(accuracies, THRESHOLD):
    pairs = []

    for i, j in combinations(range(len(accuracies)), 2):
        difference = abs(accuracies[i] - accuracies[j])

        if difference <= THRESHOLD:
            pairs.append((i, j))

    return pairs

mlp_pairs = find_pairs(mlp_accuracies, THRESHOLD)
forest_pairs = find_pairs(random_forest_accuracies, THRESHOLD)

print("Similar MLP pairs:", len(mlp_pairs))
print("Similar RF pairs:", len(forest_pairs))