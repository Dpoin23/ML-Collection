"""Train and test accuracy of scikit-learn trees on the wine dataset."""

from __future__ import annotations

from ml_collection.datasets import load_wine_split
from ml_collection.decision_tree import accuracy_versus_depth, sklearn_tree_accuracies


def test_wine_tree_accuracies_are_in_range():
    features_train, features_test, labels_train, labels_test = load_wine_split()
    _, train_accuracy, test_accuracy = sklearn_tree_accuracies(
        features_train,
        labels_train,
        features_test,
        labels_test,
        max_depth=2,
    )
    assert 0 < train_accuracy <= 1
    assert 0 < test_accuracy <= 1

    depths = [1, 2, 3]
    train_curve, test_curve = accuracy_versus_depth(
        features_train, labels_train, features_test, labels_test, depths
    )
    assert len(train_curve) == len(test_curve) == 3
