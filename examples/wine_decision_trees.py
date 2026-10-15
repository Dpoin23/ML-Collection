#!/usr/bin/env python3
"""scikit-learn decision trees on the wine cultivar dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from ml_collection.datasets import load_wine_split
from ml_collection.decision_tree import (
    accuracy_versus_depth,
    accuracy_versus_min_samples_leaf,
    sklearn_tree_accuracies,
)

DEPTHS = list(range(1, 16))
MIN_SAMPLES = list(range(1, 16))


def main():
    features_train, features_test, labels_train, labels_test = load_wine_split()
    print("train", features_train.shape, "test", features_test.shape)

    model, train_accuracy, test_accuracy = sklearn_tree_accuracies(
        features_train, labels_train, features_test, labels_test, max_depth=2
    )
    print(f"Training accuracy: {train_accuracy * 100:.2f}")
    print(f"Testing accuracy: {test_accuracy * 100:.2f}")
    plt.figure(figsize=(10, 6))
    plot_tree(model)
    plt.show()

    train_by_depth, test_by_depth = accuracy_versus_depth(
        features_train, labels_train, features_test, labels_test, DEPTHS
    )
    plt.figure(figsize=(6, 6))
    plt.plot(DEPTHS, [value * 100 for value in train_by_depth], color="blue", label="train")
    plt.plot(DEPTHS, [value * 100 for value in test_by_depth], color="red", label="test")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

    train_by_leaf, test_by_leaf = accuracy_versus_min_samples_leaf(
        features_train, labels_train, features_test, labels_test, MIN_SAMPLES, max_depth=3
    )
    plt.figure(figsize=(6, 6))
    plt.plot(MIN_SAMPLES, [value * 100 for value in train_by_leaf], color="blue", label="train")
    plt.plot(MIN_SAMPLES, [value * 100 for value in test_by_leaf], color="red", label="test")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
