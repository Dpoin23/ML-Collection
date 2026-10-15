#!/usr/bin/env python3
"""k-nearest neighbors decision boundaries and error rates for penguin bills."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

from ml_collection.k_nearest_neighbors import fit_knn, knn_error_rates, load_penguin_bill_split


def plot_boundaries(features_train, labels_train, neighbors=(1, 5, 10, 50)):
    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    colors = np.where(
        labels_train == "Adelie",
        "blue",
        np.where(labels_train == "Gentoo", "red", "green"),
    )
    for index, k in enumerate(neighbors):
        axis = axes[index // 2, index % 2]
        classifier = fit_knn(features_train, labels_train, k)
        DecisionBoundaryDisplay.from_estimator(
            classifier,
            features_train,
            response_method="predict",
            plot_method="pcolormesh",
            alpha=0.5,
            grid_resolution=200,
            ax=axis,
        )
        axis.scatter(
            features_train["bill_length_mm"],
            features_train["bill_depth_mm"],
            alpha=0.5,
            c=colors,
        )
        axis.set_xlabel("bill_length_mm")
        axis.set_ylabel("bill_depth_mm")
        axis.set_title(f"k = {k}")
    figure.tight_layout()
    return figure


def plot_error_rates(features_train, labels_train, features_test, labels_test):
    neighbors = [1, 2, 5, 10, 50, 100, 110]
    train_errors, test_errors = knn_error_rates(
        features_train, labels_train, features_test, labels_test, neighbors
    )
    plt.figure(figsize=(6, 6))
    plt.semilogx(neighbors, train_errors, color="red", label="training error rate")
    plt.semilogx(neighbors, test_errors, color="green", label="test error rate")
    plt.title("Error rate vs k")
    plt.xlabel("k")
    plt.ylabel("Error rate")
    plt.legend()
    for k, train_error, test_error in zip(neighbors, train_errors, test_errors, strict=True):
        print(f"k={k}: train error={train_error:.4f} test error={test_error:.4f}")


def main():
    features_train, features_test, labels_train, labels_test = load_penguin_bill_split()
    print(features_train.shape, features_test.shape, labels_train.shape, labels_test.shape)
    plot_boundaries(features_train, labels_train)
    plot_error_rates(features_train, labels_train, features_test, labels_test)
    plt.show()


if __name__ == "__main__":
    main()
