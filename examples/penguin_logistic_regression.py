#!/usr/bin/env python3
"""Two-feature logistic regression separating Adelie penguins from the other species."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from ml_collection.k_nearest_neighbors import load_penguin_bill_split
from ml_collection.logistic_regression import adelie_indicator, binary_cross_entropy


def plot_decision_boundary(theta_0, theta_1, theta_2, x_1, x_2, labels):
    figure, axes = plt.subplots()
    colors = np.where(labels == 1, "blue", "green")
    axes.scatter(x_1, x_2, alpha=0.5, c=colors)
    grid = np.linspace(np.min(x_1), np.max(x_1), 256)
    boundary = -1 * (theta_0 + theta_1 * grid) / theta_2
    axes.plot(grid, boundary, color="black")
    axes.set_xlabel("bill_length_mm (standardized)")
    axes.set_ylabel("bill_depth_mm (standardized)")
    return figure, axes


def main():
    features_train, _, labels_train, _ = load_penguin_bill_split()
    bill_length = features_train["bill_length_mm"]
    bill_depth = features_train["bill_depth_mm"]
    labels = adelie_indicator(labels_train)
    print(bill_length.shape, bill_depth.shape, labels.shape)

    standardized_length = (bill_length - bill_length.mean()) / bill_length.std()
    standardized_depth = (bill_depth - bill_depth.mean()) / bill_depth.std()

    def loss(theta):
        return binary_cross_entropy(
            theta[0],
            theta[1],
            theta[2],
            standardized_length,
            standardized_depth,
            labels,
        )
    bound = 20
    result = minimize(loss, np.zeros(3), bounds=[(-bound, bound)] * 3)
    print(f"theta_0={result.x[0]}, theta_1={result.x[1]}, theta_2={result.x[2]}")
    plot_decision_boundary(
        result.x[0],
        result.x[1],
        result.x[2],
        standardized_length,
        standardized_depth,
        labels,
    )
    plt.show()


if __name__ == "__main__":
    main()
