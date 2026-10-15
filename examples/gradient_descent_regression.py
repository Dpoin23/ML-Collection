#!/usr/bin/env python3
"""Gradient descent on the noisy sinh curve, then the mean-squared-error contour."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from ml_collection.constants import DEFAULT_SEED
from ml_collection.linear_regression import (
    generate_sinh_regression_data,
    gradient_descent,
    mean_squared_error,
    plot_linear_fit,
    plot_mse_contour,
)

THETA_0 = 0.5
THETA_1 = -1.0


def main():
    features, labels = generate_sinh_regression_data(30, seed=DEFAULT_SEED + 1)
    print(
        f"X range: [{features.min():.2f}, {features.max():.2f}], "
        f"Y range: [{labels.min():.2f}, {labels.max():.2f}]"
    )
    print("initial MSE:", mean_squared_error(features, labels, THETA_0, THETA_1))
    plot_linear_fit(features, labels, THETA_0, THETA_1, show=True)
    plot_mse_contour(features, labels)
    plt.show()

    _, _, losses = gradient_descent(
        features,
        labels,
        THETA_0,
        THETA_1,
        lr=0.05,
        max_iters=100,
        display_boundary=True,
    )
    _figure, axes = plt.subplots()
    axes.scatter(range(len(losses)), losses)
    axes.set_xlabel("Iteration", fontsize=14)
    axes.set_ylabel("MSE", fontsize=14)
    plt.show()
    print(f"final MSE: {losses[-1]}")


if __name__ == "__main__":
    main()
