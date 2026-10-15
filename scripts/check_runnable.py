#!/usr/bin/env python3
"""Fit the offline algorithms on local or synthetic data.

MNIST, IMDB, and the Yelp dump are left to the example scripts. This check uses the
penguins CSV, the iris measurements, and the seeded sinh curve.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import numpy as np
    from sklearn.datasets import load_iris

    from ml_collection.constants import DEFAULT_SEED
    from ml_collection.decision_tree import DecisionTree
    from ml_collection.k_nearest_neighbors import knn_error_rates, load_penguin_bill_split
    from ml_collection.kmeans import KMeansClustering
    from ml_collection.linear_regression import (
        generate_sinh_regression_data,
        mean_squared_error_gradient,
    )
    from ml_collection.polynomial_regression import select_polynomial_degree
    from ml_collection.sample_variance import sample_variance_loops, sample_variance_vectorized

    sepal_length = load_iris().data[:, 0]
    expected = float(sepal_length.var(ddof=1))
    if not math.isclose(sample_variance_loops(sepal_length), expected):
        raise SystemExit("loop variance mismatch")
    if not math.isclose(sample_variance_vectorized(sepal_length), expected):
        raise SystemExit("vectorized variance mismatch")

    features, labels = generate_sinh_regression_data(30, seed=DEFAULT_SEED + 1)
    grad_0, grad_1 = mean_squared_error_gradient(features, labels, 0.5, -1.0)
    if not math.isclose(grad_0, 1.7177774269022454):
        raise SystemExit(f"unexpected bias gradient: {grad_0}")
    if not math.isclose(grad_1, -2.181717782869463):
        raise SystemExit(f"unexpected slope gradient: {grad_1}")

    valid_x, valid_y = generate_sinh_regression_data(20, seed=DEFAULT_SEED - 1)
    degree, loss, _history = select_polynomial_degree(
        features, labels, valid_x, valid_y, max_degree=20
    )
    if not 1 <= degree <= 20 or not loss < 1:
        raise SystemExit(f"polynomial degree selection failed: degree={degree} loss={loss}")

    features_train, features_test, labels_train, labels_test = load_penguin_bill_split()
    train_errors, test_errors = knn_error_rates(
        features_train, labels_train, features_test, labels_test, [1, 5]
    )
    if len(train_errors) != 2 or np.any(test_errors < 0):
        raise SystemExit("penguin k-nn error rates failed")

    lecture_x = np.array([[1, 1], [2, 8], [4, 9], [6, 7], [7, 4], [8, 11], [3, 3], [5, 5], [9, 5], [10, 8], [11, 6], [12, 10]])
    lecture_y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    predictions = DecisionTree(max_depth=3).fit(lecture_x, lecture_y).predict(lecture_x)
    expected_labels = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    if not np.array_equal(predictions, expected_labels):
        raise SystemExit("decision tree lecture predictions failed")

    iris_xy = load_iris().data[:, :2]
    centroids = KMeansClustering(2, "random", random_state=DEFAULT_SEED).fit(iris_xy).centroids
    if centroids.shape != (2, 2):
        raise SystemExit("k-means fit failed")

    print(
        f"runnable ok: variance, gradient, polynomial degree={degree}, "
        f"penguin knn, decision tree, k-means"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:
        print(f"runnable check failed: {err}", file=sys.stderr)
        sys.exit(1)
