#!/usr/bin/env python3
"""Import the library and exercise one pure function from each algorithm module."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import numpy as np

    from ml_collection import DecisionTree, KMeansClustering, NearestCentroidClassifier
    from ml_collection.constants import DEFAULT_SEED
    from ml_collection.linear_regression import mean_squared_error, predict_linear
    from ml_collection.logistic_regression import sigmoid
    from ml_collection.polynomial_regression import polynomial_feature_matrix
    from ml_collection.sample_variance import sample_variance_vectorized

    values = np.array([1.0, 2.0, 3.0, 4.0])
    if abs(sample_variance_vectorized(values) - values.var(ddof=1)) > 1e-9:
        raise SystemExit("sample variance diverged from NumPy")
    if not np.allclose(predict_linear(values, 1.0, 2.0), 1 + 2 * values):
        raise SystemExit("linear prediction failed")
    if mean_squared_error(values, predict_linear(values, 1.0, 2.0), 1.0, 2.0) != 0:
        raise SystemExit("mean squared error failed")
    transformed = polynomial_feature_matrix(np.array([0.5]), 2)
    if not np.allclose(transformed, [[0.5, 0.25]]):
        raise SystemExit("polynomial features failed")
    if abs(float(sigmoid(0)) - 0.5) > 1e-9:
        raise SystemExit("sigmoid failed")

    features = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    nearest = NearestCentroidClassifier().fit(features, labels)
    if list(nearest.predict(features[:1])) != [0]:
        raise SystemExit("nearest centroid failed")
    tree = DecisionTree(max_depth=2).fit(features, labels)
    if tree.predict(features).shape != (4,):
        raise SystemExit("decision tree failed")
    kmeans = KMeansClustering(2, "kmeans++", random_state=DEFAULT_SEED).fit(features)
    if kmeans.centroids.shape != (2, 2):
        raise SystemExit("k-means failed")

    print(
        "smoke ok: variance, linear regression, polynomial features, "
        "logistic sigmoid, nearest centroid, decision tree, k-means"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:
        print(f"smoke failed: {err}", file=sys.stderr)
        sys.exit(1)
