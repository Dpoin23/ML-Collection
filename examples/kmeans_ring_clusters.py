#!/usr/bin/env python3
"""k-means on six ring-shaped blobs, varying k and the initialization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_collection.constants import DEFAULT_SEED
from ml_collection.kmeans import (
    KMeansClustering,
    plot_kmeans_decision_boundary,
    sample_ring_clusters,
)


def main():
    features = sample_ring_clusters(6, random_state=DEFAULT_SEED)
    print("samples", features.shape)

    for k in (2, 4, 6, 8):
        model = KMeansClustering(k, "random", random_state=DEFAULT_SEED).fit(features)
        print(f"k={k} centroids=\n{model.centroids}")
        plot_kmeans_decision_boundary(model, features, show=True)

    for random_state in (5, 6, 7, 8):
        model = KMeansClustering(6, "random", random_state=random_state).fit(features)
        print(f"random init seed={random_state}")
        plot_kmeans_decision_boundary(model, features, show=True)

    for random_state in (5, 6, 7, 8):
        model = KMeansClustering(6, "kmeans++", random_state=random_state).fit(features)
        print(f"k-means++ seed={random_state}")
        plot_kmeans_decision_boundary(model, features, show=True)


if __name__ == "__main__":
    main()
