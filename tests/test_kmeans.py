"""Assignment checks for k-means membership, initialization, and fitted centers."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from ml_collection.constants import DEFAULT_SEED
from ml_collection.kmeans import KMeansClustering, sample_ring_clusters

IRIS_XY = load_iris(return_X_y=True)[0][:, :2]


def test_compute_memberships():
    model = KMeansClustering(3, "random", random_state=DEFAULT_SEED)
    centroids = np.array([[4.5, 3.0], [5.5, 4.0]])
    which_cluster, squared_distances = model.compute_memberships(IRIS_XY[:5], centroids)
    assert np.array_equal(which_cluster, np.array([1, 0, 0, 0, 1]))
    assert np.allclose(squared_distances, np.array([0.41, 0.16, 0.08, 0.02, 0.41]), rtol=1e-4, atol=1e-4)

    model = KMeansClustering(3, "random", random_state=DEFAULT_SEED)
    centroids = np.array([[1.0, 10.0], [5.5, 4.0]])
    which_cluster, squared_distances = model.compute_memberships(IRIS_XY[:5], centroids)
    assert np.array_equal(which_cluster, np.array([1, 1, 1, 1, 1]))
    assert np.allclose(squared_distances, np.array([0.41, 1.36, 1.28, 1.62, 0.41]), rtol=1e-4, atol=1e-4)


def test_initialize_clusters():
    model = KMeansClustering(5, "random", random_state=DEFAULT_SEED)
    expected = np.array([[6.1, 3.0], [6.1, 2.9], [6.3, 2.9], [4.6, 3.4], [5.2, 2.7]])
    assert np.array_equal(model.initialize_clusters(IRIS_XY), expected)

    model = KMeansClustering(5, "kmeans++", random_state=DEFAULT_SEED)
    expected = np.array([[4.6, 3.2], [7.2, 3.6], [6.3, 2.7], [5.8, 2.7], [6.7, 2.5]])
    assert np.array_equal(model.initialize_clusters(IRIS_XY), expected)


def test_update_centroids():
    model = KMeansClustering(2, "random", random_state=DEFAULT_SEED)
    which_cluster = np.array([1, 0, 0, 1, 0])
    centroids = model.update_centroids(IRIS_XY[:5], which_cluster)
    assert np.allclose(centroids, np.array([[4.87, 3.27], [4.85, 3.3]]), rtol=1e-3, atol=1e-3)

    model = KMeansClustering(3, "random", random_state=DEFAULT_SEED)
    which_cluster = np.array([1, 2, 0, 1, 2])
    centroids = model.update_centroids(IRIS_XY[:5], which_cluster)
    assert np.allclose(centroids, np.array([[4.7, 3.2], [4.85, 3.3], [4.95, 3.3]]), rtol=1e-3, atol=1e-3)


def test_fit_centroids_on_iris_sepal_features():
    model = KMeansClustering(2, "random", random_state=DEFAULT_SEED)
    model.fit(IRIS_XY)
    assert np.allclose(model.centroids, np.array([[5.15, 3.1689], [6.5184, 2.94868]]), rtol=1e-3, atol=1e-3)

    model = KMeansClustering(5, "random", random_state=DEFAULT_SEED)
    model.fit(IRIS_XY)
    expected = np.array(
        [
            [5.3706, 3.8],
            [6.11346, 2.8673],
            [7.05, 3.0833],
            [4.828125, 3.265625],
            [5.33158, 2.52105],
        ]
    )
    assert np.allclose(model.centroids, expected, rtol=1e-3, atol=1e-3)


def test_ring_sample_shape():
    samples = sample_ring_clusters(6, n_samples_per_cluster=10, random_state=DEFAULT_SEED)
    assert samples.shape == (60, 2)
