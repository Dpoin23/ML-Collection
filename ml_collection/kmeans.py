"""k-means clustering with random and k-means++ initialization."""

from __future__ import annotations

import numpy as np


class KMeansClustering:
    """Lloyd's algorithm with either uniform or k-means++ centroid initialization.

    ``random_state`` reseeds NumPy's global generator so the assignment checks stay reproducible.
    """

    def __init__(self, k, init_strategy, random_state=None):
        np.random.seed(random_state)
        self.k = k
        self.centroids = None
        if init_strategy not in ("random", "kmeans++"):
            raise ValueError("init_strategy must be 'random' or 'kmeans++'")
        self.init_strategy = init_strategy

    def compute_memberships(self, features, centroids):
        """Assign each row to the nearest centroid.

        Returns cluster indices and the squared Euclidean distance to that centroid.
        """
        squared_distances = []
        which_cluster = []
        for point in features:
            smallest_distance = -1
            cluster = -1
            for index in range(len(centroids)):
                distance = sum((point[axis] - centroids[index][axis]) ** 2 for axis in range(len(point)))
                if distance < smallest_distance or smallest_distance == -1:
                    smallest_distance = distance
                    cluster = index
            squared_distances.append(smallest_distance)
            which_cluster.append(cluster)
        return np.asarray(which_cluster), np.asarray(squared_distances)

    def initialize_clusters(self, features):
        """Pick the starting centroids with the configured strategy."""
        if self.init_strategy == "random":
            rows = np.random.choice(len(features), self.k, replace=False)
            return np.asarray([features[index] for index in rows])

        centroids = []
        first_index = np.random.choice(features.shape[0])
        centroids.append(features[first_index])
        for _ in range(self.k - 1):
            _, squared_distances = self.compute_memberships(features, centroids)
            probabilities = squared_distances / sum(squared_distances)
            row = np.random.choice(len(features), p=probabilities)
            centroids.append(features[row])
        return np.asarray(centroids)

    def update_centroids(self, features, which_cluster):
        """Replace each centroid with the mean of the points assigned to it."""
        centroids = []
        for cluster in range(self.k):
            members = [features[index] for index in range(len(which_cluster)) if which_cluster[index] == cluster]
            centroids.append(np.mean(members, axis=0))
        return np.asarray(centroids)

    def fit(self, features, max_iter=100):
        """Run Lloyd updates until assignments stop changing or ``max_iter`` is reached."""
        centroids = self.initialize_clusters(features)
        which_cluster, _ = self.compute_memberships(features, centroids)
        converged = False
        iteration = 0
        while not converged:
            new_centroids = self.update_centroids(features, which_cluster)
            new_which_cluster, _ = self.compute_memberships(features, new_centroids)
            if np.array_equal(which_cluster, new_which_cluster) or iteration == max_iter:
                converged = True
            centroids = new_centroids
            which_cluster = new_which_cluster
            iteration += 1
        self.centroids = centroids
        return self


def sample_ring_clusters(n_clusters, n_samples_per_cluster=100, random_state=None, radius=1.0, sigma=0.1):
    """Sample isotropic blobs centered on a circle."""
    np.random.seed(random_state)
    samples = []
    angle_step = 2 * np.pi / n_clusters
    for cluster in range(n_clusters):
        theta = cluster * angle_step
        center = np.array([radius * np.cos(theta), radius * np.sin(theta)])
        blob = sigma * np.random.randn(n_samples_per_cluster, 2) + center
        samples.append(blob)
    return np.asarray(samples).reshape(-1, 2)


def plot_kmeans_decision_boundary(model, features, show=False):
    """Color the plane by cluster using the fitted centroids.

    scikit-learn draws the boundary. Its centers are replaced with this implementation's centroids.
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.inspection import DecisionBoundaryDisplay

    reference = KMeans(n_clusters=model.k)
    reference.fit(features)
    reference.cluster_centers_ = model.centroids
    boundary = DecisionBoundaryDisplay.from_estimator(
        reference,
        features,
        alpha=0.4,
        grid_resolution=250,
        response_method="predict",
    )
    boundary.ax_.scatter(*features.T, edgecolor="k", marker="o", label="data")
    boundary.ax_.scatter(
        *model.centroids.T,
        label="centroids",
        edgecolor="k",
        color="red",
        marker="x",
        s=64,
    )
    plt.legend()
    if show:
        plt.show()
    return boundary
