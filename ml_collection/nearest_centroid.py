"""Nearest-centroid classifier, accuracy, and confusion counts.

Labels are integers ``0 .. n_classes - 1``. The MNIST digit experiments use ``0 .. 9``.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid


def euclidean_distance(left, right):
    """Euclidean distance between two vectors."""
    left = np.asarray(left)
    right = np.asarray(right)
    return math.sqrt(sum((left[i] - right[i]) ** 2 for i in range(left.size)))


class NearestCentroidClassifier:
    """Assign each point to the class whose mean feature vector is closest."""

    def __init__(self):
        self.classes_ = None
        self.centroids = []

    def fit(self, features, labels):
        """Store one centroid per class, in sorted label order."""
        labels = np.asarray(labels)
        features = np.asarray(features)
        self.classes_ = np.unique(labels)
        self.centroids = []
        for label in self.classes_:
            members = features[labels == label]
            self.centroids.append(np.mean(members, axis=0))
        return self

    def predict(self, features):
        """Predict the nearest centroid's class label for each row."""
        if self.classes_ is None:
            raise RuntimeError("NearestCentroidClassifier must be fit before predict")

        predictions = []
        for vector in np.asarray(features):
            best_distance = None
            best_label = None
            for label, centroid in zip(self.classes_, self.centroids, strict=True):
                distance = euclidean_distance(vector, centroid)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_label = label
            predictions.append(best_label)
        return np.asarray(predictions)

    def plot_centroids(self, image_shape=(28, 28), show=False):
        """Draw each centroid as an image. MNIST centroids are 28 by 28."""
        import matplotlib.pyplot as plt

        count = len(self.centroids)
        columns = min(5, count)
        rows = math.ceil(count / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.2 * rows))
        flat_axes = np.atleast_1d(axes).ravel()
        plt.rcParams["image.interpolation"] = "nearest"
        plt.rcParams["image.cmap"] = "gray"

        for index, (label, centroid) in enumerate(zip(self.classes_, self.centroids, strict=True)):
            flat_axes[index].imshow(np.asarray(centroid).reshape(image_shape))
            flat_axes[index].set_title(f"Label: {label}")
        for index in range(count, len(flat_axes)):
            flat_axes[index].axis("off")

        figure.tight_layout()
        if show:
            plt.show()
        return figure, axes


def compute_accuracy(labels, predictions):
    """Fraction of predictions that match the true labels."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    matches = 0
    for index in range(labels.size):
        if labels[index] == predictions[index]:
            matches += 1
    return matches / labels.size


def confusion_matrix_counts(labels, predictions):
    """Count predictions for integer labels ``0 .. C-1``.

    Row is the true class and column is the predicted class.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    n_classes = int(max(np.max(labels), np.max(predictions))) + 1
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for index in range(labels.size):
        matrix[int(labels[index]), int(predictions[index])] += 1
    return matrix


def nearest_centroid_scores(features_train, labels_train, features_test, labels_test):
    """Train and score the local nearest-centroid classifier.

    Returns train accuracy, test accuracy, and the test confusion matrix.
    """
    model = NearestCentroidClassifier()
    model.fit(features_train, labels_train)
    train_predictions = model.predict(features_train)
    test_predictions = model.predict(features_test)
    return {
        "model": model,
        "train_accuracy": compute_accuracy(labels_train, train_predictions),
        "test_accuracy": compute_accuracy(labels_test, test_predictions),
        "confusion_matrix": confusion_matrix_counts(labels_test, test_predictions),
    }


def sklearn_nearest_centroid_scores(features_train, labels_train, features_test, labels_test):
    """Train and score ``sklearn.neighbors.NearestCentroid`` on the same split."""
    model = NearestCentroid()
    model.fit(features_train, labels_train)
    train_predictions = model.predict(features_train)
    test_predictions = model.predict(features_test)
    return {
        "model": model,
        "train_accuracy": accuracy_score(labels_train, train_predictions),
        "test_accuracy": accuracy_score(labels_test, test_predictions),
        "confusion_matrix": confusion_matrix_counts(labels_test, test_predictions),
    }
