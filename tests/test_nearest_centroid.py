"""Checks for the nearest-centroid classifier on a separable toy problem."""

from __future__ import annotations

import numpy as np

from ml_collection.nearest_centroid import (
    NearestCentroidClassifier,
    compute_accuracy,
    confusion_matrix_counts,
    nearest_centroid_scores,
    sklearn_nearest_centroid_scores,
)


def _blobs():
    features = np.array([[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.2, 4.8], [0.1, -0.2], [4.7, 5.1]])
    labels = np.array([0, 0, 1, 1, 0, 1])
    return features, labels


def test_accuracy_and_confusion_matrix():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    assert compute_accuracy(labels, predictions) == 0.75
    assert np.array_equal(
        confusion_matrix_counts(labels, predictions),
        np.array([[1, 1], [0, 2]]),
    )


def test_nearest_centroid_separates_blobs():
    features, labels = _blobs()
    model = NearestCentroidClassifier().fit(features, labels)
    assert np.array_equal(model.predict(np.array([[0.0, 0.0], [5.0, 5.0]])), np.array([0, 1]))


def test_local_classifier_matches_sklearn():
    features, labels = _blobs()
    train, test = features[:4], features[4:]
    train_labels, test_labels = labels[:4], labels[4:]
    local = nearest_centroid_scores(train, train_labels, test, test_labels)
    reference = sklearn_nearest_centroid_scores(train, train_labels, test, test_labels)
    assert local["test_accuracy"] == reference["test_accuracy"]
    assert np.array_equal(local["confusion_matrix"], reference["confusion_matrix"])
