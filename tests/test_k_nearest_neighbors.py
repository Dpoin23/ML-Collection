"""Checks for penguin k-nearest neighbors error rates."""

from __future__ import annotations

import numpy as np

from ml_collection.k_nearest_neighbors import knn_error_rates, load_penguin_bill_split


def test_penguin_split_and_error_rates():
    features_train, features_test, labels_train, labels_test = load_penguin_bill_split()
    assert len(features_train) + len(features_test) == 342
    neighbors = [1, 5, 50]
    train_errors, test_errors = knn_error_rates(
        features_train, labels_train, features_test, labels_test, neighbors
    )
    assert train_errors.shape == (3,)
    assert test_errors.shape == (3,)
    assert np.all((train_errors >= 0) & (train_errors <= 1))
    assert np.all((test_errors >= 0) & (test_errors <= 1))
    assert train_errors[0] <= train_errors[-1]
