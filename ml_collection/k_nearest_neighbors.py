"""k-nearest neighbors error rates on the Palmer penguins bill measurements."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from ml_collection.constants import DEFAULT_SEED
from ml_collection.paths import PENGUINS_CSV


def load_penguin_bill_split(csv_path=None, seed=DEFAULT_SEED, test_size=0.25):
    """Split bill length and bill depth against species.

    Returns pandas objects so column names stay available for the boundary plots.
    """
    import pandas as pd

    frame = pd.read_csv(csv_path or PENGUINS_CSV)
    features = frame[["bill_length_mm", "bill_depth_mm"]]
    labels = frame["species"]
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )


def knn_error_rates(features_train, labels_train, features_test, labels_test, neighbors):
    """Training and test error (``1 - accuracy``) for each neighbor count."""
    train_errors = []
    test_errors = []
    for k in neighbors:
        classifier = KNeighborsClassifier(k)
        classifier.fit(features_train, labels_train)
        train_predictions = classifier.predict(features_train)
        test_predictions = classifier.predict(features_test)
        train_errors.append(1 - accuracy_score(labels_train, train_predictions))
        test_errors.append(1 - accuracy_score(labels_test, test_predictions))
    return np.asarray(train_errors, dtype=float), np.asarray(test_errors, dtype=float)


def fit_knn(features_train, labels_train, n_neighbors):
    """Fit a Euclidean k-nearest neighbors classifier."""
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(features_train, labels_train)
    return classifier
