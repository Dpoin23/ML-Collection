#!/usr/bin/env python3
"""Nearest centroid on MNIST digits, compared with scikit-learn.

Downloads the OpenML MNIST copy on first run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from ml_collection.datasets import load_mnist_split
from ml_collection.nearest_centroid import (
    NearestCentroidClassifier,
    nearest_centroid_scores,
    sklearn_nearest_centroid_scores,
)


def plot_first_nine(features, labels):
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "gray"
    figure, axes = plt.subplots(3, 3, figsize=(6, 6))
    for row in range(3):
        for column in range(3):
            index = row * 3 + column
            axes[row, column].imshow(features[index].reshape(28, 28))
            axes[row, column].set_title(f"Label: {labels[index]}", fontsize=12)
    figure.tight_layout()
    return figure


def main():
    features_train, features_test, labels_train, labels_test = load_mnist_split(test_size=0.25)
    features, labels = features_train, labels_train
    plot_first_nine(features, labels)
    plt.show()

    model = NearestCentroidClassifier().fit(features_train, labels_train)
    model.plot_centroids(show=True)
    predicted = model.predict(features_test[:10])
    for index in range(10):
        print(
            f"Image {index}: predicted={predicted[index]}, actual={labels_test[index]}"
        )

    local = nearest_centroid_scores(features_train, labels_train, features_test, labels_test)
    reference = sklearn_nearest_centroid_scores(
        features_train, labels_train, features_test, labels_test
    )
    print("Local nearest centroid")
    print(f"--- Accuracy (train): {local['train_accuracy']}")
    print(f"--- Accuracy (test): {local['test_accuracy']}")
    print("scikit-learn nearest centroid")
    print(f"--- Accuracy (train): {reference['train_accuracy']}")
    print(f"--- Accuracy (test): {reference['test_accuracy']}")
    ConfusionMatrixDisplay(confusion_matrix=local["confusion_matrix"]).plot()
    ConfusionMatrixDisplay(confusion_matrix=reference["confusion_matrix"]).plot()
    plt.show()


if __name__ == "__main__":
    main()
