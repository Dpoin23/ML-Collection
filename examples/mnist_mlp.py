#!/usr/bin/env python3
"""SGD loss curves on the first 10,000 standardized MNIST training images.

Downloads the OpenML MNIST copy on first run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from ml_collection.datasets import load_mnist_split, standardize_train_test
from ml_collection.multilayer_perceptron import fit_regularized_mlp, loss_curves_by_learning_rate

LEARNING_RATES = [0.0005, 0.001, 0.005, 0.01]


def main():
    features_train, features_test, labels_train, labels_test = load_mnist_split(test_size=0.1)
    features_train, features_test, _scaler = standardize_train_test(features_train, features_test)
    validation_features = features_train[10000:20000]
    validation_labels = labels_train[10000:20000]
    features_train = features_train[:10000]
    labels_train = labels_train[:10000]
    print("train", features_train.shape, "validation", validation_features.shape)

    curves = loss_curves_by_learning_rate(
        features_train,
        labels_train,
        LEARNING_RATES,
        max_iter=100,
    )
    plt.figure(figsize=(6, 6))
    for rate, curve in curves.items():
        plt.plot(curve, label=f"lr {rate}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    _model, accuracy = fit_regularized_mlp(features_train, labels_train, features_test, labels_test)
    print(f"Test accuracy: {accuracy * 100:.2f}")
    print(f"Training images: {len(labels_train)}. Reserved validation images: {len(validation_labels)}.")


if __name__ == "__main__":
    main()
