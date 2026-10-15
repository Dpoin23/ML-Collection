#!/usr/bin/env python3
"""MNIST error of a 64-unit MLP versus logistic regression as the training set grows.

Downloads the OpenML MNIST copy on first run. This refits both models at each training size
and can take a long time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from ml_collection.datasets import load_mnist_split, standardize_train_test
from ml_collection.multilayer_perceptron import training_size_error_rates

TRAIN_SIZES = [100, 1000, 5000, 10000, 20000, 50000, 63000]


def main():
    features_train, features_test, labels_train, labels_test = load_mnist_split(test_size=0.1)
    features_train, features_test, _scaler = standardize_train_test(features_train, features_test)
    curves = training_size_error_rates(
        features_train, labels_train, features_test, labels_test, TRAIN_SIZES
    )
    plt.figure(figsize=(6, 6))
    plt.xlabel("Number of training points")
    plt.ylabel("Error rate")
    plt.semilogx(TRAIN_SIZES, curves["mlp_train_error"], color="lime", label="mlp train")
    plt.semilogx(TRAIN_SIZES, curves["mlp_test_error"], color="deepskyblue", label="mlp test")
    plt.semilogx(TRAIN_SIZES, curves["logistic_train_error"], color="red", label="logistic train")
    plt.semilogx(TRAIN_SIZES, curves["logistic_test_error"], color="black", label="logistic test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
