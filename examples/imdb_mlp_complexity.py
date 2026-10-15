#!/usr/bin/env python3
"""IMDB review sentiment: test error versus MLP hidden-layer width.

Requires the optional ``datasets`` package and a download of the IMDB corpus::

    pip install -r requirements-examples.txt

The figure is written to ``images/ERvsComp.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from ml_collection.datasets import load_imdb_tfidf
from ml_collection.multilayer_perceptron import error_rates_by_hidden_units
from ml_collection.paths import IMDB_ERROR_CURVE_PNG

HIDDEN_UNITS = [1, 2, 5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def main():
    features_train, features_test, labels_train, labels_test, _vectorizer = load_imdb_tfidf()
    train_errors, test_errors = error_rates_by_hidden_units(
        features_train,
        labels_train,
        features_test,
        labels_test,
        HIDDEN_UNITS,
    )
    for width, train_error, test_error in zip(HIDDEN_UNITS, train_errors, test_errors, strict=True):
        print(f"hidden units={width}: train error={train_error:.4f} test error={test_error:.4f}")

    plt.figure(figsize=(6, 6))
    plt.title("Error rate vs. neural network complexity")
    plt.xlabel("Hidden units")
    plt.ylabel("Error rate")
    plt.semilogx(HIDDEN_UNITS, train_errors, color="green", label="train error")
    plt.semilogx(HIDDEN_UNITS, test_errors, color="red", label="test error")
    plt.legend()
    IMDB_ERROR_CURVE_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMDB_ERROR_CURVE_PNG, dpi=150, bbox_inches="tight")
    print("saved", IMDB_ERROR_CURVE_PNG)
    plt.show()


if __name__ == "__main__":
    main()
