#!/usr/bin/env python3
"""L1 logistic regression on the local Yelp review dump.

Reads ``data/yelp_reviews.json``. Positive reviews have 4 or 5 stars. Negative reviews
have 1 or 2 stars. Three-star reviews are dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from ml_collection.logistic_regression import (
    accuracy_versus_regularization,
    fit_l1_logistic_regression,
    reviews_to_bag_of_words,
    top_weighted_terms,
    unique_word_counts,
)
from ml_collection.paths import YELP_REVIEWS_JSON


def main():
    print("Loading", YELP_REVIEWS_JSON)
    features, labels, vectorizer, _texts = reviews_to_bag_of_words(YELP_REVIEWS_JSON)
    print("Data shape:", features.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))

    counts = unique_word_counts(features)
    plt.hist(counts, bins=25)
    plt.xlabel("Unique words")
    plt.ylabel("Number of reviews")
    plt.show()

    classifier, test_accuracy, shape = fit_l1_logistic_regression(features, labels)
    print("Number of training examples:", shape[0])
    print("Vocabulary size:", shape[1])
    print(f"\nTesting accuracy: {100 * test_accuracy:.2f}")

    positive_weights, negative_weights, positive_terms, negative_terms = top_weighted_terms(
        classifier, vectorizer, k=10
    )
    print("Positive terms (higher weight means more associated with 4-5 stars):")
    for term, weight in zip(positive_terms, positive_weights, strict=True):
        print(f"{term}: {weight:.3f}")
    print("Negative terms (lower weight means more associated with 1-2 stars):")
    for term, weight in zip(negative_terms, negative_weights, strict=True):
        print(f"{term}: {weight:.3f}")

    strengths = [0, 0.1, 1, 10, 50]
    train_accuracies, test_accuracies, _coefficients = accuracy_versus_regularization(
        features, labels, strengths
    )
    for strength, train_accuracy, held_out in zip(
        strengths, train_accuracies, test_accuracies, strict=True
    ):
        print(f"C={strength}: train={train_accuracy:.4f} test={held_out:.4f}")

    _figure, axes = plt.subplots()
    positive_strengths = [strength for strength in strengths if strength > 0]
    axes.semilogx(positive_strengths, train_accuracies[1:], color="red", label="training accuracy")
    axes.semilogx(positive_strengths, test_accuracies[1:], color="blue", label="testing accuracy")
    axes.set_xlabel("inverse regularization C")
    axes.set_ylabel("accuracy")
    axes.legend()
    plt.show()


if __name__ == "__main__":
    main()
