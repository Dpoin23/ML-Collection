"""Checks for logistic probability, loss, and a tiny bag-of-words model."""

from __future__ import annotations

import json
import math

import numpy as np

from ml_collection.logistic_regression import (
    accuracy_versus_regularization,
    adelie_indicator,
    binary_cross_entropy,
    fit_l1_logistic_regression,
    logistic_probability,
    reviews_to_bag_of_words,
    sigmoid,
    top_weighted_terms,
    unique_word_counts,
)


def test_sigmoid_and_probability_at_zero_are_one_half():
    assert math.isclose(float(sigmoid(0)), 0.5)
    probability = logistic_probability(0, 0, 0, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert np.allclose(probability, [0.5, 0.5])


def test_cross_entropy_of_a_coin_flip_is_log_two():
    loss = binary_cross_entropy(0, 0, 0, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1, 0]))
    assert math.isclose(float(loss), math.log(2))


def test_adelie_indicator():
    assert np.array_equal(adelie_indicator(["Adelie", "Gentoo", "Adelie"]), np.array([1, 0, 1]))


def test_bag_of_words_drops_middle_stars(tmp_path):
    reviews = [
        {"stars": 5, "text": "great food and great service"},
        {"stars": 5, "text": "great food and lovely service"},
        {"stars": 4, "text": "great food again"},
        {"stars": 4, "text": "awesome great meal"},
        {"stars": 1, "text": "terrible food and terrible service"},
        {"stars": 1, "text": "awful terrible meal"},
        {"stars": 2, "text": "bad food and bad service"},
        {"stars": 2, "text": "horrible terrible experience"},
        {"stars": 3, "text": "this middle review should be dropped"},
    ]
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps(reviews), encoding="utf-8")

    features, labels, vectorizer, texts = reviews_to_bag_of_words(path, min_df=1)
    assert labels.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert len(texts) == 8
    assert "middle" not in vectorizer.vocabulary_
    counts = unique_word_counts(features)
    assert counts.shape == (8,)
    assert np.all(counts >= 1)

    classifier, accuracy, shape = fit_l1_logistic_regression(features, labels, test_size=0.34, C=1.0)
    assert shape[1] == len(vectorizer.vocabulary_)
    assert 0 <= accuracy <= 1
    positive_weights, negative_weights, positive_terms, negative_terms = top_weighted_terms(
        classifier, vectorizer, k=1
    )
    assert len(positive_terms) == len(negative_terms) == 1
    assert positive_weights[0] >= negative_weights[0]


def test_accuracy_versus_regularization_returns_one_entry_per_strength():
    features = np.array([[0.0], [0.1], [1.0], [1.1], [0.2], [0.9]])
    labels = np.array([0, 0, 1, 1, 0, 1])
    train_accuracies, test_accuracies, coefficients = accuracy_versus_regularization(
        features, labels, strengths=[0, 1.0]
    )
    assert len(train_accuracies) == len(test_accuracies) == len(coefficients) == 2
