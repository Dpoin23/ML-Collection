"""Binary logistic regression: from-scratch loss, and L1-regularized text classification."""

from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml_collection.constants import DEFAULT_SEED


def sigmoid(z):
    """Logistic sigmoid. Accepts scalars and arrays."""
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def logistic_probability(theta_0, theta_1, theta_2, x_1, x_2):
    """``P(y = 1 | x)`` for a two-feature logistic model."""
    return sigmoid(theta_0 + theta_1 * x_1 + theta_2 * x_2)


def binary_cross_entropy(theta_0, theta_1, theta_2, x_1, x_2, y):
    """Mean binary cross-entropy of a two-feature logistic model."""
    probability = logistic_probability(theta_0, theta_1, theta_2, x_1, x_2)
    y = np.asarray(y, dtype=float)
    loss = (-y * np.log(probability)) - ((1 - y) * np.log(1 - probability))
    return np.sum(loss) / y.size


def adelie_indicator(species):
    """Map species names to 1 for Adelie and 0 for every other species."""
    return np.asarray([1 if name == "Adelie" else 0 for name in species], dtype=int)


def reviews_to_bag_of_words(filename, min_pos=4, max_neg=2, min_df=0.02):
    """Load Yelp-style JSON reviews and build a unigram bag of words.

    Stars ``>= min_pos`` are positive (1). Stars ``<= max_neg`` are negative (0).
    Reviews between those scores are dropped. Tokens are English unigrams that appear
    in at least ``min_df`` of the kept reviews, with the English stopword list removed.
    """
    with open(filename, encoding="utf-8") as handle:
        data = json.load(handle)

    texts = []
    labels = []
    for review in data:
        stars = int(review["stars"])
        if stars >= min_pos:
            score = 1
        elif stars <= max_neg:
            score = 0
        else:
            continue
        texts.append(review["text"])
        labels.append(score)

    vectorizer = CountVectorizer(stop_words="english", min_df=min_df, ngram_range=(1, 1))
    features = vectorizer.fit_transform(texts)
    return features, np.asarray(labels), vectorizer, texts


def unique_word_counts(bag_of_words):
    """Number of vocabulary words with a nonzero count in each document."""
    dense = bag_of_words.toarray()
    return np.sum(dense >= 1, axis=1)


def make_l1_logistic_regression(C=1.0, seed=DEFAULT_SEED):
    """L1-regularized logistic regression with the liblinear solver.

    ``l1_ratio=1`` selects the L1 penalty. scikit-learn 1.8 deprecated the ``penalty`` argument.
    """
    return LogisticRegression(
        l1_ratio=1.0,
        C=C,
        solver="liblinear",
        fit_intercept=True,
        random_state=seed,
    )


def fit_l1_logistic_regression(features, labels, test_size=0.3, seed=DEFAULT_SEED, C=1.0):
    """Fit L1 logistic regression and return the model plus test accuracy."""
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed
    )
    classifier = make_l1_logistic_regression(C=C, seed=seed)
    classifier.fit(features_train, labels_train)
    test_accuracy = accuracy_score(labels_test, classifier.predict(features_test))
    return classifier, float(test_accuracy), features_train.shape


def top_weighted_terms(classifier, vectorizer, k):
    """Return the ``k`` largest and ``k`` smallest logistic coefficients and their terms.

    Order is largest positive weight first, then most negative weight first.
    """
    vocabulary = {index: word for word, index in vectorizer.vocabulary_.items()}
    coefficients = classifier.coef_[0]
    order = np.argsort(coefficients)
    positive_index = order[coefficients.size - k : coefficients.size][::-1]
    negative_index = order[:k]
    positive_terms = [vocabulary[index] for index in positive_index]
    positive_weights = [float(coefficients[index]) for index in positive_index]
    negative_terms = [vocabulary[index] for index in negative_index]
    negative_weights = [float(coefficients[index]) for index in negative_index]
    return positive_weights, negative_weights, positive_terms, negative_terms


def accuracy_versus_regularization(features, labels, strengths, seed=DEFAULT_SEED):
    """Train and test accuracy across L1 strengths.

    A strength of 0 fits an unregularized model (``C=inf``). Every other strength
    is the inverse regularization ``C`` of an L1 liblinear model.
    """
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=seed
    )
    coefficients = []
    train_accuracies = []
    test_accuracies = []
    for strength in strengths:
        if strength == 0:
            classifier = LogisticRegression(C=np.inf, fit_intercept=True, random_state=seed)
        else:
            classifier = make_l1_logistic_regression(C=strength, seed=seed)
        classifier.fit(features_train, labels_train)
        coefficients.append(classifier.coef_)
        train_accuracies.append(accuracy_score(labels_train, classifier.predict(features_train)))
        test_accuracies.append(accuracy_score(labels_test, classifier.predict(features_test)))
    return train_accuracies, test_accuracies, coefficients
