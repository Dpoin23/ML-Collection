"""Assignment checks for the from-scratch Gini decision tree."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from ml_collection.constants import DEFAULT_SEED
from ml_collection.decision_tree import DecisionTree, Node

LECTURE_FEATURES = np.array(
    [
        [1, 1],
        [2, 8],
        [4, 9],
        [6, 7],
        [7, 4],
        [8, 11],
        [3, 3],
        [5, 5],
        [9, 5],
        [10, 8],
        [11, 6],
        [12, 10],
    ]
)
LECTURE_LABELS = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


def test_class_probability_vector():
    tree = DecisionTree(max_depth=3)
    assert np.array_equal(tree.class_probability_vector(np.array([0, 0, 1, 1, 1])), np.array([0.4, 0.6]))
    assert np.array_equal(tree.class_probability_vector(np.array([1, 1, 1])), np.array([0.0, 1.0]))


def test_leaf_condition():
    tree = DecisionTree(max_depth=3)

    mixed = Node(depth=2)
    mixed.probs = np.array([0.5, 0.5])
    assert tree.leaf_condition(mixed) is False

    at_max_depth = Node(depth=3)
    at_max_depth.probs = np.array([0.5, 0.5])
    assert tree.leaf_condition(at_max_depth) is True

    pure = Node(depth=1)
    pure.probs = np.array([1.0, 0.0])
    assert tree.leaf_condition(pure) is True


def test_gini_score_rejects_an_empty_side():
    tree = DecisionTree(max_depth=3)
    features = np.array([[0.0], [0.0], [1.0]])
    labels = np.array([0, 1, 1])
    assert tree.gini_score(features, labels, 0, 0.0) is None


def test_gini_score_matches_lecture_examples():
    tree = DecisionTree(max_depth=3)
    assert np.isclose(tree.gini_score(LECTURE_FEATURES, LECTURE_LABELS, 0, 1.5), 5 / 11)
    assert np.isclose(tree.gini_score(LECTURE_FEATURES, LECTURE_LABELS, 0, 2.5), 2 / 5)
    assert np.isclose(tree.gini_score(LECTURE_FEATURES, LECTURE_LABELS, 1, 2), 5 / 11)
    assert np.isclose(tree.gini_score(LECTURE_FEATURES, LECTURE_LABELS, 1, 6.5), 4 / 9)


def test_find_best_split_matches_lecture_examples():
    tree = DecisionTree(max_depth=3)
    assert tree.find_best_split(LECTURE_FEATURES, LECTURE_LABELS) == (0, 8.5)

    left_mask = LECTURE_FEATURES[:, 0] <= 8.5
    assert tree.find_best_split(LECTURE_FEATURES[left_mask], LECTURE_LABELS[left_mask]) == (1, 6)


def test_predict_matches_lecture_labels_and_sklearn():
    expected = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    tree = DecisionTree(max_depth=3)
    tree.fit(LECTURE_FEATURES, LECTURE_LABELS)
    assert np.array_equal(tree.predict(LECTURE_FEATURES), expected)

    features, labels = make_classification(
        n_samples=250,
        n_features=5,
        n_informative=3,
        random_state=DEFAULT_SEED,
    )
    reference = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=DEFAULT_SEED)
    reference.fit(features, labels)
    tree = DecisionTree(max_depth=3)
    tree.fit(features, labels)
    assert np.array_equal(tree.predict(features), reference.predict(features))
