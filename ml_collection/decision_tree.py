"""Binary decision tree grown by exhaustive Gini splits."""

from __future__ import annotations

import numpy as np


class Node:
    """One node in a binary decision tree."""

    def __init__(self, depth):
        self.depth = depth
        self.split_feature = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.probs = None

    def __repr__(self):
        return (
            f"DT Node: \n -| Depth: {self.depth}"
            f"\n -| Split feature: {self.split_feature}"
            f"\n -| Threshold: {self.threshold}"
            f"\n -| Probs: {self.probs}"
        )


class DecisionTree:
    """Gini decision tree for labels in ``{0, 1}``."""

    def __init__(self, max_depth=3):
        self.root = None
        self.max_depth = max_depth

    def class_probability_vector(self, labels):
        """Return ``[P(y = 0), P(y = 1)]`` for the labels reaching a node."""
        count_zero = 0
        for label in labels:
            if label == 0:
                count_zero += 1
        probability_zero = count_zero / len(labels)
        probability_one = 1 - probability_zero
        return np.array([probability_zero, probability_one])

    def leaf_condition(self, node):
        """A node is a leaf at ``max_depth`` or when one class has probability 0."""
        if self.max_depth == node.depth or node.probs[0] == 0 or node.probs[1] == 0:
            return True
        return False

    def gini_score(self, features, labels, feature_index, threshold):
        """Weighted Gini impurity of splitting ``feature_index`` at ``threshold``.

        Rows with ``feature < threshold`` go left. The rest go right.
        Returns ``None`` when the threshold puts every row on one side.
        """
        num_lower = 0
        count_zero_lower = 0
        count_one_lower = 0
        count_zero_higher = 0
        count_one_higher = 0

        for row in range(len(features)):
            if features[row][feature_index] < threshold:
                num_lower += 1
                if labels[row] == 1:
                    count_one_lower += 1
                else:
                    count_zero_lower += 1
            elif labels[row] == 1:
                count_one_higher += 1
            else:
                count_zero_higher += 1

        num_higher = len(features) - num_lower
        if num_lower == 0 or num_higher == 0:
            return None
        weight_lower = num_lower / len(features)
        weight_higher = num_higher / len(features)
        probability_zero_lower = (count_zero_lower / num_lower) ** 2
        probability_one_lower = (count_one_lower / num_lower) ** 2
        probability_zero_higher = (count_zero_higher / num_higher) ** 2
        probability_one_higher = (count_one_higher / num_higher) ** 2
        return (
            weight_lower * (1 - probability_zero_lower - probability_one_lower)
            + weight_higher * (1 - probability_zero_higher - probability_one_higher)
        )

    def find_best_split(self, features, labels):
        """Feature index and midpoint threshold with the smallest Gini score.

        Candidate thresholds are midpoints between sorted values of each feature.
        Ties keep the earliest candidate.
        """
        best = ()
        best_gini = -1
        for feature_index in range(len(features[0])):
            ordered = sorted(features[row][feature_index] for row in range(len(features)))
            for row in range(len(features) - 1):
                midpoint = ordered[row] + (ordered[row + 1] - ordered[row]) / 2
                score = self.gini_score(features, labels, feature_index, midpoint)
                if score is None:
                    continue
                if best_gini == -1 or score < best_gini:
                    best = (feature_index, midpoint)
                    best_gini = score
        return best

    def fit(self, features, labels):
        """Grow the tree from the root."""
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a numpy array")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a numpy array")
        self.build_tree(features, labels, 0)
        return self

    def build_tree(self, features, labels, depth):
        """Recursively split until the leaf condition holds."""
        node = Node(depth)
        if depth == 0:
            self.root = node

        node.probs = self.class_probability_vector(labels)
        if self.leaf_condition(node):
            return node

        split_index, split_threshold = self.find_best_split(features, labels)
        left_features, left_labels = [], []
        right_features, right_labels = [], []
        for row in range(len(features)):
            if features[row][split_index] < split_threshold:
                left_features.append(features[row])
                left_labels.append(labels[row])
            else:
                right_features.append(features[row])
                right_labels.append(labels[row])

        node.split_feature = split_index
        node.threshold = split_threshold
        node.left_child = self.build_tree(left_features, left_labels, depth + 1)
        node.right_child = self.build_tree(right_features, right_labels, depth + 1)
        return node

    def predict(self, features):
        """Predict a label for every row."""
        return np.asarray([self._predict(row) for row in features])

    def _predict(self, row):
        """Walk from the root to a leaf and return the class with larger probability."""
        current = self.root
        while True:
            if self.leaf_condition(current):
                return int(np.argmax(current.probs))
            go_left = row[current.split_feature] <= current.threshold
            current = current.left_child if go_left else current.right_child

    def __repr__(self):
        return f"Decision Tree \n -| Max Depth: {self.max_depth}"


def sklearn_tree_accuracies(features_train, labels_train, features_test, labels_test, **kwargs):
    """Fit ``sklearn`` ``DecisionTreeClassifier`` and return train and test accuracy."""
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(**kwargs)
    model.fit(features_train, labels_train)
    train_accuracy = accuracy_score(labels_train, model.predict(features_train))
    test_accuracy = accuracy_score(labels_test, model.predict(features_test))
    return model, float(train_accuracy), float(test_accuracy)


def accuracy_versus_depth(features_train, labels_train, features_test, labels_test, depths):
    """Train and test accuracy of a scikit-learn tree at each ``max_depth``."""
    train_accuracies = []
    test_accuracies = []
    for depth in depths:
        _, train_accuracy, test_accuracy = sklearn_tree_accuracies(
            features_train,
            labels_train,
            features_test,
            labels_test,
            max_depth=depth,
        )
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    return train_accuracies, test_accuracies


def accuracy_versus_min_samples_leaf(
    features_train,
    labels_train,
    features_test,
    labels_test,
    min_samples,
    max_depth=3,
):
    """Train and test accuracy at each ``min_samples_leaf``, with ``max_depth`` fixed."""
    train_accuracies = []
    test_accuracies = []
    for min_samples_leaf in min_samples:
        _, train_accuracy, test_accuracy = sklearn_tree_accuracies(
            features_train,
            labels_train,
            features_test,
            labels_test,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    return train_accuracies, test_accuracies
