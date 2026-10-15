"""Assignment checks for univariate linear regression."""

from __future__ import annotations

import math

import numpy as np

from ml_collection.constants import DEFAULT_SEED
from ml_collection.linear_regression import (
    generate_sinh_regression_data,
    gradient_descent,
    mean_squared_error,
    mean_squared_error_gradient,
    predict_linear,
)


def _training_split():
    return generate_sinh_regression_data(30, seed=DEFAULT_SEED + 1)


def test_gradient_matches_assignment_values():
    features, labels = _training_split()
    grad_0, grad_1 = mean_squared_error_gradient(features, labels, 0.5, -1.0)
    assert math.isclose(grad_0, 1.7177774269022454)
    assert math.isclose(grad_1, -2.181717782869463)


def test_predict_and_mean_squared_error():
    features = np.array([0.0, 1.0, 2.0])
    labels = np.array([1.0, 3.0, 5.0])
    assert np.allclose(predict_linear(features, 1.0, 2.0), labels)
    assert mean_squared_error(features, labels, 1.0, 2.0) == 0


def test_gradient_descent_reduces_loss():
    features, labels = _training_split()
    _, _, losses = gradient_descent(features, labels, 0.5, -1.0, lr=0.05, max_iters=100)
    assert len(losses) == 101
    assert losses[-1] < losses[0]
