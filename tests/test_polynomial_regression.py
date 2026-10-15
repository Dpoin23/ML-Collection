"""Assignment checks for polynomial feature expansion."""

from __future__ import annotations

import numpy as np

from ml_collection.constants import DEFAULT_SEED
from ml_collection.linear_regression import generate_sinh_regression_data
from ml_collection.polynomial_regression import (
    polynomial_feature_matrix,
    select_polynomial_degree,
)


def test_polynomial_feature_matrix_matches_assignment_example():
    transformed = polynomial_feature_matrix(np.array([0.5, 3.0]), 4)
    expected = np.array([[0.5, 0.25, 0.125, 0.0625], [3.0, 9.0, 27.0, 81.0]])
    assert np.allclose(transformed, expected)


def test_degree_selection_returns_a_degree_in_range():
    train_x, train_y = generate_sinh_regression_data(30, seed=DEFAULT_SEED + 1)
    valid_x, valid_y = generate_sinh_regression_data(20, seed=DEFAULT_SEED - 1)
    degree, loss, history = select_polynomial_degree(train_x, train_y, valid_x, valid_y, max_degree=6)
    assert 1 <= degree <= 6
    assert loss < 1
    assert [item[0] for item in history] == [1, 2, 3, 4, 5, 6]
