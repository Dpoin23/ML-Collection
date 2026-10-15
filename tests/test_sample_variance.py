from __future__ import annotations

import math

import numpy as np
from sklearn.datasets import load_iris

from ml_collection.sample_variance import sample_variance_loops, sample_variance_vectorized


def test_sample_variance_matches_numpy_on_iris_sepal_length():
    sepal_length = load_iris().data[:, 0]
    expected = sepal_length.var(ddof=1)
    assert math.isclose(sample_variance_loops(sepal_length), expected)
    assert math.isclose(sample_variance_vectorized(sepal_length), expected)


def test_sample_variance_known_vector():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(sample_variance_loops(values), values.var(ddof=1))
    assert math.isclose(sample_variance_vectorized(values), 1.6666666666666667)
