"""A tiny MLP fit so the network helpers stay importable without MNIST."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from ml_collection.multilayer_perceptron import (
    error_rates_by_hidden_units,
    fit_regularized_mlp,
    loss_curves_by_learning_rate,
)

pytestmark = pytest.mark.filterwarnings("ignore:Stochastic Optimizer")


def test_mlp_helpers_fit_a_small_problem():
    features, labels = make_classification(n_samples=40, n_features=6, n_informative=3, random_state=0)
    _, accuracy = fit_regularized_mlp(
        features[:30],
        labels[:30],
        features[30:],
        labels[30:],
        hidden_units=8,
        batch_size=10,
        max_iter=40,
    )
    assert 0 <= accuracy <= 1

    curves = loss_curves_by_learning_rate(features[:30], labels[:30], [0.01], max_iter=30, batch_size=10)
    assert len(curves[0.01]) >= 1

    train_errors, test_errors = error_rates_by_hidden_units(
        features[:30],
        labels[:30],
        features[30:],
        labels[30:],
        [4],
        max_iter=30,
        batch_size=10,
    )
    assert len(train_errors) == len(test_errors) == 1
    assert np.all(np.asarray(train_errors) >= 0)
