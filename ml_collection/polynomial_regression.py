"""Polynomial regression by expanding powers of x and fitting ordinary least squares."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def polynomial_feature_matrix(x, degree):
    """Columns ``x, x^2, ..., x^degree``. The intercept is left to the linear model."""
    rows = []
    for value in x:
        rows.append([value ** power for power in range(1, degree + 1)])
    return np.asarray(rows, dtype=float)


def fit_polynomial_regression(x, y, degree):
    """Fit ``sklearn`` ordinary least squares on polynomial features of degree ``degree``."""
    features = polynomial_feature_matrix(x, degree)
    return LinearRegression().fit(features, y)


def polynomial_mean_squared_error(x, y, model, degree):
    """Mean squared error of a fitted polynomial model."""
    predictions = model.predict(polynomial_feature_matrix(x, degree))
    return np.sum((y - predictions) ** 2) / len(x)


def select_polynomial_degree(train_x, train_y, valid_x, valid_y, max_degree=20):
    """Choose the degree from 1..``max_degree`` with the smallest validation mean squared error.

    The search starts from a reference loss of 1. A degree is kept only when its validation
    loss is strictly smaller than that reference and than every earlier candidate. Degrees
    whose validation loss stays at or above 1 are ignored, matching the original assignment loop.
    """
    best_loss = 1.0
    best_degree = 0
    history = []
    for degree in range(1, max_degree + 1):
        model = fit_polynomial_regression(train_x, train_y, degree)
        loss = polynomial_mean_squared_error(valid_x, valid_y, model, degree)
        history.append((degree, float(loss)))
        if loss < best_loss:
            best_loss = float(loss)
            best_degree = degree
    return best_degree, best_loss, history


def plot_polynomial_fit(train_x, train_y, valid_x, valid_y, model, degree, show=False):
    """Plot train and validation points with the fitted polynomial."""
    import matplotlib.pyplot as plt

    from ml_collection.linear_regression import plot_regression_data

    fig, axes = plot_regression_data(train_x, train_y, valid_x, valid_y)
    x_plot = np.linspace(valid_x.min(), valid_x.max())
    y_plot = model.predict(polynomial_feature_matrix(x_plot, degree))
    axes.plot(x_plot, y_plot, label="Model", color="black")
    axes.legend()
    if show:
        plt.show()
    return fig, axes
