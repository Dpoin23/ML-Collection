"""Univariate linear regression: prediction, mean squared error, and gradient descent."""

from __future__ import annotations

import numpy as np

from ml_collection.constants import DEFAULT_SEED


def generate_sinh_regression_data(n, seed=DEFAULT_SEED, noise_scale=0.4):
    """Sample a noisy curve based on ``sinh`` on ``[-0.95, 0.95]``.

    The mean function is ``0.1 * (exp(3.75 x) - exp(-3.75 x) - 4)``.
    The assignment training split uses ``seed + 1`` and the validation split uses ``seed - 1``.
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(-0.95, 0.95, n)
    mean_y = (np.exp(3.75 * x) - np.exp(-3.75 * x) - 4.0) * 0.1
    y = mean_y + rng.randn(n) * noise_scale
    return x, y


def predict_linear(x, theta_0, theta_1):
    """Predict ``y = theta_0 + theta_1 * x``."""
    return theta_0 + theta_1 * x


def mean_squared_error(x, y, theta_0, theta_1):
    """Mean squared error of a univariate line."""
    residual = y - predict_linear(x, theta_0, theta_1)
    return np.sum(residual ** 2) / len(x)


def mean_squared_error_gradient(x, y, theta_0, theta_1):
    """Gradient of mean squared error with respect to bias and slope.

    Returns ``(dL/dtheta_0, dL/dtheta_1)``.
    """
    residual = y - theta_0 - theta_1 * x
    grad_0 = (-2 / len(x)) * np.sum(residual)
    grad_1 = (-2 / len(x)) * np.sum(x * residual)
    return grad_0, grad_1


def gradient_descent(
    x,
    y,
    theta_0_init,
    theta_1_init,
    lr=0.001,
    max_iters=100,
    display_boundary=False,
):
    """Batch gradient descent on univariate mean squared error.

    Returns the bias, slope, and loss recorded after initialization and after every step.
    When ``display_boundary`` is true, the current line is drawn every 20 steps.
    """
    theta_0 = theta_0_init
    theta_1 = theta_1_init
    loss = mean_squared_error(x, y, theta_0, theta_1)

    theta_0s = [theta_0]
    theta_1s = [theta_1]
    losses = [loss]

    iteration = 0
    while iteration < max_iters and loss != 0:
        if display_boundary and iteration % 20 == 0:
            plot_linear_fit(x, y, theta_0, theta_1, show=True)

        grad_0, grad_1 = mean_squared_error_gradient(x, y, theta_0, theta_1)
        theta_0 = theta_0 - lr * grad_0
        theta_1 = theta_1 - lr * grad_1
        loss = mean_squared_error(x, y, theta_0, theta_1)

        theta_0s.append(theta_0)
        theta_1s.append(theta_1)
        losses.append(loss)
        iteration += 1

    return theta_0s, theta_1s, losses


def plot_regression_data(x, y, valid_x=None, valid_y=None):
    """Scatter training points, and validation points when they are provided."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots()
    axes.scatter(x, y, label="Training")
    if valid_x is not None:
        axes.scatter(valid_x, valid_y, label="Validation")
        axes.legend()
    axes.set_xlabel("x", fontsize=14)
    axes.set_ylabel("y", fontsize=14)
    axes.set_xlim(-1.0, 1.0)
    axes.set_ylim(-4, 4)
    return fig, axes


def plot_linear_fit(x, y, theta_0, theta_1, show=False):
    """Scatter ``(x, y)`` and overlay ``theta_0 + theta_1 x``."""
    import matplotlib.pyplot as plt

    fig, axes = plot_regression_data(x, y)
    axes.plot(x, predict_linear(x, theta_0, theta_1))
    if show:
        plt.show()
    return fig, axes


def plot_mse_contour(x, y):
    """Contour plot of mean squared error over a grid of bias and slope."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots()
    theta_0_grid, theta_1_grid = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))

    losses = [
        mean_squared_error(x, y, theta_0, theta_1)
        for theta_0, theta_1 in zip(theta_0_grid.ravel(), theta_1_grid.ravel(), strict=True)
    ]
    loss_grid = np.asarray(losses).reshape(100, 100)

    axes.set_xlabel("theta_0", fontsize=14)
    axes.set_ylabel("theta_1", fontsize=14)
    axes.set_title("MSE", fontsize=18)
    image = axes.contour(theta_0_grid, theta_1_grid, loss_grid, 100)
    fig.colorbar(image, ax=axes)
    return fig, axes
