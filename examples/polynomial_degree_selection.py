#!/usr/bin/env python3
"""Pick a polynomial degree on the sinh curve by validation mean squared error."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_collection.constants import DEFAULT_SEED
from ml_collection.linear_regression import generate_sinh_regression_data
from ml_collection.polynomial_regression import (
    fit_polynomial_regression,
    plot_polynomial_fit,
    select_polynomial_degree,
)


def main():
    train_x, train_y = generate_sinh_regression_data(30, seed=DEFAULT_SEED + 1)
    valid_x, valid_y = generate_sinh_regression_data(20, seed=DEFAULT_SEED - 1)
    degree, loss, history = select_polynomial_degree(train_x, train_y, valid_x, valid_y)
    for candidate, candidate_loss in history:
        print(candidate, candidate_loss)
    print(f"Min degree = {degree} (validation MSE = {loss})")
    model = fit_polynomial_regression(train_x, train_y, degree)
    plot_polynomial_fit(train_x, train_y, valid_x, valid_y, model, degree, show=True)


if __name__ == "__main__":
    main()
