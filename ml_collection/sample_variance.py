"""Sample variance (divisor n - 1), loop form and vectorized form."""

from __future__ import annotations

import time

import numpy as np


def sample_variance_loops(values):
    """Unbiased sample variance using explicit loops.

    Matches ``numpy.var(values, ddof=1)``.
    """
    count = len(values)
    total = 0
    for value in values:
        total += value

    mean = total / count
    squared_error = 0
    for value in values:
        squared_error += (value - mean) ** 2

    return squared_error / (count - 1)


def sample_variance_vectorized(values):
    """Unbiased sample variance using NumPy reductions."""
    values = np.asarray(values)
    mean = np.mean(values)
    return np.sum((values - mean) ** 2) / (len(values) - 1)


def time_variance(values, repeats=1):
    """Return results and elapsed seconds for the loop, vectorized, and NumPy versions."""
    implementations = {
        "loops": sample_variance_loops,
        "vectorized": sample_variance_vectorized,
        "numpy": lambda array: np.asarray(array).var(ddof=1),
    }
    report = {}
    for name, function in implementations.items():
        start = time.perf_counter()
        result = None
        for _ in range(repeats):
            result = function(values)
        report[name] = {"variance": result, "seconds": time.perf_counter() - start}
    return report
