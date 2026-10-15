#!/usr/bin/env python3
"""Summarize and plot the Iris measurements."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from ml_collection.datasets import IRIS_FEATURE_NAMES, IRIS_SPECIES_NAMES, load_iris_measurements
from ml_collection.sample_variance import time_variance


def summarize_features(features):
    """Print mean, sample standard deviation, min, and max of each measurement."""
    for index, name in enumerate(IRIS_FEATURE_NAMES):
        column = features[:, index]
        print(f"{name} mean: {np.mean(column)}")
        print(f"{name} std: {np.std(column)}")
        print(f"{name} minimum: {np.min(column)}")
        print(f"{name} maximum: {np.max(column)}")
        print()


def plot_histograms(features):
    figure, axes = plt.subplots(1, 4, figsize=(12, 3))
    for index, name in enumerate(IRIS_FEATURE_NAMES):
        axes[index].hist(features[:, index], bins=7)
        axes[index].set_title(name, fontsize=14, fontweight="bold")
    figure.tight_layout()
    return figure


def plot_scatter_matrix(features, labels):
    colors = np.where(labels == 0, "blue", np.where(labels == 1, "red", "green"))
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for row in range(4):
        for column in range(4):
            axes[row, column].scatter(features[:, row], features[:, column], c=colors)
            axes[row, column].set_ylabel(IRIS_FEATURE_NAMES[row])
            axes[row, column].set_xlabel(IRIS_FEATURE_NAMES[column])
    figure.tight_layout()
    return figure


def main():
    features, labels = load_iris_measurements()
    print("species:", IRIS_SPECIES_NAMES)
    print("shape:", features.shape, labels.shape)
    print("rows 3:7 features:\n", features[3:7])
    print("rows 3:7 labels:", labels[3:7])
    summarize_features(features)
    report = time_variance(features[:, 0], repeats=20)
    for name, item in report.items():
        print(f"{name}: variance={item['variance']:.6f} seconds={item['seconds']:.6f}")
    plot_histograms(features)
    plot_scatter_matrix(features, labels)
    plt.show()


if __name__ == "__main__":
    main()
