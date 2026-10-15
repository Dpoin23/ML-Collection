"""Multilayer perceptron helpers used by the MNIST and IMDB experiments.

Callers pass arrays that are already loaded and, for MNIST, already standardized.
The full downloads live in ``examples/``.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from ml_collection.constants import DEFAULT_SEED


def error_rate(labels, predictions):
    """``1 - accuracy``."""
    return 1 - accuracy_score(labels, predictions)


def make_mlp(
    hidden_units,
    *,
    learning_rate=0.001,
    max_iter=200,
    seed=DEFAULT_SEED,
    alpha=0.0001,
    batch_size=256,
    solver="sgd",
    n_iter_no_change=10,
):
    """One hidden layer of ``hidden_units`` ReLU units."""
    return MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver=solver,
        batch_size=batch_size,
        learning_rate="constant",
        learning_rate_init=learning_rate,
        alpha=alpha,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        random_state=seed,
    )


def training_size_error_rates(
    features_train,
    labels_train,
    features_test,
    labels_test,
    train_sizes,
    seed=DEFAULT_SEED,
    max_iter=200,
):
    """Error of an MLP and of logistic regression as the training prefix grows.

    Test error uses the prefix ``features_test[:n]`` of the same length as the training prefix,
    which saturates once ``n`` exceeds the test set. That matches the original MNIST comparison.
    """
    mlp_train_error = []
    mlp_test_error = []
    logistic_train_error = []
    logistic_test_error = []

    for size in train_sizes:
        network = make_mlp(64, seed=seed, max_iter=max_iter)
        network.fit(features_train[:size], labels_train[:size])
        mlp_train_error.append(error_rate(labels_train[:size], network.predict(features_train[:size])))
        test_prefix = features_test[:size]
        label_prefix = labels_test[:size]
        mlp_test_error.append(error_rate(label_prefix, network.predict(test_prefix)))

        logistic = LogisticRegression(random_state=seed, max_iter=max_iter)
        logistic.fit(features_train[:size], labels_train[:size])
        logistic_train_error.append(error_rate(labels_train[:size], logistic.predict(features_train[:size])))
        logistic_test_error.append(error_rate(label_prefix, logistic.predict(test_prefix)))

    return {
        "train_sizes": list(train_sizes),
        "mlp_train_error": mlp_train_error,
        "mlp_test_error": mlp_test_error,
        "logistic_train_error": logistic_train_error,
        "logistic_test_error": logistic_test_error,
    }


def loss_curves_by_learning_rate(
    features,
    labels,
    learning_rates,
    seed=DEFAULT_SEED,
    max_iter=100,
    batch_size=256,
):
    """SGD loss curve for each constant learning rate, using a 64-unit hidden layer."""
    curves = {}
    for rate in learning_rates:
        network = make_mlp(
            64,
            learning_rate=rate,
            max_iter=max_iter,
            seed=seed,
            batch_size=batch_size,
            n_iter_no_change=max_iter,
        )
        network.fit(features, labels)
        curves[rate] = list(network.loss_curve_)
    return curves


def fit_regularized_mlp(
    features_train,
    labels_train,
    features_test,
    labels_test,
    *,
    hidden_units=64,
    alpha=0.005,
    batch_size=10,
    learning_rate=0.005,
    seed=DEFAULT_SEED,
    max_iter=200,
):
    """Fit an L2-regularized MLP and return ``(model, test accuracy)``."""
    network = make_mlp(
        hidden_units,
        learning_rate=learning_rate,
        max_iter=max_iter,
        seed=seed,
        alpha=alpha,
        batch_size=batch_size,
    )
    network.fit(features_train, labels_train)
    accuracy = accuracy_score(labels_test, network.predict(features_test))
    return network, float(accuracy)


def error_rates_by_hidden_units(
    features_train,
    labels_train,
    features_test,
    labels_test,
    hidden_units,
    seed=DEFAULT_SEED,
    max_iter=20,
    batch_size=256,
):
    """Train and test error of an Adam MLP for each hidden-layer width."""
    train_errors = []
    test_errors = []
    for width in hidden_units:
        network = make_mlp(
            width,
            learning_rate=0.001,
            max_iter=max_iter,
            seed=seed,
            solver="adam",
            batch_size=batch_size,
        )
        network.fit(features_train, labels_train)
        train_errors.append(error_rate(labels_train, network.predict(features_train)))
        test_errors.append(error_rate(labels_test, network.predict(features_test)))
    return train_errors, test_errors
