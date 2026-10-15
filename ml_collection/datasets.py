"""Dataset loaders for the experiments.

MNIST and IMDB are downloaded by the example scripts that need them. Penguins and the
Yelp review dump are local files under ``data/``.
"""

from __future__ import annotations

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_collection.constants import DEFAULT_SEED

IRIS_FEATURE_NAMES = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)
IRIS_SPECIES_NAMES = ("setosa", "versicolor", "virginica")


def load_iris_measurements():
    """Iris features ``(150, 4)`` and integer species labels."""
    features, labels = load_iris(return_X_y=True)
    return features, labels


def load_wine_split(test_size=0.3, seed=DEFAULT_SEED):
    """Wine cultivar features with a shuffled train/test split."""
    features, labels = load_wine(return_X_y=True)
    return train_test_split(features, labels, test_size=test_size, random_state=seed)


def load_mnist_split(test_size, seed=DEFAULT_SEED):
    """Download MNIST through OpenML and return a shuffled split of integer labels."""
    from sklearn.datasets import fetch_openml

    features, labels = fetch_openml("mnist_784", as_frame=False, return_X_y=True)
    labels = labels.astype(int)
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )


def standardize_train_test(features_train, features_test):
    """Z-score features using the training mean and variance."""
    scaler = StandardScaler()
    return scaler.fit_transform(features_train), scaler.transform(features_test), scaler


def load_imdb_tfidf(train_documents=5000, test_documents=5000, max_features=2000, seed=DEFAULT_SEED):
    """Shuffle the Hugging Face IMDB split and vectorize reviews with TF-IDF.

    Requires the optional ``datasets`` package. Positive and negative labels stay as
    the dataset's ``0`` and ``1``.
    """
    from datasets import load_dataset

    corpus = load_dataset("imdb")
    train = corpus["train"].shuffle(seed)
    test = corpus["test"].shuffle(seed)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    features_train = vectorizer.fit_transform(train["text"][:train_documents])
    features_test = vectorizer.transform(test["text"][:test_documents])
    labels_train = train["label"][:train_documents]
    labels_test = test["label"][:test_documents]
    return features_train, features_test, labels_train, labels_test, vectorizer
