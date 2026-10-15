# ML Collection

Implementations and experiments from a classical machine-learning course. The library modules contain the algorithms. `tests/` holds the checks that used to live in `problem` functions. `examples/` loads data, fits models, and draws the plots.

Every randomized routine uses seed `1234` unless a function takes a different seed on purpose. The synthetic regression training split uses seed `1235` (`1234 + 1`). The validation split uses seed `1233` (`1234 - 1`).

## Setup

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

The IMDB example also needs the Hugging Face `datasets` package:

```bash
pip install -r requirements-examples.txt
```

Run the suite:

```bash
ruff check .
pytest
python scripts/smoke.py
python scripts/check_runnable.py
```

Run an experiment from the repository root:

```bash
python examples/penguin_knn.py
```

Plotting scripts open Matplotlib windows. They need a display.

## Layout

```
ml_collection/     algorithm implementations
tests/             pytest checks, including the old problem asserts
examples/          data loading, training sweeps, and plots
data/              Palmer penguins CSV and Yelp review JSON
images/            saved IMDB error curve
scripts/           smoke import and offline runnable check used by CI
.github/           Dependabot, CodeQL, and the test workflow
```

## Index

| Path | Definition | Algorithm | Data | Expected output |
| --- | --- | --- | --- | --- |
| `ml_collection/sample_variance.py` | Unbiased sample variance, divisor `n - 1` | Loop accumulation and a vectorized NumPy form | Any 1-d sample. The test uses Iris sepal length | Matches `numpy.var(ddof=1)` |
| `ml_collection/linear_regression.py` | Univariate line `theta_0 + theta_1 x` | Prediction, mean squared error, analytic gradient, batch gradient descent | Noisy sinh curve on `[-0.95, 0.95]` | Gradient at `(0.5, -1)` is about `(1.718, -2.182)`. Loss falls across 100 steps at learning rate `0.05` |
| `ml_collection/polynomial_regression.py` | Polynomial regression without a hand-built intercept column | Powers `x .. x^degree`, then ordinary least squares | Same sinh curve. 30 train points, 20 validation points | Degree 11 has the lowest validation MSE, about `0.138` |
| `ml_collection/nearest_centroid.py` | Class mean in feature space | Nearest centroid, accuracy, confusion counts | Integer labels `0 .. C-1`. The demo uses MNIST digits | Agrees with `sklearn.neighbors.NearestCentroid` on a separable toy set |
| `ml_collection/k_nearest_neighbors.py` | Label by majority vote of the nearest training points | k-nearest neighbors | `data/penguins.csv`, bill length and bill depth, three species | k=1 training error is `0`. Test error stays near `0.035` for k in `{2, 5, 10}` |
| `ml_collection/logistic_regression.py` | `P(y=1\|x) = sigmoid(theta · x)` | Binary cross-entropy, L1-regularized logistic regression, bag-of-words | Penguin bills for the from-scratch loss. `data/yelp_reviews.json` for text | Sigmoid at 0 is `0.5`. Yelp prints test accuracy and the 10 most positive and negative terms |
| `ml_collection/decision_tree.py` | Binary tree of threshold splits | Gini impurity, grown to a max depth | Lecture 12-point set, a 250-point synthetic set, and the wine cultivars | Lecture predictions are `[0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]`. They match a depth-3 scikit-learn tree on the synthetic set |
| `ml_collection/kmeans.py` | Partition by nearest center, then update centers to means | Lloyd's algorithm, random init and k-means++ | Iris sepal length and width. Ring-shaped synthetic blobs | Two-cluster Iris centers are about `(5.15, 3.17)` and `(6.52, 2.95)` |
| `ml_collection/multilayer_perceptron.py` | One ReLU hidden layer | scikit-learn `MLPClassifier` (SGD or Adam) and logistic regression | Caller supplies arrays. MNIST and IMDB are loaded in the examples | Helpers return error rates and loss curves. Short fits are covered in `tests/test_multilayer_perceptron.py` |
| `ml_collection/datasets.py` | Loaders and feature names | None. This is I/O | Iris, wine, MNIST via OpenML, IMDB via Hugging Face | Iris is `(150, 4)`. MNIST and IMDB download on first use |

### Sample variance

`sample_variance_loops` and `sample_variance_vectorized` both compute the unbiased sample variance. `time_variance` also times `numpy.var(ddof=1)`.

`examples/iris_exploration.py` prints the three Iris species, rows 3 through 6, and the mean, population standard deviation (`numpy.std`, divisor `n`), minimum, and maximum of each measurement. It then times the variance routines on sepal length and draws a histogram per feature plus a 4 by 4 scatter matrix colored by species (setosa blue, versicolor red, virginica green).

### Linear regression and gradient descent

`generate_sinh_regression_data` draws

```
y = 0.1 * (exp(3.75 x) - exp(-3.75 x) - 4) + 0.4 * noise
```

with `x` spaced on `[-0.95, 0.95]`.

`mean_squared_error_gradient` is the analytic gradient of the mean squared error. On the 30-point training split with parameters `(0.5, -1)` the gradient is `1.7177774269022454` and `-2.181717782869463`.

`examples/gradient_descent_regression.py` prints the data ranges and the initial loss, draws the starting line, draws the loss contour over `theta_0` in `[-2, 2]` and `theta_1` in `[-1, 3]`, then runs 100 steps at learning rate `0.05` from `(0.5, -1)`. The line is redrawn every 20 steps. The script finishes with a scatter plot of loss against iteration. The loss decreases.

### Polynomial regression

`polynomial_feature_matrix(x, degree)` returns columns `x, x^2, ..., x^degree`. For `[0.5, 3]` and degree 4 the matrix is

```
[[0.5, 0.25, 0.125, 0.0625],
 [3,   9,    27,    81     ]]
```

`select_polynomial_degree` fits degrees 1 through 20 and keeps the degree with the smallest validation mean squared error, ignoring any degree whose validation loss is at least 1. On this split the winner is degree 11, validation MSE about `0.138`.

`examples/polynomial_degree_selection.py` prints every degree and loss, then plots the chosen polynomial against the training and validation points.

### Nearest centroid

`NearestCentroidClassifier` stores the mean feature vector of each integer class and predicts the nearest mean under Euclidean distance. `compute_accuracy` is the fraction of exact label matches. `confusion_matrix_counts` is a `C` by `C` count matrix for labels `0 .. C-1`, rows true and columns predicted.

`examples/mnist_nearest_centroid.py` downloads MNIST (`mnist_784` on OpenML), holds out 25 percent, and shows the first nine training digits. It draws the ten centroids as 28 by 28 images, prints predicted and true labels for the first ten test images, and plots confusion matrices for this implementation and for `sklearn.neighbors.NearestCentroid`. The first download can take a minute.

### k-nearest neighbors

`load_penguin_bill_split` reads `data/penguins.csv` (342 penguins, columns `bill_length_mm`, `bill_depth_mm`, and `species`) and holds out 25 percent. The split is 256 train and 86 test.

`examples/penguin_knn.py` draws decision boundaries for k in `{1, 5, 10, 50}` and a log-scale error curve for k in `{1, 2, 5, 10, 50, 100, 110}`.

| k | Training error | Test error |
| --- | --- | --- |
| 1 | 0.0000 | 0.0465 |
| 2 | 0.0312 | 0.0349 |
| 5 | 0.0312 | 0.0349 |
| 10 | 0.0352 | 0.0349 |
| 50 | 0.0820 | 0.0698 |
| 100 | 0.1953 | 0.2674 |
| 110 | 0.1992 | 0.2558 |

### Logistic regression

`sigmoid`, `logistic_probability`, and `binary_cross_entropy` are the from-scratch binary model with a bias and two features. A zero linear score has probability `0.5` and cross-entropy `log(2)` when labels are balanced.

`examples/penguin_logistic_regression.py` marks Adelie as class 1 and the other species as class 0, standardizes the two bill measurements with the training mean and the pandas sample standard deviation, and minimizes cross-entropy with L-BFGS-B inside bounds `[-20, 20]`. It prints `theta_0`, `theta_1`, and `theta_2`, and draws the decision line in the standardized plane.

`reviews_to_bag_of_words` reads a Yelp-style JSON list. Stars of 4 or 5 are positive. Stars of 1 or 2 are negative. Three-star reviews are dropped. The vocabulary is English unigrams that appear in at least 2 percent of the kept reviews, with scikit-learn's English stopword list removed.

`examples/yelp_sentiment.py` reads `data/yelp_reviews.json` (about 16 MB). It plots a histogram of unique words per review, fits an L1 logistic regression (`l1_ratio=1`, liblinear solver, 30 percent test split), and prints test accuracy to two decimal places as a percent. It then prints the 10 terms with the largest weights and the 10 terms with the smallest weights, to three decimal places. A further sweep over inverse regularization `C` in `{0, 0.1, 1, 10, 50}` prints train and test accuracy. `C = 0` is the unregularized model (`C = inf` in scikit-learn). The figure shows the positive `C` values on a log axis.

### Decision trees

`DecisionTree` is a binary classifier for labels in `{0, 1}`. A node becomes a leaf at `max_depth` or when one class has probability 0. Each candidate threshold is the midpoint between two consecutive sorted feature values. The split with the smallest weighted Gini impurity is kept. Ties keep the earlier candidate. Thresholds that send every row to one child are skipped.

On the 12-point lecture set the first split is feature 0 at `8.5`, and the best split of the left side is feature 1 at `6`. Gini scores used by the tests include `5/11`, `2/5`, and `4/9`.

`examples/wine_decision_trees.py` loads the scikit-learn wine dataset (13 chemical measurements, 3 cultivars) with a 30 percent test split. A depth-2 tree prints:

```
Training accuracy: 92.74
Testing accuracy: 81.48
```

It draws that tree, then accuracy versus `max_depth` from 1 to 15, then accuracy versus `min_samples_leaf` from 1 to 15 with depth fixed at 3. The highest test accuracy on the depth sweep is at depth 7 (about 94.44 percent test, 100 percent train).

### k-means

`KMeansClustering` assigns each point to the nearest centroid by squared Euclidean distance and replaces each centroid with the mean of its members. Initialization is either `random` (a sample of rows without replacement) or `kmeans++` (the next center is drawn with probability proportional to squared distance to the nearest center). `random_state` reseeds NumPy's global generator, which is what the assignment checks depend on.

`compute_memberships`, `initialize_clusters`, `update_centroids`, and `fit` are locked to the assignment's Iris sepal expectations in `tests/test_kmeans.py`. With `k=2`, `init_strategy="random"`, and seed `1234`, the fitted centers are approximately `[[5.15, 3.169], [6.518, 2.949]]`.

`examples/kmeans_ring_clusters.py` samples six Gaussian blobs on the unit circle (100 points each, noise scale `0.1`). It plots k in `{2, 4, 6, 8}` with random initialization, then `k=6` for seeds `{5, 6, 7, 8}` under both random initialization and k-means++. Cluster regions come from a scikit-learn `KMeans` boundary display whose centers are replaced by this implementation's centroids.

### Multilayer perceptrons

Each network has one hidden layer of ReLU units. `training_size_error_rates` compares a 64-unit SGD network with logistic regression as the training prefix grows. Test error uses a prefix of the test set of that same length. `loss_curves_by_learning_rate` records the SGD loss curve. `fit_regularized_mlp` uses L2 strength `alpha=0.005`, batch size 10, and learning rate `0.005`. `error_rates_by_hidden_units` trains an Adam network for each width, defaulting to 20 iterations, batch size 256, and learning rate `0.001`.

`examples/mnist_training_size.py` downloads MNIST, holds out 10 percent, standardizes with the training mean and variance, and plots train and test error at training sizes `{100, 1000, 5000, 10000, 20000, 50000, 63000}`. This refits both models at every size and takes a while.

`examples/mnist_mlp.py` uses that same standardized split, trains on the first 10,000 images, and reserves the next 10,000. It plots loss for learning rates `{0.0005, 0.001, 0.005, 0.01}` over 100 iterations, then prints test-set accuracy for the regularized 64-unit network.

`examples/imdb_mlp_complexity.py` shuffles the IMDB sentiment split, keeps 5,000 train reviews and 5,000 test reviews, and vectorizes them with TF-IDF (`max_features=2000`, English stopwords). It sweeps hidden widths from 1 through 500 and writes `images/ERvsComp.png`. The curve is error rate against hidden-unit count on a log x-axis. Train error is green. Test error is red. Install `requirements-examples.txt` before running it. The corpus download is large.

## Tests

These replace the old `problem` functions that only asserted a result:

| Test module | What it checks |
| --- | --- |
| `tests/test_sample_variance.py` | Both variance implementations match NumPy on Iris sepal length |
| `tests/test_linear_regression.py` | Assignment gradient values, perfect-fit MSE of 0, and a decreasing gradient-descent loss |
| `tests/test_polynomial_regression.py` | The degree-4 feature matrix above, and degree search on a shorter sweep |
| `tests/test_nearest_centroid.py` | Accuracy `0.75` on a hand-built confusion case, and agreement with scikit-learn |
| `tests/test_k_nearest_neighbors.py` | Penguin split size 342 and error rates in `[0, 1]` |
| `tests/test_logistic_regression.py` | Sigmoid, cross-entropy, Adelie labels, a tiny review file, and the regularization sweep |
| `tests/test_decision_tree.py` | Class probabilities, leaf checks, lecture Gini scores and splits, lecture predictions, and agreement with a depth-3 scikit-learn tree on 250 synthetic points |
| `tests/test_kmeans.py` | Membership, both initializers, centroid updates, and fitted Iris centers |
| `tests/test_wine_trees.py` | Wine tree accuracies stay inside `(0, 1]` |
| `tests/test_multilayer_perceptron.py` | The MLP helpers fit a 40-point problem without downloading MNIST |

`scripts/smoke.py` imports the library and calls one pure function from each algorithm. `scripts/check_runnable.py` reruns the known gradient, the degree-1-through-20 polynomial search, penguin k-nearest neighbors, the lecture tree, and Iris k-means. Neither script downloads data or opens a plot.

## Continuous integration

`.github/` follows the Python workflow used in Study-Bot and the Dependabot and CodeQL layout shared with Easy-Gig.

- `.github/workflows/ci.yml` runs on pushes and pull requests to `main`. It installs `requirements.txt` and `requirements-dev.txt` on Python 3.12, then runs Ruff, `compileall`, pytest, `scripts/smoke.py`, and `scripts/check_runnable.py`.
- `.github/workflows/codeql.yml` analyzes Python on the same branches and on a weekly Sunday schedule. The query suite is `security-and-quality` (`.github/codeql/codeql-config.yml`). `data/` and `images/` are excluded from the scan.
- `.github/dependabot.yml` opens weekly update pull requests for pip and GitHub Actions.

The workflows assume the default branch is named `main`.
