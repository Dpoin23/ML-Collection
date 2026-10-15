"""Repository paths for local datasets and saved figures."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DATA_DIR = REPO_ROOT / "data"
IMAGE_DIR = REPO_ROOT / "images"

PENGUINS_CSV = DATA_DIR / "penguins.csv"
YELP_REVIEWS_JSON = DATA_DIR / "yelp_reviews.json"
IMDB_ERROR_CURVE_PNG = IMAGE_DIR / "ERvsComp.png"
