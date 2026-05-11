"""
config.py
=========
Central configuration module for the Movie Recommendation System.

This module is the single source of truth for:
    * Filesystem paths (data, models, reports)
    * API credentials and endpoints (TMDB)
    * Model hyperparameters (TF-IDF, SVD, weighted-rating)
    * Reproducibility seeds
    * Logging configuration

Design principles
-----------------
1. Path objects use `pathlib.Path` — OS-independent.
2. Secrets (API keys) are loaded from a `.env` file via `python-dotenv`,
   never hard-coded. A `.env.example` is committed instead.
3. Hyperparameters live here so experiments can be tracked from one place.
4. `setup_logging()` is exposed so every module gets the same format.

Usage
-----
>>> from config import settings, setup_logging
>>> setup_logging()
>>> logger = logging.getLogger(__name__)
>>> print(settings.PATHS.RAW_DATA)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load environment variables from .env (silently fails if file is absent,
# which is fine — values can also be exported in the shell).
# -----------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_STATE: Final[int] = 42


# =============================================================================
# Paths
# =============================================================================
@dataclass(frozen=True)
class Paths:
    """Filesystem paths used throughout the project.

    All paths are absolute. Using a frozen dataclass prevents accidental
    mutation at runtime.
    """

    ROOT: Path = PROJECT_ROOT

    # Data directories
    DATA: Path = PROJECT_ROOT / "data"
    RAW_DATA: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA: Path = PROJECT_ROOT / "data" / "processed"
    EXTERNAL_DATA: Path = PROJECT_ROOT / "data" / "external"

    # Model artifacts
    MODELS: Path = PROJECT_ROOT / "models"

    # Reports / figures
    REPORTS: Path = PROJECT_ROOT / "reports"
    FIGURES: Path = PROJECT_ROOT / "reports" / "figures"

    # Notebooks
    NOTEBOOKS: Path = PROJECT_ROOT / "notebooks"

    # Frequently-referenced files
    @property
    def MOVIES_CLEAN(self) -> Path:
        return self.PROCESSED_DATA / "movies_clean.csv"

    @property
    def COUNTRY_STATS(self) -> Path:
        return self.PROCESSED_DATA / "country_stats.csv"

    @property
    def TFIDF_MATRIX(self) -> Path:
        return self.MODELS / "tfidf_matrix.pkl"

    @property
    def COSINE_SIM(self) -> Path:
        return self.MODELS / "cosine_sim.pkl"

    @property
    def SVD_MODEL(self) -> Path:
        return self.MODELS / "svd_model.pkl"

    @property
    def CONTENT_INDEX(self) -> Path:
        """Mapping: movie title/imdb_id -> matrix row index."""
        return self.MODELS / "content_index.pkl"

    def ensure_exist(self) -> None:
        """Create all directories if they are missing."""
        for attr_name in (
            "DATA", "RAW_DATA", "PROCESSED_DATA", "EXTERNAL_DATA",
            "MODELS", "REPORTS", "FIGURES", "NOTEBOOKS",
        ):
            getattr(self, attr_name).mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMDB dataset endpoints (public, no auth required)
# =============================================================================
@dataclass(frozen=True)
class IMDBConfig:
    """IMDB official dataset URLs and the local filenames they map to."""

    BASE_URL: str = "https://datasets.imdbws.com"

    # filename -> remote URL is built dynamically; we just list filenames here
    FILES: tuple[str, ...] = (
        "title.basics.tsv.gz",     # tconst, primaryTitle, year, runtime, genres
        "title.ratings.tsv.gz",    # tconst, averageRating, numVotes
        "title.crew.tsv.gz",       # tconst, directors, writers
        "title.principals.tsv.gz", # tconst, ordering, nconst, category, ...
        "name.basics.tsv.gz",      # nconst, primaryName, primaryProfession
    )

    # Restrict to feature films only (saves ~90% of memory)
    TITLE_TYPE_FILTER: tuple[str, ...] = ("movie",)

    # Quality filters applied during cleaning
    MIN_VOTES: int = 1_000           # drop obscure titles
    MIN_RUNTIME_MIN: int = 40        # drop shorts
    MAX_RUNTIME_MIN: int = 300       # drop outliers / mis-tagged content
    MIN_YEAR: int = 1920
    MAX_YEAR: int = 2026


# =============================================================================
# TMDB API
# =============================================================================
@dataclass(frozen=True)
class TMDBConfig:
    """The Movie Database (TMDB) API configuration."""

    API_KEY: str = field(
        default_factory=lambda: os.getenv("TMDB_API_KEY", "").strip()
    )
    BASE_URL: str = "https://api.themoviedb.org/3"
    FIND_ENDPOINT: str = "/find/{imdb_id}"
    MOVIE_ENDPOINT: str = "/movie/{tmdb_id}"

    # Networking
    TIMEOUT_SEC: float = 10.0
    MAX_RETRIES: int = 4
    BACKOFF_FACTOR: float = 1.5    # exponential: 1.5, 2.25, 3.375, ...
    RATE_LIMIT_SLEEP: float = 0.27  # ~3.7 req/s -> safe under 40 req / 10s

    # Local cache to avoid re-fetching the same movie
    CACHE_FILENAME: str = "tmdb_cache.json"

    def is_configured(self) -> bool:
        return bool(self.API_KEY)


# =============================================================================
# Weighted rating (IMDb formula)
# =============================================================================
@dataclass(frozen=True)
class WeightedRatingConfig:
    """
    IMDb-style weighted rating (Bayesian average).

        WR = (v / (v + m)) * R + (m / (v + m)) * C

    Where:
        R = average rating of the movie
        v = number of votes for the movie
        m = minimum votes required (we use the 80th percentile)
        C = mean rating across the whole catalog
    """

    MIN_VOTES_PERCENTILE: float = 0.80   # `m` is set to this quantile of v


# =============================================================================
# MovieLens (collaborative filtering)
# =============================================================================
@dataclass(frozen=True)
class MovieLensConfig:
    """MovieLens 25M dataset configuration."""

    DATASET_URL: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    ARCHIVE_NAME: str = "ml-25m.zip"
    EXTRACTED_DIR: str = "ml-25m"
    RATINGS_FILE: str = "ratings.csv"
    LINKS_FILE: str = "links.csv"        # movieId <-> imdbId / tmdbId
    MOVIES_FILE: str = "movies.csv"

    # Surprise reader expects (rating_min, rating_max)
    RATING_SCALE: tuple[float, float] = (0.5, 5.0)

    # For dev / smoke tests we may sample
    SAMPLE_FRACTION: float | None = None  # None = use all; 0.1 for 10% sample


# =============================================================================
# Model hyperparameters
# =============================================================================
@dataclass(frozen=True)
class ContentModelConfig:
    """TF-IDF + cosine-similarity content recommender."""

    MAX_FEATURES: int = 15_000
    NGRAM_RANGE: tuple[int, int] = (1, 2)
    MIN_DF: int = 2
    MAX_DF: float = 0.85
    STOP_WORDS: str = "english"

    # When two cosine scores are very close, break ties by weighted rating
    RERANK_BY_WEIGHTED_RATING: bool = True
    DEFAULT_TOP_K: int = 10


@dataclass(frozen=True)
class SVDConfig:
    """Surprise SVD hyperparameters."""

    N_FACTORS: int = 100
    N_EPOCHS: int = 20
    LR_ALL: float = 0.005
    REG_ALL: float = 0.02
    CV_FOLDS: int = 5
    TARGET_RMSE: float = 0.87


@dataclass(frozen=True)
class HybridConfig:
    """Hybrid recommender weighting."""

    ALPHA: float = 0.5                # 0 = collaborative, 1 = content
    DYNAMIC_ALPHA: bool = True        # adjust alpha by user history length
    COLD_START_THRESHOLD: int = 5     # ratings below this -> content-heavy
    DEFAULT_TOP_K: int = 10


# =============================================================================
# Streamlit cache settings
# =============================================================================
@dataclass(frozen=True)
class CacheConfig:
    """Streamlit cache-decorator parameters."""

    TTL_SECONDS: int = 60 * 60 * 24   # 24 h
    MAX_ENTRIES: int = 64
    SHOW_SPINNER: bool = True


# =============================================================================
# Aggregator: a single `settings` object imported everywhere
# =============================================================================
@dataclass(frozen=True)
class Settings:
    PATHS: Paths = field(default_factory=Paths)
    IMDB: IMDBConfig = field(default_factory=IMDBConfig)
    TMDB: TMDBConfig = field(default_factory=TMDBConfig)
    WEIGHTED_RATING: WeightedRatingConfig = field(default_factory=WeightedRatingConfig)
    MOVIELENS: MovieLensConfig = field(default_factory=MovieLensConfig)
    CONTENT_MODEL: ContentModelConfig = field(default_factory=ContentModelConfig)
    SVD: SVDConfig = field(default_factory=SVDConfig)
    HYBRID: HybridConfig = field(default_factory=HybridConfig)
    CACHE: CacheConfig = field(default_factory=CacheConfig)
    RANDOM_STATE: int = RANDOM_STATE


settings: Final[Settings] = Settings()


# =============================================================================
# Logging
# =============================================================================
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger.

    Idempotent: calling twice does not duplicate handlers.

    Parameters
    ----------
    level
        Logging level (default: INFO). Use logging.DEBUG for verbose.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers (e.g., when imported by Streamlit on reload)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Quiet down noisy 3rd-party libraries
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# Ensure all required directories exist on import. Cheap and idempotent.
settings.PATHS.ensure_exist()


if __name__ == "__main__":
    # Smoke-test: print the current configuration
    setup_logging()
    log = logging.getLogger("config")
    log.info("Project root: %s", settings.PATHS.ROOT)
    log.info("Random state: %d", settings.RANDOM_STATE)
    log.info("TMDB configured: %s", settings.TMDB.is_configured())
    log.info("IMDB files: %s", ", ".join(settings.IMDB.FILES))
