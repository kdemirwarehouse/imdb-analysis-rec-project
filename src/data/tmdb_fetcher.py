"""
src/data/tmdb_fetcher.py
========================
Enrichment client for **The Movie Database (TMDB) API**.

Given an IMDB ID (``ttXXXXXXX``) this module resolves the TMDB ID, then
fetches the movie details we need for analysis and modeling:

    * production_countries (ISO-3166 alpha-2)
    * budget (USD)
    * revenue (USD)
    * popularity (TMDB internal score)
    * vote_average / vote_count (TMDB community)
    * overview (English plot summary)
    * runtime, release_date, original_language

Why a custom client (instead of ``tmdbsimple``)?
------------------------------------------------
We need **explicit control** over:
* rate limiting (avoid HTTP 429 from TMDB),
* retries with exponential backoff,
* a persistent on-disk cache so re-runs are fast and respectful of the API.

Typical usage
-------------
    >>> from src.data.tmdb_fetcher import TMDBFetcher
    >>> fetcher = TMDBFetcher()
    >>> details = fetcher.fetch_one("tt0111161")        # Shawshank Redemption
    >>> df = fetcher.fetch_many(["tt0111161", "tt0068646"])
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

from config import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------------------------
_USER_AGENT = "movie-recommendation-project/0.1"
# Fields we extract from the TMDB /movie response. Everything else is dropped
# at fetch time to keep the cache small.
_KEEP_FIELDS: tuple[str, ...] = (
    "id",
    "imdb_id",
    "title",
    "original_title",
    "original_language",
    "overview",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "vote_count",
    "runtime",
    "release_date",
    "production_countries",
    "production_companies",
    "spoken_languages",
    "genres",
)


# =============================================================================
# Custom exception
# =============================================================================
class TMDBNotConfiguredError(RuntimeError):
    """Raised when the user attempts a TMDB call without an API key."""


# =============================================================================
# Main fetcher
# =============================================================================
class TMDBFetcher:
    """Thin, cache-backed, retry-aware client for the TMDB v3 API.

    Parameters
    ----------
    cache_path
        Where the JSON cache lives. Defaults to
        ``<EXTERNAL_DATA>/tmdb_cache.json``.
    flush_every
        Save the cache to disk after this many *new* fetches. A crash before
        the next flush only loses up to ``flush_every`` entries.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        *,
        flush_every: int = 50,
    ) -> None:
        if not settings.TMDB.is_configured():
            raise TMDBNotConfiguredError(
                "TMDB_API_KEY is empty. Copy `.env.example` to `.env` and set it."
            )

        self.cache_path: Path = (
            cache_path
            or settings.PATHS.EXTERNAL_DATA / settings.TMDB.CACHE_FILENAME
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._flush_every: int = max(1, flush_every)
        self._unsaved_writes: int = 0

        self._session: requests.Session = self._build_session()
        self._cache: dict[str, dict[str, Any] | None] = self._load_cache()
        self._last_request_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Session / cache lifecycle
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_session() -> requests.Session:
        sess = requests.Session()
        sess.headers.update({"User-Agent": _USER_AGENT, "Accept": "application/json"})
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8)
        sess.mount("https://", adapter)
        return sess

    def _load_cache(self) -> dict[str, dict[str, Any] | None]:
        if not self.cache_path.exists():
            logger.info("No TMDB cache found; starting empty")
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                cache = json.load(fh)
            logger.info("Loaded TMDB cache: %d entries from %s",
                        len(cache), self.cache_path)
            return cache
        except (json.JSONDecodeError, OSError):
            # Corrupt cache → back it up and start fresh.
            backup = self.cache_path.with_suffix(".json.corrupt")
            logger.exception(
                "Corrupt TMDB cache at %s — backing up to %s and starting empty",
                self.cache_path, backup,
            )
            try:
                self.cache_path.replace(backup)
            except OSError:
                pass
            return {}

    def flush(self) -> None:
        """Persist the in-memory cache to disk atomically."""
        tmp = self.cache_path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, ensure_ascii=False)
            tmp.replace(self.cache_path)
            self._unsaved_writes = 0
            logger.debug("Cache flushed (%d entries)", len(self._cache))
        except OSError:
            logger.exception("Failed to flush TMDB cache")

    def __del__(self) -> None:  # pragma: no cover — best-effort persistence
        try:
            if getattr(self, "_unsaved_writes", 0) > 0:
                self.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #
    def _respect_rate_limit(self) -> None:
        """Sleep just enough to stay under TMDB's 40 req / 10 s limit."""
        elapsed = time.monotonic() - self._last_request_ts
        wait = settings.TMDB.RATE_LIMIT_SLEEP - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    # ------------------------------------------------------------------ #
    # Low-level HTTP with retry
    # ------------------------------------------------------------------ #
    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """GET ``{BASE_URL}{path}`` with retry. Returns parsed JSON or None on 404."""
        url = f"{settings.TMDB.BASE_URL}{path}"
        params = {"api_key": settings.TMDB.API_KEY, **(params or {})}

        for attempt in range(1, settings.TMDB.MAX_RETRIES + 2):
            self._respect_rate_limit()
            try:
                resp = self._session.get(
                    url, params=params, timeout=settings.TMDB.TIMEOUT_SEC,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt > settings.TMDB.MAX_RETRIES:
                    logger.error("Network error on %s — giving up: %s", url, exc)
                    raise
                wait = settings.TMDB.BACKOFF_FACTOR ** attempt
                logger.warning("Network error (%s) attempt %d — sleep %.1fs",
                               exc, attempt, wait)
                time.sleep(wait)
                continue

            # --- Status handling ---------------------------------------- #
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    logger.error("Non-JSON 200 response from %s", url)
                    return None

            if resp.status_code == 404:
                # Not an error for our flow — many IMDB IDs simply aren't in TMDB.
                logger.debug("404 (not found): %s", url)
                return None

            if resp.status_code == 401:
                logger.error("TMDB 401 Unauthorized — check TMDB_API_KEY")
                resp.raise_for_status()

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt > settings.TMDB.MAX_RETRIES:
                    logger.error("Giving up on %s after %d attempts (HTTP %d)",
                                 url, attempt, resp.status_code)
                    resp.raise_for_status()
                # Honour Retry-After if present
                retry_after = resp.headers.get("Retry-After")
                wait = (
                    float(retry_after) if retry_after and retry_after.isdigit()
                    else settings.TMDB.BACKOFF_FACTOR ** attempt
                )
                logger.warning("HTTP %d on %s (attempt %d) — sleep %.1fs",
                               resp.status_code, url, attempt, wait)
                time.sleep(wait)
                continue

            # Other 4xx: don't retry
            logger.error("Unexpected HTTP %d on %s — not retrying",
                         resp.status_code, url)
            resp.raise_for_status()

        return None  # pragma: no cover — loop always returns or raises

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_imdb_id(imdb_id: str) -> str:
        """Normalize and validate an IMDB id (``ttXXXXXXX``)."""
        if not isinstance(imdb_id, str):
            raise TypeError(f"imdb_id must be str, got {type(imdb_id)}")
        cleaned = imdb_id.strip().lower()
        if not cleaned.startswith("tt") or not cleaned[2:].isdigit():
            raise ValueError(f"Invalid IMDB id: {imdb_id!r}")
        return cleaned

    def _slim(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep only the fields we care about (saves cache space)."""
        return {k: payload.get(k) for k in _KEEP_FIELDS if k in payload}

    def fetch_one(self, imdb_id: str, *, use_cache: bool = True) -> dict[str, Any] | None:
        """Fetch a single movie by IMDB id. Returns ``None`` if TMDB has no record.

        The result is cached on disk (eventually — see ``flush_every``).
        """
        imdb_id = self._validate_imdb_id(imdb_id)

        if use_cache and imdb_id in self._cache:
            return self._cache[imdb_id]

        # Step 1: resolve TMDB id via /find
        find_path = settings.TMDB.FIND_ENDPOINT.format(imdb_id=imdb_id)
        find_payload = self._request(find_path, params={"external_source": "imdb_id"})
        if not find_payload:
            self._cache[imdb_id] = None
            self._post_write()
            return None

        movie_results = find_payload.get("movie_results") or []
        if not movie_results:
            logger.debug("No movie_results for %s", imdb_id)
            self._cache[imdb_id] = None
            self._post_write()
            return None

        tmdb_id = movie_results[0].get("id")
        if tmdb_id is None:
            self._cache[imdb_id] = None
            self._post_write()
            return None

        # Step 2: fetch full movie details
        movie_path = settings.TMDB.MOVIE_ENDPOINT.format(tmdb_id=tmdb_id)
        movie_payload = self._request(movie_path)
        if not movie_payload:
            self._cache[imdb_id] = None
            self._post_write()
            return None

        slim = self._slim(movie_payload)
        # Ensure the imdb_id is always present (TMDB sometimes omits it)
        slim.setdefault("imdb_id", imdb_id)
        self._cache[imdb_id] = slim
        self._post_write()
        return slim

    def _post_write(self) -> None:
        self._unsaved_writes += 1
        if self._unsaved_writes >= self._flush_every:
            self.flush()

    def fetch_many(
        self,
        imdb_ids: Iterable[str],
        *,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Fetch a batch and return a tidy DataFrame.

        Movies that TMDB has no record of show up as rows with NaNs.
        """
        ids = list(imdb_ids)
        records: list[dict[str, Any]] = []
        iterable = tqdm(ids, desc="TMDB fetch", disable=not show_progress)
        for imdb_id in iterable:
            try:
                rec = self.fetch_one(imdb_id, use_cache=use_cache)
            except Exception:
                logger.exception("Unexpected error fetching %s — skipping", imdb_id)
                rec = None
            if rec is None:
                records.append({"imdb_id": imdb_id})
            else:
                records.append(rec)
        # Always flush at the end of a batch.
        self.flush()

        df = pd.DataFrame.from_records(records)
        return self._post_process(df)

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_country_codes(prod_countries: Any) -> str | None:
        """Turn TMDB's list-of-dicts into a comma-joined ISO-2 string.

        TMDB returns e.g.::

            [{"iso_3166_1": "US", "name": "United States of America"}]
        """
        if not isinstance(prod_countries, list) or not prod_countries:
            return None
        codes = [d.get("iso_3166_1") for d in prod_countries if isinstance(d, dict)]
        codes = [c for c in codes if c]
        return ",".join(codes) if codes else None

    @staticmethod
    def _extract_genres(genres: Any) -> str | None:
        if not isinstance(genres, list) or not genres:
            return None
        names = [d.get("name") for d in genres if isinstance(d, dict)]
        names = [n for n in names if n]
        return ",".join(names) if names else None

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested TMDB fields and coerce dtypes for analysis."""
        if df.empty:
            return df
        if "production_countries" in df.columns:
            df["production_countries"] = df["production_countries"].map(
                self._extract_country_codes
            )
        if "genres" in df.columns:
            df["tmdb_genres"] = df["genres"].map(self._extract_genres)
            df = df.drop(columns=["genres"])
        # Numeric coercions (TMDB sometimes returns strings for budget/revenue)
        for col in ("budget", "revenue", "popularity", "vote_average",
                    "vote_count", "runtime"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "release_date" in df.columns:
            df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        return df

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    @property
    def cache_size(self) -> int:
        return len(self._cache)


# =============================================================================
# CLI entrypoint — small smoke test
# =============================================================================
def _smoke_test() -> None:
    """Hit TMDB for a few well-known IMDB ids."""
    from config import setup_logging

    setup_logging()

    if not settings.TMDB.is_configured():
        logger.error("TMDB_API_KEY missing — aborting smoke test")
        return

    sample = [
        "tt0111161",  # The Shawshank Redemption
        "tt0068646",  # The Godfather
        "tt0468569",  # The Dark Knight
        "tt9999999",  # very likely missing
    ]
    fetcher = TMDBFetcher()
    df = fetcher.fetch_many(sample)
    logger.info("Fetched %d rows; cache now holds %d entries",
                len(df), fetcher.cache_size)
    logger.info("\n%s", df[["imdb_id", "title", "production_countries",
                            "budget", "revenue", "popularity"]].to_string())


if __name__ == "__main__":
    _smoke_test()
