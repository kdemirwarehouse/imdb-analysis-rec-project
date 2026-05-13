"""
src/data/preprocessor.py
========================
Movie-data preprocessing pipeline.

What this module does — in plain English
----------------------------------------
We have three raw data sources after Step 2:

  1. IMDB ``title.basics.tsv.gz``    — title, year, runtime, genres
  2. IMDB ``title.ratings.tsv.gz``   — averageRating, numVotes
  3. IMDB ``title.crew.tsv.gz``      — directors (IDs)
  4. IMDB ``name.basics.tsv.gz``     — director ID -> name lookup
  5. (Optional) TMDB cache JSON       — countries, budget, revenue, overview

The :class:`MovieDataPreprocessor` glues all of those together into a single
clean DataFrame and writes it to ``data/processed/movies_clean.csv``.

Every step logs ``before`` / ``after`` row counts so we can *prove* that
nulls were removed, duplicates dropped, etc. — see ``quality_report()``.

Pipeline shape
--------------
::

    load_all()         -> populates self._basics, self._ratings, ...
    merge_all()        -> self.df_  (rows after inner join with ratings)
    clean_nulls()
    drop_duplicates()
    parse_genres()
    enrich_with_directors()
    enrich_with_tmdb()   (optional)
    compute_weighted_rating()
    validate()
    save()

Each step is idempotent and prints how many rows it removed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: a small change-log so we can prove cleaning happened
# =============================================================================
@dataclass
class StepLog:
    """A single step's before/after row count, kept for the quality report."""
    step: str
    rows_before: int
    rows_after: int
    note: str = ""

    @property
    def rows_dropped(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def pct_dropped(self) -> float:
        if self.rows_before == 0:
            return 0.0
        return 100 * self.rows_dropped / self.rows_before


# =============================================================================
# Main pipeline
# =============================================================================
class MovieDataPreprocessor:
    """End-to-end pipeline from raw IMDB/TMDB files to ``movies_clean.csv``.

    Parameters
    ----------
    raw_dir
        Directory holding the raw ``.tsv.gz`` files. Defaults to the path in
        :data:`config.settings.PATHS.RAW_DATA`.
    tmdb_cache_path
        Optional path to the TMDB cache JSON written by ``TMDBFetcher``.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        raw_dir: Path | None = None,
        tmdb_cache_path: Path | None = None,
    ) -> None:
        self.raw_dir: Path = raw_dir or settings.PATHS.RAW_DATA
        self.tmdb_cache_path: Path = (
            tmdb_cache_path
            or settings.PATHS.EXTERNAL_DATA / settings.TMDB.CACHE_FILENAME
        )

        # Loaded raw frames (filled by load_*)
        self._basics: pd.DataFrame | None = None
        self._ratings: pd.DataFrame | None = None
        self._crew: pd.DataFrame | None = None
        self._names: pd.DataFrame | None = None
        self._tmdb: pd.DataFrame | None = None

        # Working frame (filled by merge_all and progressively cleaned)
        self.df_: pd.DataFrame | None = None

        # Audit trail
        self.step_logs: list[StepLog] = []

    # ------------------------------------------------------------------ #
    # Tiny utility: log + record a step
    # ------------------------------------------------------------------ #
    def _log_step(self, step: str, before: int, after: int, note: str = "") -> None:
        entry = StepLog(step, before, after, note)
        self.step_logs.append(entry)
        dropped = entry.rows_dropped
        pct = entry.pct_dropped
        suffix = f" — {note}" if note else ""
        logger.info(
            "[%s] %s → %s  (dropped %s, %.2f%%)%s",
            step, f"{before:>9,}", f"{after:>9,}", f"{dropped:>7,}", pct, suffix,
        )

    # ================================================================== #
    # 1. LOADERS — each returns nothing; results stored on self.
    # ================================================================== #
    def load_imdb_basics(self) -> pd.DataFrame:
        """Load ``title.basics.tsv.gz`` and keep only feature films.

        We apply the year / runtime / titleType filters from
        :class:`config.IMDBConfig` here, since they shrink the frame ~20x.
        """
        path = self.raw_dir / "title.basics.tsv.gz"
        usecols = [
            "tconst", "titleType", "primaryTitle", "originalTitle",
            "isAdult", "startYear", "runtimeMinutes", "genres",
        ]
        dtypes = {
            "tconst": "string", "titleType": "category",
            "primaryTitle": "string", "originalTitle": "string",
            "isAdult": "string", "startYear": "string",
            "runtimeMinutes": "string", "genres": "string",
        }
        logger.info("Loading %s", path.name)
        df = pd.read_csv(
            path, sep="\t", na_values="\\N",
            usecols=usecols, dtype=dtypes, low_memory=False,
        )
        rows_in = len(df)

        # Numeric coercions
        df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
        df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
        df["isAdult"] = (
            pd.to_numeric(df["isAdult"], errors="coerce")
            .fillna(0).astype("int8")
        )

        # Apply quality / type filter
        cfg = settings.IMDB
        mask = (
            df["titleType"].isin(cfg.TITLE_TYPE_FILTER)
            & df["startYear"].between(cfg.MIN_YEAR, cfg.MAX_YEAR)
            & df["runtimeMinutes"].between(cfg.MIN_RUNTIME_MIN, cfg.MAX_RUNTIME_MIN)
            & (df["isAdult"] == 0)
        )
        df = df.loc[mask].drop(columns=["titleType", "isAdult"]).reset_index(drop=True)
        self._log_step("load_imdb_basics", rows_in, len(df),
                       note="kept feature films within year/runtime bounds")
        self._basics = df
        return df

    def load_imdb_ratings(self) -> pd.DataFrame:
        """Load ``title.ratings.tsv.gz``."""
        path = self.raw_dir / "title.ratings.tsv.gz"
        logger.info("Loading %s", path.name)
        df = pd.read_csv(
            path, sep="\t", na_values="\\N",
            dtype={"tconst": "string",
                   "averageRating": "float32", "numVotes": "int32"},
        )
        self._log_step("load_imdb_ratings", len(df), len(df), note="raw load")
        self._ratings = df
        return df

    def load_imdb_crew(self) -> pd.DataFrame:
        """Load ``title.crew.tsv.gz`` (keep ``directors`` only)."""
        path = self.raw_dir / "title.crew.tsv.gz"
        logger.info("Loading %s", path.name)
        df = pd.read_csv(
            path, sep="\t", na_values="\\N",
            usecols=["tconst", "directors"],
            dtype={"tconst": "string", "directors": "string"},
        )
        self._log_step("load_imdb_crew", len(df), len(df), note="raw load")
        self._crew = df
        return df

    def load_imdb_names(self) -> pd.DataFrame:
        """Load ``name.basics.tsv.gz`` (only ``nconst`` → ``primaryName``)."""
        path = self.raw_dir / "name.basics.tsv.gz"
        logger.info("Loading %s", path.name)
        df = pd.read_csv(
            path, sep="\t", na_values="\\N",
            usecols=["nconst", "primaryName"],
            dtype={"nconst": "string", "primaryName": "string"},
        )
        self._log_step("load_imdb_names", len(df), len(df), note="raw load")
        self._names = df
        return df

    def load_tmdb_cache(self) -> pd.DataFrame:
        """Load TMDB enrichment from the JSON cache (skips silently if absent)."""
        if not self.tmdb_cache_path.exists():
            logger.warning(
                "TMDB cache not found at %s — TMDB columns will be missing",
                self.tmdb_cache_path,
            )
            self._tmdb = pd.DataFrame()
            return self._tmdb
        try:
            with open(self.tmdb_cache_path, "r", encoding="utf-8") as fh:
                cache: dict[str, dict[str, Any] | None] = json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read TMDB cache — proceeding without it")
            self._tmdb = pd.DataFrame()
            return self._tmdb

        records = [v for v in cache.values() if v is not None]
        df = pd.DataFrame.from_records(records)
        # Helper: nested countries / genres flatten (mirrors TMDBFetcher logic)
        def _codes(x: Any) -> str | None:
            if not isinstance(x, list) or not x:
                return None
            return ",".join(d["iso_3166_1"] for d in x if d.get("iso_3166_1")) or None
        def _gnames(x: Any) -> str | None:
            if not isinstance(x, list) or not x:
                return None
            return ",".join(d["name"] for d in x if d.get("name")) or None
        if "production_countries" in df:
            df["production_countries"] = df["production_countries"].map(_codes)
        if "genres" in df:
            df["tmdb_genres"] = df["genres"].map(_gnames)
            df = df.drop(columns=["genres"])
        # Numeric coercions
        for c in ("budget", "revenue", "popularity",
                  "vote_average", "vote_count", "runtime"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        self._log_step("load_tmdb_cache", len(cache), len(df),
                       note=f"{len(cache) - len(df)} cache entries were 404/None")
        self._tmdb = df
        return df

    def load_all(self, *, with_tmdb: bool = True) -> None:
        """Convenience: run every loader."""
        self.load_imdb_basics()
        self.load_imdb_ratings()
        self.load_imdb_crew()
        self.load_imdb_names()
        if with_tmdb:
            self.load_tmdb_cache()

    # ================================================================== #
    # 2. MERGE
    # ================================================================== #
    def merge_all(self) -> pd.DataFrame:
        """Inner-join basics + ratings (mandatory), left-join everything else."""
        if self._basics is None or self._ratings is None:
            raise RuntimeError("Call load_all() first.")

        rows_in = len(self._basics)
        df = self._basics.merge(self._ratings, on="tconst", how="inner")
        self._log_step("merge_ratings", rows_in, len(df),
                       note="inner join on tconst")

        # Apply min-votes filter (ratings statistically unreliable below this)
        before = len(df)
        df = df.loc[df["numVotes"] >= settings.IMDB.MIN_VOTES].reset_index(drop=True)
        self._log_step("filter_min_votes", before, len(df),
                       note=f"numVotes >= {settings.IMDB.MIN_VOTES}")

        # Optional: crew (left join — never drops rows)
        if self._crew is not None:
            df = df.merge(self._crew, on="tconst", how="left")
            self._log_step("merge_crew", len(df), len(df), note="left join")

        # Optional: TMDB (left join)
        if self._tmdb is not None and not self._tmdb.empty:
            tmdb_cols = [
                c for c in (
                    "imdb_id", "production_countries", "budget", "revenue",
                    "popularity", "overview", "original_language", "tmdb_genres",
                ) if c in self._tmdb.columns
            ]
            df = df.merge(
                self._tmdb[tmdb_cols].rename(columns={"imdb_id": "tconst"}),
                on="tconst", how="left",
            )
            self._log_step("merge_tmdb", len(df), len(df), note="left join")

        self.df_ = df
        return df

    # ================================================================== #
    # 3. CLEAN — nulls, duplicates, parse, enrich, score
    # ================================================================== #
    def clean_nulls(self) -> pd.DataFrame:
        """Drop rows missing values in *critical* columns.

        ``primaryTitle``, ``startYear``, ``runtimeMinutes``, ``averageRating``
        and ``numVotes`` are required. Optional columns (overview, budget,
        directors, etc.) can stay NaN.
        """
        df = self._require_df()
        critical = ["primaryTitle", "startYear", "runtimeMinutes",
                    "averageRating", "numVotes"]
        before = len(df)
        df = df.dropna(subset=critical).reset_index(drop=True)
        self._log_step("drop_nulls_critical", before, len(df),
                       note=f"required: {critical}")
        self.df_ = df
        return df

    def drop_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows (by ``tconst`` — IMDB's primary key)."""
        df = self._require_df()
        before = len(df)
        df = df.drop_duplicates(subset=["tconst"], keep="first").reset_index(drop=True)
        self._log_step("drop_duplicates", before, len(df), note="on tconst")
        self.df_ = df
        return df

    def parse_genres(self) -> pd.DataFrame:
        """Turn ``genres`` string into a list column ``genres_list``.

        IMDB stores genres as a comma-separated string: ``"Drama,Crime,Thriller"``.
        We keep both forms — the string is CSV-friendly, the list is analysis-friendly.
        """
        df = self._require_df()
        before = len(df)
        df["genres_list"] = (
            df["genres"]
            .fillna("")
            .str.split(",")
            .map(lambda xs: [x.strip() for x in xs if x.strip()] or [])
        )
        df["n_genres"] = df["genres_list"].map(len).astype("int8")
        self._log_step("parse_genres", before, len(df),
                       note="added genres_list, n_genres")
        self.df_ = df
        return df

    def enrich_with_directors(self) -> pd.DataFrame:
        """Resolve the first listed director's ``nconst`` to a human name.

        We deliberately keep only the FIRST director (covers ~95% of titles
        and avoids exploding the row count).
        """
        df = self._require_df()
        before = len(df)

        if "directors" not in df.columns or self._names is None:
            logger.warning("Director enrichment skipped — missing crew or names")
            df["director_name"] = pd.NA
        else:
            # Take first nconst from the comma-separated directors field
            first_director = (
                df["directors"]
                .fillna("")
                .str.split(",").str[0]
                .replace({"": pd.NA})
            )
            name_lookup = self._names.set_index("nconst")["primaryName"]
            df["director_name"] = first_director.map(name_lookup)
        self._log_step("enrich_with_directors", before, len(df),
                       note="added director_name")
        self.df_ = df
        return df

    def compute_weighted_rating(self) -> pd.DataFrame:
        """Add the IMDb-style Bayesian weighted rating, ``weighted_rating``.

        Formula::

            WR = v/(v+m) * R + m/(v+m) * C

        Where R = averageRating, v = numVotes,
        m = numVotes at MIN_VOTES_PERCENTILE, C = mean averageRating.
        """
        df = self._require_df()
        v = df["numVotes"].astype("float64")
        R = df["averageRating"].astype("float64")
        m = float(np.quantile(v, settings.WEIGHTED_RATING.MIN_VOTES_PERCENTILE))
        C = float(R.mean())

        df["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C
        df["weighted_rating"] = df["weighted_rating"].round(4)

        self._log_step(
            "compute_weighted_rating", len(df), len(df),
            note=f"m={m:.0f} (p{int(settings.WEIGHTED_RATING.MIN_VOTES_PERCENTILE*100)}), C={C:.3f}",
        )
        self.df_ = df
        return df

    # ================================================================== #
    # 4. VALIDATION
    # ================================================================== #
    def validate(self) -> dict[str, Any]:
        """Run sanity checks. Returns a dict of metrics; raises on hard fails."""
        df = self._require_df()

        problems: list[str] = []

        # Hard requirements
        for col in ("tconst", "primaryTitle", "startYear",
                    "runtimeMinutes", "averageRating", "numVotes",
                    "weighted_rating"):
            if col not in df.columns:
                problems.append(f"missing required column: {col}")
            elif df[col].isna().any():
                n_bad = int(df[col].isna().sum())
                problems.append(f"{n_bad} NaNs found in required column {col}")

        # Range checks
        if (df["startYear"] < 1880).any() or (df["startYear"] > 2030).any():
            problems.append("startYear out of plausible range")
        if (df["averageRating"] < 0).any() or (df["averageRating"] > 10).any():
            problems.append("averageRating out of [0, 10]")
        if (df["weighted_rating"] < 0).any() or (df["weighted_rating"] > 10).any():
            problems.append("weighted_rating out of [0, 10]")
        if df["tconst"].duplicated().any():
            problems.append("duplicate tconst values remain")

        metrics = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_unique_titles": int(df["tconst"].nunique()),
            "year_range": [int(df["startYear"].min()), int(df["startYear"].max())],
            "avg_rating_range": [float(df["averageRating"].min()),
                                 float(df["averageRating"].max())],
            "weighted_rating_range": [float(df["weighted_rating"].min()),
                                      float(df["weighted_rating"].max())],
            "problems": problems,
        }
        if problems:
            for p in problems:
                logger.error("VALIDATION: %s", p)
            raise ValueError(f"Validation failed: {problems}")
        logger.info("Validation passed: %d rows, %d cols", metrics["n_rows"], metrics["n_cols"])
        return metrics

    # ================================================================== #
    # 5. SAVE + QUALITY REPORT
    # ================================================================== #
    def save(self, path: Path | None = None) -> Path:
        """Write the cleaned DataFrame to CSV (lists become pipe-joined strings)."""
        df = self._require_df()
        path = path or settings.PATHS.MOVIES_CLEAN

        # CSV can't store Python lists. Persist a string form alongside.
        df_to_save = df.copy()
        if "genres_list" in df_to_save:
            df_to_save["genres_list"] = df_to_save["genres_list"].map(
                lambda lst: "|".join(lst) if isinstance(lst, list) else ""
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        df_to_save.to_csv(path, index=False)
        logger.info("Wrote %s (%.1f MB, %d rows)",
                    path, path.stat().st_size / 1e6, len(df_to_save))
        return path

    def quality_report(self) -> pd.DataFrame:
        """A tabular before/after of every cleaning step."""
        if not self.step_logs:
            return pd.DataFrame()
        rows = [
            {
                "step": s.step,
                "rows_before": s.rows_before,
                "rows_after": s.rows_after,
                "rows_dropped": s.rows_dropped,
                "pct_dropped": round(s.pct_dropped, 3),
                "note": s.note,
            }
            for s in self.step_logs
        ]
        return pd.DataFrame(rows)

    def null_report(self) -> pd.DataFrame:
        """Per-column null counts on the final frame.

        Note: columns of unhashable types (e.g. Python lists in
        ``genres_list``) cannot be passed to ``Series.nunique``; for those
        we report ``n_unique = NaN``.
        """
        df = self._require_df()

        # Compute nunique only on columns whose first non-null value is hashable.
        nunique_safe: dict[str, int | float] = {}
        for col in df.columns:
            sample = df[col].dropna()
            sample_val = sample.iloc[0] if len(sample) else None
            if isinstance(sample_val, (list, dict, set)):
                nunique_safe[col] = float("nan")
            else:
                try:
                    nunique_safe[col] = int(df[col].nunique(dropna=True))
                except TypeError:
                    nunique_safe[col] = float("nan")

        rep = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "null_count": df.isna().sum(),
            "null_pct": (df.isna().mean() * 100).round(3),
            "n_unique": pd.Series(nunique_safe),
        })
        return rep

    # ================================================================== #
    # 6. Convenience: run the whole pipeline
    # ================================================================== #
    def run(self, *, with_tmdb: bool = True, save: bool = True) -> pd.DataFrame:
        """Execute every step end-to-end. Returns the final clean DataFrame."""
        self.load_all(with_tmdb=with_tmdb)
        self.merge_all()
        self.clean_nulls()
        self.drop_duplicates()
        self.parse_genres()
        self.enrich_with_directors()
        self.compute_weighted_rating()
        self.validate()
        if save:
            self.save()
        return self._require_df()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _require_df(self) -> pd.DataFrame:
        if self.df_ is None:
            raise RuntimeError("Working frame is empty — call merge_all() first.")
        return self.df_


# =============================================================================
# CLI entrypoint
# =============================================================================
def main() -> None:
    from config import setup_logging
    setup_logging()
    logger.info("=== Starting preprocessing ===")
    prep = MovieDataPreprocessor()
    prep.run()
    logger.info("=== Done ===")
    logger.info("\n%s", prep.quality_report().to_string(index=False))


if __name__ == "__main__":
    main()
