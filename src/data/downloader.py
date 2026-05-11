"""
src/data/downloader.py
======================
Streaming, idempotent downloader for IMDB and MovieLens raw datasets.

Features
--------
* Streaming download (constant RAM regardless of file size).
* Skip-if-exists: a second run does nothing unless `force=True`.
* Atomic writes (write to ``<file>.part``, rename on success).
* Retry with exponential backoff on transient errors.
* Optional decompression of ``.gz`` (single file) and ``.zip`` (archive).
* Progress bar via tqdm.

Typical usage (CLI)
-------------------
    $ python -m src.data.downloader

Typical usage (Python)
----------------------
    >>> from src.data.downloader import IMDBDownloader, MovieLensDownloader
    >>> IMDBDownloader().download_all()
    >>> MovieLensDownloader().download_and_extract()
"""

from __future__ import annotations

import gzip
import logging
import shutil
import time
import zipfile
from pathlib import Path
from typing import Final

import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

from config import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------
_CHUNK_SIZE: Final[int] = 64 * 1024          # 64 KiB
_REQUEST_TIMEOUT: Final[float] = 30.0        # seconds — TCP read timeout
_MAX_RETRIES: Final[int] = 4
_BACKOFF_FACTOR: Final[float] = 1.5
_USER_AGENT: Final[str] = (
    "movie-recommendation-project/0.1 (+https://github.com/)"
)


# =============================================================================
# Generic HTTP downloader
# =============================================================================
def _build_session() -> requests.Session:
    """Create a ``requests.Session`` with a sane User-Agent and adapter."""
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _stream_download(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    timeout: float = _REQUEST_TIMEOUT,
    chunk_size: int = _CHUNK_SIZE,
    description: str | None = None,
) -> None:
    """Stream the body of ``url`` into ``dest`` atomically.

    Parameters
    ----------
    session
        Pre-configured requests session.
    url
        HTTP(S) URL to fetch.
    dest
        Final destination path. A sibling ``<dest>.part`` is used during the
        download and renamed on success.
    timeout
        Per-request socket timeout in seconds.
    chunk_size
        Size of each streamed chunk in bytes.
    description
        Label shown in the tqdm progress bar.

    Raises
    ------
    requests.HTTPError
        On non-2xx responses.
    OSError
        On filesystem errors.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    logger.info("Downloading %s -> %s", url, dest)
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("Content-Length", 0)) or None

        with (
            open(tmp_path, "wb") as fh,
            tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=description or dest.name,
                leave=False,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                pbar.update(len(chunk))

    # Atomic move — readers never observe a half-written file.
    tmp_path.replace(dest)
    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def _download_with_retry(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    max_retries: int = _MAX_RETRIES,
    backoff_factor: float = _BACKOFF_FACTOR,
    description: str | None = None,
) -> None:
    """Wrap ``_stream_download`` with exponential-backoff retries.

    Retries are attempted on:
    * Connection errors / timeouts
    * 5xx server errors

    Permanent failures (4xx other than 429) are re-raised immediately.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            _stream_download(session, url, dest, description=description)
            return
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            transient = status == 429 or (status is not None and status >= 500)
            if not transient or attempt > max_retries:
                logger.error("HTTPError on %s (status=%s) — giving up", url, status)
                raise
            wait = backoff_factor ** attempt
            logger.warning(
                "HTTP %s on %s (attempt %d/%d) — retrying in %.1fs",
                status, url, attempt, max_retries, wait,
            )
            time.sleep(wait)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt > max_retries:
                logger.error("Network error on %s — giving up: %s", url, exc)
                raise
            wait = backoff_factor ** attempt
            logger.warning(
                "Network error on %s (attempt %d/%d): %s — retrying in %.1fs",
                url, attempt, max_retries, exc, wait,
            )
            time.sleep(wait)


# =============================================================================
# IMDB downloader
# =============================================================================
class IMDBDownloader:
    """Download all IMDB ``.tsv.gz`` files listed in :class:`IMDBConfig`.

    The files are kept in their compressed form — pandas can read ``.tsv.gz``
    directly, so we skip extraction (saves ~3-4x disk space).
    """

    def __init__(self, dest_dir: Path | None = None) -> None:
        self.dest_dir: Path = dest_dir or settings.PATHS.RAW_DATA
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self._session = _build_session()

    def _remote_url(self, filename: str) -> str:
        return f"{settings.IMDB.BASE_URL}/{filename}"

    def _is_already_downloaded(self, path: Path, min_size_bytes: int = 1024) -> bool:
        """Return True if the file exists and is at least ``min_size_bytes`` big.

        The size check protects against zero-byte / truncated leftovers.
        """
        return path.exists() and path.stat().st_size >= min_size_bytes

    def download_all(self, *, force: bool = False) -> dict[str, Path]:
        """Download every IMDB file. Returns a mapping ``filename -> local path``.

        Parameters
        ----------
        force
            If True, re-download even if the local file already exists.
        """
        downloaded: dict[str, Path] = {}
        for filename in settings.IMDB.FILES:
            dest = self.dest_dir / filename
            if not force and self._is_already_downloaded(dest):
                logger.info("Skipping %s — already present", filename)
                downloaded[filename] = dest
                continue
            try:
                _download_with_retry(
                    self._session,
                    self._remote_url(filename),
                    dest,
                    description=filename,
                )
                downloaded[filename] = dest
            except Exception:
                logger.exception("Failed to download %s", filename)
                raise
        logger.info("IMDB download complete: %d files in %s",
                    len(downloaded), self.dest_dir)
        return downloaded


# =============================================================================
# MovieLens downloader (with zip extraction)
# =============================================================================
class MovieLensDownloader:
    """Download and extract the MovieLens 25M archive.

    After extraction the relevant CSV files live in
    ``<RAW_DATA>/ml-25m/{ratings,links,movies}.csv``.
    """

    def __init__(self, dest_dir: Path | None = None) -> None:
        self.dest_dir: Path = dest_dir or settings.PATHS.RAW_DATA
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self._session = _build_session()

    @property
    def archive_path(self) -> Path:
        return self.dest_dir / settings.MOVIELENS.ARCHIVE_NAME

    @property
    def extracted_dir(self) -> Path:
        return self.dest_dir / settings.MOVIELENS.EXTRACTED_DIR

    def _is_extracted(self) -> bool:
        required = (
            settings.MOVIELENS.RATINGS_FILE,
            settings.MOVIELENS.LINKS_FILE,
            settings.MOVIELENS.MOVIES_FILE,
        )
        return all((self.extracted_dir / f).exists() for f in required)

    def download_and_extract(self, *, force: bool = False) -> Path:
        """Download (if needed) and extract the MovieLens archive.

        Returns
        -------
        Path
            The directory containing the extracted CSV files.
        """
        # ---------- download ----------
        if force or not self.archive_path.exists() or self.archive_path.stat().st_size < 1024:
            try:
                _download_with_retry(
                    self._session,
                    settings.MOVIELENS.DATASET_URL,
                    self.archive_path,
                    description=settings.MOVIELENS.ARCHIVE_NAME,
                )
            except Exception:
                logger.exception("MovieLens download failed")
                raise
        else:
            logger.info("MovieLens archive already present — skipping download")

        # ---------- extract ----------
        if not force and self._is_extracted():
            logger.info("MovieLens already extracted at %s — skipping", self.extracted_dir)
            return self.extracted_dir

        logger.info("Extracting %s -> %s", self.archive_path.name, self.dest_dir)
        try:
            with zipfile.ZipFile(self.archive_path, "r") as zf:
                zf.extractall(self.dest_dir)
        except zipfile.BadZipFile:
            logger.exception("Corrupt zip — deleting %s and retry next run",
                             self.archive_path)
            self.archive_path.unlink(missing_ok=True)
            raise

        if not self._is_extracted():
            raise RuntimeError(
                f"Extraction completed but expected files missing in "
                f"{self.extracted_dir}"
            )
        logger.info("MovieLens ready at %s", self.extracted_dir)
        return self.extracted_dir


# =============================================================================
# Optional helper: decompress a single .gz file (kept for completeness)
# =============================================================================
def decompress_gz(src: Path, dest: Path | None = None, *, force: bool = False) -> Path:
    """Decompress a ``.gz`` file. Returns the path of the decompressed output.

    By default writes alongside ``src`` with the ``.gz`` suffix removed.
    """
    if src.suffix != ".gz":
        raise ValueError(f"Expected a .gz file, got: {src}")
    dest = dest or src.with_suffix("")
    if dest.exists() and not force:
        logger.info("Skipping decompression — %s already exists", dest)
        return dest
    logger.info("Decompressing %s -> %s", src.name, dest.name)
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=_CHUNK_SIZE)
    return dest


# =============================================================================
# CLI entrypoint
# =============================================================================
def main() -> None:
    """Run both downloaders end-to-end."""
    from config import setup_logging

    setup_logging()
    logger.info("=== Starting raw-data download ===")
    IMDBDownloader().download_all()
    MovieLensDownloader().download_and_extract()
    logger.info("=== Raw-data download finished ===")


if __name__ == "__main__":
    main()
