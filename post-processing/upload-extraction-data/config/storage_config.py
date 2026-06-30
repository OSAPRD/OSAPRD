"""Storage defaults for publishing extraction parquet data.

This module contains filesystem and batching defaults only. Sensitive upload
credentials live in ``tokens_config.py`` and are read from environment
variables at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path


UPLOAD_EXTRACTION_DIR = Path(__file__).resolve().parents[1]

# Docker mounts use these defaults. Local runs can override them through CLI
# flags or POST_PROCESSING_UPLOAD_EXTRACTION_* environment variables.
DEFAULT_SOURCE_DIRS = (Path("/data/input"),)
DEFAULT_OUTPUT_DIR = Path("/data/output")

# Keep the staged data subtree aligned with extraction's local output root. The
# uploader reads from extraction/data/<cohort> and writes data/<cohort>/<entity>.
DEFAULT_DATA_SUBDIR = "data"
DEFAULT_OUTPUT_BATCH_SIZE = 10_000
DEFAULT_MAX_FILES_PER_DIRECTORY = 9_500
DEFAULT_PARQUET_COMPRESSION = "zstd"
DEFAULT_STATE_DB_FILENAME = "upload_extraction_state.sqlite3"
DEFAULT_SCHEMA_VERSION = "pull_request_standardized_v12"

# Hugging Face commit/upload retry settings. The long delays are intentional:
# dataset repository commits can hit short-term and hourly service quotas.
DEFAULT_UPLOAD_MAX_RETRIES = 12
DEFAULT_UPLOAD_RETRY_BASE_SECONDS = 5 * 60.0
DEFAULT_UPLOAD_SHORT_TERM_RATE_LIMIT_WINDOW_SECONDS = 5 * 60.0
DEFAULT_UPLOAD_HOURLY_RATE_LIMIT_DELAY_SECONDS = 60 * 60.0
DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_THRESHOLD = 1
DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_DELAY_SECONDS = 5 * 60.0
DEFAULT_UPLOAD_LARGE_FOLDER_NUM_WORKERS = 1
DEFAULT_UPLOAD_LARGE_FOLDER_DIRECTORY_COOLDOWN_SECONDS = 5.0


def split_path_list(value: str | None) -> tuple[Path, ...]:
    """Parse an OS-path-separated list of input roots.

    The separator is platform-specific (`;` on Windows, `:` on Unix), matching
    how Python and shells represent path-list environment variables.
    """
    if not value:
        return ()
    paths = tuple(
        Path(part.strip()).expanduser()
        for part in value.split(os.pathsep)
        if part.strip()
    )
    return paths


def load_source_dirs() -> tuple[Path, ...]:
    """Return source roots from env aliases or the Docker-friendly default.

    ``POST_PROCESSING_LOCAL_DIRECTORIES`` remains accepted as a read-only alias
    for older post-processing command lines, but new runs should use the
    upload-extraction-specific variable.
    """
    configured = split_path_list(
        os.environ.get("POST_PROCESSING_UPLOAD_EXTRACTION_SOURCE_DIRS")
        or os.environ.get("POST_PROCESSING_UPLOAD_EXTRACTION_SOURCE_DIR")
        or os.environ.get("POST_PROCESSING_LOCAL_DIRECTORIES")
    )
    return configured or DEFAULT_SOURCE_DIRS


def load_output_dir() -> Path:
    """Return the local staging/output directory."""
    return Path(
        os.environ.get(
            "POST_PROCESSING_UPLOAD_EXTRACTION_OUTPUT_DIR",
            str(DEFAULT_OUTPUT_DIR),
        )
    ).expanduser()
