"""Storage and batching defaults for publishing curation parquet data.

This module intentionally contains no credentials. Paths are Docker-friendly by
default and can be overridden through CLI flags or
``POST_PROCESSING_UPLOAD_CURATION_*`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


UPLOAD_CURATION_DIR = Path(__file__).resolve().parents[1]

# Docker defaults mirror the documented mount contract. Local runs should
# usually override these paths through CLI flags rather than editing this file.
DEFAULT_CURATION_OUTPUTS_DIR = Path("/data/input")
DEFAULT_TOPIC_CLASSIFICATION_OUTPUTS_DIR = Path("/data/topic-classification")
DEFAULT_LONGITUDINAL_REFACTORING_OUTPUTS_DIR = Path("/data/longitudinal-refactoring")
DEFAULT_OUTPUT_DIR = Path("/data/output")

# Layout and batching defaults define the public staged package shape. Keep
# these stable unless a schema/layout version bump accompanies the change.
DEFAULT_CURATION_EXCLUDE_DIRS: tuple[str, ...] = ()
DEFAULT_TOPIC_CLASSIFICATION_TOP_K_TOPICS = 5
DEFAULT_DATA_SUBDIR = "data"
DEFAULT_OUTPUT_BATCH_SIZE = 10_000
DEFAULT_MAX_FILES_PER_DIRECTORY = 9_500
DEFAULT_PARQUET_COMPRESSION = "zstd"
DEFAULT_STATE_DB_FILENAME = "upload_curation_state.sqlite3"
DEFAULT_SCHEMA_VERSION = "curation_upload_v2"
DEFAULT_BLOB_BATCH_BYTES = 128 * 1024 * 1024

# Hugging Face repository uploads can hit request and commit quotas. Defaults
# favor conservative, resumable uploads over raw throughput.
DEFAULT_UPLOAD_MAX_RETRIES = 12
DEFAULT_UPLOAD_RETRY_BASE_SECONDS = 5 * 60.0
DEFAULT_UPLOAD_SHORT_TERM_RATE_LIMIT_WINDOW_SECONDS = 5 * 60.0
DEFAULT_UPLOAD_HOURLY_RATE_LIMIT_DELAY_SECONDS = 60 * 60.0
DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_THRESHOLD = 1
DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_DELAY_SECONDS = 5 * 60.0
DEFAULT_UPLOAD_LARGE_FOLDER_NUM_WORKERS = 1
DEFAULT_UPLOAD_LARGE_FOLDER_DIRECTORY_COOLDOWN_SECONDS = 5.0


def split_text_list(value: str | None) -> tuple[str, ...]:
    """Parse comma/semicolon-separated text values from environment variables."""
    if not value:
        return ()
    normalized = value.replace(";", ",")
    return tuple(part.strip() for part in normalized.split(",") if part.strip())


def load_curation_outputs_dir() -> Path:
    """Return the local curation output root to package."""
    return Path(
        os.environ.get(
            "POST_PROCESSING_UPLOAD_CURATION_INPUT_DIR",
            os.environ.get(
                "POST_PROCESSING_CURATION_OUTPUTS_DIR",
                str(DEFAULT_CURATION_OUTPUTS_DIR),
            ),
        )
    ).expanduser()


def load_topic_classification_outputs_dir() -> Path:
    """Return the optional topic-classification output root."""
    return Path(
        os.environ.get(
            "POST_PROCESSING_UPLOAD_CURATION_TOPIC_CLASSIFICATION_DIR",
            os.environ.get(
                "POST_PROCESSING_TOPIC_CLASSIFICATION_OUTPUTS_DIR",
                str(DEFAULT_TOPIC_CLASSIFICATION_OUTPUTS_DIR),
            ),
        )
    ).expanduser()


def load_longitudinal_refactoring_outputs_dir() -> Path:
    """Return the optional longitudinal-refactoring output root."""
    return Path(
        os.environ.get(
            "POST_PROCESSING_UPLOAD_CURATION_LONGITUDINAL_REFACTORING_DIR",
            os.environ.get(
                "POST_PROCESSING_LONGITUDINAL_REFACTORING_OUTPUTS_DIR",
                str(DEFAULT_LONGITUDINAL_REFACTORING_OUTPUTS_DIR),
            ),
        )
    ).expanduser()


def load_output_dir() -> Path:
    """Return the local staging/output directory."""
    return Path(
        os.environ.get(
            "POST_PROCESSING_UPLOAD_CURATION_OUTPUT_DIR",
            os.environ.get(
                "POST_PROCESSING_UPLOAD_CURATION_LOCAL_OUTPUT_DIR",
                str(DEFAULT_OUTPUT_DIR),
            ),
        )
    ).expanduser()


def load_curation_exclude_dirs() -> tuple[str, ...]:
    """Return source subdirectories to exclude while discovering curation runs."""
    return split_text_list(
        os.environ.get("POST_PROCESSING_UPLOAD_CURATION_EXCLUDE_DIRS")
        or os.environ.get("POST_PROCESSING_CURATION_EXCLUDE_DIRS")
    )
