"""
Storage configuration for local extraction outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

# Pure defaults used when no environment variable or CLI override is supplied.
DEFAULT_LOCAL_OUTPUT_DIR = Path("/data")
DEFAULT_BATCH_SIZE = 100

LOCAL_OUTPUT_DIR_ENV_VARS = ("EXTRACTION_LOCAL_OUTPUT_DIR", "LOCAL_OUTPUT_DIR")
BATCH_SIZE_ENV_VARS = ("EXTRACTION_BATCH_SIZE", "BATCH_SIZE")


def _env_first(*names: str) -> str | None:
    """Return the first non-empty environment variable from `names`."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return None


def _parse_positive_int(value: str | None, *, default: int, name: str) -> int:
    """Parse a positive integer environment value with a clear error message."""
    normalized = (value or "").strip()
    if not normalized:
        return int(default)
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1, got {parsed}.")
    return parsed


def load_local_output_dir() -> Path:
    """Load the configured local output root from storage env variables."""
    return Path(_env_first(*LOCAL_OUTPUT_DIR_ENV_VARS) or DEFAULT_LOCAL_OUTPUT_DIR)


def load_batch_size() -> int:
    """Load the configured parquet batch size from storage env variables."""
    return _parse_positive_int(
        _env_first(*BATCH_SIZE_ENV_VARS),
        default=DEFAULT_BATCH_SIZE,
        name=BATCH_SIZE_ENV_VARS[0],
    )

LOCAL_OUTPUT_DIR = load_local_output_dir()
BATCH_SIZE = load_batch_size()
