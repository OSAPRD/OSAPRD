"""Local storage configuration for curation inputs and outputs."""

from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_LOCAL_DATA_FORMATS = {
    "extractionpullrequests",
    "fullpullrequests_sharded",
    "curation_processed",
}
DEFAULT_LOCAL_DATA_FORMAT = "extractionpullrequests"
DEFAULT_LOCAL_DIRECTORIES = (Path("/data/input"),)
DEFAULT_LOCAL_OUTPUT_DIR = Path("/data/output")
DEFAULT_BATCH_SIZE = 100


def _env_first(*names: str) -> str | None:
    """Return the first non-empty environment variable from ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return None


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


def _parse_positive_int(value: str | None, *, default: int, name: str) -> int:
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


def _split_local_directories(raw: str | None) -> list[str]:
    """Split an OS-path-list value into local input roots."""
    if not raw or not raw.strip():
        return [str(path) for path in DEFAULT_LOCAL_DIRECTORIES]
    return [part.strip() for part in raw.split(os.pathsep) if part.strip()]


def load_local_data_format() -> str:
    """Load and validate the local parquet layout name."""
    value = (
        _env_first("CURATION_INPUT_FORMAT", "CURATION_LOCAL_DATA_FORMAT")
        or DEFAULT_LOCAL_DATA_FORMAT
    ).strip().lower()
    if value not in SUPPORTED_LOCAL_DATA_FORMATS:
        allowed = ", ".join(sorted(SUPPORTED_LOCAL_DATA_FORMATS))
        raise ValueError(f"Unsupported curation input format {value!r}. Expected one of: {allowed}.")
    return value


def load_local_directories() -> list[str]:
    """Load roots that will be scanned for local parquet inputs."""
    return _split_local_directories(
        _env_first("CURATION_INPUT_DIRS", "CURATION_INPUT_DIR", "CURATION_LOCAL_DIRECTORIES")
    )


def load_local_output_dir() -> Path:
    """Load the local output root."""
    return Path(_env_first("CURATION_OUTPUT_DIR", "LOCAL_OUTPUT_DIR") or DEFAULT_LOCAL_OUTPUT_DIR)


CURATION_LOCAL_DATA_FORMAT = load_local_data_format()

# Template used only for the legacy local sharded layout. This is a read-only
# compatibility input, not a separate processing mode.
CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE = os.environ.get(
    "CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE",
    "data/{cohort}/FullPullRequests",
).strip()
CURATION_REQUIRE_COMPLETE_SHARD_SET = _parse_bool_env(
    "CURATION_REQUIRE_COMPLETE_SHARD_SET",
    True,
)

LOCAL_DIRECTORIES = load_local_directories()
LOCAL_OUTPUT_DIR = load_local_output_dir()

_SAMPLE_HISTORY_DIR_ENV = _env_first("CURATION_SAMPLE_HISTORY_DIR", "SAMPLE_HISTORY_DIR")
SAMPLE_HISTORY_DIR = Path(_SAMPLE_HISTORY_DIR_ENV) if _SAMPLE_HISTORY_DIR_ENV else None

BATCH_SIZE = _parse_positive_int(
    _env_first("CURATION_BATCH_SIZE", "BATCH_SIZE"),
    default=DEFAULT_BATCH_SIZE,
    name="CURATION_BATCH_SIZE",
)
