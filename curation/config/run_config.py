"""Runtime knobs for the single-pass curation pipeline.

The top-level CLI resolves :class:`curation.config.settings.CurationSettings`
and writes the relevant values into environment variables before these module
constants are imported. The constants remain here because preprocessing,
sampling, hydration, and metrics modules read them directly.
"""

from __future__ import annotations

import os


def _parse_csv_env(name: str, default: tuple[str, ...]) -> list[str]:
    """Read a comma-separated string list from the environment."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_bool_env(name: str, default: bool) -> bool:
    """Read a common boolean environment value."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


def _parse_int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an integer environment value with a minimum bound."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}.")
    return parsed


COHORT = os.environ.get("CURATION_COHORT", os.environ.get("COHORT", "agentic")).strip().lower()

# Benchmark languages retained during preprocessing. The values are normalized
# later by language-selection helpers, so display casing is kept here.
TARGET_LANGUAGES = _parse_csv_env(
    "CURATION_TARGET_LANGUAGES",
    ("C++", "Java", "JavaScript", "Python"),
)

# Scientific filters used before sampling.
ONLY_MERGED_PRS = _parse_bool_env("ONLY_MERGED_PRS", True)

# Sampling sizes for the main and longitudinal subsets.
TARGET_NO_PRS = _parse_int_env("TARGET_NO_PRS", 50_000)
LONGITUDINAL_TARGET_NO_PRS = _parse_int_env("LONGITUDINAL_TARGET_NO_PRS", 5_000)

# Resume means "skip PRs already recorded as processed in checkpoint/progress
# files". It does not activate a separate processing workflow.
RESUME_PROCESSING = _parse_bool_env("RESUME_PROCESSING", True)

# Operational resume helpers for large interrupted runs.
CURATION_RESUME_FROM_EXISTING_SAMPLE = _parse_bool_env(
    "CURATION_RESUME_FROM_EXISTING_SAMPLE",
    False,
)
CURATION_SKIP_INITIAL_SAMPLE_PROCESSING = _parse_bool_env(
    "CURATION_SKIP_INITIAL_SAMPLE_PROCESSING",
    False,
)

# Snapshot deletion is opt-in because future inspection/debugging often needs
# the hydrated source trees.
DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING = _parse_bool_env(
    "DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING",
    False,
)

# Stratification controls.
TIME_BUCKET_GRANULARITY = os.environ.get("TIME_BUCKET_GRANULARITY", "hour").strip().lower()
POPULARITY_BUCKETS = _parse_int_env("POPULARITY_BUCKETS", 3, minimum=1)

# Repository-level processing concurrency. Each worker owns one repository clone
# and processes that repository's PRs sequentially.
PROCESSING_REPO_WORKERS = _parse_int_env("PROCESSING_REPO_WORKERS", 4, minimum=1)
PROCESSING_PREFETCH_REPOS = _parse_int_env("PROCESSING_PREFETCH_REPOS", 4, minimum=1)

# Two-pass streaming keeps memory bounded: pass one samples lightweight rows,
# pass two materializes full PR DTOs only for selected URLs.
USE_TWO_PASS_STREAMING_SAMPLING = _parse_bool_env("USE_TWO_PASS_STREAMING_SAMPLING", True)
STREAMING_PARQUET_BATCH_SIZE = _parse_int_env("STREAMING_PARQUET_BATCH_SIZE", 25_000, minimum=1)
CURATION_PASS2_URL_CHUNK_SIZE = _parse_int_env("CURATION_PASS2_URL_CHUNK_SIZE", 5_000, minimum=1)
CURATION_PARTITION_WRITE_BUFFER_ROWS = _parse_int_env(
    "CURATION_PARTITION_WRITE_BUFFER_ROWS",
    1_000,
    minimum=1,
)

# External analyzers/miners are often CPU- and IO-heavy. These gates keep one
# container from oversubscribing the host while repository workers are active.
EXTERNAL_TOOL_CONCURRENCY_LIMIT = _parse_int_env("EXTERNAL_TOOL_CONCURRENCY_LIMIT", 1)
CPP_TOOL_CONCURRENCY_LIMIT = _parse_int_env("CPP_TOOL_CONCURRENCY_LIMIT", 1)
