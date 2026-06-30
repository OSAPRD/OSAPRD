"""Configuration for longitudinal refactoring post-processing analysis."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path


def _bool_env(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.environ.get(name, default).split(",") if part.strip()]


def _safe_cohort_name(value: object) -> str:
    return re.sub(r"[^a-z0-9_.-]", "_", str(value or "").strip().lower())


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUN_TIMESTAMP = os.environ.get(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_RUN_TIMESTAMP",
    datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
)
_RUNS_ROOT = Path(
    os.environ.get(
        "POST_PROCESSING_RUNS_ROOT",
        os.environ.get("MOSAIC_RUNS_ROOT", str(_PROJECT_ROOT / "runs")),
    )
)
LONGITUDINAL_REFACTORING_COHORT = (
    os.environ.get(
        "POST_PROCESSING_LONGITUDINAL_REFACTORING_COHORT",
        os.environ.get("COHORT", ""),
    ).strip()
    or None
)
LONGITUDINAL_REFACTORING_INPUT_MODE = os.environ.get(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_INPUT_MODE",
    "stratified_longitudinal",
).strip().lower() or "stratified_longitudinal"


def _default_run_root() -> Path:
    name_parts = ["post_processing_longitudinal_refactoring"]
    if LONGITUDINAL_REFACTORING_INPUT_MODE == "refactoring_enriched":
        name_parts.append("refactoring_enriched")
    if LONGITUDINAL_REFACTORING_COHORT:
        name_parts.append(_safe_cohort_name(LONGITUDINAL_REFACTORING_COHORT))
    name_parts.append(_RUN_TIMESTAMP)
    root = _RUNS_ROOT / "_".join(name_parts)
    if not root.exists():
        return root
    suffix = 1
    while (root.parent / f"{root.name}_{suffix}").exists():
        suffix += 1
    return root.parent / f"{root.name}_{suffix}"


LONGITUDINAL_REFACTORING_RUN_ROOT = Path(
    os.environ.get(
        "POST_PROCESSING_LONGITUDINAL_REFACTORING_RUN_ROOT",
        os.environ.get("RUN_ROOT", str(_default_run_root())),
    )
)

# Curation output root produced by completed curation runs. This can point at a
# directory containing cohort subdirectories or directly at one curation output.
CURATION_OUTPUTS_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_CURATION_OUTPUTS_DIR",
        r"C:\path\to\curation-outputs",
    )
)

# Optional comma-separated subdirectory names or glob patterns to exclude under
# the configured curation outputs root, e.g. "validation_*,legacy_cohort".
CURATION_EXCLUDE_DIRS = _csv_env("POST_PROCESSING_CURATION_EXCLUDE_DIRS", "")

# If false, all candidates from the selected input mode are processed.
LONGITUDINAL_REFACTORING_SAMPLING_ENABLED = _bool_env(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_SAMPLING_ENABLED",
    "true",
)

# Optional target count for curation-style stratified subsampling. A value of 0
# means "process all candidates" even when sampling is enabled.
LONGITUDINAL_REFACTORING_SAMPLE_TARGET = int(
    os.environ.get("POST_PROCESSING_LONGITUDINAL_REFACTORING_SAMPLE_TARGET", "0")
)

# Number of PR records processed per prepared repository batch.
LONGITUDINAL_REFACTORING_BATCH_SIZE = int(
    os.environ.get("POST_PROCESSING_LONGITUDINAL_REFACTORING_BATCH_SIZE", "25")
)

# Concurrent repository processing settings. Each worker prepares one repository
# and processes that repository's longitudinal PRs sequentially.
LONGITUDINAL_REFACTORING_REPO_WORKERS = int(
    os.environ.get("POST_PROCESSING_LONGITUDINAL_REFACTORING_REPO_WORKERS", "4")
)

# Maximum number of repository tasks buffered ahead of workers.
LONGITUDINAL_REFACTORING_PREFETCH_REPOS = int(
    os.environ.get("POST_PROCESSING_LONGITUDINAL_REFACTORING_PREFETCH_REPOS", "4")
)

# Process-wide cap for simultaneous refactoring tool invocations in this
# post-processing run. Set to 0 to disable the gate.
LONGITUDINAL_REFACTORING_TOOL_CONCURRENCY_LIMIT = int(
    os.environ.get("POST_PROCESSING_LONGITUDINAL_REFACTORING_TOOL_CONCURRENCY_LIMIT", "2")
)

# Optional separate cap for C/C++ refactoring tools. Defaults to the general
# refactoring tool limit.
LONGITUDINAL_REFACTORING_CPP_TOOL_CONCURRENCY_LIMIT = int(
    os.environ.get(
        "POST_PROCESSING_LONGITUDINAL_REFACTORING_CPP_TOOL_CONCURRENCY_LIMIT",
        str(LONGITUDINAL_REFACTORING_TOOL_CONCURRENCY_LIMIT),
    )
)

LONGITUDINAL_REFACTORING_SNAPSHOT_LABELS = _csv_env(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_SNAPSHOT_LABELS",
    "+3d,+7d,+31d,+61d",
)

LONGITUDINAL_REFACTORING_OUTPUT_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_LONGITUDINAL_REFACTORING_OUTPUT_DIR",
        str(LONGITUDINAL_REFACTORING_RUN_ROOT / "output"),
    )
)

LONGITUDINAL_REFACTORING_CLONE_ROOT = Path(
    os.environ.get(
        "POST_PROCESSING_LONGITUDINAL_REFACTORING_CLONE_ROOT",
        str(LONGITUDINAL_REFACTORING_RUN_ROOT / "clones"),
    )
)

LONGITUDINAL_REFACTORING_CLEANUP_CLONES = _bool_env(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_CLEANUP_CLONES",
    "true",
)

LONGITUDINAL_REFACTORING_RECONSTRUCT_MISSING_SNAPSHOTS = _bool_env(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_RECONSTRUCT_MISSING_SNAPSHOTS",
    "true",
)

LONGITUDINAL_REFACTORING_HYDRATE_MISSING_FUTURE = _bool_env(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_HYDRATE_MISSING_FUTURE",
    "true",
)

LONGITUDINAL_REFACTORING_GITHUB_TOKENS = _csv_env(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_GITHUB_TOKENS",
    "",
)
