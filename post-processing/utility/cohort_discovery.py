"""Shared cohort directory discovery helpers."""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Iterable


def safe_cohort_name(value: object) -> str:
    """Return the curation-style safe cohort directory component."""
    return re.sub(r"[^a-z0-9_.-]", "_", str(value or "").strip().lower())


def cohort_name_matches(path: Path, cohort: object) -> bool:
    """Return whether a path name matches a requested cohort name."""
    requested = str(cohort or "").strip()
    if not requested:
        return True
    name = Path(path).name
    return (
        name == requested
        or name.lower() == requested.lower()
        or safe_cohort_name(name) == safe_cohort_name(requested)
    )


def normalize_exclude_patterns(exclude_dirs: Iterable[str] | None) -> tuple[str, ...]:
    """Return non-empty cohort exclusion patterns."""
    return tuple(str(pattern).strip() for pattern in (exclude_dirs or ()) if str(pattern).strip())


def path_is_excluded(path: Path, root: Path, exclude_patterns: Iterable[str] | None) -> bool:
    """Return True when a path falls under an excluded top-level cohort directory."""
    patterns = normalize_exclude_patterns(exclude_patterns)
    if not patterns:
        return False
    try:
        relative_parts = Path(path).relative_to(root).parts
    except ValueError:
        return False
    if not relative_parts:
        return False
    top_level_name = relative_parts[0]
    return any(
        fnmatch(top_level_name, pattern) or fnmatch(Path(path).name, pattern)
        for pattern in patterns
    )


def discover_cohort_dirs(
    curation_outputs_dir: Path,
    exclude_dirs: Iterable[str] | None = None,
    *,
    include_cohort: object = None,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    """Return eligible top-level cohort directories under a curation outputs root."""
    root = Path(curation_outputs_dir)
    if not root.exists():
        if log is not None:
            log(f"Curation outputs directory does not exist: {root}")
        return []
    exclude_patterns = normalize_exclude_patterns(exclude_dirs)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and cohort_name_matches(path, include_cohort)
        and not path_is_excluded(path, root, exclude_patterns)
    )
