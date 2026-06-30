"""Streaming refactoring analysis helpers.

The refactoring analysis reads persisted curation metrics and optionally
re-classifies operation names through the curation taxonomy config. Keeping the
loader local to this module avoids importing curation orchestration code during
analysis-only runs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Iterable


REFACTORING_SUCCESS_STATUS = "success"
MURPHY_HILL_COUNT_SOURCE_STORED = "stored"
MURPHY_HILL_COUNT_SOURCE_TAXONOMY = "taxonomy"
MURPHY_HILL_COUNT_SOURCES = (
    MURPHY_HILL_COUNT_SOURCE_STORED,
    MURPHY_HILL_COUNT_SOURCE_TAXONOMY,
)
MURPHY_HILL_LEVELS = ("low", "medium", "high")
MURPHY_HILL_TAXONOMY_SOURCES = ("exact", "keyword")
UNCLASSIFIED_REFACTORING_TYPE = "unclassified"
REFACTORING_LONGITUDINAL_TIMEPOINTS = ("+3d", "+7d", "+31d", "+61d")
REFACTORING_NUMERIC_METRICS = (
    "RefCount",
    "RefDensity",
    "RefDiversity",
    "RefAdded",
    "RefRemoved",
    "RefMagLines",
)
REFACTORING_DISTRIBUTION_METRICS = (
    "RefCount",
    "RefDensity",
    "RefDiversity",
    "RefMagLines",
)


def normalize_refactoring_tool_status(raw_status: Any) -> str:
    """Normalize a raw refactoring-tool status for coverage diagnostics."""
    status = str(raw_status or "").strip().lower()
    if not status:
        return "missing"
    if status == REFACTORING_SUCCESS_STATUS:
        return "success"
    if "unsupported" in status or status in {"not_supported", "not supported"}:
        return "unsupported"
    if (
        status in {"failed", "failure", "error", "timeout", "timed_out", "crashed"}
        or status.startswith("fail")
        or "failure" in status
        or "error" in status
    ):
        return "failed"
    return "other"


def _is_named_refactoring_type(label: Any) -> bool:
    """Return whether a refactoring label is a usable standardized type."""
    normalized = str(label or "").strip().casefold()
    return bool(normalized) and normalized != UNCLASSIFIED_REFACTORING_TYPE


def _named_refactoring_type_counts(type_counts: dict[str, int]) -> dict[str, int]:
    """Keep positive, standardized RefOp type counts only."""
    return {
        str(label): int(count)
        for label, count in type_counts.items()
        if _is_named_refactoring_type(label) and int(count) > 0
    }


def _validate_murphy_hill_count_source(source: str) -> str:
    """Validate whether Murphy-Hill counts use stored values or taxonomy mapping."""
    normalized = str(source or MURPHY_HILL_COUNT_SOURCE_TAXONOMY).strip().lower()
    if not normalized:
        return MURPHY_HILL_COUNT_SOURCE_TAXONOMY
    if normalized not in MURPHY_HILL_COUNT_SOURCES:
        allowed = ", ".join(MURPHY_HILL_COUNT_SOURCES)
        raise ValueError(f"Murphy-Hill count source must be one of: {allowed}")
    return normalized


def _taxonomy_config_path() -> Path:
    """Return the curation refactoring taxonomy config path."""
    repository_root = Path(__file__).resolve().parents[3]
    return repository_root / "curation" / "config" / "refactoring_taxonomy_config.py"


def _load_refactoring_taxonomy_classifier() -> Callable[[str], dict[str, Any]]:
    """Load the refactoring taxonomy classifier without importing the curation app."""
    taxonomy_path = _taxonomy_config_path()
    if not taxonomy_path.exists():
        raise RuntimeError(
            "Cannot load refactoring taxonomy config: "
            f"{taxonomy_path} does not exist"
        )
    spec = importlib.util.spec_from_file_location(
        "analysis_refactoring_taxonomy_config",
        taxonomy_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot load refactoring taxonomy config: "
            f"{taxonomy_path} is not importable"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classifier = getattr(module, "classify_refactoring_taxonomy", None)
    if not callable(classifier):
        raise RuntimeError(
            "Cannot load refactoring taxonomy config: "
            "classify_refactoring_taxonomy is missing"
        )
    return classifier


def _taxonomy_murphy_hill_level(
    refactoring_type: str,
    classifier: Callable[[str], dict[str, Any]],
) -> str | None:
    """Map a standardized RefOp type to a Murphy-Hill level via taxonomy."""
    taxonomy = classifier(refactoring_type)
    if not isinstance(taxonomy, dict):
        return None
    level = taxonomy.get("murphy_hill_level")
    meta = taxonomy.get("_meta")
    sources = meta.get("sources") if isinstance(meta, dict) else None
    level_source = (
        sources.get("murphy_hill_level") if isinstance(sources, dict) else None
    )
    if level in MURPHY_HILL_LEVELS and level_source in MURPHY_HILL_TAXONOMY_SOURCES:
        return str(level)
    return None


def parse_count_mapping(raw_value: Any) -> dict[str, int]:
    """Parse a JSON/object RefOp count mapping into positive integer counts."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        payload = raw_value
    else:
        raw_text = str(raw_value).strip()
        if not raw_text or raw_text.lower() in {"null", "none"}:
            return {}
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in payload.items():
        label = str(key or "").strip()
        if not label:
            continue
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[label] = counts.get(label, 0) + count
    return counts


def positive_refactor_count(raw_count: Any, type_counts: dict[str, int]) -> int:
    """Return standardized RefOp count from named operation-type counts."""
    del raw_count
    return sum(_named_refactoring_type_counts(type_counts).values())


def compact_refactoring_rows(rows: Iterable[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Convert compact row tuples into dictionaries for legacy callers."""
    compact: list[dict[str, Any]] = []
    for row in rows:
        if len(row) >= 13:
            (
                cohort,
                authorship_group,
                agent_label,
                language,
                refactor_count,
                _refactor_density,
                refactor_diversity,
                refactor_added_lines,
                refactor_removed_lines,
                refactor_magnitude_lines,
                nloc_before,
                refactor_type_count_json,
                refactor_murphyhill_count_json,
            ) = row[:13]
        else:
            (
                cohort,
                authorship_group,
                agent_label,
                language,
                refactor_count,
                _refactor_density,
                refactor_diversity,
                refactor_added_lines,
                refactor_removed_lines,
                refactor_magnitude_lines,
                refactor_type_count_json,
                refactor_murphyhill_count_json,
            ) = row
            nloc_before = None
        type_counts = parse_count_mapping(refactor_type_count_json)
        named_type_counts = _named_refactoring_type_counts(type_counts)
        murphy_hill_counts = parse_count_mapping(refactor_murphyhill_count_json)
        resolved_refactor_count = positive_refactor_count(refactor_count, type_counts)
        ref_density = None
        try:
            resolved_nloc_before = float(nloc_before)
        except (TypeError, ValueError):
            resolved_nloc_before = 0.0
        if resolved_nloc_before > 0:
            ref_density = resolved_refactor_count / (resolved_nloc_before / 1000.0)
        compact.append(
            {
                "cohort": str(cohort),
                "authorship_group": str(authorship_group),
                "agent_label": None if agent_label is None else str(agent_label),
                "language": None if language is None else str(language),
                "RefCount": float(resolved_refactor_count),
                "RefDensity": ref_density,
                "RefDiversity": float(refactor_diversity or 0.0),
                "RefAdded": (
                    float(refactor_added_lines or 0) / resolved_refactor_count
                    if resolved_refactor_count > 0
                    else None
                ),
                "RefRemoved": (
                    float(refactor_removed_lines or 0) / resolved_refactor_count
                    if resolved_refactor_count > 0
                    else None
                ),
                "RefMagLines": (
                    (
                        float(refactor_added_lines or 0)
                        + float(refactor_removed_lines or 0)
                    )
                    / resolved_refactor_count
                    if resolved_refactor_count > 0
                    else None
                ),
                "refactor_type_counts": named_type_counts,
                "murphy_hill_counts": murphy_hill_counts,
            }
        )
    return compact
