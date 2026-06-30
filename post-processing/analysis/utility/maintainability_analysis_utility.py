"""Streaming maintainability analysis helpers.

This module keeps the maintainability pipeline independent from a live curation
runtime. It reads persisted curation parquet fields, loads the smell taxonomy
configuration only when Mantyla-family grouping is requested, and exposes
normalized counters for JSON summaries and plots.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable


MAINTAINABILITY_SUCCESS_STATUS = "success"
UNCLASSIFIED_CODE_SMELL_TYPE = "unclassified"
TOOL_STATUS_BUCKETS = ("success", "unsupported", "failed", "missing", "other")
MANTYLA_COUNT_SOURCE_STORED = "stored"
MANTYLA_COUNT_SOURCE_TAXONOMY = "taxonomy"
MANTYLA_COUNT_SOURCES = (
    MANTYLA_COUNT_SOURCE_STORED,
    MANTYLA_COUNT_SOURCE_TAXONOMY,
)
MANTYLA_CATEGORIES = (
    "bloaters",
    "object_orientation_abusers",
    "change_preventers",
    "dispensables",
    "encapsulators",
    "couplers",
    "others",
    "unmapped",
)
MANTYLA_TAXONOMY_CATEGORIES = tuple(
    category for category in MANTYLA_CATEGORIES if category != "unmapped"
)
MANTYLA_TAXONOMY_SOURCES = ("exact_smell_type",)
SMELL_NUMERIC_METRICS = (
    "SmellCount",
    "SmellDensity",
    "SmellsDelta",
)
MAINTAINABILITY_DELTA_METRICS = (
    "MI",
    "CC",
    "HV",
    "CCDensity",
    "HVDensity",
    "DuplicationDensity",
    "CommentDensity",
    "NLOC",
    "CodeSmellDensityDelta",
)
MAINTAINABILITY_LONGITUDINAL_TIMEPOINT_DAYS = {
    "0d": 0,
    "+3d": 3,
    "+7d": 7,
    "+31d": 31,
    "+61d": 61,
}


def normalize_maintainability_tool_status(
    raw_status: Any,
    raw_indicator_status: Any = None,
) -> str:
    """Normalize maintainability-tool status for coverage diagnostics."""
    status = str(raw_status or "").strip().lower()
    indicator_status = str(raw_indicator_status or "").strip().lower()
    if not status:
        return "missing"
    if "unsupported" in status or "unsupported" in indicator_status:
        return "unsupported"
    failure_statuses = {
        "failed",
        "failure",
        "error",
        "timeout",
        "timed_out",
        "crashed",
    }
    if (
        status in failure_statuses
        or indicator_status in failure_statuses
        or status.startswith("fail")
        or indicator_status.startswith("fail")
        or "error" in status
        or "error" in indicator_status
    ):
        return "failed"
    if status == MAINTAINABILITY_SUCCESS_STATUS and (
        not indicator_status or indicator_status == MAINTAINABILITY_SUCCESS_STATUS
    ):
        return "success"
    return "other"


def _validate_mantyla_count_source(source: str) -> str:
    """Validate whether Mantyla counts use stored values or taxonomy mapping."""
    normalized = str(source or MANTYLA_COUNT_SOURCE_TAXONOMY).strip().lower()
    if not normalized:
        return MANTYLA_COUNT_SOURCE_TAXONOMY
    if normalized not in MANTYLA_COUNT_SOURCES:
        allowed = ", ".join(MANTYLA_COUNT_SOURCES)
        raise ValueError(f"Mantyla count source must be one of: {allowed}")
    return normalized


def _repository_root() -> Path:
    """Return the repository root from the analysis utility package path."""
    return Path(__file__).resolve().parents[3]


def _code_smell_taxonomy_config_path() -> Path:
    """Return the curation smell taxonomy config used for grouped counts."""
    return _repository_root() / "curation" / "config" / "code_smell_taxonomy_config.py"


def _code_smell_standardization_config_path() -> Path:
    """Return the curation smell standardization config dependency path."""
    return (
        _repository_root()
        / "curation"
        / "config"
        / "code_smell_standardization_config.py"
    )


def _load_module_from_path(module_name: str, path: Path):
    """Load a config module from a file without importing the full curation app."""
    if not path.exists():
        raise RuntimeError(f"Cannot load {module_name}: {path} does not exist")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_name}: {path} is not importable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _restore_modules(original_modules: dict[str, Any], missing_marker: object) -> None:
    """Restore ``sys.modules`` after temporary config-module loading."""
    for module_name, module in original_modules.items():
        if module is missing_marker:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = module


def _load_code_smell_taxonomy_classifier() -> Callable[..., dict[str, Any]]:
    """Load code smell taxonomy directly without importing the curation app."""
    taxonomy_path = _code_smell_taxonomy_config_path()
    standardization_path = _code_smell_standardization_config_path()
    module_names = (
        "curation",
        "curation.config",
        "curation.config.code_smell_standardization_config",
        "analysis_code_smell_taxonomy_config",
    )
    missing = object()
    original_modules = {
        module_name: sys.modules.get(module_name, missing)
        for module_name in module_names
    }
    try:
        if "curation" not in sys.modules:
            curation_module = types.ModuleType("curation")
            curation_module.__path__ = [str(_repository_root() / "curation")]
            sys.modules["curation"] = curation_module
        if "curation.config" not in sys.modules:
            config_module = types.ModuleType("curation.config")
            config_module.__path__ = [str(_repository_root() / "curation" / "config")]
            sys.modules["curation.config"] = config_module

        _load_module_from_path(
            "curation.config.code_smell_standardization_config",
            standardization_path,
        )
        taxonomy_module = _load_module_from_path(
            "analysis_code_smell_taxonomy_config",
            taxonomy_path,
        )
        classifier = getattr(taxonomy_module, "classify_code_smell_taxonomy", None)
        if not callable(classifier):
            raise RuntimeError(
                "Cannot load code smell taxonomy config: "
                "classify_code_smell_taxonomy is missing"
            )
        return classifier
    finally:
        _restore_modules(original_modules, missing)


def _taxonomy_mantyla_category(
    smell_type: str,
    classifier: Callable[..., dict[str, Any]],
) -> str | None:
    """Map a standardized smell type to a Mantyla category via taxonomy."""
    taxonomy = classifier(
        rule_id=smell_type,
        category="maintainability",
    )
    if not isinstance(taxonomy, dict):
        return None
    category = taxonomy.get("mantyla")
    meta = taxonomy.get("_meta")
    sources = meta.get("sources") if isinstance(meta, dict) else None
    source = sources.get("mantyla") if isinstance(sources, dict) else None
    if (
        category in MANTYLA_TAXONOMY_CATEGORIES
        and source in MANTYLA_TAXONOMY_SOURCES
    ):
        return str(category)
    return None


def parse_count_mapping(raw_value: Any) -> dict[str, int]:
    """Parse a JSON/object code-smell count mapping into positive integer counts."""
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
