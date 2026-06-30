"""Helpers for reading standardized refactoring outputs from curation artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _aggregate_output_root(aggregate_path: Path) -> Path | None:
    parts_lower = [part.lower() for part in Path(aggregate_path).parts]
    if "processed-data" not in parts_lower:
        return None
    processed_index = parts_lower.index("processed-data")
    if processed_index <= 0:
        return None
    return Path(*Path(aggregate_path).parts[:processed_index])


def resolve_curation_artifact_path(
    aggregate_path: Path,
    artifact_path: Any,
) -> Path | None:
    """Resolve an artifact path from a curation aggregate onto the local filesystem."""
    if artifact_path is None or not str(artifact_path).strip():
        return None
    raw_path = str(artifact_path).strip()
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    output_root = _aggregate_output_root(Path(aggregate_path))
    if output_root is None:
        return None

    normalized = raw_path.replace("\\", "/")
    for marker in ("/output/output/", "/output/"):
        if marker not in normalized:
            continue
        suffix = normalized.split(marker, 1)[1].strip("/")
        mapped = output_root.joinpath(*[part for part in suffix.split("/") if part])
        if mapped.exists():
            return mapped
    return None


def _read_json_artifact(aggregate_path: Path, artifact_path: Any) -> dict[str, Any]:
    resolved = resolve_curation_artifact_path(aggregate_path, artifact_path)
    if resolved is None:
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _operation_snapshot_label(
    operation: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> str:
    label = str(operation.get("snapshot_label") or "").strip()
    if not label and result is not None:
        label = str(result.get("snapshot_label") or "").strip()
    return label


def original_pr_refactoring_operations(
    aggregate: dict[str, Any],
    *,
    aggregate_path: Path,
) -> list[dict[str, Any]]:
    """Return standardized original-PR refactoring operations for a curation aggregate."""
    refactoring_metrics = _as_dict(_as_dict(aggregate.get("metrics")).get("refactoring"))
    operations: list[dict[str, Any]] = []

    snapshot_results = _as_list(refactoring_metrics.get("snapshot_results"))
    for result_value in snapshot_results:
        result = _as_dict(result_value)
        if str(result.get("snapshot_label") or "").strip() != "pr":
            continue
        artifact = _read_json_artifact(aggregate_path, result.get("artifact_path"))
        for operation_value in _as_list(artifact.get("standardized_operations")):
            operation = _as_dict(operation_value)
            if _operation_snapshot_label(operation, result) != "pr":
                continue
            standardized_type = str(operation.get("standardized_type") or "").strip()
            if not standardized_type:
                continue
            operations.append(dict(operation))

    if not operations:
        legacy_stage = _as_dict(refactoring_metrics.get("refactoring_operation_mining"))
        for operation_value in _as_list(legacy_stage.get("standardized_operations")):
            operation = _as_dict(operation_value)
            if _operation_snapshot_label(operation) != "pr":
                continue
            standardized_type = str(operation.get("standardized_type") or "").strip()
            if not standardized_type:
                continue
            operations.append(dict(operation))

    return operations


def original_pr_refactoring_summary(
    aggregate: dict[str, Any],
    *,
    aggregate_path: Path,
) -> dict[str, Any]:
    """Return standardized original-PR refactoring counts for a curation aggregate."""
    operations = original_pr_refactoring_operations(
        aggregate,
        aggregate_path=aggregate_path,
    )

    type_counts = Counter(
        str(operation.get("standardized_type") or "").strip()
        for operation in operations
        if str(operation.get("standardized_type") or "").strip()
    )
    return {
        "original_pr_refactoring_count": len(operations),
        "original_pr_refactoring_type_count": dict(sorted(type_counts.items())),
    }
