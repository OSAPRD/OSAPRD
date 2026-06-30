"""Repository-level PR processing pipeline.

This stage takes sampled PR DTOs, groups them by repository, prepares one clone
per repository, hydrates source snapshots, computes metrics, and persists both
per-PR aggregates and parquet rows. It owns resume checkpoints, runtime error
journals, token rotation for repository fetches, and optional snapshot cleanup.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import threading
import time
from typing import Any, Dict, Iterable, List, Set

from curation.config.storage_config import BATCH_SIZE, LOCAL_OUTPUT_DIR
from curation.config.run_config import (
    DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING,
    PROCESSING_PREFETCH_REPOS,
    PROCESSING_REPO_WORKERS,
)
from curation.config.tokens_config import TOKENS
from curation.hydration.pr_hydrator import PRHydrator, pr_to_dict
from curation.hydration.repository_hydrator import GitCommandError, RepositoryHydrator
from curation.metrics import compute_pr_metrics
from curation.pipeline.progress_context import (
    clear_current_pr_progress,
    set_current_pr_progress,
    with_pr_progress,
)
from curation.utility.language_selection import dominant_pr_language
from extraction.utility.storage_handler import StorageHandler
from extraction.utility.token_manager import TokenManager

PROCESSING_CHECKPOINT_VERSION = 4
HYDRATION_STATE_VERSION = 3
METRICS_STATE_VERSION = 4
RUN_ERROR_AGGREGATE_FLUSH_INTERVAL = 25

_CHECKPOINT_WRITE_LOCK = threading.Lock()
_CHECKPOINT_TEXT_CACHE: Dict[Path, str] = {}


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dict row or DTO-like object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_repo_from_url(url: str | None) -> tuple[str | None, str | None]:
    """Extract owner/repository from a GitHub URL when metadata is incomplete."""
    if not url:
        return None, None
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, None


def _repo_key(pr: Any) -> tuple[str | None, str | None]:
    """Return the repository grouping key for one PR."""
    base_repo = _get_attr(pr, "base_repository") or _get_attr(pr, "base_repository_full")
    name_with_owner = _get_attr(base_repo, "name_with_owner")
    if name_with_owner and "/" in name_with_owner:
        owner, repo = name_with_owner.split("/", 1)
        return owner, repo
    repo_url = _get_attr(base_repo, "url") or _get_attr(pr, "url")
    return _parse_repo_from_url(repo_url)


def _pr_id(pr: Any) -> Any:
    """Return a stable human-readable PR identifier for logs and paths."""
    return _get_attr(pr, "number") or _get_attr(pr, "id") or "unknown"


def _pr_language_label(pr: Any) -> str:
    effective = str(_get_attr(pr, "pr_primary_language_effective") or "").strip()
    if effective and effective.lower() not in {"unknown", "none", "nan"}:
        return effective
    languages = _get_attr(pr, "file_languages") or []
    if not isinstance(languages, list):
        return "unknown"
    normalized = [str(lang).strip() for lang in languages if lang]
    if not normalized:
        return "unknown"
    if len(normalized) <= 2:
        return ",".join(normalized)
    return f"{normalized[0]} (+{len(normalized)-1})"


def _benchmark_language(pr: Any) -> str | None:
    """Return dominant benchmark language label (C++, Java, JavaScript, Python) or None."""
    normalized = dominant_pr_language(
        pr,
        supported_languages=("c++", "java", "javascript", "python"),
        tie_break_priority=("c++", "java", "javascript", "python"),
    )
    label_map = {
        "c++": "C++",
        "java": "Java",
        "javascript": "JavaScript",
        "python": "Python",
    }
    return label_map.get(str(normalized)) if normalized else None


def _timing_bucket(include_future: bool) -> str:
    """Return the timing bucket for aggregate runtime statistics."""
    return "with_longitudinal" if include_future else "without_longitudinal"


def _serialize_repo_metadata(repo_obj: Any) -> Dict[str, Any]:
    """Serialize repository metadata from dicts, DTOs, or dataclasses."""
    if repo_obj is None:
        return {}
    raw = getattr(repo_obj, "__dict__", None)
    if isinstance(repo_obj, dict):
        payload = dict(repo_obj)
        return payload
    to_dict = getattr(repo_obj, "to_dict", None)
    payload: Dict[str, Any] = {}
    if callable(to_dict):
        serialized = to_dict()
        if isinstance(serialized, dict):
            payload = dict(serialized)
    if isinstance(raw, dict):
        for key in ("popularity_label", "popularity_bucket"):
            if key in raw and raw.get(key) is not None:
                payload[key] = raw.get(key)
        if not payload:
            payload = dict(raw)
    return payload


def _existing_repo_metadata(pr: Any) -> Dict[str, Any]:
    """Read repository metadata already present on the PR row."""
    base_full = _get_attr(pr, "base_repository_full")
    if base_full is not None:
        return _serialize_repo_metadata(base_full)
    base_peek = _get_attr(pr, "base_repository")
    return _serialize_repo_metadata(base_peek)


_REPO_METADATA_DROP_KEYS = {
    "allow_merge_commit",
    "allow_rebase_merge",
    "allow_squash_merge",
    "archived_at",
    "archived_reason",
    "lock_reason",
    "network_count",
}


def _sanitize_repository_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Remove fields that are out of scope for persisted repository metadata."""
    if not isinstance(metadata, dict):
        return {}
    sanitized = dict(metadata)
    for key in _REPO_METADATA_DROP_KEYS:
        sanitized.pop(key, None)
    return sanitized


def _merge_repository_metadata(pr: Any) -> Dict[str, Any]:
    """Combine repository metadata and labels into one persisted dictionary."""
    merged = _existing_repo_metadata(pr)
    if not merged.get("repository_labels"):
        labels_raw = _get_attr(pr, "labels")
        if labels_raw is None:
            labels = []
        elif isinstance(labels_raw, list):
            labels = labels_raw
        else:
            tolist = getattr(labels_raw, "tolist", None)
            if callable(tolist):
                try:
                    converted = tolist()
                    labels = converted if isinstance(converted, list) else [converted]
                except Exception:
                    labels = [labels_raw]
            else:
                labels = [labels_raw]
        label_names: List[str] = []
        if isinstance(labels, list):
            for label in labels:
                if isinstance(label, dict):
                    name = label.get("name") or label.get("id")
                else:
                    name = _get_attr(label, "name") or _get_attr(label, "id")
                if name:
                    text = str(name).strip()
                    if text:
                        label_names.append(text)
        if label_names:
            merged["repository_labels"] = sorted(set(label_names))
    if not merged.get("popularity_bucket"):
        merged["popularity_bucket"] = _get_attr(pr, "popularity_bucket")
    if not merged.get("popularity_label"):
        merged["popularity_label"] = _get_attr(pr, "popularity_label")
    if not merged.get("popularity_label") and merged.get("popularity_bucket"):
        merged["popularity_label"] = str(merged.get("popularity_bucket"))
    return _sanitize_repository_metadata(merged)


def _load_processing_progress(path: Path) -> Set[str]:
    """Load the completed-PR URL/id journal for resume handling."""
    if not path.exists():
        return set()
    completed: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if key:
                completed.add(key)
    return completed


def _append_processing_progress(path: Path, key: str) -> None:
    """Append one completed PR key to the resume journal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{key}\n")


def _sanitize_checkpoint_component(value: Any) -> str:
    """Sanitize user/repository/cohort values used in checkpoint filenames."""
    text = str(value) if value is not None else "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "unknown"


_FAILED_PR_SKIP_STAGES = {
    "pr_checkpoint_load",
    "hydration",
    "metrics_compute",
    "tool_gate",
    "tool_gate_cleanup",
    "persistence",
}


def _failed_pr_url_from_error(entry: Any) -> str | None:
    item = entry if isinstance(entry, dict) else {}
    pr_url = str(item.get("pr_url") or "").strip()
    if not pr_url:
        return None
    stage = str(item.get("stage") or "").strip()
    status = str(item.get("status") or "").strip()
    if stage in _FAILED_PR_SKIP_STAGES or status in {"failed", "not_persisted"}:
        return pr_url
    return None


def _load_failed_processing_pr_urls(output_root: Path, cohort: str) -> Set[str]:
    safe_cohort = _sanitize_checkpoint_component(cohort)
    urls: Set[str] = set()
    aggregate_path = output_root / f"run_errors_{safe_cohort}.json"
    if aggregate_path.exists():
        try:
            payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
            entries = payload.get("errors") if isinstance(payload, dict) else []
            for entry in _as_list(entries):
                pr_url = _failed_pr_url_from_error(entry)
                if pr_url:
                    urls.add(pr_url)
        except Exception:
            pass
    journal_path = output_root / f"run_errors_{safe_cohort}.runtime.jsonl"
    if journal_path.exists():
        try:
            with journal_path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        entry = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    pr_url = _failed_pr_url_from_error(entry)
                    if pr_url:
                        urls.add(pr_url)
        except Exception:
            pass
    return urls


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


_PR_METADATA_DROP_KEYS = {
    "timeline_items",
    "comments",
    "reviews",
    "comments_count",
    "reviews_count",
    "mergeable",
    "mergeable_state",
    "mergeable_method",
    "review_decision",
}


def _json_scalarize(value: Any) -> Any:
    """Return a parquet-safe scalar for arbitrary API payload values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return str(bytes(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
        except Exception:
            return str(value)
    return str(value)


def _sanitize_persist_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unstable PR API fields that can cause parquet schema drift."""
    sanitized = {str(k): _json_scalarize(v) for k, v in dict(record or {}).items()}
    for key in _PR_METADATA_DROP_KEYS:
        sanitized.pop(key, None)
    # Force repository payload columns to stable scalar text.
    for repo_key in ("base_repository", "base_repository_full"):
        if repo_key in sanitized:
            sanitized[repo_key] = _json_scalarize(sanitized.get(repo_key))
    return sanitized


def _serialize_pr_metadata(pr_obj: Any) -> Dict[str, Any]:
    if isinstance(pr_obj, dict):
        payload = dict(pr_obj)
    else:
        to_dict = getattr(pr_obj, "to_dict", None)
        if callable(to_dict):
            try:
                serialized = to_dict()
                payload = dict(serialized) if isinstance(serialized, dict) else {}
            except Exception:
                payload = {}
        elif hasattr(pr_obj, "__dict__"):
            payload = dict(getattr(pr_obj, "__dict__", {}) or {})
        else:
            payload = {}
    for key in _PR_METADATA_DROP_KEYS:
        payload.pop(key, None)
    return payload


def _unique_sorted_strings(values: List[Any]) -> List[str]:
    normalized = {
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    }
    return sorted(normalized)


def _count_values(values: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _snapshot_summary(hydration: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = _as_dict(hydration.get("snapshots"))
    future = _as_dict(snapshots.get("future"))
    return {
        "before": _as_dict(snapshots.get("before")),
        "after": _as_dict(snapshots.get("after")),
        "future": {label: _as_dict(meta) for label, meta in future.items()},
        "has_future_snapshots": bool(hydration.get("has_future_snapshots")),
    }


def _collect_stage_errors(
    stage_name: str,
    stage_status: Any,
    snapshot_results: List[Any],
) -> Dict[str, Any]:
    success_statuses = {"success", "partial_success", "completed", "skipped"}
    errors: List[Dict[str, Any]] = []

    normalized_stage_status = str(stage_status or "").strip().lower()
    if normalized_stage_status and normalized_stage_status not in success_statuses:
        errors.append(
            {
                "stage": stage_name,
                "scope": "stage",
                "status": stage_status,
            }
        )

    for result in snapshot_results:
        item = _as_dict(result)
        status = str(item.get("status") or "").strip().lower()
        if not status or status in success_statuses:
            continue
        errors.append(
            {
                "stage": stage_name,
                "scope": "snapshot",
                "snapshot_label": item.get("snapshot_label"),
                "status": item.get("status"),
                "notes": item.get("notes"),
                "return_code": item.get("return_code"),
                "artifact_path": item.get("artifact_path"),
                "stderr_path": item.get("stderr_path"),
                "stdout_path": item.get("stdout_path"),
                "tool": item.get("tool"),
            }
        )

    return {
        "count": len(errors),
        "entries": errors,
    }


def _snapshot_results_with_artifact_summaries(snapshot_results: List[Any]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for result in snapshot_results:
        item = dict(_as_dict(result))
        artifact_path_raw = item.get("artifact_path")
        if artifact_path_raw:
            artifact = _read_json_file(Path(str(artifact_path_raw)))
            if isinstance(artifact, dict):
                tool_runs = _as_list(artifact.get("tool_runs"))
                if tool_runs:
                    item["tool_runs"] = tool_runs
                for key in (
                    "stdout_preview",
                    "stderr_preview",
                    "elapsed_seconds",
                    "timed_out",
                    "stdout_path",
                    "stderr_path",
                ):
                    if key in artifact and item.get(key) in {None, ""}:
                        item[key] = artifact.get(key)
        enriched.append(item)
    return enriched


def _build_pr_metrics_aggregate(
    *,
    pr: Any,
    cohort: str,
    owner: str,
    repo: str,
    hydration: Dict[str, Any],
    metrics: Dict[str, Any],
    repository_metadata: Dict[str, Any],
    include_future: bool,
    processing_timing: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    pr_url = _get_attr(pr, "url")
    pr_number = _get_attr(pr, "number")
    pr_created_at = _get_attr(pr, "created_at")
    pr_merged_at = _get_attr(pr, "merged_at") or hydration.get("merge_time")
    pr_payload = _serialize_pr_metadata(pr)
    pr_payload.update(
        {
            "url": pr_url,
            "number": pr_number,
            "primary_language": _get_attr(pr, "primary_language"),
            "pr_primary_language_effective": _get_attr(pr, "pr_primary_language_effective"),
            "created_at": pr_created_at,
            "merged_at": pr_merged_at,
            "repository_owner": owner,
            "repository_name": repo,
            "longitudinal_selected": bool(include_future),
        }
    )

    ref_stage = _as_dict(metrics.get("refactoring_metrics"))
    ref_mining = _as_dict(ref_stage.get("refactoring_operation_mining"))
    ref_summary = _as_dict(_as_dict(ref_stage.get("refactoring_metrics")).get("metrics"))
    ref_snapshot_results = _snapshot_results_with_artifact_summaries(
        _as_list(ref_mining.get("results"))
    )
    ref_operations = _as_list(ref_mining.get("standardized_operations"))
    ref_types_raw = [
        _as_dict(operation).get("standardized_type") for operation in ref_operations
    ]
    ref_types = _unique_sorted_strings(ref_types_raw)

    maint_stage = _as_dict(metrics.get("maintainability_metrics"))
    maint_indicators = _as_dict(maint_stage.get("maintainability_indicators"))
    maint_summary = _as_dict(maint_indicators.get("summary"))
    maint_snapshot_results = _snapshot_results_with_artifact_summaries(
        _as_list(maint_indicators.get("results"))
    )
    smells = _as_list(maint_indicators.get("code_smells"))
    smell_types_raw = [
        _as_dict(smell).get("rule_id")
        or _as_dict(smell).get("rule")
        or _as_dict(smell).get("name")
        or _as_dict(smell).get("type")
        for smell in smells
    ]
    smell_types = _unique_sorted_strings(smell_types_raw)
    smell_type_count = _as_dict(maint_summary.get("smell_type_count"))
    if not smell_type_count:
        smell_type_count = _count_values([value for value in smell_types_raw if isinstance(value, str) and value.strip()])

    ref_errors = _collect_stage_errors(
        "refactoring",
        ref_stage.get("status"),
        ref_snapshot_results,
    )
    maint_errors = _collect_stage_errors(
        "maintainability",
        maint_stage.get("status"),
        maint_snapshot_results,
    )

    return {
        "schema_version": 1,
        "cohort": cohort,
        "pr": pr_payload,
        "repository_metadata": repository_metadata,
        "hydration": {
            "merge_commit": hydration.get("merge_commit"),
            "base_commit": hydration.get("base_commit"),
            "after_commit": hydration.get("after_commit"),
            "merge_time": hydration.get("merge_time"),
            "snapshots": _snapshot_summary(hydration),
            "future_snapshot_availability": _as_dict(
                hydration.get("future_snapshot_availability")
            ),
        },
        "metrics": {
            "status": metrics.get("status"),
            "metrics_backend": metrics.get("metrics_backend"),
            "processing_timing": _as_dict(processing_timing),
            "refactoring": {
                "status": ref_stage.get("status"),
                "tool": ref_mining.get("selected_tool"),
                "operation_types": ref_types,
                "operation_type_count": _as_dict(ref_summary.get("refactor_type_count")),
                "murphyhill_count": _as_dict(ref_summary.get("refactor_murphyhill_count")),
                "future_snapshot_metrics": _as_dict(
                    ref_summary.get("refactor_future_snapshot_metrics")
                ),
                "future_snapshot_availability": _as_dict(
                    ref_summary.get("refactor_future_snapshot_availability")
                ),
                "summary": ref_summary,
                "snapshot_results": ref_snapshot_results,
            },
            "maintainability": {
                "status": maint_stage.get("status"),
                "engine": maint_stage.get("engine"),
                "tools": _as_list(maint_indicators.get("selected_tools")),
                "smell_types": smell_types,
                "smell_type_count": smell_type_count,
                "future_snapshot_metrics": _as_dict(
                    maint_summary.get("maintainability_future_snapshot_metrics")
                ),
                "future_snapshot_availability": _as_dict(
                    maint_summary.get("future_snapshot_availability")
                ),
                "taxonomy_counts": {
                    "mantyla": _as_dict(maint_summary.get("smell_count_by_mantyla")),
                },
                "summary": maint_summary,
                "snapshot_results": maint_snapshot_results,
            },
            "longitudinal_commit_activity": _as_dict(
                hydration.get("longitudinal_commit_activity")
            ),
            "errors": {
                "total_count": int(ref_errors["count"]) + int(maint_errors["count"]),
                "refactoring": ref_errors,
                "maintainability": maint_errors,
            },
        },
    }


def _aggregate_metrics_path(
    output_root: Path,
    cohort: str,
    owner: str,
    repo: str,
    pr: Any,
) -> Path:
    safe_cohort = _sanitize_checkpoint_component(cohort)
    safe_owner = _sanitize_checkpoint_component(owner)
    safe_repo = _sanitize_checkpoint_component(repo)
    pr_number = _sanitize_checkpoint_component(_get_attr(pr, "number") or _get_attr(pr, "id") or "unknown")
    pr_url = str(_get_attr(pr, "url") or f"{owner}/{repo}#{pr_number}")
    digest = hashlib.sha1(pr_url.encode("utf-8")).hexdigest()[:12]
    root = output_root / "processed-data" / safe_cohort / "metrics-json" / f"{safe_owner}__{safe_repo}"
    return root / f"pr-{pr_number}__{digest}.json"


def _write_metrics_aggregate(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp_path.replace(path)


def _path_value_inside_root(value: Any, root: Path) -> bool:
    if not value:
        return False
    try:
        candidate = Path(str(value))
    except Exception:
        return False
    if not candidate.is_absolute():
        return False
    return _is_strict_child_path(candidate, root)


def _mark_cleaned_paths(item: Dict[str, Any], cleanup_root: Path) -> None:
    cleaned_any = False
    for key in (
        "artifact_path",
        "stdout_path",
        "stderr_path",
        "output_path",
        "findings_path",
        "measures_path",
        "raw_findings_path",
        "raw_measures_path",
        "raw_issues_path",
        "tool_runs_path",
    ):
        value = item.get(key)
        if _path_value_inside_root(value, cleanup_root):
            item[f"{key}_cleaned"] = value
            item[key] = None
            cleaned_any = True
    if cleaned_any:
        item["artifacts_cleaned"] = True
    for run in _as_list(item.get("tool_runs")):
        run_item = _as_dict(run)
        if run_item:
            _mark_cleaned_paths(run_item, cleanup_root)


def _mark_aggregate_artifacts_cleaned(aggregate_path: Path, cleanup_root: Path) -> None:
    payload = _read_json_file(aggregate_path)
    if not isinstance(payload, dict):
        return
    payload["artifact_cleanup"] = {
        "status": "cleaned",
        "snapshot_root_cleaned": str(cleanup_root),
        "cleaned_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }
    metrics = _as_dict(payload.get("metrics"))
    for stage_name in ("refactoring", "maintainability"):
        stage = _as_dict(metrics.get(stage_name))
        for snapshot in _as_list(stage.get("snapshot_results")):
            item = _as_dict(snapshot)
            if item:
                _mark_cleaned_paths(item, cleanup_root)
    hydration = _as_dict(payload.get("hydration"))
    if hydration:
        hydration["artifacts_cleaned"] = True
    _write_metrics_aggregate(aggregate_path, payload)


def _read_json_file(path: Path) -> Dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_run_errors_aggregate(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp_path.replace(path)


def _append_run_error_journal(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str))
        f.write("\n")


def _collect_run_errors_from_metrics(output_root: Path, cohort: str) -> Dict[str, Any]:
    safe_cohort = _sanitize_checkpoint_component(cohort)
    metrics_root = output_root / "processed-data" / safe_cohort / "metrics-json"
    if not metrics_root.exists():
        return {"entries": [], "source_files_scanned": 0}

    success_statuses = {"success", "partial_success", "completed", "missing_snapshot"}
    entries: List[Dict[str, Any]] = []
    scanned = 0
    for aggregate_path in metrics_root.rglob("*.json"):
        scanned += 1
        aggregate = _read_json_file(aggregate_path)
        if not isinstance(aggregate, dict):
            continue
        pr_info = _as_dict(aggregate.get("pr"))
        metrics = _as_dict(aggregate.get("metrics"))
        pr_url = pr_info.get("url")
        pr_number = pr_info.get("number")
        owner = pr_info.get("repository_owner")
        repo = pr_info.get("repository_name")

        for stage_name in ("refactoring", "maintainability"):
            stage = _as_dict(metrics.get(stage_name))
            stage_status = str(stage.get("status") or "").strip().lower()
            if stage_status and stage_status not in success_statuses:
                entries.append(
                    {
                        "source": "aggregate_stage",
                        "stage": stage_name,
                        "repository_owner": owner,
                        "repository_name": repo,
                        "pr_number": pr_number,
                        "pr_url": pr_url,
                        "status": stage.get("status"),
                        "aggregate_path": str(aggregate_path),
                    }
                )
            snapshot_results = _as_list(stage.get("snapshot_results"))
            for snapshot in snapshot_results:
                snapshot_item = _as_dict(snapshot)
                snapshot_status = str(snapshot_item.get("status") or "").strip().lower()
                if snapshot_status and snapshot_status not in success_statuses:
                    entries.append(
                        {
                            "source": "aggregate_snapshot",
                            "stage": stage_name,
                            "repository_owner": owner,
                            "repository_name": repo,
                            "pr_number": pr_number,
                            "pr_url": pr_url,
                            "snapshot_label": snapshot_item.get("snapshot_label"),
                            "tool": snapshot_item.get("tool"),
                            "status": snapshot_item.get("status"),
                            "notes": snapshot_item.get("notes"),
                            "return_code": snapshot_item.get("return_code"),
                            "artifact_path": snapshot_item.get("artifact_path"),
                            "aggregate_path": str(aggregate_path),
                        }
                    )

                # Include tool-level failures embedded in the aggregate. Older
                # aggregates may still require reading the per-snapshot artifact.
                tool_runs = _as_list(snapshot_item.get("tool_runs"))
                artifact_path_raw = snapshot_item.get("artifact_path")
                artifact_path: Path | None = None
                if not tool_runs and artifact_path_raw:
                    artifact_path = Path(str(artifact_path_raw))
                    if artifact_path.exists():
                        artifact = _read_json_file(artifact_path)
                        if isinstance(artifact, dict):
                            tool_runs = _as_list(artifact.get("tool_runs"))
                for run in tool_runs:
                    run_item = _as_dict(run)
                    run_status = str(run_item.get("status") or "").strip().lower()
                    if not run_status or run_status in {"completed", "skipped"}:
                        continue
                    entries.append(
                        {
                            "source": "tool_run",
                            "stage": stage_name,
                            "repository_owner": owner,
                            "repository_name": repo,
                            "pr_number": pr_number,
                            "pr_url": pr_url,
                            "snapshot_label": snapshot_item.get("snapshot_label"),
                            "tool": run_item.get("tool"),
                            "status": run_item.get("status"),
                            "notes": run_item.get("notes"),
                            "return_code": run_item.get("return_code"),
                            "artifact_path": str(artifact_path) if artifact_path else artifact_path_raw,
                            "aggregate_path": str(aggregate_path),
                        }
                    )

    return {"entries": entries, "source_files_scanned": scanned}


def _write_repository_file_list(
    *,
    snapshots_root: Path,
    owner: str,
    repo: str,
    ref: str,
    commit: str | None,
    files: List[str],
) -> Path:
    repo_root = snapshots_root / owner / repo
    repo_root.mkdir(parents=True, exist_ok=True)
    path = repo_root / "repository_file_list.json"
    payload = {
        "schema_version": 1,
        "repository_owner": owner,
        "repository_name": repo,
        "source_ref": ref,
        "source_commit": commit,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(files),
        "files": files,
    }
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


_TOOL_GATE_SUCCESS_STATUSES = {"success", "partial_success", "completed", "missing_snapshot"}
_TOOL_GATE_REF_SNAPSHOT_SUCCESS = {"success", "missing_snapshot", "future_tool_disabled"}


def _compact_reason_note(value: Any, *, max_chars: int = 220) -> str:
    text = str(value or "").strip().replace("\n", " | ")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated {len(text) - max_chars} chars]"


def _tool_gate_failure_reasons(metrics: Dict[str, Any]) -> List[str]:
    """
    Return hard tool-gate failures that should exclude a PR.

    Maintainability failures are still captured in aggregate metrics and run errors,
    but they do not gate PR inclusion.
    """
    reasons: List[str] = []
    if not isinstance(metrics, dict):
        return ["missing_metrics"]
    ref_stage = _as_dict(metrics.get("refactoring_metrics"))
    if ref_stage:
        ref_stage_status = str(ref_stage.get("status") or "").strip().lower()
        if ref_stage_status == "skipped":
            return []
        ref_snapshot_failures: List[str] = []
        ref_mining = _as_dict(ref_stage.get("refactoring_operation_mining")) or _as_dict(
            ref_stage.get("mining")
        )
        ref_snapshot_results = _as_list(ref_mining.get("snapshot_results")) or _as_list(
            ref_mining.get("results")
        ) or _as_list(
            ref_stage.get("snapshot_results")
        )
        for snapshot in ref_snapshot_results:
            snapshot_item = _as_dict(snapshot)
            status = str(snapshot_item.get("status") or "").strip().lower()
            if status and status not in _TOOL_GATE_REF_SNAPSHOT_SUCCESS:
                label = str(snapshot_item.get("snapshot_label") or "unknown")
                tool = str(snapshot_item.get("tool") or "unknown")
                reason = f"refactoring_snapshot:{label}:{tool}:{status}"
                notes = _compact_reason_note(snapshot_item.get("notes"))
                if notes:
                    reason += f"(notes={notes})"
                return_code = snapshot_item.get("return_code")
                if return_code is not None:
                    reason += f"(return_code={return_code})"
                ref_snapshot_failures.append(reason)
        reasons.extend(ref_snapshot_failures)
        if ref_stage_status and ref_stage_status not in _TOOL_GATE_SUCCESS_STATUSES and not ref_snapshot_failures:
            reasons.append(f"refactoring_stage:{ref_stage_status}(no_snapshot_failure_details)")
    else:
        reasons.append("missing_refactoring")

    return reasons


def _pr_snapshot_artifact_root(
    *,
    snapshots_root: Path,
    owner: str,
    repo: str,
    pr: Any,
) -> Path:
    pr_number = _sanitize_checkpoint_component(
        _get_attr(pr, "number") or _get_attr(pr, "id") or "unknown"
    )
    return snapshots_root / owner / repo / f"pr-{pr_number}"


def _is_strict_child_path(path: Path, parent: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved_parent = parent.resolve()
        if resolved == resolved_parent:
            return False
        resolved.relative_to(resolved_parent)
        return True
    except Exception:
        return False


def _maybe_remove_pr_snapshot_artifact_root(
    *,
    snapshots_root: Path,
    owner: str,
    repo: str,
    pr: Any,
) -> tuple[Path, bool]:
    snapshot_root = _pr_snapshot_artifact_root(
        snapshots_root=snapshots_root,
        owner=owner,
        repo=repo,
        pr=pr,
    )
    if not _is_strict_child_path(snapshot_root, snapshots_root):
        raise RuntimeError(
            f"Refusing to delete snapshot artifacts outside snapshots root: {snapshot_root}"
        )
    if not DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING:
        return snapshot_root, False
    existed = snapshot_root.exists()
    if existed:
        shutil.rmtree(snapshot_root)
    return snapshot_root, existed


def _cleanup_skipped_pr_artifacts(
    *,
    snapshots_root: Path,
    output_root: Path,
    cohort: str,
    owner: str,
    repo: str,
    pr: Any,
) -> None:
    _maybe_remove_pr_snapshot_artifact_root(
        snapshots_root=snapshots_root,
        owner=owner,
        repo=repo,
        pr=pr,
    )
    aggregate_path = _aggregate_metrics_path(output_root, cohort, owner, repo, pr)
    if aggregate_path.exists():
        try:
            aggregate_path.unlink()
        except Exception:
            pass


def _cleanup_persisted_pr_artifacts(
    *,
    snapshots_root: Path,
    owner: str,
    repo: str,
    pr: Any,
) -> tuple[Path, bool]:
    return _maybe_remove_pr_snapshot_artifact_root(
        snapshots_root=snapshots_root,
        owner=owner,
        repo=repo,
        pr=pr,
    )


def _processing_checkpoint_path(
    checkpoints_root: Path,
    cohort: str,
    owner: str,
    repo: str,
    pr: Any,
) -> Path:
    pr_url = _get_attr(pr, "url") or "unknown"
    pr_number = _get_attr(pr, "number") or _get_attr(pr, "id") or "unknown"
    digest = hashlib.sha1(pr_url.encode("utf-8")).hexdigest()[:12]
    filename = (
        f"{_sanitize_checkpoint_component(owner)}__"
        f"{_sanitize_checkpoint_component(repo)}__"
        f"pr-{_sanitize_checkpoint_component(pr_number)}__"
        f"{digest}.json"
    )
    return checkpoints_root / _sanitize_checkpoint_component(cohort) / filename


def _load_processing_checkpoint(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load processing checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid processing checkpoint payload at {path}")
    return payload


def _checkpoint_meta() -> Dict[str, int]:
    return {
        "processing_version": PROCESSING_CHECKPOINT_VERSION,
        "hydration_version": HYDRATION_STATE_VERSION,
        "metrics_version": METRICS_STATE_VERSION,
    }


def _legacy_checkpoint_record(payload: Dict[str, Any], pr_url: str) -> Dict[str, Any]:
    record = dict(payload)
    if record.get("pr_url") not in {None, pr_url}:
        return {}
    if not isinstance(record.get("hydration"), dict):
        record.pop("hydration", None)
    metrics_progress = record.get("metrics_progress")
    if not isinstance(metrics_progress, dict):
        legacy_metrics = record.get("metrics")
        if isinstance(legacy_metrics, dict):
            metrics_progress = legacy_metrics
        else:
            metrics_progress = None
    if metrics_progress is not None:
        record["metrics_progress"] = metrics_progress
    record.pop("metrics", None)
    record["_checkpoint"] = _checkpoint_meta()
    return record


def _normalize_checkpoint_record(payload: Dict[str, Any], pr_url: str) -> Dict[str, Any]:
    record = dict(payload)
    meta = record.get("_checkpoint")
    if not isinstance(meta, dict):
        return _legacy_checkpoint_record(payload, pr_url)
    if meta.get("processing_version") != PROCESSING_CHECKPOINT_VERSION:
        return {}
    if record.get("pr_url") not in {None, pr_url}:
        return {}
    if meta.get("hydration_version") != HYDRATION_STATE_VERSION:
        record.pop("hydration", None)
        record.pop("metrics_progress", None)
    elif meta.get("metrics_version") != METRICS_STATE_VERSION:
        record.pop("metrics_progress", None)
    record.pop("metrics", None)
    record["_checkpoint"] = _checkpoint_meta()
    return record


def _write_processing_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    payload_to_write = dict(payload)
    payload_to_write["_checkpoint"] = _checkpoint_meta()
    text = json.dumps(payload_to_write, ensure_ascii=False, indent=2, default=str)
    with _CHECKPOINT_WRITE_LOCK:
        if _CHECKPOINT_TEXT_CACHE.get(path) == text:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
        _CHECKPOINT_TEXT_CACHE[path] = text


def _build_pr_processing_checkpoint(
    *,
    pr_url: str,
    current_phase: str,
    hydration: Dict[str, Any] | None = None,
    metrics_progress: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the lightweight per-PR processing checkpoint payload."""
    payload: Dict[str, Any] = {
        "pr_url": pr_url,
        "current_phase": current_phase,
        "_checkpoint": _checkpoint_meta(),
    }
    if isinstance(hydration, dict):
        payload["hydration"] = hydration
    if isinstance(metrics_progress, dict):
        payload["metrics_progress"] = metrics_progress
    return payload


def _delete_processing_checkpoint(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    finally:
        with _CHECKPOINT_WRITE_LOCK:
            _CHECKPOINT_TEXT_CACHE.pop(path, None)


def _repo_prepare_checkpoint_path(
    checkpoints_root: Path,
    cohort: str,
    owner: str,
    repo: str,
) -> Path:
    filename = (
        f"{_sanitize_checkpoint_component(owner)}__"
        f"{_sanitize_checkpoint_component(repo)}__repo.json"
    )
    return checkpoints_root / _sanitize_checkpoint_component(cohort) / filename


def _repo_prepare_checkpoint_payload(
    owner: str,
    repo: str,
    default_branch: str | None,
) -> Dict[str, Any]:
    return {
        "_checkpoint": _checkpoint_meta(),
        "owner": owner,
        "repo": repo,
        "default_branch": default_branch,
        "status": "prepared",
    }


def _is_prepared_repo_checkpoint(
    path: Path,
    owner: str,
    repo: str,
) -> bool:
    payload = _load_processing_checkpoint(path)
    if not isinstance(payload, dict):
        return False
    meta = payload.get("_checkpoint")
    if not isinstance(meta, dict):
        return False
    if meta.get("processing_version") != PROCESSING_CHECKPOINT_VERSION:
        return False
    if payload.get("owner") != owner or payload.get("repo") != repo:
        return False
    return payload.get("status") == "prepared"


def process_prs(
    prs: Iterable[Any],
    cohort: str,
    resume: bool = True,
    longitudinal_urls: Iterable[str] | None = None,
    force_reprocess: bool = False,
    skip_failed_prs_on_resume: bool = False,
) -> Dict[str, Any]:
    """Process sampled PRs repo-by-repo: hydrate, compute metrics, and persist results."""
    output_root = Path(LOCAL_OUTPUT_DIR) / "output"
    clones_root = output_root / "clones"
    snapshots_root = output_root / "snapshots"
    checkpoints_root = output_root / "checkpoints" / "processing"
    progress_path = output_root / f"processing_progress_{cohort}.txt"
    output_root.mkdir(parents=True, exist_ok=True)
    clones_root.mkdir(parents=True, exist_ok=True)
    snapshots_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    # Processed rows use the extraction storage handler in durable-journal mode:
    # each PR reaches a crash-recoverable JSONL journal before parquet compaction.
    storage = StorageHandler(
        local_dir=output_root,
        batch_size=BATCH_SIZE,
        data_subdir="processed-data",
        durable_journal=True,
        durable_journal_fsync_interval=BATCH_SIZE,
    )
    failed_on_resume: Set[str] = set()
    if resume and skip_failed_prs_on_resume and not force_reprocess:
        failed_on_resume = _load_failed_processing_pr_urls(output_root, cohort)
        if failed_on_resume:
            failed_on_resume = {url for url in failed_on_resume if not storage.is_recorded(url)}

    completed = _load_processing_progress(progress_path) if resume else set()
    grouped: dict[tuple[str | None, str | None], list[Any]] = defaultdict(list)
    longitudinal_selected = {str(url) for url in (longitudinal_urls or []) if url}
    total_prs = 0
    skipped_failed_urls: Set[str] = set()
    prefiltered_completed = 0
    prefiltered_persisted = 0
    for pr in prs:
        # Filter already completed/persisted PRs before cloning repositories; on
        # large cohorts this avoids expensive fetches for work that is already
        # durable on disk.
        pr_url = _get_attr(pr, "url")
        pr_url_text = str(pr_url) if pr_url else ""
        if resume and not force_reprocess and pr_url_text:
            if pr_url_text in completed:
                prefiltered_completed += 1
                continue
            if storage.is_recorded(pr_url_text):
                prefiltered_persisted += 1
                continue
            if failed_on_resume and pr_url_text in failed_on_resume:
                skipped_failed_urls.add(pr_url_text)
                continue
        elif failed_on_resume and pr_url_text in failed_on_resume:
            skipped_failed_urls.add(pr_url_text)
            continue
        owner, repo = _repo_key(pr)
        grouped[(owner, repo)].append(pr)
        total_prs += 1
    if prefiltered_completed or prefiltered_persisted:
        print(
            "[processing] Prefiltered {total} already handled PR(s) before repo clone "
            "(completed={completed_count}, persisted={persisted_count}).".format(
                total=prefiltered_completed + prefiltered_persisted,
                completed_count=prefiltered_completed,
                persisted_count=prefiltered_persisted,
            )
        )
    if skipped_failed_urls:
        print(
            "[processing] Skipping {count} previously failed sampled PR(s) on resume; "
            "top-up will handle replacements.".format(count=len(skipped_failed_urls))
        )

    benchmark_languages = ("C++", "Java", "JavaScript", "Python")
    timing_buckets = ("with_longitudinal", "without_longitudinal")
    state: Dict[str, Any] = {
        "total_with_future": 0,
        "total_prs_processed": 0,
        "total_prs_completed": 0,
        "metrics_count": 0,
        "metrics_failed": 0,
        "before_count": 0,
        "after_count": 0,
        "future_counts": {},
        "language_time_total_seconds": {
            lang: {bucket: 0.0 for bucket in timing_buckets} for lang in benchmark_languages
        },
        "language_time_counts": {
            lang: {bucket: 0 for bucket in timing_buckets} for lang in benchmark_languages
        },
        "processed_index": 0,
        "skipped_failed_on_resume": len(skipped_failed_urls),
        "prefiltered_completed_on_resume": prefiltered_completed,
        "prefiltered_persisted_on_resume": prefiltered_persisted,
    }
    # TokenManager is optional so local/public repositories can still be
    # processed without credentials. When tokens are present, repository workers
    # rotate them on clone/fetch failures.
    token_manager = TokenManager(TOKENS) if TOKENS else None
    token_lock = threading.Lock()
    storage_lock = threading.Lock()
    state_lock = threading.Lock()
    log_lock = threading.Lock()
    completed_lock = threading.Lock()
    run_errors_lock = threading.Lock()
    run_runtime_errors: List[Dict[str, Any]] = []
    safe_cohort = _sanitize_checkpoint_component(cohort)
    run_errors_path = output_root / f"run_errors_{safe_cohort}.json"
    run_errors_journal_path = output_root / f"run_errors_{safe_cohort}.runtime.jsonl"
    try:
        run_errors_journal_path.unlink()
    except FileNotFoundError:
        pass

    def _processing_log(message: str) -> None:
        with log_lock:
            print(with_pr_progress(message))

    def _next_processed_index() -> int:
        with state_lock:
            state["processed_index"] += 1
            return int(state["processed_index"])

    def _with_state_lock(fn):
        with state_lock:
            return fn()

    def _increment_metrics_failed() -> None:
        def _inc() -> None:
            state["metrics_failed"] += 1

        _with_state_lock(_inc)

    def _flush_runtime_run_errors(
        entries_snapshot: List[Dict[str, Any]] | None = None,
        *,
        force: bool = False,
    ) -> None:
        if entries_snapshot is None:
            with run_errors_lock:
                entries_snapshot = list(run_runtime_errors)
        if (
            entries_snapshot
            and not force
            and len(entries_snapshot) % RUN_ERROR_AGGREGATE_FLUSH_INTERVAL != 0
        ):
            return
        errors_by_stage: Dict[str, int] = {}
        errors_by_status: Dict[str, int] = {}
        for entry in entries_snapshot:
            item = _as_dict(entry)
            stage_key = str(item.get("stage") or "unknown")
            status_key = str(item.get("status") or "unknown")
            errors_by_stage[stage_key] = int(errors_by_stage.get(stage_key, 0)) + 1
            errors_by_status[status_key] = int(errors_by_status.get(status_key, 0)) + 1
        payload = {
            "schema_version": 1,
            "cohort": cohort,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": {
                "total_errors": len(entries_snapshot),
                "runtime_errors": len(entries_snapshot),
                "aggregated_errors": 0,
                "aggregate_files_scanned": 0,
                "runtime_error_journal": str(run_errors_journal_path),
                "errors_by_stage": dict(sorted(errors_by_stage.items())),
                "errors_by_status": dict(sorted(errors_by_status.items())),
            },
            "errors": entries_snapshot,
        }
        try:
            _write_run_errors_aggregate(run_errors_path, payload)
        except Exception as exc:
            _processing_log(f"[processing] Failed to persist incremental run errors: {exc}")

    def _record_runtime_error(
        *,
        stage: str,
        owner: str | None,
        repo: str | None,
        pr_url: str | None = None,
        pr_number: Any = None,
        status: str = "failed",
        notes: str | None = None,
    ) -> None:
        entry = {
            "source": "runtime",
            "stage": stage,
            "repository_owner": owner,
            "repository_name": repo,
            "pr_number": pr_number,
            "pr_url": pr_url,
            "status": status,
            "notes": notes,
        }
        with run_errors_lock:
            run_runtime_errors.append(entry)
            try:
                _append_run_error_journal(run_errors_journal_path, entry)
            except Exception as exc:
                print(with_pr_progress(f"[processing] Failed to append runtime error journal: {exc}"))
            snapshot = list(run_runtime_errors)
        _flush_runtime_run_errors(snapshot)

    _flush_runtime_run_errors([], force=True)

    def _process_repo(owner: str, repo: str, repo_prs: List[Any]) -> None:
        _processing_log(f"[processing] Repository {owner}/{repo}: {len(repo_prs)} PRs")
        default_branch = None
        base_full = _get_attr(repo_prs[0], "base_repository_full")
        if base_full:
            default_branch = _get_attr(base_full, "default_branch")
        repo_hydrator = RepositoryHydrator(
            owner=owner,
            name=repo,
            clone_root=clones_root,
            default_branch=default_branch,
        )
        repo_checkpoint_path = _repo_prepare_checkpoint_path(checkpoints_root, cohort, owner, repo)
        prepared = False
        repo_prepare_started_at = time.perf_counter()
        repo_prepare_elapsed = 0.0
        attempts = 0
        max_attempts = (len(TOKENS) if TOKENS else 0) + 1
        try:
            if (
                resume
                and _is_prepared_repo_checkpoint(repo_checkpoint_path, owner, repo)
                and (repo_hydrator.repo_dir / ".git").exists()
            ):
                _processing_log(f"[processing] Reusing prepared clone for {owner}/{repo}.")
                prepared = True
                repo_prepare_elapsed = time.perf_counter() - repo_prepare_started_at
        except Exception as exc:
            _processing_log(f"[processing] Repo checkpoint load failed for {owner}/{repo}: {exc}")
            _record_runtime_error(
                stage="repo_checkpoint_load",
                owner=owner,
                repo=repo,
                status="failed",
                notes=str(exc),
            )

        while not prepared and attempts < max_attempts:
            attempts += 1
            token = None
            try:
                with token_lock:
                    token = token_manager.get_token() if token_manager else None
            except RuntimeError as exc:
                _processing_log(
                    f"[processing] No usable tokens available: {exc}. "
                    "Falling back to unauthenticated clone/fetch for public repos."
                )
                token = None
            repo_hydrator.set_token(token)
            try:
                _processing_log(
                    f"[processing] Cloning/updating {owner}/{repo} (attempt {attempts}/{max_attempts})..."
                )
                repo_hydrator.prepare()
                _write_processing_checkpoint(
                    repo_checkpoint_path,
                    _repo_prepare_checkpoint_payload(owner, repo, default_branch),
                )
                _processing_log(f"[processing] Clone ready for {owner}/{repo}.")
                prepared = True
                repo_prepare_elapsed = time.perf_counter() - repo_prepare_started_at
            except GitCommandError as exc:
                message = str(exc).lower()
                _processing_log(f"[processing] Failed to prepare repo {owner}/{repo}: {exc}")
                if repo_hydrator.repo_dir.exists() and not (repo_hydrator.repo_dir / ".git").exists():
                    repo_hydrator.cleanup()
                if token is None or not token_manager:
                    break
                if (
                    "authentication" in message
                    or "bad credentials" in message
                    or "access denied" in message
                ):
                    try:
                        with token_lock:
                            token_manager.invalidate_current()
                    except RuntimeError as exc:
                        _processing_log(f"[processing] Token invalidation failed: {exc}")
                        break
                else:
                    try:
                        with token_lock:
                            token_manager.rotate_token()
                    except RuntimeError as exc:
                        _processing_log(f"[processing] Token rotation failed: {exc}")
                        break
        if not prepared:
            _record_runtime_error(
                stage="repo_prepare",
                owner=owner,
                repo=repo,
                status="failed",
                notes="Unable to prepare repository clone after retries.",
            )
            repo_hydrator.cleanup()
            return

        try:
            pr_numbers: List[int] = []
            for repo_pr in repo_prs:
                value = _get_attr(repo_pr, "number")
                try:
                    pr_numbers.append(int(value))
                except (TypeError, ValueError):
                    continue
            fetched_refs = repo_hydrator.prefetch_pull_refs(pr_numbers)
            if fetched_refs:
                _processing_log(
                    f"[processing] Repository {owner}/{repo}: prefetched {fetched_refs} PR refs."
                )
        except Exception as exc:
            _processing_log(f"[processing] Repository {owner}/{repo}: PR ref prefetch skipped: {exc}")

        try:
            default_branch_for_manifest = repo_hydrator.resolve_default_branch()
            preferred_ref = (
                f"origin/{default_branch_for_manifest}" if default_branch_for_manifest else "HEAD"
            )
            repo_files = repo_hydrator.list_repository_files(preferred_ref)
            used_ref = preferred_ref
            if not repo_files and preferred_ref != "HEAD":
                repo_files = repo_hydrator.list_repository_files("HEAD")
                used_ref = "HEAD"
            ref_sha = repo_hydrator.resolve_ref_sha(used_ref)
            manifest_path = _write_repository_file_list(
                snapshots_root=snapshots_root,
                owner=owner,
                repo=repo,
                ref=used_ref,
                commit=ref_sha,
                files=repo_files,
            )
            _processing_log(
                f"[processing] Repository {owner}/{repo}: file list saved ({len(repo_files)} files) -> {manifest_path}"
            )
        except Exception as exc:
            _processing_log(f"[processing] Repository {owner}/{repo}: failed to save file list: {exc}")
            _record_runtime_error(
                stage="repository_file_list",
                owner=owner,
                repo=repo,
                status="failed",
                notes=str(exc),
            )

        pr_hydrator = PRHydrator(repo_hydrator, snapshots_root)
        future_count = 0
        clone_overhead_per_pr = (
            (repo_prepare_elapsed / float(len(repo_prs)))
            if repo_prs and repo_prepare_elapsed > 0
            else 0.0
        )

        for pr in repo_prs:
            pr_url = _get_attr(pr, "url") or "unknown"
            pr_id = _pr_id(pr)
            pr_language = _pr_language_label(pr)
            benchmark_language = _benchmark_language(pr)
            current_pr_index = _next_processed_index()
            set_current_pr_progress(current_pr_index, total_prs)
            pr_started_at = time.perf_counter()

            def _current_pr_timing() -> Dict[str, float]:
                elapsed = time.perf_counter() - pr_started_at
                elapsed_with_clone = elapsed + clone_overhead_per_pr
                return {
                    "processing_time_seconds": float(elapsed_with_clone),
                    "processing_time_seconds_excluding_clone": float(elapsed),
                    "clone_time_share_seconds": float(clone_overhead_per_pr),
                }

            def _log_pr_elapsed(status: str, timing_override: Dict[str, float] | None = None) -> None:
                timing = timing_override or _current_pr_timing()
                elapsed_with_clone = float(timing.get("processing_time_seconds", 0.0))
                if benchmark_language and status not in {
                    "already_completed",
                    "already_persisted",
                    "checkpoint_load_failed",
                }:
                    bucket = _timing_bucket(include_future)

                    def _update_timing() -> None:
                        state["language_time_total_seconds"][benchmark_language][bucket] += elapsed_with_clone
                        state["language_time_counts"][benchmark_language][bucket] += 1

                    _with_state_lock(_update_timing)
                _processing_log(
                    "[processing] PR {pr_id} (lang={lang}, status={status}) took {seconds:.2f}s (includes {clone_share:.2f}s clone share)".format(
                        pr_id=pr_id,
                        lang=pr_language,
                        status=status,
                        seconds=elapsed_with_clone,
                        clone_share=clone_overhead_per_pr,
                    )
                )
                def _inc_completed() -> int:
                    state["total_prs_completed"] += 1
                    return int(state["total_prs_completed"])

                completed_count = _with_state_lock(_inc_completed)
                _processing_log(
                    f"[processing] Progress: completed {completed_count}/{total_prs} PRs"
                )

            checkpoint_path = _processing_checkpoint_path(checkpoints_root, cohort, owner, repo, pr)
            include_future = pr_url in longitudinal_selected
            try:
                with completed_lock:
                    if not force_reprocess and pr_url in completed:
                        _delete_processing_checkpoint(checkpoint_path)
                        _log_pr_elapsed("already_completed")
                        continue
                with storage_lock:
                    already_recorded = (not force_reprocess) and storage.is_recorded(pr_url)
                if already_recorded:
                    _processing_log(f"[processing] PR {pr_url} already persisted; restoring progress marker.")
                    with completed_lock:
                        _append_processing_progress(progress_path, pr_url)
                        completed.add(pr_url)
                    _delete_processing_checkpoint(checkpoint_path)
                    _log_pr_elapsed("already_persisted")
                    continue
                _processing_log(f"[processing] PR {current_pr_index}/{total_prs}: {pr_url}")

                try:
                    checkpoint_record = _load_processing_checkpoint(checkpoint_path) if resume else None
                except Exception as exc:
                    _processing_log(f"[processing] PR {pr_url} checkpoint load failed: {exc}")
                    _record_runtime_error(
                        stage="pr_checkpoint_load",
                        owner=owner,
                        repo=repo,
                        pr_url=pr_url,
                        pr_number=pr_id,
                        status="failed",
                        notes=str(exc),
                    )
                    _log_pr_elapsed("checkpoint_load_failed")
                    continue

                checkpoint_state: Dict[str, Any] = {}
                if isinstance(checkpoint_record, dict):
                    normalized_checkpoint = _normalize_checkpoint_record(checkpoint_record, pr_url)
                    if not normalized_checkpoint and checkpoint_record:
                        _processing_log(f"[processing] PR {pr_url}: discarding stale checkpoint state.")
                    checkpoint_state = normalized_checkpoint
                hydration = (
                    checkpoint_state.get("hydration")
                    if isinstance(checkpoint_state.get("hydration"), dict)
                    else None
                )
                if hydration is not None and bool(hydration.get("longitudinal_selected")) != include_future:
                    _processing_log(
                        f"[processing] PR {pr_url}: discarding checkpoint because longitudinal selection changed."
                    )
                    hydration = None
                    checkpoint_state.pop("hydration", None)
                    checkpoint_state.pop("metrics_progress", None)
                if hydration:
                    _processing_log(
                        f"[processing] PR {pr_url}: resuming from checkpointed hydration/metrics state."
                    )

                def _checkpoint_hydration_progress(hydration_payload: Dict[str, Any]) -> None:
                    _write_processing_checkpoint(
                        checkpoint_path,
                        _build_pr_processing_checkpoint(
                            pr_url=pr_url,
                            current_phase="hydrating",
                            hydration=hydration_payload,
                            metrics_progress=checkpoint_state.get("metrics_progress")
                            if isinstance(checkpoint_state.get("metrics_progress"), dict)
                            else None,
                        ),
                    )

                try:
                    hydration = pr_hydrator.hydrate(
                        pr,
                        include_future=include_future,
                        existing_hydration=hydration,
                        progress_callback=_checkpoint_hydration_progress,
                    )
                    _write_processing_checkpoint(
                        checkpoint_path,
                        _build_pr_processing_checkpoint(
                            pr_url=pr_url,
                            current_phase="hydrated",
                            hydration=hydration,
                            metrics_progress=checkpoint_state.get("metrics_progress")
                            if isinstance(checkpoint_state.get("metrics_progress"), dict)
                            else None,
                        ),
                    )
                except Exception as exc:
                    _processing_log(f"[processing] PR {pr_url} hydration failed: {exc}")
                    _record_runtime_error(
                        stage="hydration",
                        owner=owner,
                        repo=repo,
                        pr_url=pr_url,
                        pr_number=pr_id,
                        status="failed",
                        notes=str(exc),
                    )
                    _log_pr_elapsed("hydration_failed")
                    continue

                existing_metrics = (
                    checkpoint_state.get("metrics_progress")
                    if isinstance(checkpoint_state.get("metrics_progress"), dict)
                    else None
                )

                def _checkpoint_metric_progress(metrics_payload: Dict[str, Any]) -> None:
                    checkpoint_state["metrics_progress"] = metrics_payload
                    _write_processing_checkpoint(
                        checkpoint_path,
                        _build_pr_processing_checkpoint(
                            pr_url=pr_url,
                            current_phase=str(metrics_payload.get("current_phase", "metrics")),
                            hydration=hydration,
                            metrics_progress=metrics_payload,
                        ),
                    )

                try:
                    metrics = compute_pr_metrics(
                        pr,
                        hydration,
                        repo_hydrator,
                        cohort=cohort,
                        include_future=include_future,
                        existing_metrics=existing_metrics,
                        progress_callback=_checkpoint_metric_progress,
                    )
                except Exception as exc:
                    _processing_log(f"[processing] PR {pr_url} metric computation failed: {exc}")
                    _record_runtime_error(
                        stage="metrics_compute",
                        owner=owner,
                        repo=repo,
                        pr_url=pr_url,
                        pr_number=pr_id,
                        status="failed",
                        notes=str(exc),
                    )
                    _increment_metrics_failed()
                    _log_pr_elapsed("metrics_failed")
                    continue

                tool_gate_failures = _tool_gate_failure_reasons(metrics)
                if tool_gate_failures:
                    _processing_log(
                        "[processing] PR {url} skipped: tool gate failed ({reasons})".format(
                            url=pr_url,
                            reasons=", ".join(tool_gate_failures),
                        )
                    )
                    _record_runtime_error(
                        stage="tool_gate",
                        owner=owner,
                        repo=repo,
                        pr_url=pr_url,
                        pr_number=pr_id,
                        status="failed",
                        notes="; ".join(tool_gate_failures),
                    )
                    try:
                        _cleanup_skipped_pr_artifacts(
                            snapshots_root=snapshots_root,
                            output_root=output_root,
                            cohort=cohort,
                            owner=owner,
                            repo=repo,
                            pr=pr,
                        )
                    except Exception as exc:
                        _processing_log(
                            f"[processing] PR {pr_url} artifact cleanup after tool gate failure failed: {exc}"
                        )
                        _record_runtime_error(
                            stage="tool_gate_cleanup",
                            owner=owner,
                            repo=repo,
                            pr_url=pr_url,
                            pr_number=pr_id,
                            status="failed",
                            notes=str(exc),
                        )
                    _increment_metrics_failed()
                    _delete_processing_checkpoint(checkpoint_path)
                    _log_pr_elapsed("tool_gate_failed")
                    continue

                _write_processing_checkpoint(
                    checkpoint_path,
                    _build_pr_processing_checkpoint(
                        pr_url=pr_url,
                        current_phase="persisting",
                        hydration=hydration,
                        metrics_progress=checkpoint_state.get("metrics_progress")
                        if isinstance(checkpoint_state.get("metrics_progress"), dict)
                        else None,
                    ),
                )
                record = _sanitize_persist_record(pr_to_dict(pr))
                record["primary_language"] = _get_attr(pr, "primary_language")
                record["pr_primary_language_effective"] = _get_attr(
                    pr, "pr_primary_language_effective"
                )
                record["longitudinal_selected"] = include_future
                repository_metadata = _merge_repository_metadata(pr)
                record["repository_metadata"] = repository_metadata
                record["hydration"] = hydration
                record["metrics"] = metrics
                record["metrics_backend"] = (
                    metrics.get("metrics_backend") if isinstance(metrics, dict) else None
                )
                pr_timing = _current_pr_timing()
                record["processing_timing"] = pr_timing
                if isinstance(metrics, dict):
                    metrics["processing_timing"] = pr_timing
                aggregate_path: Path | None = None
                try:
                    aggregate_payload = _build_pr_metrics_aggregate(
                        pr=pr,
                        cohort=cohort,
                        owner=owner,
                        repo=repo,
                        hydration=hydration,
                        metrics=metrics,
                        repository_metadata=repository_metadata,
                        include_future=include_future,
                        processing_timing=pr_timing,
                    )
                    # Keep the same detailed aggregate payload in parquet rows as well.
                    record["metrics_aggregate"] = aggregate_payload
                    aggregate_path = _aggregate_metrics_path(output_root, cohort, owner, repo, pr)
                    _write_metrics_aggregate(aggregate_path, aggregate_payload)
                    record["metrics_aggregate_path"] = str(aggregate_path)
                except Exception as exc:
                    _processing_log(f"[processing] PR {pr_url} aggregate metrics write failed: {exc}")
                    _record_runtime_error(
                        stage="aggregate_write",
                        owner=owner,
                        repo=repo,
                        pr_url=pr_url,
                        pr_number=pr_id,
                        status="failed",
                        notes=str(exc),
                    )
                try:
                    with storage_lock:
                        stored = storage.persist_one(record, base_name="processed_pr", group=cohort)
                        persisted = bool(stored or storage.is_recorded(pr_url))
                except Exception as exc:
                    _processing_log(f"[processing] PR {pr_url} persistence failed: {exc}")
                    _record_runtime_error(
                        stage="persistence",
                        owner=owner,
                        repo=repo,
                        pr_url=pr_url,
                        pr_number=pr_id,
                        status="failed",
                        notes=str(exc),
                    )
                    _log_pr_elapsed("persistence_failed")
                    continue
                if not persisted:
                    _processing_log(f"[processing] PR {pr_url} was not persisted; leaving checkpoint in place.")
                    _record_runtime_error(
                        stage="persistence",
                        owner=owner,
                        repo=repo,
                        pr_url=pr_url,
                        pr_number=pr_id,
                        status="not_persisted",
                        notes="Record not persisted by storage handler.",
                    )
                    _log_pr_elapsed("not_persisted")
                    continue
                with completed_lock:
                    _append_processing_progress(progress_path, pr_url)
                    completed.add(pr_url)
                _delete_processing_checkpoint(checkpoint_path)
                try:
                    cleaned_snapshot_root, snapshot_root_deleted = _cleanup_persisted_pr_artifacts(
                        snapshots_root=snapshots_root,
                        owner=owner,
                        repo=repo,
                        pr=pr,
                    )
                    if (
                        snapshot_root_deleted
                        and aggregate_path is not None
                        and aggregate_path.exists()
                    ):
                        try:
                            _mark_aggregate_artifacts_cleaned(
                                aggregate_path,
                                cleaned_snapshot_root,
                            )
                        except Exception as exc:
                            _processing_log(
                                "[processing] PR {url} aggregate artifact cleanup marker failed: {error}".format(
                                    url=pr_url,
                                    error=exc,
                                )
                            )
                    if snapshot_root_deleted:
                        _processing_log(
                            "[processing] PR {url}: cleaned transient artifacts at {root}".format(
                                url=pr_url,
                                root=cleaned_snapshot_root,
                            )
                        )
                    elif DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING:
                        _processing_log(
                            "[processing] PR {url}: no transient snapshot artifacts found at {root}".format(
                                url=pr_url,
                                root=cleaned_snapshot_root,
                            )
                        )
                    else:
                        _processing_log(
                            "[processing] PR {url}: retained snapshot artifacts at {root}".format(
                                url=pr_url,
                                root=cleaned_snapshot_root,
                            )
                        )
                except Exception as exc:
                    _processing_log(
                        f"[processing] PR {pr_url}: transient artifact cleanup after persistence failed: {exc}"
                    )
                    _record_runtime_error(
                        stage="artifact_cleanup",
                        owner=owner,
                        repo=repo,
                        pr_url=pr_url,
                        pr_number=pr_id,
                        status="warning",
                        notes=str(exc),
                    )

                snapshots = hydration.get("snapshots") or {}
                before_present = bool(snapshots.get("before"))
                after_present = bool(snapshots.get("after"))
                future_snapshots = snapshots.get("future") or {}

                def _update_success_state() -> None:
                    state["total_prs_processed"] += 1
                    state["metrics_count"] += 1
                    if before_present:
                        state["before_count"] += 1
                    if after_present:
                        state["after_count"] += 1
                    for label, meta in future_snapshots.items():
                        if (
                            isinstance(meta, dict)
                            and meta.get("commit")
                            and meta.get("available") is not False
                        ):
                            current = int(state["future_counts"].get(label, 0))
                            state["future_counts"][label] = current + 1
                    if hydration.get("has_future_snapshots"):
                        state["total_with_future"] += 1

                _with_state_lock(_update_success_state)
                if hydration.get("has_future_snapshots"):
                    future_count += 1
                if before_present or after_present:
                    _processing_log(
                        "[processing] PR {url}: current snapshots before={before} after={after}".format(
                            url=pr_url,
                            before="yes" if before_present else "no",
                            after="yes" if after_present else "no",
                        )
                    )
                if hydration.get("has_future_snapshots"):
                    _processing_log(f"[processing] PR {pr_url}: future snapshots found.")
                _processing_log(
                    "[processing] PR {url}: snapshots before={before} after={after} future={future}".format(
                        url=pr_url,
                        before="yes" if hydration["snapshots"].get("before") else "no",
                        after="yes" if hydration["snapshots"].get("after") else "no",
                        future="yes" if hydration.get("has_future_snapshots") else "no",
                    )
                )
                _log_pr_elapsed("success", timing_override=pr_timing)
            finally:
                clear_current_pr_progress()

        _processing_log(
            f"[processing] Repository {owner}/{repo}: future snapshots on {future_count}/{len(repo_prs)} PRs"
        )
        repo_hydrator.cleanup()
        _delete_processing_checkpoint(repo_checkpoint_path)

    repo_items = [((owner, repo), repo_prs) for (owner, repo), repo_prs in grouped.items()]
    unknown_repo_prs = 0
    for (owner, repo), repo_prs in repo_items:
        if not owner or not repo:
            unknown_repo_prs += len(repo_prs)
    if unknown_repo_prs:
        _processing_log("[processing] Skipping PRs with unknown repository identifiers.")

    valid_repo_items = [item for item in repo_items if item[0][0] and item[0][1]]
    repo_workers = max(1, int(PROCESSING_REPO_WORKERS))
    prefetch_limit = max(repo_workers, int(PROCESSING_PREFETCH_REPOS))
    _processing_log(
        f"[processing] Parallel repo processing enabled: workers={repo_workers} prefetch={prefetch_limit}"
    )

    repo_queue: Queue[Any] = Queue(maxsize=prefetch_limit)
    sentinel = object()

    def _repo_worker_loop() -> None:
        while True:
            try:
                item = repo_queue.get(timeout=0.25)
            except Empty:
                continue
            if item is sentinel:
                repo_queue.task_done()
                break
            (owner, repo), repo_prs = item
            try:
                _process_repo(str(owner), str(repo), list(repo_prs))
            except Exception as exc:
                _processing_log(f"[processing] Repository {owner}/{repo}: worker failed: {exc}")
                _record_runtime_error(
                    stage="repo_worker",
                    owner=str(owner),
                    repo=str(repo),
                    status="failed",
                    notes=str(exc),
                )
            finally:
                repo_queue.task_done()

    workers = [
        threading.Thread(target=_repo_worker_loop, name=f"repo-worker-{idx+1}", daemon=True)
        for idx in range(repo_workers)
    ]
    for worker in workers:
        worker.start()

    for item in valid_repo_items:
        repo_queue.put(item)
    for _ in workers:
        repo_queue.put(sentinel)

    repo_queue.join()
    for worker in workers:
        worker.join()

    with storage_lock:
        storage.flush_local()
    language_time_avg_seconds: Dict[str, Dict[str, float]] = {
        lang: {} for lang in benchmark_languages
    }
    language_time_counts = state["language_time_counts"]
    language_time_total_seconds = state["language_time_total_seconds"]
    for lang in benchmark_languages:
        for bucket in timing_buckets:
            count = int(language_time_counts.get(lang, {}).get(bucket, 0))
            total = float(language_time_total_seconds.get(lang, {}).get(bucket, 0.0))
            language_time_avg_seconds[lang][bucket] = (total / count) if count else 0.0
    _processing_log("[processing] Average PR processing time by language (temporary benchmark, split):")
    for lang in benchmark_languages:
        with_count = int(language_time_counts.get(lang, {}).get("with_longitudinal", 0))
        with_avg = float(language_time_avg_seconds.get(lang, {}).get("with_longitudinal", 0.0))
        without_count = int(language_time_counts.get(lang, {}).get("without_longitudinal", 0))
        without_avg = float(
            language_time_avg_seconds.get(lang, {}).get("without_longitudinal", 0.0)
        )
        _processing_log(
            "[processing]   - {lang}: with_longitudinal avg={with_avg:.2f}s count={with_count}; "
            "without_longitudinal avg={without_avg:.2f}s count={without_count}".format(
                lang=lang,
                with_avg=with_avg,
                with_count=with_count,
                without_avg=without_avg,
                without_count=without_count,
            )
        )
    _processing_log(f"[processing] Completed. PRs with future snapshots: {int(state['total_with_future'])}")
    harvested = _collect_run_errors_from_metrics(output_root, cohort)
    with run_errors_lock:
        runtime_entries = list(run_runtime_errors)
    all_entries = runtime_entries + _as_list(harvested.get("entries"))
    errors_by_stage: Dict[str, int] = {}
    errors_by_status: Dict[str, int] = {}
    for entry in all_entries:
        item = _as_dict(entry)
        stage_key = str(item.get("stage") or "unknown")
        status_key = str(item.get("status") or "unknown")
        errors_by_stage[stage_key] = int(errors_by_stage.get(stage_key, 0)) + 1
        errors_by_status[status_key] = int(errors_by_status.get(status_key, 0)) + 1
    run_errors_payload = {
        "schema_version": 1,
        "cohort": cohort,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_errors": len(all_entries),
            "runtime_errors": len(runtime_entries),
            "aggregated_errors": len(_as_list(harvested.get("entries"))),
            "aggregate_files_scanned": int(harvested.get("source_files_scanned", 0) or 0),
            "runtime_error_journal": str(run_errors_journal_path),
            "errors_by_stage": dict(sorted(errors_by_stage.items())),
            "errors_by_status": dict(sorted(errors_by_status.items())),
        },
        "errors": all_entries,
    }
    _write_run_errors_aggregate(run_errors_path, run_errors_payload)
    _processing_log(f"[processing] Run error report written: {run_errors_path}")
    return {
        "processed_prs": int(state["total_prs_processed"]),
        "longitudinal_selected_prs": len(longitudinal_selected),
        "metrics_computed": int(state["metrics_count"]),
        "metrics_failed": int(state["metrics_failed"]),
        "with_before": int(state["before_count"]),
        "with_after": int(state["after_count"]),
        "with_future_any": int(state["total_with_future"]),
        "future_counts": dict(sorted(state["future_counts"].items())),
        "language_time_total_seconds": {
            lang: dict(language_time_total_seconds.get(lang, {})) for lang in benchmark_languages
        },
        "language_time_counts": {
            lang: dict(language_time_counts.get(lang, {})) for lang in benchmark_languages
        },
        "language_time_avg_seconds": {
            lang: dict(language_time_avg_seconds.get(lang, {})) for lang in benchmark_languages
        },
        "run_errors_path": str(run_errors_path),
        "run_errors_journal_path": str(run_errors_journal_path),
        "run_errors_total": len(all_entries),
        "skipped_failed_on_resume": int(state["skipped_failed_on_resume"]),
        "prefiltered_completed_on_resume": int(state["prefiltered_completed_on_resume"]),
        "prefiltered_persisted_on_resume": int(state["prefiltered_persisted_on_resume"]),
    }
