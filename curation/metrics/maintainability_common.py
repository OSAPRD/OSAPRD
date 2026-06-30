"""Shared maintainability helpers used by the active Multimetric stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from curation.config.maintainability_config import FUTURE_MAINTAINABILITY_SNAPSHOT_LABELS
from curation.pipeline.progress_context import with_pr_progress

LOG_PREFIX = "[maintainability]"
logger = logging.getLogger(__name__)


def _get_pr_number(pr: Any) -> Any:
    """Return a PR number from either a dict row or DTO-like object."""
    return getattr(pr, "number", None) if not isinstance(pr, dict) else pr.get("number")


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from a dict or object."""
    return getattr(obj, name, default) if not isinstance(obj, dict) else obj.get(name, default)


def _log_info(message: str) -> None:
    """Log and print a progress-aware maintainability message."""
    logger.info(message)
    print(with_pr_progress(f"{LOG_PREFIX} {message}"))


def _log_error(message: str) -> None:
    """Log and print a progress-aware maintainability error."""
    logger.error(message)
    print(with_pr_progress(f"{LOG_PREFIX} ERROR: {message}"))


def _write_json_artifact(path: Path, payload: Any) -> str:
    """Write one JSON artifact atomically and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)
    return str(path)


def _future_impact_summary(hydration: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize future diff overlap with the PR's changed files and lines."""
    future_tracking = (hydration.get("diff_tracking") or {}).get("future") or {}
    per_snapshot: Dict[str, Dict[str, Any]] = {}
    snapshots_with_file_overlap = 0
    snapshots_with_line_overlap = 0
    total_touched_pr_lines = 0
    for label in FUTURE_MAINTAINABILITY_SNAPSHOT_LABELS:
        tracking = future_tracking.get(label)
        if not isinstance(tracking, dict):
            continue
        overlap = tracking.get("overlap_with_pr") or {}
        summary = {
            "changed_files_count": tracking.get("changed_files_count", 0),
            "changed_paths": tracking.get("changed_paths") or [],
            "touched_pr_files_count": overlap.get("touched_pr_files_count", 0),
            "touched_pr_files": overlap.get("touched_pr_files") or [],
            "touched_pr_lines_count": overlap.get("touched_pr_lines_count", 0),
            "touched_pr_line_files": overlap.get("touched_pr_line_files") or [],
            "has_file_overlap": bool(overlap.get("has_file_overlap")),
            "has_line_overlap": bool(overlap.get("has_line_overlap")),
        }
        per_snapshot[label] = summary
        if summary["has_file_overlap"]:
            snapshots_with_file_overlap += 1
        if summary["has_line_overlap"]:
            snapshots_with_line_overlap += 1
        total_touched_pr_lines += int(summary["touched_pr_lines_count"])
    return {
        "per_snapshot": per_snapshot,
        "snapshots_with_file_overlap": snapshots_with_file_overlap,
        "snapshots_with_line_overlap": snapshots_with_line_overlap,
        "total_touched_pr_lines": total_touched_pr_lines,
    }


def _future_snapshot_has_no_pr_file_overlap(hydration: Dict[str, Any], label: str) -> bool:
    """Return True when a computed future diff does not touch any PR file."""
    future_tracking = (hydration.get("diff_tracking") or {}).get("future") or {}
    tracking = future_tracking.get(label)
    if not isinstance(tracking, dict):
        return False
    overlap = tracking.get("overlap_with_pr")
    if not isinstance(overlap, dict):
        return False
    return not bool(overlap.get("has_file_overlap"))


def _future_snapshot_cumulative_touch_count(
    hydration: Dict[str, Any],
    label: str,
) -> Optional[int]:
    """Return commits touching PR files from after-PR through the target snapshot."""
    activity = hydration.get("longitudinal_commit_activity")
    if not isinstance(activity, dict):
        return None
    if int(activity.get("tracked_pr_files_count") or 0) <= 0:
        return None
    intervals = activity.get("future_snapshot_intervals")
    if not isinstance(intervals, dict) or label not in intervals:
        return None
    total = 0
    seen_target = False
    for current_label in FUTURE_MAINTAINABILITY_SNAPSHOT_LABELS:
        interval = intervals.get(current_label)
        if not isinstance(interval, dict):
            continue
        if interval.get("commit_activity_status") != "ok":
            return None
        if interval.get("ancestry_status") is not True:
            return None
        count = interval.get("touching_commits_count")
        if count is None:
            return None
        total += int(count)
        if current_label == label:
            seen_target = True
            break
    return total if seen_target else None


def _future_snapshot_tool_skip_reason(
    hydration: Dict[str, Any],
    label: str,
) -> Optional[str]:
    """Return why a future metric snapshot can reuse the after-PR result."""
    cumulative_touches = _future_snapshot_cumulative_touch_count(hydration, label)
    if cumulative_touches == 0:
        return "future_no_cumulative_commit_touches"
    if cumulative_touches is not None:
        return None
    if _future_snapshot_has_no_pr_file_overlap(hydration, label):
        return "future_no_pr_file_overlap"
    return None
