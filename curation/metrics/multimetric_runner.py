"""Shared helpers for running Multimetric over materialized snapshots.

The maintainability stage builds one :class:`SnapshotTask` per before/after or
future snapshot. This module handles source-file discovery, command execution,
tool output normalization, and checkpoint-row reuse.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SOURCE_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".cc",
    ".cpp",
    ".cxx",
    ".hpp",
    ".hh",
    ".hxx",
}
EXCLUDED_DIRECTORY_NAMES = {".git", "node_modules", "maintainability"}
METRIC_FIELDS = (
    "loc",
    "comment_ratio",
    "cyclomatic_complexity",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugprop",
    "halstead_timerequired",
    "maintainability_index",
    "fanout_internal",
    "fanout_external",
    "operands_sum",
    "operands_uniq",
    "operators_sum",
    "operators_uniq",
    "pylint",
    "tiobe",
    "tiobe_complexity",
    "tiobe_duplication",
    "tiobe_functional",
)
DEFAULT_MAINTINDEX_MODE = "sei"


@dataclass(frozen=True)
class SnapshotTask:
    """One maintainability snapshot to analyze."""

    cohort: str | None
    repository_owner: str | None
    repository_name: str | None
    repository_key: str | None
    pr_number: int | None
    pr_url: str | None
    snapshot_kind: str
    snapshot_label: str
    days_after_merge: int | None
    snapshot_commit: str | None
    snapshot_path: Path | None


def utc_now_text() -> str:
    """Return the current UTC timestamp in JSON-friendly text form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_id(*parts: object) -> str:
    """Return a stable SHA-256 id for a tuple of identifying parts."""
    normalized = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def json_dumps(value: Any) -> str | None:
    """Serialize nested metric payloads for parquet-compatible fields."""
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def safe_float(value: Any) -> float | None:
    """Coerce a value to float, preserving None for missing/bad values."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_excluded_source_path(path: Path) -> bool:
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in path.parts):
        return True
    name = path.name
    if name == "diff":
        return True
    if name.upper().startswith("README"):
        return True
    if name.endswith("_tool_results.json"):
        return True
    return False


def discover_source_files(snapshot_path: Path) -> list[Path]:
    """Return supported source files under a hydrated snapshot."""
    if not snapshot_path.exists() or not snapshot_path.is_dir():
        return []
    source_files: list[Path] = []
    for path in snapshot_path.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(snapshot_path)
        except ValueError:
            rel = path
        if _is_excluded_source_path(rel):
            continue
        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        source_files.append(path)
    return sorted(source_files, key=lambda item: item.relative_to(snapshot_path).as_posix())


def tool_version(multimetric_bin: str = "multimetric") -> str | None:
    """Return the installed Multimetric package version when available."""
    try:
        return importlib.metadata.version("multimetric")
    except importlib.metadata.PackageNotFoundError:
        pass
    return "available" if shutil.which(multimetric_bin) else None


def multimetric_command(
    *,
    multimetric_bin: str,
    maintindex_mode: str,
    relative_files: Sequence[str],
    jobs: int | None = None,
) -> list[str]:
    """Build the Multimetric CLI command for one snapshot."""
    command = [multimetric_bin, "--maintindex", maintindex_mode]
    if jobs is not None:
        command.extend(["--jobs", str(max(1, int(jobs)))])
    command.extend(relative_files)
    return command


def _command_length(command: Sequence[str]) -> int:
    """Estimate command-line length for platform guardrails."""
    return sum(len(str(part)) + 1 for part in command)


def _command_length_limit() -> int:
    """Return a conservative command-length limit for the current platform."""
    return 30_000 if os.name == "nt" else 120_000


def snapshot_metric_id(task: SnapshotTask) -> str:
    """Return the stable row id for one snapshot metric task."""
    return stable_id(
        "snapshot",
        task.repository_key,
        task.pr_number,
        task.snapshot_label,
        task.snapshot_commit,
        task.snapshot_path,
    )


def empty_snapshot_row(
    task: SnapshotTask,
    *,
    status: str,
    error_message: str | None,
    files_considered: int,
    files_analyzed: int,
    tool_version_value: str | None,
    maintindex_mode: str,
    created_at_utc: str,
    snapshot_metric_id_value: str | None = None,
) -> dict[str, Any]:
    """Build a complete snapshot row for missing/failed/no-source cases."""
    row = {
        "snapshot_metric_id": snapshot_metric_id_value or snapshot_metric_id(task),
        "cohort": task.cohort,
        "repository_owner": task.repository_owner,
        "repository_name": task.repository_name,
        "repository_key": task.repository_key,
        "pr_number": task.pr_number,
        "pr_url": task.pr_url,
        "snapshot_kind": task.snapshot_kind,
        "snapshot_label": task.snapshot_label,
        "days_after_merge": task.days_after_merge,
        "snapshot_commit": task.snapshot_commit,
        "snapshot_path": str(task.snapshot_path) if task.snapshot_path else None,
        "tool": "multimetric",
        "tool_version": tool_version_value,
        "maintindex_mode": maintindex_mode,
        "status": status,
        "error_message": error_message,
        "files_considered": files_considered,
        "files_analyzed": files_analyzed,
        "multimetric_overall_json": None,
        "multimetric_stats_json": None,
        "multimetric_reused": False,
        "reused_from_snapshot_metric_id": None,
        "reuse_reason": None,
        "created_at_utc": created_at_utc,
    }
    for field in METRIC_FIELDS:
        row[field] = None
    return row


def _metric_columns_from_mapping(mapping: Any) -> dict[str, float | None]:
    """Extract known numeric metric columns from a Multimetric mapping."""
    source = mapping if isinstance(mapping, dict) else {}
    return {field: safe_float(source.get(field)) for field in METRIC_FIELDS}


def _count_multimetric_file_items(raw_files: Any) -> int:
    """Count file-level entries in Multimetric's raw output payload."""
    if isinstance(raw_files, dict):
        return sum(1 for value in raw_files.values() if isinstance(value, dict))
    if isinstance(raw_files, list):
        count = 0
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            if len(item) == 1:
                _, only_value = next(iter(item.items()))
                if isinstance(only_value, dict):
                    count += 1
                    continue
            count += 1
        return count
    return 0


def row_from_multimetric_payload(
    *,
    task: SnapshotTask,
    payload: dict[str, Any],
    source_files: Sequence[Path],
    tool_version_value: str | None,
    maintindex_mode: str,
    created_at_utc: str,
    snapshot_metric_id_value: str | None = None,
) -> dict[str, Any]:
    """Normalize Multimetric JSON output into the persisted snapshot row."""
    row_id = snapshot_metric_id_value or snapshot_metric_id(task)
    overall = payload.get("overall")
    if not isinstance(overall, dict):
        return empty_snapshot_row(
            task,
            status="parse_failed",
            error_message="multimetric JSON did not contain an object at 'overall'",
            files_considered=len(source_files),
            files_analyzed=0,
            tool_version_value=tool_version_value,
            maintindex_mode=maintindex_mode,
            created_at_utc=created_at_utc,
            snapshot_metric_id_value=row_id,
        )

    files_reported = _count_multimetric_file_items(payload.get("files"))
    row = empty_snapshot_row(
        task,
        status="success",
        error_message=None,
        files_considered=len(source_files),
        files_analyzed=files_reported if files_reported else len(source_files),
        tool_version_value=tool_version_value,
        maintindex_mode=maintindex_mode,
        created_at_utc=created_at_utc,
        snapshot_metric_id_value=row_id,
    )
    row.update(_metric_columns_from_mapping(overall))
    row["multimetric_overall_json"] = json_dumps(overall)
    row["multimetric_stats_json"] = json_dumps(payload.get("stats"))
    return row


def run_multimetric_for_snapshot(
    task: SnapshotTask,
    *,
    multimetric_bin: str = "multimetric",
    maintindex_mode: str = DEFAULT_MAINTINDEX_MODE,
    jobs: int | None = None,
    tool_version_value: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run Multimetric for one snapshot and return one normalized row."""
    created_at = created_at_utc or utc_now_text()
    version = tool_version_value if tool_version_value is not None else tool_version(multimetric_bin)
    snapshot_path = task.snapshot_path
    if snapshot_path is None or not snapshot_path.exists() or not snapshot_path.is_dir():
        return empty_snapshot_row(
            task,
            status="missing_snapshot",
            error_message="snapshot path is missing or is not a directory",
            files_considered=0,
            files_analyzed=0,
            tool_version_value=version,
            maintindex_mode=maintindex_mode,
            created_at_utc=created_at,
        )

    source_files = discover_source_files(snapshot_path)
    if not source_files:
        return empty_snapshot_row(
            task,
            status="no_source_files",
            error_message="no supported source files found in snapshot",
            files_considered=0,
            files_analyzed=0,
            tool_version_value=version,
            maintindex_mode=maintindex_mode,
            created_at_utc=created_at,
        )

    relative_files = [path.relative_to(snapshot_path).as_posix() for path in source_files]
    command = multimetric_command(
        multimetric_bin=multimetric_bin,
        maintindex_mode=maintindex_mode,
        relative_files=relative_files,
        jobs=jobs,
    )
    if _command_length(command) > _command_length_limit():
        return empty_snapshot_row(
            task,
            status="tool_failed",
            error_message="command_too_long",
            files_considered=len(source_files),
            files_analyzed=0,
            tool_version_value=version,
            maintindex_mode=maintindex_mode,
            created_at_utc=created_at,
        )

    try:
        proc = subprocess.run(
            command,
            cwd=snapshot_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return empty_snapshot_row(
            task,
            status="tool_failed",
            error_message=str(exc),
            files_considered=len(source_files),
            files_analyzed=0,
            tool_version_value=version,
            maintindex_mode=maintindex_mode,
            created_at_utc=created_at,
        )

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        return empty_snapshot_row(
            task,
            status="tool_failed",
            error_message=message[:4000] if message else f"multimetric exited {proc.returncode}",
            files_considered=len(source_files),
            files_analyzed=0,
            tool_version_value=version,
            maintindex_mode=maintindex_mode,
            created_at_utc=created_at,
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return empty_snapshot_row(
            task,
            status="parse_failed",
            error_message=f"invalid multimetric JSON: {exc}",
            files_considered=len(source_files),
            files_analyzed=0,
            tool_version_value=version,
            maintindex_mode=maintindex_mode,
            created_at_utc=created_at,
        )
    if not isinstance(payload, dict):
        return empty_snapshot_row(
            task,
            status="parse_failed",
            error_message="multimetric JSON root was not an object",
            files_considered=len(source_files),
            files_analyzed=0,
            tool_version_value=version,
            maintindex_mode=maintindex_mode,
            created_at_utc=created_at,
        )

    return row_from_multimetric_payload(
        task=task,
        payload=payload,
        source_files=source_files,
        tool_version_value=version,
        maintindex_mode=maintindex_mode,
        created_at_utc=created_at,
    )


def reuse_snapshot_row(
    *,
    source_row: dict[str, Any],
    target_task: SnapshotTask,
    reuse_reason: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a target snapshot row by reusing another snapshot's metrics."""
    target_id = snapshot_metric_id(target_task)
    reused = dict(source_row)
    source_id = reused.get("snapshot_metric_id")
    reused.update(
        {
            "snapshot_metric_id": target_id,
            "cohort": target_task.cohort,
            "repository_owner": target_task.repository_owner,
            "repository_name": target_task.repository_name,
            "repository_key": target_task.repository_key,
            "pr_number": target_task.pr_number,
            "pr_url": target_task.pr_url,
            "snapshot_kind": target_task.snapshot_kind,
            "snapshot_label": target_task.snapshot_label,
            "days_after_merge": target_task.days_after_merge,
            "snapshot_commit": target_task.snapshot_commit,
            "snapshot_path": str(target_task.snapshot_path) if target_task.snapshot_path else None,
            "status": "success",
            "error_message": None,
            "multimetric_reused": True,
            "reused_from_snapshot_metric_id": str(source_id) if source_id else None,
            "reuse_reason": reuse_reason,
            "created_at_utc": created_at_utc or utc_now_text(),
        }
    )
    stats = {}
    raw_stats = reused.get("multimetric_stats_json")
    if isinstance(raw_stats, str) and raw_stats.strip():
        try:
            loaded = json.loads(raw_stats)
            if isinstance(loaded, dict):
                stats.update(loaded)
        except Exception:
            pass
    stats["_reuse"] = {
        "reused": True,
        "reused_from_snapshot_metric_id": source_id,
        "reuse_reason": reuse_reason,
    }
    reused["multimetric_stats_json"] = json_dumps(stats)
    return reused
