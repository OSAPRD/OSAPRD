"""Hydrate before/after/future artifacts for one pull request.

`PRHydrator` turns an enriched PR row plus a prepared repository clone into the
local source snapshots consumed by metric stages. It also records structured
diff information so later stages can track original PR lines through future
commits without rerunning refactoring tools on those future snapshots.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any, Callable, Optional

from curation.hydration.repository_hydrator import RepositoryHydrator
from curation.pipeline.progress_context import with_pr_progress
from extraction.utility.language_labeller import infer_language


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dict row or DTO-like object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse GitHub-style ISO timestamps into datetimes when possible."""
    if not dt_str:
        return None
    value = dt_str.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_iso(dt: datetime) -> str:
    """Format a datetime as a UTC GitHub-style timestamp."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_HUNK_PATTERN = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)

_LANGUAGE_ALIASES = {
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "py": "python",
    "c#": "c#",
    "cs": "c#",
    "cpp": "c++",
    "cxx": "c++",
    "c++": "c++",
}


def _normalize_diff_path(path: Optional[str]) -> Optional[str]:
    """Normalize paths from git diffs into repository-relative paths."""
    if not path:
        return None
    text = str(path).strip()
    if not text or text == "/dev/null":
        return None
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]
    return text.replace("\\", "/").lstrip("/") or None


def _normalize_language_token(language: Optional[str]) -> Optional[str]:
    """Normalize language labels used for path filtering."""
    if not language:
        return None
    token = str(language).strip().lower()
    if not token:
        return None
    return _LANGUAGE_ALIASES.get(token, token)


def _line_range(start: int, count: int) -> Optional[dict[str, int]]:
    """Return an inclusive line range or None for zero-line hunks."""
    if count <= 0:
        return None
    return {"start": start, "end": start + count - 1}


def _anchor_line(start: int, count: int) -> Optional[int]:
    """Return the anchor line for insert/delete-only hunks."""
    if count > 0:
        return None
    return max(1, start)


def _ranges_by_file(files: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, int]]]:
    """Group parsed hunk line ranges by old or new file path."""
    result: dict[str, list[dict[str, int]]] = {}
    for entry in files:
        path = entry.get(key)
        if not path:
            continue
        ranges = entry.get("old_line_ranges" if key == "old_path" else "new_line_ranges") or []
        if not ranges:
            continue
        result.setdefault(str(path), []).extend(ranges)
    return result


def _anchors_by_file(files: list[dict[str, Any]], key: str) -> dict[str, list[int]]:
    """Group zero-length hunk anchor lines by old or new file path."""
    result: dict[str, list[int]] = {}
    anchor_key = "old_anchor_lines" if key == "old_path" else "new_anchor_lines"
    for entry in files:
        path = entry.get(key)
        if not path:
            continue
        anchors = [int(value) for value in (entry.get(anchor_key) or []) if value is not None]
        if not anchors:
            continue
        result.setdefault(str(path), []).extend(anchors)
    return result


def _count_range_lines(ranges_by_file: dict[str, list[dict[str, int]]]) -> int:
    """Count total covered lines across grouped inclusive ranges."""
    total = 0
    for ranges in ranges_by_file.values():
        for line_range in ranges:
            start = int(line_range.get("start", 0))
            end = int(line_range.get("end", start - 1))
            if end >= start:
                total += end - start + 1
    return total


def _merge_line_ranges(ranges: list[dict[str, int]]) -> list[dict[str, int]]:
    """Merge overlapping or adjacent inclusive line ranges."""
    normalized: list[dict[str, int]] = []
    for line_range in ranges:
        start = int(line_range.get("start", 0))
        end = int(line_range.get("end", start - 1))
        if end < start:
            continue
        normalized.append({"start": start, "end": end})
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, int]] = []
    for line_range in normalized:
        if not merged or line_range["start"] > merged[-1]["end"] + 1:
            merged.append(dict(line_range))
            continue
        merged[-1]["end"] = max(merged[-1]["end"], line_range["end"])
    return merged


def _parse_structured_diff(diff_text: str) -> dict[str, Any]:
    """Parse a unified git diff into per-file changed line ranges."""
    files: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                files.append(current)
            parts = line.split()
            old_path = _normalize_diff_path(parts[2]) if len(parts) > 2 else None
            new_path = _normalize_diff_path(parts[3]) if len(parts) > 3 else None
            current = {
                "old_path": old_path,
                "new_path": new_path,
                "path": new_path or old_path,
                "old_line_ranges": [],
                "new_line_ranges": [],
                "old_anchor_lines": [],
                "new_anchor_lines": [],
            }
            continue
        if current is None:
            continue
        if line.startswith("--- "):
            current["old_path"] = _normalize_diff_path(line[4:])
            current["path"] = current.get("new_path") or current.get("old_path")
            continue
        if line.startswith("+++ "):
            current["new_path"] = _normalize_diff_path(line[4:])
            current["path"] = current.get("new_path") or current.get("old_path")
            continue
        match = _HUNK_PATTERN.match(line)
        if not match:
            continue
        old_start = int(match.group("old_start"))
        old_count = int(match.group("old_count") or "1")
        new_start = int(match.group("new_start"))
        new_count = int(match.group("new_count") or "1")
        old_range = _line_range(old_start, old_count)
        new_range = _line_range(new_start, new_count)
        old_anchor = _anchor_line(old_start, old_count)
        new_anchor = _anchor_line(new_start, new_count)
        if old_range:
            current["old_line_ranges"].append(old_range)
        if new_range:
            current["new_line_ranges"].append(new_range)
        if old_anchor is not None:
            current["old_anchor_lines"].append(old_anchor)
        if new_anchor is not None:
            current["new_anchor_lines"].append(new_anchor)
    if current is not None:
        files.append(current)

    old_line_ranges_by_file = _ranges_by_file(files, "old_path")
    new_line_ranges_by_file = _ranges_by_file(files, "new_path")
    old_anchor_lines_by_file = _anchors_by_file(files, "old_path")
    new_anchor_lines_by_file = _anchors_by_file(files, "new_path")
    changed_paths = sorted({str(entry.get("path")) for entry in files if entry.get("path")})
    return {
        "changed_files_count": len(changed_paths),
        "changed_paths": changed_paths,
        "files": files,
        "old_line_ranges_by_file": old_line_ranges_by_file,
        "new_line_ranges_by_file": new_line_ranges_by_file,
        "old_anchor_lines_by_file": old_anchor_lines_by_file,
        "new_anchor_lines_by_file": new_anchor_lines_by_file,
        "old_changed_line_count": _count_range_lines(old_line_ranges_by_file),
        "new_changed_line_count": _count_range_lines(new_line_ranges_by_file),
    }


def _overlapping_line_count(
    ranges_a: list[dict[str, int]],
    ranges_b: list[dict[str, int]],
) -> int:
    """Count intersecting line numbers between two range collections."""
    merged_a = _merge_line_ranges(ranges_a)
    merged_b = _merge_line_ranges(ranges_b)
    overlap = 0
    index_b = 0
    for left in merged_a:
        left_start = int(left.get("start", 0))
        left_end = int(left.get("end", left_start - 1))
        while index_b < len(merged_b) and int(merged_b[index_b].get("end", -1)) < left_start:
            index_b += 1
        scan_index = index_b
        while scan_index < len(merged_b):
            right = merged_b[scan_index]
            right_start = int(right.get("start", 0))
            right_end = int(right.get("end", right_start - 1))
            if right_start > left_end:
                break
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if end >= start:
                overlap += end - start + 1
            scan_index += 1
    return overlap


def _snapshot_export_complete(snapshot_meta: Any) -> bool:
    """Return True when a snapshot metadata entry points to an exported tree."""
    if not isinstance(snapshot_meta, dict):
        return False
    commit = snapshot_meta.get("commit")
    path = snapshot_meta.get("path")
    if not commit or not path:
        return False
    return Path(path).exists()


def _diff_artifact_complete(diff_meta: Any) -> bool:
    """Return True when a diff metadata entry points to an existing diff file."""
    if not isinstance(diff_meta, dict):
        return False
    diff_path = diff_meta.get("diff_path")
    if not diff_path:
        return False
    return Path(diff_path).exists()


def _empty_overlap_summary() -> dict[str, Any]:
    """Return the neutral overlap summary used for unavailable future diffs."""
    return {
        "touched_pr_files_count": 0,
        "touched_pr_files": [],
        "touched_pr_lines_count": 0,
        "touched_pr_line_files": [],
        "touched_pr_line_counts_by_file": {},
        "has_file_overlap": False,
        "has_line_overlap": False,
    }


def _empty_future_diff_record(after_sha: Optional[str]) -> dict[str, Any]:
    """Return a placeholder future-diff record anchored at the after-PR commit."""
    return {
        "start_commit": after_sha,
        "end_commit": None,
        "diff_path": None,
        "changed_files_count": 0,
        "changed_paths": [],
        "files": [],
        "old_line_ranges_by_file": {},
        "new_line_ranges_by_file": {},
        "old_anchor_lines_by_file": {},
        "new_anchor_lines_by_file": {},
        "old_changed_line_count": 0,
        "new_changed_line_count": 0,
        "overlap_with_pr": _empty_overlap_summary(),
    }


def _after_side_overlap_ranges(diff_summary: dict[str, Any]) -> dict[str, list[dict[str, int]]]:
    """Return PR after-side ranges used as the baseline for future tracking."""
    ranges_by_file = {
        str(path): list(ranges)
        for path, ranges in (diff_summary.get("new_line_ranges_by_file") or {}).items()
    }
    for path, anchors in (diff_summary.get("new_anchor_lines_by_file") or {}).items():
        if not anchors:
            continue
        ranges_by_file.setdefault(str(path), []).extend(
            {"start": int(anchor), "end": int(anchor)} for anchor in anchors
        )
    return ranges_by_file


def _compute_diff_overlap(
    pr_diff: dict[str, Any],
    future_diff: dict[str, Any],
) -> dict[str, Any]:
    """Compare original PR ranges with a future diff range summary."""
    pr_ranges = _after_side_overlap_ranges(pr_diff)
    future_ranges = future_diff.get("old_line_ranges_by_file") or {}
    touched_files = sorted(set(pr_ranges) & set(future_ranges))
    overlapping_lines = {
        path: _overlapping_line_count(pr_ranges.get(path, []), future_ranges.get(path, []))
        for path in touched_files
    }
    overlapping_lines = {
        path: count for path, count in overlapping_lines.items() if count > 0
    }
    return {
        "touched_pr_files_count": len(touched_files),
        "touched_pr_files": touched_files,
        "touched_pr_lines_count": sum(overlapping_lines.values()),
        "touched_pr_line_files": sorted(overlapping_lines),
        "touched_pr_line_counts_by_file": overlapping_lines,
        "has_file_overlap": bool(touched_files),
        "has_line_overlap": bool(overlapping_lines),
    }


def _missing_reason_from_file_attrition(
    *,
    deleted_files: list[str],
    renamed_files: list[dict[str, Any]],
    unknown_missing_files: list[str],
) -> str:
    """Summarize why tracked PR files are absent in a future snapshot."""
    if deleted_files and renamed_files:
        return "tracked_files_deleted_or_renamed"
    if deleted_files and not renamed_files and not unknown_missing_files:
        return "tracked_files_deleted"
    if renamed_files and not deleted_files and not unknown_missing_files:
        return "tracked_files_renamed"
    if deleted_files or renamed_files:
        return "tracked_files_deleted_or_renamed"
    return "missing_snapshot_unknown"


def _future_file_availability(
    *,
    repo: RepositoryHydrator,
    after_sha: Optional[str],
    snapshot_commit: Optional[str],
    files_after: list[str],
    copied_files: list[str],
) -> dict[str, Any]:
    """Report whether after-PR files still exist at a future snapshot commit."""
    expected_files = sorted({str(path).replace("\\", "/").lstrip("/") for path in files_after if str(path).strip()})
    copied_set = {str(path).replace("\\", "/").lstrip("/") for path in copied_files if str(path).strip()}
    if not snapshot_commit:
        return {
            "files_expected": len(expected_files),
            "files_copied": 0,
            "files_present": [],
            "files_missing": None,
            "missing_files": [],
            "deleted_files": [],
            "renamed_files": [],
            "unknown_missing_files": [],
            "file_availability_status": "not_evaluated",
        }

    present_files = sorted(
        path for path in expected_files if path in copied_set or repo.path_exists_at_commit(snapshot_commit, path)
    )
    present_set = set(present_files)
    missing_files = sorted(path for path in expected_files if path not in present_set)
    path_statuses = repo.path_statuses_between(after_sha or "", snapshot_commit, missing_files)
    deleted_files: list[str] = []
    renamed_files: list[dict[str, Any]] = []
    unknown_missing_files: list[str] = []
    for path in missing_files:
        status = path_statuses.get(path) or {}
        status_name = str(status.get("status") or "").strip().lower()
        if status_name == "deleted":
            deleted_files.append(path)
        elif status_name == "renamed":
            renamed_files.append(
                {
                    "old_path": path,
                    "new_path": status.get("new_path"),
                    "raw_status": status.get("raw_status"),
                }
            )
        else:
            unknown_missing_files.append(path)

    if not expected_files:
        availability_status = "not_evaluated"
    elif not missing_files:
        availability_status = "complete"
    elif present_files:
        availability_status = "partial_missing"
    else:
        availability_status = "all_missing"

    return {
        "files_expected": len(expected_files),
        "files_copied": len(present_files),
        "files_present": present_files,
        "files_missing": len(missing_files),
        "missing_files": missing_files,
        "deleted_files": sorted(deleted_files),
        "renamed_files": sorted(renamed_files, key=lambda item: str(item.get("old_path") or "")),
        "unknown_missing_files": sorted(unknown_missing_files),
        "file_availability_status": availability_status,
    }


def _dominant_pr_language(pr: Any) -> Optional[str]:
    """Return the already-resolved dominant PR language, if present."""
    explicit = _normalize_language_token(
        _get_attr(pr, "pr_primary_language_effective") or _get_attr(pr, "primary_language")
    )
    return explicit


def _filter_paths_by_language(paths: list[str], language_token: Optional[str]) -> list[str]:
    """Filter repository paths to those matching the PR benchmark language."""
    normalized_language = _normalize_language_token(language_token)
    if not normalized_language:
        return []
    selected: list[str] = []
    for path in paths:
        inferred = _normalize_language_token(infer_language(path))
        if inferred == normalized_language:
            selected.append(path.replace("\\", "/").lstrip("/"))
    return sorted(set(selected))


def _compute_future_commit_activity(
    *,
    repo: RepositoryHydrator,
    after_sha: Optional[str],
    tracked_pr_files: list[str],
    future_commits: list[tuple[str, str]],
) -> dict[str, Any]:
    """Compute commit-touch metadata for consecutive future snapshot intervals."""
    per_snapshot: dict[str, Any] = {}
    interval_start_commit = after_sha
    interval_start_label = "after"
    for label, interval_end_commit in future_commits:
        interval_end_commit = str(interval_end_commit or "").strip()
        if not interval_start_commit or not interval_end_commit:
            interval_start_commit = interval_end_commit or interval_start_commit
            interval_start_label = str(label)
            continue
        ancestry_status = repo.is_ancestor(interval_start_commit, interval_end_commit)
        commit_activity_status = "ok"
        touch_events: Optional[list[dict[str, object]]]
        if ancestry_status is False:
            touch_events = None
            commit_activity_status = "non_ancestral_interval"
        elif ancestry_status is None:
            touch_events = None
            commit_activity_status = "ancestry_unknown"
        else:
            touch_events = repo.commits_touching_paths_between(
                interval_start_commit,
                interval_end_commit,
                tracked_pr_files,
            )
            if touch_events is None:
                commit_activity_status = "commit_log_failed"
        touched_files: set[str] = set()
        commit_shas: list[str] = []
        for event in touch_events or []:
            sha = str(event.get("sha") or "").strip()
            if sha:
                commit_shas.append(sha)
            for touched_path in event.get("touched_paths") or []:
                touched_files.add(str(touched_path))
        touched_count = len(touched_files)
        total_files = len(tracked_pr_files)
        coverage_pct = (100.0 * touched_count / total_files) if total_files else 0.0
        per_snapshot[str(label)] = {
            "interval_start_label": interval_start_label,
            "interval_start_commit": interval_start_commit,
            "interval_end_commit": interval_end_commit,
            "commit_activity_status": commit_activity_status,
            "ancestry_status": ancestry_status,
            "touching_commits_count": (
                len(touch_events) if touch_events is not None else None
            ),
            "touching_commit_shas": commit_shas,
            "touched_pr_files_count": touched_count,
            "touched_pr_files": sorted(touched_files),
            "touch_coverage_pct": coverage_pct,
        }
        interval_start_commit = interval_end_commit
        interval_start_label = str(label)
    return per_snapshot


def _reliable_cumulative_touch_count(
    intervals: dict[str, Any],
    label: str,
    labels: tuple[str, ...],
) -> Optional[int]:
    """Return cumulative touch count only when all intervals are trustworthy."""
    total = 0
    seen_target = False
    for current_label in labels:
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


def _can_skip_future_snapshot_materialization(
    *,
    files_after: list[str],
    tracked_pr_files: list[str],
    intervals: dict[str, Any],
    label: str,
    labels: tuple[str, ...],
) -> bool:
    """Return True when a future snapshot would be byte-identical for tracked files."""
    expected = {
        str(path).strip().replace("\\", "/").lstrip("/")
        for path in files_after
        if str(path).strip()
    }
    tracked = {
        str(path).strip().replace("\\", "/").lstrip("/")
        for path in tracked_pr_files
        if str(path).strip()
    }
    if not expected or expected != tracked:
        return False
    return _reliable_cumulative_touch_count(intervals, label, labels) == 0


class PRHydrator:
    """Create local source snapshots and future-tracking metadata for one PR."""

    FUTURE_OFFSETS_DAYS = (3, 7, 31, 61)
    NON_CODE_LANGUAGES = {
        "JSON",
        "YAML",
        "TOML",
        "INI",
        "Markdown",
        "reStructuredText",
        "HTML",
        "CSS",
        "SCSS",
        "Sass",
        "Less",
        "XML",
        "Vue",
        "Svelte",
        "Jupyter Notebook",
    }

    def __init__(self, repo: RepositoryHydrator, snapshots_root: Path) -> None:
        """Bind the hydrator to one prepared repository and output root."""
        self.repo = repo
        self.snapshots_root = snapshots_root

    def hydrate(
        self,
        pr: Any,
        *,
        include_future: bool = True,
        existing_hydration: Optional[dict[str, Any]] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict:
        """Hydrate before/after snapshots, optional future snapshots, and diff metadata."""
        pr_number = _get_attr(pr, "number")
        pr_url = _get_attr(pr, "url")
        pr_id = pr_number if pr_number is not None else _get_attr(pr, "id", "unknown")

        merge_sha = _get_attr(pr, "merge_commit_sha")
        base_sha = _get_attr(pr, "base_commit_sha")
        head_sha = _get_attr(pr, "head_commit_sha")
        pr_number_int: Optional[int]
        try:
            pr_number_int = int(pr_number) if pr_number is not None else None
        except (TypeError, ValueError):
            pr_number_int = None
        merged_at = _parse_iso(_get_attr(pr, "merged_at"))
        print(
            with_pr_progress(
                "[hydration] PR {url}: merge_sha={merge_sha} base_sha={base_sha} head_sha={head_sha}".format(
                    url=pr_url or "unknown",
                    merge_sha=merge_sha or "none",
                    base_sha=base_sha or "none",
                    head_sha=head_sha or "none",
                )
            )
        )

        branch = self.repo.resolve_default_branch()
        latest_observed_commit_at = _as_utc(_parse_iso(self.repo.latest_commit_timestamp(branch)))

        # Ensure commit objects are locally available; stale/default clones may miss PR-only refs.
        if merge_sha and not self.repo.ensure_commit(merge_sha, pr_number=pr_number_int):
            print(
                with_pr_progress(
                    "[hydration] PR {url}: merge commit unavailable locally, falling back when possible.".format(
                        url=pr_url or "unknown"
                    )
                )
            )
            merge_sha = None
        if head_sha and not self.repo.ensure_commit(head_sha, pr_number=pr_number_int):
            print(
                with_pr_progress(
                    "[hydration] PR {url}: head commit unavailable locally.".format(
                        url=pr_url or "unknown"
                    )
                )
            )
            head_sha = None
        if base_sha and not self.repo.ensure_commit(base_sha, pr_number=pr_number_int):
            print(
                with_pr_progress(
                    "[hydration] PR {url}: base commit unavailable locally.".format(
                        url=pr_url or "unknown"
                    )
                )
            )
            base_sha = None

        if not base_sha and merge_sha:
            base_sha = self.repo.parent_commit(merge_sha)
            if base_sha and not self.repo.ensure_commit(base_sha, pr_number=pr_number_int):
                base_sha = None

        after_sha = merge_sha or head_sha

        snapshot_dir = self.snapshots_root / self.repo.owner / self.repo.name / f"pr-{pr_id}"
        diff_path = snapshot_dir / "diff"
        before_dir = snapshot_dir / "before"
        after_dir = snapshot_dir / "after"
        future_dir = snapshot_dir / "future"

        hydration: dict[str, Any] = (
            dict(existing_hydration) if isinstance(existing_hydration, dict) else {}
        )
        longitudinal_commit_activity: dict[str, Any] = (
            dict(hydration.get("longitudinal_commit_activity"))
            if isinstance(hydration.get("longitudinal_commit_activity"), dict)
            else {}
        )
        snapshots_value = hydration.get("snapshots")
        snapshots: dict[str, Any] = dict(snapshots_value) if isinstance(snapshots_value, dict) else {}
        future_snapshots_value = snapshots.get("future")
        snapshots["future"] = (
            dict(future_snapshots_value) if isinstance(future_snapshots_value, dict) else {}
        )
        diff_tracking_value = hydration.get("diff_tracking")
        diff_tracking: dict[str, Any] = (
            dict(diff_tracking_value) if isinstance(diff_tracking_value, dict) else {}
        )
        future_tracking_value = diff_tracking.get("future")
        diff_tracking["future"] = (
            dict(future_tracking_value) if isinstance(future_tracking_value, dict) else {}
        )
        future_availability_value = hydration.get("future_snapshot_availability")
        future_snapshot_availability: dict[str, Any] = (
            dict(future_availability_value) if isinstance(future_availability_value, dict) else {}
        )

        def _build_payload() -> dict[str, Any]:
            has_future_snapshots = any(
                isinstance(meta, dict)
                and meta.get("commit")
                and meta.get("available") is not False
                for meta in (snapshots.get("future") or {}).values()
            )
            return {
                "pr_url": pr_url,
                "pr_number": pr_number,
                "longitudinal_selected": bool(include_future),
                "merge_commit": merge_sha,
                "base_commit": base_sha,
                "after_commit": after_sha,
                "snapshots": snapshots,
                "future_snapshot_availability": future_snapshot_availability,
                "diff_tracking": diff_tracking,
                "diff_path": str(diff_path) if diff_path.exists() else None,
                "readme_path": hydration.get("readme_path"),
                "readme": hydration.get("readme"),
                "has_future_snapshots": has_future_snapshots,
                "merge_time": merged_at.isoformat() if merged_at else None,
                "longitudinal_commit_activity": longitudinal_commit_activity,
            }

        def _persist_progress() -> dict[str, Any]:
            payload = _build_payload()
            hydration.clear()
            hydration.update(payload)
            if progress_callback:
                progress_callback(payload)
            return payload

        files_before, files_after = self._extract_file_paths(pr)
        print(
            with_pr_progress(
                "[hydration] PR {url}: file paths before={before_count} after={after_count}".format(
                    url=pr_url or "unknown",
                    before_count=len(files_before),
                    after_count=len(files_after),
                )
            )
        )
        before_snapshot = snapshots.get("before")
        if base_sha and not _snapshot_export_complete(before_snapshot):
            copied = self.repo.export_files(base_sha, files_before, before_dir)
            snapshots["before"] = {
                "commit": base_sha,
                "path": str(before_dir),
                "files_copied": len(copied),
            }
            print(
                with_pr_progress(
                    "[hydration] PR {url}: before files_copied={count}".format(
                        url=pr_url or "unknown",
                        count=len(copied),
                    )
                )
            )
            _persist_progress()

        after_snapshot = snapshots.get("after")
        if after_sha and not _snapshot_export_complete(after_snapshot):
            copied = self.repo.export_files(after_sha, files_after, after_dir)
            snapshots["after"] = {
                "commit": after_sha,
                "path": str(after_dir),
                "files_copied": len(copied),
            }
            print(
                with_pr_progress(
                    "[hydration] PR {url}: after files_copied={count}".format(
                        url=pr_url or "unknown",
                        count=len(copied),
                    )
                )
            )
            _persist_progress()

        pr_diff_tracking = diff_tracking.get("pr")
        pr_diff_summary: Optional[dict[str, Any]] = (
            pr_diff_tracking if isinstance(pr_diff_tracking, dict) else None
        )
        if base_sha and after_sha and not _diff_artifact_complete(pr_diff_tracking):
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_text = self.repo.diff_text_between(
                base_sha,
                after_sha,
                paths=sorted(set(files_before + files_after)),
                unified=0,
            )
            diff_path.write_text(diff_text, encoding="utf-8", errors="ignore")
            pr_diff_summary = _parse_structured_diff(diff_text)
            diff_tracking["pr"] = {
                "start_commit": base_sha,
                "end_commit": after_sha,
                "diff_path": str(diff_path),
                **pr_diff_summary,
            }
            _persist_progress()
        elif isinstance(pr_diff_tracking, dict):
            pr_diff_summary = pr_diff_tracking

        dominant_language = _dominant_pr_language(pr)
        tracked_pr_files = _filter_paths_by_language(files_after, dominant_language)
        future_targets: list[dict[str, Any]] = []
        if include_future and merged_at:
            target_timestamps: list[str] = []
            for offset in self.FUTURE_OFFSETS_DAYS:
                target_dt = _as_utc(merged_at + timedelta(days=offset))
                if target_dt is None:
                    continue
                label = f"+{offset}d"
                timestamp = _format_iso(target_dt)
                future_targets.append(
                    {
                        "label": label,
                        "offset": offset,
                        "target_dt": target_dt,
                        "target_timestamp": timestamp,
                    }
                )
                target_timestamps.append(timestamp)
            commits_by_timestamp = self.repo.commits_after(target_timestamps, branch)
            for target in future_targets:
                target["commit"] = commits_by_timestamp.get(str(target["target_timestamp"]))
        per_snapshot = _compute_future_commit_activity(
            repo=self.repo,
            after_sha=after_sha,
            tracked_pr_files=tracked_pr_files,
            future_commits=[
                (str(target["label"]), str(target.get("commit") or ""))
                for target in future_targets
                if target.get("commit")
            ],
        )
        longitudinal_commit_activity.clear()
        longitudinal_commit_activity.update(
            {
                "original_pr_files_count": len(files_after),
                "tracked_pr_files_count": len(tracked_pr_files),
                "tracked_pr_files": tracked_pr_files,
                "future_snapshot_intervals": per_snapshot,
            }
        )
        _persist_progress()

        if include_future and merged_at:
            for target in future_targets:
                offset = int(target["offset"])
                target_dt = target["target_dt"]
                target_timestamp = str(target["target_timestamp"])
                commit = str(target.get("commit") or "").strip()
                label = str(target["label"])
                future_snapshots: dict[str, Any] = snapshots.get("future") or {}
                future_tracking: dict[str, Any] = diff_tracking.get("future") or {}
                if commit:
                    existing_future_snapshot = future_snapshots.get(label)
                    dest = future_dir / label
                    skip_materialization = (
                        _can_skip_future_snapshot_materialization(
                            files_after=files_after,
                            tracked_pr_files=tracked_pr_files,
                            intervals=per_snapshot,
                            label=label,
                            labels=tuple(f"+{offset}d" for offset in self.FUTURE_OFFSETS_DAYS),
                        )
                    )
                    if skip_materialization:
                        existing_snapshot_path = (
                            str(existing_future_snapshot.get("path"))
                            if _snapshot_export_complete(existing_future_snapshot)
                            else None
                        )
                        expected_files = sorted(
                            {
                                str(path).replace("\\", "/").lstrip("/")
                                for path in files_after
                                if str(path).strip()
                            }
                        )
                        availability_record = {
                            "label": label,
                            "target_offset_days": offset,
                            "target_timestamp": target_timestamp,
                            "repository_observation_cutoff": (
                                _format_iso(latest_observed_commit_at)
                                if latest_observed_commit_at
                                else None
                            ),
                            "available": True,
                            "missing_reason": None,
                            "snapshot_commit": commit,
                            "snapshot_path": existing_snapshot_path,
                            "files_expected": len(expected_files),
                            "files_copied": 0,
                            "files_present": expected_files,
                            "files_missing": 0,
                            "missing_files": [],
                            "deleted_files": [],
                            "renamed_files": [],
                            "unknown_missing_files": [],
                            "file_availability_status": "complete",
                            "materialization_skipped": True,
                            "skip_reason": "future_no_cumulative_commit_touches",
                        }
                        future_snapshot_availability[label] = availability_record
                        future_snapshots[label] = {
                            "commit": commit,
                            "path": existing_snapshot_path,
                            "target_timestamp": target_timestamp,
                            "repository_observation_cutoff": availability_record["repository_observation_cutoff"],
                            "available": True,
                            "missing_reason": None,
                            "files_expected": availability_record["files_expected"],
                            "files_copied": 0,
                            "files_present": expected_files,
                            "files_missing": 0,
                            "missing_files": [],
                            "deleted_files": [],
                            "renamed_files": [],
                            "file_availability_status": "complete",
                            "diff_path": None,
                            "materialization_skipped": True,
                            "skip_reason": "future_no_cumulative_commit_touches",
                        }
                        future_tracking[label] = _empty_future_diff_record(after_sha)
                        future_tracking[label]["end_commit"] = commit
                        future_tracking[label]["target_timestamp"] = target_timestamp
                        future_tracking[label]["available"] = True
                        future_tracking[label]["missing_reason"] = None
                        future_tracking[label]["materialization_skipped"] = True
                        future_tracking[label]["skip_reason"] = "future_no_cumulative_commit_touches"
                        snapshots["future"] = future_snapshots
                        diff_tracking["future"] = future_tracking
                        _persist_progress()
                        print(
                            with_pr_progress(
                                "[hydration] PR {url}: future {label} materialization skipped (no PR-file touches)".format(
                                    url=pr_url or "unknown",
                                    label=label,
                                )
                            )
                        )
                        continue

                    if not _snapshot_export_complete(existing_future_snapshot):
                        copied = self.repo.export_files(commit, files_after, dest)
                        future_snapshots[label] = {
                            "commit": commit,
                            "path": str(dest),
                            "files_copied": len(copied),
                            "diff_path": str(dest / "diff") if after_sha else None,
                        }
                        snapshots["future"] = future_snapshots
                        _persist_progress()
                    else:
                        copied = []
                    file_availability = _future_file_availability(
                        repo=self.repo,
                        after_sha=after_sha,
                        snapshot_commit=commit,
                        files_after=files_after,
                        copied_files=copied,
                    )
                    snapshot_available = (
                        file_availability.get("file_availability_status")
                        in {"complete", "partial_missing"}
                    )
                    missing_reason = None
                    if not snapshot_available:
                        missing_reason = _missing_reason_from_file_attrition(
                            deleted_files=list(file_availability.get("deleted_files") or []),
                            renamed_files=list(file_availability.get("renamed_files") or []),
                            unknown_missing_files=list(file_availability.get("unknown_missing_files") or []),
                        )
                    availability_record = {
                        "label": label,
                        "target_offset_days": offset,
                        "target_timestamp": target_timestamp,
                        "repository_observation_cutoff": (
                            _format_iso(latest_observed_commit_at)
                            if latest_observed_commit_at
                            else None
                        ),
                        "available": bool(snapshot_available),
                        "missing_reason": missing_reason,
                        "snapshot_commit": commit,
                        "snapshot_path": str(dest),
                        **file_availability,
                    }
                    future_snapshot_availability[label] = availability_record
                    future_snapshots[label] = {
                        **(future_snapshots.get(label) or {}),
                        "commit": commit,
                        "path": str(dest),
                        "target_timestamp": availability_record["target_timestamp"],
                        "repository_observation_cutoff": availability_record["repository_observation_cutoff"],
                        "available": bool(snapshot_available),
                        "missing_reason": missing_reason,
                        "files_expected": availability_record["files_expected"],
                        "files_copied": availability_record["files_copied"],
                        "files_missing": availability_record["files_missing"],
                        "missing_files": availability_record["missing_files"],
                        "deleted_files": availability_record["deleted_files"],
                        "renamed_files": availability_record["renamed_files"],
                        "file_availability_status": availability_record["file_availability_status"],
                        "diff_path": str(dest / "diff") if after_sha else None,
                    }
                    snapshots["future"] = future_snapshots
                    future_diff_path = dest / "diff"
                    future_diff_summary = (
                        future_tracking.get(label)
                        if isinstance(future_tracking.get(label), dict)
                        else None
                    )
                    if after_sha and not _diff_artifact_complete(future_diff_summary):
                        future_diff_text = self.repo.diff_text_between(
                            after_sha,
                            commit,
                            paths=files_after,
                            unified=0,
                        )
                        future_diff_path.parent.mkdir(parents=True, exist_ok=True)
                        future_diff_path.write_text(
                            future_diff_text,
                            encoding="utf-8",
                            errors="ignore",
                        )
                        future_diff_summary = _parse_structured_diff(future_diff_text)
                        future_tracking[label] = {
                            "start_commit": after_sha,
                            "end_commit": commit,
                            "diff_path": str(future_diff_path) if after_sha else None,
                            "target_timestamp": availability_record["target_timestamp"],
                            "available": bool(snapshot_available),
                            "missing_reason": missing_reason,
                            **future_diff_summary,
                            "overlap_with_pr": (
                                _compute_diff_overlap(pr_diff_summary, future_diff_summary)
                                if pr_diff_summary and future_diff_summary
                                else _empty_overlap_summary()
                            ),
                        }
                        diff_tracking["future"] = future_tracking
                        _persist_progress()
                    elif not isinstance(future_tracking.get(label), dict):
                        future_tracking[label] = _empty_future_diff_record(after_sha)
                        future_tracking[label]["end_commit"] = commit
                        future_tracking[label]["diff_path"] = (
                            str(future_diff_path) if after_sha else None
                        )
                        future_tracking[label]["target_timestamp"] = availability_record["target_timestamp"]
                        future_tracking[label]["available"] = bool(snapshot_available)
                        future_tracking[label]["missing_reason"] = missing_reason
                        diff_tracking["future"] = future_tracking
                        _persist_progress()
                    else:
                        future_tracking[label]["target_timestamp"] = availability_record["target_timestamp"]
                        future_tracking[label]["available"] = bool(snapshot_available)
                        future_tracking[label]["missing_reason"] = missing_reason
                        diff_tracking["future"] = future_tracking
                        _persist_progress()
                    overlap_summary = (diff_tracking.get("future") or {}).get(label, {}).get(
                        "overlap_with_pr",
                        _empty_overlap_summary(),
                    )
                    print(
                        with_pr_progress(
                            "[hydration] PR {url}: future {label} files_copied={count} touched_pr_files={files} touched_pr_lines={lines}".format(
                                url=pr_url or "unknown",
                                label=label,
                                count=(future_snapshots.get(label) or {}).get("files_copied", len(copied)),
                                files=overlap_summary["touched_pr_files_count"],
                                lines=overlap_summary["touched_pr_lines_count"],
                            )
                        )
                    )
                else:
                    target_utc = _as_utc(target_dt)
                    if latest_observed_commit_at and target_utc and target_utc > latest_observed_commit_at:
                        missing_reason = "insufficient_elapsed_time"
                    elif latest_observed_commit_at:
                        missing_reason = "no_future_commit_after_target"
                    else:
                        missing_reason = "missing_snapshot_unknown"
                    availability_record = {
                        "label": label,
                        "target_offset_days": offset,
                        "target_timestamp": target_timestamp,
                        "repository_observation_cutoff": (
                            _format_iso(latest_observed_commit_at)
                            if latest_observed_commit_at
                            else None
                        ),
                        "available": False,
                        "missing_reason": missing_reason,
                        "snapshot_commit": None,
                        "snapshot_path": None,
                        "files_expected": len(files_after),
                        "files_copied": 0,
                        "files_present": [],
                        "files_missing": None,
                        "missing_files": [],
                        "deleted_files": [],
                        "renamed_files": [],
                        "unknown_missing_files": [],
                        "file_availability_status": "not_evaluated",
                    }
                    future_snapshot_availability[label] = availability_record
                    future_snapshots[label] = {
                        "commit": None,
                        "path": None,
                        "target_timestamp": availability_record["target_timestamp"],
                        "repository_observation_cutoff": availability_record["repository_observation_cutoff"],
                        "available": False,
                        "missing_reason": missing_reason,
                        "files_expected": len(files_after),
                        "files_copied": 0,
                        "files_missing": None,
                        "missing_files": [],
                        "deleted_files": [],
                        "renamed_files": [],
                        "file_availability_status": "not_evaluated",
                    }
                    future_tracking[label] = _empty_future_diff_record(after_sha)
                    future_tracking[label]["target_timestamp"] = availability_record["target_timestamp"]
                    future_tracking[label]["available"] = False
                    future_tracking[label]["missing_reason"] = missing_reason
                    snapshots["future"] = future_snapshots
                    diff_tracking["future"] = future_tracking
                    _persist_progress()
        elif not include_future:
            snapshots["future"] = {}
            diff_tracking["future"] = {}
            future_snapshot_availability.clear()
            _persist_progress()

        readme = None
        readme_path = None
        if after_sha:
            existing_readme_path = hydration.get("readme_path")
            if existing_readme_path and Path(existing_readme_path).exists():
                readme_path = Path(existing_readme_path)
                readme = hydration.get("readme")
            else:
                readme_name, readme = self.repo.read_readme_at_commit(after_sha)
                readme_path = Path(readme_name) if readme_name else None
            if readme is not None:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                readme_out = snapshot_dir / "README.md"
                readme_out.write_text(readme, encoding="utf-8", errors="ignore")
                readme_path = readme_out
                hydration["readme_path"] = str(readme_path)
                hydration["readme"] = readme
                _persist_progress()

        return _persist_progress()

    def _extract_file_paths(self, pr: Any) -> tuple[list[str], list[str]]:
        """Return code-file paths needed for before and after snapshots."""
        files = _get_attr(pr, "files") or []
        before_paths: list[str] = []
        after_paths: list[str] = []
        for item in files:
            path = _get_attr(item, "path") or _get_attr(item, "filename") or _get_attr(item, "file")
            prev = _get_attr(item, "previous_filename")
            if path and not self._is_code_path(str(path)):
                path = None
            if prev and not self._is_code_path(str(prev)):
                prev = None
            if prev:
                before_paths.append(str(prev))
            if path:
                after_paths.append(str(path))
                if not prev:
                    before_paths.append(str(path))
        before_paths = list(dict.fromkeys(before_paths))
        after_paths = list(dict.fromkeys(after_paths))
        return before_paths, after_paths

    def _is_code_path(self, path: str) -> bool:
        """Return True when a path maps to a supported code language."""
        language = infer_language(path)
        if not language:
            return False
        return language not in self.NON_CODE_LANGUAGES


def pr_to_dict(pr: Any) -> dict:
    """Serialize a PR DTO/dict/value into a plain dictionary."""
    if hasattr(pr, "to_dict"):
        return pr.to_dict()
    if hasattr(pr, "__dict__"):
        try:
            return asdict(pr)
        except TypeError:
            return dict(pr.__dict__)
    if isinstance(pr, dict):
        return dict(pr)
    return {"value": pr}
