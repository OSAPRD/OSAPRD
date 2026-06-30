"""Maintainability metrics stage backed by Multimetric plus custom duplication.

Every selected snapshot is analyzed with Multimetric for source-code metrics.
Duplicated-lines density is then computed by the in-repository deterministic
scanner and merged into the same snapshot row and delta summaries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from curation.config.maintainability_config import (
    FUTURE_MAINTAINABILITY_SNAPSHOT_LABELS,
    MULTIMETRIC_COMMAND,
    MULTIMETRIC_MAINTINDEX_MODE,
)
from curation.hydration.repository_hydrator import RepositoryHydrator
from curation.metrics.code_smell_metrics import (
    CODE_SMELL_SELECTED_TOOLS,
    CodeSmellDetectionStage,
    summarize_code_smells,
)
from curation.metrics.duplication_metrics import compute_snapshot_duplication
from curation.metrics.maintainability_common import (
    _future_impact_summary,
    _future_snapshot_tool_skip_reason,
    _get_attr,
    _get_pr_number,
    _log_error,
    _log_info,
    _write_json_artifact,
)
from curation.metrics.multimetric_runner import (
    METRIC_FIELDS,
    SnapshotTask,
    json_dumps,
    reuse_snapshot_row,
    run_multimetric_for_snapshot,
    safe_float,
    tool_version,
    utc_now_text,
)


_SUCCESS_STATUSES = {"success", "missing_snapshot", "no_source_files"}
CUSTOM_DUPLICATION_FIELDS = (
    "custom_duplication_ncloc",
    "duplicated_lines",
    "duplicated_lines_density",
)
MAINTAINABILITY_METRIC_FIELDS = (*METRIC_FIELDS, *CUSTOM_DUPLICATION_FIELDS)


def _days_after_merge(label: str) -> int | None:
    """Parse labels such as ``+31d`` into an integer day offset."""
    match = re.match(r"^\+(\d+)d$", str(label or "").strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _snapshot_targets(hydration: Dict[str, Any], *, include_future: bool) -> List[Dict[str, Any]]:
    """Return maintainability snapshot targets from hydration metadata."""
    snapshots = hydration.get("snapshots") or {}
    targets: List[Dict[str, Any]] = []
    before = snapshots.get("before")
    if isinstance(before, dict) and before.get("commit"):
        targets.append(
            {
                "label": "before",
                "kind": "before",
                "commit": before.get("commit"),
                "snapshot": before,
            }
        )
    after = snapshots.get("after")
    if isinstance(after, dict) and after.get("commit"):
        targets.append(
            {
                "label": "after",
                "kind": "after",
                "commit": after.get("commit"),
                "snapshot": after,
            }
        )
    if include_future:
        future = snapshots.get("future") or {}
        for label in FUTURE_MAINTAINABILITY_SNAPSHOT_LABELS:
            snapshot_meta = future.get(label)
            if (
                isinstance(snapshot_meta, dict)
                and snapshot_meta.get("commit")
                and snapshot_meta.get("available") is not False
            ):
                targets.append(
                    {
                        "label": str(label),
                        "kind": "future",
                        "commit": snapshot_meta.get("commit"),
                        "snapshot": snapshot_meta,
                    }
                )
    return targets


def _snapshot_task(
    *,
    cohort: str | None,
    owner: str | None,
    repo: str | None,
    pr: Any,
    target: Dict[str, Any],
) -> SnapshotTask:
    """Build a Multimetric task from a hydration snapshot target."""
    label = str(target.get("label") or "")
    snapshot = target.get("snapshot") if isinstance(target.get("snapshot"), dict) else {}
    raw_path = snapshot.get("path")
    return SnapshotTask(
        cohort=cohort,
        repository_owner=owner,
        repository_name=repo,
        repository_key=f"{owner}/{repo}" if owner and repo else None,
        pr_number=_safe_pr_number(_get_pr_number(pr)),
        pr_url=_get_attr(pr, "url"),
        snapshot_kind=str(target.get("kind") or ("future" if label.startswith("+") else label)),
        snapshot_label=label,
        days_after_merge=_days_after_merge(label),
        snapshot_commit=str(target.get("commit") or snapshot.get("commit") or "") or None,
        snapshot_path=Path(str(raw_path)) if raw_path else None,
    )


def _safe_pr_number(value: Any) -> int | None:
    """Coerce PR number metadata to an int when possible."""
    try:
        return int(value)
    except Exception:
        return None


def _artifact_path_for_result(
    *,
    task: SnapshotTask,
    fallback_snapshot_path: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Return the artifact directory and file for a snapshot result."""
    base = task.snapshot_path if task.snapshot_path else fallback_snapshot_path
    if not base:
        return None, None
    maintainability_root = base / "maintainability"
    suffix = task.snapshot_label.replace("+", "plus").replace("/", "_")
    filename = (
        "multimetric_tool_results.json"
        if task.snapshot_label in {"before", "after"}
        else f"multimetric_{suffix}_tool_results.json"
    )
    return maintainability_root, maintainability_root / filename


def _write_multimetric_artifact(
    *,
    task: SnapshotTask,
    row: Dict[str, Any],
    command_reused: bool,
    fallback_snapshot_path: Path | None = None,
) -> tuple[str | None, str | None]:
    """Persist a snapshot-level Multimetric/custom-duplication artifact."""
    maintainability_root, artifact_path = _artifact_path_for_result(
        task=task,
        fallback_snapshot_path=fallback_snapshot_path,
    )
    if artifact_path is None:
        return (
            str(maintainability_root) if maintainability_root else None,
            None,
        )
    payload = {
        "schema_version": 1,
        "engine": "multimetric",
        "tool": "multimetric",
        "snapshot_label": task.snapshot_label,
        "snapshot_kind": task.snapshot_kind,
        "snapshot_commit": task.snapshot_commit,
        "snapshot_path": str(task.snapshot_path) if task.snapshot_path else None,
        "status": row.get("status"),
        "error_message": row.get("error_message"),
        "files_considered": row.get("files_considered"),
        "files_analyzed": row.get("files_analyzed"),
        "multimetric_reused": bool(command_reused),
        "reused_from_snapshot_metric_id": row.get("reused_from_snapshot_metric_id"),
        "reuse_reason": row.get("reuse_reason"),
        "snapshot_metric_row": row,
    }
    _write_json_artifact(artifact_path, payload)
    return str(maintainability_root), str(artifact_path)


def _measures_from_row(
    row: Dict[str, Any],
    smell_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract user-facing maintainability measures from one snapshot row."""
    smell_count = (
        int(smell_result.get("issue_count", 0))
        if isinstance(smell_result, dict)
        else 0
    )
    measures = {field: row.get(field) for field in MAINTAINABILITY_METRIC_FIELDS}
    measures.update(
        {
            "ncloc": row.get("loc"),
            "complexity": row.get("cyclomatic_complexity"),
            "duplicated_lines_density": row.get("duplicated_lines_density"),
            "duplicated_lines_density_source": "custom",
            "code_smells": smell_count,
            "code_smell_detection_skipped": False,
            "code_smell_detection_status": (
                smell_result.get("status") if isinstance(smell_result, dict) else None
            ),
            "multimetric_reused": bool(row.get("multimetric_reused")),
        }
    )
    return measures


def _snapshot_row_by_label(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index snapshot metric rows by snapshot label."""
    by_label: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("snapshot_label") or "")
        if label:
            by_label[label] = row
    return by_label


def _future_metric_deltas(rows: List[Dict[str, Any]], hydration: Dict[str, Any]) -> Dict[str, Any]:
    """Compute after-to-future maintainability deltas for longitudinal snapshots."""
    by_label = _snapshot_row_by_label(rows)
    after = by_label.get("after") or {}
    output: Dict[str, Any] = {}
    availability = hydration.get("future_snapshot_availability") or {}
    for label in FUTURE_MAINTAINABILITY_SNAPSHOT_LABELS:
        future = by_label.get(label)
        if not future:
            continue
        item: Dict[str, Any] = {
            "status": future.get("status"),
            "snapshot_commit": future.get("snapshot_commit"),
            "snapshot_metric_id": future.get("snapshot_metric_id"),
            "multimetric_reused": bool(future.get("multimetric_reused")),
            "reuse_reason": future.get("reuse_reason"),
            "availability": availability.get(label) if isinstance(availability, dict) else None,
            "metrics": {},
        }
        for field in MAINTAINABILITY_METRIC_FIELDS:
            current = safe_float(future.get(field))
            baseline = safe_float(after.get(field))
            item["metrics"][field] = {
                "after": baseline,
                "future": current,
                "delta": (current - baseline) if current is not None and baseline is not None else None,
            }
        output[label] = item
    return output


def _pre_post_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute before-to-after maintainability metrics for the original PR."""
    by_label = _snapshot_row_by_label(rows)
    pre = by_label.get("before") or {}
    post = by_label.get("after") or {}
    output: Dict[str, Any] = {}
    for field in MAINTAINABILITY_METRIC_FIELDS:
        pre_value = safe_float(pre.get(field))
        post_value = safe_float(post.get(field))
        output[f"{field}_pre"] = pre_value
        output[f"{field}_post"] = post_value
        output[f"{field}_delta"] = (
            post_value - pre_value if pre_value is not None and post_value is not None else None
        )
    return output


def _overall_status(results: List[Dict[str, Any]]) -> str:
    """Summarize snapshot-row statuses into one stage status."""
    if not results:
        return "missing_snapshot"
    statuses = {str(item.get("status") or "").strip().lower() for item in results}
    if statuses == {"success"}:
        return "success"
    if statuses <= _SUCCESS_STATUSES:
        return "partial_success"
    if "success" in statuses:
        return "partial_failure"
    return "failed"


def _checkpointed_results(existing_stage: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return reusable snapshot rows from an existing in-progress payload."""
    if not isinstance(existing_stage, dict):
        return {}
    indicators = existing_stage.get("maintainability_indicators")
    summary = indicators.get("summary") if isinstance(indicators, dict) else None
    rows = summary.get("multimetric_snapshot_rows") if isinstance(summary, dict) else None
    if not isinstance(rows, list):
        return {}
    by_label: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("snapshot_label") or "")
        status = str(row.get("status") or "").strip().lower()
        if label and status == "success":
            by_label[label] = row
    return by_label


class MultimetricMaintainabilityMetricsStage:
    """Compute maintainability metrics for before/after and future snapshots."""

    def __init__(
        self,
        *,
        cohort: str | None = None,
        multimetric_bin: str = MULTIMETRIC_COMMAND,
        maintindex_mode: str = MULTIMETRIC_MAINTINDEX_MODE,
    ) -> None:
        """Configure the Multimetric command and cohort metadata."""
        self.cohort = cohort
        self.multimetric_bin = multimetric_bin
        self.maintindex_mode = maintindex_mode
        self._tool_version = tool_version(multimetric_bin)
        self._smell_stage = CodeSmellDetectionStage()

    def _build_progress(
        self,
        *,
        targets: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
        smell_findings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a progress payload after each snapshot is processed."""
        status = _overall_status(results)
        smell_findings = smell_findings or []
        return {
            "engine": "multimetric",
            "status": status,
            "maintainability_indicators": {
                "engine": "multimetric",
                "selected_tools": ["multimetric", *CODE_SMELL_SELECTED_TOOLS],
                "code_smell_detection_skipped": False,
                "code_smells": smell_findings,
                "summary": {
                    "target_snapshots": len(targets),
                    "completed_snapshots": len(results),
                    "smell_count": len(smell_findings),
                    "code_smell_detection_skipped": False,
                    "multimetric_snapshot_rows": rows,
                },
                "results": results,
            },
        }

    def compute(
        self,
        pr: Any,
        hydration: Dict[str, Any],
        repo_hydrator: RepositoryHydrator,
        *,
        include_future: bool = True,
        existing_stage: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the maintainability stage for one hydrated PR."""
        pr_number = _get_pr_number(pr)
        owner = getattr(repo_hydrator, "owner", None)
        repo = getattr(repo_hydrator, "name", None)
        targets = _snapshot_targets(hydration, include_future=include_future)
        checkpointed = _checkpointed_results(existing_stage)
        rows: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        smell_findings: List[Dict[str, Any]] = []
        smell_results: List[Dict[str, Any]] = []
        after_row: Dict[str, Any] | None = None
        after_snapshot_path: Path | None = None
        created_at = utc_now_text()

        _log_info(
            "PR {pr_number}: multimetric snapshots={count}".format(
                pr_number=pr_number,
                count=len(targets),
            )
        )

        for target in targets:
            task = _snapshot_task(
                cohort=self.cohort,
                owner=owner,
                repo=repo,
                pr=pr,
                target=target,
            )
            label = task.snapshot_label
            row: Dict[str, Any]
            if label in checkpointed:
                row = dict(checkpointed[label])
                _log_info(f"PR {pr_number}: reusing checkpointed multimetric result for {label}")
            else:
                skip_reason = _future_snapshot_tool_skip_reason(hydration, label)
                if skip_reason and after_row:
                    row = reuse_snapshot_row(
                        source_row=after_row,
                        target_task=task,
                        reuse_reason=skip_reason,
                        created_at_utc=created_at,
                    )
                    _log_info(
                        f"PR {pr_number}: reused after multimetric metrics for {label} ({skip_reason})."
                    )
                else:
                    row = run_multimetric_for_snapshot(
                        task,
                        multimetric_bin=self.multimetric_bin,
                        maintindex_mode=self.maintindex_mode,
                        jobs=None,
                        tool_version_value=self._tool_version,
                        created_at_utc=created_at,
                    )
                    if str(row.get("status") or "") not in _SUCCESS_STATUSES:
                        _log_error(
                            "PR {pr_number}: multimetric {label} status={status} error={error}".format(
                                pr_number=pr_number,
                                label=label,
                                status=row.get("status"),
                                error=row.get("error_message"),
                            )
                        )

            if label == "after" and str(row.get("status") or "").strip().lower() == "success":
                after_row = row
                if task.snapshot_path:
                    after_snapshot_path = task.snapshot_path

            if not row.get("multimetric_reused") or row.get("duplicated_lines_density") is None:
                row.update(compute_snapshot_duplication(task.snapshot_path))

            smell_result = self._smell_stage.analyze_snapshot(task)
            smell_results.append(smell_result)
            smell_findings.extend(
                finding
                for finding in smell_result.get("findings", [])
                if isinstance(finding, dict)
            )

            output_path, artifact_path = _write_multimetric_artifact(
                task=task,
                row=row,
                command_reused=bool(row.get("multimetric_reused")),
                fallback_snapshot_path=after_snapshot_path,
            )
            result = {
                "engine": "multimetric",
                "tool": "multimetric",
                "snapshot_label": label,
                "snapshot_commit": task.snapshot_commit,
                "status": row.get("status"),
                "issue_count": int(smell_result.get("issue_count", 0)),
                "output_path": output_path,
                "artifact_path": artifact_path,
                "code_smell_artifact_path": smell_result.get("artifact_path"),
                "code_smell_status": smell_result.get("status"),
                "code_smell_summary": smell_result.get("summary"),
                "notes": row.get("error_message"),
                "measures": _measures_from_row(row, smell_result),
                "multimetric_snapshot_metric_id": row.get("snapshot_metric_id"),
                "multimetric_reused": bool(row.get("multimetric_reused")),
                "reused_from_snapshot_metric_id": row.get("reused_from_snapshot_metric_id"),
                "reuse_reason": row.get("reuse_reason"),
            }
            rows.append(row)
            results.append(result)
            if progress_callback:
                progress_callback(
                    self._build_progress(
                        targets=targets,
                        results=results,
                        rows=rows,
                        smell_findings=smell_findings,
                    )
                )

        status = _overall_status(results)
        smell_summary = summarize_code_smells(smell_findings)
        summary = {
            "target_snapshots": len(targets),
            "completed_snapshots": len(results),
            **smell_summary,
            "code_smell_detection_skipped": False,
            "code_smell_snapshot_results": smell_results,
            "multimetric_snapshot_rows": rows,
            "multimetric_snapshot_rows_json": json_dumps(rows),
            "snapshot_measures": {
                str(result.get("snapshot_label")): result.get("measures") for result in results
            },
            "snapshot_metric_ids": {
                str(row.get("snapshot_label")): row.get("snapshot_metric_id") for row in rows
            },
            "snapshot_pre_post_metrics": _pre_post_metrics(rows),
            "maintainability_future_snapshot_metrics": _future_metric_deltas(rows, hydration),
            "future_snapshot_availability": hydration.get("future_snapshot_availability") or {},
            "future_impact_summary": _future_impact_summary(hydration),
        }
        payload = {
            "engine": "multimetric",
            "status": status,
            "maintainability_indicators": {
                "engine": "multimetric",
                "selected_tools": ["multimetric", *CODE_SMELL_SELECTED_TOOLS],
                "code_smell_detection_skipped": False,
                "code_smells": smell_findings,
                "summary": summary,
                "results": results,
                "other_indicators": [],
                "notes": (
                    "Multimetric emits numeric maintainability measures. Code-smell "
                    "instances are collected by DesigniteJava, DesignitePython/DPy, "
                    "PMD, ESLint, Cppcheck, and clang-tidy when those tools are configured."
                ),
            },
        }
        _log_info(
            "PR {pr_number}: maintainability(multimetric) status={status} snapshots={snapshots}".format(
                pr_number=pr_number,
                status=status,
                snapshots=len(results),
            )
        )
        return payload
