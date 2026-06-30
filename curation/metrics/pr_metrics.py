"""Per-PR metric orchestration for curation.

This module is the only active metric stage selector. Every PR runs original
before/after refactoring mining and Multimetric maintainability with custom
duplicated-lines density.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from curation.hydration.repository_hydrator import RepositoryHydrator
from curation.metrics.maintainability_multimetric_metrics import (
    MultimetricMaintainabilityMetricsStage,
)
from curation.metrics.refactoring_metrics import RefactoringMetricsStage

METRICS_BACKEND = "multimetric_plus_custom_duplicated_lines_density"


def _get_pr_number(pr: Any) -> Any:
    """Return the PR number from either a DTO-like object or a dict."""
    return getattr(pr, "number", None) if not isinstance(pr, dict) else pr.get("number")


def _stage_status(stage_payload: Optional[Dict[str, Any]]) -> str:
    """Return a normalized stage status for overall-progress reporting."""
    if not isinstance(stage_payload, dict):
        return "not_run"
    return str(stage_payload.get("status", "not_run"))


def _overall_status(
    refactoring_metrics: Optional[Dict[str, Any]],
    maintainability_metrics: Optional[Dict[str, Any]],
) -> str:
    """Summarize the overall PR-metrics status from the two stage payloads."""
    statuses = {
        _stage_status(refactoring_metrics),
        _stage_status(maintainability_metrics),
    }
    statuses.discard("skipped")
    if not statuses:
        return "skipped"
    success_like = {"success", "partial_success"}
    failure_like = {"failed", "partial_failure", "analyzer_failed"}
    if statuses == {"success"}:
        return "success"
    if statuses <= success_like:
        return "partial_success"
    if "not_run" in statuses:
        return "in_progress"
    if statuses.intersection(success_like) and statuses.intersection(failure_like):
        return "partial_failure"
    if statuses.intersection(failure_like):
        return "failed"
    return "partial_failure"


def _assemble_metrics_payload(
    pr: Any,
    hydration: Dict[str, Any],
    repo_hydrator: RepositoryHydrator,
    *,
    include_future: bool,
    refactoring_metrics: Optional[Dict[str, Any]],
    maintainability_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the persisted metrics payload, including partial in-progress state."""
    return {
        "status": _overall_status(refactoring_metrics, maintainability_metrics),
        "metrics_backend": METRICS_BACKEND,
        "repository_owner": repo_hydrator.owner,
        "repository_name": repo_hydrator.name,
        "pr_number": _get_pr_number(pr),
        "has_hydration": bool(hydration),
        "longitudinal_selected": bool(include_future),
        "has_future_snapshots": bool(hydration.get("has_future_snapshots")),
        "refactoring_metrics": refactoring_metrics,
        "maintainability_metrics": maintainability_metrics,
        "notes": (
            "Refactoring-operation mining runs on the original before/after PR. "
            "Maintainability uses Multimetric for source metrics and a custom "
            "duplicated-lines-density implementation."
        ),
    }


def _assemble_metrics_progress(
    pr: Any,
    hydration: Dict[str, Any],
    repo_hydrator: RepositoryHydrator,
    *,
    current_phase: str,
    include_future: bool,
    refactoring_metrics: Optional[Dict[str, Any]],
    maintainability_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a lightweight checkpoint payload for in-flight metric computation."""
    return {
        "status": _overall_status(refactoring_metrics, maintainability_metrics),
        "metrics_backend": METRICS_BACKEND,
        "current_phase": current_phase,
        "repository_owner": repo_hydrator.owner,
        "repository_name": repo_hydrator.name,
        "pr_number": _get_pr_number(pr),
        "has_hydration": bool(hydration),
        "longitudinal_selected": bool(include_future),
        "has_future_snapshots": bool(hydration.get("has_future_snapshots")),
        "refactoring_metrics": refactoring_metrics,
        "maintainability_metrics": maintainability_metrics,
    }


def compute_pr_metrics(
    pr: Any,
    hydration: Dict[str, Any],
    repo_hydrator: RepositoryHydrator,
    *,
    cohort: str | None = None,
    include_future: bool = True,
    existing_metrics: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Compute the two active metric stages for one PR.

    The only supported mode is:
    1. refactoring mining on the original before/after PR snapshots;
    2. Multimetric maintainability plus custom duplicated-lines density.

    """
    existing_metrics = existing_metrics if isinstance(existing_metrics, dict) else {}
    refactoring_metrics = existing_metrics.get("refactoring_metrics")
    maintainability_metrics = existing_metrics.get("maintainability_metrics")

    def _emit_progress(current_phase: str) -> None:
        if progress_callback:
            progress_callback(
                _assemble_metrics_progress(
                    pr,
                    hydration,
                    repo_hydrator,
                    current_phase=current_phase,
                    include_future=include_future,
                    refactoring_metrics=refactoring_metrics,
                    maintainability_metrics=maintainability_metrics,
                )
            )

    def _on_refactoring_progress(stage_payload: Dict[str, Any]) -> None:
        nonlocal refactoring_metrics
        refactoring_metrics = stage_payload
        _emit_progress("refactoring_metrics")

    refactoring_stage = RefactoringMetricsStage()
    refactoring_metrics = refactoring_stage.compute(
        pr,
        hydration,
        repo_hydrator,
        include_future=include_future,
        existing_stage=existing_metrics.get("refactoring_metrics"),
        progress_callback=_on_refactoring_progress,
    )

    def _on_maintainability_progress(stage_payload: Dict[str, Any]) -> None:
        nonlocal maintainability_metrics
        maintainability_metrics = stage_payload
        _emit_progress("maintainability_metrics")

    maintainability_stage = MultimetricMaintainabilityMetricsStage(cohort=cohort)
    maintainability_metrics = maintainability_stage.compute(
        pr,
        hydration,
        repo_hydrator,
        include_future=include_future,
        existing_stage=existing_metrics.get("maintainability_metrics"),
        progress_callback=_on_maintainability_progress,
    )

    return _assemble_metrics_payload(
        pr,
        hydration,
        repo_hydrator,
        include_future=include_future,
        refactoring_metrics=refactoring_metrics,
        maintainability_metrics=maintainability_metrics,
    )
