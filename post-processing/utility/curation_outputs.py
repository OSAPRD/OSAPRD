"""Helpers for discovering files in curation output directories."""

from __future__ import annotations

from pathlib import Path


def candidate_curation_roots(cohort_dir: Path) -> tuple[Path, ...]:
    """Return possible roots that directly contain curation run files."""
    root = Path(cohort_dir)
    roots = [root]
    nested_output = root / "output"
    if nested_output.exists():
        roots.append(nested_output)
    outputs_dir = root / "outputs"
    if outputs_dir.exists():
        for nested_cohort in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
            roots.append(nested_cohort)
            nested_cohort_output = nested_cohort / "output"
            if nested_cohort_output.exists():
                roots.append(nested_cohort_output)
    return tuple(roots)


def discover_aggregate_metric_paths(cohort_dir: Path) -> list[Path]:
    """Find aggregate metrics JSON files below a curation output/cohort directory."""
    paths: dict[Path, None] = {}
    for root in candidate_curation_roots(cohort_dir):
        processed_data_dir = root / "output" / "processed-data"
        if not processed_data_dir.exists():
            continue
        for metrics_dir in processed_data_dir.glob("*/metrics-json"):
            if not metrics_dir.is_dir() or metrics_dir.name.lower() != "metrics-json":
                continue
            for aggregate_path in metrics_dir.rglob("*.json"):
                paths[aggregate_path] = None
    return sorted(paths)


def discover_processing_checkpoint_paths(cohort_dir: Path) -> list[Path]:
    """Find processing checkpoint JSON files below a curation output/cohort directory."""
    paths: dict[Path, None] = {}
    for root in candidate_curation_roots(cohort_dir):
        checkpoints_dir = root / "output" / "checkpoints" / "processing"
        if checkpoints_dir.exists():
            for path in checkpoints_dir.rglob("*.json"):
                paths[path] = None
    return sorted(paths)


def discover_longitudinal_pr_jsonl_paths(cohort_dir: Path) -> list[Path]:
    """Find longitudinal PR JSONL files below a curation output/cohort directory."""
    paths: dict[Path, None] = {}
    for root in candidate_curation_roots(cohort_dir):
        if not root.exists():
            continue
        for path in root.glob("longitudinal_prs_*.jsonl"):
            paths[path] = None
    return sorted(paths)


def root_looks_like_curation_output(path: Path) -> bool:
    """Return whether a path appears to be an individual curation output root."""
    root = Path(path)
    return (
        bool(discover_longitudinal_pr_jsonl_paths(root))
        or bool(discover_aggregate_metric_paths(root))
        or bool(discover_processing_checkpoint_paths(root))
    )
