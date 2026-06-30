"""Discovery helpers for curation processed PR parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CohortParquetFiles:
    """Processed PR parquet files for one curation cohort."""

    cohort: str
    paths: tuple[Path, ...]


def discover_processed_pr_parquets(curation_data_dir: Path) -> list[CohortParquetFiles]:
    """Return ``processed_pr_batch-*.parquet`` inputs grouped by cohort directory."""
    root = Path(curation_data_dir)
    if not root.exists():
        return []

    cohorts: list[CohortParquetFiles] = []
    for cohort_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = tuple(
            sorted(
                (
                    cohort_dir / "output" / "processed-data"
                ).glob("*/processed_pr_batch-*.parquet")
            )
        )
        if paths:
            cohorts.append(CohortParquetFiles(cohort=cohort_dir.name, paths=paths))
    return cohorts


def filter_excluded_cohorts(
    cohort_inputs: list[CohortParquetFiles],
    excluded_agents: tuple[str, ...],
) -> list[CohortParquetFiles]:
    """Drop excluded agent cohorts before opening their parquet files."""
    excluded = {agent.strip().casefold() for agent in excluded_agents if agent.strip()}
    if not excluded:
        return cohort_inputs
    return [
        cohort_input
        for cohort_input in cohort_inputs
        if cohort_input.cohort.strip().casefold() not in excluded
    ]
