"""Refactoring-focused streaming analysis for the post-processing pipeline.

This pipeline reads original-PR refactoring metrics produced during curation.
It does not invoke refactoring tools; it summarizes persisted metrics, writes
the refactoring result JSON, and fans out into the characteristics and
longitudinal refactoring companion analyses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ANALYSIS_DIR / "config"
PLOTTER_DIR = ANALYSIS_DIR / "plotters"
UTILITY_DIR = ANALYSIS_DIR / "utility"
for candidate in (CONFIG_DIR, PLOTTER_DIR, UTILITY_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import analysis_config
import storage_config
from analysis_runtime_utility import (
    AnalysisLogger,
    log_cohort_input_counts,
    make_progress_logger,
    release_process_memory,
)
from characteristics_analysis_utility import load_characteristics_topic_groups
from characteristics_refactoring_pipeline import (
    run_characteristics_refactoring_analysis_from_payload,
)
from curation_parquet_utility import (
    CohortParquetFiles,
    discover_processed_pr_parquets,
    filter_excluded_cohorts,
)
from longitudinal_refactoring_pipeline import (
    run_longitudinal_refactoring_analysis_from_payload,
)
from plotting_utility import set_plot_data_writes_enabled
from refactoring_analysis_plotter import (
    PLOT_OUTPUT_STEMS,
    write_refactoring_analysis_plots_from_payload,
)
from refactoring_analysis_utility import MURPHY_HILL_COUNT_SOURCE_TAXONOMY
from refactoring_streaming_analysis_utility import stream_refactoring_analysis


OUTPUT_FILENAME = "refactoring_analysis_results.json"


def _output_path(analysis_output_dir: Path) -> Path:
    """Return the canonical refactoring JSON output path."""
    return Path(analysis_output_dir) / "refactoring" / OUTPUT_FILENAME


def _plots_dir(analysis_output_dir: Path) -> Path:
    """Return the refactoring plot output directory."""
    return Path(analysis_output_dir) / "refactoring" / "plots"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a stable, sorted JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _run_with_streaming_inputs(
    *,
    cohort_inputs: list[CohortParquetFiles],
    excluded_agents: tuple[str, ...],
    murphy_hill_count_source: str,
    plots_output_dir: Path | None = None,
    analysis_output_dir: Path | None = None,
    topic_classification_output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Stream curation rows and write all refactoring companion artifacts."""
    logger = AnalysisLogger("refactoring")
    log_cohort_input_counts(logger, cohort_inputs)
    logger.log("streaming_refactoring_analysis_start")
    topic_group_records, include_domain = load_characteristics_topic_groups(
        topic_classification_output_dir
    )
    accumulator = stream_refactoring_analysis(
        cohort_inputs,
        excluded_agents=excluded_agents,
        murphy_hill_count_source=murphy_hill_count_source,
        topic_group_records=topic_group_records,
        progress_logger=make_progress_logger(logger, prefix="streaming"),
    )
    logger.log("streaming_refactoring_analysis_complete")
    results = accumulator.result_payload()
    compact_payload = accumulator.compact_payload()
    if plots_output_dir is not None:
        logger.log("writing_refactoring_plots")
        write_refactoring_analysis_plots_from_payload(
            compact_payload,
            plots_output_dir,
            logger=logger,
        )
        release_process_memory(logger, stage="refactoring_plot_memory_released")
        logger.log("refactoring_plots_written")
    if analysis_output_dir is not None:
        logger.log("running_refactoring_characteristics_analysis")
        run_characteristics_refactoring_analysis_from_payload(
            compact_payload,
            analysis_output_dir=analysis_output_dir,
            include_domain=include_domain,
            logger=logger,
        )
        release_process_memory(
            logger,
            stage="refactoring_characteristics_memory_released",
        )
        logger.log("refactoring_characteristics_analysis_written")
        logger.log("running_refactoring_longitudinal_analysis")
        run_longitudinal_refactoring_analysis_from_payload(
            compact_payload.get("longitudinal_values", {}),
            analysis_output_dir=analysis_output_dir,
            logger=logger,
        )
        release_process_memory(
            logger,
            stage="refactoring_longitudinal_memory_released",
        )
        logger.log("refactoring_longitudinal_analysis_written")
    return results


def run_refactoring_analysis_pipeline(
    *,
    curation_data_dir: Path | str | None = None,
    analysis_output_dir: Path | str | None = None,
    topic_classification_output_dir: Path | str | None = None,
    excluded_agents: tuple[str, ...] | None = None,
    murphy_hill_count_source: str | None = None,
) -> dict[str, Any]:
    """Run refactoring analysis and write JSON plus plot outputs."""
    resolved_curation_data_dir = Path(
        curation_data_dir or storage_config.CURATION_DATA_DIR
    )
    resolved_analysis_output_dir = Path(
        analysis_output_dir or storage_config.ANALYSIS_OUTPUT_DIR
    )
    resolved_topic_classification_output_dir = (
        topic_classification_output_dir
        if topic_classification_output_dir is not None
        else storage_config.TOPIC_CLASSIFICATION_OUTPUT_DIR
    )

    resolved_excluded_agents = tuple(
        excluded_agents
        if excluded_agents is not None
        else getattr(analysis_config, "EXCLUDED_AGENTS", ())
    )
    cohort_inputs = filter_excluded_cohorts(
        discover_processed_pr_parquets(resolved_curation_data_dir),
        resolved_excluded_agents,
    )
    resolved_murphy_hill_count_source = (
        murphy_hill_count_source
        if murphy_hill_count_source is not None
        else getattr(
            analysis_config,
            "MURPHY_HILL_COUNT_SOURCE",
            MURPHY_HILL_COUNT_SOURCE_TAXONOMY,
        )
    )
    plot_mode = bool(getattr(analysis_config, "PLOT_MODE", False))
    previous_plot_data_writes = set_plot_data_writes_enabled(not plot_mode)
    try:
        results = _run_with_streaming_inputs(
            cohort_inputs=cohort_inputs,
            excluded_agents=resolved_excluded_agents,
            murphy_hill_count_source=resolved_murphy_hill_count_source,
            plots_output_dir=_plots_dir(resolved_analysis_output_dir),
            analysis_output_dir=resolved_analysis_output_dir,
            topic_classification_output_dir=resolved_topic_classification_output_dir,
        )
        _write_json(_output_path(resolved_analysis_output_dir), results)
        return results
    finally:
        set_plot_data_writes_enabled(previous_plot_data_writes)


def main() -> None:
    """Run the refactoring pipeline when this file is invoked directly."""
    results = run_refactoring_analysis_pipeline()
    output_path = _output_path(storage_config.ANALYSIS_OUTPUT_DIR)
    if bool(getattr(analysis_config, "PLOT_MODE", False)):
        print(
            "[post-processing/analysis] Wrote refactoring analysis and plots "
            f"in plot mode to {output_path}"
        )
        return
    print(f"[post-processing/analysis] Wrote refactoring analysis to {output_path}")
    print(
        "[post-processing/analysis] Eligible PR count: "
        f"{results['eligible_pr_count']}; total refops: "
        f"{results['overall']['total_standardized_refops']}"
    )


if __name__ == "__main__":
    main()
