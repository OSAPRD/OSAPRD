"""Dataset-level counts for the streaming post-processing analysis pipeline.

The dataset pipeline is the first of the three analysis families. It reads the
same processed curation parquet files as the refactoring and maintainability
pipelines, joins optional topic-domain records, and writes aggregate counts plus
composition plots under ``<analysis-output>/dataset``.
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
from curation_parquet_utility import (
    CohortParquetFiles,
    discover_processed_pr_parquets,
    filter_excluded_cohorts,
)
from data_analysis_plotter import (
    PLOT_OUTPUT_STEMS,
    write_data_analysis_plots_from_payload,
)
from dataset_streaming_analysis_utility import stream_dataset_analysis
from plotting_utility import set_plot_data_writes_enabled
from topic_groups_utility import TopicGroupRecord, load_topic_group_records


OUTPUT_FILENAME = "data_analysis_results.json"


def _output_path(analysis_output_dir: Path) -> Path:
    """Return the canonical dataset JSON output path."""
    return Path(analysis_output_dir) / "dataset" / OUTPUT_FILENAME


def _plots_dir(analysis_output_dir: Path) -> Path:
    """Return the dataset plot output directory."""
    return Path(analysis_output_dir) / "dataset" / "plots"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a stable, sorted JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _run_with_streaming_inputs(
    *,
    cohort_inputs: list[CohortParquetFiles],
    topic_group_records: list[TopicGroupRecord],
    excluded_agents: tuple[str, ...],
    plots_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Stream curation inputs and optionally render dataset plots."""
    logger = AnalysisLogger("dataset")
    log_cohort_input_counts(logger, cohort_inputs)
    logger.log("streaming_dataset_analysis_start")
    accumulator = stream_dataset_analysis(
        cohort_inputs,
        topic_group_records=topic_group_records,
        excluded_agents=excluded_agents,
        progress_logger=make_progress_logger(logger, prefix="streaming"),
    )
    logger.log("streaming_dataset_analysis_complete")
    results = accumulator.result_payload()
    if plots_output_dir is not None:
        logger.log("writing_dataset_plots")
        write_data_analysis_plots_from_payload(
            accumulator.plot_payload(),
            accumulator.popularity_scheme,
            plots_output_dir,
            logger=logger,
        )
        release_process_memory(logger, stage="dataset_plot_memory_released")
        logger.log("dataset_plots_written")
    return results


def run_data_analysis_pipeline(
    *,
    curation_data_dir: Path | str | None = None,
    topic_classification_output_dir: Path | str | None = None,
    analysis_output_dir: Path | str | None = None,
    excluded_agents: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run dataset-level analysis counts and write the JSON result."""
    resolved_curation_data_dir = Path(
        curation_data_dir or storage_config.CURATION_DATA_DIR
    )
    resolved_topic_output_dir = Path(
        topic_classification_output_dir
        or storage_config.TOPIC_CLASSIFICATION_OUTPUT_DIR
    )
    resolved_analysis_output_dir = Path(
        analysis_output_dir or storage_config.ANALYSIS_OUTPUT_DIR
    )

    plot_mode = bool(getattr(analysis_config, "PLOT_MODE", False))
    resolved_excluded_agents = tuple(
        excluded_agents
        if excluded_agents is not None
        else getattr(analysis_config, "EXCLUDED_AGENTS", ())
    )
    cohort_inputs = filter_excluded_cohorts(
        discover_processed_pr_parquets(resolved_curation_data_dir),
        resolved_excluded_agents,
    )
    previous_plot_data_writes = set_plot_data_writes_enabled(not plot_mode)
    try:
        topic_group_records = load_topic_group_records(resolved_topic_output_dir)
        results = _run_with_streaming_inputs(
            cohort_inputs=cohort_inputs,
            topic_group_records=topic_group_records,
            excluded_agents=resolved_excluded_agents,
            plots_output_dir=_plots_dir(resolved_analysis_output_dir),
        )
        _write_json(_output_path(resolved_analysis_output_dir), results)
        return results
    finally:
        set_plot_data_writes_enabled(previous_plot_data_writes)


def main() -> None:
    """Run the dataset pipeline when this file is invoked directly."""
    results = run_data_analysis_pipeline()
    output_path = _output_path(storage_config.ANALYSIS_OUTPUT_DIR)
    if bool(getattr(analysis_config, "PLOT_MODE", False)):
        print(
            "[post-processing/analysis] Wrote data analysis results and plots "
            f"in plot mode to {output_path}"
        )
        return
    print(f"[post-processing/analysis] Wrote data analysis results to {output_path}")
    print(
        "[post-processing/analysis] Overall PR count: "
        f"{results['overall']['pull_request_count']} "
        "(longitudinal: "
        f"{results['overall']['longitudinal_pull_request_count']})"
    )


if __name__ == "__main__":
    main()
