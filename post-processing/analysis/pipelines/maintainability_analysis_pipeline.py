"""Maintainability-focused streaming analysis for the post-processing pipeline.

This pipeline consumes maintainability and code-smell metrics already written by
curation. It summarizes the embedded values, writes plots/JSON, and optionally
runs the Multimetric detail companion analysis from embedded input rows or a
legacy external Multimetric folder.
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
from characteristics_maintainability_metrics_pipeline import (
    run_characteristics_maintainability_metrics_analysis_from_payload,
)
from curation_parquet_utility import (
    CohortParquetFiles,
    discover_processed_pr_parquets,
    filter_excluded_cohorts,
)
from longitudinal_maintainability_metrics_pipeline import (
    run_longitudinal_maintainability_metrics_analysis_from_payload,
)
from maintainability_analysis_plotter import (
    PLOT_OUTPUT_STEMS,
    write_maintainability_analysis_plots_from_payload,
)
from maintainability_analysis_utility import MANTYLA_COUNT_SOURCE_TAXONOMY
from maintainability_multimetrics_pipeline import (
    run_maintainability_multimetrics_analysis_from_payload,
)
from maintainability_streaming_analysis_utility import stream_maintainability_analysis
from plotting_utility import set_plot_data_writes_enabled


OUTPUT_FILENAME = "maintainability_analysis_results.json"
OUTPUT_DIR_NAME = "maintainability"
FILTERED_ON_REFOPS_OUTPUT_DIR_NAME = "maintainability-filtered-on-refops"


def _output_dir_name(require_refops: bool = False) -> str:
    """Return the output group name for the active maintainability filter."""
    return FILTERED_ON_REFOPS_OUTPUT_DIR_NAME if require_refops else OUTPUT_DIR_NAME


def _output_path(analysis_output_dir: Path, require_refops: bool = False) -> Path:
    """Return the canonical maintainability JSON output path."""
    return Path(analysis_output_dir) / _output_dir_name(require_refops) / OUTPUT_FILENAME


def _plots_dir(analysis_output_dir: Path, require_refops: bool = False) -> Path:
    """Return the maintainability plot output directory."""
    return Path(analysis_output_dir) / _output_dir_name(require_refops) / "plots"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a stable, sorted JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve_multimetric_options(
    configured_source: str | None,
    configured_dir: Path | str | None,
) -> tuple[str, Path | None]:
    """Validate the requested Multimetric detail source.

    ``input`` reads rows embedded in processed curation parquet. ``external`` is
    kept for older runs that wrote ``multimetric_snapshot_metrics.parquet``
    files separately. ``auto`` prefers embedded input rows and falls back to the
    external directory only when one exists.
    """
    normalized_source = str(configured_source or "auto").strip().casefold()
    allowed = {"auto", "input", "external", "off"}
    if normalized_source not in allowed:
        joined = ", ".join(sorted(allowed))
        raise ValueError(f"multimetric_source must be one of: {joined}")
    if normalized_source == "off":
        return normalized_source, None
    if configured_dir is None:
        if normalized_source == "external":
            raise ValueError(
                "multimetric_source=external requires "
                "POST_PROCESSING_ANALYSIS_MULTIMETRIC_OUTPUT_DIR or "
                "--multimetric-output-dir."
            )
        return normalized_source, None
    resolved_dir = Path(configured_dir)
    if not resolved_dir.exists():
        if normalized_source == "external":
            raise FileNotFoundError(
                "Configured multimetric output directory does not exist: "
                f"{resolved_dir}"
            )
        return normalized_source, None
    return normalized_source, resolved_dir


def _run_with_streaming_inputs(
    *,
    cohort_inputs: list[CohortParquetFiles],
    excluded_agents: tuple[str, ...],
    mantyla_count_source: str,
    require_refops: bool,
    plots_output_dir: Path | None = None,
    analysis_output_dir: Path | None = None,
    analysis_kind: str = OUTPUT_DIR_NAME,
    topic_classification_output_dir: Path | str | None = None,
    multimetric_output_dir: Path | str | None = None,
    multimetric_source: str = "auto",
) -> dict[str, Any]:
    """Stream curation rows and write maintainability companion artifacts."""
    logger = AnalysisLogger(analysis_kind)
    log_cohort_input_counts(logger, cohort_inputs)
    logger.log("streaming_maintainability_analysis_start")
    topic_group_records, include_domain = load_characteristics_topic_groups(
        topic_classification_output_dir
    )
    accumulator = stream_maintainability_analysis(
        cohort_inputs,
        excluded_agents=excluded_agents,
        mantyla_count_source=mantyla_count_source,
        require_refops=require_refops,
        topic_group_records=topic_group_records,
        progress_logger=make_progress_logger(logger, prefix="streaming"),
    )
    logger.log("streaming_maintainability_analysis_complete")
    results = accumulator.result_payload()
    compact_payload = accumulator.compact_payload()
    if plots_output_dir is not None:
        logger.log("writing_maintainability_plots")
        write_maintainability_analysis_plots_from_payload(
            compact_payload,
            plots_output_dir,
            logger=logger,
        )
        release_process_memory(
            logger,
            stage="maintainability_plot_memory_released",
        )
        logger.log("maintainability_plots_written")
    if analysis_output_dir is not None:
        logger.log("running_maintainability_characteristics_analysis")
        run_characteristics_maintainability_metrics_analysis_from_payload(
            compact_payload,
            analysis_output_dir=analysis_output_dir,
            analysis_kind=analysis_kind,
            include_domain=include_domain,
            logger=logger,
        )
        release_process_memory(
            logger,
            stage="maintainability_characteristics_memory_released",
        )
        logger.log("maintainability_characteristics_analysis_written")
        logger.log("running_maintainability_longitudinal_analysis")
        run_longitudinal_maintainability_metrics_analysis_from_payload(
            compact_payload.get("longitudinal_values", {}),
            analysis_output_dir=analysis_output_dir,
            analysis_kind=analysis_kind,
            logger=logger,
        )
        release_process_memory(
            logger,
            stage="maintainability_longitudinal_memory_released",
        )
        logger.log("maintainability_longitudinal_analysis_written")
        if multimetric_source != "off":
            logger.log(
                "running_maintainability_multimetric_analysis",
                source=multimetric_source,
                external_dir=str(multimetric_output_dir)
                if multimetric_output_dir is not None
                else None,
            )
            multimetric_results = run_maintainability_multimetrics_analysis_from_payload(
                analysis_output_dir=analysis_output_dir,
                multimetric_output_dir=multimetric_output_dir,
                maintainability_payload=compact_payload,
                include_domain=include_domain,
                logger=logger,
                multimetric_source=multimetric_source,
            )
            if multimetric_results is not None:
                release_process_memory(
                    logger,
                    stage="maintainability_multimetric_memory_released",
                )
                logger.log("maintainability_multimetric_analysis_written")
            else:
                logger.log("maintainability_multimetric_analysis_skipped")
    return results


def run_maintainability_analysis_pipeline(
    *,
    curation_data_dir: Path | str | None = None,
    analysis_output_dir: Path | str | None = None,
    topic_classification_output_dir: Path | str | None = None,
    excluded_agents: tuple[str, ...] | None = None,
    mantyla_count_source: str | None = None,
    require_refops: bool | None = None,
    multimetric_output_dir: Path | str | None = None,
    multimetric_source: str | None = None,
) -> dict[str, Any]:
    """Run maintainability analysis and write JSON plus plot outputs."""
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
    resolved_multimetric_output_dir = (
        multimetric_output_dir
        if multimetric_output_dir is not None
        else storage_config.MULTIMETRIC_OUTPUT_DIR
    )
    resolved_multimetric_source = (
        multimetric_source
        if multimetric_source is not None
        else getattr(analysis_config, "MULTIMETRIC_SOURCE", "auto")
    )
    (
        resolved_multimetric_source,
        resolved_multimetric_output_dir,
    ) = _resolve_multimetric_options(
        resolved_multimetric_source,
        resolved_multimetric_output_dir,
    )

    resolved_require_refops = (
        require_refops
        if require_refops is not None
        else bool(getattr(analysis_config, "MAINTAINABILITY_REQUIRE_REFOPS", False))
    )
    output_dir_name = _output_dir_name(resolved_require_refops)
    resolved_excluded_agents = tuple(
        excluded_agents
        if excluded_agents is not None
        else getattr(analysis_config, "EXCLUDED_AGENTS", ())
    )
    cohort_inputs = filter_excluded_cohorts(
        discover_processed_pr_parquets(resolved_curation_data_dir),
        resolved_excluded_agents,
    )
    resolved_mantyla_count_source = (
        mantyla_count_source
        if mantyla_count_source is not None
        else getattr(
            analysis_config,
            "MANTYLA_COUNT_SOURCE",
            MANTYLA_COUNT_SOURCE_TAXONOMY,
        )
    )
    plot_mode = bool(getattr(analysis_config, "PLOT_MODE", False))
    previous_plot_data_writes = set_plot_data_writes_enabled(not plot_mode)
    try:
        results = _run_with_streaming_inputs(
            cohort_inputs=cohort_inputs,
            excluded_agents=resolved_excluded_agents,
            mantyla_count_source=resolved_mantyla_count_source,
            require_refops=resolved_require_refops,
            plots_output_dir=_plots_dir(
                resolved_analysis_output_dir,
                resolved_require_refops,
            ),
            analysis_output_dir=resolved_analysis_output_dir,
            analysis_kind=output_dir_name,
            topic_classification_output_dir=resolved_topic_classification_output_dir,
            multimetric_output_dir=resolved_multimetric_output_dir,
            multimetric_source=resolved_multimetric_source,
        )
        _write_json(_output_path(resolved_analysis_output_dir, resolved_require_refops), results)
        return results
    finally:
        set_plot_data_writes_enabled(previous_plot_data_writes)


def main() -> None:
    """Run the maintainability pipeline when this file is invoked directly."""
    results = run_maintainability_analysis_pipeline()
    require_refops = bool(
        getattr(analysis_config, "MAINTAINABILITY_REQUIRE_REFOPS", False)
    )
    output_path = _output_path(storage_config.ANALYSIS_OUTPUT_DIR, require_refops)
    if bool(getattr(analysis_config, "PLOT_MODE", False)):
        print(
            "[post-processing/analysis] Wrote maintainability analysis and plots "
            f"in plot mode to {output_path}"
        )
        return
    print(f"[post-processing/analysis] Wrote maintainability analysis to {output_path}")
    print(
        "[post-processing/analysis] Eligible PR count: "
        f"{results['eligible_pr_count']}"
    )


if __name__ == "__main__":
    main()
