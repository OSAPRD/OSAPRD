"""Maintainability detail analysis for Multimetric snapshot outputs.

The primary maintainability pipeline uses compact curation summaries. This
companion consumes either embedded curation snapshot rows or legacy external
``multimetric_snapshot_metrics.parquet`` files to produce detailed Multimetric
boxplots, characteristic heatmaps, longitudinal plots, and coverage JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PLOTTER_DIR = ANALYSIS_DIR / "plotters"
UTILITY_DIR = ANALYSIS_DIR / "utility"
for candidate in (PLOTTER_DIR, UTILITY_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from longitudinal_analysis_plotter import (
    longitudinal_line_plot_grid_stems,
    longitudinal_line_plot_stems,
    write_longitudinal_line_plot_grids,
    write_longitudinal_line_plots,
)
from longitudinal_analysis_utility import build_longitudinal_results_from_payload
from characteristics_analysis_plotter import (
    characteristic_heatmap_stems,
    extended_characteristic_heatmap_stems,
    write_characteristics_heatmaps,
    write_extended_characteristics_heatmaps,
    write_characteristics_plots,
)
from characteristics_analysis_utility import (
    build_characteristics_results_from_payload,
    characteristics_output_dir,
    characteristics_output_path,
    write_characteristics_json,
)
from maintainability_multimetrics_plotter import (
    boxplot_output_stems,
    multimetric_quality_grid_stems,
    write_multimetric_boxplots_from_payload,
    write_multimetric_quality_boxplot_grid_from_payload,
)
from maintainability_multimetrics_streaming_utility import (
    stream_embedded_multimetric_analysis,
    stream_multimetric_analysis,
)
from maintainability_multimetrics_utility import (
    MULTIMETRIC_METRICS,
    MULTIMETRIC_PLOT_METRICS,
    MULTIMETRIC_QUALITY_GRID_METRICS,
    OUTPUT_FILENAME,
)
from plotting_utility import remove_plot_outputs


MULTIMETRIC_LONGITUDINAL_QUALITY_GRID_METRICS = (
    "original_smell_count",
    "original_code_smell_density",
    "original_duplication_density",
    "comment_ratio",
    "cyclomatic_complexity_per_kloc",
    "halstead_volume_per_kloc",
    "maintainability_index",
)
MULTIMETRIC_LONGITUDINAL_QUALITY_GRID_WITH_FANOUT_METRICS = (
    "original_smell_count",
    "original_code_smell_density",
    "original_duplication_density",
    "comment_ratio",
    "fanout_external_per_kloc",
    "cyclomatic_complexity_per_kloc",
    "halstead_volume_per_kloc",
    "maintainability_index",
)
MULTIMETRIC_LONGITUDINAL_GRID_SPECS = (
    {
        "stem": "maintainability_quality_longitudinal_line_grid",
        "rows": 2,
        "columns": 4,
        "center_incomplete_final_row": True,
        "figure_width_inches": 9.16,
        "figure_height_inches": 2.25,
        "line_width": 0.75,
        "marker_size": 1.25,
        "y_axis_label_fontsize": 6.5,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "legend_max_columns": 20,
        "bottom_margin": 0.14,
        "hspace": 0.2,
        "wspace": 0.35,
        "allow_zero_floor_when_positive": False,
        "final_row_wspace_multiplier": 2.0,
        "log_scale_metrics": ("halstead_volume_per_kloc",),
        "y_axis_label_overrides": {
            "original_duplication_density": "Duplicated Lines\nDensity (%)",
            "cyclomatic_complexity_per_kloc": "Cyclomatic Complexity\nper KLOC",
            "halstead_volume_per_kloc": "Halstead Volume\nper KLOC",
        },
        "metrics": MULTIMETRIC_LONGITUDINAL_QUALITY_GRID_METRICS,
    },
    {
        "stem": "maintainability_quality_longitudinal_line_grid_with_fanout",
        "rows": 2,
        "columns": 4,
        "center_incomplete_final_row": False,
        "figure_width_inches": 11.16,
        "figure_height_inches": 2.25,
        "line_width": 0.75,
        "marker_size": 1.25,
        "y_axis_label_fontsize": 6.5,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "legend_max_columns": 20,
        "bottom_margin": 0.14,
        "hspace": 0.2,
        "wspace": 0.35,
        "allow_zero_floor_when_positive": False,
        "final_row_wspace_multiplier": 2.0,
        "log_scale_metrics": ("halstead_volume_per_kloc",),
        "y_axis_label_overrides": {
            "original_duplication_density": "Duplicated Lines\nDensity (%)",
            "cyclomatic_complexity_per_kloc": "Cyclomatic Complexity\nper KLOC",
            "halstead_volume_per_kloc": "Halstead Volume\nper KLOC",
            "fanout_external_per_kloc": "Fan Out per KLOC",
        },
        "metrics": MULTIMETRIC_LONGITUDINAL_QUALITY_GRID_WITH_FANOUT_METRICS,
    },
)
MULTIMETRIC_CHARACTERISTICS_HEATMAP_METRICS = MULTIMETRIC_QUALITY_GRID_METRICS
MULTIMETRIC_FANOUT_EXTENDED_HEATMAP_SUFFIX = "_with_fanout"
MULTIMETRIC_CHARACTERISTICS_HEATMAP_WITH_FANOUT_METRICS = (
    "original_code_smell_density_delta",
    "original_duplication_density",
    "comment_ratio",
    "fanout_external_per_kloc",
    "cyclomatic_complexity_per_kloc",
    "halstead_volume_per_kloc",
    "maintainability_index",
)
MULTIMETRIC_QUALITY_GRID_WITH_FANOUT_METRICS = (
    "original_smells_delta",
    "original_code_smell_density_delta",
    "original_duplication_density",
    "comment_ratio",
    "fanout_external_per_kloc",
    "cyclomatic_complexity_per_kloc",
    "halstead_volume_per_kloc",
    "maintainability_index",
)
CHARACTERISTICS_OUTPUT_FILENAME = "characteristics_maintainability_multimetric_results.json"
LONGITUDINAL_OUTPUT_FILENAME = "longitudinal_maintainability_multimetric_results.json"
PLOT_OUTPUT_STEMS = (
    tuple(
        f"boxplots/multimetric/{stem.split('/', 1)[1]}"
        for stem in boxplot_output_stems(MULTIMETRIC_PLOT_METRICS)
    )
    + multimetric_quality_grid_stems()
    + longitudinal_line_plot_stems(MULTIMETRIC_PLOT_METRICS)
    + longitudinal_line_plot_grid_stems(MULTIMETRIC_LONGITUDINAL_GRID_SPECS)
    + extended_characteristic_heatmap_stems(include_domain=True)
    + characteristic_heatmap_stems(
        include_domain=True,
        stem_suffix=MULTIMETRIC_FANOUT_EXTENDED_HEATMAP_SUFFIX,
    )
    + extended_characteristic_heatmap_stems(
        include_domain=True,
        stem_suffix=MULTIMETRIC_FANOUT_EXTENDED_HEATMAP_SUFFIX,
    )
)
REMOVED_PLOT_OUTPUT_STEMS = tuple(
    f"boxplots/multimetric/{metric}_delta_boxplot_by_cohort"
    for metric in MULTIMETRIC_METRICS
    if metric not in MULTIMETRIC_PLOT_METRICS
) + ("boxplots-grid/multimetric/maintainability_quality_metrics_boxplots_by_cohort",)


def plot_output_stems() -> tuple[str, ...]:
    """Return expected Multimetric detail plot stems."""
    return PLOT_OUTPUT_STEMS


def _output_dir(analysis_output_dir: Path | str) -> Path:
    """Return the maintainability output directory for Multimetric results."""
    return Path(analysis_output_dir) / "maintainability"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a stable, sorted JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _cohort_values_from_scope_payload(
    metric_values_by_scope: dict[str, Any],
    metric: str,
) -> dict[str, list[float]]:
    """Extract per-cohort metric lists from the main maintainability payload."""
    by_scope = metric_values_by_scope.get(metric, {})
    if not isinstance(by_scope, dict):
        return {}
    values_by_cohort: dict[str, list[float]] = {}
    for scope, values in by_scope.items():
        scope_text = str(scope)
        if not scope_text.startswith("cohort:") or not isinstance(values, list):
            continue
        values_by_cohort[scope_text[len("cohort:") :]] = [
            float(value)
            for value in values
            if value is not None
        ]
    return values_by_cohort


def _inject_original_maintainability_metrics(
    payload: Any,
    maintainability_payload: dict[str, Any],
) -> None:
    """Merge curation-native smell/duplication values into Multimetric payloads.

    Multimetric does not own the custom duplicated-lines density or the
    standardized smell-density values. Those values are already available in the
    maintainability accumulator, so they are injected before plotting.
    """
    original_delta_metrics = {
        "SmellsDelta": "original_smells_delta",
        "CodeSmellDensityDelta": "original_code_smell_density_delta",
        "DuplicationDensity": "original_duplication_density",
    }
    original_longitudinal_metrics = {
        "SmellCount": "original_smell_count",
        "CodeSmellDensity": "original_code_smell_density",
        "DuplicationDensity": "original_duplication_density",
    }
    metric_values_by_scope = maintainability_payload.get("metric_values_by_scope") or {}
    if isinstance(metric_values_by_scope, dict):
        for source_metric, target_metric in original_delta_metrics.items():
            original_delta = _cohort_values_from_scope_payload(
                metric_values_by_scope,
                source_metric,
            )
            if original_delta:
                payload.delta_values_by_metric[target_metric] = original_delta

    original_characteristics = maintainability_payload.get(
        "characteristic_metric_values",
    ) or {}
    if isinstance(original_characteristics, dict):
        for dimension, by_level in original_characteristics.items():
            if not isinstance(by_level, dict):
                continue
            for level, by_metric in by_level.items():
                if not isinstance(by_metric, dict):
                    continue
                for source_metric, target_metric in original_delta_metrics.items():
                    source = by_metric.get(source_metric)
                    if not isinstance(source, dict):
                        continue
                    payload.characteristic_metric_values[dimension][level][
                        target_metric
                    ] = {
                        str(scope): [
                            float(value)
                            for value in values
                            if value is not None
                        ]
                        for scope, values in source.items()
                        if isinstance(values, list)
                    }

    original_longitudinal = maintainability_payload.get("longitudinal_values") or {}
    if isinstance(original_longitudinal, dict):
        for cohort, by_label in original_longitudinal.items():
            if not isinstance(by_label, dict):
                continue
            for label, by_metric in by_label.items():
                if not isinstance(by_metric, dict):
                    continue
                for source_metric, target_metric in original_longitudinal_metrics.items():
                    values = by_metric.get(source_metric)
                    if not isinstance(values, list):
                        continue
                    payload.longitudinal_values[cohort][label][target_metric] = [
                        float(value)
                        for value in values
                        if value is not None
                    ]


def run_maintainability_multimetrics_analysis_from_payload(
    *,
    analysis_output_dir: Path | str,
    multimetric_output_dir: Path | str | None,
    maintainability_payload: dict[str, Any],
    include_domain: bool,
    logger: Any | None = None,
    multimetric_source: str = "auto",
) -> dict[str, Any] | None:
    """Run Multimetric companion analysis from embedded or external rows."""
    normalized_source = str(multimetric_source or "auto").strip().casefold()
    if normalized_source == "off":
        return None
    pr_index = list(maintainability_payload.get("multimetric_pr_index", []))
    embedded_rows = [
        row
        for row in maintainability_payload.get("embedded_multimetric_snapshot_rows", [])
        if isinstance(row, dict)
    ]
    payload = None
    if normalized_source in {"auto", "input"} and embedded_rows:
        payload = stream_embedded_multimetric_analysis(
            snapshot_rows=embedded_rows,
            pr_index=pr_index,
            logger=logger,
        )
    elif normalized_source == "input":
        raise FileNotFoundError(
            "multimetric_source=input was requested, but the curation input "
            "does not contain embedded Multimetric snapshot rows."
        )
    if payload is None and normalized_source in {"auto", "external"}:
        try:
            payload = stream_multimetric_analysis(
                multimetric_output_dir=multimetric_output_dir,
                pr_index=pr_index,
                logger=logger,
            )
        except FileNotFoundError:
            if normalized_source == "external":
                raise
            if logger is not None:
                logger.log(
                    "maintainability_multimetric_external_source_missing",
                    external_dir=str(multimetric_output_dir)
                    if multimetric_output_dir is not None
                    else None,
                )
            payload = None
    if payload is None and normalized_source == "external":
        raise FileNotFoundError(
            "multimetric_source=external was requested, but no external "
            "Multimetric snapshot parquet rows were found."
        )
    if payload is None:
        return None
    _inject_original_maintainability_metrics(payload, maintainability_payload)
    if logger is not None:
        coverage = payload.coverage_summary()
        logger.log(
            "maintainability_multimetric_coverage",
            eligible_prs=coverage["eligible_pr_count"],
            matched_prs=coverage["matched_pr_count"],
            unmatched_prs=coverage["unmatched_pr_count"],
            delta_ready_prs=coverage["delta_pr_count"],
            delta_missing_prs=coverage["delta_missing_pr_count"],
            longitudinal_ready_prs=coverage["longitudinal_pr_count"],
            longitudinal_missing_prs=coverage["longitudinal_missing_pr_count"],
            failed_parquets=len(payload.failed_parquets),
        )
    output_dir = _output_dir(analysis_output_dir)
    plots_dir = output_dir / "plots"
    plot_payload = payload.plot_payload()
    remove_plot_outputs(plots_dir, REMOVED_PLOT_OUTPUT_STEMS)
    if logger is not None:
        logger.log("writing_maintainability_multimetric_boxplots")
    write_multimetric_boxplots_from_payload(
        delta_values_by_metric=plot_payload["delta_values_by_metric"],
        delta_metadata_by_metric=plot_payload["delta_metadata_by_metric"],
        output_dir=plots_dir,
        metrics=MULTIMETRIC_PLOT_METRICS,
        boxplot_subdir="boxplots/multimetric",
    )
    if logger is not None:
        logger.log("writing_maintainability_multimetric_boxplot_grids")
    write_multimetric_quality_boxplot_grid_from_payload(
        delta_values_by_metric=plot_payload["delta_values_by_metric"],
        output_dir=plots_dir,
        metrics=MULTIMETRIC_QUALITY_GRID_METRICS,
    )
    write_multimetric_quality_boxplot_grid_from_payload(
        delta_values_by_metric=plot_payload["delta_values_by_metric"],
        output_dir=plots_dir,
        metrics=MULTIMETRIC_QUALITY_GRID_WITH_FANOUT_METRICS,
        stem="maintainability_quality_metrics_normalized_boxplots_by_cohort_with_fanout",
        layout="3x3",
        figsize=(7.16, 5.25),
        ncols=3,
        center_incomplete_final_row=True,
        comparison_stat_y=-0.15,
        comparison_stat_label_size=4.8,
        metric_axis_labels={
            "original_smells_delta": "\u0394 Smell Count",
            "comment_ratio": "\u0394 Comment Ratio (%)",
            "fanout_external_per_kloc": "\u0394 Fan Out per KLOC",
        },
    )
    write_multimetric_quality_boxplot_grid_from_payload(
        delta_values_by_metric=plot_payload["delta_values_by_metric"],
        output_dir=plots_dir,
        metrics=MULTIMETRIC_QUALITY_GRID_WITH_FANOUT_METRICS,
        stem="maintainability_quality_metrics_normalized_boxplots_by_cohort_with_fanout_2x4",
        layout="2x4",
        figsize=(9.16, 3.5),
        ncols=4,
        wspace=0.25,
        comparison_stat_y=-0.15,
        comparison_stat_label_size=4.8,
        metric_axis_labels={
            "original_smells_delta": "\u0394 Smell Count",
            "comment_ratio": "\u0394 Comment Ratio (%)",
            "fanout_external_per_kloc": "\u0394 Fan Out per KLOC",
        },
    )
    if logger is not None:
        logger.log("writing_maintainability_multimetric_characteristics_heatmaps")
    characteristic_results = build_characteristics_results_from_payload(
        plot_payload["characteristic_metric_values"],
        metrics=MULTIMETRIC_CHARACTERISTICS_HEATMAP_METRICS,
        include_domain=include_domain,
    )
    characteristic_output_dir = characteristics_output_dir(
        analysis_output_dir,
        "maintainability-multimetric",
    )
    write_characteristics_plots(
        None,
        output_dir=characteristic_output_dir,
        metrics=MULTIMETRIC_CHARACTERISTICS_HEATMAP_METRICS,
        results=characteristic_results,
        include_domain=include_domain,
    )
    if logger is not None:
        logger.log("writing_maintainability_multimetric_extended_heatmaps")
    write_extended_characteristics_heatmaps(
        output_dir=characteristic_output_dir,
        companion_output_dir=characteristics_output_dir(
            analysis_output_dir,
            "refactoring",
        ),
        primary_analysis_group="Maintainability",
        companion_analysis_group="Refactoring",
        include_domain=include_domain,
        write_combined_grid_without_domain=True,
    )
    if logger is not None:
        logger.log("writing_maintainability_multimetric_fanout_extended_heatmaps")
    fanout_characteristic_results = build_characteristics_results_from_payload(
        plot_payload["characteristic_metric_values"],
        metrics=MULTIMETRIC_CHARACTERISTICS_HEATMAP_WITH_FANOUT_METRICS,
        include_domain=include_domain,
    )
    write_characteristics_heatmaps(
        fanout_characteristic_results,
        output_dir=characteristic_output_dir,
        metrics=MULTIMETRIC_CHARACTERISTICS_HEATMAP_WITH_FANOUT_METRICS,
        include_domain=include_domain,
        stem_suffix=MULTIMETRIC_FANOUT_EXTENDED_HEATMAP_SUFFIX,
    )
    write_extended_characteristics_heatmaps(
        output_dir=characteristic_output_dir,
        companion_output_dir=characteristics_output_dir(
            analysis_output_dir,
            "refactoring",
        ),
        primary_analysis_group="Maintainability",
        companion_analysis_group="Refactoring",
        include_domain=include_domain,
        primary_heatmap_stem_suffix=MULTIMETRIC_FANOUT_EXTENDED_HEATMAP_SUFFIX,
        extended_stem_suffix=MULTIMETRIC_FANOUT_EXTENDED_HEATMAP_SUFFIX,
        write_combined_grid_without_domain=True,
    )
    if logger is not None:
        logger.log("writing_maintainability_multimetric_line_plots")
    longitudinal_metrics = tuple(
        dict.fromkeys(
            tuple(MULTIMETRIC_PLOT_METRICS)
            + tuple(MULTIMETRIC_LONGITUDINAL_QUALITY_GRID_METRICS)
            + tuple(MULTIMETRIC_LONGITUDINAL_QUALITY_GRID_WITH_FANOUT_METRICS)
        )
    )
    longitudinal_results = build_longitudinal_results_from_payload(
        plot_payload["longitudinal_values"],
        metrics=longitudinal_metrics,
    )
    longitudinal_output_dir = (
        Path(analysis_output_dir) / "longitudinal" / "maintainability-multimetric"
    )
    write_longitudinal_line_plots(
        output_dir=longitudinal_output_dir,
        metrics=MULTIMETRIC_PLOT_METRICS,
        results=longitudinal_results,
        x_labelpad=5.0,
    )
    if logger is not None:
        logger.log("writing_maintainability_multimetric_line_plot_grids")
    write_longitudinal_line_plot_grids(
        output_dir=longitudinal_output_dir,
        grid_specs=MULTIMETRIC_LONGITUDINAL_GRID_SPECS,
        results=longitudinal_results,
    )
    results = payload.result_payload()
    _write_json(output_dir / OUTPUT_FILENAME, results)
    write_characteristics_json(
        characteristics_output_path(
            analysis_output_dir,
            "maintainability-multimetric",
            CHARACTERISTICS_OUTPUT_FILENAME,
        ),
        {
            "metrics": characteristic_results.get("metrics", []),
            "fanout_extended_heatmap_metrics": list(
                MULTIMETRIC_CHARACTERISTICS_HEATMAP_WITH_FANOUT_METRICS
            ),
            "language_scope": characteristic_results.get("language_scope", []),
            "popularity_buckets": characteristic_results.get("popularity_buckets", {}),
            "domain_enabled": bool(characteristic_results.get("domain_enabled", False)),
            "dimensions": list((characteristic_results.get("dimensions") or {}).keys()),
            **(
                {"domain_scope": characteristic_results.get("domain_scope", [])}
                if bool(characteristic_results.get("domain_enabled", False))
                else {}
            ),
        },
    )
    _write_json(
        longitudinal_output_dir / LONGITUDINAL_OUTPUT_FILENAME,
        {
            "eligible_pr_count": longitudinal_results.get("eligible_pr_count", 0),
            "timepoints": longitudinal_results.get("timepoints", []),
            "metrics": list((longitudinal_results.get("metrics") or {}).keys()),
            "plot_metrics": list(MULTIMETRIC_PLOT_METRICS),
            "grid_metrics": list(MULTIMETRIC_LONGITUDINAL_QUALITY_GRID_METRICS),
            "fanout_grid_metrics": list(
                MULTIMETRIC_LONGITUDINAL_QUALITY_GRID_WITH_FANOUT_METRICS
            ),
        },
    )
    return results
