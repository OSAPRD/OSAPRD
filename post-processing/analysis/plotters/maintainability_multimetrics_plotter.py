"""Plotting for Multimetric maintainability analysis.

The maintainability pipeline emits compact payloads for before/after deltas,
longitudinal trends, and characteristic slices. This module turns those payloads
into deterministic figure and plot-data files without re-reading curation
parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
UTILITY_DIR = ANALYSIS_DIR / "utility"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from balance_statistics_utility import (
    add_cliffs_delta_ci,
    apply_fdr_correction,
    mann_whitney_u_test,
    numeric_distribution_summary,
)
from maintainability_multimetrics_utility import (
    MULTIMETRIC_METRICS,
    MULTIMETRIC_QUALITY_GRID_METRICS,
    MULTIMETRIC_REPLACEMENT_METRICS,
)
from maintainability_analysis_plotter import (
    _MaintainabilityPlotPayloadConnection,
    _plot_delta_boxplot_grid,
)
from plotting_utility import (
    add_human_median_baseline,
    add_violin_underlay,
    apply_ieee_plot_style,
    apply_percentile_capped_y_axis,
    apply_symlog_y_axis_if_range_exceeds,
    cohort_color_map,
    display_group_labels,
    ieee_boxplot_kwargs,
    order_humans_first,
    require_matplotlib,
    save_figure,
    style_ieee_boxplot,
    write_plot_data,
)


def sql_string_literal(value: object) -> str:
    """Return a single-quoted SQL string literal for legacy query helpers."""
    return "'" + str(value).replace("'", "''") + "'"


MULTIMETRIC_LABELS = {
    "loc": "Lines of Code",
    "comment_ratio": "Comment Ratio (%)",
    "cyclomatic_complexity": "Cyclomatic Complexity",
    "halstead_volume": "Halstead Volume",
    "halstead_difficulty": "Halstead Difficulty",
    "halstead_effort": "Halstead Effort",
    "halstead_bugprop": "Halstead Delivered Bugs",
    "halstead_timerequired": "Halstead Time Required",
    "maintainability_index": "Maintainability Index",
    "fanout_internal": "Fanout Internal",
    "fanout_external": "Fanout External",
    "operands_sum": "Sum of Operands",
    "operands_uniq": "Unique Operands",
    "operators_sum": "Sum of Operators",
    "operators_uniq": "Unique Operators",
    "pylint": "Pylint",
    "tiobe": "TIOBE Quality Score",
    "tiobe_complexity": "TIOBE Complexity",
    "tiobe_duplication": "TIOBE Duplication",
    "tiobe_functional": "Functional Defect Score",
    "halstead_volume_per_kloc": "Halstead Volume per KLOC",
    "halstead_difficulty_per_kloc": "Halstead Difficulty per KLOC",
    "halstead_effort_per_kloc": "Halstead Effort per KLOC",
    "cyclomatic_complexity_per_kloc": "Cyclomatic Complexity per KLOC",
    "halstead_bugprop_per_kloc": "Halstead Delivered Bugs per KLOC",
    "fanout_external_per_kloc": "Fan Out per KLOC",
    "halstead_timerequired_per_kloc": "Halstead Time Required per KLOC",
    "multimetric_duplication_score": "Duplication Score",
    "original_code_smell_density_delta": "Smells per KLOC",
    "original_duplication_density": "Duplicated Lines Density (%)",
}
MULTIMETRIC_BOXPLOT_P_THRESHOLD = 0.001
MULTIMETRIC_BOXPLOT_AXIS_PADDING = 0.1
MILLION_SCALE_METRICS: set[str] = set()
TEN_TO_MINUS_EIGHT_SCALE_METRICS: set[str] = set()
MULTIMETRIC_GRID_METRIC_ALIASES = {
    "original_smells_delta": "SmellsDelta",
    "original_code_smell_density_delta": "CodeSmellDensityDelta",
    "original_duplication_density": "DuplicationDensity",
    "comment_ratio": "CommentDensity",
    "fanout_external": "MultimetricFanOut",
    "fanout_external_per_kloc": "MultimetricFanOutPerKLOC",
    "cyclomatic_complexity_per_kloc": "CCDensity",
    "halstead_volume_per_kloc": "HVDensity",
    "maintainability_index": "MI",
}


def metric_label(metric: str, *, delta: bool = False) -> str:
    """Return the display label for a Multimetric metric."""
    label = MULTIMETRIC_LABELS.get(metric, metric)
    return f"\u0394 {label}" if delta else label


def _plot_value_scale(metric: str) -> float:
    if metric in TEN_TO_MINUS_EIGHT_SCALE_METRICS:
        return 1.0e-8
    return 1.0 / 1_000_000.0 if metric in MILLION_SCALE_METRICS else 1.0


def metric_stem(metric: str) -> str:
    """Return the filename-safe stem for one Multimetric metric."""
    return str(metric).lower()


def boxplot_output_stems(metrics: tuple[str, ...] = MULTIMETRIC_METRICS) -> tuple[str, ...]:
    """Return expected delta boxplot output stems for selected metrics."""
    return tuple(
        f"boxplots/{metric_stem(metric)}_delta_boxplot_by_cohort"
        for metric in metrics
    )


def multimetric_quality_grid_stems() -> tuple[str, ...]:
    """Return expected normalized quality-grid output stems."""
    return (
        "boxplots-grid/multimetric/maintainability_quality_metrics_normalized_boxplots_by_cohort",
        "boxplots-grid/multimetric/maintainability_quality_metrics_normalized_boxplots_by_cohort_with_fanout",
        "boxplots-grid/multimetric/maintainability_quality_metrics_normalized_boxplots_by_cohort_with_fanout_2x4",
    )


def _summary_payload(values: list[float]) -> dict[str, Any]:
    return {**numeric_distribution_summary(values), "n": len(values)}


def _format_delta(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def _format_p_value(value: float | None) -> str:
    if value is None:
        return "NA"
    if value <= MULTIMETRIC_BOXPLOT_P_THRESHOLD:
        return "< 0.001"
    return f"= {value:.3f}"


def _nonempty_groups(con, metric: str) -> dict[str, list[float]]:
    rows = con.execute(
        f"""
        SELECT cohort, value
        FROM analysis_multimetric_delta_metrics
        WHERE metric = {sql_string_literal(metric)}
          AND value IS NOT NULL
          AND cohort IS NOT NULL
          AND NULLIF(trim(CAST(cohort AS VARCHAR)), '') IS NOT NULL
        ORDER BY cohort, value
        """
    ).fetchall()
    grouped: dict[str, list[float]] = {}
    value_scale = _plot_value_scale(metric)
    for cohort, value in rows:
        grouped.setdefault(str(cohort), []).append(float(value) * value_scale)
    return grouped


def _metric_metadata(con, metric: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT files_considered, files_analyzed, tool_version, maintindex_mode
        FROM analysis_multimetric_delta_metrics
        WHERE metric = {sql_string_literal(metric)}
        """
    ).fetchall()
    files_considered = [
        float(row[0]) for row in rows if row[0] is not None
    ]
    files_analyzed = [
        float(row[1]) for row in rows if row[1] is not None
    ]
    tool_versions = sorted(
        {str(row[2]) for row in rows if row[2] is not None and str(row[2]).strip()}
    )
    maintindex_modes = sorted(
        {str(row[3]) for row in rows if row[3] is not None and str(row[3]).strip()}
    )
    return {
        "files_considered": _summary_payload(files_considered),
        "files_analyzed": _summary_payload(files_analyzed),
        "tool_versions": tool_versions,
        "maintindex_modes": maintindex_modes,
    }


def _human_group(groups: dict[str, list[float]]) -> str | None:
    for candidate in ("human", "humans"):
        if candidate in {group.casefold(): group for group in groups}:
            return {group.casefold(): group for group in groups}[candidate]
    return None


def _comparison_payloads(groups: list[str], grouped: dict[str, list[float]]) -> dict[str, Any]:
    human = _human_group(grouped)
    if human is None:
        return {}
    payload: dict[str, Any] = {}
    human_values = grouped.get(human, [])
    for group in groups:
        if group == human:
            continue
        group_values = grouped.get(group, [])
        test = mann_whitney_u_test(group_values, human_values)
        add_cliffs_delta_ci(test, group_values, human_values)
        test["first_group"] = group
        test["second_group"] = human
        payload[group] = {"mann_whitney_u": test}
    return apply_fdr_correction(payload)


def _add_zero_reference_line_if_needed(ax, grouped: dict[str, list[float]]) -> None:
    values = [value for group_values in grouped.values() for value in group_values]
    if values and min(values) < 0.0 < max(values):
        ax.axhline(0.0, color="0.25", linewidth=0.55, alpha=0.7, zorder=1)


def _add_comparison_annotations(
    ax,
    groups: list[str],
    comparison_payload: dict[str, Any],
    *,
    font_size: float = 5.5,
    y: float = -0.11,
) -> None:
    transform = ax.get_xaxis_transform()
    for index, group in enumerate(groups, start=1):
        payload = comparison_payload.get(group)
        if not payload:
            continue
        test = payload.get("mann_whitney_u") or {}
        delta = test.get("cliffs_delta")
        adjusted_p = test.get("adjusted_p_value")
        ax.text(
            index,
            y,
            f"\u03b4 = {_format_delta(delta)}\np {_format_p_value(adjusted_p)}",
            transform=transform,
            ha="center",
            va="top",
            fontsize=font_size,
            clip_on=False,
        )


def _draw_boxplot(
    ax,
    grouped: dict[str, list[float]],
    groups: list[str],
    metric: str,
    *,
    show_x_label: bool = True,
    x_tick_label_size: float = 5.3,
    y_tick_label_size: float = 7.0,
    axis_label_size: float | None = None,
    y_labelpad: float | None = None,
    y_tick_labelpad: float | None = None,
) -> dict[str, Any] | None:
    colors = cohort_color_map(groups)
    ordered_colors = [colors[group] for group in groups]
    if not groups:
        ax.set_xlabel("Cohort" if show_x_label else "", fontsize=axis_label_size)
        ax.set_ylabel(
            metric_label(metric, delta=True),
            fontsize=axis_label_size,
            labelpad=y_labelpad,
        )
        return None
    add_violin_underlay(
        ax,
        [grouped[group] for group in groups],
        colors=ordered_colors,
    )
    try:
        boxplot = ax.boxplot(
            [grouped[group] for group in groups],
            tick_labels=display_group_labels(groups),
            **ieee_boxplot_kwargs(),
        )
    except TypeError:
        boxplot = ax.boxplot(
            [grouped[group] for group in groups],
            labels=display_group_labels(groups),
            **ieee_boxplot_kwargs(),
        )
    style_ieee_boxplot(boxplot, ordered_colors)
    add_human_median_baseline(ax, grouped)
    _add_zero_reference_line_if_needed(ax, grouped)
    y_axis_limits = apply_percentile_capped_y_axis(
        ax,
        grouped,
        show_note=False,
        force_zero_for_nonnegative=False,
    )
    lower, upper = ax.get_ylim()
    if lower >= 0.0:
        lower = max(0.0, lower - MULTIMETRIC_BOXPLOT_AXIS_PADDING)
    else:
        lower = lower - MULTIMETRIC_BOXPLOT_AXIS_PADDING
    ax.set_ylim(lower, upper)
    y_axis_scale = apply_symlog_y_axis_if_range_exceeds(ax, range_threshold=100.0)
    ax.set_xlim(0.45, len(groups) + 0.75)
    ax.set_xlabel("Cohort" if show_x_label else "", fontsize=axis_label_size)
    ax.set_ylabel(
        metric_label(metric, delta=True),
        fontsize=axis_label_size,
        labelpad=y_labelpad,
    )
    ax.tick_params(axis="x", labelsize=x_tick_label_size, pad=1.0)
    y_tick_kwargs: dict[str, float] = {"labelsize": y_tick_label_size}
    if y_tick_labelpad is not None:
        y_tick_kwargs["pad"] = float(y_tick_labelpad)
    ax.tick_params(axis="y", **y_tick_kwargs)
    ax.grid(axis="y", alpha=0.3)
    if y_axis_limits is None:
        return {"scale": y_axis_scale}
    return {**y_axis_limits, "scale": y_axis_scale, "visual_lower": float(lower)}


def write_multimetric_boxplots(
    con,
    *,
    output_dir: Path | str,
    metrics: tuple[str, ...] = MULTIMETRIC_METRICS,
    boxplot_subdir: str = "boxplots",
) -> None:
    """Write individual multimetrics delta boxplots and sidecar JSON."""
    plt, _mdates = require_matplotlib()
    apply_ieee_plot_style()
    resolved_output_dir = Path(output_dir) / boxplot_subdir
    for metric in metrics:
        grouped = _nonempty_groups(con, metric)
        groups = order_humans_first(grouped)
        comparisons = _comparison_payloads(groups, grouped)
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        y_axis_limits = _draw_boxplot(ax, grouped, groups, metric)
        _add_comparison_annotations(ax, groups, comparisons)
        ax.xaxis.set_label_coords(0.5, -0.2)
        fig.subplots_adjust(bottom=0.36)
        stem = f"{metric_stem(metric)}_delta_boxplot_by_cohort"
        save_figure(fig, resolved_output_dir, stem)
        write_plot_data(
            resolved_output_dir,
            stem,
            {
                "plot": stem,
                "plot_type": "boxplot",
                "metric": metric,
                "group_field": "cohort",
                "x_axis": "Cohort",
                "y_axis": metric_label(metric, delta=True),
                "percentile_capped_y_axis": y_axis_limits,
                "groups": {
                    group: _summary_payload(grouped.get(group, []))
                    for group in groups
                },
                "multimetric_metadata": _metric_metadata(con, metric),
                "human_baseline_tests": comparisons,
                "comparison_label_p_threshold": MULTIMETRIC_BOXPLOT_P_THRESHOLD,
                "figure": {
                    "width_inches": 3.5,
                    "height_inches": 2.0,
                    "bottom": 0.36,
                    "comparison_stat_label_size": 5.5,
                    "comparison_stat_y": -0.11,
                    "x_axis_label_y": -0.2,
                },
            },
        )
        plt.close(fig)
        del grouped, groups, comparisons


def write_multimetric_boxplots_from_payload(
    *,
    delta_values_by_metric: dict[str, dict[str, list[float]]],
    delta_metadata_by_metric: dict[str, dict[str, Any]],
    output_dir: Path | str,
    metrics: tuple[str, ...] = MULTIMETRIC_METRICS,
    boxplot_subdir: str = "boxplots",
) -> None:
    """Write individual multimetrics delta boxplots from streaming payloads."""
    plt, _mdates = require_matplotlib()
    apply_ieee_plot_style()
    resolved_output_dir = Path(output_dir) / boxplot_subdir
    for metric in metrics:
        grouped = _payload_groups(delta_values_by_metric, metric)
        groups = order_humans_first(grouped)
        comparisons = _comparison_payloads(groups, grouped)
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        y_axis_limits = _draw_boxplot(ax, grouped, groups, metric)
        _add_comparison_annotations(ax, groups, comparisons)
        ax.xaxis.set_label_coords(0.5, -0.2)
        fig.subplots_adjust(bottom=0.36)
        stem = f"{metric_stem(metric)}_delta_boxplot_by_cohort"
        save_figure(fig, resolved_output_dir, stem)
        write_plot_data(
            resolved_output_dir,
            stem,
            {
                "plot": stem,
                "plot_type": "boxplot",
                "metric": metric,
                "source_metric": metric,
                "group_field": "cohort",
                "x_axis": "Cohort",
                "y_axis": metric_label(metric, delta=True),
                "plot_value_scale": _plot_value_scale(metric),
                "percentile_capped_y_axis": y_axis_limits,
                "groups": {
                    group: _summary_payload(grouped.get(group, []))
                    for group in groups
                },
                "multimetric_metadata": delta_metadata_by_metric.get(metric, {}),
                "human_baseline_tests": comparisons,
                "comparison_label_p_threshold": MULTIMETRIC_BOXPLOT_P_THRESHOLD,
                "figure": {
                    "width_inches": 3.5,
                    "height_inches": 2.0,
                    "bottom": 0.36,
                    "comparison_stat_label_size": 5.5,
                    "comparison_stat_y": -0.11,
                    "x_axis_label_y": -0.2,
                },
            },
        )
        plt.close(fig)
        del grouped, groups, comparisons


def _payload_groups(
    delta_values_by_metric: dict[str, dict[str, list[float]]],
    metric: str,
) -> dict[str, list[float]]:
    return {
        str(cohort): [
            float(value) * _plot_value_scale(metric)
            for value in values
        ]
        for cohort, values in delta_values_by_metric.get(metric, {}).items()
        if values
    }


def _payload_scoped_groups(
    delta_values_by_metric: dict[str, dict[str, list[float]]],
    metric: str,
) -> dict[str, list[float]]:
    return {
        f"cohort:{cohort}": values
        for cohort, values in _payload_groups(delta_values_by_metric, metric).items()
    }


def write_multimetric_quality_boxplot_grid_from_payload(
    *,
    delta_values_by_metric: dict[str, dict[str, list[float]]],
    output_dir: Path | str,
    metrics: tuple[str, ...] = MULTIMETRIC_QUALITY_GRID_METRICS,
    stem: str = "maintainability_quality_metrics_normalized_boxplots_by_cohort",
    layout: str = "2x3",
    figsize: tuple[float, float] = (7.16, 3.5),
    ncols: int = 3,
    center_incomplete_final_row: bool = False,
    metric_axis_labels: dict[str, str] | None = None,
    wspace: float = 0.45,
    tick_label_size: float = 6.5,
    tick_label_rotation: float = 0.0,
    comparison_stat_label_size: float = 5.0,
    comparison_stat_y: float = -0.11,
) -> None:
    """Write the maintainability-style multimetric quality boxplot grid."""
    apply_ieee_plot_style()
    resolved_output_dir = Path(output_dir) / "boxplots-grid" / "multimetric"
    aliased_metrics = tuple(
        MULTIMETRIC_GRID_METRIC_ALIASES[metric]
        for metric in metrics
        if metric in MULTIMETRIC_GRID_METRIC_ALIASES
    )
    alias_by_multimetric_metric = {
        MULTIMETRIC_GRID_METRIC_ALIASES[metric]: metric
        for metric in metrics
        if metric in MULTIMETRIC_GRID_METRIC_ALIASES
    }
    metric_values_by_scope: dict[str, dict[str, list[float]]] = {}
    group_scopes: set[str] = set()
    for alias_metric, multimetric_metric in alias_by_multimetric_metric.items():
        scoped_values = _payload_scoped_groups(
            delta_values_by_metric,
            multimetric_metric,
        )
        metric_values_by_scope[alias_metric] = scoped_values
        group_scopes.update(scoped_values)
    metric_values_by_scope["SmellCount"] = {
        scope: [0.0]
        for scope in group_scopes
    }
    payload = {"metric_values_by_scope": metric_values_by_scope}
    _plot_delta_boxplot_grid(
        _MaintainabilityPlotPayloadConnection(payload),
        resolved_output_dir,
        metrics=aliased_metrics,
        stem=stem,
        layout=layout,
        figsize=figsize,
        tick_label_size=tick_label_size,
        y_tick_label_size=6.5,
        axis_label_size=7.5,
        supxlabel_font_size=7.5,
        comparison_stat_label_size=comparison_stat_label_size,
        comparison_stat_y=comparison_stat_y,
        tick_label_rotation=tick_label_rotation,
        left=0.01,
        right=0.99,
        bottom=0.28,
        top=0.947,
        wspace=wspace,
        tight_layout_w_pad=wspace,
        hspace=0.45,
        ncols=ncols,
        supxlabel_y=-0.03,
        y_labelpad=0.75,
        y_tick_labelpad=0.5,
        center_incomplete_final_row=center_incomplete_final_row,
        metric_axis_labels={
            alias_metric: (
                metric_axis_labels.get(multimetric_metric)
                if metric_axis_labels and multimetric_metric in metric_axis_labels
                else metric_label(multimetric_metric, delta=True)
            )
            for alias_metric, multimetric_metric in alias_by_multimetric_metric.items()
        },
    )
