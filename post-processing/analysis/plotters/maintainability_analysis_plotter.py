"""Plots for maintainability analysis outputs.

Maintainability plots cover tool coverage, Mantyla smell-category composition,
before/after metric deltas, and paired before/after views. Payload adapters let
the same plotting code operate on streaming outputs without requiring DuckDB.
"""

from __future__ import annotations

import sys
import re
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
from analysis_runtime_utility import release_process_memory
from maintainability_analysis_utility import (
    MAINTAINABILITY_DELTA_METRICS,
    MANTYLA_CATEGORIES,
    TOOL_STATUS_BUCKETS,
)
from plotting_utility import (
    add_stacked_bar_percentage_callouts,
    add_human_median_baseline,
    add_xtick_count_sublabels,
    add_violin_underlay,
    apply_percentile_capped_y_axis,
    apply_ieee_plot_style,
    cohort_color_map,
    display_group_label,
    display_group_labels,
    ieee_boxplot_kwargs,
    order_labels_by_average_percentage,
    order_humans_first,
    ranked_stacked_bar_colors,
    require_matplotlib,
    remove_plot_outputs,
    save_figure,
    apply_symlog_y_axis_if_range_exceeds,
    stacked_bar_visual_metadata,
    stacked_bar_visual_percentages,
    style_ieee_boxplot,
    write_plot_data,
)

MANTYLA_PLOT_CATEGORIES = tuple(
    category for category in MANTYLA_CATEGORIES if category != "unmapped"
)
MANTYLA_PLOT_CATEGORY_NAMES = {
    "bloaters": "Bloaters",
    "object_orientation_abusers": "Object-Oriented Abusers",
    "change_preventers": "Change Preventers",
    "dispensables": "Dispensables",
    "encapsulators": "Encapsulators",
    "couplers": "Couplers",
    "others": "Others",
    "unmapped": "Unmapped",
}
MANTYLA_PLOT_COLORS = {
    "bloaters": "#C9C6C6",
    "object_orientation_abusers": "#0072B2",
    "change_preventers": "#D55E00",
    "dispensables": "#009E73",
    "encapsulators": "#E69F00",
    "couplers": "#CC79A7",
    "others": "#F0E442",
    "unmapped": "#D9D9D9",
}
STACKED_BAR_PERCENTAGE_FONT_SIZE = 6.0
STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET = -0.2
MANTYLA_STACKED_BAR_PERCENTAGE_FONT_SIZE = 5.5
MANTYLA_LEGEND_FONT_SIZE = 6.5
MANTYLA_LEGEND_ROW_COUNTS = (3, 4)
MANTYLA_LEGEND_COLUMNS = max(MANTYLA_LEGEND_ROW_COUNTS)
MANTYLA_LEGEND_CENTER_X = 0.43
MANTYLA_PERCENTAGE_LABEL_Y_OFFSET = -0.6
MANTYLA_PERCENTAGE_LABEL_MINIMUM = 4.5
MANTYLA_PERCENTAGE_CALLOUT_MAXIMUM = 4.5
MANTYLA_MINIMUM_VISIBLE_SEGMENT_HEIGHT = 1.0
MANTYLA_COUNT_LABEL_FONT_SIZE = 5.0
MANTYLA_PERCENTAGE_CALLOUT_FONT_SIZE = MANTYLA_COUNT_LABEL_FONT_SIZE - 0.25
MANTYLA_COUNT_LABEL_Y = -0.15
MANTYLA_WITH_PR_COUNT_X_LABELPAD = 15
MANTYLA_DISTRIBUTION_FIGSIZE = (3.5, 2.0)
MAINTAINABILITY_BOXPLOT_AXIS_LOWER_BOUND = 0.0
MAINTAINABILITY_BOXPLOT_AXIS_LOWER_PADDING = 0.1
MAINTAINABILITY_BOXPLOT_MINIMUM_UPPER_PADDING = 0.1
MAINTAINABILITY_BOXPLOT_P_THRESHOLD = 0.001
GRID_BOXPLOT_X_MARGIN = 0.1
GRID_BOXPLOT_DEFAULT_HALF_WIDTH = 0.25
MAINTAINABILITY_BOXPLOT_SYMLOG_RANGE_THRESHOLD = 0.0
PAIRED_PLOT_BEFORE_COLOR = "#6F6F6F"
PAIRED_PLOT_AFTER_COLOR = "#0072B2"
GROUP_FIELD_SQL = {
    "cohort": "cohort",
    "authorship_group": "authorship_group",
}
METRIC_SQL = {
    "SmellsDelta": '"SmellsDelta"',
    "SmellCount": '"SmellCount"',
    "SmellDensity": '"SmellDensity"',
    "MultimetricFanOut": '"MultimetricFanOut"',
    "MultimetricFanOutPerKLOC": '"MultimetricFanOutPerKLOC"',
    **{metric: f'"{metric}"' for metric in MAINTAINABILITY_DELTA_METRICS},
}
BEFORE_AFTER_COLUMNS = {
    "MI": ('"MI_Before"', '"MI_Post"'),
    "CC": ('"CC_Before"', '"CC_Post"'),
    "HV": ('"HV_Before"', '"HV_Post"'),
    "CCDensity": ('"CCDensity_Before"', '"CCDensity_Post"'),
    "HVDensity": ('"HVDensity_Before"', '"HVDensity_Post"'),
    "DuplicationDensity": (
        '"DuplicationDensity_Before"',
        '"DuplicationDensity_Post"',
    ),
    "CommentDensity": ('"CommentDensity_Before"', '"CommentDensity_Post"'),
    "CodeSmellDensity": ('"CodeSmellDensity_Before"', '"CodeSmellDensity_Post"'),
    "SmellCount": ('("SmellCount" - "SmellsDelta")', '"SmellCount"'),
}
BEFORE_AFTER_AXIS_LABELS = {
    "MI": "Maintainability Index",
    "CC": "Cyclomatic Complexity",
    "HV": "Halstead Volume",
    "CCDensity": "Cyclomatic Complexity per KLOC",
    "HVDensity": "Halstead Volume per KLOC",
    "DuplicationDensity": "Duplicated Lines Density (%)",
    "CommentDensity": "Comment Lines Density (%)",
    "CodeSmellDensity": "Smells per KLOC",
    "SmellCount": "Code Smell Count",
}
TOOL_STATUS_LABELS = {
    "success": "Success",
    "unsupported": "Unsupported",
    "failed": "Failed",
    "missing": "Missing",
    "other": "Other",
}
TOOL_STATUS_COLORS = {
    "success": "#009E73",
    "unsupported": "#BDBDBD",
    "failed": "#D55E00",
    "missing": "#F0F0F0",
    "other": "#E69F00",
}
METRIC_STEMS = {
    "SmellsDelta": "smell_count_delta",
    "SmellCount": "smell_count",
    "SmellDensity": "smell_density",
    "SmellsFixed": "smells_fixed",
    "MI": "mi_delta",
    "CC": "cc_delta",
    "HV": "hv_delta",
    "CCDensity": "cc_density_delta",
    "HVDensity": "hv_density_delta",
    "DuplicationDensity": "duplication_density_delta",
    "CommentDensity": "comment_density_delta",
    "NLOC": "nloc_delta",
    "CodeSmellDensityDelta": "code_smell_density_delta",
    "CodeSmellIntroRate": "code_smell_intro_rate",
    "CodeSmellFixRate": "code_smell_fix_rate",
}
METRIC_AXIS_LABELS = {
    "SmellsDelta": "\u0394 Smell Count",
    "SmellCount": "Smell Count",
    "SmellDensity": "Smells per KLOC",
    "MI": "\u0394 Maintainability Index",
    "CC": "\u0394 Cyclomatic Complexity",
    "HV": "\u0394 Halstead Volume",
    "CCDensity": "\u0394 Cyclomatic Complexity per KLOC",
    "HVDensity": "\u0394 Halstead Volume per KLOC",
    "DuplicationDensity": "\u0394 Duplicated Lines Density (%)",
    "CommentDensity": "\u0394 Comment Lines Density (%)",
    "NLOC": "\u0394 NLOC",
    "CodeSmellDensityDelta": "\u0394 Smells per KLOC",
    "CodeSmellIntroRate": "Smell Intro Rate (%)",
    "CodeSmellFixRate": "Smell Fix Rate (%)",
}
PERCENTAGE_RATE_METRICS = {
    "CodeSmellIntroRate",
    "CodeSmellFixRate",
}
TEN_THOUSAND_SCALE_METRICS: set[str] = set()
TEN_TO_MINUS_EIGHT_SCALE_METRICS: set[str] = set()
MILLION_SCALE_METRICS: set[str] = set()
PERCENTAGE_RATE_AXIS_MINIMUM = 0.0
PERCENTAGE_RATE_AXIS_MAXIMUM = 100.0
MAINTAINABILITY_QUALITY_GRID_METRICS = (
    "CodeSmellDensityDelta",
    "DuplicationDensity",
    "CommentDensity",
    "CC",
    "HV",
    "MI",
)
MAINTAINABILITY_QUALITY_NORMALIZED_GRID_METRICS = (
    "CodeSmellDensityDelta",
    "DuplicationDensity",
    "CommentDensity",
    "CCDensity",
    "HVDensity",
    "MI",
)
MAINTAINABILITY_SMELL_COUNT_QUALITY_GRID_METRICS = (
    "SmellsDelta",
    *MAINTAINABILITY_QUALITY_GRID_METRICS,
)
MAINTAINABILITY_SMELL_GRID_METRICS = (
    "SmellsDelta",
    "CodeSmellFixRate",
    "CodeSmellIntroRate",
)
MAINTAINABILITY_PLOT_METRICS = tuple(
    ("SmellsDelta",)
    + tuple(
        metric
        for metric in MAINTAINABILITY_DELTA_METRICS
        if metric not in {"NLOC", "CodeSmellIntroRate", "CodeSmellFixRate"}
    )
)
MAINTAINABILITY_SIGNED_BOXPLOT_METRICS = {
    "SmellsDelta",
    "MI",
    "CC",
    "HV",
    "CCDensity",
    "HVDensity",
    "DuplicationDensity",
    "CommentDensity",
    "NLOC",
    "CodeSmellDensityDelta",
    "MultimetricFanOutPerKLOC",
}
REMOVED_PLOT_OUTPUT_STEMS = (
    "tool_coverage_by_cohort",
    "tool_coverage_by_authorship",
    "mantyla_distribution_by_authorship",
    "boxplots/maintainability_metrics_boxplots_by_cohort",
    "boxplots-grid/maintainability_metrics_boxplots_by_cohort",
    "boxplots-grid/maintainability_count_density_quality_metrics_boxplots_by_cohort",
    "boxplots-grid/maintainability_smell_metrics_boxplots_by_cohort",
    "boxplots-grid/maintainability_smell_count_quality_metrics_boxplots_by_cohort",
    "boxplots/maintainability_quality_metrics_boxplots_by_cohort",
    "boxplots/maintainability_smell_metrics_boxplots_by_cohort",
    f"boxplots/{METRIC_STEMS['NLOC']}_boxplot_by_cohort",
    f"boxplots/{METRIC_STEMS['NLOC']}_boxplot_by_authorship",
    f"boxplots/{METRIC_STEMS['SmellsFixed']}_boxplot_by_cohort",
    f"boxplots/{METRIC_STEMS['SmellsFixed']}_boxplot_by_authorship",
    "boxplots/smell_fixed_boxplot_by_cohort",
    "boxplots/smell_fixed_boxplot_by_authorship",
    f"boxplots/{METRIC_STEMS['CodeSmellIntroRate']}_boxplot_by_cohort",
    f"boxplots/{METRIC_STEMS['CodeSmellIntroRate']}_boxplot_by_authorship",
    f"boxplots/{METRIC_STEMS['CodeSmellFixRate']}_boxplot_by_cohort",
    f"boxplots/{METRIC_STEMS['CodeSmellFixRate']}_boxplot_by_authorship",
    f"violin_plots/{METRIC_STEMS['NLOC']}_violin_by_cohort",
    f"violin_plots/{METRIC_STEMS['NLOC']}_violin_by_authorship",
    f"violin_plots/{METRIC_STEMS['SmellsFixed']}_violin_by_cohort",
    f"violin_plots/{METRIC_STEMS['SmellsFixed']}_violin_by_authorship",
    f"violin_plots/{METRIC_STEMS['CodeSmellIntroRate']}_violin_by_cohort",
    f"violin_plots/{METRIC_STEMS['CodeSmellIntroRate']}_violin_by_authorship",
    f"violin_plots/{METRIC_STEMS['CodeSmellFixRate']}_violin_by_cohort",
    f"violin_plots/{METRIC_STEMS['CodeSmellFixRate']}_violin_by_authorship",
    *(
        f"boxplots/{METRIC_STEMS[metric]}_boxplot_by_authorship"
        for metric in MAINTAINABILITY_PLOT_METRICS
    ),
    *(
        f"violin_plots/{METRIC_STEMS[metric]}_violin_by_cohort"
        for metric in MAINTAINABILITY_PLOT_METRICS
    ),
    *(
        f"violin_plots/{METRIC_STEMS[metric]}_violin_by_authorship"
        for metric in MAINTAINABILITY_PLOT_METRICS
    ),
    *(
        f"paired_plots/{_metric_stem}_paired_by_authorship"
        for _metric_stem in (
            "mi",
            "cc",
            "hv",
            "duplication_density",
            "comment_density",
            "code_smell_density",
            "smell_count",
        )
    ),
    *(
        f"paired_plots/{_metric_stem}_paired_by_cohort"
        for _metric_stem in (
            "mi",
            "cc",
            "hv",
            "duplication_density",
            "comment_density",
            "code_smell_density",
            "smell_count",
        )
    ),
)


PLOT_OUTPUT_STEMS = (
    "mantyla_distribution_by_cohort",
    *(
        f"boxplots/{METRIC_STEMS[metric]}_boxplot_by_cohort"
        for metric in MAINTAINABILITY_PLOT_METRICS
    ),
    "boxplots-grid/maintainability_quality_metrics_boxplots_by_cohort",
    "boxplots-grid/maintainability_quality_metrics_normalized_boxplots_by_cohort",
)


def _format_plot_stat(value: float | None) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.2f}"


def _scale_from_y_axis_limits(y_axis_limits: dict[str, object] | None) -> str:
    if y_axis_limits is None:
        return "linear"
    return str(y_axis_limits.get("scale") or "linear")


def _format_effect_size_compact(value: float | None) -> str:
    if value is None:
        return "\u03b4 = NA"
    return f"\u03b4 = {value:.2f}"


def _format_effect_size_with_significance_star(
    value: float | None,
    adjusted_p_value: float | None,
    *,
    threshold: float = MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
) -> str:
    label = _format_effect_size_compact(value)
    if adjusted_p_value is None:
        return label
    try:
        is_significant = float(adjusted_p_value) <= float(threshold)
    except (TypeError, ValueError):
        is_significant = False
    return f"{label}*" if is_significant else label


def _format_adjusted_p_threshold(value: float | None) -> str:
    if value is None:
        return "p = NA"
    if float(value) <= MAINTAINABILITY_BOXPLOT_P_THRESHOLD:
        return f"p \u2264 {MAINTAINABILITY_BOXPLOT_P_THRESHOLD:.3f}"
    return f"p > {MAINTAINABILITY_BOXPLOT_P_THRESHOLD:.3f}"


def _format_adjusted_p_threshold_compact(value: float | None) -> str:
    if value is None:
        return "p = NA"
    if float(value) <= MAINTAINABILITY_BOXPLOT_P_THRESHOLD:
        return f"p \u2264 {MAINTAINABILITY_BOXPLOT_P_THRESHOLD:.3f}"
    return f"p > {MAINTAINABILITY_BOXPLOT_P_THRESHOLD:.3f}"


def _human_baseline_group(groups: list[str]) -> str | None:
    for group in groups:
        if group.strip().casefold() in {"human", "humans"}:
            return group
    return None


def _summary_payload(values: list[float]) -> dict[str, object]:
    return {
        **numeric_distribution_summary(values),
        "n": len(values),
    }


def _test_payload(
    *,
    first_group: str,
    second_group: str | None,
    first_values: list[float],
    second_values: list[float],
) -> dict[str, object]:
    if second_group is None:
        test = mann_whitney_u_test([], [])
    else:
        test = mann_whitney_u_test(first_values, second_values)
        add_cliffs_delta_ci(test, first_values, second_values)
    test["first_group"] = first_group
    test["second_group"] = second_group
    return test


def _comparison_payloads(
    groups: list[str],
    grouped: dict[str, list[float]],
    group_field: str,
) -> dict[str, object]:
    baseline_group = _human_baseline_group(groups)
    comparisons: dict[str, object] = {}
    for group in groups:
        if group == baseline_group:
            comparisons[group] = {
                "baseline": True,
                "mann_whitney_u": _test_payload(
                    first_group=group,
                    second_group=None,
                    first_values=[],
                    second_values=[],
                ),
            }
            continue
        if (
            group_field == "authorship_group"
            and group.casefold() not in {"agent", "agents"}
        ):
            baseline_for_group = None
        else:
            baseline_for_group = baseline_group
        comparisons[group] = {
            "baseline": False,
            "mann_whitney_u": _test_payload(
                first_group=group,
                second_group=baseline_for_group,
                first_values=grouped.get(group, []),
                second_values=(
                    grouped.get(baseline_for_group, [])
                    if baseline_for_group is not None
                    else []
                ),
            ),
        }
    return apply_fdr_correction(
        {
            "baseline_group": baseline_group,
            "comparisons": comparisons,
        }
    )


def _boxplot_comparison_stat_labels(
    groups: list[str],
    comparison_payload: dict[str, object] | None,
    *,
    label_style: str = "delta_p",
    significance_threshold: float = MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
) -> list[str]:
    comparisons = {}
    if comparison_payload:
        comparisons = comparison_payload.get("comparisons", {}) or {}
    labels: list[str] = []
    for group in groups:
        comparison = comparisons.get(group, {}) if isinstance(comparisons, dict) else {}
        if comparison.get("baseline"):
            labels.append("-")
            continue
        test = comparison.get("mann_whitney_u", {}) if isinstance(comparison, dict) else {}
        if label_style == "delta_star":
            labels.append(
                _format_effect_size_with_significance_star(
                    test.get("cliffs_delta"),
                    test.get("adjusted_p_value"),
                    threshold=significance_threshold,
                )
            )
            continue
        labels.append(
            f"{_format_effect_size_compact(test.get('cliffs_delta'))}\n"
            f"{_format_adjusted_p_threshold_compact(test.get('adjusted_p_value'))}"
        )
    return labels


def _boxplot_comparison_tick_labels(
    groups: list[str],
    comparison_payload: dict[str, object] | None,
) -> list[str]:
    comparisons = {}
    if comparison_payload:
        comparisons = comparison_payload.get("comparisons", {}) or {}
    labels: list[str] = []
    for group in groups:
        display_label = display_group_label(group)
        comparison = comparisons.get(group, {}) if isinstance(comparisons, dict) else {}
        if comparison.get("baseline"):
            labels.append(f"{display_label}\n-")
            continue
        test = comparison.get("mann_whitney_u", {}) if isinstance(comparison, dict) else {}
        labels.append(
            f"{display_label}\n"
            f"{_format_effect_size_compact(test.get('cliffs_delta'))}\n"
            f"{_format_adjusted_p_threshold_compact(test.get('adjusted_p_value'))}"
        )
    return labels


def _add_boxplot_comparison_annotations(
    ax,
    groups: list[str],
    comparison_payload: dict[str, object] | None,
    *,
    font_size: float = 6.0,
    y_offset: float = -0.1,
    label_style: str = "delta_p",
    significance_threshold: float = MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
) -> None:
    if not groups:
        return
    for index, label in enumerate(
        _boxplot_comparison_stat_labels(
            groups,
            comparison_payload,
            label_style=label_style,
            significance_threshold=significance_threshold,
        ),
        start=1,
    ):
        ax.annotate(
            label,
            xy=(index, y_offset),
            xycoords=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=font_size,
            linespacing=0.85,
            clip_on=False,
            zorder=50,
        )


def _has_negative_values(grouped: dict[str, list[float]]) -> bool:
    return any(value < 0.0 for values in grouped.values() for value in values)


def _max_group_value(grouped: dict[str, list[float]]) -> float:
    return max(
        (value for values in grouped.values() for value in values),
        default=0.0,
    )


def _add_zero_reference_line_if_needed(ax, grouped: dict[str, list[float]]) -> None:
    if not _has_negative_values(grouped):
        return
    ax.axhline(
        0.0,
        color="0.25",
        linestyle=(0, (3, 2)),
        linewidth=0.7,
        alpha=0.35,
        zorder=0,
    )


def _set_nonnegative_y_axis_if_possible(ax, grouped: dict[str, list[float]]) -> None:
    if _has_negative_values(grouped):
        return
    max_value = _max_group_value(grouped)
    ax.set_ylim(bottom=0.0, top=max(1.0, max_value * 1.05))


def _set_y_axis_for_values(ax, values_by_group: list[list[float]]) -> None:
    values = [value for group in values_by_group for value in group]
    if not values:
        return
    minimum = min(values)
    maximum = max(values)
    if minimum >= 0:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=max(0.0, bottom), top=top)
    if minimum < 0 < maximum:
        ax.axhline(
            0.0,
            color="0.35",
            linestyle=(0, (3, 2)),
            linewidth=0.6,
            alpha=0.55,
            zorder=1,
        )


def _flatten_numeric_groups(values_by_group) -> list[float]:
    if isinstance(values_by_group, dict):
        groups = values_by_group.values()
    else:
        groups = values_by_group
    flattened: list[float] = []
    for values in groups:
        for value in values:
            flattened.append(float(value))
    return flattened


def _clamp_percentage_rate_axis(
    ax,
    metric_name: str,
    y_axis_limits: dict[str, object] | None,
) -> dict[str, object] | None:
    if metric_name not in PERCENTAGE_RATE_METRICS:
        return y_axis_limits
    bottom, top = ax.get_ylim()
    visual_lower = max(PERCENTAGE_RATE_AXIS_MINIMUM, float(bottom))
    visual_upper = min(PERCENTAGE_RATE_AXIS_MAXIMUM, float(top))
    if visual_upper <= visual_lower:
        visual_upper = PERCENTAGE_RATE_AXIS_MAXIMUM
    ax.set_ylim(visual_lower, visual_upper)
    payload = {
        "percentage_rate_axis_minimum": PERCENTAGE_RATE_AXIS_MINIMUM,
        "percentage_rate_axis_maximum": PERCENTAGE_RATE_AXIS_MAXIMUM,
        "percentage_rate_axis_clamped": True,
        "visual_lower": visual_lower,
        "visual_upper": visual_upper,
    }
    if y_axis_limits is None:
        return payload
    return {
        **y_axis_limits,
        **payload,
    }


def _apply_boxplot_lower_bound(
    ax,
    metric_name: str,
    values_by_group,
    y_axis_limits: dict[str, object] | None,
) -> dict[str, object] | None:
    if metric_name in PERCENTAGE_RATE_METRICS:
        return _clamp_percentage_rate_axis(ax, metric_name, y_axis_limits)
    if metric_name in MAINTAINABILITY_SIGNED_BOXPLOT_METRICS:
        flattened = _flatten_numeric_groups(values_by_group)
        has_negative_values = any(float(value) < 0.0 for value in flattened)
        current_lower, visible_upper = ax.get_ylim()
        if has_negative_values:
            current_lower = min(float(current_lower), 0.0)
        visual_lower = (
            float(current_lower) - MAINTAINABILITY_BOXPLOT_AXIS_LOWER_PADDING
        )
        ax.set_ylim(bottom=visual_lower, top=float(visible_upper))
        if y_axis_limits is None:
            return None
        below_count = sum(
            1
            for value in flattened
            if float(value) < float(current_lower)
        )
        return {
            **y_axis_limits,
            "lower": float(current_lower),
            "visual_lower_padding": MAINTAINABILITY_BOXPLOT_AXIS_LOWER_PADDING,
            "visual_lower": visual_lower,
            "below_count": below_count,
            "is_clipped": bool(
                below_count or int(y_axis_limits.get("above_count", 0))
            ),
        }
    visible_upper = ax.get_ylim()[1]
    ax.set_ylim(
        bottom=(
            MAINTAINABILITY_BOXPLOT_AXIS_LOWER_BOUND
            - MAINTAINABILITY_BOXPLOT_AXIS_LOWER_PADDING
        ),
        top=max(
            float(visible_upper),
            MAINTAINABILITY_BOXPLOT_AXIS_LOWER_BOUND
            + MAINTAINABILITY_BOXPLOT_MINIMUM_UPPER_PADDING,
        ),
    )
    if y_axis_limits is None:
        return None
    below_count = sum(
        1
        for value in _flatten_numeric_groups(values_by_group)
        if float(value) < MAINTAINABILITY_BOXPLOT_AXIS_LOWER_BOUND
    )
    return {
        **y_axis_limits,
        "lower": MAINTAINABILITY_BOXPLOT_AXIS_LOWER_BOUND,
        "visual_lower_padding": MAINTAINABILITY_BOXPLOT_AXIS_LOWER_PADDING,
        "visual_lower": (
            MAINTAINABILITY_BOXPLOT_AXIS_LOWER_BOUND
            - MAINTAINABILITY_BOXPLOT_AXIS_LOWER_PADDING
        ),
        "below_count": below_count,
        "is_clipped": bool(
            below_count or int(y_axis_limits.get("above_count", 0))
        ),
    }


def _hex_text_color(hex_color: str) -> str:
    text = str(hex_color or "").strip().lstrip("#")
    if len(text) != 6:
        return "black"
    try:
        red = int(text[0:2], 16) / 255.0
        green = int(text[2:4], 16) / 255.0
        blue = int(text[4:6], 16) / 255.0
    except ValueError:
        return "black"
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.55 else "white"


def _group_sql(group_field: str) -> str:
    try:
        return GROUP_FIELD_SQL[group_field]
    except KeyError as exc:
        raise ValueError(f"Unknown maintainability group field: {group_field}") from exc


def _metric_sql(metric_name: str) -> str:
    try:
        return METRIC_SQL[metric_name]
    except KeyError as exc:
        raise ValueError(f"Unknown maintainability metric: {metric_name}") from exc


def _metric_stem(metric_name: str) -> str:
    try:
        return METRIC_STEMS[metric_name]
    except KeyError as exc:
        raise ValueError(f"Unknown maintainability metric: {metric_name}") from exc


def _metric_axis_label(metric_name: str) -> str:
    try:
        return METRIC_AXIS_LABELS[metric_name]
    except KeyError as exc:
        raise ValueError(f"Unknown maintainability metric: {metric_name}") from exc


def _plot_value_scale(metric_name: str) -> float:
    if metric_name in PERCENTAGE_RATE_METRICS:
        return 100.0
    if metric_name in TEN_THOUSAND_SCALE_METRICS:
        return 1.0 / 10_000.0
    if metric_name in TEN_TO_MINUS_EIGHT_SCALE_METRICS:
        return 1.0e-8
    if metric_name in MILLION_SCALE_METRICS:
        return 1.0 / 1_000_000.0
    return 1.0


def _plot_value_unit(metric_name: str) -> str:
    if metric_name in PERCENTAGE_RATE_METRICS:
        return "percentage"
    if metric_name in TEN_THOUSAND_SCALE_METRICS:
        return "ten_thousands"
    if metric_name in TEN_TO_MINUS_EIGHT_SCALE_METRICS:
        return "ten_to_minus_eight"
    if metric_name in MILLION_SCALE_METRICS:
        return "millions"
    return "original_metric_unit"


def _nonempty_sql(column_sql: str) -> str:
    return (
        f"{column_sql} IS NOT NULL "
        f"AND NULLIF(trim(CAST({column_sql} AS VARCHAR)), '') IS NOT NULL"
    )


def _groups_for_field(con, group_field: str) -> list[str]:
    group_column = _group_sql(group_field)
    rows = con.execute(
        f"""
        SELECT DISTINCT {group_column}
        FROM analysis_maintainability_prs
        WHERE {_nonempty_sql(group_column)}
        ORDER BY {group_column}
        """
    ).fetchall()
    return order_humans_first(str(row[0]) for row in rows)


class _PayloadQueryResult:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._rows)

    def fetchone(self) -> tuple[object, ...] | None:
        return self._rows[0] if self._rows else None


class _MaintainabilityPlotPayloadConnection:
    """Adapter for rendering existing maintainability plots from payloads."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def execute(self, sql: str) -> _PayloadQueryResult:
        normalized_sql = " ".join(str(sql).strip().lower().split())
        if "select distinct" in normalized_sql and "from analysis_maintainability_prs" in normalized_sql:
            return _PayloadQueryResult([(group,) for group in self._groups(_field_from_sql(normalized_sql))])
        if "with per_pr as" in normalized_sql and "analysis_mantyla_category_counts" in normalized_sql:
            return _PayloadQueryResult(self._per_pr_mantyla_rows(_field_from_sql(normalized_sql)))
        if "from analysis_mantyla_category_counts" in normalized_sql and "sum(smell_count)" in normalized_sql:
            return _PayloadQueryResult(self._aggregate_mantyla_rows(_field_from_sql(normalized_sql)))
        if "count(*)" in normalized_sql and "from analysis_maintainability_prs" in normalized_sql:
            return _PayloadQueryResult(self._eligible_pr_count_rows(_field_from_sql(normalized_sql)))
        if "from analysis_maintainability_prs" in normalized_sql:
            field = _field_from_sql(normalized_sql)
            metric = _metric_from_sql(normalized_sql)
            if metric is not None:
                return _PayloadQueryResult(self._metric_rows(field, metric))
        raise ValueError(f"Unsupported maintainability plot payload query: {sql}")

    def _groups(self, field: str) -> list[str]:
        scopes = self.payload.get("metric_values_by_scope") or {}
        smell_scopes = scopes.get("SmellCount", {}) if isinstance(scopes, dict) else {}
        prefix = f"{field}:"
        groups = [
            str(scope)[len(prefix) :]
            for scope in smell_scopes
            if str(scope).startswith(prefix)
        ]
        if not groups and isinstance(scopes, dict):
            group_set: set[str] = set()
            for by_scope in scopes.values():
                if not isinstance(by_scope, dict):
                    continue
                for scope in by_scope:
                    scope_text = str(scope)
                    if scope_text.startswith(prefix):
                        group_set.add(scope_text[len(prefix) :])
            groups = sorted(group_set)
        return order_humans_first(groups)

    def _metric_rows(self, field: str, metric: str) -> list[tuple[object, ...]]:
        metric_values = self.payload.get("metric_values_by_scope") or {}
        by_scope = metric_values.get(metric, {}) if isinstance(metric_values, dict) else {}
        rows: list[tuple[object, ...]] = []
        for group in self._groups(field):
            scope = f"{field}:{group}"
            for value in by_scope.get(scope, []):
                rows.append((group, float(value) * _plot_value_scale(metric)))
        return rows

    def _aggregate_mantyla_rows(self, field: str) -> list[tuple[object, ...]]:
        counts = self.payload.get("mantyla_counts_by_scope") or {}
        rows: list[tuple[object, ...]] = []
        for group in self._groups(field):
            scope = f"{field}:{group}"
            group_counts = counts.get(scope, {}) if isinstance(counts, dict) else {}
            for category in MANTYLA_PLOT_CATEGORIES:
                rows.append((group, category, int(group_counts.get(category, 0) or 0)))
        return rows

    def _eligible_pr_count_rows(self, field: str) -> list[tuple[object, ...]]:
        scopes = self.payload.get("metric_values_by_scope") or {}
        smell_counts = scopes.get("SmellCount", {}) if isinstance(scopes, dict) else {}
        rows: list[tuple[object, ...]] = []
        for group in self._groups(field):
            scope = f"{field}:{group}"
            values = smell_counts.get(scope, []) if isinstance(smell_counts, dict) else []
            rows.append((group, len(values)))
        return rows

    def _per_pr_mantyla_rows(self, field: str) -> list[tuple[object, ...]]:
        counts = self.payload.get("per_pr_mantyla_counts_by_scope") or {}
        rows: list[tuple[object, ...]] = []
        for group in self._groups(field):
            scope = f"{field}:{group}"
            by_category = counts.get(scope, {}) if isinstance(counts, dict) else {}
            max_count = max(
                (len(by_category.get(category, [])) for category in MANTYLA_PLOT_CATEGORIES),
                default=0,
            )
            for index in range(max_count):
                rows.append(
                    tuple(
                        [group]
                        + [
                            float(_list_value(by_category.get(category, []), index))
                            for category in MANTYLA_PLOT_CATEGORIES
                        ]
                    )
                )
        return rows


def _field_from_sql(sql: str) -> str:
    if "authorship_group" in sql:
        return "authorship_group"
    return "cohort"


def _metric_from_sql(sql: str) -> str | None:
    selected_columns_match = re.search(
        r"\bselect\s+.+?,\s+(.*?)\s+from\s+analysis_maintainability_prs\b",
        sql,
    )
    selected_metric_sql = selected_columns_match.group(1).strip() if selected_columns_match else sql
    selected_metric_name = selected_metric_sql.strip().strip('"')
    for metric in tuple(METRIC_SQL.keys()) + tuple(MAINTAINABILITY_PLOT_METRICS):
        if metric.lower() == selected_metric_name:
            return metric
    return None


def _list_value(values: object, index: int) -> float:
    if not isinstance(values, list) or index >= len(values):
        return 0.0
    try:
        return float(values[index] or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _group_values(
    con,
    group_field: str,
    metric_name: str,
) -> dict[str, list[float]]:
    group_column = _group_sql(group_field)
    metric_column = _metric_sql(metric_name)
    value_scale = _plot_value_scale(metric_name)
    grouped: dict[str, list[float]] = {}
    rows = con.execute(
        f"""
        SELECT {group_column}, {metric_column}
        FROM analysis_maintainability_prs
        WHERE {_nonempty_sql(group_column)}
          AND {metric_column} IS NOT NULL
        ORDER BY {group_column}, {metric_column}
        """
    ).fetchall()
    for group, value in rows:
        grouped.setdefault(str(group), []).append(float(value) * value_scale)
    return {group: grouped[group] for group in order_humans_first(grouped)}


def _nonempty_groups(
    con,
    group_field: str,
    metric_name: str,
) -> dict[str, list[float]]:
    return {
        group: values
        for group, values in _group_values(con, group_field, metric_name).items()
        if values
    }


def _mantyla_category_count_groups(
    con,
    group_field: str,
) -> dict[str, dict[str, list[float]]]:
    group_column = _group_sql(group_field)
    category_selects = ",\n".join(
        f"""
                COALESCE(
                    SUM(
                        CASE
                            WHEN counts.mantyla_category = '{category}'
                            THEN counts.smell_count
                            ELSE 0
                        END
                    ),
                    0
                ) AS {category}_count"""
        for category in MANTYLA_PLOT_CATEGORIES
    )
    rows = con.execute(
        f"""
        WITH per_pr AS (
            SELECT
                prs.analysis_row_id,
                prs.{group_column} AS group_key,
                {category_selects}
            FROM analysis_maintainability_prs AS prs
            LEFT JOIN analysis_mantyla_category_counts AS counts
                ON prs.analysis_row_id = counts.analysis_row_id
            WHERE {_nonempty_sql(f"prs.{group_column}")}
            GROUP BY prs.analysis_row_id, group_key
        )
        SELECT
            group_key,
            bloaters_count,
            object_orientation_abusers_count,
            change_preventers_count,
            dispensables_count,
            encapsulators_count,
            couplers_count,
            others_count
        FROM per_pr
        ORDER BY group_key, analysis_row_id
        """
    ).fetchall()
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        group = str(row[0])
        grouped.setdefault(
            group,
            {category: [] for category in MANTYLA_PLOT_CATEGORIES},
        )
        for category, value in zip(MANTYLA_PLOT_CATEGORIES, row[1:]):
            grouped[group][category].append(float(value or 0.0))
    return {group: grouped[group] for group in order_humans_first(grouped)}


def _eligible_pr_counts_by_group(con, group_field: str) -> dict[str, int]:
    group_column = _group_sql(group_field)
    rows = con.execute(
        f"""
        SELECT {group_column} AS group_key, COUNT(*) AS pull_request_count
        FROM analysis_maintainability_prs
        WHERE {_nonempty_sql(group_column)}
        GROUP BY group_key
        ORDER BY group_key
        """
    ).fetchall()
    return {
        str(group): int(count or 0)
        for group, count in rows
        if group is not None
    }


def _coverage_counts_by_group(con, group_field: str) -> dict[str, dict[str, int]]:
    group_column = _group_sql(group_field)
    rows = con.execute(
        f"""
        SELECT
            {group_column} AS group_key,
            tool_status,
            COALESCE(SUM(pull_request_count), 0) AS pull_request_count
        FROM analysis_maintainability_tool_coverage
        WHERE {_nonempty_sql(group_column)}
        GROUP BY group_key, tool_status
        ORDER BY group_key, tool_status
        """
    ).fetchall()
    grouped: dict[str, dict[str, int]] = {}
    for group, status, count in rows:
        group_key = str(group)
        status_key = str(status or "other")
        if status_key not in TOOL_STATUS_BUCKETS:
            status_key = "other"
        grouped.setdefault(
            group_key,
            {bucket: 0 for bucket in TOOL_STATUS_BUCKETS},
        )
        grouped[group_key][status_key] += int(count or 0)
    return {group: grouped[group] for group in order_humans_first(grouped)}


def _mantyla_percentages_by_group(
    counts: dict[str, dict[str, int]],
    groups: list[str],
) -> dict[str, dict[str, float]]:
    percentages: dict[str, dict[str, float]] = {}
    for group in groups:
        total = sum(counts[group].values())
        percentages[group] = {
            category: 100.0 * counts[group][category] / total if total else 0.0
            for category in MANTYLA_PLOT_CATEGORIES
        }
    return percentages


def _add_centered_mantyla_legend_rows(
    ax,
    *,
    labels: list[str],
    colors: list[str],
) -> None:
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
    from matplotlib.patches import Rectangle

    def legend_item(label: str, color: str):
        handle = DrawingArea(6.0, 5.0, 0.0, 0.0)
        handle.add_artist(
            Rectangle(
                (0.0, 1.0),
                5.0,
                3.0,
                facecolor=color,
                edgecolor="black",
                linewidth=0.25,
            )
        )
        text = TextArea(
            label,
            textprops={
                "fontsize": MANTYLA_LEGEND_FONT_SIZE,
                "va": "center",
            },
        )
        return HPacker(
            children=[handle, text],
            align="center",
            pad=0.0,
            sep=2.0,
        )

    rows: list[list[tuple[str, str]]] = []
    start = 0
    for row_count in MANTYLA_LEGEND_ROW_COUNTS:
        if start >= len(labels):
            break
        rows.append(list(zip(labels[start : start + row_count], colors[start : start + row_count])))
        start += row_count
    if start < len(labels):
        rows.extend(
            list(zip(
                labels[index : index + MANTYLA_LEGEND_COLUMNS],
                colors[index : index + MANTYLA_LEGEND_COLUMNS],
            ))
            for index in range(start, len(labels), MANTYLA_LEGEND_COLUMNS)
        )
    row_boxes = [
        HPacker(
            children=[
                legend_item(label, color)
                for label, color in row
            ],
            align="center",
            pad=0.0,
            sep=6.0,
        )
        for row in rows
    ]
    legend_box = VPacker(
        children=row_boxes,
        align="center",
        pad=0.0,
        sep=1.0,
    )
    ax.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=legend_box,
            pad=0.0,
            borderpad=0.0,
            frameon=False,
            bbox_to_anchor=(MANTYLA_LEGEND_CENTER_X, 1.03),
            bbox_transform=ax.transAxes,
        )
    )


def _mantyla_legend_label_rows(labels: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    start = 0
    for row_count in MANTYLA_LEGEND_ROW_COUNTS:
        if start >= len(labels):
            break
        rows.append(labels[start : start + row_count])
        start += row_count
    if start < len(labels):
        rows.extend(
            labels[index : index + MANTYLA_LEGEND_COLUMNS]
            for index in range(start, len(labels), MANTYLA_LEGEND_COLUMNS)
        )
    return rows


def _plot_tool_coverage(
    con,
    group_field: str,
    output_dir: Path,
    stem: str,
) -> None:
    plt, _mdates = require_matplotlib()
    counts = _coverage_counts_by_group(con, group_field)
    groups = order_humans_first(counts)
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    x_values = list(range(len(groups)))
    bottoms = [0.0 for _group in groups]
    true_by_group: dict[str, dict[str, float]] = {}
    visual_by_group: dict[str, dict[str, float]] = {}
    for group in groups:
        total = sum(counts[group].values())
        percentages = [
            100.0 * counts[group].get(status, 0) / total if total else 0.0
            for status in TOOL_STATUS_BUCKETS
        ]
        visual_percentages = stacked_bar_visual_percentages(percentages)
        true_by_group[group] = {
            status: percentages[index]
            for index, status in enumerate(TOOL_STATUS_BUCKETS)
        }
        visual_by_group[group] = {
            status: visual_percentages[index]
            for index, status in enumerate(TOOL_STATUS_BUCKETS)
        }
    for status in TOOL_STATUS_BUCKETS:
        percentages = [
            true_by_group[group].get(status, 0.0)
            for group in groups
        ]
        visual_percentages = [
            visual_by_group[group].get(status, 0.0)
            for group in groups
        ]
        previous_bottoms = list(bottoms)
        ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=TOOL_STATUS_LABELS[status],
            color=TOOL_STATUS_COLORS[status],
            edgecolor="black",
            linewidth=0.25,
        )
        for x_value, bottom, percentage, visual_percentage in zip(
            x_values,
            previous_bottoms,
            percentages,
            visual_percentages,
        ):
            if percentage <= 0.0:
                continue
            ax.text(
                x_value,
                bottom
                + visual_percentage / 2.0
                + STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=STACKED_BAR_PERCENTAGE_FONT_SIZE,
                color=_hex_text_color(TOOL_STATUS_COLORS[status]),
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    ax.set_xlabel("Cohort" if group_field == "cohort" else "Authorship Group")
    ax.set_ylabel("Percentage of PRs (%)")
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(groups), rotation=0, ha="center")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(groups) - 0.5))
    ax.legend(
        frameon=False,
        ncol=len(TOOL_STATUS_BUCKETS),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.35,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(left=0.01, right=0.99)
    save_figure(fig, output_dir, stem, use_tight_layout=False)
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "group_field": group_field,
            "x_axis": "Cohort" if group_field == "cohort" else "Authorship Group",
            "y_axis": "Percentage of PRs (%)",
            **stacked_bar_visual_metadata(),
            "status_order": list(TOOL_STATUS_BUCKETS),
            "groups": {
                group: {
                    "total_pull_request_count": int(sum(counts[group].values())),
                    "statuses": {
                        status: {
                            "pull_request_count": int(counts[group].get(status, 0)),
                            "percentage": (
                                100.0
                                * int(counts[group].get(status, 0))
                                / sum(counts[group].values())
                                if sum(counts[group].values())
                                else 0.0
                            ),
                            "visual_percentage": visual_by_group[group].get(
                                status,
                                0.0,
                            ),
                        }
                        for status in TOOL_STATUS_BUCKETS
                    },
                }
                for group in groups
            },
        },
    )
    plt.close(fig)
    del grouped, groups, human_baseline_tests


def _plot_mantyla_distribution(
    con,
    group_field: str,
    output_dir: Path,
    stem: str,
) -> None:
    plt, _mdates = require_matplotlib()
    group_column = _group_sql(group_field)
    groups = _groups_for_field(con, group_field)
    counts = {
        group: {category: 0 for category in MANTYLA_PLOT_CATEGORIES}
        for group in groups
    }
    rows = con.execute(
        f"""
        SELECT
            {group_column} AS group_key,
            mantyla_category,
            COALESCE(SUM(smell_count), 0) AS smell_count
        FROM analysis_mantyla_category_counts
        WHERE {_nonempty_sql(group_column)}
        GROUP BY group_key, mantyla_category
        ORDER BY group_key, mantyla_category
        """
    ).fetchall()
    for group, category, count in rows:
        if group is None or category is None or str(group) not in counts:
            continue
        if str(category) in counts[str(group)]:
            counts[str(group)][str(category)] += int(count or 0)

    percentages_by_group = _mantyla_percentages_by_group(counts, groups)
    category_order = order_labels_by_average_percentage(
        MANTYLA_PLOT_CATEGORIES,
        groups,
        percentages_by_group,
    )
    category_colors = ranked_stacked_bar_colors(category_order)

    group_totals = {
        group: int(sum(counts[group].values()))
        for group in groups
    }
    eligible_pr_counts = _eligible_pr_counts_by_group(con, group_field)
    fig, ax = plt.subplots(figsize=MANTYLA_DISTRIBUTION_FIGSIZE)
    x_values = list(range(len(groups)))
    bottoms = [0.0 for _group in groups]
    visual_percentages_by_group = {
        group: {
            category: visual_percentage
            for category, visual_percentage in zip(
                category_order,
                stacked_bar_visual_percentages(
                    [
                        percentages_by_group[group].get(category, 0.0)
                        for category in category_order
                    ],
                    minimum_segment_height=MANTYLA_MINIMUM_VISIBLE_SEGMENT_HEIGHT,
                ),
            )
        }
        for group in groups
    }
    legend_labels = []
    legend_colors = []
    callout_segments_by_group: list[list[tuple[float, float]]] = [
        [] for _group in groups
    ]
    for category in category_order:
        percentages = [
            percentages_by_group[group].get(category, 0.0)
            for group in groups
        ]
        visual_percentages = [
            visual_percentages_by_group[group].get(category, 0.0)
            for group in groups
        ]
        bars = ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=MANTYLA_PLOT_CATEGORY_NAMES[category],
            color=category_colors[category],
            edgecolor="black",
            linewidth=0.25,
        )
        if any(percentage > 0.0 for percentage in percentages):
            legend_labels.append(MANTYLA_PLOT_CATEGORY_NAMES[category])
            legend_colors.append(category_colors[category])
        for index, (bar, percentage) in enumerate(zip(bars, percentages)):
            if percentage < MANTYLA_PERCENTAGE_LABEL_MINIMUM:
                continue
            if percentage < MANTYLA_PERCENTAGE_CALLOUT_MAXIMUM:
                callout_segments_by_group[index].append(
                    (
                        float(bar.get_y())
                        + float(bar.get_height()) / 2.0
                        - float(percentage) / 2.0,
                        float(percentage),
                    )
                )
                continue
            label_x = float(bar.get_x()) + float(bar.get_width()) / 2.0
            label_y = (
                float(bar.get_y())
                + float(bar.get_height()) / 2.0
                + MANTYLA_PERCENTAGE_LABEL_Y_OFFSET
            )
            percentage_label = f"{percentage:.1f}%"
            ax.text(
                label_x,
                label_y,
                percentage_label,
                ha="center",
                va="center",
                fontsize=MANTYLA_STACKED_BAR_PERCENTAGE_FONT_SIZE,
                color=_hex_text_color(category_colors[category]),
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    for x_value, segments in zip(x_values, callout_segments_by_group):
        add_stacked_bar_percentage_callouts(
            ax,
            x_value,
            segments,
            x_offset=0.46,
            font_size=MANTYLA_PERCENTAGE_CALLOUT_FONT_SIZE,
        )
    ax.set_xlabel(
        "Cohort" if group_field == "cohort" else "Authorship Group",
        labelpad=MANTYLA_WITH_PR_COUNT_X_LABELPAD,
    )
    ax.set_ylabel("Percentage of Smells (%)")
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(groups), rotation=0, ha="center")
    add_xtick_count_sublabels(
        ax,
        x_values,
        [group_totals[group] for group in groups],
        font_size=MANTYLA_COUNT_LABEL_FONT_SIZE,
        y=MANTYLA_COUNT_LABEL_Y,
        secondary_counts=[eligible_pr_counts.get(group, 0) for group in groups],
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(groups) - 0.5))
    _add_centered_mantyla_legend_rows(
        ax,
        labels=legend_labels,
        colors=legend_colors,
    )
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, output_dir, stem)
    per_pr_counts = _mantyla_category_count_groups(con, group_field)
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "plot_type": "mantyla_distribution_stacked_bars",
            "group_field": group_field,
            "x_axis": "Cohort" if group_field == "cohort" else "Authorship Group",
            "y_axis": "Percentage of Smells (%)",
            "figure": {
                "width_inches": MANTYLA_DISTRIBUTION_FIGSIZE[0],
                "height_inches": MANTYLA_DISTRIBUTION_FIGSIZE[1],
            },
            "x_tick_count_label_font_size": MANTYLA_COUNT_LABEL_FONT_SIZE,
            "x_tick_count_labels": {
                group: f"n = {group_totals[group]:,}"
                for group in groups
            },
            "x_tick_pr_count_labels": {
                group: f"p = {eligible_pr_counts.get(group, 0):,}"
                for group in groups
            },
            "x_tick_pr_count_label_font_size": MANTYLA_COUNT_LABEL_FONT_SIZE,
            **stacked_bar_visual_metadata(
                minimum_segment_height=MANTYLA_MINIMUM_VISIBLE_SEGMENT_HEIGHT
            ),
            "percentage_label_minimum": MANTYLA_PERCENTAGE_LABEL_MINIMUM,
            "percentage_callouts": {
                "enabled": True,
                "minimum_percentage": MANTYLA_PERCENTAGE_LABEL_MINIMUM,
                "maximum_percentage_exclusive": MANTYLA_PERCENTAGE_CALLOUT_MAXIMUM,
                "font_size": MANTYLA_PERCENTAGE_CALLOUT_FONT_SIZE,
            },
            "categories": list(category_order),
            "category_labels": {
                category: MANTYLA_PLOT_CATEGORY_NAMES[category]
                for category in category_order
            },
            "category_colors": category_colors,
            "category_text_colors": {
                category: _hex_text_color(color)
                for category, color in category_colors.items()
            },
            "stack_order_basis": "descending_average_percentage_across_cohorts",
            "legend": {
                "rows": _mantyla_legend_label_rows(legend_labels),
                "columns": MANTYLA_LEGEND_COLUMNS,
                "row_counts": list(MANTYLA_LEGEND_ROW_COUNTS),
                "font_size": MANTYLA_LEGEND_FONT_SIZE,
                "center_x": MANTYLA_LEGEND_CENTER_X,
                "centered_rows": True,
            },
            "groups": {
                group: {
                    "total_code_smell_count": group_totals[group],
                    "eligible_pull_request_count": eligible_pr_counts.get(group, 0),
                    "categories": {
                        category: {
                            "code_smell_count": int(counts[group][category]),
                            "percentage": (
                                100.0
                                * int(counts[group][category])
                                / sum(counts[group].values())
                                if sum(counts[group].values())
                                else 0.0
                            ),
                            "visual_percentage": (
                                visual_percentages_by_group[group].get(
                                    category,
                                    0.0,
                                )
                            ),
                            "per_pr_summary": _summary_payload(
                                per_pr_counts.get(group, {}).get(category, [])
                            ),
                        }
                        for category in category_order
                    },
                }
                for group in groups
            },
            "human_baseline_tests_by_category": {
                category: _comparison_payloads(
                    groups,
                    {
                        group: per_pr_counts.get(group, {}).get(category, [])
                        for group in groups
                    },
                    group_field,
                )
                for category in category_order
            },
        },
    )
    plt.close(fig)


def render_mantyla_distribution_from_payload(
    payload: dict[str, object],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render one Mantyla-category stacked bar chart from stored plot data."""
    groups_payload = payload.get("groups")
    categories = payload.get("categories")
    if not isinstance(groups_payload, dict) or not isinstance(categories, list):
        raise ValueError("Mäntylä stacked-bar metadata requires groups and categories")
    groups = order_humans_first(groups_payload.keys())
    category_order = [str(category) for category in categories]
    category_labels = (
        payload.get("category_labels")
        if isinstance(payload.get("category_labels"), dict)
        else {}
    )
    payload_category_colors = (
        payload.get("category_colors")
        if isinstance(payload.get("category_colors"), dict)
        else {}
    )
    category_colors = {
        category: str(
            payload_category_colors.get(
                category,
                MANTYLA_PLOT_COLORS.get(category, "#56B4E9"),
            )
        )
        for category in category_order
    }
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(figsize=MANTYLA_DISTRIBUTION_FIGSIZE)
    x_values = list(range(len(groups)))
    bottoms = [0.0 for _group in groups]
    legend_labels = []
    legend_colors = []
    group_totals: list[int] = []
    eligible_pr_counts: list[int] = []
    callout_segments_by_group: list[list[tuple[float, float]]] = [
        [] for _group in groups
    ]
    for group in groups:
        group_payload = groups_payload.get(group, {})
        total = (
            group_payload.get("total_code_smell_count")
            if isinstance(group_payload, dict)
            else 0
        )
        if not total and isinstance(group_payload, dict):
            category_payloads = group_payload.get("categories", {})
            if isinstance(category_payloads, dict):
                total = sum(
                    int(category_payload.get("code_smell_count") or 0)
                    for category_payload in category_payloads.values()
                    if isinstance(category_payload, dict)
                )
        try:
            group_totals.append(int(total or 0))
        except (TypeError, ValueError):
            group_totals.append(0)
        eligible_count = (
            group_payload.get("eligible_pull_request_count")
            if isinstance(group_payload, dict)
            else 0
        )
        try:
            eligible_pr_counts.append(int(eligible_count or 0))
        except (TypeError, ValueError):
            eligible_pr_counts.append(0)
    for category in category_order:
        percentages = []
        visual_percentages = []
        for group in groups:
            group_payload = groups_payload.get(group, {})
            category_payload = (
                group_payload.get("categories", {}).get(category, {})
                if isinstance(group_payload, dict)
                else {}
            )
            percentages.append(float(category_payload.get("percentage") or 0.0))
            visual_percentages.append(
                float(
                    category_payload.get(
                        "visual_percentage",
                        category_payload.get("percentage") or 0.0,
                    )
                )
            )
        bars = ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=str(category_labels.get(category, MANTYLA_PLOT_CATEGORY_NAMES.get(category, category))),
            color=category_colors[category],
            edgecolor="black",
            linewidth=0.25,
        )
        if any(percentage > 0.0 for percentage in percentages):
            legend_labels.append(
                str(category_labels.get(category, MANTYLA_PLOT_CATEGORY_NAMES.get(category, category)))
            )
            legend_colors.append(category_colors[category])
        for index, (bar, percentage) in enumerate(zip(bars, percentages)):
            if percentage < MANTYLA_PERCENTAGE_LABEL_MINIMUM:
                continue
            if percentage < MANTYLA_PERCENTAGE_CALLOUT_MAXIMUM:
                callout_segments_by_group[index].append(
                    (
                        float(bar.get_y())
                        + float(bar.get_height()) / 2.0
                        - float(percentage) / 2.0,
                        float(percentage),
                    )
                )
                continue
            ax.text(
                float(bar.get_x()) + float(bar.get_width()) / 2.0,
                float(bar.get_y())
                + float(bar.get_height()) / 2.0
                + MANTYLA_PERCENTAGE_LABEL_Y_OFFSET,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=MANTYLA_STACKED_BAR_PERCENTAGE_FONT_SIZE,
                color=_hex_text_color(category_colors[category]),
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    for x_value, segments in zip(x_values, callout_segments_by_group):
        add_stacked_bar_percentage_callouts(
            ax,
            x_value,
            segments,
            x_offset=0.46,
            font_size=MANTYLA_PERCENTAGE_CALLOUT_FONT_SIZE,
        )
    ax.set_xlabel(
        str(payload.get("x_axis") or "Cohort"),
        labelpad=MANTYLA_WITH_PR_COUNT_X_LABELPAD,
    )
    ax.set_ylabel(str(payload.get("y_axis") or "Percentage of Smells (%)"))
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(groups), rotation=0, ha="center")
    add_xtick_count_sublabels(
        ax,
        x_values,
        group_totals,
        font_size=MANTYLA_COUNT_LABEL_FONT_SIZE,
        y=MANTYLA_COUNT_LABEL_Y,
        secondary_counts=eligible_pr_counts if any(eligible_pr_counts) else None,
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(groups) - 0.5))
    _add_centered_mantyla_legend_rows(
        ax,
        labels=legend_labels,
        colors=legend_colors,
    )
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, Path(output_dir), stem)
    plt.close(fig)


def _plot_delta_boxplot(
    con,
    group_field: str,
    metric_name: str,
    output_dir: Path,
    stem: str,
    figure_height: float = 1.75,
) -> None:
    plt, _mdates = require_matplotlib()
    grouped = _nonempty_groups(con, group_field, metric_name)
    groups = order_humans_first(grouped)
    human_baseline_tests = _comparison_payloads(groups, grouped, group_field)
    comparison_stat_label_size = 5.5
    comparison_stat_y = -0.15
    x_axis_label_y = -0.27
    bottom = 0.36
    fig, ax = plt.subplots(figsize=(3.5, figure_height))
    y_axis_limits = _draw_delta_boxplot_axis(
        ax,
        grouped,
        groups,
        metric_name,
        xlabel="Cohort" if group_field == "cohort" else "Authorship Group",
        ylabel=_metric_axis_label(metric_name),
        comparison_payload=human_baseline_tests,
        comparison_stat_label_size=comparison_stat_label_size,
        comparison_stat_y=comparison_stat_y,
    )
    if x_axis_label_y is not None:
        ax.xaxis.set_label_coords(0.5, x_axis_label_y)
    fig.subplots_adjust(bottom=bottom)
    save_figure(fig, output_dir, stem)
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "plot_type": "boxplot",
            "metric": metric_name,
            "group_field": group_field,
            "x_axis": "Cohort" if group_field == "cohort" else "Authorship Group",
            "y_axis": _metric_axis_label(metric_name),
            "plot_value_scale": _plot_value_scale(metric_name),
            "plot_value_unit": _plot_value_unit(metric_name),
            "scale": _scale_from_y_axis_limits(y_axis_limits),
            "percentile_capped_y_axis": y_axis_limits,
            "groups": {
                group: _summary_payload(grouped.get(group, [])) for group in groups
            },
            "human_baseline_tests": human_baseline_tests,
            "comparison_label_p_threshold": MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
            "figure": {
                "width_inches": 3.5,
                "height_inches": figure_height,
                "bottom": bottom,
                "comparison_stat_label_size": comparison_stat_label_size,
                "comparison_stat_y": comparison_stat_y,
                "x_axis_label_y": x_axis_label_y,
            },
        },
    )
    plt.close(fig)


def _draw_delta_boxplot_axis(
    ax,
    grouped: dict[str, list[float]],
    groups: list[str],
    metric_name: str,
    *,
    xlabel: str | None,
    ylabel: str | None,
    tick_label_size: float | None = None,
    y_tick_label_size: float | None = None,
    axis_label_size: float | None = None,
    tick_label_rotation: float = 0.0,
    comparison_payload: dict[str, object] | None = None,
    comparison_stat_label_size: float | None = None,
    comparison_stat_y: float = -0.1,
    comparison_label_style: str = "delta_p",
    significance_threshold: float = MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
    y_labelpad: float | None = None,
    x_labelpad: float | None = None,
    y_tick_labelpad: float | None = None,
    tight_x_margins: bool = False,
    allow_symlog_y_axis: bool = True,
) -> dict[str, object] | None:
    if groups:
        tick_labels = (
            display_group_labels(groups)
            if comparison_stat_label_size is not None
            else _boxplot_comparison_tick_labels(groups, comparison_payload)
        )
        boxplot_kwargs = ieee_boxplot_kwargs()
        colors = cohort_color_map(groups)
        ordered_colors = [colors[group] for group in groups]
        add_violin_underlay(
            ax,
            [grouped[group] for group in groups],
            colors=ordered_colors,
        )
        try:
            boxplot = ax.boxplot(
                [grouped[group] for group in groups],
                tick_labels=tick_labels,
                **boxplot_kwargs,
            )
        except TypeError:
            boxplot = ax.boxplot(
                [grouped[group] for group in groups],
                labels=tick_labels,
                **boxplot_kwargs,
        )
        style_ieee_boxplot(boxplot, ordered_colors)
        add_human_median_baseline(ax, grouped)
        _add_zero_reference_line_if_needed(ax, grouped)
        y_axis_limits = apply_percentile_capped_y_axis(ax, grouped, show_note=False)
        y_axis_limits = _apply_boxplot_lower_bound(
            ax,
            metric_name,
            grouped,
            y_axis_limits,
        )
        if allow_symlog_y_axis:
            y_axis_scale = apply_symlog_y_axis_if_range_exceeds(
                ax,
                range_threshold=MAINTAINABILITY_BOXPLOT_SYMLOG_RANGE_THRESHOLD,
            )
        else:
            ax.set_yscale("linear")
            y_axis_scale = "linear"
        if y_axis_limits is not None:
            y_axis_limits = {
                **y_axis_limits,
                "scale": y_axis_scale,
                "scale_range_threshold": (
                    MAINTAINABILITY_BOXPLOT_SYMLOG_RANGE_THRESHOLD
                ),
            }
        if tight_x_margins:
            ax.set_xlim(
                1.0 - GRID_BOXPLOT_DEFAULT_HALF_WIDTH - GRID_BOXPLOT_X_MARGIN,
                len(groups)
                + GRID_BOXPLOT_DEFAULT_HALF_WIDTH
                + GRID_BOXPLOT_X_MARGIN,
            )
        else:
            ax.set_xlim(0.45, len(groups) + 0.75)
    else:
        y_axis_limits = None
    if xlabel is not None:
        if x_labelpad is None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(xlabel, labelpad=x_labelpad)
    if ylabel is not None:
        if y_labelpad is None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(ylabel, labelpad=y_labelpad)
        if axis_label_size is not None:
            ax.yaxis.label.set_size(axis_label_size)
    ax.tick_params(axis="x", rotation=tick_label_rotation)
    if tick_label_size is not None:
        ax.tick_params(axis="x", labelsize=tick_label_size, pad=1.0)
    if y_tick_label_size is not None:
        ax.tick_params(axis="y", labelsize=y_tick_label_size)
    if y_tick_labelpad is not None:
        ax.tick_params(axis="y", pad=y_tick_labelpad)
    for label in ax.get_xticklabels():
        label.set_rotation(tick_label_rotation)
        label.set_ha("right" if tick_label_rotation else "center")
        label.set_linespacing(0.85)
    if groups and comparison_stat_label_size is not None:
        _add_boxplot_comparison_annotations(
            ax,
            groups,
            comparison_payload,
            font_size=comparison_stat_label_size,
            y_offset=comparison_stat_y,
            label_style=comparison_label_style,
            significance_threshold=significance_threshold,
        )
    ax.grid(axis="y", alpha=0.3)
    return y_axis_limits


def _center_incomplete_final_row_axes(
    axes,
    *,
    used_count: int,
    rows: int,
    columns: int,
) -> bool:
    remainder = used_count % columns
    if rows <= 1 or columns <= 1 or remainder == 0:
        return False
    final_row = used_count // columns
    if final_row >= rows or not hasattr(axes, "__getitem__"):
        return False
    active_axes = [axes[final_row, column] for column in range(remainder)]
    if not active_axes:
        return False
    first_position = axes[final_row, 0].get_position()
    last_position = axes[final_row, columns - 1].get_position()
    if columns > 1:
        second_position = axes[final_row, 1].get_position()
        column_step = float(second_position.x0 - first_position.x0)
    else:
        column_step = float(first_position.width)
    axis_width = float(first_position.width)
    full_left = float(first_position.x0)
    full_width = float(last_position.x0 + last_position.width - full_left)
    current_gap = max(0.0, column_step - axis_width)
    target_step = axis_width + current_gap
    target_group_width = axis_width + (remainder - 1) * target_step
    start_x = full_left + max(0.0, (full_width - target_group_width) / 2.0)
    for index, ax in enumerate(active_axes):
        position = ax.get_position()
        ax.set_position(
            [
                start_x + index * target_step,
                position.y0,
                position.width,
                position.height,
            ]
        )
    return True


def _plot_delta_boxplot_grid(
    con,
    output_dir: Path,
    *,
    metrics: tuple[str, ...],
    stem: str,
    layout: str,
    figsize: tuple[float, float],
    tick_label_size: float,
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float,
    hspace: float = 0.25,
    tick_label_rotation: float = 0.0,
    y_tick_label_size: float | None = None,
    axis_label_size: float | None = None,
    supxlabel_font_size: float | None = None,
    tight_layout_w_pad: float | None = None,
    ncols: int | None = None,
    comparison_stat_label_size: float = 6.0,
    comparison_stat_y: float = -0.1,
    comparison_label_style: str = "delta_p",
    significance_caption: str | None = None,
    significance_caption_font_size: float = 6.0,
    supxlabel_y: float = 0.03,
    y_labelpad: float = 3.0,
    y_tick_labelpad: float = 1.0,
    center_incomplete_final_row: bool = False,
    linear_scale_metrics: tuple[str, ...] = (),
    metric_axis_labels: dict[str, str] | None = None,
) -> None:
    plt, _mdates = require_matplotlib()
    resolved_ncols = ncols or len(metrics)
    resolved_nrows = (len(metrics) + resolved_ncols - 1) // resolved_ncols
    fig, axes = plt.subplots(
        resolved_nrows,
        resolved_ncols,
        figsize=figsize,
        constrained_layout=False,
    )
    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    elif hasattr(axes, "__iter__"):
        axes_list = list(axes)
    else:
        axes_list = [axes]
    for index, ax in enumerate(axes_list):
        row_index = index // resolved_ncols
        ax.set_zorder(resolved_nrows - row_index)

    metric_axis_pairs = list(zip(metrics, axes_list))
    metric_positions = {
        metric_name: {
            "row": index // resolved_ncols,
            "column": index % resolved_ncols,
        }
        for index, metric_name in enumerate(metrics)
    }
    center_final_row_after_layout = False
    if center_incomplete_final_row and resolved_ncols > 1:
        remainder = len(metrics) % resolved_ncols
        if remainder:
            full_count = len(metrics) - remainder
            final_row_start = full_count
            axis_indices = list(range(full_count)) + list(
                range(final_row_start, final_row_start + remainder)
            )
            metric_axis_pairs = [
                (metric_name, axes_list[axis_index])
                for metric_name, axis_index in zip(metrics, axis_indices)
            ]
            metric_positions = {
                metric_name: {
                    "row": axis_index // resolved_ncols,
                    "column": axis_index % resolved_ncols,
                }
                for metric_name, axis_index in zip(metrics, axis_indices)
            }
            center_final_row_after_layout = True

    panels: dict[str, object] = {}
    used_axes = set()
    linear_scale_metric_set = {str(metric) for metric in linear_scale_metrics}
    resolved_metric_axis_labels = metric_axis_labels or {}
    for metric_name, ax in metric_axis_pairs:
        used_axes.add(id(ax))
        grouped = _nonempty_groups(con, "cohort", metric_name)
        groups = order_humans_first(grouped)
        human_baseline_tests = _comparison_payloads(groups, grouped, "cohort")
        if metric_name in resolved_metric_axis_labels:
            axis_label = resolved_metric_axis_labels[metric_name]
        else:
            axis_label = _metric_axis_label(metric_name)
        y_axis_limits = _draw_delta_boxplot_axis(
            ax,
            grouped,
            groups,
            metric_name,
            xlabel=None,
            ylabel=axis_label,
            tick_label_size=tick_label_size,
            y_tick_label_size=y_tick_label_size,
            axis_label_size=axis_label_size,
            tick_label_rotation=tick_label_rotation,
            comparison_payload=human_baseline_tests,
            comparison_stat_label_size=comparison_stat_label_size,
            comparison_stat_y=comparison_stat_y,
            comparison_label_style=comparison_label_style,
            significance_threshold=MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
            y_labelpad=y_labelpad,
            y_tick_labelpad=y_tick_labelpad,
            tight_x_margins=True,
            allow_symlog_y_axis=metric_name not in linear_scale_metric_set,
        )
        panels[metric_name] = {
            "metric": metric_name,
            "x_axis": "Cohort",
            "y_axis": axis_label,
            "plot_value_scale": _plot_value_scale(metric_name),
            "plot_value_unit": _plot_value_unit(metric_name),
            "scale": _scale_from_y_axis_limits(y_axis_limits),
            "percentile_capped_y_axis": y_axis_limits,
            "groups": {
                group: _summary_payload(grouped.get(group, []))
                for group in groups
            },
            "human_baseline_tests": human_baseline_tests,
            "comparison_label_p_threshold": MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
            **metric_positions.get(metric_name, {}),
        }
        del grouped, groups, human_baseline_tests
    for ax in axes_list:
        if id(ax) not in used_axes:
            ax.set_visible(False)

    fig.supxlabel("Cohort", y=supxlabel_y, fontsize=supxlabel_font_size)
    if significance_caption:
        fig.text(
            left,
            supxlabel_y,
            significance_caption,
            ha="left",
            va="center",
            fontsize=significance_caption_font_size,
        )
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    tight_layout_kwargs = {"pad": 0.01, "h_pad": 1.4}
    if tight_layout_w_pad is not None:
        tight_layout_kwargs["w_pad"] = float(tight_layout_w_pad)
    centered_final_row_holder: dict[str, bool] = {"centered": False}

    def _pre_save_callback(_fig) -> None:
        if center_final_row_after_layout:
            centered_final_row_holder["centered"] = _center_incomplete_final_row_axes(
                axes,
                used_count=len(metrics),
                rows=resolved_nrows,
                columns=resolved_ncols,
            )

    save_figure(
        fig,
        output_dir,
        stem,
        tight_layout_kwargs=tight_layout_kwargs,
        pre_save_callback=_pre_save_callback,
    )
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "plot_type": "boxplot_grid",
            "layout": layout,
            "metrics": list(metrics),
            "group_field": "cohort",
            "figure": {
                "width_inches": float(figsize[0]),
                "height_inches": float(figsize[1]),
                "left": float(left),
                "right": float(right),
                "bottom": float(bottom),
                "top": float(top),
                "wspace": float(wspace),
                "hspace": float(hspace),
                "tick_label_size": float(tick_label_size),
                "y_tick_label_size": (
                    float(y_tick_label_size)
                    if y_tick_label_size is not None
                    else None
                ),
                "axis_label_size": (
                    float(axis_label_size)
                    if axis_label_size is not None
                    else None
                ),
                "supxlabel_font_size": (
                    float(supxlabel_font_size)
                    if supxlabel_font_size is not None
                    else None
                ),
                "tick_label_rotation": float(tick_label_rotation),
                "comparison_stat_label_size": float(comparison_stat_label_size),
                "comparison_stat_y": float(comparison_stat_y),
                "comparison_label_style": comparison_label_style,
                "significance_caption": significance_caption,
                "significance_caption_font_size": float(significance_caption_font_size),
                "significance_threshold": MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
                "supxlabel_y": float(supxlabel_y),
                "y_labelpad": float(y_labelpad),
                "y_tick_labelpad": float(y_tick_labelpad),
                "ncols": int(resolved_ncols),
                "nrows": int(resolved_nrows),
                "center_incomplete_final_row": bool(
                    centered_final_row_holder["centered"]
                ),
                "tight_layout_kwargs": tight_layout_kwargs,
            },
            "panels": panels,
        },
    )
    plt.close(fig)


def _plot_maintainability_metric_boxplot_grids(
    con,
    output_dir: Path,
) -> None:
    _plot_delta_boxplot_grid(
        con,
        output_dir,
        metrics=MAINTAINABILITY_QUALITY_GRID_METRICS,
        stem="maintainability_quality_metrics_boxplots_by_cohort",
        layout="2x3",
        figsize=(7.16, 3.75),
        tick_label_size=6.5,
        y_tick_label_size=6.5,
        axis_label_size=7.5,
        supxlabel_font_size=7.5,
        comparison_stat_label_size=5.0,
        comparison_stat_y=-0.11,
        tick_label_rotation=0.0,
        left=0.01,
        right=0.99,
        bottom=0.28,
        top=0.947,
        wspace=0.45,
        tight_layout_w_pad=0.45,
        hspace=0.45,
        ncols=3,
        supxlabel_y=-0.03,
        y_labelpad=0.75,
        y_tick_labelpad=0.5,
    )
    _plot_delta_boxplot_grid(
        con,
        output_dir,
        metrics=MAINTAINABILITY_QUALITY_NORMALIZED_GRID_METRICS,
        stem="maintainability_quality_metrics_normalized_boxplots_by_cohort",
        layout="2x3",
        figsize=(7.16, 3.5),
        tick_label_size=6.5,
        y_tick_label_size=6.5,
        axis_label_size=7.5,
        supxlabel_font_size=7.5,
        comparison_stat_label_size=5.0,
        comparison_stat_y=-0.11,
        tick_label_rotation=0.0,
        left=0.01,
        right=0.99,
        bottom=0.28,
        top=0.947,
        wspace=0.45,
        tight_layout_w_pad=0.45,
        hspace=0.45,
        ncols=3,
        supxlabel_y=-0.03,
        y_labelpad=0.75,
        y_tick_labelpad=0.5,
    )


def _before_after_stem(metric_name: str) -> str:
    stems = {
        "MI": "mi",
        "CC": "cc",
        "HV": "hv",
        "CCDensity": "cc_density",
        "HVDensity": "hv_density",
        "DuplicationDensity": "duplication_density",
        "CommentDensity": "comment_density",
        "CodeSmellDensity": "code_smell_density",
        "SmellCount": "smell_count",
    }
    return stems[metric_name]


def _paired_group_values(
    con,
    *,
    group_field: str,
    before_column: str,
    after_column: str,
) -> dict[str, dict[str, list[float]]]:
    group_column = _group_sql(group_field)
    before_sql = before_column
    after_sql = after_column
    rows = con.execute(
        f"""
        SELECT
            {group_column},
            {before_sql} AS before_value,
            {after_sql} AS after_value
        FROM analysis_maintainability_prs
        WHERE {_nonempty_sql(group_column)}
          AND ({before_sql} IS NOT NULL OR {after_sql} IS NOT NULL)
        ORDER BY {group_column}, before_value, after_value
        """
    ).fetchall()
    grouped: dict[str, dict[str, list[float]]] = {}
    for group, before_value, after_value in rows:
        group_key = str(group)
        grouped.setdefault(group_key, {"before": [], "after": [], "delta": []})
        if before_value is not None:
            grouped[group_key]["before"].append(float(before_value))
        if after_value is not None:
            grouped[group_key]["after"].append(float(after_value))
        if before_value is not None and after_value is not None:
            grouped[group_key]["delta"].append(
                float(after_value) - float(before_value)
            )
    return {group: grouped[group] for group in order_humans_first(grouped)}


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _plot_before_after_paired(
    con,
    *,
    group_field: str,
    metric_name: str,
    output_dir: Path,
    stem: str,
) -> None:
    plt, _mdates = require_matplotlib()
    before_column, after_column = BEFORE_AFTER_COLUMNS[metric_name]
    grouped = _paired_group_values(
        con,
        group_field=group_field,
        before_column=before_column,
        after_column=after_column,
    )
    groups = order_humans_first(grouped)
    delta_grouped = {
        group: values["delta"]
        for group, values in grouped.items()
        if values["delta"]
    }
    human_baseline_tests = _comparison_payloads(groups, delta_grouped, group_field)
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    values_by_position: list[list[float]] = []
    positions: list[float] = []
    colors: list[str] = []
    for index, group in enumerate(groups, start=1):
        before_values = grouped[group]["before"]
        after_values = grouped[group]["after"]
        positions.extend([index - 0.16, index + 0.16])
        values_by_position.extend([before_values, after_values])
        colors.extend([PAIRED_PLOT_BEFORE_COLOR, PAIRED_PLOT_AFTER_COLOR])
    if values_by_position:
        add_violin_underlay(
            ax,
            values_by_position,
            positions=positions,
            colors=colors,
            width=0.22,
        )
        boxplot = ax.boxplot(
            values_by_position,
            positions=positions,
            widths=0.24,
            **ieee_boxplot_kwargs(),
        )
        style_ieee_boxplot(boxplot, colors)
        for index, group in enumerate(groups, start=1):
            before_values = grouped[group]["before"]
            after_values = grouped[group]["after"]
            before_median = _median(before_values)
            after_median = _median(after_values)
            if before_median is not None and after_median is not None:
                ax.plot(
                    [index - 0.16, index + 0.16],
                    [before_median, after_median],
                    color="black",
                    linestyle=(0, (3, 2)),
                    marker="D",
                    markersize=1.8,
                    linewidth=0.7,
                    label="Median" if index == 1 else None,
                    zorder=4,
                )
        _set_y_axis_for_values(ax, values_by_position)
        y_axis_limits = apply_percentile_capped_y_axis(
            ax,
            values_by_position,
            show_note=False,
        )
        y_axis_limits = _apply_boxplot_lower_bound(
            ax,
            metric_name,
            values_by_position,
            y_axis_limits,
        )
    else:
        y_axis_limits = None
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(display_group_labels(groups), rotation=0, ha="center", fontsize=5.3)
    ax.set_xlabel("Cohort" if group_field == "cohort" else "Authorship Group")
    ax.set_ylabel(BEFORE_AFTER_AXIS_LABELS[metric_name])
    _add_boxplot_comparison_annotations(ax, groups, human_baseline_tests)
    ax.grid(axis="y", alpha=0.3)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=PAIRED_PLOT_BEFORE_COLOR,
            markeredgewidth=1.0,
            markersize=5,
            label="Before",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=PAIRED_PLOT_AFTER_COLOR,
            markeredgewidth=1.0,
            markersize=5,
            label="After",
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=False, ncol=2)
    fig.subplots_adjust(bottom=0.34)
    save_figure(fig, output_dir, stem)
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "plot_type": "paired_before_after_boxplot",
            "metric": metric_name,
            "group_field": group_field,
            "x_axis": "Cohort" if group_field == "cohort" else "Authorship Group",
            "y_axis": BEFORE_AFTER_AXIS_LABELS[metric_name],
            "percentile_capped_y_axis": y_axis_limits,
            "groups": {
                group: {
                    "before": {
                        **_summary_payload(grouped[group]["before"]),
                        "mean_marker": _mean(grouped[group]["before"]),
                        "median_marker": _median(grouped[group]["before"]),
                    },
                    "after": {
                        **_summary_payload(grouped[group]["after"]),
                        "mean_marker": _mean(grouped[group]["after"]),
                        "median_marker": _median(grouped[group]["after"]),
                    },
                    "delta": {
                        **_summary_payload(grouped[group]["delta"]),
                        "calculation": "after - before",
                    },
                }
                for group in groups
            },
            "human_baseline_tests": human_baseline_tests,
            "comparison_value": "after_minus_before",
            "comparison_label_p_threshold": MAINTAINABILITY_BOXPLOT_P_THRESHOLD,
        },
    )
    plt.close(fig)


def write_maintainability_analysis_plots(
    con,
    output_dir: Path,
    logger: Any | None = None,
) -> None:
    """Write maintainability coverage, smell-category, and metric delta plots."""
    apply_ieee_plot_style()
    remove_plot_outputs(output_dir, REMOVED_PLOT_OUTPUT_STEMS)
    if logger is not None:
        logger.log("writing_maintainability_mantyla_distribution")
    _plot_mantyla_distribution(
        con,
        "cohort",
        output_dir,
        "mantyla_distribution_by_cohort",
    )
    release_process_memory(logger, stage="maintainability_mantyla_plot_memory_released")
    if logger is not None:
        logger.log("writing_maintainability_boxplots")
    for metric_name in MAINTAINABILITY_PLOT_METRICS:
        _plot_delta_boxplot(
            con,
            "cohort",
            metric_name,
            output_dir / "boxplots",
            f"{_metric_stem(metric_name)}_boxplot_by_cohort",
        )
    release_process_memory(logger, stage="maintainability_boxplots_memory_released")
    if logger is not None:
        logger.log("writing_maintainability_boxplot_grids")
    _plot_maintainability_metric_boxplot_grids(
        con,
        output_dir / "boxplots-grid",
    )
    release_process_memory(logger, stage="maintainability_boxplot_grids_memory_released")


def write_maintainability_analysis_plots_from_payload(
    payload: dict[str, object],
    output_dir: Path,
    logger: Any | None = None,
) -> None:
    """Render maintainability plots from compact streaming payloads."""
    write_maintainability_analysis_plots(
        _MaintainabilityPlotPayloadConnection(payload),
        output_dir,
        logger=logger,
    )
