"""Plotting helpers for characteristics analyses.

Characteristic plots compare metric distributions by language, repository
popularity, and topic/domain. The writer functions emit both figures and
plot-data JSON so charts can be regenerated without restreaming parquet input.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

from balance_statistics_utility import (
    add_cliffs_delta_ci,
    apply_fdr_correction,
    median_confidence_interval,
    numeric_distribution_summary,
)
from characteristics_analysis_utility import (
    CHARACTERISTIC_DOMAINS,
    CHARACTERISTIC_LANGUAGES,
    CHARACTERISTIC_LANGUAGE_LABELS,
    POPULARITY_BUCKET_ORDER,
)
from plotting_utility import (
    add_human_median_baseline,
    add_violin_underlay,
    apply_percentile_capped_y_axis,
    apply_ieee_plot_style,
    display_group_labels,
    human_baseline_group,
    ieee_boxplot_kwargs,
    order_humans_first,
    require_matplotlib,
    remove_plot_outputs,
    save_figure,
    stacked_bar_visual_metadata,
    stacked_bar_visual_percentages,
    style_ieee_boxplot,
    write_plot_data,
)


def sql_string_literal(value: object) -> str:
    """Return a single-quoted SQL string literal for legacy query helpers."""
    return "'" + str(value).replace("'", "''") + "'"


AUTHORSHIP_COLORS = {
    "human": "#0072B2",
    "agent": "#D55E00",
}
DIMENSION_AXIS_LABELS = {
    "language": "Programming Language",
    "popularity": "Popularity",
    "domain": "Domain",
}
HEATMAP_COLORMAP = "RdBu_r"
HEATMAP_COLOR_MIN = -1.0
HEATMAP_COLOR_MAX = 1.0
HEATMAP_COLOR_CENTER = 0.0
HEATMAP_COLOR_LABEL = "Cliff's Delta (\u03b4)"
HEATMAP_FIGURE_WIDTH = 7.16
HEATMAP_SIGNIFICANCE_P_THRESHOLD = 0.001
HEATMAP_SIGNIFICANCE_CAPTION = "* BH-Adjusted p ≤ 0.001"
HEATMAP_XTICK_ROTATION_DEGREES = 15
HEATMAP_YTICK_ROTATION_DEGREES = 0
HEATMAP_AXIS_TICK_LABEL_SIZE = 7.5
HEATMAP_AXIS_LABEL_SIZE = 8.0
HEATMAP_X_AXIS_LABEL_PAD = 3.6
HEATMAP_X_AXIS_LABEL_Y = -0.24
HEATMAP_COLORBAR_LABEL_SIZE = 8.0
HEATMAP_COLORBAR_TICK_LABEL_SIZE = 7.5
HEATMAP_ANNOTATION_FONT_SIZE = 8.0
HEATMAP_SIGNIFICANCE_CAPTION_FONT_SIZE = 7.5
HEATMAP_SIGNIFICANCE_CAPTION_X = -0.5
HEATMAP_SIGNIFICANCE_CAPTION_Y = HEATMAP_X_AXIS_LABEL_Y
EXTENDED_HEATMAP_ANNOTATION_FONT_SIZE = 8.0
EXTENDED_HEATMAP_COMPACT_ANNOTATION_FONT_SIZE = 8.0
EXTENDED_HEATMAP_COLUMN_GROUP_LABEL_SIZE = 8.0
EXTENDED_HEATMAP_COLUMN_GROUP_LABEL_Y = 1.015
COMBINED_HEATMAP_ROW_GROUP_LABEL_SIZE = 8.0
HEATMAP_SECTION_SEPARATOR_LINEWIDTH = 1.65
HEATMAP_SECTION_SEPARATOR_ALPHA = 0.85
COMBINED_HEATMAP_ROW_GROUP_LABEL_X = -0.15
COMBINED_HEATMAP_ROW_GROUP_LABEL_PADDING = 0.1
COMBINED_HEATMAP_COLORBAR_WIDTH = "2%"
COMBINED_HEATMAP_COLORBAR_PAD = 0.045
HEATMAP_PLOT_DATA_SCHEMA = "characteristics_effect_size_heatmap/v2"
EXTENDED_HEATMAP_PLOT_DATA_SCHEMA = "characteristics_extended_effect_size_heatmap/v1"
EXTENDED_HEATMAP_GRID_PLOT_DATA_SCHEMA = (
    "characteristics_extended_effect_size_heatmap_grid/v1"
)
EXTENDED_HEATMAP_GRID_STEM = "extended_effect_size_heatmaps_grid"
HEATMAP_EXCLUDED_METRICS = {"CodeSmellFixRate", "CodeSmellIntroRate"}
DOMAIN_HEATMAP_LABELS = {
    "AI, Data, and Science": "AI, Data,\nand Science",
    "Backend, APIs, and Security": "Backend, APIs,\nand Security",
    "Distributed and Embedded Systems": "Distributed and\nEmbedded Systems",
}
LANGUAGE_COLORS = {
    "python": "#C9C6C6",
    "javascript": "#0072B2",
    "java": "#D55E00",
    "c++": "#009E73",
}
LANGUAGE_TEXT_COLORS = {
    "python": "black",
    "javascript": "white",
    "java": "white",
    "c++": "white",
}
POPULARITY_COLORS = {
    "low": "#C9C6C6",
    "medium": "#0072B2",
    "high": "#D55E00",
}
POPULARITY_TEXT_COLORS = {
    "low": "black",
    "medium": "white",
    "high": "white",
}
POPULARITY_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}
DOMAIN_COLORS = {
    "AI, Data, and Science": "#C9C6C6",
    "Backend, APIs, and Security": "#0072B2",
    "Distributed and Embedded Systems": "#D55E00",
    "Graphics": "#009E73",
    "Web and Mobile": "#CC79A7",
}
DOMAIN_TEXT_COLORS = {
    "AI, Data, and Science": "black",
    "Backend, APIs, and Security": "white",
    "Distributed and Embedded Systems": "white",
    "Graphics": "white",
    "Web and Mobile": "white",
}
DOMAIN_STACKED_LABELS = {
    "AI, Data, and Science": "AI/Data\nScience",
    "Web and Mobile": "Web/\nMobile",
    "Backend, APIs, and Security": "Backend/API\nSecurity",
    "Graphics": "Graphics",
    "Distributed and Embedded Systems": "Distributed/\nEmbedded",
}
DOMAIN_DOTPLOT_LABELS = {
    "AI, Data, and Science": "AI, Data,\nand Science",
    "Web and Mobile": "Web and\nMobile",
    "Backend, APIs, and Security": "Backend, APIs,\nand Security",
    "Graphics": "Graphics",
    "Distributed and Embedded Systems": "Distributed and\nEmbedded Systems",
}
DOMAIN_DOTPLOT_SINGLE_LINE_LABELS = {
    domain: domain for domain in CHARACTERISTIC_DOMAINS
}
STACKED_BAR_PERCENTAGE_FONT_SIZE = 6.0
STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET = -0.2
STACKED_BAR_OUTPUT_DIR = "stacked_bars"
DOTPLOT_OUTPUT_DIR = "dotplots"
DOTPLOT_PLOT_DATA_SCHEMA = "characteristics_median_ci_dotplot/v1"
DOTPLOT_CONFIDENCE_LEVEL = 0.95
DOTPLOT_LEGEND_FONT_SIZE = 5.0
DOTPLOT_COMBINED_FIGURE_WIDTH = 7.16
SMELLDENSITY_DOTPLOT_COMBINED_FIGURE_WIDTH = DOTPLOT_COMBINED_FIGURE_WIDTH
DOTPLOT_COMBINED_FIGURE_HEIGHT = 2.0
DOTPLOT_COMBINED_WSPACE = 0.45
DOTPLOT_TICK_LABEL_SIZE = 6.5
DOTPLOT_NONNEGATIVE_VISUAL_BOTTOM = -0.1
DOTPLOT_LEGEND_TITLE_ANCHOR_Y = 1.22
DOMAIN_DOTPLOT_LEGEND_TITLE_ANCHOR_Y = 1.24
CHARACTERISTICS_BOXPLOT_AXIS_LOWER_BOUND = 0.0
CHARACTERISTICS_BOXPLOT_AXIS_LOWER_PADDING = 0.1
CHARACTERISTICS_BOXPLOT_MINIMUM_UPPER_PADDING = 0.1


def metric_stem(metric: str) -> str:
    """Return a filename-safe stem for one characteristic metric."""
    stem = []
    for character in str(metric):
        if character.isalnum():
            stem.append(character.lower())
        else:
            stem.append("_")
    return "_".join(part for part in "".join(stem).split("_") if part)


def characteristic_plot_stems(
    *,
    metrics: tuple[str, ...],
    include_domain: bool,
) -> tuple[str, ...]:
    """Return expected legacy characteristic plot stems for cleanup checks."""
    del metrics
    stems: list[str] = []
    stems.extend(
        [
            "heatmaps/language_effect_size_heatmap",
            "heatmaps/popularity_effect_size_heatmap",
        ]
    )
    if include_domain:
        stems.append("heatmaps/domain_effect_size_heatmap")
    return tuple(stems)


def characteristic_composition_stacked_bar_stems(
    *,
    stem_prefix: str,
    include_domain: bool,
) -> tuple[str, ...]:
    """Return expected composition stacked-bar output stems."""
    stems = [
        f"{STACKED_BAR_OUTPUT_DIR}/{stem_prefix}_by_language_cohort_100pct_stacked_bars",
        f"{STACKED_BAR_OUTPUT_DIR}/{stem_prefix}_by_popularity_cohort_100pct_stacked_bars",
    ]
    if include_domain:
        stems.extend(
            [
                f"{STACKED_BAR_OUTPUT_DIR}/{stem_prefix}_by_domain_cohort_100pct_stacked_bars",
                f"{STACKED_BAR_OUTPUT_DIR}/{stem_prefix}_characteristics_by_cohort_100pct_stacked_bars",
            ]
        )
    return tuple(stems)


def characteristic_median_ci_dotplot_stems(
    *,
    metric: str,
    include_domain: bool,
) -> tuple[str, ...]:
    """Return expected median/CI dotplot output stems for one metric."""
    metric_name = metric_stem(metric)
    stems = [
        f"{DOTPLOT_OUTPUT_DIR}/{metric_name}_by_language_cohort_median_ci_dotplot",
        f"{DOTPLOT_OUTPUT_DIR}/{metric_name}_by_popularity_cohort_median_ci_dotplot",
    ]
    if include_domain:
        stems.extend(
            [
                f"{DOTPLOT_OUTPUT_DIR}/{metric_name}_by_domain_cohort_median_ci_dotplot",
                f"{DOTPLOT_OUTPUT_DIR}/{metric_name}_characteristics_by_cohort_median_ci_dotplots",
            ]
        )
    return tuple(stems)


def removed_characteristic_boxplot_stems(
    *,
    metrics: tuple[str, ...],
) -> tuple[str, ...]:
    """Return obsolete characteristic boxplot stems that should be removed."""
    return tuple(
        f"boxplots/{metric_stem(metric)}_popularity_boxplot_by_authorship"
        for metric in metrics
    )


def characteristic_heatmap_stems(
    *,
    include_domain: bool,
    stem_prefix: str = "",
    stem_suffix: str = "",
) -> tuple[str, ...]:
    """Return expected compact effect-size heatmap stems."""
    stems = [
        f"heatmaps/{stem_prefix}language_effect_size_heatmap{stem_suffix}",
        f"heatmaps/{stem_prefix}popularity_effect_size_heatmap{stem_suffix}",
    ]
    if include_domain:
        stems.append(f"heatmaps/{stem_prefix}domain_effect_size_heatmap{stem_suffix}")
    return tuple(stems)


def extended_characteristic_heatmap_stems(
    *,
    include_domain: bool,
    stem_suffix: str = "",
) -> tuple[str, ...]:
    """Return expected detailed effect-size heatmap stems."""
    stems = [
        f"heatmaps/language_effect_size_heatmap_extended{stem_suffix}",
        f"heatmaps/popularity_effect_size_heatmap_extended{stem_suffix}",
    ]
    if include_domain:
        stems.append(f"heatmaps/domain_effect_size_heatmap_extended{stem_suffix}")
        stems.append(f"heatmaps/{EXTENDED_HEATMAP_GRID_STEM}{stem_suffix}")
    return tuple(stems)


def _metric_sql(metric: str) -> str:
    return '"' + str(metric).replace('"', '""') + '"'


def _display_metric(metric: str) -> str:
    absolute_labels = {
        "CodeSmellDensity_Post": "Smells per KLOC",
        "MI_Post": "Maintainability Index",
        "CC_Post": "Cyclomatic Complexity",
        "HV_Post": "Halstead Volume",
        "DuplicationDensity_Post": "Duplicated Lines Density (%)",
        "CommentDensity_Post": "Comment Lines Density (%)",
    }
    if metric in absolute_labels:
        return absolute_labels[metric]
    labels = {
        "RefCount": "RefOp Count",
        "RefDensity": "RefOps per KLOC",
        "RefDiversity": "RefOp Diversity",
        "RefMagLines": "Lines Changed per RefOp",
        "RefAdded": "Lines Added per RefOp",
        "RefAddedLines": "Lines Added per RefOp",
        "RefRemoved": "Lines Deleted per RefOp",
        "RefDeletedLines": "Lines Deleted per RefOp",
        "SmellDensity": "Smells per KLOC",
        "SmellsDelta": "\u0394 Smell Count",
        "MI": "\u0394 Maintainability Index",
        "CC": "\u0394 Cyclomatic Complexity",
        "HV": "\u0394 Halstead Volume",
        "DuplicationDensity": "\u0394 Duplicated Lines Density (%)",
        "CommentDensity": "\u0394 Comment Lines Density (%)",
        "CCDensity": "\u0394 Cyclomatic Complexity per KLOC",
        "HVDensity": "\u0394 Halstead Volume per KLOC",
        "CodeSmellDensityDelta": "\u0394 Smells per KLOC",
        "CodeSmellIntroRate": "Smell Intro Rate (%)",
        "CodeSmellFixRate": "Smell Fix Rate (%)",
        "SmellRegressionRate": "Smell Regression Rate (%)",
        "SmellRegRate": "Smell Regression Rate (%)",
        "cyclomatic_complexity_per_kloc": "\u0394 Cyclomatic Complexity per KLOC",
        "halstead_volume_per_kloc": "\u0394 Halstead Volume per KLOC",
        "comment_ratio": "\u0394 Comment Ratio (%)",
        "maintainability_index": "\u0394 Maintainability Index",
        "multimetric_duplication_score": "\u0394 Duplication Score",
        "original_code_smell_density_delta": "\u0394 Smells per KLOC",
        "original_duplication_density": "\u0394 Duplicated Lines Density (%)",
        "fanout_external": "\u0394 Fan Out",
        "fanout_external_per_kloc": "\u0394 Fan Out per KLOC",
        "tiobe": "\u0394 TIOBE Quality Score",
        "halstead_bugprop_per_kloc": "\u0394 Halstead Delivered Bugs per KLOC",
        "halstead_difficulty_per_kloc": "\u0394 Halstead Difficulty per KLOC",
        "halstead_effort_per_kloc": "\u0394 Halstead Effort per KLOC",
        "halstead_timerequired_per_kloc": "\u0394 Halstead Time Required per KLOC",
        "tiobe_complexity": "\u0394 TIOBE Complexity",
        "tiobe_duplication": "\u0394 TIOBE Duplication",
    }
    return labels.get(metric, metric)


def _wrap_to_two_lines(label: str, *, max_length: int = 22) -> str:
    text = str(label)
    if len(text) <= max_length or "\n" in text:
        return text
    words = text.split()
    if len(words) < 2:
        return text
    best_index = 1
    best_balance = len(text)
    for index in range(1, len(words)):
        left = " ".join(words[:index])
        right = " ".join(words[index:])
        balance = abs(len(left) - len(right))
        if balance < best_balance:
            best_index = index
            best_balance = balance
    return " ".join(words[:best_index]) + "\n" + " ".join(words[best_index:])


def _display_level_label(dimension: str, label: Any) -> str:
    text = str(label)
    if dimension != "domain":
        return text
    return DOMAIN_HEATMAP_LABELS.get(text, _wrap_to_two_lines(text))


def _display_dimension_axis_label(label: str) -> str:
    if str(label) == "Programming Language":
        return "Programming\nLanguage"
    return str(label)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric_value):
        return None
    return numeric_value


def _heatmap_test_payload(test: dict[str, Any]) -> dict[str, Any]:
    return {
        "u_statistic": _finite_float_or_none(test.get("u_statistic")),
        "p": _finite_float_or_none(test.get("adjusted_p_value")),
        "p_value": _finite_float_or_none(test.get("p_value")),
        "adjusted_p_value": _finite_float_or_none(test.get("adjusted_p_value")),
        "fdr_method": test.get("fdr_method"),
        "significant_after_fdr": test.get("significant_after_fdr"),
        "cliffs_delta": _finite_float_or_none(test.get("cliffs_delta")),
        "cliffs_delta_ci95_low": _finite_float_or_none(
            test.get("cliffs_delta_ci95_low")
        ),
        "cliffs_delta_ci95_high": _finite_float_or_none(
            test.get("cliffs_delta_ci95_high")
        ),
        "n_first": test.get("n_first"),
        "n_second": test.get("n_second"),
        "first_group": test.get("first_group"),
        "second_group": test.get("second_group"),
    }


def _format_heatmap_delta(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def _heatmap_significance_star(adjusted_p_value: float | None) -> bool:
    return (
        adjusted_p_value is not None
        and float(adjusted_p_value) <= HEATMAP_SIGNIFICANCE_P_THRESHOLD
    )


def _heatmap_annotation_text(
    delta: float | None,
    adjusted_p_value: float | None,
) -> str:
    suffix = "*" if _heatmap_significance_star(adjusted_p_value) else ""
    return f"{_format_heatmap_delta(delta)}{suffix}"


def _add_heatmap_significance_caption(
    ax,
    *,
    x: float = HEATMAP_SIGNIFICANCE_CAPTION_X,
    y: float = HEATMAP_SIGNIFICANCE_CAPTION_Y,
    coordinate_system: str = "data_x_axes_y",
) -> None:
    from matplotlib.transforms import blended_transform_factory

    transform = (
        ax.transAxes
        if coordinate_system == "axes"
        else blended_transform_factory(ax.transData, ax.transAxes)
    )
    ax.text(
        x,
        y,
        HEATMAP_SIGNIFICANCE_CAPTION,
        transform=transform,
        ha="left",
        va="center",
        fontsize=HEATMAP_SIGNIFICANCE_CAPTION_FONT_SIZE,
        clip_on=False,
    )


def _align_heatmap_metric_label(ax) -> None:
    ax.xaxis.set_label_coords(0.5, HEATMAP_X_AXIS_LABEL_Y)


def _values(
    con,
    *,
    table_name: str,
    metric: str,
    where_sql: str,
) -> list[float]:
    rows = con.execute(
        f"""
        SELECT {_metric_sql(metric)}
        FROM {table_name}
        WHERE {where_sql}
          AND {_metric_sql(metric)} IS NOT NULL
        ORDER BY {_metric_sql(metric)}
        """
    ).fetchall()
    return [float(row[0]) for row in rows if row[0] is not None]


def _cohorts(con, *, base_where_sql: str) -> list[str]:
    rows = con.execute(
        f"""
        SELECT DISTINCT cohort
        FROM analysis_characteristic_prs
        WHERE {base_where_sql}
          AND cohort IS NOT NULL
          AND NULLIF(trim(CAST(cohort AS VARCHAR)), '') IS NOT NULL
        ORDER BY cohort
        """
    ).fetchall()
    return order_humans_first(str(row[0]) for row in rows)


def _stacked_bar_percentage_maps(
    *,
    cohorts: list[str],
    labels: list[str],
    counts_by_cohort: dict[str, dict[str, int]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    true_by_cohort: dict[str, dict[str, float]] = {}
    visual_by_cohort: dict[str, dict[str, float]] = {}
    for cohort in cohorts:
        total = sum(counts_by_cohort.get(cohort, {}).get(label, 0) for label in labels)
        percentages = [
            100.0 * counts_by_cohort.get(cohort, {}).get(label, 0) / total
            if total
            else 0.0
            for label in labels
        ]
        visual_percentages = stacked_bar_visual_percentages(percentages)
        true_by_cohort[cohort] = {
            label: percentages[index]
            for index, label in enumerate(labels)
        }
        visual_by_cohort[cohort] = {
            label: visual_percentages[index]
            for index, label in enumerate(labels)
        }
    return true_by_cohort, visual_by_cohort


def _composition_counts_by_cohort(
    con,
    *,
    count_table_name: str,
    count_column: str,
    dimension: str,
    levels: list[str],
    base_where_sql: str,
) -> dict[str, dict[str, int]]:
    cohorts = _cohorts(con, base_where_sql=base_where_sql)
    nested = {
        cohort: {level: 0 for level in levels}
        for cohort in cohorts
    }
    if dimension == "domain":
        source_table_name = "analysis_characteristic_domain_prs"
        level_column = "domain"
    elif dimension == "language":
        source_table_name = "analysis_characteristic_prs"
        level_column = "characteristic_language"
    elif dimension == "popularity":
        source_table_name = "analysis_characteristic_prs"
        level_column = "popularity_group"
    else:
        raise ValueError(f"Unsupported characteristic dimension: {dimension}")

    level_filter = ", ".join(sql_string_literal(level) for level in levels)
    rows = con.execute(
        f"""
        SELECT
            prs.cohort,
            prs.{level_column} AS level_key,
            COALESCE(SUM(counts.{_metric_sql(count_column)}), 0) AS item_count
        FROM {source_table_name} AS prs
        INNER JOIN {count_table_name} AS counts
            ON counts.analysis_row_id = prs.analysis_row_id
        WHERE {base_where_sql}
          AND prs.cohort IS NOT NULL
          AND NULLIF(trim(CAST(prs.cohort AS VARCHAR)), '') IS NOT NULL
          AND prs.{level_column} IN ({level_filter})
        GROUP BY prs.cohort, level_key
        ORDER BY prs.cohort, level_key
        """
    ).fetchall()
    for cohort, level, count in rows:
        if cohort is None or level is None:
            continue
        cohort_key = str(cohort)
        level_key = str(level)
        if cohort_key not in nested:
            nested[cohort_key] = {candidate: 0 for candidate in levels}
        if level_key in nested[cohort_key]:
            nested[cohort_key][level_key] += int(count or 0)
    return nested


def _composition_panel_payload(
    *,
    cohorts: list[str],
    levels: list[str],
    counts_by_cohort: dict[str, dict[str, int]],
    count_key: str,
    total_key: str,
) -> dict[str, Any]:
    true_by_cohort, visual_by_cohort = _stacked_bar_percentage_maps(
        cohorts=cohorts,
        labels=levels,
        counts_by_cohort=counts_by_cohort,
    )
    return {
        cohort: {
            total_key: int(
                sum(counts_by_cohort.get(cohort, {}).get(level, 0) for level in levels)
            ),
            "groups": {
                level: {
                    count_key: int(counts_by_cohort.get(cohort, {}).get(level, 0)),
                    "percentage": true_by_cohort[cohort].get(level, 0.0),
                    "visual_percentage": visual_by_cohort[cohort].get(level, 0.0),
                }
                for level in levels
            },
        }
        for cohort in cohorts
    }


def _draw_characteristic_composition_stacked_bars(
    ax,
    *,
    cohorts: list[str],
    levels: list[str],
    counts_by_cohort: dict[str, dict[str, int]],
    display_labels: dict[str, str],
    colors: dict[str, str],
    text_colors: dict[str, str],
    y_axis_label: str,
    show_ylabel: bool = True,
    legend_fontsize: float | None = None,
) -> None:
    bottoms = [0.0 for _cohort in cohorts]
    x_values = list(range(len(cohorts)))
    true_by_cohort, visual_by_cohort = _stacked_bar_percentage_maps(
        cohorts=cohorts,
        labels=levels,
        counts_by_cohort=counts_by_cohort,
    )
    for level in levels:
        percentages = [
            true_by_cohort[cohort].get(level, 0.0)
            for cohort in cohorts
        ]
        visual_percentages = [
            visual_by_cohort[cohort].get(level, 0.0)
            for cohort in cohorts
        ]
        ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=display_labels.get(level, level),
            color=colors[level],
            edgecolor="black",
            linewidth=0.3,
        )
        for x_value, bottom, percentage, visual_percentage in zip(
            x_values,
            bottoms,
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
                color=text_colors.get(level, "black"),
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    ax.set_xlabel("Cohort")
    ax.set_ylabel(y_axis_label if show_ylabel else "", labelpad=3.0)
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(cohorts), rotation=0, ha="center")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.tick_params(axis="y", pad=1.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(cohorts) - 0.5))
    legend_kwargs: dict[str, Any] = {}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    ax.legend(
        frameon=False,
        ncol=len(levels),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0.0,
        columnspacing=0.45,
        handlelength=0.8,
        handletextpad=0.25,
        **legend_kwargs,
    )
    ax.grid(axis="y", alpha=0.3)


def _draw_characteristic_composition_stacked_bars_from_payload(
    ax,
    panel_payload: dict[str, Any],
    *,
    cohorts: list[str],
    y_axis_label: str,
    show_ylabel: bool = True,
) -> None:
    levels = [str(level) for level in panel_payload.get("levels", [])]
    groups_payload = panel_payload.get("groups")
    if not levels or not isinstance(groups_payload, dict):
        raise ValueError("characteristics stacked-bar metadata requires levels and groups")
    display_labels = (
        panel_payload.get("display_labels")
        if isinstance(panel_payload.get("display_labels"), dict)
        else {}
    )
    colors = panel_payload.get("colors") if isinstance(panel_payload.get("colors"), dict) else {}
    text_colors = (
        panel_payload.get("text_colors")
        if isinstance(panel_payload.get("text_colors"), dict)
        else {}
    )
    bottoms = [0.0 for _cohort in cohorts]
    x_values = list(range(len(cohorts)))
    for level in levels:
        percentages = []
        visual_percentages = []
        for cohort in cohorts:
            cohort_payload = groups_payload.get(cohort, {})
            level_payload = (
                cohort_payload.get("levels", {}).get(level, {})
                if isinstance(cohort_payload, dict)
                else {}
            )
            percentages.append(float(level_payload.get("percentage") or 0.0))
            visual_percentages.append(
                float(
                    level_payload.get(
                        "visual_percentage",
                        level_payload.get("percentage") or 0.0,
                    )
                )
            )
        ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=str(display_labels.get(level, level)),
            color=str(colors.get(level, "#56B4E9")),
            edgecolor="black",
            linewidth=0.3,
        )
        for x_value, bottom, percentage, visual_percentage in zip(
            x_values,
            bottoms,
            percentages,
            visual_percentages,
        ):
            if percentage <= 0.0:
                continue
            ax.text(
                x_value,
                bottom + visual_percentage / 2.0 + STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=STACKED_BAR_PERCENTAGE_FONT_SIZE,
                color=str(text_colors.get(level, "black")),
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    ax.set_xlabel("Cohort")
    ax.set_ylabel(y_axis_label if show_ylabel else "", labelpad=3.0)
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(cohorts), rotation=0, ha="center")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.tick_params(axis="y", pad=1.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(cohorts) - 0.5))
    legend_kwargs: dict[str, Any] = {}
    if panel_payload.get("legend_fontsize") is not None:
        legend_kwargs["fontsize"] = panel_payload.get("legend_fontsize")
    ax.legend(
        frameon=False,
        ncol=len(levels),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0.0,
        columnspacing=0.45,
        handlelength=0.8,
        handletextpad=0.25,
        **legend_kwargs,
    )
    ax.grid(axis="y", alpha=0.3)


def _composition_dimension_configs(*, include_domain: bool) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {
            "dimension": "language",
            "title": "Programming Language",
            "levels": list(CHARACTERISTIC_LANGUAGES),
            "display_labels": dict(CHARACTERISTIC_LANGUAGE_LABELS),
            "colors": LANGUAGE_COLORS,
            "text_colors": LANGUAGE_TEXT_COLORS,
            "legend_fontsize": None,
        },
        {
            "dimension": "popularity",
            "title": "Popularity",
            "levels": list(POPULARITY_BUCKET_ORDER),
            "display_labels": POPULARITY_LABELS,
            "colors": POPULARITY_COLORS,
            "text_colors": POPULARITY_TEXT_COLORS,
            "legend_fontsize": None,
        },
    ]
    if include_domain:
        configs.append(
            {
                "dimension": "domain",
                "title": "Domain",
                "levels": list(CHARACTERISTIC_DOMAINS),
                "display_labels": DOMAIN_STACKED_LABELS,
                "colors": DOMAIN_COLORS,
                "text_colors": DOMAIN_TEXT_COLORS,
                "legend_fontsize": 5.0,
            }
        )
    return configs


def _write_characteristic_composition_single_plot(
    *,
    output_dir: Path,
    stem: str,
    cohorts: list[str],
    config: dict[str, Any],
    counts_by_cohort: dict[str, dict[str, int]],
    y_axis_label: str,
    count_key: str,
    total_key: str,
    domain_enabled: bool,
) -> None:
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    _draw_characteristic_composition_stacked_bars(
        ax,
        cohorts=cohorts,
        levels=config["levels"],
        counts_by_cohort=counts_by_cohort,
        display_labels=config["display_labels"],
        colors=config["colors"],
        text_colors=config["text_colors"],
        y_axis_label=y_axis_label,
        show_ylabel=True,
        legend_fontsize=config.get("legend_fontsize"),
    )
    save_figure(fig, output_dir / STACKED_BAR_OUTPUT_DIR, stem)
    write_plot_data(
        output_dir / STACKED_BAR_OUTPUT_DIR,
        stem,
        {
            "plot": stem,
            "plot_type": "characteristics_composition_stacked_bars",
            "dimension": config["dimension"],
            "x_axis": "Cohort",
            "y_axis": y_axis_label,
            **stacked_bar_visual_metadata(),
            "domain_enabled": bool(domain_enabled),
            "cohorts": cohorts,
            "levels": config["levels"],
            "display_labels": config["display_labels"],
            "colors": config["colors"],
            "text_colors": config["text_colors"],
            "legend_fontsize": config.get("legend_fontsize"),
            "groups": _composition_panel_payload(
                cohorts=cohorts,
                levels=config["levels"],
                counts_by_cohort=counts_by_cohort,
                count_key=count_key,
                total_key=total_key,
            ),
        },
    )
    plt.close(fig)


def _write_characteristic_composition_combined_plot(
    *,
    output_dir: Path,
    stem: str,
    cohorts: list[str],
    configs: list[dict[str, Any]],
    counts_by_dimension: dict[str, dict[str, dict[str, int]]],
    y_axis_label: str,
    count_key: str,
    total_key: str,
    domain_enabled: bool,
) -> None:
    if len(configs) != 3:
        return
    plt, _mdates = require_matplotlib()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOTPLOT_COMBINED_FIGURE_WIDTH, 2.25),
        sharey=True,
    )
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else list(axes)
    for index, (ax, config) in enumerate(zip(axes_list, configs)):
        _draw_characteristic_composition_stacked_bars(
            ax,
            cohorts=cohorts,
            levels=config["levels"],
            counts_by_cohort=counts_by_dimension[config["dimension"]],
            display_labels=config["display_labels"],
            colors=config["colors"],
            text_colors=config["text_colors"],
            y_axis_label=y_axis_label,
            show_ylabel=(index == 0),
            legend_fontsize=config.get("legend_fontsize"),
        )
        if index > 0:
            ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.28, top=0.947, wspace=0.35)
    save_figure(fig, output_dir / STACKED_BAR_OUTPUT_DIR, stem)
    write_plot_data(
        output_dir / STACKED_BAR_OUTPUT_DIR,
        stem,
        {
            "plot": stem,
            "plot_type": "characteristics_composition_stacked_bars_grid",
            "layout": "1x3",
            "x_axis": "Cohort",
            "y_axis": y_axis_label,
            **stacked_bar_visual_metadata(),
            "domain_enabled": bool(domain_enabled),
            "cohorts": cohorts,
            "figure": {
                "width_inches": DOTPLOT_COMBINED_FIGURE_WIDTH,
                "height_inches": 2.25,
                "left": 0.01,
                "right": 0.99,
                "bottom": 0.28,
                "top": 0.947,
                "wspace": 0.35,
            },
            "panels": {
                config["dimension"]: {
                    "dimension": config["dimension"],
                    "title": config["title"],
                    "levels": config["levels"],
                    "display_labels": config["display_labels"],
                    "colors": config["colors"],
                    "text_colors": config["text_colors"],
                    "legend_fontsize": config.get("legend_fontsize"),
                    "groups": _composition_panel_payload(
                        cohorts=cohorts,
                        levels=config["levels"],
                        counts_by_cohort=counts_by_dimension[config["dimension"]],
                        count_key=count_key,
                        total_key=total_key,
                    ),
                }
                for config in configs
            },
        },
    )
    plt.close(fig)


def render_characteristics_composition_stacked_bars_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render one characteristic composition stacked bar from plot data."""
    if payload.get("plot_type") != "characteristics_composition_stacked_bars":
        raise ValueError("unsupported characteristics stacked-bar metadata")
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("characteristics stacked-bar metadata requires groups")
    cohorts = [
        str(cohort)
        for cohort in (
            payload.get("cohorts")
            if isinstance(payload.get("cohorts"), list)
            else order_humans_first(groups)
        )
    ]
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    _draw_characteristic_composition_stacked_bars_from_payload(
        ax,
        payload,
        cohorts=cohorts,
        y_axis_label=str(payload.get("y_axis") or "Percentage (%)"),
        show_ylabel=True,
    )
    save_figure(fig, Path(output_dir), stem)
    plt.close(fig)


def render_characteristics_composition_stacked_bars_grid_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render a multi-panel characteristic composition grid from plot data."""
    if payload.get("plot_type") != "characteristics_composition_stacked_bars_grid":
        raise ValueError("unsupported characteristics stacked-bar grid metadata")
    panels = payload.get("panels")
    if not isinstance(panels, dict) or not panels:
        raise ValueError("characteristics stacked-bar grid metadata requires panels")
    figure = payload.get("figure") if isinstance(payload.get("figure"), dict) else {}
    panel_items = list(panels.items())
    cohorts = [
        str(cohort)
        for cohort in (
            payload.get("cohorts")
            if isinstance(payload.get("cohorts"), list)
            else []
        )
    ]
    if not cohorts:
        first_panel = next(iter(panels.values()))
        first_groups = first_panel.get("groups") if isinstance(first_panel, dict) else {}
        cohorts = order_humans_first(first_groups) if isinstance(first_groups, dict) else []
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, axes = plt.subplots(
        1,
        len(panel_items),
        figsize=(
            float(figure.get("width_inches", DOTPLOT_COMBINED_FIGURE_WIDTH)),
            float(figure.get("height_inches", 2.25)),
        ),
        sharey=True,
    )
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for index, (ax, (_dimension, panel)) in enumerate(zip(axes_list, panel_items)):
        if not isinstance(panel, dict):
            ax.set_visible(False)
            continue
        _draw_characteristic_composition_stacked_bars_from_payload(
            ax,
            panel,
            cohorts=cohorts,
            y_axis_label=str(payload.get("y_axis") or "Percentage (%)"),
            show_ylabel=(index == 0),
        )
        if index > 0:
            ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=float(figure.get("bottom", 0.28)),
        top=float(figure.get("top", 0.947)),
        wspace=float(figure.get("wspace", 0.35)),
    )
    save_figure(fig, Path(output_dir), stem)
    plt.close(fig)


def write_characteristics_composition_stacked_bars(
    con,
    *,
    output_dir: Path,
    count_table_name: str,
    count_column: str,
    stem_prefix: str,
    y_axis_label: str,
    count_key: str,
    total_key: str,
    include_domain: bool,
    base_where_sql: str = "TRUE",
) -> None:
    """Write characteristic composition stacked bars from query-like input."""
    apply_ieee_plot_style()
    output_dir = Path(output_dir)
    all_stems = set(
        characteristic_composition_stacked_bar_stems(
            stem_prefix=stem_prefix,
            include_domain=True,
        )
    )
    kept_stems = set(
        characteristic_composition_stacked_bar_stems(
            stem_prefix=stem_prefix,
            include_domain=include_domain,
        )
    )
    remove_plot_outputs(output_dir, tuple(sorted(all_stems - kept_stems)))
    cohorts = _cohorts(con, base_where_sql=base_where_sql)
    configs = _composition_dimension_configs(include_domain=include_domain)
    counts_by_dimension: dict[str, dict[str, dict[str, int]]] = {}
    for config in configs:
        counts_by_dimension[config["dimension"]] = _composition_counts_by_cohort(
            con,
            count_table_name=count_table_name,
            count_column=count_column,
            dimension=config["dimension"],
            levels=config["levels"],
            base_where_sql=base_where_sql,
        )
        _write_characteristic_composition_single_plot(
            output_dir=output_dir,
            stem=(
                f"{stem_prefix}_by_{config['dimension']}_cohort_"
                "100pct_stacked_bars"
            ),
            cohorts=cohorts,
            config=config,
            counts_by_cohort=counts_by_dimension[config["dimension"]],
            y_axis_label=y_axis_label,
            count_key=count_key,
            total_key=total_key,
            domain_enabled=include_domain,
        )
    if include_domain:
        _write_characteristic_composition_combined_plot(
            output_dir=output_dir,
            stem=f"{stem_prefix}_characteristics_by_cohort_100pct_stacked_bars",
            cohorts=cohorts,
            configs=configs,
            counts_by_dimension=counts_by_dimension,
            y_axis_label=y_axis_label,
            count_key=count_key,
            total_key=total_key,
            domain_enabled=include_domain,
        )


def write_characteristics_composition_stacked_bars_from_payload(
    *,
    counts_by_dimension: dict[str, dict[str, dict[str, int]]],
    output_dir: Path,
    stem_prefix: str,
    y_axis_label: str,
    count_key: str,
    total_key: str,
    include_domain: bool,
) -> None:
    """Write characteristic composition plots from streaming payload counts."""
    apply_ieee_plot_style()
    output_dir = Path(output_dir)
    all_stems = set(
        characteristic_composition_stacked_bar_stems(
            stem_prefix=stem_prefix,
            include_domain=True,
        )
    )
    kept_stems = set(
        characteristic_composition_stacked_bar_stems(
            stem_prefix=stem_prefix,
            include_domain=include_domain,
        )
    )
    remove_plot_outputs(output_dir, tuple(sorted(all_stems - kept_stems)))
    configs = _composition_dimension_configs(include_domain=include_domain)
    cohorts = order_humans_first(
        {
            cohort
            for by_cohort in counts_by_dimension.values()
            for cohort in by_cohort.keys()
        }
    )
    normalized_counts: dict[str, dict[str, dict[str, int]]] = {}
    for config in configs:
        dimension = str(config["dimension"])
        levels = list(config["levels"])
        normalized_counts[dimension] = {
            cohort: {
                level: int(
                    counts_by_dimension
                    .get(dimension, {})
                    .get(cohort, {})
                    .get(level, 0)
                )
                for level in levels
            }
            for cohort in cohorts
        }
        _write_characteristic_composition_single_plot(
            output_dir=output_dir,
            stem=(
                f"{stem_prefix}_by_{dimension}_cohort_"
                "100pct_stacked_bars"
            ),
            cohorts=cohorts,
            config=config,
            counts_by_cohort=normalized_counts[dimension],
            y_axis_label=y_axis_label,
            count_key=count_key,
            total_key=total_key,
            domain_enabled=include_domain,
        )
    if include_domain:
        _write_characteristic_composition_combined_plot(
            output_dir=output_dir,
            stem=f"{stem_prefix}_characteristics_by_cohort_100pct_stacked_bars",
            cohorts=cohorts,
            configs=configs,
            counts_by_dimension=normalized_counts,
            y_axis_label=y_axis_label,
            count_key=count_key,
            total_key=total_key,
            domain_enabled=include_domain,
        )


def _dotplot_dimension_configs(*, include_domain: bool) -> list[dict[str, Any]]:
    configs = _composition_dimension_configs(include_domain=include_domain)
    for config in configs:
        config["legend_fontsize"] = DOTPLOT_LEGEND_FONT_SIZE
        config["show_human_reference_lines"] = True
        if config["dimension"] == "domain":
            config["display_labels"] = DOMAIN_DOTPLOT_LABELS
            config["legend_ncol"] = 3
            config["legend_label_alignment"] = "center"
            config["legend_centered_rows"] = True
            config["legend_row_counts"] = [2, 3]
            config["legend_anchor_y"] = DOMAIN_DOTPLOT_LEGEND_TITLE_ANCHOR_Y
    return configs


def _dotplot_source_for_dimension(dimension: str) -> tuple[str, str]:
    if dimension == "domain":
        return "analysis_characteristic_domain_prs", "domain"
    if dimension == "language":
        return "analysis_characteristic_prs", "characteristic_language"
    if dimension == "popularity":
        return "analysis_characteristic_prs", "popularity_group"
    raise ValueError(f"Unsupported characteristic dimension: {dimension}")


def _metric_values_by_cohort_dimension(
    con,
    *,
    metric: str,
    dimension: str,
    levels: list[str],
    cohorts: list[str],
    base_where_sql: str,
) -> dict[str, dict[str, list[float]]]:
    values_by_cohort = {
        cohort: {level: [] for level in levels}
        for cohort in cohorts
    }
    source_table_name, level_column = _dotplot_source_for_dimension(dimension)
    level_filter = ", ".join(sql_string_literal(level) for level in levels)
    rows = con.execute(
        f"""
        SELECT
            cohort,
            {level_column} AS level_key,
            {_metric_sql(metric)} AS metric_value
        FROM {source_table_name}
        WHERE {base_where_sql}
          AND cohort IS NOT NULL
          AND NULLIF(trim(CAST(cohort AS VARCHAR)), '') IS NOT NULL
          AND {level_column} IN ({level_filter})
          AND {_metric_sql(metric)} IS NOT NULL
        ORDER BY cohort, level_key, metric_value
        """
    ).fetchall()
    for cohort, level, value in rows:
        if cohort is None or level is None or value is None:
            continue
        cohort_key = str(cohort)
        level_key = str(level)
        if cohort_key not in values_by_cohort:
            values_by_cohort[cohort_key] = {candidate: [] for candidate in levels}
        if level_key in values_by_cohort[cohort_key]:
            values_by_cohort[cohort_key][level_key].append(float(value))
    return values_by_cohort


def _median_ci_summary(values: list[float]) -> dict[str, Any]:
    return median_confidence_interval(
        values,
        confidence=DOTPLOT_CONFIDENCE_LEVEL,
    )


def _dotplot_panel_payload(
    *,
    cohorts: list[str],
    levels: list[str],
    values_by_cohort: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    return {
        cohort: {
            level: _median_ci_summary(
                values_by_cohort.get(cohort, {}).get(level, [])
            )
            for level in levels
        }
        for cohort in cohorts
    }


def _dotplot_offsets(level_count: int) -> list[float]:
    if level_count <= 1:
        return [0.0]
    step = min(0.16, 0.70 / max(1, level_count - 1))
    center = (level_count - 1) / 2.0
    return [(index - center) * step for index in range(level_count)]


def _set_dotplot_y_axis(
    ax,
    *,
    summaries_by_cohort: dict[str, dict[str, dict[str, Any]]],
    cohorts: list[str],
    levels: list[str],
) -> None:
    _set_dotplot_y_axis_from_summaries(
        ax,
        summaries_list=[summaries_by_cohort],
        cohorts=cohorts,
        levels_by_summary=[levels],
    )


def _set_dotplot_y_axis_from_summaries(
    ax,
    *,
    summaries_list: list[dict[str, dict[str, dict[str, Any]]]],
    cohorts: list[str],
    levels_by_summary: list[list[str]],
) -> None:
    values: list[float] = []
    for summaries_by_cohort, levels in zip(summaries_list, levels_by_summary):
        for cohort in cohorts:
            for level in levels:
                summary = summaries_by_cohort.get(cohort, {}).get(level, {})
                for key in ("median", "ci95_low", "ci95_high"):
                    value = summary.get(key)
                    if value is None:
                        continue
                    numeric = float(value)
                    if isfinite(numeric):
                        values.append(numeric)
    if not values:
        return
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = max(abs(maximum) * 0.08, 0.05)
    else:
        padding = max((maximum - minimum) * 0.08, 0.05)
    bottom = minimum - padding
    top = maximum + padding
    if minimum >= 0.0:
        bottom = DOTPLOT_NONNEGATIVE_VISUAL_BOTTOM
    ax.set_ylim(bottom, top)
    if minimum >= 0.0:
        nonnegative_ticks = [
            tick for tick in ax.get_yticks()
            if float(tick) >= 0.0
        ]
        if nonnegative_ticks:
            ax.set_yticks(nonnegative_ticks)


def _dotplot_human_reference_lines(
    *,
    cohorts: list[str],
    config: dict[str, Any],
    summaries_by_cohort: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    baseline_cohort = human_baseline_group(cohorts)
    if baseline_cohort is None:
        return {}
    lines: dict[str, dict[str, Any]] = {}
    for level in config["levels"]:
        median_value = (
            summaries_by_cohort
            .get(baseline_cohort, {})
            .get(level, {})
            .get("median")
        )
        if median_value is None:
            continue
        lines[level] = {
            "baseline_cohort": baseline_cohort,
            "level": level,
            "level_label": config["display_labels"].get(level, level),
            "median": float(median_value),
            "color": config["colors"][level],
            "linestyle": "dashed",
            "linewidth": 0.8,
            "alpha": 0.35,
        }
    return lines


def _draw_dotplot_human_reference_lines(
    ax,
    *,
    cohorts: list[str],
    config: dict[str, Any],
    summaries_by_cohort: dict[str, dict[str, dict[str, Any]]],
) -> None:
    if not config.get("show_human_reference_lines"):
        return
    for line in _dotplot_human_reference_lines(
        cohorts=cohorts,
        config=config,
        summaries_by_cohort=summaries_by_cohort,
    ).values():
        ax.axhline(
            float(line["median"]),
            color=str(line["color"]),
            linestyle=(0, (4, 2)),
            linewidth=float(line["linewidth"]),
            alpha=float(line["alpha"]),
            zorder=1,
        )


def _add_centered_dotplot_legend_rows(
    ax,
    *,
    config: dict[str, Any],
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker

    levels = list(config["levels"])
    row_counts = list(config.get("legend_row_counts") or (2, 3))
    legend_fontsize = float(config.get("legend_fontsize") or DOTPLOT_LEGEND_FONT_SIZE)
    legend_anchor_y = float(
        config.get("legend_anchor_y", DOTPLOT_LEGEND_TITLE_ANCHOR_Y)
    )
    title = str(config.get("title") or "")

    def legend_item(level: str):
        color = config["colors"][level]
        handle = DrawingArea(7.0, 6.0, 0.0, 0.0)
        handle.add_artist(
            Line2D(
                [3.5],
                [3.0],
                marker="o",
                linestyle="None",
                color=color,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.25,
                markersize=3.4,
            )
        )
        label = str(config["display_labels"].get(level, level))
        return HPacker(
            children=[
                handle,
                TextArea(
                    label,
                    textprops={
                        "fontsize": legend_fontsize,
                        "va": "center",
                    },
                ),
            ],
            align="center",
            pad=0.0,
            sep=2.0,
        )

    rows: list[list[str]] = []
    start = 0
    for row_count in row_counts:
        if start >= len(levels):
            break
        rows.append(levels[start : start + int(row_count)])
        start += int(row_count)
    if start < len(levels):
        fallback_columns = int(config.get("legend_ncol") or 3)
        rows.extend(
            levels[index : index + fallback_columns]
            for index in range(start, len(levels), fallback_columns)
        )

    children = []
    if title:
        children.append(
            TextArea(
                title,
                textprops={
                    "fontsize": 7.0,
                    "ha": "center",
                    "fontweight": "normal",
                },
            )
        )
    children.extend(
        HPacker(
            children=[legend_item(level) for level in row],
            align="center",
            pad=0.0,
            sep=6.0,
        )
        for row in rows
    )
    legend_box = VPacker(children=children, align="center", pad=0.0, sep=1.0)
    ax.add_artist(
        AnchoredOffsetbox(
            loc="upper center",
            child=legend_box,
            pad=0.0,
            borderpad=0.0,
            frameon=False,
            bbox_to_anchor=(0.5, legend_anchor_y),
            bbox_transform=ax.transAxes,
        )
    )


def _draw_characteristic_median_ci_dotplot(
    ax,
    *,
    cohorts: list[str],
    config: dict[str, Any],
    summaries_by_cohort: dict[str, dict[str, dict[str, Any]]],
    y_axis_label: str,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    apply_y_axis: bool = True,
) -> None:
    levels = list(config["levels"])
    offsets = _dotplot_offsets(len(levels))
    x_values = list(range(len(cohorts)))
    _draw_dotplot_human_reference_lines(
        ax,
        cohorts=cohorts,
        config=config,
        summaries_by_cohort=summaries_by_cohort,
    )
    for level, offset in zip(levels, offsets):
        plot_x: list[float] = []
        plot_y: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        for cohort_index, cohort in enumerate(cohorts):
            summary = summaries_by_cohort.get(cohort, {}).get(level, {})
            median_value = summary.get("median")
            if median_value is None:
                continue
            median_float = float(median_value)
            ci_low = summary.get("ci95_low")
            ci_high = summary.get("ci95_high")
            low_float = median_float if ci_low is None else float(ci_low)
            high_float = median_float if ci_high is None else float(ci_high)
            plot_x.append(cohort_index + offset)
            plot_y.append(median_float)
            lower_errors.append(max(0.0, median_float - low_float))
            upper_errors.append(max(0.0, high_float - median_float))
        if not plot_x:
            continue
        color = config["colors"][level]
        ax.errorbar(
            plot_x,
            plot_y,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.75,
            capsize=2.0,
            capthick=0.65,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.25,
            markersize=3.4,
            label=config["display_labels"].get(level, level),
            zorder=3,
        )
    ax.set_xlabel("Cohort" if show_xlabel else "")
    ax.set_ylabel(y_axis_label if show_ylabel else "", labelpad=3.0)
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        display_group_labels(cohorts),
        rotation=0,
        ha="center",
        fontsize=DOTPLOT_TICK_LABEL_SIZE,
    )
    ax.tick_params(axis="y", pad=1.0, labelsize=DOTPLOT_TICK_LABEL_SIZE)
    ax.set_xlim(-0.45, max(0.45, len(cohorts) - 0.55))
    if apply_y_axis:
        _set_dotplot_y_axis(
            ax,
            summaries_by_cohort=summaries_by_cohort,
            cohorts=cohorts,
            levels=levels,
        )
    ax.grid(axis="y", alpha=0.3)
    if config.get("legend_centered_rows"):
        _add_centered_dotplot_legend_rows(ax, config=config)
        return
    legend_ncol = config.get(
        "legend_ncol",
        len(levels) if len(levels) <= 4 else 3,
    )
    legend = ax.legend(
        frameon=False,
        title=config.get("title"),
        title_fontsize=7.0,
        ncol=legend_ncol,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            float(config.get("legend_anchor_y", DOTPLOT_LEGEND_TITLE_ANCHOR_Y)),
        ),
        borderaxespad=0.0,
        columnspacing=config.get("legend_columnspacing", 0.45),
        handlelength=0.8,
        handletextpad=0.25,
        fontsize=config.get("legend_fontsize"),
    )
    if config.get("legend_label_alignment") == "center":
        legend.get_title().set_ha("center")
        for text in legend.get_texts():
            text.set_multialignment("center")
            text.set_ha("center")


def _write_characteristic_median_ci_single_dotplot(
    *,
    output_dir: Path,
    stem: str,
    metric: str,
    cohorts: list[str],
    config: dict[str, Any],
    summaries_by_cohort: dict[str, dict[str, dict[str, Any]]],
    domain_enabled: bool,
) -> None:
    plt, _mdates = require_matplotlib()
    metric_label = _display_metric(metric)
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    _draw_characteristic_median_ci_dotplot(
        ax,
        cohorts=cohorts,
        config=config,
        summaries_by_cohort=summaries_by_cohort,
        y_axis_label=metric_label,
    )
    save_figure(fig, output_dir / DOTPLOT_OUTPUT_DIR, stem)
    write_plot_data(
        output_dir / DOTPLOT_OUTPUT_DIR,
        stem,
        {
            "plot": stem,
            "plot_type": "characteristics_median_ci_dotplot",
            "plot_data_schema": DOTPLOT_PLOT_DATA_SCHEMA,
            "metric": metric,
            "metric_label": metric_label,
            "dimension": config["dimension"],
            "x_axis": "Cohort",
            "y_axis": metric_label,
            "confidence_level": DOTPLOT_CONFIDENCE_LEVEL,
            "domain_enabled": bool(domain_enabled),
            "cohorts": cohorts,
            "levels": config["levels"],
            "display_labels": config["display_labels"],
            "colors": config["colors"],
            "legend_ncol": config.get("legend_ncol"),
            "legend_label_alignment": config.get("legend_label_alignment"),
            "legend_centered_rows": config.get("legend_centered_rows", False),
            "legend_row_counts": config.get("legend_row_counts"),
            "legend_fontsize": config.get("legend_fontsize"),
            "legend_anchor_y": config.get("legend_anchor_y"),
            "show_human_reference_lines": config.get(
                "show_human_reference_lines",
                False,
            ),
            "human_reference_lines": _dotplot_human_reference_lines(
                cohorts=cohorts,
                config=config,
                summaries_by_cohort=summaries_by_cohort,
            ),
            "groups": summaries_by_cohort,
        },
    )
    plt.close(fig)


def _write_characteristic_median_ci_combined_dotplot(
    *,
    output_dir: Path,
    stem: str,
    metric: str,
    cohorts: list[str],
    configs: list[dict[str, Any]],
    summaries_by_dimension: dict[str, dict[str, dict[str, dict[str, Any]]]],
    domain_enabled: bool,
) -> None:
    if len(configs) != 3:
        return
    plt, _mdates = require_matplotlib()
    metric_label = _display_metric(metric)
    figure_width = (
        SMELLDENSITY_DOTPLOT_COMBINED_FIGURE_WIDTH
        if metric == "SmellDensity"
        else DOTPLOT_COMBINED_FIGURE_WIDTH
    )
    render_configs = [dict(config) for config in configs]
    for config in render_configs:
        config["show_human_reference_lines"] = True
    if metric in {"SmellDensity", "RefDensity"}:
        for config in render_configs:
            if config.get("dimension") == "domain":
                config["display_labels"] = DOMAIN_DOTPLOT_SINGLE_LINE_LABELS
                config["legend_ncol"] = 3
                config["legend_label_alignment"] = "center"
                config["legend_centered_rows"] = True
                config["legend_row_counts"] = [2, 3]
    independent_y_axis = True
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(figure_width, DOTPLOT_COMBINED_FIGURE_HEIGHT),
        sharey=not independent_y_axis,
    )
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else list(axes)
    for index, (ax, config) in enumerate(zip(axes_list, render_configs)):
        _draw_characteristic_median_ci_dotplot(
            ax,
            cohorts=cohorts,
            config=config,
            summaries_by_cohort=summaries_by_dimension[config["dimension"]],
            y_axis_label=metric_label,
            show_ylabel=(index == 0),
            show_xlabel=False,
            apply_y_axis=independent_y_axis,
        )
        if not independent_y_axis and index > 0:
            ax.tick_params(axis="y", labelleft=False)
    if not independent_y_axis:
        _set_dotplot_y_axis_from_summaries(
            axes_list[0],
            summaries_list=[
                summaries_by_dimension[config["dimension"]]
                for config in render_configs
            ],
            cohorts=cohorts,
            levels_by_summary=[list(config["levels"]) for config in render_configs],
        )
        bottom, top = axes_list[0].get_ylim()
        panel_y_axis_limits = {
            config["dimension"]: {
                "bottom": float(bottom),
                "top": float(top),
            }
            for config in render_configs
        }
    else:
        panel_y_axis_limits = {
            config["dimension"]: {
                "bottom": float(ax.get_ylim()[0]),
                "top": float(ax.get_ylim()[1]),
            }
            for ax, config in zip(axes_list, render_configs)
        }
    fig.supxlabel("Cohort", y=-0.02)
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.28,
        top=0.947,
        wspace=DOTPLOT_COMBINED_WSPACE,
    )
    save_figure(
        fig,
        output_dir / DOTPLOT_OUTPUT_DIR,
        stem,
        tight_layout_kwargs={"w_pad": DOTPLOT_COMBINED_WSPACE},
    )
    write_plot_data(
        output_dir / DOTPLOT_OUTPUT_DIR,
        stem,
        {
            "plot": stem,
            "plot_type": "characteristics_median_ci_dotplot_grid",
            "plot_data_schema": DOTPLOT_PLOT_DATA_SCHEMA,
            "layout": "1x3",
            "metric": metric,
            "metric_label": metric_label,
            "x_axis": "Cohort",
            "y_axis": metric_label,
            "confidence_level": DOTPLOT_CONFIDENCE_LEVEL,
            "domain_enabled": bool(domain_enabled),
            "cohorts": cohorts,
            "figure": {
                "width_inches": figure_width,
                "height_inches": DOTPLOT_COMBINED_FIGURE_HEIGHT,
                "left": 0.01,
                "right": 0.99,
                "bottom": 0.28,
                "top": 0.947,
                "wspace": DOTPLOT_COMBINED_WSPACE,
                "tight_layout_kwargs": {"w_pad": DOTPLOT_COMBINED_WSPACE},
                "supxlabel_y": -0.02,
                "shared_y_axis": not independent_y_axis,
                "y_axis_scaling": (
                    "per_panel_data_range"
                    if independent_y_axis
                    else "shared_all_panel_data_range"
                ),
            },
            "panels": {
                config["dimension"]: {
                    "dimension": config["dimension"],
                    "title": config["title"],
                    "levels": config["levels"],
                    "display_labels": config["display_labels"],
                    "colors": config["colors"],
                    "legend_ncol": config.get("legend_ncol"),
                    "legend_label_alignment": config.get("legend_label_alignment"),
                    "legend_centered_rows": config.get("legend_centered_rows", False),
                    "legend_row_counts": config.get("legend_row_counts"),
                    "legend_fontsize": config.get("legend_fontsize"),
                    "legend_anchor_y": config.get("legend_anchor_y"),
                    "show_human_reference_lines": config.get(
                        "show_human_reference_lines",
                        False,
                    ),
                    "human_reference_lines": _dotplot_human_reference_lines(
                        cohorts=cohorts,
                        config=config,
                        summaries_by_cohort=summaries_by_dimension[config["dimension"]],
                    ),
                    "y_axis_limits": panel_y_axis_limits.get(config["dimension"]),
                    "groups": summaries_by_dimension[config["dimension"]],
                }
                for config in render_configs
            },
        },
    )
    plt.close(fig)


def _dotplot_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    dimension = str(payload.get("dimension") or "")
    levels = [str(level) for level in payload.get("levels", [])]
    display_labels = payload.get("display_labels")
    colors = payload.get("colors")
    config: dict[str, Any] = {
        "dimension": dimension,
        "title": payload.get("title") or DIMENSION_AXIS_LABELS.get(dimension, dimension),
        "levels": levels,
        "display_labels": display_labels if isinstance(display_labels, dict) else {},
        "colors": colors if isinstance(colors, dict) else {},
        "legend_fontsize": payload.get("legend_fontsize") or DOTPLOT_LEGEND_FONT_SIZE,
    }
    if payload.get("legend_ncol") is not None:
        config["legend_ncol"] = payload.get("legend_ncol")
    if payload.get("legend_label_alignment") is not None:
        config["legend_label_alignment"] = payload.get("legend_label_alignment")
    if payload.get("legend_centered_rows") is not None:
        config["legend_centered_rows"] = payload.get("legend_centered_rows")
    if payload.get("legend_row_counts") is not None:
        config["legend_row_counts"] = payload.get("legend_row_counts")
    if payload.get("legend_anchor_y") is not None:
        config["legend_anchor_y"] = payload.get("legend_anchor_y")
    if payload.get("show_human_reference_lines") is not None:
        config["show_human_reference_lines"] = payload.get("show_human_reference_lines")
    if dimension == "domain":
        config.setdefault("legend_ncol", 3)
        config["legend_label_alignment"] = "center"
        config.setdefault("legend_centered_rows", True)
        config.setdefault("legend_row_counts", [2, 3])
        config.setdefault("legend_anchor_y", DOMAIN_DOTPLOT_LEGEND_TITLE_ANCHOR_Y)
    return config


def render_characteristics_dotplot_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render one characteristic median/CI dotplot from plot data."""
    if payload.get("plot_data_schema") != DOTPLOT_PLOT_DATA_SCHEMA:
        raise ValueError("unsupported characteristics dotplot metadata schema")
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("characteristics dotplot metadata requires groups")
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    _draw_characteristic_median_ci_dotplot(
        ax,
        cohorts=[
            str(cohort)
            for cohort in (
                payload.get("cohorts")
                if isinstance(payload.get("cohorts"), list)
                else order_humans_first(groups)
            )
        ],
        config=_dotplot_config_from_payload(payload),
        summaries_by_cohort=groups,
        y_axis_label=str(payload.get("y_axis") or payload.get("metric_label") or ""),
    )
    save_figure(fig, Path(output_dir), stem)
    plt.close(fig)


def render_characteristics_dotplot_grid_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render a characteristic median/CI dotplot grid from plot data."""
    if payload.get("plot_data_schema") != DOTPLOT_PLOT_DATA_SCHEMA:
        raise ValueError("unsupported characteristics dotplot grid metadata schema")
    panels = payload.get("panels")
    if not isinstance(panels, dict) or not panels:
        raise ValueError("characteristics dotplot grid metadata requires panels")
    figure = payload.get("figure") if isinstance(payload.get("figure"), dict) else {}
    metric = str(payload.get("metric") or "")
    metric_label = str(payload.get("y_axis") or payload.get("metric_label") or metric)
    cohorts = [
        str(cohort)
        for cohort in (
            payload.get("cohorts")
            if isinstance(payload.get("cohorts"), list)
            else []
        )
    ]
    if not cohorts:
        first_panel = next(iter(panels.values()))
        if isinstance(first_panel, dict) and isinstance(first_panel.get("groups"), dict):
            cohorts = order_humans_first(first_panel["groups"])
    panel_items = list(panels.items())
    figure_width = DOTPLOT_COMBINED_FIGURE_WIDTH
    figure_height = DOTPLOT_COMBINED_FIGURE_HEIGHT
    shared_y_axis = False
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, axes = plt.subplots(1, len(panel_items), figsize=(figure_width, figure_height), sharey=shared_y_axis)
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for index, (ax, (_dimension, panel)) in enumerate(zip(axes_list, panel_items)):
        if not isinstance(panel, dict) or not isinstance(panel.get("groups"), dict):
            ax.set_visible(False)
            continue
        config = _dotplot_config_from_payload(panel)
        _draw_characteristic_median_ci_dotplot(
            ax,
            cohorts=cohorts,
            config=config,
            summaries_by_cohort=panel["groups"],
            y_axis_label=metric_label,
            show_ylabel=(index == 0),
            show_xlabel=False,
            apply_y_axis=not shared_y_axis,
        )
        if shared_y_axis and index > 0:
            ax.tick_params(axis="y", labelleft=False)
    if shared_y_axis:
        summaries_list = [
            panel["groups"]
            for _dimension, panel in panel_items
            if isinstance(panel, dict) and isinstance(panel.get("groups"), dict)
        ]
        levels_by_summary = [
            [str(level) for level in panel.get("levels", [])]
            for _dimension, panel in panel_items
            if isinstance(panel, dict)
        ]
        if summaries_list and levels_by_summary:
            _set_dotplot_y_axis_from_summaries(
                axes_list[0],
                summaries_list=summaries_list,
                cohorts=cohorts,
                levels_by_summary=levels_by_summary,
            )
    fig.supxlabel(
        str(payload.get("x_axis") or "Cohort"),
        y=float(figure.get("supxlabel_y", -0.02)),
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=float(figure.get("bottom", 0.28)),
        top=float(figure.get("top", 0.947)),
        wspace=DOTPLOT_COMBINED_WSPACE,
    )
    tight_layout_kwargs = figure.get("tight_layout_kwargs")
    resolved_tight_layout_kwargs = (
        dict(tight_layout_kwargs) if isinstance(tight_layout_kwargs, dict) else {}
    )
    resolved_tight_layout_kwargs["w_pad"] = DOTPLOT_COMBINED_WSPACE
    save_figure(
        fig,
        Path(output_dir),
        stem,
        tight_layout_kwargs=resolved_tight_layout_kwargs,
    )
    plt.close(fig)


def write_characteristics_median_ci_dotplots(
    con,
    *,
    output_dir: Path,
    metric: str,
    include_domain: bool,
    base_where_sql: str = "TRUE",
) -> None:
    """Write median/CI dotplots for characteristic slices from results."""
    apply_ieee_plot_style()
    output_dir = Path(output_dir)
    all_stems = set(
        characteristic_median_ci_dotplot_stems(
            metric=metric,
            include_domain=True,
        )
    )
    kept_stems = set(
        characteristic_median_ci_dotplot_stems(
            metric=metric,
            include_domain=include_domain,
        )
    )
    remove_plot_outputs(output_dir, tuple(sorted(all_stems - kept_stems)))
    cohorts = _cohorts(con, base_where_sql=base_where_sql)
    configs = _dotplot_dimension_configs(include_domain=include_domain)
    summaries_by_dimension: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for config in configs:
        values_by_cohort = _metric_values_by_cohort_dimension(
            con,
            metric=metric,
            dimension=config["dimension"],
            levels=config["levels"],
            cohorts=cohorts,
            base_where_sql=base_where_sql,
        )
        summaries_by_dimension[config["dimension"]] = _dotplot_panel_payload(
            cohorts=cohorts,
            levels=config["levels"],
            values_by_cohort=values_by_cohort,
        )
        _write_characteristic_median_ci_single_dotplot(
            output_dir=output_dir,
            stem=(
                f"{metric_stem(metric)}_by_{config['dimension']}_cohort_"
                "median_ci_dotplot"
            ),
            metric=metric,
            cohorts=cohorts,
            config=config,
            summaries_by_cohort=summaries_by_dimension[config["dimension"]],
            domain_enabled=include_domain,
        )
    if include_domain:
        _write_characteristic_median_ci_combined_dotplot(
            output_dir=output_dir,
            stem=f"{metric_stem(metric)}_characteristics_by_cohort_median_ci_dotplots",
            metric=metric,
            cohorts=cohorts,
            configs=configs,
            summaries_by_dimension=summaries_by_dimension,
            domain_enabled=include_domain,
        )


def write_characteristics_median_ci_dotplots_from_payload(
    *,
    characteristic_metric_values: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    output_dir: Path,
    metric: str,
    include_domain: bool,
) -> None:
    """Write median/CI dotplots from streaming characteristic metric payloads."""
    apply_ieee_plot_style()
    output_dir = Path(output_dir)
    all_stems = set(
        characteristic_median_ci_dotplot_stems(
            metric=metric,
            include_domain=True,
        )
    )
    kept_stems = set(
        characteristic_median_ci_dotplot_stems(
            metric=metric,
            include_domain=include_domain,
        )
    )
    remove_plot_outputs(output_dir, tuple(sorted(all_stems - kept_stems)))
    configs = _dotplot_dimension_configs(include_domain=include_domain)
    cohorts = order_humans_first(
        {
            scope[len("cohort:") :]
            for by_level in characteristic_metric_values.values()
            for by_metric in by_level.values()
            for by_scope in by_metric.values()
            for scope, values in by_scope.items()
            if scope.startswith("cohort:") and values
        }
    )
    summaries_by_dimension: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for config in configs:
        dimension = str(config["dimension"])
        values_by_cohort = {
            cohort: {level: [] for level in config["levels"]}
            for cohort in cohorts
        }
        for level in config["levels"]:
            scoped = (
                characteristic_metric_values
                .get(dimension, {})
                .get(level, {})
                .get(metric, {})
            )
            for cohort in cohorts:
                values_by_cohort[cohort][level] = [
                    float(value)
                    for value in scoped.get(f"cohort:{cohort}", [])
                ]
        summaries_by_dimension[dimension] = _dotplot_panel_payload(
            cohorts=cohorts,
            levels=config["levels"],
            values_by_cohort=values_by_cohort,
        )
        _write_characteristic_median_ci_single_dotplot(
            output_dir=output_dir,
            stem=(
                f"{metric_stem(metric)}_by_{dimension}_cohort_"
                "median_ci_dotplot"
            ),
            metric=metric,
            cohorts=cohorts,
            config=config,
            summaries_by_cohort=summaries_by_dimension[dimension],
            domain_enabled=include_domain,
        )
    if include_domain:
        _write_characteristic_median_ci_combined_dotplot(
            output_dir=output_dir,
            stem=f"{metric_stem(metric)}_characteristics_by_cohort_median_ci_dotplots",
            metric=metric,
            cohorts=cohorts,
            configs=configs,
            summaries_by_dimension=summaries_by_dimension,
            domain_enabled=include_domain,
        )


def _summary_payload(values: list[float]) -> dict[str, Any]:
    return {
        **numeric_distribution_summary(values),
        "n": len(values),
    }


def _compact_test_payload(test: dict[str, Any]) -> dict[str, Any]:
    return {
        "p": test.get("adjusted_p_value"),
        "p_value": test.get("p_value"),
        "adjusted_p_value": test.get("adjusted_p_value"),
        "fdr_method": test.get("fdr_method"),
        "significant_after_fdr": test.get("significant_after_fdr"),
        "cliffs_delta": test.get("cliffs_delta"),
        "cliffs_delta_ci95_low": test.get("cliffs_delta_ci95_low"),
        "cliffs_delta_ci95_high": test.get("cliffs_delta_ci95_high"),
        "first_group": test.get("first_group"),
        "second_group": test.get("second_group"),
    }


def _dimension_test(
    results: dict[str, Any],
    *,
    dimension: str,
    level: str,
    metric: str,
    comparison: str,
) -> dict[str, Any]:
    test = (
        results.get("dimensions", {})
        .get(dimension, {})
        .get(level, {})
        .get(metric, {})
        .get(comparison, {})
        .get("mann_whitney_u", {})
    )
    return _compact_test_payload(test)


def _dimension_cohort_test(
    results: dict[str, Any],
    *,
    dimension: str,
    level: str,
    metric: str,
    cohort: str,
) -> dict[str, Any]:
    test = (
        results.get("dimensions", {})
        .get(dimension, {})
        .get(level, {})
        .get(metric, {})
        .get("cohort_vs_humans", {})
        .get(cohort, {})
        .get("mann_whitney_u", {})
    )
    return _compact_test_payload(test)


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


def _apply_boxplot_lower_bound(
    ax,
    metric: str,
    values_by_group,
    y_axis_limits: dict[str, object] | None,
) -> dict[str, object] | None:
    visible_upper = ax.get_ylim()[1]
    ax.set_ylim(
        bottom=(
            CHARACTERISTICS_BOXPLOT_AXIS_LOWER_BOUND
            - CHARACTERISTICS_BOXPLOT_AXIS_LOWER_PADDING
        ),
        top=max(
            float(visible_upper),
            CHARACTERISTICS_BOXPLOT_AXIS_LOWER_BOUND
            + CHARACTERISTICS_BOXPLOT_MINIMUM_UPPER_PADDING,
        ),
    )
    if y_axis_limits is None:
        return None
    below_count = sum(
        1
        for value in _flatten_numeric_groups(values_by_group)
        if float(value) < CHARACTERISTICS_BOXPLOT_AXIS_LOWER_BOUND
    )
    return {
        **y_axis_limits,
        "lower": CHARACTERISTICS_BOXPLOT_AXIS_LOWER_BOUND,
        "visual_lower": (
            CHARACTERISTICS_BOXPLOT_AXIS_LOWER_BOUND
            - CHARACTERISTICS_BOXPLOT_AXIS_LOWER_PADDING
        ),
        "below_count": below_count,
        "is_clipped": bool(
            below_count or int(y_axis_limits.get("above_count", 0))
        ),
    }


def _plot_popularity_boxplot_by_authorship(
    con,
    *,
    metric: str,
    base_where_sql: str,
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    positions: list[float] = []
    values_by_position: list[list[float]] = []
    colors: list[str] = []
    plot_buckets: dict[str, Any] = {}
    for index, bucket in enumerate(POPULARITY_BUCKET_ORDER, start=1):
        human_values = _values(
            con,
            table_name="analysis_characteristic_prs",
            metric=metric,
            where_sql=(
                f"{base_where_sql} "
                f"AND popularity_group = {sql_string_literal(bucket)} "
                "AND authorship_group = 'human'"
            ),
        )
        agent_values = _values(
            con,
            table_name="analysis_characteristic_prs",
            metric=metric,
            where_sql=(
                f"{base_where_sql} "
                f"AND popularity_group = {sql_string_literal(bucket)} "
                "AND authorship_group = 'agent'"
            ),
        )
        positions.extend([index - 0.17, index + 0.17])
        values_by_position.extend([human_values, agent_values])
        colors.extend([AUTHORSHIP_COLORS["human"], AUTHORSHIP_COLORS["agent"]])
        plot_buckets[bucket] = {
            "human": _summary_payload(human_values),
            "agent": _summary_payload(agent_values),
            "agents_vs_humans": _dimension_test(
                results,
                dimension="popularity",
                level=bucket,
                metric=metric,
                comparison="agents_vs_humans",
            ),
        }
    if positions:
        add_violin_underlay(
            ax,
            values_by_position,
            positions=positions,
            colors=colors,
            width=0.23,
        )
        boxplot = ax.boxplot(
            values_by_position,
            positions=positions,
            widths=0.25,
            **ieee_boxplot_kwargs(),
        )
        style_ieee_boxplot(boxplot, colors)
        grouped_for_baseline = {
            "human": [
                value
                for bucket in POPULARITY_BUCKET_ORDER
                for value in _values(
                    con,
                    table_name="analysis_characteristic_prs",
                    metric=metric,
                    where_sql=(
                        f"{base_where_sql} "
                        f"AND popularity_group = {sql_string_literal(bucket)} "
                        "AND authorship_group = 'human'"
                    ),
                )
            ]
        }
        add_human_median_baseline(ax, grouped_for_baseline)
        y_axis_limits = apply_percentile_capped_y_axis(
            ax,
            values_by_position,
            show_note=False,
        )
    else:
        y_axis_limits = None
    ax.set_xticks(range(1, len(POPULARITY_BUCKET_ORDER) + 1))
    ax.set_xticklabels(["Low", "Medium", "High"], rotation=0, fontsize=6.0)
    ax.set_xlabel("Popularity")
    ax.set_ylabel(_display_metric(metric))
    ax.grid(axis="y", alpha=0.3)
    _set_y_axis_for_values(ax, values_by_position)
    y_axis_limits = _apply_boxplot_lower_bound(
        ax,
        metric,
        values_by_position,
        y_axis_limits,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=AUTHORSHIP_COLORS["human"],
            markersize=5,
            label="Humans",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=AUTHORSHIP_COLORS["agent"],
            markersize=5,
            label="Agents",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    stem = f"{metric_stem(metric)}_popularity_boxplot_by_authorship"
    save_figure(fig, output_dir / "boxplots", stem)
    write_plot_data(
        output_dir / "boxplots",
        stem,
        {
            "plot": stem,
            "plot_type": "boxplot",
            "metric": metric,
            "dimension": "popularity",
            "x_axis": "Popularity",
            "y_axis": _display_metric(metric),
            "percentile_capped_y_axis": y_axis_limits,
            "buckets": plot_buckets,
        },
    )
    plt.close(fig)


def _plot_effect_size_heatmap(
    results: dict[str, Any],
    *,
    dimension: str,
    metrics: tuple[str, ...],
    levels: tuple[str, ...],
    level_labels: dict[str, str],
    output_dir: Path,
    stem: str,
) -> None:
    plt, _mdates = require_matplotlib()
    matrix: list[list[float]] = []
    delta_matrix: list[list[float | None]] = []
    p_value_matrix: list[list[float | None]] = []
    adjusted_p_value_matrix: list[list[float | None]] = []
    significant_matrix: list[list[bool | None]] = []
    delta_ci_low_matrix: list[list[float | None]] = []
    delta_ci_high_matrix: list[list[float | None]] = []
    labels: list[list[str]] = []
    label_colors: list[list[str]] = []
    cells: list[dict[str, Any]] = []
    for level in levels:
        level_payload = results.get("dimensions", {}).get(dimension, {}).get(level, {})
        row: list[float] = []
        delta_row: list[float | None] = []
        p_value_row: list[float | None] = []
        adjusted_p_value_row: list[float | None] = []
        significant_row: list[bool | None] = []
        delta_ci_low_row: list[float | None] = []
        delta_ci_high_row: list[float | None] = []
        label_row: list[str] = []
        label_color_row: list[str] = []
        for metric in metrics:
            comparison = (
                level_payload.get(metric, {})
                .get("agents_vs_humans", {})
            )
            test = comparison.get("mann_whitney_u", {})
            heatmap_test = _heatmap_test_payload(test)
            delta = heatmap_test.get("cliffs_delta")
            p_value = heatmap_test.get("p_value")
            adjusted_p_value = heatmap_test.get("adjusted_p_value")
            significant_after_fdr = test.get("significant_after_fdr")
            delta_ci_low = heatmap_test.get("cliffs_delta_ci95_low")
            delta_ci_high = heatmap_test.get("cliffs_delta_ci95_high")
            delta_value = _finite_float_or_none(delta)
            row.append(float("nan") if delta is None else float(delta))
            delta_row.append(delta_value)
            p_value_row.append(p_value)
            adjusted_p_value_row.append(adjusted_p_value)
            significant_row.append(significant_after_fdr)
            delta_ci_low_row.append(delta_ci_low)
            delta_ci_high_row.append(delta_ci_high)
            annotation_text = _heatmap_annotation_text(
                delta_value,
                adjusted_p_value,
            )
            annotation_color = (
                "white"
                if delta_value is not None and abs(delta_value) >= 0.55
                else "black"
            )
            label_row.append(annotation_text)
            label_color_row.append(annotation_color)
            cells.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "level_label": level_labels.get(level, level),
                    "metric": metric,
                    "metric_label": _display_metric(metric),
                    "row_index": len(delta_matrix),
                    "column_index": len(delta_row) - 1,
                    "value_field": "cliffs_delta",
                    "cliffs_delta": delta_value,
                    "p": adjusted_p_value,
                    "p_value": p_value,
                    "adjusted_p_value": adjusted_p_value,
                    "fdr_method": test.get("fdr_method"),
                    "significant_after_fdr": significant_after_fdr,
                    "significance_star": _heatmap_significance_star(
                        adjusted_p_value
                    ),
                    "cliffs_delta_ci95_low": delta_ci_low,
                    "cliffs_delta_ci95_high": delta_ci_high,
                    "annotation_text": annotation_text,
                    "annotation_text_color": annotation_color,
                    "comparison": {
                        "compared_group": "agent",
                        "baseline_group": "human",
                        "positive_delta_interpretation": (
                            "agents tend higher than humans"
                        ),
                        "statistics": comparison.get("statistics", {}),
                        "mann_whitney_u": heatmap_test,
                    },
                }
            )
        matrix.append(row)
        delta_matrix.append(delta_row)
        p_value_matrix.append(p_value_row)
        adjusted_p_value_matrix.append(adjusted_p_value_row)
        significant_matrix.append(significant_row)
        delta_ci_low_matrix.append(delta_ci_low_row)
        delta_ci_high_matrix.append(delta_ci_high_row)
        labels.append(label_row)
        label_colors.append(label_color_row)
    fig_height = max(1.75, 0.36 * len(levels) + 0.9)
    fig, ax = plt.subplots(figsize=(HEATMAP_FIGURE_WIDTH, fig_height))
    image = ax.imshow(
        matrix,
        cmap=HEATMAP_COLORMAP,
        vmin=HEATMAP_COLOR_MIN,
        vmax=HEATMAP_COLOR_MAX,
        aspect="auto",
    )
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(
        [_display_metric(metric) for metric in metrics],
        rotation=HEATMAP_XTICK_ROTATION_DEGREES,
        ha="right",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(
        [
            _display_level_label(dimension, level_labels.get(level, level))
            for level in levels
        ],
        rotation=HEATMAP_YTICK_ROTATION_DEGREES,
        ha="right",
        va="center",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    for y_index, row in enumerate(labels):
        for x_index, label in enumerate(row):
            ax.text(
                x_index,
                y_index,
                label,
                ha="center",
                va="center",
                fontsize=HEATMAP_ANNOTATION_FONT_SIZE,
                color=label_colors[y_index][x_index],
            )
    ax.set_xlabel(
        "Metric",
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
        labelpad=HEATMAP_X_AXIS_LABEL_PAD,
    )
    _align_heatmap_metric_label(ax)
    ax.set_ylabel(
        DIMENSION_AXIS_LABELS[dimension],
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(HEATMAP_COLOR_LABEL, fontsize=HEATMAP_COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_LABEL_SIZE)
    _add_heatmap_significance_caption(ax)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.32)
    save_figure(fig, output_dir / "heatmaps", stem)
    write_plot_data(
        output_dir / "heatmaps",
        stem,
        {
            "plot": stem,
            "plot_data_schema": "characteristics_effect_size_heatmap/v2",
            "plot_type": "effect_size_heatmap",
            "dimension": dimension,
            "x_axis": "Metric",
            "y_axis": DIMENSION_AXIS_LABELS[dimension],
            "color": HEATMAP_COLOR_LABEL,
            "color_encoding": {
                "field": "cliffs_delta",
                "label": HEATMAP_COLOR_LABEL,
                "colormap": HEATMAP_COLORMAP,
                "minimum": HEATMAP_COLOR_MIN,
                "maximum": HEATMAP_COLOR_MAX,
                "center": HEATMAP_COLOR_CENTER,
                "positive_interpretation": "agents tend higher than humans",
                "negative_interpretation": "agents tend lower than humans",
            },
            "annotation_encoding": {
                "lines": ["cliffs_delta"],
                "number_format": "{:.2f}",
                "significance_marker": "*",
                "significance_field": "adjusted_p_value",
                "significance_threshold": HEATMAP_SIGNIFICANCE_P_THRESHOLD,
                "significance_caption": HEATMAP_SIGNIFICANCE_CAPTION,
                "missing_value_label": "NA",
            },
            "comparison": {
                "compared_group": "agent",
                "baseline_group": "human",
                "test": "mann_whitney_u",
                "effect_size": "cliffs_delta",
                "p_value_display_field": "adjusted_p_value",
                "fdr_method": "benjamini_hochberg",
            },
            "figure": {
                "width_inches": HEATMAP_FIGURE_WIDTH,
                "height_inches": fig_height,
                "x_axis_label_y": HEATMAP_X_AXIS_LABEL_Y,
                "significance_caption_position": {
                    "x": HEATMAP_SIGNIFICANCE_CAPTION_X,
                    "y": HEATMAP_SIGNIFICANCE_CAPTION_Y,
                    "coordinate_system": "data_x_axes_y",
                    "horizontal_alignment": "left",
                    "vertical_alignment": "center",
                },
            },
            "metrics": list(metrics),
            "metric_columns": [
                {
                    "index": index,
                    "metric": metric,
                    "metric_label": _display_metric(metric),
                }
                for index, metric in enumerate(metrics)
            ],
            "levels": list(levels),
            "level_rows": [
                {
                    "index": index,
                    "level": level,
                    "level_label": level_labels.get(level, level),
                    "display_level_label": _display_level_label(
                        dimension,
                        level_labels.get(level, level),
                    ),
                }
                for index, level in enumerate(levels)
            ],
            "matrix": {
                "cliffs_delta": delta_matrix,
                "p_value": p_value_matrix,
                "adjusted_p_value": adjusted_p_value_matrix,
                "significant_after_fdr": significant_matrix,
                "cliffs_delta_ci95_low": delta_ci_low_matrix,
                "cliffs_delta_ci95_high": delta_ci_high_matrix,
                "annotation_text": labels,
                "annotation_text_color": label_colors,
            },
            "cells": cells,
        },
    )
    plt.close(fig)


def _heatmap_json_path(
    output_dir: Path,
    dimension: str,
    *,
    stem_prefix: str = "",
    stem_suffix: str = "",
) -> Path:
    return (
        Path(output_dir)
        / "heatmaps"
        / f"{stem_prefix}{dimension}_effect_size_heatmap{stem_suffix}.json"
    )


def _extended_heatmap_stem(dimension: str, stem_suffix: str = "") -> str:
    return f"{dimension}_effect_size_heatmap_extended{stem_suffix}"


def _load_reusable_heatmap_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("plot_data_schema") != HEATMAP_PLOT_DATA_SCHEMA:
        return None
    if payload.get("plot_type") != "effect_size_heatmap":
        return None
    if not isinstance(payload.get("metric_columns"), list):
        return None
    if not isinstance(payload.get("level_rows"), list):
        return None
    if not isinstance(payload.get("cells"), list):
        return None
    return payload


def _heatmap_levels(payload: dict[str, Any]) -> list[str]:
    return [str(row.get("level")) for row in payload.get("level_rows", [])]


def _heatmap_metric_columns(
    payload: dict[str, Any],
    *,
    analysis_group: str,
    start_index: int,
) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for offset, column in enumerate(payload.get("metric_columns", [])):
        metric = str(column.get("metric"))
        if metric in HEATMAP_EXCLUDED_METRICS:
            continue
        columns.append(
            {
                "index": start_index + offset,
                "source_index": column.get("index"),
                "source_analysis_group": analysis_group,
                "metric": metric,
                "metric_label": column.get("metric_label") or _display_metric(metric),
            }
        )
    return columns


def _heatmap_cells_by_level_metric(
    payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in payload.get("cells", []):
        level = str(cell.get("level"))
        metric = str(cell.get("metric"))
        cells[(level, metric)] = dict(cell)
    return cells


def _extended_annotation_color(delta: float | None) -> str:
    return "white" if delta is not None and abs(delta) >= 0.55 else "black"


def _extended_cell_payload(
    *,
    source_cell: dict[str, Any] | None,
    source_analysis_group: str,
    row_index: int,
    column_index: int,
    level: str,
    level_label: str,
    metric: str,
    metric_label: str,
    dimension: str,
) -> dict[str, Any]:
    source_cell = source_cell or {}
    delta = _finite_float_or_none(source_cell.get("cliffs_delta"))
    p_value = _finite_float_or_none(source_cell.get("p_value"))
    adjusted_p_value = _finite_float_or_none(source_cell.get("adjusted_p_value"))
    delta_ci_low = _finite_float_or_none(source_cell.get("cliffs_delta_ci95_low"))
    delta_ci_high = _finite_float_or_none(source_cell.get("cliffs_delta_ci95_high"))
    annotation_text = _heatmap_annotation_text(delta, adjusted_p_value)
    return {
        **source_cell,
        "dimension": dimension,
        "level": level,
        "level_label": level_label,
        "display_level_label": _display_level_label(dimension, level_label),
        "metric": metric,
        "metric_label": metric_label,
        "row_index": row_index,
        "column_index": column_index,
        "source_analysis_group": source_analysis_group,
        "value_field": "cliffs_delta",
        "cliffs_delta": delta,
        "p": adjusted_p_value,
        "p_value": p_value,
        "adjusted_p_value": adjusted_p_value,
        "significance_star": _heatmap_significance_star(adjusted_p_value),
        "cliffs_delta_ci95_low": delta_ci_low,
        "cliffs_delta_ci95_high": delta_ci_high,
        "annotation_text": annotation_text,
        "annotation_text_color": _extended_annotation_color(delta),
    }


def _write_extended_heatmap_from_payloads(
    *,
    primary_payload: dict[str, Any],
    companion_payload: dict[str, Any],
    primary_path: Path,
    companion_path: Path,
    output_dir: Path,
    dimension: str,
    primary_analysis_group: str,
    companion_analysis_group: str,
    stem_suffix: str = "",
) -> bool:
    if primary_payload.get("dimension") != dimension:
        return False
    if companion_payload.get("dimension") != dimension:
        return False
    primary_levels = _heatmap_levels(primary_payload)
    companion_levels = _heatmap_levels(companion_payload)
    if not primary_levels or primary_levels != companion_levels:
        return False

    level_rows = [
        {
            "index": index,
            "level": str(row.get("level")),
            "level_label": row.get("level_label") or str(row.get("level")),
            "display_level_label": _display_level_label(
                dimension,
                row.get("level_label") or str(row.get("level")),
            ),
        }
        for index, row in enumerate(primary_payload.get("level_rows", []))
    ]
    primary_columns = _heatmap_metric_columns(
        primary_payload,
        analysis_group=primary_analysis_group,
        start_index=0,
    )
    companion_columns = _heatmap_metric_columns(
        companion_payload,
        analysis_group=companion_analysis_group,
        start_index=0,
    )
    if not primary_columns or not companion_columns:
        return False
    columns_by_group = {
        primary_analysis_group: primary_columns,
        companion_analysis_group: companion_columns,
    }
    requested_group_order = ["Refactoring", "Maintainability"]
    ordered_groups = [
        group
        for requested in requested_group_order
        for group in columns_by_group
        if str(group).casefold() == requested.casefold()
    ]
    ordered_groups.extend(
        group for group in columns_by_group if group not in ordered_groups
    )
    metric_columns = [
        {**column, "index": index}
        for index, column in enumerate(
            column
            for group in ordered_groups
            for column in columns_by_group[group]
        )
    ]

    source_cells = {
        primary_analysis_group: _heatmap_cells_by_level_metric(primary_payload),
        companion_analysis_group: _heatmap_cells_by_level_metric(companion_payload),
    }
    delta_matrix: list[list[float | None]] = []
    p_value_matrix: list[list[float | None]] = []
    adjusted_p_value_matrix: list[list[float | None]] = []
    significant_matrix: list[list[bool | None]] = []
    delta_ci_low_matrix: list[list[float | None]] = []
    delta_ci_high_matrix: list[list[float | None]] = []
    labels: list[list[str]] = []
    label_colors: list[list[str]] = []
    plot_matrix: list[list[float]] = []
    cells: list[dict[str, Any]] = []

    for row_index, level_row in enumerate(level_rows):
        level = str(level_row["level"])
        level_label = str(level_row["level_label"])
        delta_row: list[float | None] = []
        p_value_row: list[float | None] = []
        adjusted_p_value_row: list[float | None] = []
        significant_row: list[bool | None] = []
        delta_ci_low_row: list[float | None] = []
        delta_ci_high_row: list[float | None] = []
        label_row: list[str] = []
        label_color_row: list[str] = []
        plot_row: list[float] = []
        for column in metric_columns:
            analysis_group = str(column["source_analysis_group"])
            metric = str(column["metric"])
            metric_label = str(column["metric_label"])
            source_cell = source_cells[analysis_group].get((level, metric))
            cell = _extended_cell_payload(
                source_cell=source_cell,
                source_analysis_group=analysis_group,
                row_index=row_index,
                column_index=int(column["index"]),
                level=level,
                level_label=level_label,
                metric=metric,
                metric_label=metric_label,
                dimension=dimension,
            )
            delta = cell["cliffs_delta"]
            plot_row.append(float("nan") if delta is None else float(delta))
            delta_row.append(delta)
            p_value_row.append(cell["p_value"])
            adjusted_p_value_row.append(cell["adjusted_p_value"])
            significant_row.append(cell.get("significant_after_fdr"))
            delta_ci_low_row.append(cell["cliffs_delta_ci95_low"])
            delta_ci_high_row.append(cell["cliffs_delta_ci95_high"])
            label_row.append(str(cell["annotation_text"]))
            label_color_row.append(str(cell["annotation_text_color"]))
            cells.append(cell)
        plot_matrix.append(plot_row)
        delta_matrix.append(delta_row)
        p_value_matrix.append(p_value_row)
        adjusted_p_value_matrix.append(adjusted_p_value_row)
        significant_matrix.append(significant_row)
        delta_ci_low_matrix.append(delta_ci_low_row)
        delta_ci_high_matrix.append(delta_ci_high_row)
        labels.append(label_row)
        label_colors.append(label_color_row)

    plt, _mdates = require_matplotlib()
    figure_width = HEATMAP_FIGURE_WIDTH
    figure_height = max(1.52, 0.288 * len(level_rows) + 0.88)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(
        plot_matrix,
        cmap=HEATMAP_COLORMAP,
        vmin=HEATMAP_COLOR_MIN,
        vmax=HEATMAP_COLOR_MAX,
        aspect="auto",
    )
    ax.set_xticks(range(len(metric_columns)))
    ax.set_xticklabels(
        [str(column["metric_label"]) for column in metric_columns],
        rotation=HEATMAP_XTICK_ROTATION_DEGREES,
        ha="right",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    ax.set_yticks(range(len(level_rows)))
    ax.set_yticklabels(
        [
            _display_level_label(dimension, row["level_label"])
            for row in level_rows
        ],
        rotation=HEATMAP_YTICK_ROTATION_DEGREES,
        ha="right",
        va="center",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    for y_index, row in enumerate(labels):
        for x_index, label in enumerate(row):
            ax.text(
                x_index,
                y_index,
                label,
                ha="center",
                va="center",
                fontsize=(
                    EXTENDED_HEATMAP_COMPACT_ANNOTATION_FONT_SIZE
                    if len(metric_columns) > 8
                    else EXTENDED_HEATMAP_ANNOTATION_FONT_SIZE
                ),
                color=label_colors[y_index][x_index],
                linespacing=0.9,
            )
    group_spans = []
    next_start_column = 0
    for group in ordered_groups:
        group_column_count = len(columns_by_group[group])
        group_spans.append(
            {
                "analysis_group": group,
                "start_column": next_start_column,
                "end_column": next_start_column + group_column_count - 1,
            }
        )
        next_start_column += group_column_count
    if len(group_spans) > 1:
        for span in group_spans[:-1]:
            ax.axvline(
                span["end_column"] + 0.5,
                color="black",
                linewidth=HEATMAP_SECTION_SEPARATOR_LINEWIDTH,
                alpha=HEATMAP_SECTION_SEPARATOR_ALPHA,
            )
    for span in group_spans:
        center = (span["start_column"] + span["end_column"]) / 2.0
        ax.text(
            center,
            EXTENDED_HEATMAP_COLUMN_GROUP_LABEL_Y,
            str(span["analysis_group"]),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=EXTENDED_HEATMAP_COLUMN_GROUP_LABEL_SIZE,
            fontweight="bold",
            clip_on=False,
        )
    ax.set_xlabel(
        "Metric",
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
        labelpad=HEATMAP_X_AXIS_LABEL_PAD,
    )
    _align_heatmap_metric_label(ax)
    ax.set_ylabel(
        DIMENSION_AXIS_LABELS[dimension],
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.025)
    colorbar.set_label(HEATMAP_COLOR_LABEL, fontsize=HEATMAP_COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_LABEL_SIZE)
    _add_heatmap_significance_caption(ax)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.35, top=0.86)
    stem = _extended_heatmap_stem(dimension, stem_suffix)
    save_figure(fig, output_dir / "heatmaps", stem)
    write_plot_data(
        output_dir / "heatmaps",
        stem,
        {
            "plot": stem,
            "plot_data_schema": EXTENDED_HEATMAP_PLOT_DATA_SCHEMA,
            "plot_type": "extended_effect_size_heatmap",
            "dimension": dimension,
            "x_axis": "Metric",
            "y_axis": DIMENSION_AXIS_LABELS[dimension],
            "color": HEATMAP_COLOR_LABEL,
            "color_encoding": {
                "field": "cliffs_delta",
                "label": HEATMAP_COLOR_LABEL,
                "colormap": HEATMAP_COLORMAP,
                "minimum": HEATMAP_COLOR_MIN,
                "maximum": HEATMAP_COLOR_MAX,
                "center": HEATMAP_COLOR_CENTER,
                "positive_interpretation": "agents tend higher than humans",
                "negative_interpretation": "agents tend lower than humans",
            },
            "annotation_encoding": {
                "lines": ["cliffs_delta"],
                "number_format": "{:.2f}",
                "significance_marker": "*",
                "significance_field": "adjusted_p_value",
                "significance_threshold": HEATMAP_SIGNIFICANCE_P_THRESHOLD,
                "significance_caption": HEATMAP_SIGNIFICANCE_CAPTION,
                "missing_value_label": "NA",
            },
            "comparison": {
                "compared_group": "agent",
                "baseline_group": "human",
                "test": "mann_whitney_u",
                "effect_size": "cliffs_delta",
                "p_value_display_field": "adjusted_p_value",
                "fdr_method": "benjamini_hochberg",
            },
            "source_heatmaps": [
                {
                    "analysis_group": primary_analysis_group,
                    "path": primary_path,
                    "plot": primary_payload.get("plot"),
                    "metrics": [
                        column["metric"]
                        for column in primary_columns
                    ],
                },
                {
                    "analysis_group": companion_analysis_group,
                    "path": companion_path,
                    "plot": companion_payload.get("plot"),
                    "metrics": [
                        column["metric"]
                        for column in companion_columns
                    ],
                },
            ],
            "column_groups": group_spans,
            "figure": {
                "width_inches": figure_width,
                "height_inches": figure_height,
                "x_axis_label_y": HEATMAP_X_AXIS_LABEL_Y,
                "significance_caption_position": {
                    "x": HEATMAP_SIGNIFICANCE_CAPTION_X,
                    "y": HEATMAP_SIGNIFICANCE_CAPTION_Y,
                    "coordinate_system": "data_x_axes_y",
                    "horizontal_alignment": "left",
                    "vertical_alignment": "center",
                },
            },
            "metrics": [column["metric"] for column in metric_columns],
            "metric_columns": metric_columns,
            "levels": [row["level"] for row in level_rows],
            "level_rows": level_rows,
            "matrix": {
                "cliffs_delta": delta_matrix,
                "p_value": p_value_matrix,
                "adjusted_p_value": adjusted_p_value_matrix,
                "significant_after_fdr": significant_matrix,
                "cliffs_delta_ci95_low": delta_ci_low_matrix,
                "cliffs_delta_ci95_high": delta_ci_high_matrix,
                "annotation_text": labels,
                "annotation_text_color": label_colors,
            },
            "cells": cells,
        },
    )
    plt.close(fig)
    return True


def _extended_heatmap_json_path(
    output_dir: Path,
    dimension: str,
    *,
    stem_suffix: str = "",
) -> Path:
    return (
        Path(output_dir)
        / "heatmaps"
        / f"{_extended_heatmap_stem(dimension, stem_suffix)}.json"
    )


def _load_extended_heatmap_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("plot_data_schema") != EXTENDED_HEATMAP_PLOT_DATA_SCHEMA:
        return None
    if payload.get("plot_type") != "extended_effect_size_heatmap":
        return None
    if not isinstance(payload.get("metric_columns"), list):
        return None
    if not isinstance(payload.get("level_rows"), list):
        return None
    matrix = payload.get("matrix")
    if not isinstance(matrix, dict):
        return None
    if not isinstance(matrix.get("cliffs_delta"), list):
        return None
    if not isinstance(matrix.get("annotation_text"), list):
        return None
    if not isinstance(matrix.get("annotation_text_color"), list):
        return None
    return payload


def _extended_heatmap_metric_signature(payload: dict[str, Any]) -> list[str]:
    return [
        str(column.get("metric"))
        for column in payload.get("metric_columns", [])
    ]


def _matrix_for_plot(payload: dict[str, Any]) -> list[list[float]]:
    matrix = payload.get("matrix", {})
    rows = matrix.get("cliffs_delta", [])
    plot_rows: list[list[float]] = []
    for row in rows:
        plot_row = []
        for value in row:
            if value is None:
                plot_row.append(float("nan"))
            else:
                try:
                    plot_row.append(float(value))
                except (TypeError, ValueError):
                    plot_row.append(float("nan"))
        plot_rows.append(plot_row)
    return plot_rows


def _annotate_heatmap_axis(
    ax,
    labels: list[list[Any]],
    label_colors: list[list[Any]],
    *,
    column_count: int,
) -> None:
    font_size = (
        EXTENDED_HEATMAP_COMPACT_ANNOTATION_FONT_SIZE
        if column_count > 8
        else EXTENDED_HEATMAP_ANNOTATION_FONT_SIZE
    )
    for y_index, row in enumerate(labels):
        for x_index, label in enumerate(row):
            ax.text(
                x_index,
                y_index,
                str(label),
                ha="center",
                va="center",
                fontsize=font_size,
                color=str(label_colors[y_index][x_index]),
                linespacing=0.9,
            )


def _draw_column_group_spans(
    ax,
    column_groups: list[dict[str, Any]],
    *,
    show_labels: bool,
) -> None:
    for span in column_groups[:-1]:
        try:
            end_column = float(span["end_column"])
        except (KeyError, TypeError, ValueError):
            continue
        ax.axvline(
            end_column + 0.5,
            color="black",
            linewidth=HEATMAP_SECTION_SEPARATOR_LINEWIDTH,
            alpha=HEATMAP_SECTION_SEPARATOR_ALPHA,
        )
    if not show_labels:
        return
    for span in column_groups:
        try:
            start_column = float(span["start_column"])
            end_column = float(span["end_column"])
        except (KeyError, TypeError, ValueError):
            continue
        ax.text(
            (start_column + end_column) / 2.0,
            EXTENDED_HEATMAP_COLUMN_GROUP_LABEL_Y,
            str(span.get("analysis_group", "")),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=EXTENDED_HEATMAP_COLUMN_GROUP_LABEL_SIZE,
            fontweight="bold",
            clip_on=False,
        )


def _heatmap_metric_labels(payload: dict[str, Any]) -> list[str]:
    columns = payload.get("metric_columns")
    if not isinstance(columns, list):
        raise ValueError("heatmap metadata requires metric_columns")
    return [
        str(column.get("metric_label") or column.get("metric"))
        for column in columns
        if isinstance(column, dict)
    ]


def _heatmap_level_labels(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("level_rows")
    if not isinstance(rows, list):
        raise ValueError("heatmap metadata requires level_rows")
    return [
        str(
            row.get("display_level_label")
            or row.get("level_label")
            or row.get("level")
        )
        for row in rows
        if isinstance(row, dict)
    ]


def _heatmap_annotation_payload(payload: dict[str, Any]) -> tuple[list[list[Any]], list[list[Any]]]:
    matrix = payload.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("heatmap metadata requires matrix")
    labels = matrix.get("annotation_text")
    label_colors = matrix.get("annotation_text_color")
    if not isinstance(labels, list) or not isinstance(label_colors, list):
        raise ValueError("heatmap metadata requires annotation matrices")
    return labels, label_colors


def render_characteristics_effect_size_heatmap_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render one compact characteristic effect-size heatmap from plot data."""
    if payload.get("plot_data_schema") != HEATMAP_PLOT_DATA_SCHEMA:
        raise ValueError("unsupported characteristics heatmap metadata schema")
    figure = payload.get("figure") if isinstance(payload.get("figure"), dict) else {}
    metric_labels = _heatmap_metric_labels(payload)
    level_labels = _heatmap_level_labels(payload)
    annotation_text, annotation_colors = _heatmap_annotation_payload(payload)
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(
        figsize=(
            HEATMAP_FIGURE_WIDTH,
            float(figure.get("height_inches", max(1.75, 0.36 * len(level_labels) + 0.9))),
        )
    )
    image = ax.imshow(
        _matrix_for_plot(payload),
        cmap=HEATMAP_COLORMAP,
        vmin=HEATMAP_COLOR_MIN,
        vmax=HEATMAP_COLOR_MAX,
        aspect="auto",
    )
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(
        metric_labels,
        rotation=HEATMAP_XTICK_ROTATION_DEGREES,
        ha="right",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    ax.set_yticks(range(len(level_labels)))
    ax.set_yticklabels(
        level_labels,
        rotation=HEATMAP_YTICK_ROTATION_DEGREES,
        ha="right",
        va="center",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    _annotate_heatmap_axis(
        ax,
        annotation_text,
        annotation_colors,
        column_count=len(metric_labels),
    )
    ax.set_xlabel(
        str(payload.get("x_axis") or "Metric"),
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
        labelpad=HEATMAP_X_AXIS_LABEL_PAD,
    )
    _align_heatmap_metric_label(ax)
    ax.set_ylabel(
        str(payload.get("y_axis") or DIMENSION_AXIS_LABELS.get(str(payload.get("dimension")), "")),
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(HEATMAP_COLOR_LABEL, fontsize=HEATMAP_COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_LABEL_SIZE)
    _add_heatmap_significance_caption(ax)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.32)
    save_figure(fig, Path(output_dir), stem)
    plt.close(fig)


def render_characteristics_extended_heatmap_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render one detailed characteristic effect-size heatmap from plot data."""
    if payload.get("plot_data_schema") != EXTENDED_HEATMAP_PLOT_DATA_SCHEMA:
        raise ValueError("unsupported characteristics extended heatmap metadata schema")
    figure = payload.get("figure") if isinstance(payload.get("figure"), dict) else {}
    metric_labels = _heatmap_metric_labels(payload)
    level_labels = _heatmap_level_labels(payload)
    annotation_text, annotation_colors = _heatmap_annotation_payload(payload)
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(
        figsize=(
            HEATMAP_FIGURE_WIDTH,
            float(figure.get("height_inches", max(1.52, 0.288 * len(level_labels) + 0.88))),
        )
    )
    image = ax.imshow(
        _matrix_for_plot(payload),
        cmap=HEATMAP_COLORMAP,
        vmin=HEATMAP_COLOR_MIN,
        vmax=HEATMAP_COLOR_MAX,
        aspect="auto",
    )
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(
        metric_labels,
        rotation=HEATMAP_XTICK_ROTATION_DEGREES,
        ha="right",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    ax.set_yticks(range(len(level_labels)))
    ax.set_yticklabels(
        level_labels,
        rotation=HEATMAP_YTICK_ROTATION_DEGREES,
        ha="right",
        va="center",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    _annotate_heatmap_axis(
        ax,
        annotation_text,
        annotation_colors,
        column_count=len(metric_labels),
    )
    _draw_column_group_spans(
        ax,
        payload.get("column_groups", []),
        show_labels=True,
    )
    ax.set_xlabel(
        str(payload.get("x_axis") or "Metric"),
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
        labelpad=HEATMAP_X_AXIS_LABEL_PAD,
    )
    _align_heatmap_metric_label(ax)
    ax.set_ylabel(
        str(payload.get("y_axis") or DIMENSION_AXIS_LABELS.get(str(payload.get("dimension")), "")),
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.025)
    colorbar.set_label(HEATMAP_COLOR_LABEL, fontsize=HEATMAP_COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_LABEL_SIZE)
    _add_heatmap_significance_caption(ax)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.35, top=0.86)
    save_figure(fig, Path(output_dir), stem)
    plt.close(fig)


def render_characteristics_combined_extended_heatmap_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render a combined detailed characteristic heatmap grid from plot data."""
    if payload.get("plot_data_schema") != EXTENDED_HEATMAP_GRID_PLOT_DATA_SCHEMA:
        raise ValueError("unsupported characteristics combined heatmap metadata schema")
    figure = payload.get("figure") if isinstance(payload.get("figure"), dict) else {}
    metric_labels = _heatmap_metric_labels(payload)
    level_labels = _heatmap_level_labels(payload)
    annotation_text, annotation_colors = _heatmap_annotation_payload(payload)
    row_groups = payload.get("row_groups") if isinstance(payload.get("row_groups"), list) else []
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(
        figsize=(
            HEATMAP_FIGURE_WIDTH,
            float(figure.get("height_inches", max(2.0, 0.23 * len(level_labels) + 1.25))),
        )
    )
    image = ax.imshow(
        _matrix_for_plot(payload),
        cmap=HEATMAP_COLORMAP,
        vmin=HEATMAP_COLOR_MIN,
        vmax=HEATMAP_COLOR_MAX,
        aspect="auto",
    )
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(
        metric_labels,
        rotation=HEATMAP_XTICK_ROTATION_DEGREES,
        ha="right",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    ax.set_yticks(range(len(level_labels)))
    ax.set_yticklabels(
        level_labels,
        rotation=HEATMAP_YTICK_ROTATION_DEGREES,
        ha="right",
        va="center",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    ax.set_xlabel(
        str(payload.get("x_axis") or "Metric"),
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
        labelpad=HEATMAP_X_AXIS_LABEL_PAD,
    )
    _align_heatmap_metric_label(ax)
    ax.set_ylabel("")
    _annotate_heatmap_axis(
        ax,
        annotation_text,
        annotation_colors,
        column_count=len(metric_labels),
    )
    _draw_column_group_spans(
        ax,
        payload.get("column_groups", []),
        show_labels=True,
    )
    for group in row_groups[:-1]:
        ax.axhline(
            float(group["end_row"]) + 0.5,
            color="black",
            linewidth=HEATMAP_SECTION_SEPARATOR_LINEWIDTH,
            alpha=HEATMAP_SECTION_SEPARATOR_ALPHA,
        )
    for group in row_groups:
        center = (float(group["start_row"]) + float(group["end_row"])) / 2.0
        ax.text(
            COMBINED_HEATMAP_ROW_GROUP_LABEL_X,
            center,
            _display_dimension_axis_label(str(group["dimension_label"])),
            transform=ax.get_yaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=COMBINED_HEATMAP_ROW_GROUP_LABEL_SIZE,
            fontweight="bold",
            clip_on=False,
        )

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes(
        "right",
        size=COMBINED_HEATMAP_COLORBAR_WIDTH,
        pad=COMBINED_HEATMAP_COLORBAR_PAD,
    )
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(HEATMAP_COLOR_LABEL, fontsize=HEATMAP_COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_LABEL_SIZE)
    _add_heatmap_significance_caption(
        ax,
        x=COMBINED_HEATMAP_ROW_GROUP_LABEL_X,
        coordinate_system="axes",
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=float(figure.get("bottom", 0.15)),
        top=float(figure.get("top", 0.88)),
    )
    save_figure(fig, Path(output_dir), stem, use_tight_layout=False)
    plt.close(fig)


def _write_extended_heatmap_grid(
    output_dir: Path,
    *,
    stem_suffix: str = "",
    dimensions: tuple[str, ...] = ("language", "popularity", "domain"),
) -> None:
    heatmap_dir = Path(output_dir) / "heatmaps"
    grid_stem = f"{EXTENDED_HEATMAP_GRID_STEM}{stem_suffix}"
    payloads: list[dict[str, Any]] = []
    for dimension in dimensions:
        payload = _load_extended_heatmap_payload(
            _extended_heatmap_json_path(
                output_dir,
                dimension,
                stem_suffix=stem_suffix,
            )
        )
        if payload is None:
            remove_plot_outputs(heatmap_dir, (grid_stem,))
            return
        payloads.append(payload)

    metric_signature = _extended_heatmap_metric_signature(payloads[0])
    if not metric_signature or any(
        _extended_heatmap_metric_signature(payload) != metric_signature
        for payload in payloads[1:]
    ):
        remove_plot_outputs(heatmap_dir, (grid_stem,))
        return

    metric_columns = payloads[0].get("metric_columns", [])
    if not metric_columns:
        remove_plot_outputs(heatmap_dir, (grid_stem,))
        return

    combined_plot_matrix: list[list[float]] = []
    combined_delta_matrix: list[list[Any]] = []
    combined_p_value_matrix: list[list[Any]] = []
    combined_adjusted_p_value_matrix: list[list[Any]] = []
    combined_significant_matrix: list[list[Any]] = []
    combined_delta_ci_low_matrix: list[list[Any]] = []
    combined_delta_ci_high_matrix: list[list[Any]] = []
    combined_annotation_text: list[list[Any]] = []
    combined_annotation_color: list[list[Any]] = []
    combined_level_rows: list[dict[str, Any]] = []
    combined_cells: list[dict[str, Any]] = []
    row_groups: list[dict[str, Any]] = []
    row_offset = 0

    for payload in payloads:
        dimension = str(payload.get("dimension"))
        level_rows = payload.get("level_rows", [])
        matrix = payload.get("matrix", {})
        plot_matrix = _matrix_for_plot(payload)
        row_count = len(level_rows)
        if row_count == 0:
            continue

        start_row = row_offset
        end_row = row_offset + row_count - 1
        row_groups.append(
            {
                "dimension": dimension,
                "dimension_label": DIMENSION_AXIS_LABELS.get(
                    dimension,
                    dimension,
                ),
                "start_row": start_row,
                "end_row": end_row,
            }
        )

        combined_plot_matrix.extend(plot_matrix)
        combined_delta_matrix.extend(matrix.get("cliffs_delta", []))
        combined_p_value_matrix.extend(matrix.get("p_value", []))
        combined_adjusted_p_value_matrix.extend(
            matrix.get("adjusted_p_value", [])
        )
        combined_significant_matrix.extend(
            matrix.get("significant_after_fdr", [])
        )
        combined_delta_ci_low_matrix.extend(
            matrix.get("cliffs_delta_ci95_low", [])
        )
        combined_delta_ci_high_matrix.extend(
            matrix.get("cliffs_delta_ci95_high", [])
        )
        combined_annotation_text.extend(matrix.get("annotation_text", []))
        combined_annotation_color.extend(matrix.get("annotation_text_color", []))

        for row_index, row in enumerate(level_rows):
            display_label = row.get("display_level_label") or _display_level_label(
                dimension,
                row.get("level_label"),
            )
            combined_row = dict(row)
            combined_row.update(
                {
                    "source_dimension": dimension,
                    "source_row_index": row_index,
                    "row_index": row_offset + row_index,
                    "dimension_label": DIMENSION_AXIS_LABELS.get(
                        dimension,
                        dimension,
                    ),
                    "display_level_label": display_label,
                }
            )
            combined_level_rows.append(combined_row)

        for cell in payload.get("cells", []):
            combined_cell = dict(cell)
            source_row_index = combined_cell.get("row_index")
            try:
                source_row_index_int = int(source_row_index)
            except (TypeError, ValueError):
                source_row_index_int = 0
            combined_cell.update(
                {
                    "source_dimension": dimension,
                    "source_row_index": source_row_index_int,
                    "row_index": row_offset + source_row_index_int,
                    "dimension_label": DIMENSION_AXIS_LABELS.get(
                        dimension,
                        dimension,
                    ),
                }
            )
            combined_cells.append(combined_cell)

        row_offset += row_count

    if not combined_level_rows:
        remove_plot_outputs(heatmap_dir, (grid_stem,))
        return

    plt, _mdates = require_matplotlib()
    width = HEATMAP_FIGURE_WIDTH
    figure_height = 3.75
    fig, ax = plt.subplots(
        figsize=(width, figure_height),
        constrained_layout=False,
    )
    image = ax.imshow(
        combined_plot_matrix,
        cmap=HEATMAP_COLORMAP,
        vmin=HEATMAP_COLOR_MIN,
        vmax=HEATMAP_COLOR_MAX,
        aspect="auto",
    )
    ax.set_xticks(range(len(metric_columns)))
    ax.set_xticklabels(
        [str(column.get("metric_label")) for column in metric_columns],
        rotation=HEATMAP_XTICK_ROTATION_DEGREES,
        ha="right",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    ax.set_yticks(range(len(combined_level_rows)))
    ax.set_yticklabels(
        [
            str(row.get("display_level_label") or row.get("level_label"))
            for row in combined_level_rows
        ],
        rotation=HEATMAP_YTICK_ROTATION_DEGREES,
        ha="right",
        va="center",
        fontsize=HEATMAP_AXIS_TICK_LABEL_SIZE,
    )
    ax.set_xlabel(
        "Metric",
        fontsize=HEATMAP_AXIS_LABEL_SIZE,
        labelpad=HEATMAP_X_AXIS_LABEL_PAD,
    )
    _align_heatmap_metric_label(ax)
    ax.set_ylabel("")
    _annotate_heatmap_axis(
        ax,
        combined_annotation_text,
        combined_annotation_color,
        column_count=len(metric_columns),
    )
    _draw_column_group_spans(
        ax,
        payloads[0].get("column_groups", []),
        show_labels=True,
    )
    for group in row_groups[:-1]:
        ax.axhline(
            float(group["end_row"]) + 0.5,
            color="black",
            linewidth=HEATMAP_SECTION_SEPARATOR_LINEWIDTH,
            alpha=HEATMAP_SECTION_SEPARATOR_ALPHA,
        )
    for group in row_groups:
        start_row = float(group["start_row"])
        end_row = float(group["end_row"])
        center = (start_row + end_row) / 2.0
        label = _display_dimension_axis_label(str(group["dimension_label"]))
        ax.text(
            COMBINED_HEATMAP_ROW_GROUP_LABEL_X,
            center,
            label,
            transform=ax.get_yaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=COMBINED_HEATMAP_ROW_GROUP_LABEL_SIZE,
            fontweight="bold",
            clip_on=False,
        )

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes(
        "right",
        size=COMBINED_HEATMAP_COLORBAR_WIDTH,
        pad=COMBINED_HEATMAP_COLORBAR_PAD,
    )
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(HEATMAP_COLOR_LABEL, fontsize=HEATMAP_COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_LABEL_SIZE)
    _add_heatmap_significance_caption(
        ax,
        x=COMBINED_HEATMAP_ROW_GROUP_LABEL_X,
        coordinate_system="axes",
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.15,
        top=0.88,
    )
    save_figure(
        fig,
        heatmap_dir,
        grid_stem,
        use_tight_layout=False,
    )
    write_plot_data(
        heatmap_dir,
        grid_stem,
        {
            "plot": grid_stem,
            "plot_data_schema": EXTENDED_HEATMAP_GRID_PLOT_DATA_SCHEMA,
            "plot_type": "extended_effect_size_heatmap_combined",
            "layout": "combined_rows",
            "dimensions": dimensions,
            "x_axis": "Metric",
            "row_group_axis": "Characteristic",
            "color": HEATMAP_COLOR_LABEL,
            "color_encoding": {
                "field": "cliffs_delta",
                "label": HEATMAP_COLOR_LABEL,
                "colormap": HEATMAP_COLORMAP,
                "minimum": HEATMAP_COLOR_MIN,
                "maximum": HEATMAP_COLOR_MAX,
                "center": HEATMAP_COLOR_CENTER,
                "positive_interpretation": "agents tend higher than humans",
                "negative_interpretation": "agents tend lower than humans",
            },
            "annotation_encoding": {
                "lines": ["cliffs_delta"],
                "number_format": "{:.2f}",
                "significance_marker": "*",
                "significance_field": "adjusted_p_value",
                "significance_threshold": HEATMAP_SIGNIFICANCE_P_THRESHOLD,
                "significance_caption": HEATMAP_SIGNIFICANCE_CAPTION,
                "missing_value_label": "NA",
            },
            "figure": {
                "width_inches": width,
                "height_inches": figure_height,
                "bottom": 0.15,
                "top": 0.88,
                "left": 0.01,
                "right": 0.99,
                "x_axis_label_y": HEATMAP_X_AXIS_LABEL_Y,
                "significance_caption_position": {
                    "x": COMBINED_HEATMAP_ROW_GROUP_LABEL_X,
                    "y": HEATMAP_SIGNIFICANCE_CAPTION_Y,
                    "coordinate_system": "axes",
                    "horizontal_alignment": "left",
                    "vertical_alignment": "center",
                },
                "row_group_label_x": COMBINED_HEATMAP_ROW_GROUP_LABEL_X,
                "row_group_label_padding": (
                    COMBINED_HEATMAP_ROW_GROUP_LABEL_PADDING
                ),
                "colorbar": {
                    "height_matches_heatmap": True,
                    "width": COMBINED_HEATMAP_COLORBAR_WIDTH,
                    "pad": COMBINED_HEATMAP_COLORBAR_PAD,
                    "placement": "attached_right_axis",
                },
            },
            "source_heatmaps": [
                {
                    "dimension": str(payload.get("dimension")),
                    "plot": payload.get("plot"),
                    "path": str(
                        _extended_heatmap_json_path(
                            output_dir,
                            str(payload.get("dimension")),
                            stem_suffix=stem_suffix,
                        )
                    ),
                }
                for payload in payloads
            ],
            "row_groups": row_groups,
            "metrics": [column["metric"] for column in metric_columns],
            "metric_columns": metric_columns,
            "levels": [
                {
                    "dimension": row.get("source_dimension"),
                    "level": row.get("level"),
                }
                for row in combined_level_rows
            ],
            "level_rows": combined_level_rows,
            "matrix": {
                "cliffs_delta": combined_delta_matrix,
                "p_value": combined_p_value_matrix,
                "adjusted_p_value": combined_adjusted_p_value_matrix,
                "significant_after_fdr": combined_significant_matrix,
                "cliffs_delta_ci95_low": combined_delta_ci_low_matrix,
                "cliffs_delta_ci95_high": combined_delta_ci_high_matrix,
                "annotation_text": combined_annotation_text,
                "annotation_text_color": combined_annotation_color,
            },
            "cells": combined_cells,
            "column_groups": payloads[0].get("column_groups", []),
            "panels": {
                str(payload.get("dimension")): {
                    "dimension": payload.get("dimension"),
                    "y_axis": payload.get("y_axis"),
                    "metric_columns": payload.get("metric_columns", []),
                    "level_rows": payload.get("level_rows", []),
                    "matrix": payload.get("matrix", {}),
                    "cells": payload.get("cells", []),
                    "column_groups": payload.get("column_groups", []),
                }
                for payload in payloads
            },
        },
    )
    plt.close(fig)


def write_extended_characteristics_heatmaps(
    *,
    output_dir: Path,
    companion_output_dir: Path,
    primary_analysis_group: str,
    companion_analysis_group: str,
    include_domain: bool,
    primary_heatmap_stem_prefix: str = "",
    primary_heatmap_stem_suffix: str = "",
    extended_stem_suffix: str = "",
    write_combined_grid_without_domain: bool = False,
) -> None:
    """Write extended heatmaps by merging existing heatmap JSON outputs."""
    apply_ieee_plot_style()
    dimensions = ["language", "popularity"]
    if include_domain:
        dimensions.append("domain")
    else:
        stems_to_remove = [_extended_heatmap_stem("domain", extended_stem_suffix)]
        if not write_combined_grid_without_domain:
            stems_to_remove.append(f"{EXTENDED_HEATMAP_GRID_STEM}{extended_stem_suffix}")
        remove_plot_outputs(output_dir / "heatmaps", tuple(stems_to_remove))

    for dimension in dimensions:
        primary_path = _heatmap_json_path(
            output_dir,
            dimension,
            stem_prefix=primary_heatmap_stem_prefix,
            stem_suffix=primary_heatmap_stem_suffix,
        )
        companion_path = _heatmap_json_path(companion_output_dir, dimension)
        primary_payload = _load_reusable_heatmap_payload(primary_path)
        companion_payload = _load_reusable_heatmap_payload(companion_path)
        if primary_payload is None or companion_payload is None:
            remove_plot_outputs(
                output_dir / "heatmaps",
                (_extended_heatmap_stem(dimension, extended_stem_suffix),),
            )
            continue
        wrote = _write_extended_heatmap_from_payloads(
            primary_payload=primary_payload,
            companion_payload=companion_payload,
            primary_path=primary_path,
            companion_path=companion_path,
            output_dir=output_dir,
            dimension=dimension,
            primary_analysis_group=primary_analysis_group,
            companion_analysis_group=companion_analysis_group,
            stem_suffix=extended_stem_suffix,
        )
        if not wrote:
            remove_plot_outputs(
                output_dir / "heatmaps",
                (_extended_heatmap_stem(dimension, extended_stem_suffix),),
            )
    if include_domain or write_combined_grid_without_domain:
        _write_extended_heatmap_grid(
            output_dir,
            stem_suffix=extended_stem_suffix,
            dimensions=tuple(dimensions),
        )


def write_characteristics_heatmaps(
    results: dict[str, Any],
    *,
    output_dir: Path,
    metrics: tuple[str, ...],
    include_domain: bool,
    stem_prefix: str = "",
    stem_suffix: str = "",
) -> None:
    """Write compact and detailed characteristic effect-size heatmaps."""
    apply_ieee_plot_style()
    resolved_metrics = tuple(
        metric for metric in metrics if metric not in HEATMAP_EXCLUDED_METRICS
    )
    if not include_domain:
        remove_plot_outputs(
            output_dir,
            characteristic_heatmap_stems(
                include_domain=True,
                stem_prefix=stem_prefix,
                stem_suffix=stem_suffix,
            )[2:],
        )
    _plot_effect_size_heatmap(
        results,
        dimension="language",
        metrics=resolved_metrics,
        levels=CHARACTERISTIC_LANGUAGES,
        level_labels=CHARACTERISTIC_LANGUAGE_LABELS,
        output_dir=output_dir,
        stem=f"{stem_prefix}language_effect_size_heatmap{stem_suffix}",
    )
    _plot_effect_size_heatmap(
        results,
        dimension="popularity",
        metrics=resolved_metrics,
        levels=POPULARITY_BUCKET_ORDER,
        level_labels={"low": "Low", "medium": "Medium", "high": "High"},
        output_dir=output_dir,
        stem=f"{stem_prefix}popularity_effect_size_heatmap{stem_suffix}",
    )
    if include_domain:
        _plot_effect_size_heatmap(
            results,
            dimension="domain",
            metrics=resolved_metrics,
            levels=CHARACTERISTIC_DOMAINS,
            level_labels={domain: domain for domain in CHARACTERISTIC_DOMAINS},
            output_dir=output_dir,
            stem=f"{stem_prefix}domain_effect_size_heatmap{stem_suffix}",
        )


def write_characteristics_plots(
    con,
    *,
    output_dir: Path,
    metrics: tuple[str, ...],
    results: dict[str, Any],
    include_domain: bool,
    base_where_sql: str = "TRUE",
    heatmap_metrics: tuple[str, ...] | None = None,
) -> None:
    """Write the full characteristic analysis plot family."""
    del con
    apply_ieee_plot_style()
    removed_plot_stems = list(removed_characteristic_boxplot_stems(metrics=metrics))
    for metric in metrics:
        metric_name = metric_stem(metric)
        removed_plot_stems.extend(
            [
                f"violin_plots/{metric_name}_language_violin_by_cohort",
                f"violin_plots/{metric_name}_popularity_violin_by_cohort",
                f"violin_plots/{metric_name}_domain_violin_by_cohort",
                f"violin_plots/{metric_name}_language_violin_by_authorship",
                f"violin_plots/{metric_name}_popularity_violin_by_authorship",
                f"violin_plots/{metric_name}_domain_violin_by_authorship",
            ]
        )
    remove_plot_outputs(output_dir, tuple(removed_plot_stems))
    if not include_domain:
        all_stems = set(characteristic_plot_stems(metrics=metrics, include_domain=True))
        kept_stems = set(characteristic_plot_stems(metrics=metrics, include_domain=False))
        remove_plot_outputs(output_dir, tuple(sorted(all_stems - kept_stems)))
    resolved_heatmap_metrics = tuple(
        metric
        for metric in (heatmap_metrics or metrics)
        if metric not in HEATMAP_EXCLUDED_METRICS
    )
    _plot_effect_size_heatmap(
        results,
        dimension="language",
        metrics=resolved_heatmap_metrics,
        levels=CHARACTERISTIC_LANGUAGES,
        level_labels=CHARACTERISTIC_LANGUAGE_LABELS,
        output_dir=output_dir,
        stem="language_effect_size_heatmap",
    )
    _plot_effect_size_heatmap(
        results,
        dimension="popularity",
        metrics=resolved_heatmap_metrics,
        levels=POPULARITY_BUCKET_ORDER,
        level_labels={"low": "Low", "medium": "Medium", "high": "High"},
        output_dir=output_dir,
        stem="popularity_effect_size_heatmap",
    )
    if include_domain:
        _plot_effect_size_heatmap(
            results,
            dimension="domain",
            metrics=resolved_heatmap_metrics,
            levels=CHARACTERISTIC_DOMAINS,
            level_labels={domain: domain for domain in CHARACTERISTIC_DOMAINS},
            output_dir=output_dir,
            stem="domain_effect_size_heatmap",
        )


def render_characteristics_plot_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> bool:
    """Dispatch saved characteristic plot data to the correct renderer."""
    plot_type = str(payload.get("plot_type") or "")
    if plot_type == "characteristics_composition_stacked_bars":
        render_characteristics_composition_stacked_bars_from_payload(
            payload,
            output_dir,
            stem,
        )
        return True
    if plot_type == "characteristics_composition_stacked_bars_grid":
        render_characteristics_composition_stacked_bars_grid_from_payload(
            payload,
            output_dir,
            stem,
        )
        return True
    if plot_type == "characteristics_median_ci_dotplot":
        render_characteristics_dotplot_from_payload(payload, output_dir, stem)
        return True
    if plot_type == "characteristics_median_ci_dotplot_grid":
        render_characteristics_dotplot_grid_from_payload(payload, output_dir, stem)
        return True
    if plot_type == "effect_size_heatmap":
        render_characteristics_effect_size_heatmap_from_payload(
            payload,
            output_dir,
            stem,
        )
        return True
    if plot_type == "extended_effect_size_heatmap":
        render_characteristics_extended_heatmap_from_payload(
            payload,
            output_dir,
            stem,
        )
        return True
    if plot_type == "extended_effect_size_heatmap_combined":
        render_characteristics_combined_extended_heatmap_from_payload(
            payload,
            output_dir,
            stem,
        )
        return True
    return False
