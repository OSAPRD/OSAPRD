"""Plots for dataset-level analysis outputs.

Dataset plots summarize cohort composition, repository popularity, topic/domain
coverage, PR size, creation timing, and longitudinal snapshot attrition. The
same rendering functions can consume SQL-like adapters or compact streaming
payloads produced by the dataset pipeline.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
UTILITY_DIR = ANALYSIS_DIR / "utility"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from balance_statistics_utility import numeric_distribution_summary
from analysis_runtime_utility import release_process_memory
from plotting_utility import (
    add_human_median_baseline,
    add_violin_underlay,
    apply_percentile_capped_y_axis,
    apply_ieee_plot_style,
    cohort_color_map,
    contrasting_text_color,
    display_group_label,
    display_group_labels,
    ieee_boxplot_kwargs,
    order_labels_by_average_percentage,
    order_humans_first,
    ranked_stacked_bar_colors,
    require_matplotlib,
    save_figure,
    apply_symlog_y_axis_if_range_exceeds,
    stacked_bar_visual_metadata,
    stacked_bar_visual_percentages,
    style_ieee_boxplot,
    write_plot_data,
)
from topic_groups_utility import TOPIC_CONFIDENCE_THRESHOLD, TOPIC_GROUP_ORDER
from longitudinal_analysis_utility import (
    LONGITUDINAL_TIMEPOINT_DAYS,
    LONGITUDINAL_TIMEPOINTS,
)
from longitudinal_analysis_plotter import (
    ATTRITION_AXIS_LOWER_BOUND,
    ATTRITION_AXIS_LOWER_PADDING,
    ATTRITION_AXIS_UPPER_BOUND,
    ATTRITION_AXIS_UPPER_PADDING,
    ATTRITION_FIGURE_HEIGHT,
    ATTRITION_LINE_WIDTH,
    ATTRITION_MARKER_SIZE,
    LONGITUDINAL_X_AXIS_PADDING,
    TIMEPOINT_X_POSITIONS,
    _resolved_cohort_colors,
    _timepoint_tick_labels,
    render_longitudinal_attrition_plot_from_payload,
)


TIME_PLOT_X_AXIS_START = datetime(2025, 5, 1)
TIME_PLOT_X_AXIS_END = datetime(2025, 12, 31, 23, 59, 59)
TIME_PLOT_X_AXIS_FINAL_LABEL = datetime(2026, 1, 1)
PLOT_OUTPUT_STEMS = (
    "creation_time_weekly_cumulative_counts_by_cohort",
    "creation_time_hour_of_day_distribution_overall",
    "language_by_cohort_100pct_stacked_bars",
    "domain_prs_by_cohort_100pct_stacked_bars",
    "domain_repos_by_cohort_100pct_stacked_bars",
    "domain_confidence_boxplot_by_domain",
    "language_popularity_by_cohort_100pct_stacked_bars",
    "popularity_size_by_cohort_boxplots",
    "pr_size_changed_files_boxplot_by_cohort",
    "pr_size_changed_lines_boxplot_by_cohort",
    "popularity_stargazer_boxplot",
    "popularity_bucket_100pct_stacked_bars",
    "longitudinal_attrition_by_cohort",
)
LANGUAGE_ORDER = ("python", "javascript", "java", "c++")
LANGUAGE_LABELS = {
    "python": "Python",
    "javascript": "JavaScript",
    "java": "Java",
    "c++": "C++",
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
DOMAIN_COLORS = {
    "AI, Data, and Science": "#C9C6C6",
    "Web and Mobile": "#E69F00",
    "Backend, APIs, and Security": "#0072B2",
    "Graphics": "#009E73",
    "Distributed and Embedded Systems": "#D55E00",
}
DOMAIN_TEXT_COLORS = {
    "AI, Data, and Science": "black",
    "Web and Mobile": "black",
    "Backend, APIs, and Security": "white",
    "Graphics": "white",
    "Distributed and Embedded Systems": "white",
}
DOMAIN_AXIS_LABELS = {
    "AI, Data, and Science": "AI/Data\nScience",
    "Web and Mobile": "Web/\nMobile",
    "Backend, APIs, and Security": "Backend/API\nSecurity",
    "Graphics": "Graphics",
    "Distributed and Embedded Systems": "Distributed/\nEmbedded",
}
DOMAIN_CONFIDENCE_AXIS_LABELS = {
    "AI, Data, and Science": "AI, Data,\nand Science",
    "Web and Mobile": "Web and\nMobile",
    "Backend, APIs, and Security": "Backend, APIs,\nand Security",
    "Graphics": "Graphics",
    "Distributed and Embedded Systems": "Distributed and\nEmbedded Systems",
}
DOMAIN_STACKED_BAR_LEGEND_CENTER_X = 0.43
STACKED_BAR_PERCENTAGE_FONT_SIZE = 6.0
STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET = -0.3
PR_SIZE_AXIS_LOWER_BOUND = 1.0
PR_SIZE_AXIS_LOWER_PADDING = 0.1
POPULARITY_AXIS_LOWER_BOUND = 0.0
POPULARITY_AXIS_LOWER_PADDING = 0.1
GRID_BOXPLOT_X_MARGIN = 0.1
GRID_BOXPLOT_DEFAULT_HALF_WIDTH = 0.25
INDIVIDUAL_BOXPLOT_WIDTH = 3.5
INDIVIDUAL_BOXPLOT_HEIGHT = 2.0
INDIVIDUAL_BOXPLOT_BOTTOM = 0.36
INDIVIDUAL_BOXPLOT_X_AXIS_LABEL_Y = -0.25
POPULARITY_BUCKET_COLORS = {
    "pop0": "#C9C6C6",
    "pop1": "#0072B2",
    "pop2": "#D55E00",
}
POPULARITY_BUCKET_TEXT_COLORS = {
    "pop0": "black",
    "pop1": "white",
    "pop2": "white",
}


def _simple_count_map(
    rows: list[tuple[Any, Any]],
    *,
    labels: list[str] | None = None,
) -> dict[str, int]:
    counts = {label: 0 for label in labels or []}
    for group_key, count in rows:
        if group_key is None:
            continue
        counts[str(group_key)] = counts.get(str(group_key), 0) + int(count or 0)
    return counts


def _summary_payload(values: list[float] | list[int]) -> dict[str, Any]:
    return {
        **numeric_distribution_summary(values),
        "n": len(values),
    }


def _count_percentage_payload(
    counts: dict[str, int],
    labels: tuple[str, ...] | list[str],
    *,
    count_key: str = "count",
) -> dict[str, dict[str, float | int]]:
    total = sum(int(counts.get(label, 0)) for label in labels)
    percentages = [
        100.0 * int(counts.get(label, 0)) / total if total else 0.0
        for label in labels
    ]
    visual_percentages = stacked_bar_visual_percentages(percentages)
    return {
        str(label): {
            count_key: int(counts.get(label, 0)),
            "percentage": percentages[index],
            "visual_percentage": visual_percentages[index],
        }
        for index, label in enumerate(labels)
    }


def _stacked_bar_percentage_maps(
    *,
    cohorts: list[str],
    labels: list[str],
    counts_by_cohort: dict[str, dict[str, int]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    true_by_cohort: dict[str, dict[str, float]] = {}
    visual_by_cohort: dict[str, dict[str, float]] = {}
    for cohort in cohorts:
        counts = counts_by_cohort.get(cohort, {})
        total = sum(int(counts.get(label, 0)) for label in labels)
        percentages = [
            100.0 * int(counts.get(label, 0)) / total if total else 0.0
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


def _cohorts(con) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT cohort FROM analysis_prs ORDER BY cohort"
    ).fetchall()
    return order_humans_first(
        str(row[0]) for row in rows if row[0] is not None and str(row[0]).strip()
    )


class _PayloadQueryResult:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None


class _DatasetPlotPayloadConnection:
    """Small adapter letting existing dataset plot renderers consume payloads."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def execute(self, sql: str) -> _PayloadQueryResult:
        normalized_sql = " ".join(str(sql).strip().lower().split())
        if "select distinct cohort from analysis_prs" in normalized_sql:
            return _PayloadQueryResult([(cohort,) for cohort in self._cohorts()])
        if "created_week_start" in normalized_sql and "group by cohort" in normalized_sql:
            return _PayloadQueryResult(self._creation_week_rows())
        if "cast(created_hour as varchar)" in normalized_sql:
            return _PayloadQueryResult(self._creation_hour_rows())
        if "select cohort, popularity_bucket" in normalized_sql:
            return _PayloadQueryResult(self._cohort_count_rows("popularity_counts_by_cohort"))
        if "select cohort, language" in normalized_sql:
            return _PayloadQueryResult(self._cohort_count_rows("language_counts_by_cohort"))
        if "from analysis_pr_domains" in normalized_sql and "confidence" in normalized_sql:
            return _PayloadQueryResult(self._domain_confidence_rows())
        if "from analysis_pr_domains" in normalized_sql and "count(distinct repository_identity)" in normalized_sql:
            return _PayloadQueryResult(self._cohort_count_rows("domain_repo_counts_by_cohort"))
        if "from analysis_pr_domains" in normalized_sql and "count(*) as item_count" in normalized_sql:
            return _PayloadQueryResult(self._cohort_count_rows("domain_pr_counts_by_cohort"))
        if "from analysis_prs" in normalized_sql and "order by cohort," in normalized_sql:
            metric = self._metric_from_numeric_query(normalized_sql)
            if metric is not None:
                return _PayloadQueryResult(self._numeric_rows(metric))
        if "from analysis_dataset_longitudinal_snapshots" in normalized_sql:
            return _PayloadQueryResult(self._longitudinal_availability_rows())
        raise ValueError(f"Unsupported dataset plot payload query: {sql}")

    def _cohorts(self) -> list[str]:
        cohorts = [
            str(cohort)
            for cohort in self.payload.get("cohorts", [])
            if str(cohort).strip()
        ]
        if cohorts:
            return order_humans_first(cohorts)
        observed: set[str] = set()
        for key in (
            "language_counts_by_cohort",
            "popularity_counts_by_cohort",
            "domain_pr_counts_by_cohort",
            "numeric_values_by_cohort",
            "longitudinal_availability",
        ):
            value = self.payload.get(key) or {}
            if key == "numeric_values_by_cohort":
                for by_cohort in value.values():
                    observed.update(str(cohort) for cohort in by_cohort)
            else:
                observed.update(str(cohort) for cohort in value)
        return order_humans_first(observed)

    def _creation_week_rows(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for cohort, counts in (self.payload.get("creation_week_counts_by_cohort") or {}).items():
            for week_start, count in counts.items():
                try:
                    parsed_week = datetime.fromisoformat(str(week_start)).date()
                except ValueError:
                    continue
                rows.append((str(cohort), parsed_week, int(count or 0)))
        return sorted(rows, key=lambda row: (row[0], row[1]))

    def _creation_hour_rows(self) -> list[tuple[Any, ...]]:
        counts = self.payload.get("creation_hour_counts") or {}
        return [
            (str(hour), int(counts.get(str(hour), 0) or 0))
            for hour in sorted((str(key) for key in counts), key=lambda value: int(value))
        ]

    def _cohort_count_rows(self, payload_key: str) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for cohort, counts in (self.payload.get(payload_key) or {}).items():
            for label, count in counts.items():
                rows.append((str(cohort), str(label), int(count or 0)))
        return sorted(rows, key=lambda row: (row[0], row[1]))

    def _domain_confidence_rows(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for domain, values in (self.payload.get("domain_confidence_values") or {}).items():
            for value in values:
                rows.append((str(domain), float(value)))
        return sorted(rows, key=lambda row: (row[0], row[1]))

    def _numeric_rows(self, metric: str) -> list[tuple[Any, ...]]:
        by_cohort = (self.payload.get("numeric_values_by_cohort") or {}).get(metric, {})
        rows: list[tuple[Any, ...]] = []
        for cohort, values in by_cohort.items():
            for value in values:
                rows.append((str(cohort), float(value)))
        return sorted(rows, key=lambda row: (row[0], row[1]))

    def _longitudinal_availability_rows(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for cohort, by_label in (self.payload.get("longitudinal_availability") or {}).items():
            for label, counter in by_label.items():
                days = LONGITUDINAL_TIMEPOINT_DAYS.get(str(label), 0)
                total = int(counter.get(f"{label}:total", 0) or 0)
                available = int(counter.get(f"{label}:available", 0) or 0)
                rows.append((str(cohort), str(label), int(days), total, available))
        return sorted(rows, key=lambda row: (row[0], row[2]))

    @staticmethod
    def _metric_from_numeric_query(sql: str) -> str | None:
        selected_columns_match = re.search(
            r"\bselect\s+cohort\s*,\s+(.*?)\s+from\s+analysis_prs\b",
            sql,
        )
        selected_metric_sql = selected_columns_match.group(1).strip() if selected_columns_match else sql
        selected_metric_name = selected_metric_sql.strip().strip('"')
        for metric in ("changed_files_count", "changed_line_count", "stargazer_count"):
            if metric == selected_metric_name:
                return metric
        return None


def _popularity_counts_by_cohort(
    con,
    scheme: dict[str, Any],
) -> dict[str, dict[str, dict[str, int]]]:
    labels = [str(label) for label in scheme.get("bucket_labels") or []]
    nested = {
        cohort: {label: {"pull_request_count": 0} for label in labels}
        for cohort in _cohorts(con)
    }
    rows = con.execute(
        """
        SELECT cohort, popularity_bucket, COUNT(*) AS pull_request_count
        FROM analysis_prs
        GROUP BY cohort, popularity_bucket
        ORDER BY cohort, popularity_bucket
        """
    ).fetchall()
    for cohort, bucket, count in rows:
        if cohort is None or bucket is None:
            continue
        cohort_key = str(cohort)
        nested.setdefault(cohort_key, {})
        for label in labels:
            nested[cohort_key].setdefault(label, {"pull_request_count": 0})
        nested[cohort_key][str(bucket)] = {"pull_request_count": int(count or 0)}
    return nested


def _flat_popularity_counts_by_cohort(
    con,
    scheme: dict[str, Any],
) -> dict[str, dict[str, int]]:
    labels = [str(label) for label in scheme.get("bucket_labels") or []]
    nested_counts = _popularity_counts_by_cohort(con, scheme)
    return {
        cohort: {
            label: int(
                nested_counts
                .get(cohort, {})
                .get(label, {})
                .get("pull_request_count", 0)
            )
            for label in labels
        }
        for cohort in _cohorts(con)
    }


def _language_counts_by_cohort(con) -> dict[str, dict[str, int]]:
    nested = {
        cohort: {language: 0 for language in LANGUAGE_ORDER}
        for cohort in _cohorts(con)
    }
    rows = con.execute(
        """
        SELECT cohort, language, COUNT(*) AS pull_request_count
        FROM analysis_prs
        WHERE language IS NOT NULL
        GROUP BY cohort, language
        ORDER BY cohort, language
        """
    ).fetchall()
    for cohort, language, count in rows:
        if cohort is None or language is None:
            continue
        cohort_key = str(cohort)
        language_key = str(language).strip().lower()
        nested.setdefault(cohort_key, {label: 0 for label in LANGUAGE_ORDER})
        if language_key in LANGUAGE_ORDER:
            nested[cohort_key][language_key] = int(count or 0)
    return nested


def _draw_cohort_percentage_stacked_bars(
    ax,
    *,
    cohorts: list[str],
    labels: list[str],
    counts_by_cohort: dict[str, dict[str, int]],
    display_labels: dict[str, str],
    colors: dict[str, str],
    text_colors: dict[str, str],
    xlabel: str | None = "Cohort",
    ylabel: str = "Percentage of PRs (%)",
    show_ylabel: bool = True,
    legend_fontsize: float | None = None,
    legend_title: str | None = None,
    legend_title_fontsize: float = 7.0,
    y_labelpad: float | None = None,
    y_tick_labelpad: float | None = None,
) -> None:
    bottoms = [0.0 for _cohort in cohorts]
    x_values = list(range(len(cohorts)))
    true_by_cohort, visual_by_cohort = _stacked_bar_percentage_maps(
        cohorts=cohorts,
        labels=labels,
        counts_by_cohort=counts_by_cohort,
    )
    for label in labels:
        percentages = [
            true_by_cohort[cohort].get(label, 0.0)
            for cohort in cohorts
        ]
        visual_percentages = [
            visual_by_cohort[cohort].get(label, 0.0)
            for cohort in cohorts
        ]
        ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=display_labels.get(label, label),
            color=colors[label],
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
                color=text_colors.get(label, "black"),
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if y_labelpad is None:
        ax.set_ylabel(ylabel if show_ylabel else "")
    else:
        ax.set_ylabel(ylabel if show_ylabel else "", labelpad=y_labelpad)
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(cohorts), rotation=0, ha="center")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    if y_tick_labelpad is not None:
        ax.tick_params(axis="y", pad=y_tick_labelpad)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(cohorts) - 0.5))
    legend_kwargs: dict[str, Any] = {}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    if legend_title is not None:
        legend_kwargs["title"] = legend_title
        legend_kwargs["title_fontsize"] = legend_title_fontsize
    ax.legend(
        frameon=False,
        ncol=len(labels),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        columnspacing=0.45,
        handlelength=0.8,
        handletextpad=0.25,
        **legend_kwargs,
    )


def _stacked_bar_panel_payload(
    *,
    cohorts: list[str],
    labels: list[str],
    counts_by_cohort: dict[str, dict[str, int]],
    count_key: str,
    total_key: str,
) -> dict[str, Any]:
    true_by_cohort, visual_by_cohort = _stacked_bar_percentage_maps(
        cohorts=cohorts,
        labels=labels,
        counts_by_cohort=counts_by_cohort,
    )
    return {
        cohort: {
            total_key: int(
                sum(counts_by_cohort.get(cohort, {}).get(label, 0) for label in labels)
            ),
            "groups": {
                label: {
                    count_key: int(counts_by_cohort.get(cohort, {}).get(label, 0)),
                    "percentage": true_by_cohort[cohort].get(label, 0.0),
                    "visual_percentage": visual_by_cohort[cohort].get(label, 0.0),
                }
                for label in labels
            },
        }
        for cohort in cohorts
    }


def _domain_counts_by_cohort(
    con,
    *,
    count_kind: str,
) -> dict[str, dict[str, int]]:
    nested = {
        cohort: {domain: 0 for domain in TOPIC_GROUP_ORDER}
        for cohort in _cohorts(con)
    }
    if count_kind == "pull_request":
        rows = con.execute(
            """
            SELECT cohort, topic_group, COUNT(*) AS item_count
            FROM analysis_pr_domains
            GROUP BY cohort, topic_group
            ORDER BY cohort, topic_group
            """
        ).fetchall()
    elif count_kind == "repository":
        rows = con.execute(
            """
            SELECT
                cohort,
                topic_group,
                COUNT(DISTINCT repository_identity) AS item_count
            FROM analysis_pr_domains
            WHERE repository_identity IS NOT NULL
            GROUP BY cohort, topic_group
            ORDER BY cohort, topic_group
            """
        ).fetchall()
    else:
        raise ValueError(f"Unsupported domain count kind: {count_kind}")

    for cohort, topic_group, count in rows:
        if cohort is None or topic_group is None:
            continue
        cohort_key = str(cohort)
        domain_key = str(topic_group)
        nested.setdefault(cohort_key, {domain: 0 for domain in TOPIC_GROUP_ORDER})
        if domain_key in TOPIC_GROUP_ORDER:
            nested[cohort_key][domain_key] = int(count or 0)
    return nested


def _format_month_axis(
    ax,
    mdates,
    *,
    left_limit: Any | None = None,
    right_limit: Any | None = None,
    date_format: str = "%b",
) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    if left_limit is None:
        left_limit = TIME_PLOT_X_AXIS_START
    if right_limit is None:
        right_limit = TIME_PLOT_X_AXIS_END
    ax.set_xlim(left=left_limit, right=right_limit)


def _plot_creation_time_weekly_cumulative_counts_by_cohort(
    con,
    output_dir: Path,
) -> None:
    plt, mdates = require_matplotlib()
    rows = con.execute(
        """
        SELECT cohort, created_week_start, COUNT(*) AS pull_request_count
        FROM analysis_prs
        WHERE created_week_start IS NOT NULL
        GROUP BY cohort, created_week_start
        ORDER BY cohort, created_week_start
        """
    ).fetchall()
    by_cohort: dict[str, dict[Any, int]] = {}
    all_weeks: set[Any] = set()
    for cohort, week_start, count in rows:
        by_cohort.setdefault(str(cohort), {})[week_start] = int(count or 0)
        all_weeks.add(week_start)

    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    colors = cohort_color_map(by_cohort)
    if all_weeks:
        min_week = min(all_weeks)
        max_week = max(all_weeks)
        weeks = []
        current_week = min_week
        while current_week <= max_week:
            weeks.append(current_week)
            current_week = current_week + timedelta(days=7)
        for cohort in order_humans_first(by_cohort):
            cumulative_counts = []
            cumulative = 0
            for week in weeks:
                cumulative += by_cohort[cohort].get(week, 0)
                cumulative_counts.append(cumulative)
            ax.plot(
                weeks,
                cumulative_counts,
                marker="o",
                label=display_group_label(cohort),
                color=colors[cohort],
                linewidth=0.9,
                markersize=2.0,
            )
        _format_month_axis(
            ax,
            mdates,
            right_limit=TIME_PLOT_X_AXIS_FINAL_LABEL,
            date_format="%m.%y",
        )
        ax.legend(frameon=False, ncol=2)
    ax.set_xlabel("PR Creation Date (MM.YY)")
    ax.set_ylabel("Cumulative PR Count")
    ax.grid(alpha=0.3)
    save_figure(
        fig,
        output_dir,
        "creation_time_weekly_cumulative_counts_by_cohort",
    )
    write_plot_data(
        output_dir,
        "creation_time_weekly_cumulative_counts_by_cohort",
        {
            "plot": "creation_time_weekly_cumulative_counts_by_cohort",
            "x_axis": "PR Creation Date (MM.YY)",
            "y_axis": "Cumulative PR Count",
            "series": [
                {
                    "cohort": cohort,
                    "points": [
                        {
                            "created_week_start": str(week),
                            "weekly_pull_request_count": int(
                                by_cohort[cohort].get(week, 0)
                            ),
                            "cumulative_pull_request_count": int(
                                sum(
                                    by_cohort[cohort].get(prior_week, 0)
                                    for prior_week in weeks[: index + 1]
                                )
                            ),
                        }
                        for index, week in enumerate(weeks)
                    ],
                }
                for cohort in order_humans_first(by_cohort)
            ],
        },
    )
    plt.close(fig)


def _plot_creation_time_hour_of_day_distribution_overall(
    con,
    output_dir: Path,
) -> None:
    plt, _mdates = require_matplotlib()
    counts = _simple_count_map(
        con.execute(
            """
            SELECT
                CAST(created_hour AS VARCHAR) AS created_hour,
                COUNT(*) AS pull_request_count
            FROM analysis_prs
            WHERE created_hour IS NOT NULL
            GROUP BY created_hour
            ORDER BY created_hour
            """
        ).fetchall(),
        labels=[str(hour) for hour in range(24)],
    )
    total = sum(counts.values())
    proportions = [
        counts[str(hour)] / total if total else 0.0
        for hour in range(24)
    ]
    percentages = [proportion * 100.0 for proportion in proportions]

    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    ax.bar(
        range(24),
        percentages,
        width=1.0,
        align="edge",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.3,
    )
    for hour, percentage in enumerate(percentages):
        if percentage <= 0:
            continue
        ax.text(
            hour + 0.5,
            percentage + 0.08,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=STACKED_BAR_PERCENTAGE_FONT_SIZE,
            rotation=90,
        )
    ax.set_xlabel("PR Creation Hour (UTC)")
    ax.set_ylabel("Percentage of PRs (%)")
    hour_ticks = list(range(0, 25, 3))
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f"{hour:02d}:00" for hour in hour_ticks])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, max(percentages) * 1.18 if percentages else 1)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, output_dir, "creation_time_hour_of_day_distribution_overall")
    write_plot_data(
        output_dir,
        "creation_time_hour_of_day_distribution_overall",
        {
            "plot": "creation_time_hour_of_day_distribution_overall",
            "x_axis": "PR Creation Hour (UTC)",
            "y_axis": "Percentage of PRs (%)",
            "total_pull_request_count": int(total),
            "hours": [
                {
                    "hour": hour,
                    "label": f"{hour:02d}:00",
                    "pull_request_count": int(counts[str(hour)]),
                    "percentage": percentages[hour],
                    "proportion": proportions[hour],
                }
                for hour in range(24)
            ],
        },
    )
    plt.close(fig)


def _plot_language_by_cohort_100pct_stacked_bars(con, output_dir: Path) -> None:
    plt, _mdates = require_matplotlib()
    cohorts = _cohorts(con)
    counts_by_cohort = _language_counts_by_cohort(con)
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    bottoms = [0.0 for _cohort in cohorts]
    x_values = list(range(len(cohorts)))
    true_by_cohort, visual_by_cohort = _stacked_bar_percentage_maps(
        cohorts=cohorts,
        labels=list(LANGUAGE_ORDER),
        counts_by_cohort=counts_by_cohort,
    )
    for language in LANGUAGE_ORDER:
        percentages = [
            true_by_cohort[cohort].get(language, 0.0)
            for cohort in cohorts
        ]
        visual_percentages = [
            visual_by_cohort[cohort].get(language, 0.0)
            for cohort in cohorts
        ]
        ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=LANGUAGE_LABELS[language],
            color=LANGUAGE_COLORS[language],
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
                color=LANGUAGE_TEXT_COLORS[language],
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Percentage of PRs (%)")
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(cohorts), rotation=0, ha="center")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(cohorts) - 0.5))
    ax.legend(
        frameon=False,
        ncol=len(LANGUAGE_ORDER),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        columnspacing=0.7,
        handlelength=0.9,
        handletextpad=0.35,
    )
    save_figure(fig, output_dir, "language_by_cohort_100pct_stacked_bars")
    write_plot_data(
        output_dir,
        "language_by_cohort_100pct_stacked_bars",
        {
            "plot": "language_by_cohort_100pct_stacked_bars",
            "x_axis": "Cohort",
            "y_axis": "Percentage of PRs (%)",
            **stacked_bar_visual_metadata(),
            "language_order": list(LANGUAGE_ORDER),
            "language_labels": LANGUAGE_LABELS,
            "cohorts": {
                cohort: {
                    "total_pull_request_count": int(
                        sum(
                            counts_by_cohort.get(cohort, {}).get(language, 0)
                            for language in LANGUAGE_ORDER
                        )
                    ),
                    "languages": _count_percentage_payload(
                        counts_by_cohort.get(cohort, {}),
                        LANGUAGE_ORDER,
                    ),
                }
                for cohort in cohorts
            },
        },
    )
    plt.close(fig)


def _add_centered_domain_legend_rows(
    ax,
    *,
    domains: list[str],
    legend_columns: int,
    legend_fontsize: float,
    domain_colors: dict[str, str] | None = None,
) -> None:
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
    from matplotlib.patches import Rectangle
    resolved_domain_colors = domain_colors or DOMAIN_COLORS

    def legend_item(domain: str):
        handle = DrawingArea(6.0, 5.0, 0.0, 0.0)
        handle.add_artist(
            Rectangle(
                (0.0, 1.0),
                5.0,
                3.0,
                facecolor=resolved_domain_colors[domain],
                edgecolor="black",
                linewidth=0.3,
            )
        )
        label = TextArea(
            domain,
            textprops={
                "fontsize": legend_fontsize,
                "va": "center",
            },
        )
        return HPacker(
            children=[handle, label],
            align="center",
            pad=0.0,
            sep=2.0,
        )

    rows = [
        domains[index : index + legend_columns]
        for index in range(0, len(domains), legend_columns)
    ]
    row_boxes = [
        HPacker(
            children=[legend_item(domain) for domain in row],
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
            bbox_to_anchor=(DOMAIN_STACKED_BAR_LEGEND_CENTER_X, 1.02),
            bbox_transform=ax.transAxes,
        )
    )


def _plot_domain_by_cohort_100pct_stacked_bars(
    con,
    output_dir: Path,
    *,
    count_kind: str,
    ylabel: str,
    stem: str,
    total_key: str,
    count_key: str,
    percentage_label_y_offset: float = STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET,
    legend_columns: int | None = None,
    legend_fontsize: float = 4.8,
) -> None:
    plt, _mdates = require_matplotlib()
    domains = list(TOPIC_GROUP_ORDER)
    cohorts = _cohorts(con)
    counts_by_cohort = _domain_counts_by_cohort(con, count_kind=count_kind)
    true_by_cohort, visual_by_cohort = _stacked_bar_percentage_maps(
        cohorts=cohorts,
        labels=domains,
        counts_by_cohort=counts_by_cohort,
    )
    domains = order_labels_by_average_percentage(
        domains,
        cohorts,
        true_by_cohort,
    )
    domain_colors = ranked_stacked_bar_colors(domains)
    domain_text_colors = {
        domain: contrasting_text_color(color)
        for domain, color in domain_colors.items()
    }
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    bottoms = [0.0 for _cohort in cohorts]
    x_values = list(range(len(cohorts)))
    legend_handles = []
    for domain in domains:
        percentages = [
            true_by_cohort[cohort].get(domain, 0.0)
            for cohort in cohorts
        ]
        visual_percentages = [
            visual_by_cohort[cohort].get(domain, 0.0)
            for cohort in cohorts
        ]
        bar_container = ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=domain,
            color=domain_colors[domain],
            edgecolor="black",
            linewidth=0.3,
        )
        legend_handles.append(bar_container[0])
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
                + percentage_label_y_offset,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=STACKED_BAR_PERCENTAGE_FONT_SIZE,
                color=domain_text_colors[domain],
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    ax.set_xlabel("Cohort")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(cohorts), rotation=0, ha="center")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(cohorts) - 0.5))
    if legend_columns is not None and len(domains) > legend_columns:
        _add_centered_domain_legend_rows(
            ax,
            domains=domains,
            legend_columns=legend_columns,
            legend_fontsize=legend_fontsize,
            domain_colors=domain_colors,
        )
    else:
        ax.legend(
            legend_handles,
            domains,
            frameon=False,
            ncol=legend_columns or len(domains),
            loc="lower center",
            bbox_to_anchor=(DOMAIN_STACKED_BAR_LEGEND_CENTER_X, 1.02),
            borderaxespad=0.0,
            columnspacing=0.45,
            handlelength=0.8,
            handletextpad=0.25,
            fontsize=legend_fontsize,
        )
    save_figure(fig, output_dir, stem)
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "x_axis": "Cohort",
            "y_axis": ylabel,
            **stacked_bar_visual_metadata(),
            "legend": {
                "center_x": DOMAIN_STACKED_BAR_LEGEND_CENTER_X,
                "centered_on": "plot_and_y_axis_label",
                "columns": legend_columns or len(domains),
                "font_size": legend_fontsize,
                "rows": (
                    [
                        domains[:legend_columns],
                        domains[legend_columns:],
                    ]
                    if legend_columns is not None
                    and len(domains) > legend_columns
                    else [domains]
                ),
            },
            "confidence_threshold": TOPIC_CONFIDENCE_THRESHOLD,
            "domain_order": domains,
            "domain_colors": domain_colors,
            "domain_text_colors": domain_text_colors,
            "stack_order_basis": "descending_average_percentage_across_cohorts",
            "cohorts": {
                cohort: {
                    total_key: int(
                        sum(
                            counts_by_cohort.get(cohort, {}).get(domain, 0)
                            for domain in domains
                        )
                    ),
                    "domains": _count_percentage_payload(
                        counts_by_cohort.get(cohort, {}),
                        domains,
                        count_key=count_key,
                    ),
                }
                for cohort in cohorts
            },
        },
    )
    plt.close(fig)


def _plot_domain_confidence_boxplot_by_domain(con, output_dir: Path) -> None:
    plt, _mdates = require_matplotlib()
    rows = con.execute(
        """
        SELECT topic_group, confidence
        FROM (
            SELECT DISTINCT
                repository_identity,
                topic_group,
                confidence
            FROM analysis_pr_domains
            WHERE repository_identity IS NOT NULL
              AND confidence IS NOT NULL
        )
        ORDER BY topic_group, confidence
        """
    ).fetchall()
    by_domain: dict[str, list[float]] = {domain: [] for domain in TOPIC_GROUP_ORDER}
    for topic_group, confidence in rows:
        domain = str(topic_group)
        if domain in by_domain and confidence is not None:
            by_domain[domain].append(float(confidence))

    domains = [domain for domain in TOPIC_GROUP_ORDER if by_domain.get(domain)]
    domain_confidence_height = 1.5
    fig, ax = plt.subplots(
        figsize=(INDIVIDUAL_BOXPLOT_WIDTH, domain_confidence_height)
    )
    if domains:
        values_by_domain = [by_domain[domain] for domain in domains]
        colors = [DOMAIN_COLORS[domain] for domain in domains]
        add_violin_underlay(ax, values_by_domain, colors=colors)
        try:
            boxplot = ax.boxplot(
                values_by_domain,
                tick_labels=[
                    DOMAIN_CONFIDENCE_AXIS_LABELS[domain]
                    for domain in domains
                ],
                **ieee_boxplot_kwargs(),
            )
        except TypeError:
            boxplot = ax.boxplot(
                values_by_domain,
                labels=[
                    DOMAIN_CONFIDENCE_AXIS_LABELS[domain]
                    for domain in domains
                ],
                **ieee_boxplot_kwargs(),
            )
        style_ieee_boxplot(boxplot, colors)
        ax.axhline(
            TOPIC_CONFIDENCE_THRESHOLD,
            color="0.35",
            linestyle=(0, (3, 2)),
            linewidth=0.55,
            alpha=0.6,
            zorder=1,
        )
        ax.set_ylim(
            bottom=max(0.0, TOPIC_CONFIDENCE_THRESHOLD - 0.03),
            top=1.01,
        )
        ax.tick_params(axis="x", rotation=0, labelsize=5.5)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Topic Confidence")
    ax.xaxis.set_label_coords(0.5, INDIVIDUAL_BOXPLOT_X_AXIS_LABEL_Y)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=INDIVIDUAL_BOXPLOT_BOTTOM)
    save_figure(fig, output_dir, "domain_confidence_boxplot_by_domain")
    write_plot_data(
        output_dir,
        "domain_confidence_boxplot_by_domain",
        {
            "plot": "domain_confidence_boxplot_by_domain",
            "metric": "topic_confidence",
            "x_axis": "Domain",
            "y_axis": "Topic Confidence",
            "confidence_threshold": TOPIC_CONFIDENCE_THRESHOLD,
            "domain_order": list(TOPIC_GROUP_ORDER),
            "figure": {
                "width_inches": INDIVIDUAL_BOXPLOT_WIDTH,
                "height_inches": domain_confidence_height,
                "bottom": INDIVIDUAL_BOXPLOT_BOTTOM,
                "x_axis_label_y": INDIVIDUAL_BOXPLOT_X_AXIS_LABEL_Y,
            },
            "groups": {
                domain: _summary_payload(by_domain.get(domain, []))
                for domain in TOPIC_GROUP_ORDER
            },
        },
    )
    plt.close(fig)


def _plot_language_popularity_by_cohort_100pct_stacked_bars(
    con,
    scheme: dict[str, Any],
    output_dir: Path,
) -> None:
    plt, _mdates = require_matplotlib()
    cohorts = _cohorts(con)
    language_counts_by_cohort = _language_counts_by_cohort(con)
    popularity_labels = [str(label) for label in scheme.get("bucket_labels") or []]
    popularity_counts_by_cohort = _flat_popularity_counts_by_cohort(con, scheme)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.0),
        sharey=True,
    )
    _draw_cohort_percentage_stacked_bars(
        axes[0],
        cohorts=cohorts,
        labels=list(LANGUAGE_ORDER),
        counts_by_cohort=language_counts_by_cohort,
        display_labels=LANGUAGE_LABELS,
        colors=LANGUAGE_COLORS,
        text_colors=LANGUAGE_TEXT_COLORS,
        ylabel="Percentage of PRs (%)",
        show_ylabel=True,
        y_labelpad=3.0,
        y_tick_labelpad=1.0,
        xlabel=None,
        legend_title="Programming Language",
        legend_title_fontsize=7.5,
    )
    _draw_cohort_percentage_stacked_bars(
        axes[1],
        cohorts=cohorts,
        labels=popularity_labels,
        counts_by_cohort=popularity_counts_by_cohort,
        display_labels={
            label: _popularity_bucket_display_label(label)
            for label in popularity_labels
        },
        colors=POPULARITY_BUCKET_COLORS,
        text_colors=POPULARITY_BUCKET_TEXT_COLORS,
        ylabel="Percentage of PRs (%)",
        show_ylabel=False,
        y_labelpad=3.0,
        y_tick_labelpad=1.0,
        xlabel=None,
        legend_title="Popularity",
        legend_title_fontsize=7.5,
    )
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].yaxis.label.set_size(7.5)
    fig.supxlabel("Cohort", y=-0.02, fontsize=7.5)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.55)
    stem = "language_popularity_by_cohort_100pct_stacked_bars"
    save_figure(fig, output_dir, stem, tight_layout_kwargs={"w_pad": 0.55})
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "layout": "1x2",
            "x_axis": "Cohort",
            "y_axis": "Percentage of PRs (%)",
            "axis_label_font_size": 7.5,
            **stacked_bar_visual_metadata(),
            "panels": {
                "language": {
                    "labels": list(LANGUAGE_ORDER),
                    "display_labels": LANGUAGE_LABELS,
                    "legend_title": "Programming Language",
                    "legend_title_font_size": 7.5,
                    "cohorts": _stacked_bar_panel_payload(
                        cohorts=cohorts,
                        labels=list(LANGUAGE_ORDER),
                        counts_by_cohort=language_counts_by_cohort,
                        count_key="pull_request_count",
                        total_key="total_pull_request_count",
                    ),
                },
                "popularity": {
                    "labels": popularity_labels,
                    "display_labels": {
                        label: _popularity_bucket_display_label(label)
                        for label in popularity_labels
                    },
                    "legend_title": "Popularity",
                    "legend_title_font_size": 7.5,
                    "cohorts": _stacked_bar_panel_payload(
                        cohorts=cohorts,
                        labels=popularity_labels,
                        counts_by_cohort=popularity_counts_by_cohort,
                        count_key="pull_request_count",
                        total_key="total_pull_request_count",
                    ),
                },
            },
        },
    )
    plt.close(fig)


def _cohort_numeric_values(
    con,
    *,
    metric_column: str,
    integer_values: bool = False,
) -> dict[str, list[float]]:
    rows = con.execute(
        f"""
        SELECT cohort, {metric_column}
        FROM analysis_prs
        WHERE {metric_column} IS NOT NULL
        ORDER BY cohort, {metric_column}
        """
    ).fetchall()
    by_cohort: dict[str, list[float]] = {}
    for cohort, value in rows:
        numeric = int(value or 0) if integer_values else float(value or 0.0)
        by_cohort.setdefault(str(cohort), []).append(float(numeric))
    return by_cohort


def _draw_cohort_boxplot(
    ax,
    by_cohort: dict[str, list[float]],
    *,
    ylabel: str,
    value_axis_lower_bound: float,
    value_axis_lower_padding: float,
    showmeans: bool = True,
    add_zero_baseline: bool = False,
    zero_tick_candidates: tuple[int, ...] | None = None,
    minimum_upper_padding: float = 0.1,
    y_labelpad: float | None = None,
    y_tick_labelpad: float | None = None,
    tight_x_margins: bool = False,
    xlabel: str | None = "Cohort",
) -> dict[str, float | int | bool] | None:
    cohorts = order_humans_first(by_cohort)
    if not cohorts:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if y_labelpad is None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(ylabel, labelpad=y_labelpad)
        ax.grid(axis="y", alpha=0.3)
        return None

    boxplot_kwargs = ieee_boxplot_kwargs()
    if not showmeans:
        boxplot_kwargs["showmeans"] = False
        boxplot_kwargs.pop("meanprops", None)
    colors = cohort_color_map(cohorts)
    ordered_colors = [colors[cohort] for cohort in cohorts]
    values_by_cohort = [by_cohort[cohort] for cohort in cohorts]
    add_violin_underlay(
        ax,
        values_by_cohort,
        colors=ordered_colors,
    )
    try:
        boxplot = ax.boxplot(
            values_by_cohort,
            tick_labels=display_group_labels(cohorts),
            **boxplot_kwargs,
        )
    except TypeError:
        boxplot = ax.boxplot(
            values_by_cohort,
            labels=display_group_labels(cohorts),
            **boxplot_kwargs,
        )
    style_ieee_boxplot(boxplot, ordered_colors)
    add_human_median_baseline(ax, by_cohort)
    y_axis_limits = apply_percentile_capped_y_axis(ax, by_cohort, show_note=False)
    visible_upper = ax.get_ylim()[1]
    ax.set_ylim(
        bottom=float(value_axis_lower_bound) - float(value_axis_lower_padding),
        top=max(
            float(visible_upper),
            float(value_axis_lower_bound) + float(minimum_upper_padding),
        ),
    )
    if add_zero_baseline:
        ax.axhline(
            value_axis_lower_bound,
            color="0.2",
            linewidth=0.45,
            alpha=0.45,
            zorder=1,
        )
    if zero_tick_candidates is not None:
        upper_tick_limit = max(
            float(visible_upper),
            float(value_axis_lower_bound) + float(minimum_upper_padding),
        )
        ax.set_yticks([tick for tick in zero_tick_candidates if tick <= upper_tick_limit])
    y_axis_scale = apply_symlog_y_axis_if_range_exceeds(ax)
    if y_axis_limits is not None:
        below_count = (
            0
            if value_axis_lower_bound == 0
            else sum(
                1
                for values in by_cohort.values()
                for value in values
                if float(value) < float(value_axis_lower_bound)
            )
        )
        y_axis_limits = {
            **y_axis_limits,
            "lower": float(value_axis_lower_bound),
            "visual_lower": float(value_axis_lower_bound)
            - float(value_axis_lower_padding),
            "below_count": below_count,
            "is_clipped": bool(
                below_count or int(y_axis_limits.get("above_count", 0))
            ),
            "scale": y_axis_scale,
            "scale_range_threshold": 100.0,
        }
    ax.tick_params(axis="x", rotation=0)
    if y_tick_labelpad is not None:
        ax.tick_params(axis="y", pad=y_tick_labelpad)
    if tight_x_margins:
        ax.set_xlim(
            1.0 - GRID_BOXPLOT_DEFAULT_HALF_WIDTH - GRID_BOXPLOT_X_MARGIN,
            len(cohorts) + GRID_BOXPLOT_DEFAULT_HALF_WIDTH + GRID_BOXPLOT_X_MARGIN,
        )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")
    if y_labelpad is None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(ylabel, labelpad=y_labelpad)
    ax.grid(axis="y", alpha=0.3)
    return y_axis_limits


def _scale_from_y_axis_limits(
    y_axis_limits: dict[str, Any] | None,
) -> str:
    if y_axis_limits is None:
        return "linear"
    return str(y_axis_limits.get("scale") or "linear")


def _plot_pr_size_boxplot(
    con,
    output_dir: Path,
    *,
    metric_column: str,
    ylabel: str,
    stem: str,
    value_axis_lower_bound: float = PR_SIZE_AXIS_LOWER_BOUND,
    value_axis_lower_padding: float = PR_SIZE_AXIS_LOWER_PADDING,
) -> None:
    plt, _mdates = require_matplotlib()
    by_cohort = _cohort_numeric_values(con, metric_column=metric_column)
    fig, ax = plt.subplots(
        figsize=(INDIVIDUAL_BOXPLOT_WIDTH, INDIVIDUAL_BOXPLOT_HEIGHT)
    )
    y_axis_limits = _draw_cohort_boxplot(
        ax,
        by_cohort,
        ylabel=ylabel,
        value_axis_lower_bound=value_axis_lower_bound,
        value_axis_lower_padding=value_axis_lower_padding,
    )
    ax.xaxis.set_label_coords(0.5, INDIVIDUAL_BOXPLOT_X_AXIS_LABEL_Y)
    fig.subplots_adjust(bottom=INDIVIDUAL_BOXPLOT_BOTTOM)
    cohorts = order_humans_first(by_cohort)
    save_figure(fig, output_dir, stem)
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "metric": metric_column,
            "x_axis": "Cohort",
            "y_axis": ylabel,
            "scale": _scale_from_y_axis_limits(y_axis_limits),
            "percentile_capped_y_axis": y_axis_limits,
            "figure": {
                "width_inches": INDIVIDUAL_BOXPLOT_WIDTH,
                "height_inches": INDIVIDUAL_BOXPLOT_HEIGHT,
                "bottom": INDIVIDUAL_BOXPLOT_BOTTOM,
                "x_axis_label_y": INDIVIDUAL_BOXPLOT_X_AXIS_LABEL_Y,
            },
            "groups": {
                cohort: _summary_payload(by_cohort.get(cohort, []))
                for cohort in cohorts
            },
        },
    )
    plt.close(fig)


def _popularity_bucket_display_label(label: str) -> str:
    return {
        "pop0": "Low",
        "pop1": "Medium",
        "pop2": "High",
    }.get(label, label)


def _plot_popularity_stargazer_boxplot(con, output_dir: Path) -> None:
    plt, _mdates = require_matplotlib()
    by_cohort = _cohort_numeric_values(
        con,
        metric_column="stargazer_count",
        integer_values=True,
    )
    fig, ax = plt.subplots(
        figsize=(INDIVIDUAL_BOXPLOT_WIDTH, INDIVIDUAL_BOXPLOT_HEIGHT)
    )
    y_axis_limits = _draw_cohort_boxplot(
        ax,
        by_cohort,
        ylabel="Repository Stars",
        value_axis_lower_bound=POPULARITY_AXIS_LOWER_BOUND,
        value_axis_lower_padding=POPULARITY_AXIS_LOWER_PADDING,
        showmeans=False,
        add_zero_baseline=True,
        zero_tick_candidates=(0, 1, 10, 100, 1000, 10000, 100000),
        minimum_upper_padding=1.1,
    )
    ax.xaxis.set_label_coords(0.5, INDIVIDUAL_BOXPLOT_X_AXIS_LABEL_Y)
    fig.subplots_adjust(bottom=INDIVIDUAL_BOXPLOT_BOTTOM)
    cohorts = order_humans_first(by_cohort)
    save_figure(fig, output_dir, "popularity_stargazer_boxplot")
    write_plot_data(
        output_dir,
        "popularity_stargazer_boxplot",
        {
            "plot": "popularity_stargazer_boxplot",
            "metric": "stargazer_count",
            "x_axis": "Cohort",
            "y_axis": "Repository Stars",
            "scale": _scale_from_y_axis_limits(y_axis_limits),
            "percentile_capped_y_axis": y_axis_limits,
            "figure": {
                "width_inches": INDIVIDUAL_BOXPLOT_WIDTH,
                "height_inches": INDIVIDUAL_BOXPLOT_HEIGHT,
                "bottom": INDIVIDUAL_BOXPLOT_BOTTOM,
                "x_axis_label_y": INDIVIDUAL_BOXPLOT_X_AXIS_LABEL_Y,
            },
            "groups": {
                cohort: _summary_payload(by_cohort.get(cohort, []))
                for cohort in cohorts
            },
        },
    )
    plt.close(fig)


def _plot_popularity_size_by_cohort_boxplots(con, output_dir: Path) -> None:
    plt, _mdates = require_matplotlib()
    panels = [
        {
            "key": "popularity",
            "metric": "stargazer_count",
            "metric_column": "stargazer_count",
            "ylabel": "Repository Stars",
            "value_axis_lower_bound": POPULARITY_AXIS_LOWER_BOUND,
            "value_axis_lower_padding": POPULARITY_AXIS_LOWER_PADDING,
            "showmeans": False,
            "add_zero_baseline": True,
            "zero_tick_candidates": (0, 1, 10, 100, 1000, 10000, 100000),
            "minimum_upper_padding": 1.1,
            "integer_values": True,
        },
        {
            "key": "changed_files",
            "metric": "changed_files_count",
            "metric_column": "changed_files_count",
            "ylabel": "Changed Files",
            "value_axis_lower_bound": PR_SIZE_AXIS_LOWER_BOUND,
            "value_axis_lower_padding": PR_SIZE_AXIS_LOWER_PADDING,
            "showmeans": True,
            "add_zero_baseline": False,
            "zero_tick_candidates": None,
            "minimum_upper_padding": 0.1,
            "integer_values": False,
        },
        {
            "key": "changed_lines",
            "metric": "changed_line_count",
            "metric_column": "changed_line_count",
            "ylabel": "Changed Lines",
            "value_axis_lower_bound": 0.0,
            "value_axis_lower_padding": 0.1,
            "showmeans": True,
            "add_zero_baseline": False,
            "zero_tick_candidates": None,
            "minimum_upper_padding": 0.1,
            "integer_values": False,
        },
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.5))
    plot_payload: dict[str, Any] = {
        "plot": "popularity_size_by_cohort_boxplots",
        "layout": "1x3",
        "x_axis": "Cohort",
        "shared_x_axis_label": True,
        "axis_label_font_size": 7.5,
        "x_tick_label_font_size": 6.5,
        "panels": {},
    }
    for ax, panel in zip(axes, panels):
        by_cohort = _cohort_numeric_values(
            con,
            metric_column=str(panel["metric_column"]),
            integer_values=bool(panel["integer_values"]),
        )
        y_axis_limits = _draw_cohort_boxplot(
            ax,
            by_cohort,
            ylabel=str(panel["ylabel"]),
            value_axis_lower_bound=float(panel["value_axis_lower_bound"]),
            value_axis_lower_padding=float(panel["value_axis_lower_padding"]),
            showmeans=bool(panel["showmeans"]),
            add_zero_baseline=bool(panel["add_zero_baseline"]),
            zero_tick_candidates=panel["zero_tick_candidates"],
            minimum_upper_padding=float(panel["minimum_upper_padding"]),
            y_labelpad=3.0,
            y_tick_labelpad=1.0,
            tight_x_margins=True,
            xlabel=None,
        )
        ax.tick_params(axis="x", labelsize=6.5)
        ax.yaxis.label.set_size(7.5)
        cohorts = order_humans_first(by_cohort)
        plot_payload["panels"][str(panel["key"])] = {
            "metric": panel["metric"],
            "y_axis": panel["ylabel"],
            "scale": _scale_from_y_axis_limits(y_axis_limits),
            "percentile_capped_y_axis": y_axis_limits,
            "groups": {
                cohort: _summary_payload(by_cohort.get(cohort, []))
                for cohort in cohorts
            },
        }
    fig.supxlabel("Cohort", y=-0.02, fontsize=7.5)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.28, top=0.947, wspace=0.45)
    stem = "popularity_size_by_cohort_boxplots"
    save_figure(fig, output_dir, stem, tight_layout_kwargs={"w_pad": 0.45})
    write_plot_data(output_dir, stem, plot_payload)
    plt.close(fig)


def _plot_popularity_bucket_100pct_stacked_bars(
    con,
    scheme: dict[str, Any],
    output_dir: Path,
) -> None:
    plt, _mdates = require_matplotlib()
    labels = [str(label) for label in scheme.get("bucket_labels") or []]
    cohorts = _cohorts(con)
    counts_by_cohort = _popularity_counts_by_cohort(con, scheme)
    flat_counts_by_cohort = _flat_popularity_counts_by_cohort(con, scheme)
    true_by_cohort, visual_by_cohort = _stacked_bar_percentage_maps(
        cohorts=cohorts,
        labels=labels,
        counts_by_cohort=flat_counts_by_cohort,
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    bottoms = [0.0 for _cohort in cohorts]
    x_values = list(range(len(cohorts)))
    for label in labels:
        percentages = [
            true_by_cohort[cohort].get(label, 0.0)
            for cohort in cohorts
        ]
        visual_percentages = [
            visual_by_cohort[cohort].get(label, 0.0)
            for cohort in cohorts
        ]
        ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=_popularity_bucket_display_label(label),
            color=POPULARITY_BUCKET_COLORS.get(label, "#56B4E9"),
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
            label_y = (
                bottom
                + visual_percentage / 2.0
                + STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET
            )
            percentage_label = f"{percentage:.1f}%"
            ax.text(
                x_value,
                label_y,
                percentage_label,
                ha="center",
                va="center",
                fontsize=STACKED_BAR_PERCENTAGE_FONT_SIZE,
                color=POPULARITY_BUCKET_TEXT_COLORS.get(label, "black"),
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Percentage of PRs (%)")
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(cohorts), rotation=0, ha="center")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(cohorts) - 0.5))
    ax.legend(
        frameon=False,
        ncol=len(labels),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.35,
    )
    save_figure(fig, output_dir, "popularity_bucket_100pct_stacked_bars")
    write_plot_data(
        output_dir,
        "popularity_bucket_100pct_stacked_bars",
        {
            "plot": "popularity_bucket_100pct_stacked_bars",
            "x_axis": "Cohort",
            "y_axis": "Percentage of PRs (%)",
            **stacked_bar_visual_metadata(),
            "bucket_labels": labels,
            "bucket_display_labels": {
                label: _popularity_bucket_display_label(label)
                for label in labels
            },
            "cohorts": {
                cohort: {
                    "total_pull_request_count": int(
                        sum(
                            counts_by_cohort
                            .get(cohort, {})
                            .get(bucket, {})
                            .get("pull_request_count", 0)
                            for bucket in labels
                        )
                    ),
                    "buckets": _count_percentage_payload(
                        flat_counts_by_cohort.get(cohort, {}),
                        labels,
                        count_key="pull_request_count",
                    ),
                }
                for cohort in cohorts
            },
        },
    )
    plt.close(fig)


def _plot_longitudinal_attrition_by_cohort(con, output_dir: Path) -> None:
    cohorts = _cohorts(con)
    if not cohorts:
        return
    rows = con.execute(
        """
        SELECT
            cohort,
            timepoint_label,
            days_after_merge,
            COUNT(*) AS total_longitudinal_pull_request_count,
            SUM(CASE WHEN snapshot_available THEN 1 ELSE 0 END)
                AS available_pull_request_count
        FROM analysis_dataset_longitudinal_snapshots
        GROUP BY cohort, timepoint_label, days_after_merge
        ORDER BY cohort, days_after_merge
        """
    ).fetchall()
    by_cohort: dict[str, dict[str, dict[str, Any]]] = {
        cohort: {
            label: {
                "days_after_merge": LONGITUDINAL_TIMEPOINT_DAYS[label],
                "pull_request_count": 0,
                "baseline_pull_request_count": 0,
                "total_longitudinal_pull_request_count": 0,
                "retention_proportion": 0.0,
                "retention_percentage": 0.0,
            }
            for label in LONGITUDINAL_TIMEPOINTS
        }
        for cohort in cohorts
    }
    for (
        cohort,
        timepoint_label,
        days_after_merge,
        total_count,
        available_count,
    ) in rows:
        cohort_key = str(cohort)
        if cohort_key not in by_cohort:
            continue
        label = str(timepoint_label)
        if label not in by_cohort[cohort_key]:
            continue
        total = int(total_count or 0)
        available = int(available_count or 0)
        percentage = 100.0 * available / total if total else 0.0
        by_cohort[cohort_key][label] = {
            "days_after_merge": int(
                days_after_merge or LONGITUDINAL_TIMEPOINT_DAYS[label]
            ),
            "pull_request_count": available,
            "baseline_pull_request_count": total,
            "total_longitudinal_pull_request_count": total,
            "retention_proportion": percentage / 100.0,
            "retention_percentage": percentage,
        }
    payload = {
        "plot": "longitudinal_attrition_by_cohort",
        "plot_type": "longitudinal_attrition_line_plot",
        "x_axis": "Time After Merge (Days)",
        "x_axis_scale": "ordinal_timepoints",
        "x_axis_padding": LONGITUDINAL_X_AXIS_PADDING,
        "x_axis_positions": TIMEPOINT_X_POSITIONS,
        "x_axis_tick_labels": _timepoint_tick_labels(),
        "cohorts": cohorts,
        "cohort_colors": _resolved_cohort_colors(cohorts),
        "line_width": ATTRITION_LINE_WIDTH,
        "marker_size": ATTRITION_MARKER_SIZE,
        "figure": {
            "width_inches": 3.5,
            "height_inches": ATTRITION_FIGURE_HEIGHT,
        },
        "y_axis": "PRs with Available Snapshot (%)",
        "y_value_field": "retention_percentage",
        "y_value_unit": "percent",
        "numerator": "pull_requests_with_available_snapshot",
        "denominator": "total_longitudinal_pull_request_count",
        "logical_lower_bound": ATTRITION_AXIS_LOWER_BOUND,
        "logical_upper_bound": ATTRITION_AXIS_UPPER_BOUND,
        "visual_lower_padding": ATTRITION_AXIS_LOWER_PADDING,
        "visual_lower": ATTRITION_AXIS_LOWER_BOUND - ATTRITION_AXIS_LOWER_PADDING,
        "visual_upper_padding": ATTRITION_AXIS_UPPER_PADDING,
        "visual_upper": ATTRITION_AXIS_UPPER_BOUND + ATTRITION_AXIS_UPPER_PADDING,
        "availability_sources": [
            "analysis_dataset_longitudinal_snapshots",
        ],
        "availability_rule": (
            "0d is available for every longitudinal PR; future timepoints are "
            "available when either compact future snapshot source reports "
            "available, snapshot_available, or snapshot_commit."
        ),
        "timepoints": list(LONGITUDINAL_TIMEPOINTS),
        "by_cohort": by_cohort,
    }
    stem = "longitudinal_attrition_by_cohort"
    write_plot_data(output_dir, stem, payload)
    render_longitudinal_attrition_plot_from_payload(payload, output_dir, stem)


def write_data_analysis_plots(
    con,
    scheme: dict[str, Any],
    output_dir: Path,
    logger: Any | None = None,
) -> None:
    """Write the full dataset-composition and sampling diagnostics plot set."""
    apply_ieee_plot_style()
    if logger is not None:
        logger.log("writing_dataset_creation_time_plots")
    _plot_creation_time_weekly_cumulative_counts_by_cohort(con, output_dir)
    _plot_creation_time_hour_of_day_distribution_overall(con, output_dir)
    release_process_memory(logger, stage="dataset_creation_time_plot_memory_released")
    if logger is not None:
        logger.log("writing_dataset_composition_stacked_bars")
    _plot_language_by_cohort_100pct_stacked_bars(con, output_dir)
    _plot_domain_by_cohort_100pct_stacked_bars(
        con,
        output_dir,
        count_kind="pull_request",
        ylabel="Percentage of PRs (%)",
        stem="domain_prs_by_cohort_100pct_stacked_bars",
        total_key="total_pull_request_count",
        count_key="pull_request_count",
        percentage_label_y_offset=STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET,
        legend_columns=3,
        legend_fontsize=5.5,
    )
    _plot_domain_by_cohort_100pct_stacked_bars(
        con,
        output_dir,
        count_kind="repository",
        ylabel="Percentage of Repos (%)",
        stem="domain_repos_by_cohort_100pct_stacked_bars",
        total_key="total_repository_count",
        count_key="repository_count",
        percentage_label_y_offset=STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET,
        legend_columns=3,
        legend_fontsize=5.5,
    )
    release_process_memory(logger, stage="dataset_composition_plot_memory_released")
    if logger is not None:
        logger.log("writing_dataset_domain_confidence_boxplot")
    _plot_domain_confidence_boxplot_by_domain(con, output_dir)
    _plot_language_popularity_by_cohort_100pct_stacked_bars(con, scheme, output_dir)
    release_process_memory(logger, stage="dataset_domain_plot_memory_released")
    if logger is not None:
        logger.log("writing_dataset_size_and_popularity_boxplots")
    _plot_pr_size_boxplot(
        con,
        output_dir,
        metric_column="changed_files_count",
        ylabel="Changed Files",
        stem="pr_size_changed_files_boxplot_by_cohort",
    )
    _plot_pr_size_boxplot(
        con,
        output_dir,
        metric_column="changed_line_count",
        ylabel="Changed Lines",
        stem="pr_size_changed_lines_boxplot_by_cohort",
        value_axis_lower_bound=0.0,
        value_axis_lower_padding=0.1,
    )
    _plot_popularity_stargazer_boxplot(con, output_dir)
    _plot_popularity_size_by_cohort_boxplots(con, output_dir)
    release_process_memory(logger, stage="dataset_size_plot_memory_released")
    if logger is not None:
        logger.log("writing_dataset_popularity_stacked_bars")
    _plot_popularity_bucket_100pct_stacked_bars(con, scheme, output_dir)
    release_process_memory(logger, stage="dataset_popularity_plot_memory_released")
    if logger is not None:
        logger.log("writing_dataset_longitudinal_attrition")
    _plot_longitudinal_attrition_by_cohort(con, output_dir)
    release_process_memory(logger, stage="dataset_attrition_plot_memory_released")


def write_data_analysis_plots_from_payload(
    payload: dict[str, Any],
    scheme: dict[str, Any],
    output_dir: Path,
    logger: Any | None = None,
) -> None:
    """Render dataset plots from compact streaming payloads."""
    write_data_analysis_plots(
        _DatasetPlotPayloadConnection(payload),
        scheme,
        output_dir,
        logger=logger,
    )
