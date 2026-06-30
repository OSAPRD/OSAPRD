"""Plots for refactoring analysis outputs.

Refactoring plots cover tool coverage, Murphy-Hill category composition, and
distributional summaries for operation counts, density, diversity, and line
impact. Payload adapters keep plot generation independent from a database
runtime.
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
from plotting_utility import (
    add_human_median_baseline,
    add_xtick_count_sublabels,
    add_violin_underlay,
    apply_percentile_capped_y_axis,
    apply_ieee_plot_style,
    cohort_color_map,
    display_group_label,
    display_group_labels,
    ieee_boxplot_kwargs,
    order_humans_first,
    require_matplotlib,
    remove_plot_outputs,
    save_figure,
    apply_symlog_y_axis_if_range_exceeds,
    stacked_bar_visual_metadata,
    stacked_bar_visual_percentages,
    style_ieee_boxplot,
    write_plot_data,
)
from refactoring_analysis_utility import (
    REFACTORING_DISTRIBUTION_METRICS,
    REFACTORING_NUMERIC_METRICS,
)


MURPHY_HILL_PLOT_LEVELS = ("low", "medium", "high")
MURPHY_HILL_PLOT_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}
MURPHY_HILL_PLOT_COLORS = {
    "low": "#C9C6C6",
    "medium": "#0072B2",
    "high": "#D55E00",
}
MURPHY_HILL_PLOT_TEXT_COLORS = {
    "low": "black",
    "medium": "white",
    "high": "white",
}
STACKED_BAR_PERCENTAGE_FONT_SIZE = 6.0
STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET = -0.2
STACKED_BAR_COUNT_LABEL_FONT_SIZE = 5.5
STACKED_BAR_COUNT_LABEL_Y = -0.16
STACKED_BAR_WITH_PR_COUNT_X_LABELPAD = 24
MURPHY_HILL_DISTRIBUTION_FIGSIZE = (3.5, 2.0)
REFACTORING_BOXPLOT_AXIS_LOWER_BOUND = 0.0
REFACTORING_BOXPLOT_AXIS_LOWER_PADDING = 0.1
REFACTORING_BOXPLOT_MINIMUM_UPPER_PADDING = 0.1
REFACTORING_BOXPLOT_P_THRESHOLD = 0.001
GRID_BOXPLOT_X_MARGIN = 0.1
GRID_BOXPLOT_DEFAULT_HALF_WIDTH = 0.25
REFACTORING_BOXPLOT_METRIC_LOWER_BOUNDS = {
    "RefCount": 1.0,
}
REFACTORING_BOXPLOT_METRIC_LOWER_PADDING = {
    "RefCount": 0.1,
}
GROUP_FIELD_SQL = {
    "cohort": "cohort",
    "authorship_group": "authorship_group",
}
METRIC_SQL = {metric: f'"{metric}"' for metric in REFACTORING_NUMERIC_METRICS}
METRIC_AXIS_LABELS = {
    "RefCount": "RefOp Count",
    "RefDensity": "RefOps per KLOC",
    "RefDiversity": "RefOp Diversity",
    "RefMagLines": "Lines Changed per RefOp",
    "RefAdded": "Lines Added per RefOp",
    "RefAddedLines": "Lines Added per RefOp",
    "RefRemoved": "Lines Deleted per RefOp",
    "RefDeletedLines": "Lines Deleted per RefOp",
}


def _metric_stem(metric_name: str) -> str:
    return metric_name.lower()


def _metric_axis_label(metric_name: str) -> str:
    return METRIC_AXIS_LABELS.get(metric_name, metric_name)


PLOT_OUTPUT_STEMS = (
    "murphy_hill_distribution_by_cohort",
    *(
        f"boxplots/{_metric_stem(metric)}_boxplot_by_cohort"
        for metric in REFACTORING_NUMERIC_METRICS
    ),
    "boxplots-grid/refactoring_metrics_boxplots_by_cohort",
    "boxplots-grid/refadded_refremoved_boxplots_by_cohort",
)
REMOVED_PLOT_OUTPUT_STEMS = (
    "tool_coverage_by_cohort",
    "tool_coverage_by_authorship",
    "murphy_hill_distribution_by_authorship",
    "boxplots/refactoring_metrics_boxplots_by_cohort",
    "boxplots/refadded_refremoved_boxplots_by_cohort",
    *(
        f"boxplots/{_metric_stem(metric)}_boxplot_by_authorship"
        for metric in REFACTORING_NUMERIC_METRICS
    ),
    *(
        f"violin_plots/{_metric_stem(metric)}_{plot_type}_by_cohort"
        for metric in REFACTORING_DISTRIBUTION_METRICS
        for plot_type in ("violin",)
    ),
    *(
        f"violin_plots/{_metric_stem(metric)}_{plot_type}_by_authorship"
        for metric in REFACTORING_DISTRIBUTION_METRICS
        for plot_type in ("violin",)
    ),
)


def _format_plot_stat(value: float | None) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.2f}"


def _format_p_value(value: float | None) -> str:
    if value is None:
        return "p = NA"
    if value < 0.001:
        return "p < 0.001"
    return "p > 0.001"


def _format_effect_size(value: float | None) -> str:
    if value is None:
        return "\u03b4 = NA"
    return f"\u03b4 = {value:.2f}"


def _format_effect_size_compact(value: float | None) -> str:
    if value is None:
        return "\u03b4 = NA"
    return f"\u03b4 = {value:.2f}"


def _format_adjusted_p_threshold(value: float | None) -> str:
    if value is None:
        return "p = NA"
    if float(value) <= REFACTORING_BOXPLOT_P_THRESHOLD:
        return f"p \u2264 {REFACTORING_BOXPLOT_P_THRESHOLD:.3f}"
    return f"p > {REFACTORING_BOXPLOT_P_THRESHOLD:.3f}"


def _format_adjusted_p_threshold_compact(value: float | None) -> str:
    if value is None:
        return "p = NA"
    if float(value) <= REFACTORING_BOXPLOT_P_THRESHOLD:
        return f"p \u2264 {REFACTORING_BOXPLOT_P_THRESHOLD:.3f}"
    return f"p > {REFACTORING_BOXPLOT_P_THRESHOLD:.3f}"


def _boxplot_axis_lower_bound(metric_name: str) -> float:
    return REFACTORING_BOXPLOT_METRIC_LOWER_BOUNDS.get(
        metric_name,
        REFACTORING_BOXPLOT_AXIS_LOWER_BOUND,
    )


def _boxplot_axis_lower_padding(metric_name: str) -> float:
    if metric_name in REFACTORING_BOXPLOT_METRIC_LOWER_PADDING:
        return REFACTORING_BOXPLOT_METRIC_LOWER_PADDING[metric_name]
    return REFACTORING_BOXPLOT_AXIS_LOWER_PADDING


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


def _comparison_tick_labels(
    groups: list[str],
    grouped_values: dict[str, list[float]],
    group_field: str,
) -> list[str]:
    """Compatibility wrapper for local callers using raw values."""
    return _boxplot_comparison_tick_labels(
        groups,
        _comparison_payloads(groups, grouped_values, group_field),
    )


def _boxplot_comparison_stat_labels(
    groups: list[str],
    comparison_payload: dict[str, object] | None,
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
        labels.append(
            f"{_format_effect_size_compact(test.get('cliffs_delta'))}\n"
            f"{_format_adjusted_p_threshold_compact(test.get('adjusted_p_value'))}"
        )
    return labels


def _plot_p_value(test: dict[str, object]) -> float | None:
    value = test.get("adjusted_p_value")
    return None if value is None else float(value)


def _murphy_hill_comparison_tick_labels(
    groups: list[str],
    grouped: dict[str, dict[str, list[float]]],
    group_field: str,
) -> list[str]:
    baseline_group = _human_baseline_group(groups)
    labels = []
    for group in groups:
        group_label = display_group_label(group)
        if group == baseline_group:
            labels.append(f"{group_label}\n-")
            continue
        level_lines = []
        for level in MURPHY_HILL_PLOT_LEVELS:
            prefix = level[0].upper()
            if baseline_group is None:
                level_lines.append(f"{prefix}: \u03b4 = NA, p = NA")
                continue
            test = mann_whitney_u_test(
                grouped.get(group, {}).get(level, []),
                grouped.get(baseline_group, {}).get(level, []),
            )
            level_lines.append(
                f"{prefix}: {_format_effect_size(test['cliffs_delta'])}, "
                f"{_format_p_value(test['p_value'])}"
            )
        labels.append(f"{group_label}\n" + "\n".join(level_lines))
    return labels


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
        raise ValueError(f"Unknown refactoring group field: {group_field}") from exc


def _metric_sql(metric_name: str) -> str:
    try:
        return METRIC_SQL[metric_name]
    except KeyError as exc:
        raise ValueError(f"Unknown refactoring metric: {metric_name}") from exc


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
        FROM analysis_refactoring_success_prs
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


class _RefactoringPlotPayloadConnection:
    """Adapter for rendering existing refactoring plots from compact payloads."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def execute(self, sql: str) -> _PayloadQueryResult:
        normalized_sql = " ".join(str(sql).strip().lower().split())
        if "select distinct" in normalized_sql and "from analysis_refactoring_success_prs" in normalized_sql:
            return _PayloadQueryResult([(group,) for group in self._groups(_field_from_sql(normalized_sql))])
        if "with per_pr as" in normalized_sql and "analysis_refactoring_murphy_hill_counts" in normalized_sql:
            return _PayloadQueryResult(self._per_pr_murphy_rows(_field_from_sql(normalized_sql)))
        if "from analysis_refactoring_murphy_hill_counts" in normalized_sql and "sum(refop_count)" in normalized_sql:
            return _PayloadQueryResult(self._aggregate_murphy_rows(_field_from_sql(normalized_sql)))
        if "count(*)" in normalized_sql and "from analysis_refactoring_success_prs" in normalized_sql:
            return _PayloadQueryResult(self._eligible_pr_count_rows(_field_from_sql(normalized_sql)))
        if "from analysis_refactoring_success_prs" in normalized_sql:
            field = _field_from_sql(normalized_sql)
            metric = _metric_from_sql(normalized_sql)
            if metric is not None:
                refactored_only = '"refcount" > 0' in normalized_sql or "prs.\"refcount\" > 0" in normalized_sql
                return _PayloadQueryResult(self._metric_rows(field, metric, refactored_only=refactored_only))
        raise ValueError(f"Unsupported refactoring plot payload query: {sql}")

    def _groups(self, field: str) -> list[str]:
        scopes = self.payload.get("metric_values_by_scope") or {}
        ref_count_scopes = scopes.get("RefCount", {}) if isinstance(scopes, dict) else {}
        prefix = f"{field}:"
        groups = [
            str(scope)[len(prefix) :]
            for scope in ref_count_scopes
            if str(scope).startswith(prefix)
        ]
        return order_humans_first(groups)

    def _metric_rows(
        self,
        field: str,
        metric: str,
        *,
        refactored_only: bool,
    ) -> list[tuple[object, ...]]:
        metric_values = self.payload.get("metric_values_by_scope") or {}
        if refactored_only:
            refop_positive_metric_values = (
                self.payload.get("refop_positive_metric_values_by_scope") or {}
            )
            positive_by_scope = (
                refop_positive_metric_values.get(metric, {})
                if isinstance(refop_positive_metric_values, dict)
                else {}
            )
            if positive_by_scope:
                rows: list[tuple[object, ...]] = []
                for group in self._groups(field):
                    scope = f"{field}:{group}"
                    for value in positive_by_scope.get(scope, []):
                        rows.append((group, float(value)))
                return rows
        by_scope = metric_values.get(metric, {}) if isinstance(metric_values, dict) else {}
        ref_counts = metric_values.get("RefCount", {}) if isinstance(metric_values, dict) else {}
        rows: list[tuple[object, ...]] = []
        for group in self._groups(field):
            scope = f"{field}:{group}"
            values = list(by_scope.get(scope, []))
            if refactored_only:
                ref_values = list(ref_counts.get(scope, []))
                values = [
                    value
                    for value, ref_count in zip(values, ref_values)
                    if float(ref_count or 0.0) > 0.0
                ]
            for value in values:
                rows.append((group, float(value)))
        return rows

    def _aggregate_murphy_rows(self, field: str) -> list[tuple[object, ...]]:
        counts = self.payload.get("murphy_counts_by_scope") or {}
        rows: list[tuple[object, ...]] = []
        for group in self._groups(field):
            scope = f"{field}:{group}"
            group_counts = counts.get(scope, {}) if isinstance(counts, dict) else {}
            for level in MURPHY_HILL_PLOT_LEVELS:
                rows.append((group, level, int(group_counts.get(level, 0) or 0)))
        return rows

    def _per_pr_murphy_rows(self, field: str) -> list[tuple[object, ...]]:
        counts = self.payload.get("per_pr_murphy_counts_by_scope") or {}
        rows: list[tuple[object, ...]] = []
        for group in self._groups(field):
            scope = f"{field}:{group}"
            by_level = counts.get(scope, {}) if isinstance(counts, dict) else {}
            max_count = max((len(by_level.get(level, [])) for level in MURPHY_HILL_PLOT_LEVELS), default=0)
            for index in range(max_count):
                rows.append(
                    (
                        group,
                        float(_list_value(by_level.get("low", []), index)),
                        float(_list_value(by_level.get("medium", []), index)),
                        float(_list_value(by_level.get("high", []), index)),
                    )
                )
        return rows

    def _eligible_pr_count_rows(self, field: str) -> list[tuple[object, ...]]:
        scopes = self.payload.get("metric_values_by_scope") or {}
        ref_counts = scopes.get("RefCount", {}) if isinstance(scopes, dict) else {}
        rows: list[tuple[object, ...]] = []
        for group in self._groups(field):
            scope = f"{field}:{group}"
            values = ref_counts.get(scope, []) if isinstance(ref_counts, dict) else []
            rows.append((group, len(values)))
        return rows


def _field_from_sql(sql: str) -> str:
    if "authorship_group" in sql:
        return "authorship_group"
    return "cohort"


def _metric_from_sql(sql: str) -> str | None:
    selected_columns_match = re.search(
        r"\bselect\s+.+?,\s+(.*?)\s+from\s+analysis_refactoring_success_prs\b",
        sql,
    )
    selected_metric_sql = selected_columns_match.group(1).strip() if selected_columns_match else sql
    selected_metric_name = selected_metric_sql.strip().strip('"')
    for metric in REFACTORING_NUMERIC_METRICS:
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
    *,
    refactored_only: bool = False,
) -> dict[str, list[float]]:
    group_column = _group_sql(group_field)
    metric_column = _metric_sql(metric_name)
    refactored_filter = ' AND "RefCount" > 0' if refactored_only else ""
    grouped: dict[str, list[float]] = {}
    rows = con.execute(
        f"""
        SELECT {group_column}, {metric_column}
        FROM analysis_refactoring_success_prs
        WHERE {_nonempty_sql(group_column)}
          {refactored_filter}
        ORDER BY {group_column}, {metric_column}
        """
    ).fetchall()
    for group, value in rows:
        grouped.setdefault(str(group), []).append(float(value or 0.0))
    return {group: grouped[group] for group in order_humans_first(grouped)}


def _nonempty_groups(
    con,
    group_field: str,
    metric_name: str,
    *,
    refactored_only: bool = False,
) -> dict[str, list[float]]:
    return {
        group: values
        for group, values in _group_values(
            con,
            group_field,
            metric_name,
            refactored_only=refactored_only,
        ).items()
        if values
    }


def _murphy_hill_level_count_groups(
    con,
    group_field: str,
) -> dict[str, dict[str, list[float]]]:
    group_column = _group_sql(group_field)
    level_selects = ",\n".join(
        f"""
                COALESCE(
                    SUM(
                        CASE
                            WHEN counts.murphy_hill_level = '{level}'
                            THEN counts.refop_count
                            ELSE 0
                        END
                    ),
                    0
                ) AS {level}_count"""
        for level in MURPHY_HILL_PLOT_LEVELS
    )
    rows = con.execute(
        f"""
        WITH per_pr AS (
            SELECT
                prs.analysis_row_id,
                prs.{group_column} AS group_key,
                {level_selects}
            FROM analysis_refactoring_success_prs AS prs
            LEFT JOIN analysis_refactoring_murphy_hill_counts AS counts
                ON prs.analysis_row_id = counts.analysis_row_id
            WHERE {_nonempty_sql(f"prs.{group_column}")}
              AND prs."RefCount" > 0
            GROUP BY prs.analysis_row_id, group_key
        )
        SELECT
            group_key,
            low_count,
            medium_count,
            high_count
        FROM per_pr
        ORDER BY group_key, analysis_row_id
        """
    ).fetchall()
    grouped: dict[str, dict[str, list[float]]] = {}
    for group, low_count, medium_count, high_count in rows:
        group_key = str(group)
        grouped.setdefault(
            group_key,
            {level: [] for level in MURPHY_HILL_PLOT_LEVELS},
        )
        grouped[group_key]["low"].append(float(low_count or 0.0))
        grouped[group_key]["medium"].append(float(medium_count or 0.0))
        grouped[group_key]["high"].append(float(high_count or 0.0))
    return {
        group: grouped[group]
        for group in order_humans_first(grouped)
    }


def _eligible_pr_counts_by_group(con, group_field: str) -> dict[str, int]:
    group_column = _group_sql(group_field)
    rows = con.execute(
        f"""
        SELECT {group_column} AS group_key, COUNT(*) AS pull_request_count
        FROM analysis_refactoring_success_prs
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


def _scale_from_y_axis_limits(y_axis_limits: dict[str, object] | None) -> str:
    if y_axis_limits is None:
        return "linear"
    return str(y_axis_limits.get("scale") or "linear")


def _plot_murphy_hill_distribution(
    con,
    group_field: str,
    output_dir: Path,
    stem: str,
) -> None:
    plt, _mdates = require_matplotlib()
    group_column = _group_sql(group_field)
    groups = _groups_for_field(con, group_field)
    counts = {
        group: {level: 0 for level in MURPHY_HILL_PLOT_LEVELS}
        for group in groups
    }
    rows = con.execute(
        f"""
        SELECT
            {group_column} AS group_key,
            murphy_hill_level,
            COALESCE(SUM(refop_count), 0) AS refop_count
        FROM analysis_refactoring_murphy_hill_counts
        WHERE {_nonempty_sql(group_column)}
        GROUP BY group_key, murphy_hill_level
        ORDER BY group_key, murphy_hill_level
        """
    ).fetchall()
    for group, level, count in rows:
        if group is None or level is None or str(group) not in counts:
            continue
        if str(level) in counts[str(group)]:
            counts[str(group)][str(level)] += int(count or 0)

    group_totals = {
        group: int(sum(counts[group].values()))
        for group in groups
    }
    eligible_pr_counts = _eligible_pr_counts_by_group(con, group_field)
    fig, ax = plt.subplots(figsize=MURPHY_HILL_DISTRIBUTION_FIGSIZE)
    x_values = list(range(len(groups)))
    bottoms = [0.0 for _group in groups]
    true_by_group: dict[str, dict[str, float]] = {}
    visual_by_group: dict[str, dict[str, float]] = {}
    for group in groups:
        total = sum(counts[group].values())
        percentages = [
            100.0 * counts[group][level] / total if total else 0.0
            for level in MURPHY_HILL_PLOT_LEVELS
        ]
        visual_percentages = stacked_bar_visual_percentages(percentages)
        true_by_group[group] = {
            level: percentages[index]
            for index, level in enumerate(MURPHY_HILL_PLOT_LEVELS)
        }
        visual_by_group[group] = {
            level: visual_percentages[index]
            for index, level in enumerate(MURPHY_HILL_PLOT_LEVELS)
        }
    for level in MURPHY_HILL_PLOT_LEVELS:
        percentages = [
            true_by_group[group].get(level, 0.0)
            for group in groups
        ]
        visual_percentages = [
            visual_by_group[group].get(level, 0.0)
            for group in groups
        ]
        previous_bottoms = list(bottoms)
        ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=MURPHY_HILL_PLOT_LABELS[level],
            color=MURPHY_HILL_PLOT_COLORS[level],
            edgecolor="black",
            linewidth=0.3,
        )
        for x_value, bottom, percentage, visual_percentage in zip(
            x_values,
            previous_bottoms,
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
                color=MURPHY_HILL_PLOT_TEXT_COLORS[level],
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    ax.set_xlabel(
        "Cohort" if group_field == "cohort" else "Authorship Group",
        labelpad=STACKED_BAR_WITH_PR_COUNT_X_LABELPAD,
    )
    ax.set_ylabel("Percentage of RefOps (%)")
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(groups), rotation=0, ha="center")
    add_xtick_count_sublabels(
        ax,
        x_values,
        [group_totals[group] for group in groups],
        font_size=STACKED_BAR_COUNT_LABEL_FONT_SIZE,
        y=STACKED_BAR_COUNT_LABEL_Y,
        secondary_counts=[eligible_pr_counts.get(group, 0) for group in groups],
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(groups) - 0.5))
    ax.legend(
        frameon=False,
        ncol=len(MURPHY_HILL_PLOT_LEVELS),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.35,
        fontsize=7.0,
    )
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, output_dir, stem)
    per_pr_counts = _murphy_hill_level_count_groups(con, group_field)
    write_plot_data(
        output_dir,
        stem,
        {
            "plot": stem,
            "plot_type": "murphy_hill_distribution_stacked_bars",
            "group_field": group_field,
            "x_axis": "Cohort" if group_field == "cohort" else "Authorship Group",
            "y_axis": "Percentage of RefOps (%)",
            "figure": {
                "width_inches": MURPHY_HILL_DISTRIBUTION_FIGSIZE[0],
                "height_inches": MURPHY_HILL_DISTRIBUTION_FIGSIZE[1],
            },
            "x_tick_count_label_font_size": STACKED_BAR_COUNT_LABEL_FONT_SIZE,
            "x_tick_count_labels": {
                group: f"n = {group_totals[group]:,}"
                for group in groups
            },
            "x_tick_pr_count_labels": {
                group: f"p = {eligible_pr_counts.get(group, 0):,}"
                for group in groups
            },
            "x_tick_pr_count_label_font_size": STACKED_BAR_COUNT_LABEL_FONT_SIZE,
            **stacked_bar_visual_metadata(),
            "levels": list(MURPHY_HILL_PLOT_LEVELS),
            "groups": {
                group: {
                    "total_refop_count": group_totals[group],
                    "eligible_pull_request_count": eligible_pr_counts.get(group, 0),
                    "levels": {
                        level: {
                            "refop_count": int(counts[group][level]),
                            "percentage": true_by_group[group].get(level, 0.0),
                            "visual_percentage": visual_by_group[group].get(
                                level,
                                0.0,
                            ),
                            "per_pr_summary": _summary_payload(
                                per_pr_counts.get(group, {}).get(level, [])
                            ),
                        }
                        for level in MURPHY_HILL_PLOT_LEVELS
                    },
                }
                for group in groups
            },
            "human_baseline_tests_by_level": {
                level: _comparison_payloads(
                    groups,
                    {
                        group: per_pr_counts.get(group, {}).get(level, [])
                        for group in groups
                    },
                    group_field,
                )
                for level in MURPHY_HILL_PLOT_LEVELS
            },
        },
    )
    plt.close(fig)


def render_murphy_hill_distribution_from_payload(
    payload: dict[str, object],
    output_dir: Path | str,
    stem: str,
) -> None:
    """Render one Murphy-Hill stacked bar chart from stored plot data."""
    groups_payload = payload.get("groups")
    levels = payload.get("levels")
    if not isinstance(groups_payload, dict) or not isinstance(levels, list):
        raise ValueError("Murphy-Hill stacked-bar metadata requires groups and levels")
    groups = order_humans_first(groups_payload.keys())
    level_order = [str(level) for level in levels]
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(figsize=MURPHY_HILL_DISTRIBUTION_FIGSIZE)
    x_values = list(range(len(groups)))
    bottoms = [0.0 for _group in groups]
    group_totals: list[int] = []
    eligible_pr_counts: list[int] = []
    for group in groups:
        group_payload = groups_payload.get(group, {})
        total = (
            group_payload.get("total_refop_count")
            if isinstance(group_payload, dict)
            else 0
        )
        if not total and isinstance(group_payload, dict):
            level_payloads = group_payload.get("levels", {})
            if isinstance(level_payloads, dict):
                total = sum(
                    int(level_payload.get("refop_count") or 0)
                    for level_payload in level_payloads.values()
                    if isinstance(level_payload, dict)
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
    for level in level_order:
        percentages = []
        visual_percentages = []
        for group in groups:
            group_payload = groups_payload.get(group, {})
            level_payload = (
                group_payload.get("levels", {}).get(level, {})
                if isinstance(group_payload, dict)
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
        previous_bottoms = list(bottoms)
        ax.bar(
            x_values,
            visual_percentages,
            bottom=bottoms,
            label=MURPHY_HILL_PLOT_LABELS.get(level, level),
            color=MURPHY_HILL_PLOT_COLORS.get(level, "#56B4E9"),
            edgecolor="black",
            linewidth=0.3,
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
                bottom + visual_percentage / 2.0 + STACKED_BAR_PERCENTAGE_LABEL_Y_OFFSET,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=STACKED_BAR_PERCENTAGE_FONT_SIZE,
                color=MURPHY_HILL_PLOT_TEXT_COLORS.get(level, "black"),
            )
        bottoms = [
            bottom + visual_percentage
            for bottom, visual_percentage in zip(bottoms, visual_percentages)
        ]
    ax.set_xlabel(
        str(payload.get("x_axis") or "Cohort"),
        labelpad=STACKED_BAR_WITH_PR_COUNT_X_LABELPAD,
    )
    ax.set_ylabel(str(payload.get("y_axis") or "Percentage of RefOps (%)"))
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_group_labels(groups), rotation=0, ha="center")
    add_xtick_count_sublabels(
        ax,
        x_values,
        group_totals,
        font_size=STACKED_BAR_COUNT_LABEL_FONT_SIZE,
        y=STACKED_BAR_COUNT_LABEL_Y,
        secondary_counts=eligible_pr_counts if any(eligible_pr_counts) else None,
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.5, max(0.5, len(groups) - 0.5))
    ax.legend(
        frameon=False,
        ncol=len(level_order),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.35,
        fontsize=7.0,
    )
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, Path(output_dir), stem)
    plt.close(fig)


def _plot_metric_boxplot(
    con,
    group_field: str,
    metric_name: str,
    output_dir: Path,
    stem: str,
) -> None:
    plt, _mdates = require_matplotlib()
    grouped = _nonempty_groups(
        con,
        group_field,
        metric_name,
        refactored_only=True,
    )
    groups = order_humans_first(grouped)
    human_baseline_tests = _comparison_payloads(
        groups,
        grouped,
        group_field,
    )
    figure_height = 2.0
    comparison_stat_label_size = 5.5
    comparison_stat_y = -0.11
    x_axis_label_y = -0.2
    bottom = 0.36
    fig, ax = plt.subplots(figsize=(3.5, figure_height))
    y_axis_limits = _draw_metric_boxplot_axis(
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
            "filtered_to_refcount_positive": True,
            "scale": _scale_from_y_axis_limits(y_axis_limits),
            "percentile_capped_y_axis": y_axis_limits,
            "groups": {
                group: _summary_payload(grouped.get(group, []))
                for group in groups
            },
            "human_baseline_tests": human_baseline_tests,
            "comparison_label_p_threshold": REFACTORING_BOXPLOT_P_THRESHOLD,
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


def _draw_metric_boxplot_axis(
    ax,
    grouped: dict[str, list[float]],
    groups: list[str],
    metric_name: str,
    *,
    xlabel: str | None,
    ylabel: str | None,
    tick_label_size: float | None = None,
    tick_label_rotation: float = 0.0,
    comparison_payload: dict[str, object] | None = None,
    comparison_stat_label_size: float | None = None,
    comparison_stat_y: float = -0.12,
    y_labelpad: float | None = None,
    y_tick_labelpad: float | None = None,
    tight_x_margins: bool = False,
) -> dict[str, float | int | bool] | None:
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
        visible_upper = ax.get_ylim()[1]
        lower_bound = _boxplot_axis_lower_bound(metric_name)
        lower_padding = _boxplot_axis_lower_padding(metric_name)
        ax.set_ylim(
            bottom=lower_bound - lower_padding,
            top=max(
                float(visible_upper),
                (
                    lower_bound
                    + REFACTORING_BOXPLOT_MINIMUM_UPPER_PADDING
                ),
                ),
            )
        if y_axis_limits is not None:
            below_count = sum(
                1
                for values in grouped.values()
                for value in values
                if float(value) < lower_bound
            )
            y_axis_limits = {
                **y_axis_limits,
                "lower": lower_bound,
                "visual_lower": lower_bound - lower_padding,
                "below_count": below_count,
                "is_clipped": bool(
                    below_count or int(y_axis_limits.get("above_count", 0))
                ),
            }
        y_axis_scale = apply_symlog_y_axis_if_range_exceeds(ax)
        if y_axis_limits is not None:
            y_axis_limits = {
                **y_axis_limits,
                "scale": y_axis_scale,
                "scale_range_threshold": 100.0,
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
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        if y_labelpad is None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(ylabel, labelpad=y_labelpad)
    ax.tick_params(axis="x", rotation=tick_label_rotation)
    if tick_label_size is not None:
        ax.tick_params(axis="x", labelsize=tick_label_size, pad=1.0)
    if y_tick_labelpad is not None:
        ax.tick_params(axis="y", pad=y_tick_labelpad)
    for label in ax.get_xticklabels():
        label.set_rotation(tick_label_rotation)
        label.set_ha("right" if tick_label_rotation else "center")
        label.set_linespacing(0.85)
    if groups and comparison_stat_label_size is not None:
        for index, label in enumerate(
            _boxplot_comparison_stat_labels(groups, comparison_payload),
            start=1,
        ):
            ax.annotate(
                label,
                xy=(index, comparison_stat_y),
                xycoords=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=comparison_stat_label_size,
                linespacing=0.85,
                clip_on=False,
                zorder=50,
            )
    ax.grid(axis="y", alpha=0.3)
    return y_axis_limits


def _plot_metric_boxplot_grid(
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
    hspace: float = 0.35,
    tick_label_rotation: float = 0.0,
    ncols: int | None = None,
    comparison_stat_label_size: float | None = None,
    comparison_stat_y: float = -0.12,
    supxlabel_y: float = 0.03,
    y_labelpad: float = 3.0,
    y_tick_labelpad: float = 1.0,
    tight_layout_h_pad: float | None = 1.4,
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
    panels: dict[str, object] = {}
    for ax, metric_name in zip(axes_list, metrics):
        grouped = _nonempty_groups(
            con,
            "cohort",
            metric_name,
            refactored_only=True,
        )
        groups = order_humans_first(grouped)
        human_baseline_tests = _comparison_payloads(
            groups,
            grouped,
            "cohort",
        )
        y_axis_limits = _draw_metric_boxplot_axis(
            ax,
            grouped,
            groups,
            metric_name,
            xlabel=None,
            ylabel=_metric_axis_label(metric_name),
            tick_label_size=tick_label_size,
            tick_label_rotation=tick_label_rotation,
            comparison_payload=human_baseline_tests,
            comparison_stat_label_size=comparison_stat_label_size,
            comparison_stat_y=comparison_stat_y,
            y_labelpad=y_labelpad,
            y_tick_labelpad=y_tick_labelpad,
            tight_x_margins=True,
        )
        panels[metric_name] = {
            "metric": metric_name,
            "x_axis": "Cohort",
            "y_axis": _metric_axis_label(metric_name),
            "scale": _scale_from_y_axis_limits(y_axis_limits),
            "percentile_capped_y_axis": y_axis_limits,
            "groups": {
                group: _summary_payload(grouped.get(group, []))
                for group in groups
            },
            "human_baseline_tests": human_baseline_tests,
            "comparison_label_p_threshold": REFACTORING_BOXPLOT_P_THRESHOLD,
        }
        del grouped, groups, human_baseline_tests
    for ax in axes_list[len(metrics) :]:
        ax.set_visible(False)
    fig.supxlabel("Cohort", y=supxlabel_y)
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    tight_layout_kwargs: dict[str, float] = {"pad": 0.01, "w_pad": wspace}
    if tight_layout_h_pad is not None:
        tight_layout_kwargs["h_pad"] = tight_layout_h_pad
    post_tight_layout_adjust_kwargs = {"hspace": hspace}
    save_figure(
        fig,
        output_dir,
        stem,
        tight_layout_kwargs=tight_layout_kwargs,
        post_tight_layout_adjust_kwargs=post_tight_layout_adjust_kwargs,
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
            "filtered_to_refcount_positive": True,
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
                "tick_label_rotation": float(tick_label_rotation),
                "comparison_stat_label_size": comparison_stat_label_size,
                "comparison_stat_y": float(comparison_stat_y),
                "supxlabel_y": float(supxlabel_y),
                "y_labelpad": float(y_labelpad),
                "y_tick_labelpad": float(y_tick_labelpad),
                "ncols": int(resolved_ncols),
                "nrows": int(resolved_nrows),
                "tight_layout_kwargs": tight_layout_kwargs,
                "post_tight_layout_adjust_kwargs": post_tight_layout_adjust_kwargs,
            },
            "panels": panels,
        },
    )
    plt.close(fig)


def _plot_combined_refactoring_metric_boxplots(
    con,
    output_dir: Path,
) -> None:
    _plot_metric_boxplot_grid(
        con,
        output_dir,
        metrics=tuple(REFACTORING_DISTRIBUTION_METRICS),
        stem="refactoring_metrics_boxplots_by_cohort",
        layout="2x2",
        figsize=(7.16, 3.5),
        tick_label_size=7.0,
        comparison_stat_label_size=6.5,
        comparison_stat_y=-0.11,
        tick_label_rotation=0.0,
        left=0.01,
        right=0.99,
        bottom=0.20,
        top=0.965,
        wspace=0.45,
        hspace=0.25,
        ncols=2,
        supxlabel_y=-0.03,
        y_labelpad=2.0,
        tight_layout_h_pad=None,
    )


def _plot_refadded_refremoved_boxplots(
    con,
    output_dir: Path,
) -> None:
    _plot_metric_boxplot_grid(
        con,
        output_dir,
        metrics=("RefAdded", "RefRemoved"),
        stem="refadded_refremoved_boxplots_by_cohort",
        layout="1x2",
        figsize=(7.16, 2.0),
        tick_label_size=7.0,
        comparison_stat_label_size=6.5,
        comparison_stat_y=-0.11,
        left=0.01,
        right=0.99,
        bottom=0.34,
        top=0.96,
        wspace=0.45,
        supxlabel_y=-0.05,
        y_labelpad=2.0,
    )


def write_refactoring_analysis_plots(
    con,
    output_dir: Path,
    logger: Any | None = None,
) -> None:
    """Write refactoring coverage, Murphy-Hill, and metric distribution plots."""
    apply_ieee_plot_style()
    remove_plot_outputs(output_dir, REMOVED_PLOT_OUTPUT_STEMS)
    if logger is not None:
        logger.log("writing_refactoring_murphy_hill_distribution")
    _plot_murphy_hill_distribution(
        con,
        "cohort",
        output_dir,
        "murphy_hill_distribution_by_cohort",
    )
    release_process_memory(logger, stage="refactoring_murphy_hill_plot_memory_released")
    if logger is not None:
        logger.log("writing_refactoring_boxplots")
    for metric_name in REFACTORING_NUMERIC_METRICS:
        _plot_metric_boxplot(
            con,
            "cohort",
            metric_name,
            output_dir / "boxplots",
            f"{_metric_stem(metric_name)}_boxplot_by_cohort",
        )
    release_process_memory(logger, stage="refactoring_boxplots_memory_released")
    if logger is not None:
        logger.log("writing_refactoring_boxplot_grids")
    _plot_combined_refactoring_metric_boxplots(
        con,
        output_dir / "boxplots-grid",
    )
    _plot_refadded_refremoved_boxplots(
        con,
        output_dir / "boxplots-grid",
    )
    release_process_memory(logger, stage="refactoring_boxplot_grids_memory_released")


def write_refactoring_analysis_plots_from_payload(
    payload: dict[str, object],
    output_dir: Path,
    logger: Any | None = None,
) -> None:
    """Render refactoring plots from compact streaming payloads."""
    write_refactoring_analysis_plots(
        _RefactoringPlotPayloadConnection(payload),
        output_dir,
        logger=logger,
    )
