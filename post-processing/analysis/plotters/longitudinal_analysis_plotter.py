"""Line plots for longitudinal analysis outputs.

The longitudinal plotter renders shared timepoint series for refactoring,
maintainability, and Multimetric metrics. It keeps scaling, nonnegative-axis
handling, and attrition plotting in one place so all longitudinal figures use
the same visual conventions.
"""

from __future__ import annotations

import colorsys
import sys
import math
from pathlib import Path
from typing import Any

from matplotlib.ticker import NullFormatter, NullLocator

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
UTILITY_DIR = ANALYSIS_DIR / "utility"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from longitudinal_analysis_utility import (
    LONGITUDINAL_TIMEPOINTS,
    NONNEGATIVE_LONGITUDINAL_METRICS,
)
from plotting_utility import (
    apply_ieee_plot_style,
    cohort_color_map,
    display_group_label,
    order_humans_first,
    require_matplotlib,
    remove_plot_outputs,
    save_figure,
    write_plot_data,
)


METRIC_AXIS_LABELS = {
    "RefCount": "RefOp Count",
    "RefDensity": "RefOps per KLOC",
    "RefDiversity": "RefOp Diversity",
    "RefMagLines": "Lines Changed per RefOp",
    "RefAdded": "Lines Added per RefOp",
    "RefAddedLines": "Lines Added per RefOp",
    "RefRemoved": "Lines Deleted per RefOp",
    "RefDeletedLines": "Lines Deleted per RefOp",
    "RefRetentionRate": "RefOp Retention Rate (%)",
    "RefZoneFutureTouchedLines": "Future Lines Touched per RefOp",
    "FutureTouchingCommits": "Commit Count Touching PR Files",
    "MI": "Maintainability Index",
    "CC": "Cyclomatic Complexity",
    "HV": "Halstead Volume (×10⁴)",
    "CCDensity": "Cyclomatic Complexity per KLOC",
    "HVDensity": "Halstead Volume per KLOC",
    "DuplicationDensity": "Duplicated Lines Density (%)",
    "CommentDensity": "Comment Lines Density (%)",
    "SmellCount": "Smell Count",
    "CodeSmellDensity": "Smells per KLOC",
    "SmellIntroCount": "Smell Intro Count",
    "SmellIntroRate": "Smell Intro Rate (%)",
    "SmellRegressionCount": "Smell Regression Count",
    "SmellRegressionRate": "Smell Regression Rate (%)",
    "SmellRegRate": "Smell Regression Rate (%)",
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
    "halstead_difficulty_per_kloc": "Halstead Difficulty per KLOC",
    "halstead_effort_per_kloc": "Halstead Effort per KLOC",
    "cyclomatic_complexity_per_kloc": "Cyclomatic Complexity per KLOC",
    "halstead_volume_per_kloc": "Halstead Volume per KLOC",
    "halstead_bugprop_per_kloc": "Halstead Delivered Bugs per KLOC",
    "fanout_external_per_kloc": "Fan Out per KLOC",
    "halstead_timerequired_per_kloc": "Halstead Time Required per KLOC",
    "multimetric_duplication_score": "Duplication Score",
    "original_smell_count": "Smells",
    "original_code_smell_density": "Smells per KLOC",
    "original_duplication_density": "Duplicated Lines Density (%)",
}
PERCENTAGE_RATE_METRICS = {
    "RefRetentionRate",
    "SmellIntroRate",
    "SmellRegressionRate",
    "SmellRegRate",
}
TEN_THOUSAND_SCALE_LINE_METRICS = {"HV"}
TEN_TO_MINUS_EIGHT_SCALE_LINE_METRICS: set[str] = set()
MILLION_SCALE_LINE_METRICS: set[str] = set()
SYMLOG_SCALE_LINE_METRICS = {"RefZoneFutureTouchedLines"}
LOG_SCALE_LINE_METRICS: set[str] = set()
LOG_SCALE_LINE_LINTHRESH = 1.0
PERCENTAGE_RATE_AXIS_MINIMUM = 0.0
PERCENTAGE_RATE_AXIS_MAXIMUM = 100.0
PERCENTAGE_RATE_ZOOM_METRICS = {"RefRetentionRate"}
PERCENTAGE_RATE_ZOOM_LOWER_STEP = 10.0
LONGITUDINAL_LINE_PLOT_GRID_OUTPUT_DIR = "line_plots-grid"
LEGACY_LONGITUDINAL_LINE_PLOT_GRID_OUTPUT_DIR = "box_line_plots"
LONGITUDINAL_GRID_LEGEND_SINGLE_ROW_Y = 0.89
LONGITUDINAL_GRID_LEGEND_MULTI_ROW_Y = 0.83
LONGITUDINAL_GRID_LEGEND_DOTPLOT_ALIGNED_Y_OFFSET_POINTS = 4.0
LONGITUDINAL_LEGEND_COLUMN_SPACING = 0.96
LONGITUDINAL_GRID_X_TICK_LABELPAD = 1.0
LONGITUDINAL_GRID_Y_TICK_LABELPAD = 0.5
LONGITUDINAL_GRID_Y_LABELPAD = 0.75
LONGITUDINAL_GRID_X_LABEL_Y = 0.0
LONGITUDINAL_GRID_WSPACE = 0.25
LONGITUDINAL_GRID_FIGURE_WIDTH = 7.16
LONGITUDINAL_LINE_LEGEND_BASE_Y = 1.025
LONGITUDINAL_DEFAULT_LEGEND_ROWS = 2
PLOT_SCALED_NUMERIC_FIELDS = {
    "min",
    "q1",
    "median",
    "mean",
    "q3",
    "max",
    "iqr",
    "mean_ci95_low",
    "mean_ci95_high",
}
ATTRITION_AXIS_LOWER_BOUND = 0.0
ATTRITION_AXIS_LOWER_PADDING = 0.0
ATTRITION_AXIS_UPPER_BOUND = 100.0
ATTRITION_AXIS_UPPER_PADDING = 5.0
ATTRITION_FIGURE_HEIGHT = 1.75
ATTRITION_LINE_WIDTH = 0.75
ATTRITION_MARKER_SIZE = 1.0
LONGITUDINAL_LINE_AXIS_LOWER_PADDING = 0.1
LONGITUDINAL_X_AXIS_PADDING = 0.0
LONGITUDINAL_CI_BAND_ALPHA = 0.08
LONGITUDINAL_CI_BAND_SATURATION_MULTIPLIER = 1.25
LONGITUDINAL_STANDALONE_LINE_WIDTH = 1.0
LONGITUDINAL_GRID_LINE_WIDTH = 0.85
LONGITUDINAL_LINE_MARKER_SIZE = 2.3
LONGITUDINAL_LEGEND_MARKER_SIZE = 4.0
LONGITUDINAL_LEGEND_FONT_SIZE = 6.5
LONGITUDINAL_LEGEND_HANDLE_TEXT_PAD = 0.697
LONGITUDINAL_POSITIVE_AXIS_ZOOM_LOWER_TO_SPAN_RATIO = 1.0
LONGITUDINAL_POSITIVE_AXIS_ZOOM_LOWER_TO_UPPER_RATIO = 0.10
LONGITUDINAL_POSITIVE_AXIS_ZOOM_PADDING_FRACTION = 0.18
TIMEPOINT_X_POSITIONS = {
    label: index for index, label in enumerate(LONGITUDINAL_TIMEPOINTS)
}


def _saturate_hex_color(
    color: str,
    *,
    multiplier: float = LONGITUDINAL_CI_BAND_SATURATION_MULTIPLIER,
) -> str:
    """Increase RGB saturation for hex colors while preserving lightness."""
    text = str(color or "").strip()
    if not text.startswith("#") or len(text) != 7:
        return text
    try:
        red = int(text[1:3], 16) / 255.0
        green = int(text[3:5], 16) / 255.0
        blue = int(text[5:7], 16) / 255.0
    except ValueError:
        return text
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    saturated = min(1.0, saturation * max(0.0, float(multiplier)))
    new_red, new_green, new_blue = colorsys.hls_to_rgb(
        hue, lightness, saturated
    )
    return "#{:02X}{:02X}{:02X}".format(
        round(new_red * 255),
        round(new_green * 255),
        round(new_blue * 255),
    )


def metric_stem(metric: str) -> str:
    """Return a filename-safe stem for one longitudinal metric."""
    stem = []
    for character in str(metric):
        if character.isalnum():
            stem.append(character.lower())
        else:
            stem.append("_")
    return "_".join(part for part in "".join(stem).split("_") if part)


def _plot_value_scale(metric: str) -> float:
    if metric in PERCENTAGE_RATE_METRICS:
        return 100.0
    if metric in TEN_THOUSAND_SCALE_LINE_METRICS:
        return 1.0 / 10_000.0
    if metric in TEN_TO_MINUS_EIGHT_SCALE_LINE_METRICS:
        return 1.0e-8
    if metric in MILLION_SCALE_LINE_METRICS:
        return 1.0 / 1_000_000.0
    return 1.0


def _scale_numeric_value(value: Any, scale: float) -> Any:
    if value is None:
        return None
    try:
        return float(value) * float(scale)
    except (TypeError, ValueError):
        return value


def _scale_plot_payload(value: Any, scale: float, key: str | None = None) -> Any:
    if scale == 1.0:
        return value
    if isinstance(value, dict):
        return {
            str(child_key): _scale_plot_payload(
                child_value,
                scale,
                str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _scale_plot_payload(child_value, scale, key)
            for child_value in value
        ]
    if key in PLOT_SCALED_NUMERIC_FIELDS:
        return _scale_numeric_value(value, scale)
    return value


def _percentage_zoom_lower_bound(value: float) -> float:
    return max(
        PERCENTAGE_RATE_AXIS_MINIMUM,
        math.floor(
            (float(value) + 1e-9) / PERCENTAGE_RATE_ZOOM_LOWER_STEP
        )
        * PERCENTAGE_RATE_ZOOM_LOWER_STEP,
    )


def longitudinal_line_plot_stems(metrics: tuple[str, ...]) -> tuple[str, ...]:
    """Return expected output stems for individual longitudinal line plots."""
    return tuple(
        f"line_plots/{metric_stem(metric)}_longitudinal_line_plot"
        for metric in metrics
    )


def longitudinal_line_plot_grid_stems(
    grid_specs: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    """Return expected output stems for grouped longitudinal plot grids."""
    return tuple(
        f"{LONGITUDINAL_LINE_PLOT_GRID_OUTPUT_DIR}/{spec['stem']}"
        for spec in grid_specs
    )


def _timepoint_x(label: str) -> float:
    return float(TIMEPOINT_X_POSITIONS.get(str(label), 0))


def _timepoint_tick_label(label: str) -> str:
    return str(label).removesuffix("d")


def _timepoint_tick_labels() -> list[str]:
    return [_timepoint_tick_label(label) for label in LONGITUDINAL_TIMEPOINTS]


def _legend_column_count(group_count: int, max_columns: int | None = None) -> int:
    if group_count <= 0:
        return 1
    if group_count <= 2:
        return group_count
    if max_columns is not None:
        try:
            return min(group_count, max(1, int(max_columns)))
        except (TypeError, ValueError):
            pass
    return max(
        1,
        (int(group_count) + LONGITUDINAL_DEFAULT_LEGEND_ROWS - 1)
        // LONGITUDINAL_DEFAULT_LEGEND_ROWS,
    )


def _legend_row_count(
    group_count: int,
    max_columns: int | None = None,
) -> int:
    columns = _legend_column_count(group_count, max_columns)
    return max(1, (group_count + columns - 1) // columns)


def _grid_legend_y(group_count: int) -> float:
    if _legend_row_count(group_count) > 1:
        return LONGITUDINAL_GRID_LEGEND_MULTI_ROW_Y
    return LONGITUDINAL_GRID_LEGEND_SINGLE_ROW_Y


def _axes_y_offset_from_points(ax, points: float) -> float:
    fig = ax.figure
    axes_height_inches = max(
        float(ax.get_position().height) * float(fig.get_figheight()),
        0.01,
    )
    return float(points) / (axes_height_inches * 72.0)


def _add_top_legend(
    ax,
    *,
    group_count: int,
    legend_y_offset_points: float = 0.0,
    legend_base_y: float = 1.02,
    legend_max_columns: int | None = None,
    center_last_row: bool = False,
) -> dict[str, Any] | None:
    plt, _mdates = require_matplotlib()
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=handle.get_color(),
            markeredgecolor=handle.get_color(),
            markersize=LONGITUDINAL_LEGEND_MARKER_SIZE,
        )
        for handle in handles
    ]
    if center_last_row and marker_handles and labels:
        columns = _legend_column_count(group_count, legend_max_columns)
        rows = _legend_row_count(group_count, legend_max_columns)
        if rows > 1:
            complete_row_count, last_row_count = divmod(len(marker_handles), columns)
            if last_row_count:
                full_handles = marker_handles[: complete_row_count * columns]
                full_labels = labels[: complete_row_count * columns]
                last_handles = marker_handles[complete_row_count * columns :]
                last_labels = labels[complete_row_count * columns :]

                center_positions: set[int] = _centered_last_row_positions(
                    columns,
                    last_row_count,
                )
                padded_last_handles: list[Any] = []
                padded_last_labels: list[str] = []
                last_index = 0
                for position in range(columns):
                    if position in center_positions and last_index < last_row_count:
                        padded_last_handles.append(last_handles[last_index])
                        padded_last_labels.append(str(last_labels[last_index]))
                        last_index += 1
                    else:
                        padded_last_handles.append(
                            plt.Line2D(
                                [0],
                                [0],
                                marker="",
                                linestyle="",
                                linewidth=0.0,
                                alpha=0.0,
                            )
                        )
                        padded_last_labels.append("")

                marker_handles = (
                    full_handles + padded_last_handles
                    if complete_row_count
                    else padded_last_handles
                )
                labels = (
                    full_labels + padded_last_labels
                    if complete_row_count
                    else padded_last_labels
                )
                columns = _legend_column_count(len(marker_handles), legend_max_columns)
                rows = _legend_row_count(len(marker_handles), legend_max_columns)

    columns = _legend_column_count(group_count, legend_max_columns)
    rows = _legend_row_count(group_count, legend_max_columns)
    legend_y = legend_base_y + _axes_y_offset_from_points(ax, legend_y_offset_points)
    ax.legend(
        marker_handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        ncol=columns,
        columnspacing=LONGITUDINAL_LEGEND_COLUMN_SPACING,
        fontsize=LONGITUDINAL_LEGEND_FONT_SIZE,
        handlelength=0.0,
        handletextpad=LONGITUDINAL_LEGEND_HANDLE_TEXT_PAD,
        borderaxespad=0.0,
    )
    return {
        "location": "above_plot",
        "columns": columns,
        "rows": rows,
        "max_columns": 4,
        "icon": "marker_only",
        "marker_size": LONGITUDINAL_LEGEND_MARKER_SIZE,
        "font_size": LONGITUDINAL_LEGEND_FONT_SIZE,
        "handle_text_pad": LONGITUDINAL_LEGEND_HANDLE_TEXT_PAD,
        "column_spacing": LONGITUDINAL_LEGEND_COLUMN_SPACING,
        "legend_base_y": legend_base_y,
        "legend_y_offset_points": legend_y_offset_points,
        "bbox_to_anchor": [0.5, legend_y],
    }


def _centered_last_row_positions(columns: int, item_count: int) -> set[int]:
    """Return column positions that best center a partial legend row."""
    if columns <= 0 or item_count <= 0:
        return set()
    if item_count >= columns:
        return set(range(min(columns, item_count)))
    if item_count == 1:
        return {columns // 2}

    center = (columns - 1) / 2.0
    from itertools import combinations

    best_positions = list(range(item_count))
    best_cost = float("inf")
    for positions in combinations(range(columns), item_count):
        mean_position = sum(positions) / item_count
        spread = (
            max(positions) - min(positions)
            if item_count > 1
            else 0.0
        )
        cost = abs(mean_position - center) + 0.002 * spread
        if cost < best_cost:
            best_cost = cost
            best_positions = list(positions)
    return set(best_positions)


def _figure_y_offset_from_points(fig, points: float) -> float:
    height_inches = max(float(fig.get_figheight()), 0.01)
    return float(points) / (height_inches * 72.0)


def _active_axes_top_y(axes) -> float:
    active_tops: list[float] = []
    for ax in axes:
        if ax is None or not ax.get_visible():
            continue
        active_tops.append(float(ax.get_position().y1))
    return max(active_tops) if active_tops else 1.0


def _add_grid_top_legend(
    fig,
    handles,
    labels,
    *,
    group_count: int,
    legend_y_offset_points: float = 0.0,
    axes=None,
    legend_max_columns: int | None = None,
) -> dict[str, Any] | None:
    plt, _mdates = require_matplotlib()
    if not handles:
        return None
    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=handle.get_color(),
            markeredgecolor=handle.get_color(),
            markersize=LONGITUDINAL_LEGEND_MARKER_SIZE,
        )
        for handle in handles
    ]
    legend_item_count = max(1, len(labels))
    columns = _legend_column_count(legend_item_count, legend_max_columns)
    rows = _legend_row_count(legend_item_count, legend_max_columns)
    axes_top_y = None
    legend_location = "above_grid"
    legend_anchor_reference = "fixed_figure_y"
    legend_loc = "upper center"
    if axes is not None:
        axes_top_y = _active_axes_top_y(axes)
        legend_y = axes_top_y + _figure_y_offset_from_points(
            fig,
            legend_y_offset_points,
        )
        legend_anchor_reference = "active_axes_top"
        legend_loc = "lower center"
    else:
        legend_y = _grid_legend_y(legend_item_count) + _figure_y_offset_from_points(
            fig,
            legend_y_offset_points,
        )
    fig.legend(
        marker_handles,
        labels,
        loc=legend_loc,
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        ncol=columns,
        columnspacing=LONGITUDINAL_LEGEND_COLUMN_SPACING,
        fontsize=LONGITUDINAL_LEGEND_FONT_SIZE,
        handlelength=0.0,
        handletextpad=LONGITUDINAL_LEGEND_HANDLE_TEXT_PAD,
        borderaxespad=0.0,
    )
    return {
        "location": legend_location,
        "columns": columns,
        "rows": rows,
        "max_columns": columns,
        "icon": "marker_only",
        "marker_size": LONGITUDINAL_LEGEND_MARKER_SIZE,
        "font_size": LONGITUDINAL_LEGEND_FONT_SIZE,
        "handle_text_pad": LONGITUDINAL_LEGEND_HANDLE_TEXT_PAD,
        "column_spacing": LONGITUDINAL_LEGEND_COLUMN_SPACING,
        "legend_y_offset_points": legend_y_offset_points,
        "legend_anchor_reference": legend_anchor_reference,
        "active_axes_top_y": axes_top_y,
        "bbox_to_anchor": [0.5, legend_y],
    }


def _top_margin_for_legend(
    group_count: int,
    max_columns: int | None = None,
) -> float:
    return 0.76 if _legend_row_count(group_count, max_columns) > 1 else 0.84


def _top_margin_for_grid_legend(
    group_count: int,
    max_columns: int | None = None,
) -> float:
    return 0.76 if _legend_row_count(group_count, max_columns) > 1 else 0.84


def _mean_range(series: dict[str, list[dict[str, Any]]]) -> tuple[float, float] | None:
    values: list[float] = []
    for points in series.values():
        for point in points:
            mean = point.get("mean")
            low = point.get("mean_ci95_low")
            high = point.get("mean_ci95_high")
            for value in (mean, low, high):
                if value is not None:
                    values.append(float(value))
    if not values:
        return None
    return min(values), max(values)


def _positive_mean_range(
    series: dict[str, list[dict[str, Any]]],
) -> tuple[float, float] | None:
    values: list[float] = []
    for points in series.values():
        for point in points:
            mean = point.get("mean")
            low = point.get("mean_ci95_low")
            high = point.get("mean_ci95_high")
            for value in (mean, low, high):
                if value is not None and float(value) > 0.0:
                    values.append(float(value))
    if not values:
        return None
    return min(values), max(values)


def _set_line_plot_y_limits(
    ax,
    value_range: tuple[float, float] | None,
    *,
    metric: str,
    clamp_nonnegative_axis_to_zero: bool = False,
    negative_visual_lower_limit: float | None = None,
    axis_padding_fraction: float | None = None,
    allow_zero_floor_when_positive: bool = True,
    force_log_scale: bool = False,
    positive_value_range: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    if value_range is None:
        return None
    lower, upper = value_range
    if metric in SYMLOG_SCALE_LINE_METRICS:
        ax.set_yscale("symlog", linthresh=LOG_SCALE_LINE_LINTHRESH)
        logical_lower = 0.0 if lower >= 0.0 else float(lower)
        visual_lower = logical_lower
        visual_upper = max(float(upper) * 1.18, LOG_SCALE_LINE_LINTHRESH)
        if visual_upper <= visual_lower:
            visual_upper = visual_lower + LOG_SCALE_LINE_LINTHRESH
        ax.set_ylim(visual_lower, visual_upper)
        return {
            "lower": float(logical_lower),
            "visual_lower_padding": 0.0,
            "visual_lower": float(visual_lower),
            "upper": float(upper),
            "visual_upper": float(visual_upper),
            "clamp_nonnegative_axis_to_zero": bool(clamp_nonnegative_axis_to_zero),
            "axis_scale": "symlog",
            "log_scale_requested": True,
            "linthresh": LOG_SCALE_LINE_LINTHRESH,
            "zero_compatible_log_scale": True,
        }
    if force_log_scale or metric in LOG_SCALE_LINE_METRICS:
        positive_range = positive_value_range
        if positive_range is None:
            positive_values = [
                float(value)
                for value in (lower, upper)
                if value is not None and float(value) > 0.0
            ]
            positive_range = (
                (min(positive_values), max(positive_values))
                if positive_values
                else None
            )
        if positive_range is not None:
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            positive_lower, positive_upper = positive_range
            visual_lower = positive_lower / 1.18
            visual_upper = positive_upper * 1.18
            if visual_upper <= visual_lower:
                visual_upper = visual_lower * 1.18
            ax.set_ylim(visual_lower, visual_upper)
            return {
                "lower": float(positive_lower),
                "visual_lower_padding": 0.0,
                "visual_lower": float(visual_lower),
                "upper": float(positive_upper),
                "visual_upper": float(visual_upper),
                "clamp_nonnegative_axis_to_zero": bool(clamp_nonnegative_axis_to_zero),
                "axis_scale": "log",
                "log_scale_requested": True,
                "positive_log_scale": True,
            }
    span = upper - lower
    padding_fraction = (
        max(0.0, min(0.40, float(axis_padding_fraction or 0.22)))
    )
    upper_padding = span * padding_fraction if span > 0 else max(abs(upper), 1.0) * 0.18
    if lower >= 0.0:
        zoom_positive_axis = (
            not clamp_nonnegative_axis_to_zero
            and (
                metric not in PERCENTAGE_RATE_METRICS
                or metric in PERCENTAGE_RATE_ZOOM_METRICS
            )
            and span > 0.0
            and lower > span * LONGITUDINAL_POSITIVE_AXIS_ZOOM_LOWER_TO_SPAN_RATIO
            and lower > upper * LONGITUDINAL_POSITIVE_AXIS_ZOOM_LOWER_TO_UPPER_RATIO
        )
        if zoom_positive_axis:
            logical_lower = float(lower)
            if metric in PERCENTAGE_RATE_ZOOM_METRICS:
                visual_lower = _percentage_zoom_lower_bound(logical_lower)
            else:
                lower_padding = max(
                    span * LONGITUDINAL_POSITIVE_AXIS_ZOOM_PADDING_FRACTION,
                    LONGITUDINAL_LINE_AXIS_LOWER_PADDING,
                )
                visual_lower = max(0.0, logical_lower - lower_padding)
        else:
            if allow_zero_floor_when_positive:
                logical_lower = 0.0
                visual_lower = (
                    logical_lower
                    if clamp_nonnegative_axis_to_zero
                    else logical_lower
                    - (
                        span * padding_fraction
                        if span > 0
                        else LONGITUDINAL_LINE_AXIS_LOWER_PADDING
                    )
                )
            else:
                logical_lower = float(lower)
                visual_lower = logical_lower - (
                    span * padding_fraction
                    if span > 0
                    else LONGITUDINAL_LINE_AXIS_LOWER_PADDING
                )
    else:
        zoom_positive_axis = False
        logical_lower = float(lower)
        visual_lower = logical_lower - LONGITUDINAL_LINE_AXIS_LOWER_PADDING
    visual_upper = float(upper) + upper_padding
    if metric in PERCENTAGE_RATE_METRICS:
        logical_lower = max(PERCENTAGE_RATE_AXIS_MINIMUM, float(logical_lower))
        visual_lower = max(PERCENTAGE_RATE_AXIS_MINIMUM, float(visual_lower))
        visual_upper = min(PERCENTAGE_RATE_AXIS_MAXIMUM, float(visual_upper))
        if visual_upper <= visual_lower:
            visual_upper = PERCENTAGE_RATE_AXIS_MAXIMUM
    nonnegative_axis_lower_clamped = False
    if metric in NONNEGATIVE_LONGITUDINAL_METRICS and visual_lower < 0.0:
        visual_lower = 0.0
        logical_lower = max(0.0, float(logical_lower))
        nonnegative_axis_lower_clamped = True
    negative_visual_lower_limit_applied = False
    if (
        negative_visual_lower_limit is not None
        and lower < 0.0
        and metric not in NONNEGATIVE_LONGITUDINAL_METRICS
        and visual_lower < float(negative_visual_lower_limit)
    ):
        visual_lower = float(negative_visual_lower_limit)
        negative_visual_lower_limit_applied = True
    ax.set_ylim(visual_lower, visual_upper)
    payload = {
        "lower": float(logical_lower),
        "visual_lower_padding": LONGITUDINAL_LINE_AXIS_LOWER_PADDING,
        "visual_lower": float(visual_lower),
        "upper": float(upper),
        "visual_upper": float(visual_upper),
        "clamp_nonnegative_axis_to_zero": bool(clamp_nonnegative_axis_to_zero),
        "positive_axis_zoom_applied": bool(zoom_positive_axis),
        "nonnegative_axis_lower_clamped": bool(nonnegative_axis_lower_clamped),
    }
    if zoom_positive_axis:
        payload.update(
            {
                "positive_axis_zoom_lower_to_span_ratio": (
                    LONGITUDINAL_POSITIVE_AXIS_ZOOM_LOWER_TO_SPAN_RATIO
                ),
                "positive_axis_zoom_lower_to_upper_ratio": (
                    LONGITUDINAL_POSITIVE_AXIS_ZOOM_LOWER_TO_UPPER_RATIO
                ),
                "positive_axis_zoom_padding_fraction": (
                    LONGITUDINAL_POSITIVE_AXIS_ZOOM_PADDING_FRACTION
                ),
            }
        )
    if negative_visual_lower_limit is not None:
        payload.update(
            {
                "negative_visual_lower_limit": float(negative_visual_lower_limit),
                "negative_visual_lower_limit_applied": negative_visual_lower_limit_applied,
            }
        )
    if metric in PERCENTAGE_RATE_METRICS:
        payload.update(
            {
                "percentage_rate_axis_minimum": PERCENTAGE_RATE_AXIS_MINIMUM,
                "percentage_rate_axis_maximum": PERCENTAGE_RATE_AXIS_MAXIMUM,
                "percentage_rate_axis_clamped": True,
                "percentage_rate_axis_zoom_allowed": (
                    metric in PERCENTAGE_RATE_ZOOM_METRICS
                ),
                "percentage_rate_zoom_lower_step": (
                    PERCENTAGE_RATE_ZOOM_LOWER_STEP
                    if metric in PERCENTAGE_RATE_ZOOM_METRICS
                    else None
                ),
            }
        )
    return payload


def _metric_plot_payload(
    *,
    metric: str,
    metric_payload: dict[str, Any],
) -> dict[str, Any]:
    plot_value_scale = _plot_value_scale(metric)
    series = _scale_plot_payload(
        metric_payload.get("plot_series", {}),
        plot_value_scale,
    )
    return {
        "plot_value_scale": plot_value_scale,
        "plot_value_unit": (
            "percentage"
            if metric in PERCENTAGE_RATE_METRICS
            else "ten_thousands"
            if metric in TEN_THOUSAND_SCALE_LINE_METRICS
            else "ten_to_minus_eight"
            if metric in TEN_TO_MINUS_EIGHT_SCALE_LINE_METRICS
            else "millions"
            if metric in MILLION_SCALE_LINE_METRICS
            else "original_metric_unit"
        ),
        "series": series,
        "overall": _scale_plot_payload(
            metric_payload.get("overall", {}),
            plot_value_scale,
        ),
        "per_cohort": _scale_plot_payload(
            metric_payload.get("per_cohort", {}),
            plot_value_scale,
        ),
        "agents_vs_humans": _scale_plot_payload(
            metric_payload.get("agents_vs_humans", {}),
            plot_value_scale,
        ),
    }


def _resolved_cohort_colors(
    cohorts: list[str] | tuple[str, ...],
    source_colors: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve longitudinal cohort colors through the shared fixed palette."""
    colors: dict[str, str] = {}
    if source_colors is not None:
        colors.update({str(key): str(value) for key, value in source_colors.items()})
    colors.update(cohort_color_map(cohorts))
    return {
        str(cohort): str(colors.get(str(cohort), "#4D4D4D"))
        for cohort in cohorts
    }


def _draw_metric_line_axis(
    ax,
    *,
    metric: str,
    metric_payload: dict[str, Any],
    show_x_axis_label: bool = True,
    show_legend: bool = True,
    x_tick_fontsize: float = 5.6,
    x_tick_labelpad: float | None = None,
    y_tick_labelpad: float | None = None,
    y_labelpad: float | None = None,
    y_axis_label_override: str | None = None,
    x_labelpad: float | None = None,
    legend_y_offset_points: float = 0.0,
    legend_base_y: float = 1.02,
    legend_max_columns: int | None = None,
    cohort_order: list[str] | tuple[str, ...] | None = None,
    fixed_cohort_colors: dict[str, str] | None = None,
    clamp_nonnegative_axis_to_zero: bool = False,
    negative_visual_lower_limit: float | None = None,
    line_width: float = LONGITUDINAL_STANDALONE_LINE_WIDTH,
    marker_size: float = LONGITUDINAL_LINE_MARKER_SIZE,
    y_axis_label_fontsize: float | None = None,
    axis_padding_fraction: float | None = None,
    allow_zero_floor_when_positive: bool = True,
    center_last_row: bool = False,
    force_log_scale: bool = False,
) -> dict[str, Any]:
    payload = _metric_plot_payload(metric=metric, metric_payload=metric_payload)
    series = payload["series"]
    cohorts = order_humans_first(
        cohort_order if cohort_order is not None else series.keys()
    )
    cohort_colors = _resolved_cohort_colors(cohorts, fixed_cohort_colors)
    plotted_any = False
    plotted_cohort_count = 0
    for cohort in cohorts:
        points = [
            point for point in series.get(cohort, [])
            if point.get("mean") is not None and point.get("n", 0) > 0
        ]
        if not points:
            continue
        plotted_any = True
        plotted_cohort_count += 1
        x_values = [
            _timepoint_x(str(point.get("timepoint_label", "0d")))
            for point in points
        ]
        y_values = [float(point["mean"]) for point in points]
        color = cohort_colors.get(cohort, "#4D4D4D")
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=line_width,
            markersize=marker_size,
            color=color,
            label=display_group_label(cohort),
            zorder=3,
        )
        low_values = []
        high_values = []
        has_band = False
        for point in points:
            low = point.get("mean_ci95_low")
            high = point.get("mean_ci95_high")
            mean = point.get("mean")
            if low is not None and high is not None:
                low_values.append(float(low))
                high_values.append(float(high))
                has_band = True
            else:
                low_values.append(float(mean))
                high_values.append(float(mean))
        if has_band:
            ax.fill_between(
                x_values,
                low_values,
                high_values,
                color=_saturate_hex_color(color),
                alpha=LONGITUDINAL_CI_BAND_ALPHA,
                linewidth=0,
                zorder=2,
            )
    if x_labelpad is None:
        ax.set_xlabel("Time After Merge (Days)")
    else:
        ax.set_xlabel("Time After Merge (Days)", labelpad=x_labelpad)
    if not show_x_axis_label:
        ax.set_xlabel("")
    y_axis_label = y_axis_label_override or METRIC_AXIS_LABELS.get(metric, metric)
    if y_labelpad is None:
        if y_axis_label_fontsize is None:
            ax.set_ylabel(y_axis_label)
        else:
            ax.set_ylabel(y_axis_label, fontsize=float(y_axis_label_fontsize))
    else:
        if y_axis_label_fontsize is None:
            ax.set_ylabel(y_axis_label, labelpad=y_labelpad)
        else:
            ax.set_ylabel(
                y_axis_label,
                labelpad=y_labelpad,
                fontsize=float(y_axis_label_fontsize),
            )
    ax.set_xticks(list(range(len(LONGITUDINAL_TIMEPOINTS))))
    ax.set_xticklabels(
        _timepoint_tick_labels(),
        rotation=0,
        ha="center",
        fontsize=x_tick_fontsize,
    )
    ax.set_xlim(
        -LONGITUDINAL_X_AXIS_PADDING,
        len(LONGITUDINAL_TIMEPOINTS) - 1 + LONGITUDINAL_X_AXIS_PADDING,
    )
    if x_tick_labelpad is not None:
        ax.tick_params(axis="x", pad=x_tick_labelpad)
    if y_tick_labelpad is not None:
        ax.tick_params(axis="y", pad=y_tick_labelpad)
    ax.grid(axis="y", color="0.88", linestyle="-", linewidth=0.35)
    legend_payload = None
    if plotted_any and show_legend:
        legend_payload = _add_top_legend(
            ax,
            group_count=plotted_cohort_count,
            legend_y_offset_points=legend_y_offset_points,
            legend_base_y=legend_base_y,
            legend_max_columns=legend_max_columns,
            center_last_row=center_last_row,
        )
    value_range = _mean_range(series)
    y_axis_limits = _set_line_plot_y_limits(
        ax,
        value_range,
        metric=metric,
        clamp_nonnegative_axis_to_zero=clamp_nonnegative_axis_to_zero,
        negative_visual_lower_limit=negative_visual_lower_limit,
        axis_padding_fraction=axis_padding_fraction,
        allow_zero_floor_when_positive=allow_zero_floor_when_positive,
        force_log_scale=force_log_scale,
        positive_value_range=(
            _positive_mean_range(series) if force_log_scale else None
        ),
    )
    payload.update(
        {
            "metric": metric,
            "metric_label": METRIC_AXIS_LABELS.get(metric, metric),
            "cohorts": cohorts,
            "cohort_colors": cohort_colors,
            "plotted_cohort_count": plotted_cohort_count,
            "legend": legend_payload,
            "line_width": line_width,
            "marker_size": marker_size,
            "y_axis_limits": y_axis_limits,
        }
    )
    return payload


def _draw_metric_line_axis_from_plot_payload(
    ax,
    payload: dict[str, Any],
    *,
    show_x_axis_label: bool = True,
    show_legend: bool = True,
    x_tick_fontsize: float = 5.6,
    x_tick_labelpad: float | None = None,
    y_tick_labelpad: float | None = None,
    y_labelpad: float | None = None,
    x_labelpad: float | None = None,
    legend_y_offset_points: float = 0.0,
    legend_base_y: float = 1.02,
    legend_max_columns: int | None = None,
    center_last_row: bool = False,
) -> dict[str, Any]:
    metric = str(payload.get("metric") or "")
    series = payload.get("plot_series")
    if not isinstance(series, dict):
        raise ValueError("longitudinal line plot metadata requires plot_series")
    cohorts = order_humans_first(
        payload.get("cohorts") if isinstance(payload.get("cohorts"), list) else series.keys()
    )
    source_colors = (
        payload.get("cohort_colors")
        if isinstance(payload.get("cohort_colors"), dict)
        else None
    )
    cohort_colors = _resolved_cohort_colors(cohorts, source_colors)
    timepoints = [
        str(item)
        for item in (
            payload.get("timepoints")
            if isinstance(payload.get("timepoints"), list)
            else LONGITUDINAL_TIMEPOINTS
        )
    ]
    timepoint_positions = {
        str(key): float(value)
        for key, value in (
            payload.get("x_axis_positions")
            if isinstance(payload.get("x_axis_positions"), dict)
            else TIMEPOINT_X_POSITIONS
        ).items()
    }
    tick_labels = (
        [str(item) for item in payload.get("x_axis_tick_labels")]
        if isinstance(payload.get("x_axis_tick_labels"), list)
        else [_timepoint_tick_label(label) for label in timepoints]
    )
    line_width = float(payload.get("line_width") or LONGITUDINAL_STANDALONE_LINE_WIDTH)
    marker_size = float(payload.get("marker_size") or LONGITUDINAL_LINE_MARKER_SIZE)
    ci_alpha = float(
        payload.get("confidence_interval_band_alpha")
        or LONGITUDINAL_CI_BAND_ALPHA
    )
    ci_saturation_multiplier = float(
        payload.get("confidence_interval_band_saturation_multiplier")
        or LONGITUDINAL_CI_BAND_SATURATION_MULTIPLIER
    )
    plotted_any = False
    plotted_cohort_count = 0
    for cohort in cohorts:
        raw_points = series.get(cohort, [])
        points = [
            point for point in raw_points
            if isinstance(point, dict)
            and point.get("mean") is not None
            and int(point.get("n") or 0) > 0
        ]
        if not points:
            continue
        plotted_any = True
        plotted_cohort_count += 1
        x_values = [
            timepoint_positions.get(
                str(point.get("timepoint_label") or point.get("timepoint") or "0d"),
                0.0,
            )
            for point in points
        ]
        y_values = [float(point["mean"]) for point in points]
        color = cohort_colors.get(cohort, "#4D4D4D")
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=line_width,
            markersize=marker_size,
            color=color,
            label=display_group_label(cohort),
            zorder=3,
        )
        low_values = []
        high_values = []
        has_band = False
        for point in points:
            mean = point.get("mean")
            low = point.get("mean_ci95_low")
            high = point.get("mean_ci95_high")
            if low is not None and high is not None:
                low_values.append(float(low))
                high_values.append(float(high))
                has_band = True
            else:
                low_values.append(float(mean))
                high_values.append(float(mean))
        if has_band:
            ax.fill_between(
                x_values,
                low_values,
                high_values,
                color=_saturate_hex_color(
                    color,
                    multiplier=ci_saturation_multiplier,
                ),
                alpha=ci_alpha,
                linewidth=0,
                zorder=2,
            )
    x_axis_label = str(payload.get("x_axis") or "Time After Merge (Days)")
    if x_labelpad is None:
        ax.set_xlabel(x_axis_label)
    else:
        ax.set_xlabel(x_axis_label, labelpad=x_labelpad)
    if not show_x_axis_label:
        ax.set_xlabel("")
    y_axis_label = str(payload.get("y_axis") or payload.get("metric_label") or metric)
    if y_labelpad is None:
        ax.set_ylabel(y_axis_label)
    else:
        y_axis_label_fontsize = float(
            grid_spec.get("y_axis_label_fontsize", 7.0)
        )
        ax.set_ylabel(
            y_axis_label,
            labelpad=y_labelpad,
            fontsize=y_axis_label_fontsize,
        )
    ax.set_xticks([timepoint_positions.get(label, index) for index, label in enumerate(timepoints)])
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=x_tick_fontsize)
    padding = float(payload.get("x_axis_padding") or LONGITUDINAL_X_AXIS_PADDING)
    ax.set_xlim(-padding, len(timepoints) - 1 + padding)
    if x_tick_labelpad is not None:
        ax.tick_params(axis="x", pad=x_tick_labelpad)
    if y_tick_labelpad is not None:
        ax.tick_params(axis="y", pad=y_tick_labelpad)
    ax.grid(axis="y", color="0.88", linestyle="-", linewidth=0.35)
    y_axis_limits = payload.get("y_axis_limits")
    if isinstance(y_axis_limits, dict):
        axis_scale = str(y_axis_limits.get("axis_scale") or "").strip()
        if axis_scale == "symlog":
            ax.set_yscale(
                "symlog",
                linthresh=float(
                    y_axis_limits.get("linthresh") or LOG_SCALE_LINE_LINTHRESH
                ),
            )
        elif axis_scale == "log":
            ax.set_yscale("log")
        visual_lower = y_axis_limits.get("visual_lower")
        visual_upper = y_axis_limits.get("visual_upper")
        if visual_lower is not None and visual_upper is not None:
            ax.set_ylim(float(visual_lower), float(visual_upper))
    legend_payload = None
    if plotted_any and show_legend:
        legend_payload = _add_top_legend(
            ax,
            group_count=plotted_cohort_count,
            legend_y_offset_points=legend_y_offset_points,
            legend_base_y=legend_base_y,
            legend_max_columns=legend_max_columns,
            center_last_row=center_last_row,
        )
    return {
        "plotted_cohort_count": plotted_cohort_count,
        "legend": legend_payload,
    }


def render_longitudinal_line_plot_from_payload(
    payload: dict[str, Any],
    output_dir: Path,
    stem: str,
) -> None:
    """Render one longitudinal line plot from saved plot-data payload."""
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    figure_height_inches = float(
        payload.get("figure_height_inches") or 2.35
    )
    fig, ax = plt.subplots(figsize=(3.5, figure_height_inches))
    legend_max_columns = (
        int(payload["legend_max_columns"])
        if payload.get("legend_max_columns") is not None
        else None
    )
    draw_payload = _draw_metric_line_axis_from_plot_payload(
        ax,
        payload,
        x_labelpad=(
            float(payload["x_axis_labelpad"])
            if payload.get("x_axis_labelpad") is not None
            else None
        ),
        legend_base_y=float(
            payload.get("legend_base_y") or LONGITUDINAL_LINE_LEGEND_BASE_Y
        ),
        center_last_row=bool(payload.get("center_last_legend_row", False)),
        legend_max_columns=legend_max_columns,
    )
    fig.subplots_adjust(
        bottom=0.20,
        top=_top_margin_for_legend(
            draw_payload["plotted_cohort_count"],
            legend_max_columns,
        ),
    )
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def _write_metric_line_plot(
    *,
    output_dir: Path,
    metric: str,
    metric_payload: dict[str, Any],
    x_labelpad: float | None = None,
    legend_y_offset_points: float = 0.0,
    legend_base_y: float = LONGITUDINAL_LINE_LEGEND_BASE_Y,
    negative_visual_lower_limit: float | None = None,
    figure_height_inches: float | None = None,
    legend_max_columns: int | None = None,
    center_last_legend_row: bool = False,
    axis_padding_fraction: float | None = None,
    allow_zero_floor_when_positive: bool = True,
    extra_plot_metadata: dict[str, Any] | None = None,
) -> None:
    (Path(output_dir) / "line_plots").mkdir(parents=True, exist_ok=True)
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    figure_height = 2.35 if figure_height_inches is None else float(figure_height_inches)
    fig, ax = plt.subplots(figsize=(3.5, figure_height))
    plot_payload = _draw_metric_line_axis(
        ax,
        metric=metric,
        metric_payload=metric_payload,
        x_labelpad=x_labelpad,
        legend_y_offset_points=legend_y_offset_points,
        legend_base_y=legend_base_y,
        legend_max_columns=legend_max_columns,
        center_last_row=center_last_legend_row,
        negative_visual_lower_limit=negative_visual_lower_limit,
        axis_padding_fraction=axis_padding_fraction,
        allow_zero_floor_when_positive=allow_zero_floor_when_positive,
    )
    fig.subplots_adjust(
        bottom=0.20,
        top=_top_margin_for_legend(
            plot_payload["plotted_cohort_count"],
            legend_max_columns,
        ),
    )
    stem = f"{metric_stem(metric)}_longitudinal_line_plot"
    save_figure(fig, output_dir / "line_plots", stem)
    write_plot_data(
        output_dir / "line_plots",
        stem,
        {
            "plot": stem,
            "plot_type": "longitudinal_line_plot",
            "metric": metric,
            "x_axis": "Time After Merge (Days)",
            "x_axis_scale": "ordinal_timepoints",
            "x_axis_padding": LONGITUDINAL_X_AXIS_PADDING,
            "x_axis_positions": TIMEPOINT_X_POSITIONS,
            "x_axis_tick_labels": _timepoint_tick_labels(),
            "legend": plot_payload["legend"],
            "cohorts": plot_payload["cohorts"],
            "cohort_colors": plot_payload["cohort_colors"],
            "line_width": plot_payload["line_width"],
            "marker_size": plot_payload["marker_size"],
            "x_axis_labelpad": x_labelpad,
            "legend_base_y": legend_base_y,
            "legend_max_columns": legend_max_columns,
            "center_last_legend_row": center_last_legend_row,
            "confidence_interval_band_alpha": LONGITUDINAL_CI_BAND_ALPHA,
            "confidence_interval_band_saturation_multiplier": (
                LONGITUDINAL_CI_BAND_SATURATION_MULTIPLIER
            ),
            "negative_visual_lower_limit": negative_visual_lower_limit,
            "y_axis": METRIC_AXIS_LABELS.get(metric, metric),
            "plot_value_scale": plot_payload["plot_value_scale"],
            "plot_value_unit": plot_payload["plot_value_unit"],
            "figure_height_inches": figure_height,
            "y_axis_limits": plot_payload["y_axis_limits"],
            "timepoints": list(LONGITUDINAL_TIMEPOINTS),
            "overall": plot_payload["overall"],
            "per_cohort": plot_payload["per_cohort"],
            "agents_vs_humans": plot_payload["agents_vs_humans"],
            "plot_series": plot_payload["series"],
            **(
                {"multimetric_metadata": extra_plot_metadata}
                if extra_plot_metadata is not None
                else {}
            ),
        },
    )
    plt.close(fig)


def _grid_figure_size(
    rows: int,
    columns: int,
    *,
    width_override: float | None = None,
    height_override: float | None = None,
) -> tuple[float, float]:
    del columns, width_override
    width = LONGITUDINAL_GRID_FIGURE_WIDTH
    height = max(2.25, rows * 2.15)
    if height_override is not None:
        height = float(height_override)
    return (
        width,
        height,
    )


def _center_incomplete_final_row_axes(
    axes,
    *,
    used_count: int,
    rows: int,
    columns: int,
    final_row_wspace_multiplier: float = 1.0,
) -> bool:
    remainder = used_count % columns
    if rows <= 1 or columns <= 1 or remainder == 0:
        return False
    final_row = used_count // columns
    if final_row >= rows:
        return False
    active_axes = [axes[final_row, column] for column in range(remainder)]
    if not active_axes:
        return False
    if columns > 1:
        first_position = axes[final_row, 0].get_position()
        second_position = axes[final_row, 1].get_position()
        column_step = second_position.x0 - first_position.x0
    else:
        column_step = active_axes[0].get_position().width
    first_position = axes[final_row, 0].get_position()
    last_position = axes[final_row, columns - 1].get_position()
    axis_width = float(first_position.width)
    current_gap = max(0.0, float(column_step) - axis_width)
    target_gap = current_gap * max(1.0, float(final_row_wspace_multiplier))
    target_step = axis_width + target_gap
    full_left = float(first_position.x0)
    full_width = float(last_position.x0 + last_position.width - full_left)
    target_group_width = axis_width + (remainder - 1) * target_step
    if target_group_width > full_width:
        target_step = column_step
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


def render_longitudinal_line_plot_grid_from_payload(
    payload: dict[str, Any],
    output_dir: Path,
    stem: str,
) -> None:
    """Render a multi-metric longitudinal grid from saved plot data."""
    panels = payload.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("longitudinal line plot grid metadata requires panels")
    layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
    rows = int(layout.get("rows") or 1)
    columns = int(layout.get("columns") or max(1, len(panels)))
    width = LONGITUDINAL_GRID_FIGURE_WIDTH
    height = float(
        layout.get("figure_height_inches")
        or _grid_figure_size(rows, columns)[1]
    )
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(width, height),
        squeeze=False,
        constrained_layout=False,
    )
    for index, ax in enumerate(axes.flat):
        row_index = index // columns
        ax.set_zorder(rows - row_index)
    legend_by_label: dict[str, Any] = {}
    max_plotted_cohort_count = 0
    used_count = 0
    for index, ax in enumerate(axes.flat):
        if index >= len(panels):
            ax.axis("off")
            continue
        panel = panels[index]
        if not isinstance(panel, dict):
            ax.axis("off")
            continue
        used_count += 1
        draw_payload = _draw_metric_line_axis_from_plot_payload(
            ax,
            panel,
            show_x_axis_label=False,
            show_legend=False,
            x_tick_fontsize=7.0,
            x_tick_labelpad=float(
                layout.get("x_tick_labelpad", LONGITUDINAL_GRID_X_TICK_LABELPAD)
            ),
            y_tick_labelpad=float(
                layout.get("y_tick_labelpad", LONGITUDINAL_GRID_Y_TICK_LABELPAD)
            ),
            y_labelpad=float(layout.get("y_labelpad", LONGITUDINAL_GRID_Y_LABELPAD)),
        )
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_by_label.setdefault(label, handle)
        max_plotted_cohort_count = max(
            max_plotted_cohort_count,
            int(draw_payload["plotted_cohort_count"] or 0),
        )
    cohorts = (
        [str(item) for item in payload.get("cohorts")]
        if isinstance(payload.get("cohorts"), list)
        else []
    )
    legend_labels = [
        display_group_label(cohort)
        for cohort in cohorts
        if display_group_label(cohort) in legend_by_label
    ]
    if not legend_labels:
        legend_labels = list(legend_by_label)
    legend_handles = [legend_by_label[label] for label in legend_labels]
    legend_y_offset_points = float(
        payload.get("legend", {}).get(
            "legend_y_offset_points",
            LONGITUDINAL_GRID_LEGEND_DOTPLOT_ALIGNED_Y_OFFSET_POINTS,
        )
        if isinstance(payload.get("legend"), dict)
        else LONGITUDINAL_GRID_LEGEND_DOTPLOT_ALIGNED_Y_OFFSET_POINTS
    )
    fig.supxlabel(
        str(payload.get("x_axis") or "Time After Merge (Days)"),
        fontsize=8,
        y=float(layout.get("x_axis_label_y", LONGITUDINAL_GRID_X_LABEL_Y)),
    )
    row_gap = float(layout.get("hspace", 0.45 if rows > 1 else 0.20))
    column_gap = float(layout.get("wspace", LONGITUDINAL_GRID_WSPACE))
    legend_max_columns = layout.get("legend_max_columns")
    if legend_max_columns is not None:
        legend_max_columns = int(legend_max_columns)
    final_row_wspace_multiplier = float(
        layout.get("final_row_wspace_multiplier", 1.0)
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=float(layout.get("bottom_margin", 0.18 if rows == 1 else 0.14)),
        top=_top_margin_for_grid_legend(
            max(len(legend_handles), max_plotted_cohort_count),
            legend_max_columns,
        ),
        wspace=column_gap,
        hspace=row_gap,
    )
    def add_legend_after_layout(_fig) -> None:
        if bool(layout.get("center_incomplete_final_row")):
            _center_incomplete_final_row_axes(
                axes,
                used_count=used_count,
                rows=rows,
                columns=columns,
                final_row_wspace_multiplier=final_row_wspace_multiplier,
            )
        if not legend_handles:
            return
        _add_grid_top_legend(
            fig,
            legend_handles,
            legend_labels,
            group_count=max(len(legend_handles), max_plotted_cohort_count),
            legend_y_offset_points=legend_y_offset_points,
            axes=[ax for ax in axes.flat if ax.get_visible()],
            legend_max_columns=legend_max_columns,
        )

    save_figure(
        fig,
        output_dir,
        stem,
        tight_layout_kwargs={"w_pad": column_gap},
        post_tight_layout_adjust_kwargs={
            "wspace": column_gap,
            "hspace": row_gap,
        },
        pre_save_callback=add_legend_after_layout,
    )
    plt.close(fig)


def _write_metric_line_plot_grid(
    *,
    output_dir: Path,
    grid_spec: dict[str, Any],
    results: dict[str, Any],
) -> None:
    metrics = tuple(grid_spec["metrics"])
    rows = int(grid_spec["rows"])
    columns = int(grid_spec["columns"])
    if not metrics:
        return
    output_dir = Path(output_dir)
    (output_dir / LONGITUDINAL_LINE_PLOT_GRID_OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=_grid_figure_size(
            rows,
            columns,
            width_override=grid_spec.get("figure_width_inches"),
            height_override=grid_spec.get("figure_height_inches"),
        ),
        squeeze=False,
    )
    grid_cohort_candidates: list[str] = []
    for metric in metrics:
        metric_payload = results.get("metrics", {}).get(metric, {})
        grid_cohort_candidates.extend(
            str(cohort)
            for cohort in (metric_payload.get("plot_series", {}) or {}).keys()
        )
    grid_cohorts = order_humans_first(grid_cohort_candidates)
    grid_cohort_colors = _resolved_cohort_colors(grid_cohorts)
    panels: list[dict[str, Any]] = []
    max_plotted_cohort_count = 0
    legend_by_label: dict[str, Any] = {}
    clamp_nonnegative_axis_to_zero = bool(
        grid_spec.get("clamp_nonnegative_axis_to_zero", False)
    )
    negative_visual_lower_limit = (
        float(grid_spec["negative_visual_lower_limit"])
        if grid_spec.get("negative_visual_lower_limit") is not None
        else None
    )
    line_width = float(grid_spec.get("line_width", LONGITUDINAL_GRID_LINE_WIDTH))
    marker_size = float(grid_spec.get("marker_size", LONGITUDINAL_LINE_MARKER_SIZE))
    grid_y_axis_label_overrides = (
        grid_spec.get("y_axis_label_overrides") or {}
    )
    if not isinstance(grid_y_axis_label_overrides, dict):
        grid_y_axis_label_overrides = {}
    grid_log_scale_metrics = {
        str(metric)
        for metric in (grid_spec.get("log_scale_metrics") or ())
    }
    cden_label_override = (
        "Cyclomatic Complexity\nper KLOC"
        if (
            str(grid_spec.get("stem"))
            == "maintainability_smell_quality_normalized_longitudinal_line_grid"
            and "CCDensity" in grid_spec.get("metrics", ())
        )
        else None
    )
    for index, ax in enumerate(axes.flat):
        if index >= len(metrics):
            ax.axis("off")
            continue
        metric = metrics[index]
        metric_payload = results.get("metrics", {}).get(metric, {})
        panel_payload = _draw_metric_line_axis(
            ax,
            metric=metric,
            metric_payload=metric_payload,
            show_x_axis_label=False,
            show_legend=False,
            y_axis_label_fontsize=float(
                grid_spec.get("y_axis_label_fontsize", 8.0)
            ),
            y_axis_label_override=(
                (str(override).strip() if isinstance(override, str) else None)
                if (override := grid_y_axis_label_overrides.get(metric)) is not None
                else cden_label_override
                if metric == "CCDensity"
                else None
            ),
            x_tick_fontsize=7.0,
            x_tick_labelpad=LONGITUDINAL_GRID_X_TICK_LABELPAD,
            y_tick_labelpad=LONGITUDINAL_GRID_Y_TICK_LABELPAD,
            y_labelpad=LONGITUDINAL_GRID_Y_LABELPAD,
            cohort_order=grid_cohorts,
            fixed_cohort_colors=grid_cohort_colors,
            clamp_nonnegative_axis_to_zero=clamp_nonnegative_axis_to_zero,
            negative_visual_lower_limit=negative_visual_lower_limit,
            line_width=line_width,
            marker_size=marker_size,
            axis_padding_fraction=float(
                grid_spec.get("axis_padding_fraction", 0.08)
            ),
            allow_zero_floor_when_positive=bool(
                grid_spec.get("allow_zero_floor_when_positive", True)
            ),
            force_log_scale=metric in grid_log_scale_metrics,
        )
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            legend_by_label.setdefault(label, handle)
        max_plotted_cohort_count = max(
            max_plotted_cohort_count,
            int(panel_payload["plotted_cohort_count"] or 0),
        )
        panels.append(
            {
                "metric": metric,
                "metric_label": METRIC_AXIS_LABELS.get(metric, metric),
                "row": index // columns,
                "column": index % columns,
                "plot_value_scale": panel_payload["plot_value_scale"],
                "plot_value_unit": panel_payload["plot_value_unit"],
                "y_axis_limits": panel_payload["y_axis_limits"],
                "overall": panel_payload["overall"],
                "per_cohort": panel_payload["per_cohort"],
                "agents_vs_humans": panel_payload["agents_vs_humans"],
                "cohorts": panel_payload["cohorts"],
                "cohort_colors": panel_payload["cohort_colors"],
                "line_width": panel_payload["line_width"],
                "marker_size": panel_payload["marker_size"],
                "plot_series": panel_payload["series"],
            }
        )
    legend_labels = [
        display_group_label(cohort)
        for cohort in grid_cohorts
        if display_group_label(cohort) in legend_by_label
    ]
    legend_handles = [legend_by_label[label] for label in legend_labels]
    legend_y_offset_points = float(
        grid_spec.get(
            "legend_y_offset_points",
            LONGITUDINAL_GRID_LEGEND_DOTPLOT_ALIGNED_Y_OFFSET_POINTS,
        )
    )
    x_axis_label_y = float(
        grid_spec.get("x_axis_label_y", LONGITUDINAL_GRID_X_LABEL_Y)
    )
    fig.supxlabel(
        "Time After Merge (Days)",
        fontsize=8,
        y=x_axis_label_y,
    )
    bottom_margin = float(
        grid_spec.get("bottom_margin", 0.18 if rows == 1 else 0.14)
    )
    row_gap = float(
        grid_spec.get("hspace", 0.45 if rows > 1 else 0.20)
    )
    column_gap = float(grid_spec.get("wspace", LONGITUDINAL_GRID_WSPACE))
    legend_max_columns = grid_spec.get("legend_max_columns")
    if legend_max_columns is not None:
        legend_max_columns = int(legend_max_columns)
    final_row_wspace_multiplier = float(
        grid_spec.get("final_row_wspace_multiplier", 1.0)
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=bottom_margin,
        top=_top_margin_for_grid_legend(
            max(len(legend_handles), max_plotted_cohort_count),
            legend_max_columns,
        ),
        wspace=column_gap,
        hspace=row_gap,
    )
    centered_final_row_holder: dict[str, bool] = {"centered": False}
    legend_payload_holder: dict[str, Any] = {"legend": None}

    def add_legend_after_layout(_fig) -> None:
        if bool(grid_spec.get("center_incomplete_final_row")):
            centered_final_row_holder["centered"] = _center_incomplete_final_row_axes(
                axes,
                used_count=len(metrics),
                rows=rows,
                columns=columns,
                final_row_wspace_multiplier=final_row_wspace_multiplier,
            )
        if not legend_handles:
            return
        legend_payload_holder["legend"] = _add_grid_top_legend(
            fig,
            legend_handles,
            legend_labels,
            group_count=max(len(legend_handles), max_plotted_cohort_count),
            legend_y_offset_points=legend_y_offset_points,
            axes=[ax for ax in axes.flat if ax.get_visible()],
            legend_max_columns=legend_max_columns,
        )

    stem = str(grid_spec["stem"])
    remove_plot_outputs(
        output_dir,
        (f"{LEGACY_LONGITUDINAL_LINE_PLOT_GRID_OUTPUT_DIR}/{stem}",),
    )
    save_figure(
        fig,
        output_dir / LONGITUDINAL_LINE_PLOT_GRID_OUTPUT_DIR,
        stem,
        tight_layout_kwargs={"w_pad": column_gap},
        post_tight_layout_adjust_kwargs={
            "wspace": column_gap,
            "hspace": row_gap,
        },
        pre_save_callback=add_legend_after_layout,
    )
    write_plot_data(
        output_dir / LONGITUDINAL_LINE_PLOT_GRID_OUTPUT_DIR,
        stem,
        {
            "plot": stem,
            "plot_type": "longitudinal_line_plot_grid",
            "layout": {
                "rows": rows,
                "columns": columns,
                "hidden_empty_panels": rows * columns - len(metrics),
                "figure_width_inches": _grid_figure_size(
                    rows,
                    columns,
                    width_override=grid_spec.get("figure_width_inches"),
                    height_override=grid_spec.get("figure_height_inches"),
                )[0],
                "figure_height_inches": _grid_figure_size(
                    rows,
                    columns,
                    width_override=grid_spec.get("figure_width_inches"),
                    height_override=grid_spec.get("figure_height_inches"),
                )[1],
                "bottom_margin": bottom_margin,
                "wspace": column_gap,
                "final_row_wspace_multiplier": final_row_wspace_multiplier,
                "hspace": row_gap,
                "legend_max_columns": legend_max_columns,
                "x_tick_labelpad": LONGITUDINAL_GRID_X_TICK_LABELPAD,
                "y_tick_labelpad": LONGITUDINAL_GRID_Y_TICK_LABELPAD,
                "y_labelpad": LONGITUDINAL_GRID_Y_LABELPAD,
                "x_axis_label_y": x_axis_label_y,
                "center_incomplete_final_row": centered_final_row_holder["centered"],
                "clamp_nonnegative_axis_to_zero": clamp_nonnegative_axis_to_zero,
                "negative_visual_lower_limit": negative_visual_lower_limit,
                "line_width": line_width,
                "marker_size": marker_size,
            },
            "metrics": list(metrics),
            "cohorts": grid_cohorts,
            "cohort_colors": grid_cohort_colors,
            "x_axis": "Time After Merge (Days)",
            "x_axis_scale": "ordinal_timepoints",
            "x_axis_padding": LONGITUDINAL_X_AXIS_PADDING,
            "x_axis_positions": TIMEPOINT_X_POSITIONS,
            "x_axis_tick_labels": _timepoint_tick_labels(),
            "legend": legend_payload_holder["legend"],
            "confidence_interval_band_alpha": LONGITUDINAL_CI_BAND_ALPHA,
            "confidence_interval_band_saturation_multiplier": (
                LONGITUDINAL_CI_BAND_SATURATION_MULTIPLIER
            ),
            "timepoints": list(LONGITUDINAL_TIMEPOINTS),
            "panels": panels,
        },
    )
    plt.close(fig)


def render_longitudinal_attrition_plot_from_payload(
    payload: dict[str, Any],
    output_dir: Path,
    stem: str,
) -> None:
    """Render longitudinal sample-retention attrition from saved plot data."""
    by_cohort = payload.get("by_cohort")
    if not isinstance(by_cohort, dict):
        raise ValueError("longitudinal attrition metadata requires by_cohort")
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, ATTRITION_FIGURE_HEIGHT))
    cohorts = order_humans_first(
        payload.get("cohorts")
        if isinstance(payload.get("cohorts"), list)
        else by_cohort.keys()
    )
    colors_source = (
        payload.get("cohort_colors")
        if isinstance(payload.get("cohort_colors"), dict)
        else None
    )
    cohort_colors = _resolved_cohort_colors(cohorts, colors_source)
    timepoints = [
        str(item)
        for item in (
            payload.get("timepoints")
            if isinstance(payload.get("timepoints"), list)
            else LONGITUDINAL_TIMEPOINTS
        )
    ]
    tick_labels = (
        [str(item) for item in payload.get("x_axis_tick_labels")]
        if isinstance(payload.get("x_axis_tick_labels"), list)
        else [_timepoint_tick_label(label) for label in timepoints]
    )
    positions = {
        str(key): float(value)
        for key, value in (
            payload.get("x_axis_positions")
            if isinstance(payload.get("x_axis_positions"), dict)
            else TIMEPOINT_X_POSITIONS
        ).items()
    }
    line_width = float(payload.get("line_width") or ATTRITION_LINE_WIDTH)
    marker_size = float(payload.get("marker_size") or ATTRITION_MARKER_SIZE)
    for cohort in cohorts:
        points = [
            by_cohort.get(cohort, {}).get(label, {})
            for label in timepoints
        ]
        x_values = [positions.get(label, index) for index, label in enumerate(timepoints)]
        y_values = _attrition_retention_percentage_values(points)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=line_width,
            markersize=marker_size,
            color=cohort_colors.get(cohort, "#4D4D4D"),
            label=display_group_label(cohort),
        )
    ax.set_xlabel(str(payload.get("x_axis") or "Time After Merge (Days)"))
    ax.set_ylabel("PRs with Available Snapshot (%)")
    ax.set_xticks([positions.get(label, index) for index, label in enumerate(timepoints)])
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=5.6)
    padding = float(payload.get("x_axis_padding") or LONGITUDINAL_X_AXIS_PADDING)
    ax.set_xlim(-padding, len(timepoints) - 1 + padding)
    ax.grid(axis="y", color="0.88", linestyle="-", linewidth=0.35)
    visual_lower = payload.get("visual_lower")
    if visual_lower is None:
        visual_lower = ATTRITION_AXIS_LOWER_BOUND - ATTRITION_AXIS_LOWER_PADDING
    visual_upper = payload.get("visual_upper")
    if visual_upper is None:
        logical_upper = payload.get("logical_upper_bound")
        if logical_upper is None:
            logical_upper = ATTRITION_AXIS_UPPER_BOUND
        visual_upper = float(logical_upper) + ATTRITION_AXIS_UPPER_PADDING
    ax.set_ylim(bottom=float(visual_lower), top=float(visual_upper))
    if cohorts:
        _add_top_legend(
            ax,
            group_count=len(cohorts),
            legend_max_columns=max(1, len(cohorts)),
        )
    fig.subplots_adjust(
        bottom=0.20,
        top=_top_margin_for_legend(len(cohorts)),
    )
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def _attrition_retention_percentage_values(
    points: list[dict[str, Any]],
) -> list[float]:
    if not points:
        return []
    baseline_raw = (
        points[0].get("baseline_pull_request_count")
        if isinstance(points[0], dict)
        else None
    )
    if baseline_raw is None and isinstance(points[0], dict):
        baseline_raw = points[0].get("pull_request_count")
    try:
        baseline = float(baseline_raw or 0.0)
    except (TypeError, ValueError):
        baseline = 0.0
    values: list[float] = []
    for index, point in enumerate(points):
        if index == 0 and point and baseline > 0.0:
            values.append(100.0)
            continue
        if baseline > 0.0:
            try:
                count = float(point.get("pull_request_count", 0.0) or 0.0)
            except (TypeError, ValueError):
                count = 0.0
            values.append(100.0 * count / baseline)
            continue
        try:
            values.append(float(point.get("retention_percentage", 0.0) or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def _write_attrition_plot(
    *,
    output_dir: Path,
    results: dict[str, Any],
) -> None:
    attrition_payload = results.get("attrition", {})
    attrition = attrition_payload.get("by_cohort", {})
    if not attrition:
        return
    (Path(output_dir) / "attrition").mkdir(parents=True, exist_ok=True)
    apply_ieee_plot_style()
    plt, _mdates = require_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, ATTRITION_FIGURE_HEIGHT))
    cohorts = order_humans_first(attrition.keys())
    cohort_colors = _resolved_cohort_colors(cohorts)
    for cohort in cohorts:
        points = [
            attrition.get(cohort, {}).get(label, {})
            for label in LONGITUDINAL_TIMEPOINTS
        ]
        x_values = [_timepoint_x(label) for label in LONGITUDINAL_TIMEPOINTS]
        y_values = _attrition_retention_percentage_values(points)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=ATTRITION_LINE_WIDTH,
            markersize=ATTRITION_MARKER_SIZE,
            color=cohort_colors.get(cohort, "#4D4D4D"),
            label=display_group_label(cohort),
        )
    ax.set_xlabel("Time After Merge (Days)")
    ax.set_ylabel("PRs with Available Snapshot (%)")
    ax.set_xticks(list(range(len(LONGITUDINAL_TIMEPOINTS))))
    ax.set_xticklabels(
        _timepoint_tick_labels(),
        rotation=0,
        ha="center",
        fontsize=5.6,
    )
    ax.set_xlim(
        -LONGITUDINAL_X_AXIS_PADDING,
        len(LONGITUDINAL_TIMEPOINTS) - 1 + LONGITUDINAL_X_AXIS_PADDING,
    )
    ax.grid(axis="y", color="0.88", linestyle="-", linewidth=0.35)
    ax.set_ylim(
        bottom=ATTRITION_AXIS_LOWER_BOUND - ATTRITION_AXIS_LOWER_PADDING,
        top=ATTRITION_AXIS_UPPER_BOUND + ATTRITION_AXIS_UPPER_PADDING,
    )
    legend_payload = None
    if cohorts:
        legend_payload = _add_top_legend(
            ax,
            group_count=len(cohorts),
            legend_max_columns=max(1, len(cohorts)),
        )
    fig.subplots_adjust(
        bottom=0.20,
        top=_top_margin_for_legend(len(cohorts)),
    )
    save_figure(fig, output_dir / "attrition", "longitudinal_attrition_by_cohort")
    write_plot_data(
        output_dir / "attrition",
        "longitudinal_attrition_by_cohort",
        {
            "plot": "longitudinal_attrition_by_cohort",
            "plot_type": "longitudinal_attrition_line_plot",
            "x_axis": "Time After Merge (Days)",
            "x_axis_scale": "ordinal_timepoints",
            "x_axis_padding": LONGITUDINAL_X_AXIS_PADDING,
            "x_axis_positions": TIMEPOINT_X_POSITIONS,
            "x_axis_tick_labels": _timepoint_tick_labels(),
            "legend": legend_payload,
            "cohorts": cohorts,
            "cohort_colors": cohort_colors,
            "line_width": ATTRITION_LINE_WIDTH,
            "marker_size": ATTRITION_MARKER_SIZE,
            "figure": {
                "width_inches": 3.5,
                "height_inches": ATTRITION_FIGURE_HEIGHT,
            },
            "y_axis": "PRs with Available Snapshot (%)",
            "y_value_field": "retention_percentage",
            "y_value_unit": "percent",
            "numerator": attrition_payload.get(
                "numerator",
                "pull_requests_with_available_snapshot",
            ),
            "denominator": attrition_payload.get(
                "denominator",
                "total_longitudinal_pull_request_count",
            ),
            "logical_lower_bound": ATTRITION_AXIS_LOWER_BOUND,
            "logical_upper_bound": ATTRITION_AXIS_UPPER_BOUND,
            "visual_lower_padding": ATTRITION_AXIS_LOWER_PADDING,
            "visual_lower": ATTRITION_AXIS_LOWER_BOUND - ATTRITION_AXIS_LOWER_PADDING,
            "visual_upper_padding": ATTRITION_AXIS_UPPER_PADDING,
            "visual_upper": ATTRITION_AXIS_UPPER_BOUND + ATTRITION_AXIS_UPPER_PADDING,
            "timepoints": list(LONGITUDINAL_TIMEPOINTS),
            "by_cohort": attrition,
            "by_authorship": results.get("attrition", {}).get("by_authorship", {}),
        },
    )
    plt.close(fig)


def write_longitudinal_line_plots(
    *,
    output_dir: Path | str,
    metrics: tuple[str, ...],
    results: dict[str, Any],
    x_labelpad: float | None = None,
    legend_y_offset_points: float = 0.0,
    legend_base_y: float = LONGITUDINAL_LINE_LEGEND_BASE_Y,
    negative_visual_lower_limit: float | None = None,
    figure_height_inches: float | None = None,
    center_last_legend_row_by_metric: dict[str, bool] | None = None,
    extra_plot_metadata_by_metric: dict[str, Any] | None = None,
) -> None:
    """Write one longitudinal line plot per metric from result summaries."""
    resolved_output_dir = Path(output_dir)
    for metric in metrics:
        metric_payload = results.get("metrics", {}).get(metric, {})
        _write_metric_line_plot(
            output_dir=resolved_output_dir,
            metric=metric,
            metric_payload=metric_payload,
            x_labelpad=x_labelpad,
            legend_y_offset_points=legend_y_offset_points,
            legend_base_y=legend_base_y,
            negative_visual_lower_limit=negative_visual_lower_limit,
            figure_height_inches=figure_height_inches,
            center_last_legend_row=(
                bool(
                    (center_last_legend_row_by_metric or {}).get(metric),
                )
            ),
            extra_plot_metadata=(
                (extra_plot_metadata_by_metric or {}).get(metric)
            ),
        )


def write_longitudinal_line_plot_grids(
    *,
    output_dir: Path | str,
    grid_specs: tuple[dict[str, Any], ...],
    results: dict[str, Any],
) -> None:
    """Write grouped longitudinal grid plots from result summaries."""
    resolved_output_dir = Path(output_dir)
    for grid_spec in grid_specs:
        _write_metric_line_plot_grid(
            output_dir=resolved_output_dir,
            grid_spec=grid_spec,
            results=results,
        )


def render_longitudinal_plot_from_payload(
    payload: dict[str, Any],
    output_dir: Path | str,
    stem: str,
) -> bool:
    """Dispatch a saved longitudinal plot-data payload to its renderer."""
    resolved_output_dir = Path(output_dir)
    plot_type = str(payload.get("plot_type") or "")
    if plot_type == "longitudinal_line_plot":
        render_longitudinal_line_plot_from_payload(payload, resolved_output_dir, stem)
        return True
    if plot_type == "longitudinal_line_plot_grid":
        render_longitudinal_line_plot_grid_from_payload(
            payload,
            resolved_output_dir,
            stem,
        )
        return True
    if plot_type == "longitudinal_attrition_line_plot":
        render_longitudinal_attrition_plot_from_payload(
            payload,
            resolved_output_dir,
            stem,
        )
        return True
    return False
