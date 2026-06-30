"""Plotting helpers shared by analysis pipelines."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from statistics import median
from math import isfinite
from typing import Any, Iterable


PLOT_FONT_FAMILY = "Times New Roman"
PLOT_EDGE_MARGIN = 0.01
PLOT_EDGE_RIGHT = 1.0 - PLOT_EDGE_MARGIN
PLOT_TIGHT_LAYOUT_PAD = 0.01
PLOT_SAVEFIG_PAD = 0.01
STACKED_BAR_MINIMUM_SEGMENT_HEIGHT = 6.0
Y_AXIS_SYMLOG_RANGE_THRESHOLD = 100.0
PLOT_FONT_FALLBACKS = (
    PLOT_FONT_FAMILY,
    "Liberation Serif",
    "DejaVu Serif",
)
_PLOT_FONT_FAMILY_CACHE: str | None = None
_PLOT_DATA_WRITES_ENABLED = True


# High-contrast fallback colors for cohorts not explicitly named below.
COHORT_COLORS = (
    "#0072B2",
    "#C44E52",
    "#E69F00",
    "#009E73",
    "#6A3D9A",
    "#CC79A7",
    "#4D4D4D",
    "#56B4E9",
    "#8C564B",
    "#17BECF",
    "#BCBD22",
    "#000000",
)
STACKED_BAR_RANK_COLORS = (
    "#C9C6C6",
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#F0E442",
)
PREFERRED_COHORT_ORDER = (
    "human",
    "humans",
    "codex",
    "copilot",
    "claude",
    "cursor",
    "jules",
    "devin",
)
STANDARD_COHORT_COLORS = {
    "human": "#0072B2",
    "humans": "#0072B2",
    "agent": "#D55E00",
    "agents": "#D55E00",
    "codex": "#C44E52",
    "copilot": "#E69F00",
    "claude": "#009E73",
    "cursor": "#6A3D9A",
    "jules": "#CC79A7",
    "devin": "#4D4D4D",
    "openhands": "#56B4E9",
    "junie": "#44AA99",
    "codegen": "#56B4E9",
    "cosine": "#44AA99",
}


def order_humans_first(groups: Iterable[str]) -> list[str]:
    """Return unique groups in the project-wide cohort display order."""
    normalized: dict[str, str] = {}
    for group in groups:
        label = str(group).strip()
        if not label:
            continue
        key = label.casefold()
        normalized.setdefault(key, label)
    preferred_order = {
        name: index for index, name in enumerate(PREFERRED_COHORT_ORDER)
    }
    def sort_key(label: str) -> tuple[int, str, str]:
        normalized_label = label.casefold()
        if normalized_label in preferred_order:
            return (preferred_order[normalized_label], normalized_label, label)
        if normalized_label in {"agent", "agents"}:
            return (len(preferred_order), normalized_label, label)
        return (len(preferred_order) + 1, normalized_label, label)

    return sorted(normalized.values(), key=sort_key)


def display_group_label(group: object) -> str:
    """Return a display label with the first visible character capitalized."""
    label = str(group).strip()
    if not label:
        return label
    return label[:1].upper() + label[1:]


def display_group_labels(groups: Iterable[object]) -> list[str]:
    """Return display labels for cohort/authorship groups."""
    return [display_group_label(group) for group in groups]


def add_xtick_count_sublabels(
    ax,
    x_values: Iterable[float],
    counts: Iterable[int | float],
    *,
    font_size: float = 6.5,
    y: float = -0.14,
    prefix: str = "n = ",
    secondary_counts: Iterable[int | float] | None = None,
    secondary_prefix: str = "p = ",
) -> None:
    """Add compact count labels below existing x-axis tick labels."""
    secondary_values = (
        list(secondary_counts) if secondary_counts is not None else None
    )

    def _format_count(value: int | float) -> str:
        try:
            display_count = int(round(float(value)))
        except (TypeError, ValueError):
            display_count = 0
        return f"{display_count:,}"

    for index, (x_value, count) in enumerate(zip(x_values, counts)):
        label = f"{prefix}{_format_count(count)}"
        if secondary_values is not None and index < len(secondary_values):
            label = (
                f"{label}\n"
                f"{secondary_prefix}{_format_count(secondary_values[index])}"
            )
        ax.text(
            float(x_value),
            y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=font_size,
            clip_on=False,
        )


def contrasting_text_color(hex_color: str) -> str:
    """Return black or white text for readable labels over a hex fill color."""
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


def order_labels_by_average_percentage(
    labels: Iterable[str],
    groups: Iterable[str],
    percentages_by_group: dict[str, dict[str, float]],
) -> list[str]:
    """Order labels by descending average percentage across groups."""
    label_list = list(labels)
    group_list = list(groups)
    original_index = {label: index for index, label in enumerate(label_list)}
    denominator = float(len(group_list)) if group_list else 1.0

    def average_percentage(label: str) -> float:
        return sum(
            float(percentages_by_group.get(group, {}).get(label, 0.0))
            for group in group_list
        ) / denominator

    return sorted(
        label_list,
        key=lambda label: (-average_percentage(label), original_index[label]),
    )


def ranked_stacked_bar_colors(
    labels: Iterable[str],
    palette: tuple[str, ...] = STACKED_BAR_RANK_COLORS,
) -> dict[str, str]:
    """Assign fill colors by stack rank rather than label identity."""
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(labels)
    }


def require_matplotlib():
    """Import matplotlib with a non-interactive backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "matplotlib is required for analysis plot generation."
        ) from exc
    return plt, mdates


def _selected_plot_font_family() -> str:
    """Return the first installed Times-style serif plotting font."""
    global _PLOT_FONT_FAMILY_CACHE
    if _PLOT_FONT_FAMILY_CACHE is not None:
        return _PLOT_FONT_FAMILY_CACHE
    try:
        from matplotlib import font_manager

        for family in PLOT_FONT_FALLBACKS:
            try:
                font_manager.findfont(family, fallback_to_default=False)
            except Exception:
                continue
            _PLOT_FONT_FAMILY_CACHE = family
            return family
    except Exception:
        pass
    _PLOT_FONT_FAMILY_CACHE = PLOT_FONT_FALLBACKS[-1]
    return _PLOT_FONT_FAMILY_CACHE


def apply_ieee_plot_style() -> None:
    """Apply compact serif plotting defaults suitable for paper figures."""
    plt, _mdates = require_matplotlib()
    plot_font_family = _selected_plot_font_family()
    plt.rcParams.update(
        {
            "font.family": [plot_font_family],
            "font.serif": list(PLOT_FONT_FALLBACKS),
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "mathtext.fontset": "dejavuserif",
            "mathtext.default": "regular",
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.2,
            "lines.markersize": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _json_ready(value: Any) -> Any:
    """Convert common analysis/plot values into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        converted = {str(key): _json_ready(child) for key, child in value.items()}
        if "adjusted_p_value" in converted and "p" not in converted:
            converted["p"] = converted.get("adjusted_p_value")
        return converted
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(child) for child in value]
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            pass
    return str(value)


def write_plot_data(output_dir: Path | str, stem: str, payload: dict[str, Any]) -> None:
    """Write compact sibling JSON data for a generated plot."""
    if not _PLOT_DATA_WRITES_ENABLED:
        return
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    with (resolved_output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def set_plot_data_writes_enabled(enabled: bool) -> bool:
    """Set whether plot sidecar JSON writes/deletes are enabled; return old value."""
    global _PLOT_DATA_WRITES_ENABLED
    previous = _PLOT_DATA_WRITES_ENABLED
    _PLOT_DATA_WRITES_ENABLED = bool(enabled)
    return previous


def plot_data_writes_enabled() -> bool:
    """Return whether plot sidecar JSON writes/deletes are currently enabled."""
    return _PLOT_DATA_WRITES_ENABLED


def remove_plot_outputs(output_dir: Path | str, stems: tuple[str, ...]) -> None:
    """Remove generated plot files and sibling JSON for disabled/stale stems."""
    resolved_output_dir = Path(output_dir)
    suffixes = (".pdf", ".png", ".json") if _PLOT_DATA_WRITES_ENABLED else (".pdf", ".png")
    for stem in stems:
        for suffix in suffixes:
            (resolved_output_dir / f"{stem}{suffix}").unlink(missing_ok=True)


def cohort_color_map(cohorts: Iterable[str]) -> dict[str, str]:
    """Return a stable colorblind-safe color map for cohorts/authorship groups."""
    color_map: dict[str, str] = {}
    fallback_index = 0
    for cohort in order_humans_first(str(cohort) for cohort in cohorts):
        normalized = str(cohort).strip().casefold()
        if normalized in STANDARD_COHORT_COLORS:
            color_map[str(cohort)] = STANDARD_COHORT_COLORS[normalized]
            continue
        while COHORT_COLORS[fallback_index % len(COHORT_COLORS)] in set(
            STANDARD_COHORT_COLORS.values()
        ):
            fallback_index += 1
        color_map[str(cohort)] = COHORT_COLORS[fallback_index % len(COHORT_COLORS)]
        fallback_index += 1
    return color_map


def human_baseline_group(groups: Iterable[str]) -> str | None:
    """Return the human/humans group label when present."""
    for group in groups:
        label = str(group).strip()
        if label.casefold() in {"human", "humans"}:
            return str(group)
    return None


def add_human_median_baseline(ax, grouped: dict[str, Iterable[float]]) -> None:
    """Draw a faint blue dashed reference line at the human baseline median."""
    baseline_group = human_baseline_group(grouped.keys())
    if baseline_group is None:
        return
    values = [float(value) for value in grouped.get(baseline_group, [])]
    if not values:
        return
    ax.axhline(
        float(median(values)),
        color=STANDARD_COHORT_COLORS["human"],
        linestyle=(0, (4, 2)),
        linewidth=0.8,
        alpha=0.35,
        zorder=1,
    )


def _finite_float_values(values: Iterable[object]) -> list[float]:
    """Return finite floats while ignoring nulls, strings, and infinities."""
    finite_values: list[float] = []
    for value in values:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if isfinite(numeric):
            finite_values.append(numeric)
    return finite_values


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for sorted plot bounds."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    bounded = min(max(float(percentile), 0.0), 100.0)
    position = (bounded / 100.0) * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def percentile_capped_y_limits(
    grouped: dict[str, Iterable[float]] | Iterable[Iterable[float]] | Iterable[float],
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    force_zero_for_nonnegative: bool = True,
) -> dict[str, float | int | bool] | None:
    """Return display-only y-limits based on percentile-capped plot ranges."""
    if isinstance(grouped, dict):
        raw_groups = list(grouped.values())
    else:
        raw_groups = list(grouped)
    if not raw_groups:
        return None

    flattened: list[float] = []
    if all(isinstance(item, (list, tuple, set)) for item in raw_groups):
        for values in raw_groups:  # type: ignore[assignment]
            flattened.extend(_finite_float_values(values))
    else:
        flattened = _finite_float_values(raw_groups)  # type: ignore[arg-type]
    if not flattened:
        return None

    actual_min = min(flattened)
    actual_max = max(flattened)
    lower = _percentile(flattened, lower_percentile)
    upper = _percentile(flattened, upper_percentile)

    if force_zero_for_nonnegative and actual_min >= 0.0:
        lower = 0.0
    elif actual_max <= 0.0:
        upper = 0.0
    elif lower < 0.0 < upper:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)

    if lower == upper:
        padding = max(abs(lower) * 0.08, 0.05)
        lower -= 0.0 if force_zero_for_nonnegative and actual_min >= 0.0 else padding
        upper += padding
    else:
        padding = (upper - lower) * 0.06
        if force_zero_for_nonnegative and actual_min >= 0.0:
            upper += padding
        else:
            lower -= padding
            upper += padding

    if force_zero_for_nonnegative and actual_min >= 0.0:
        lower = 0.0

    below = sum(1 for value in flattened if value < lower)
    above = sum(1 for value in flattened if value > upper)
    return {
        "lower": float(lower),
        "upper": float(upper),
        "below_count": int(below),
        "above_count": int(above),
        "total_count": int(len(flattened)),
        "is_clipped": bool(below or above),
    }


def apply_percentile_capped_y_axis(
    ax,
    grouped: dict[str, Iterable[float]] | Iterable[Iterable[float]] | Iterable[float],
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    force_zero_for_nonnegative: bool = True,
    note_font_size: float = 4.6,
    show_note: bool = True,
) -> dict[str, float | int | bool] | None:
    """Set plot y-limits to percentile-capped display ranges without changing data."""
    limits = percentile_capped_y_limits(
        grouped,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        force_zero_for_nonnegative=force_zero_for_nonnegative,
    )
    if limits is None:
        return None
    ax.set_ylim(float(limits["lower"]), float(limits["upper"]))
    if show_note and limits["is_clipped"]:
        parts = []
        if int(limits["above_count"]):
            parts.append(f"{int(limits['above_count'])} Above")
        if int(limits["below_count"]):
            parts.append(f"{int(limits['below_count'])} Below")
        ax.text(
            0.995,
            0.985,
            (
                f"Axis Clipped To P{lower_percentile:g}-P{upper_percentile:g}; "
                + ", ".join(parts)
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=note_font_size,
            color="0.25",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )
    return limits


def apply_symlog_y_axis_if_range_exceeds(
    ax,
    *,
    range_threshold: float = Y_AXIS_SYMLOG_RANGE_THRESHOLD,
    linthresh: float = 1.0,
    small_negative_padding_threshold: float = 1.0,
) -> str:
    """Use symlog only when the visible y-axis range is large enough."""
    bottom, top = ax.get_ylim()
    effective_bottom = float(bottom)
    if (
        effective_bottom < 0.0
        and float(top) > 0.0
        and abs(effective_bottom) <= float(small_negative_padding_threshold)
    ):
        effective_bottom = 0.0
    plotted_range = float(top) - effective_bottom
    if plotted_range > float(range_threshold):
        ax.set_yscale("symlog", linthresh=linthresh)
        return "symlog"
    ax.set_yscale("linear")
    return "linear"


def add_boxplot_summary_labels(
    ax,
    groups: Iterable[str],
    labels_by_group: dict[str, str],
    *,
    font_size: float = 4.6,
) -> None:
    """Add compact above-axis boxplot summaries at a common height."""
    ordered_groups = [str(group) for group in groups]
    if not ordered_groups:
        return
    for index, group in enumerate(ordered_groups, start=1):
        label = labels_by_group.get(group)
        if not label:
            continue
        ax.text(
            index,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=font_size,
            linespacing=0.82,
            clip_on=False,
        )


def ieee_boxplot_kwargs() -> dict[str, Any]:
    """Return compact boxplot defaults for paper figures."""
    return {
        "patch_artist": True,
        "showfliers": False,
        "showmeans": True,
        "meanline": False,
        "meanprops": {
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 0.45,
            "markersize": 3.0,
            "zorder": 5,
        },
        "medianprops": {
            "color": "black",
            "linestyle": "-",
            "linewidth": 1.45,
            "zorder": 4,
        },
        "boxprops": {
            "facecolor": "white",
            "linewidth": 1.0,
            "zorder": 2.5,
        },
        "whiskerprops": {
            "linewidth": 0.8,
            "zorder": 2.5,
        },
        "capprops": {
            "linewidth": 0.8,
            "zorder": 2.5,
        },
        "flierprops": {
            "marker": ".",
            "markersize": 1.2,
            "markerfacecolor": "0.25",
            "markeredgecolor": "0.25",
            "alpha": 0.22,
        },
    }


def style_ieee_boxplot(
    boxplot: dict[str, Any],
    colors: Iterable[str],
    *,
    box_alpha: float = 0.96,
) -> None:
    """Style boxplots as white boxes with colored borders."""
    color_list = [str(color) for color in colors]
    for index, (patch, color) in enumerate(zip(boxplot.get("boxes", []), color_list)):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_alpha(box_alpha)
        patch.set_linewidth(1.0)
        if hasattr(patch, "set_zorder"):
            patch.set_zorder(2.5)
        for artist in boxplot.get("whiskers", [])[2 * index : 2 * index + 2]:
            artist.set_color(color)
            artist.set_linewidth(0.8)
            if hasattr(artist, "set_alpha"):
                artist.set_alpha(0.95)
        for artist in boxplot.get("caps", [])[2 * index : 2 * index + 2]:
            artist.set_color(color)
            artist.set_linewidth(0.8)
            if hasattr(artist, "set_alpha"):
                artist.set_alpha(0.95)
    for artist in boxplot.get("medians", []):
        artist.set_color("black")
        artist.set_linewidth(1.45)
        if hasattr(artist, "set_zorder"):
            artist.set_zorder(4)
    for artist in boxplot.get("means", []):
        artist.set_marker("D")
        artist.set_markerfacecolor("white")
        artist.set_markeredgecolor("black")
        artist.set_markeredgewidth(0.45)
        artist.set_markersize(3.0)
        if hasattr(artist, "set_zorder"):
            artist.set_zorder(5)
    for artist in boxplot.get("fliers", []):
        artist.set_marker(".")
        artist.set_markersize(1.2)
        artist.set_markerfacecolor("0.25")
        artist.set_markeredgecolor("0.25")
        artist.set_alpha(0.22)


def add_violin_underlay(
    ax,
    values_by_position: Iterable[Iterable[object]],
    *,
    positions: Iterable[float] | None = None,
    colors: Iterable[str] | None = None,
    width: float = 0.72,
    alpha: float = 0.24,
) -> None:
    """Draw a faded violin distribution behind a boxplot."""
    groups = [_finite_float_values(values) for values in values_by_position]
    position_list = (
        [float(position) for position in positions]
        if positions is not None
        else [float(index) for index in range(1, len(groups) + 1)]
    )
    del colors
    neutral_color = "#BDBDBD"
    plotted_groups: list[list[float]] = []
    plotted_positions: list[float] = []
    for index, values in enumerate(groups):
        if index >= len(position_list) or len(set(values)) < 2:
            continue
        plotted_groups.append(values)
        plotted_positions.append(position_list[index])
    if not plotted_groups:
        return
    parts = ax.violinplot(
        plotted_groups,
        positions=plotted_positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts.get("bodies", []):
        body.set_facecolor(neutral_color)
        body.set_edgecolor(neutral_color)
        body.set_alpha(alpha)
        body.set_linewidth(0.45)
        body.set_zorder(1)

def style_violin_inner_lines(
    parts: dict[str, object],
    *,
    color: str = "black",
    linewidth: float = 0.45,
) -> None:
    """Make violin mean/median/extrema bars thin and black."""
    for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
        artist = parts.get(key)
        if artist is None:
            continue
        if hasattr(artist, "set_color"):
            artist.set_color(color)
        if hasattr(artist, "set_edgecolor"):
            artist.set_edgecolor(color)
        if hasattr(artist, "set_linewidth"):
            artist.set_linewidth(linewidth)
        if hasattr(artist, "set_alpha"):
            artist.set_alpha(0.95)


def save_figure(
    fig,
    output_dir: Path | str,
    stem: str,
    *,
    use_tight_layout: bool = True,
    tight_layout_kwargs: dict[str, Any] | None = None,
    post_tight_layout_adjust_kwargs: dict[str, Any] | None = None,
    pre_save_callback: Any | None = None,
) -> None:
    """Save a figure as PDF and 300-dpi PNG."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    plot_font_family = _selected_plot_font_family()
    for text_artist in fig.findobj(
        lambda artist: hasattr(artist, "set_fontfamily")
        and hasattr(artist, "get_text")
    ):
        text_artist.set_fontfamily(plot_font_family)
    if use_tight_layout:
        layout_kwargs: dict[str, Any] = {"pad": PLOT_TIGHT_LAYOUT_PAD}
        if tight_layout_kwargs is not None:
            layout_kwargs.update(tight_layout_kwargs)
        layout_kwargs["pad"] = PLOT_TIGHT_LAYOUT_PAD
        layout_kwargs["rect"] = (
            PLOT_EDGE_MARGIN,
            0.0,
            PLOT_EDGE_RIGHT,
            1.0,
        )
        fig.tight_layout(**layout_kwargs)
    if post_tight_layout_adjust_kwargs is not None:
        fig.subplots_adjust(**post_tight_layout_adjust_kwargs)
    if pre_save_callback is not None:
        pre_save_callback(fig)
    fig.savefig(
        resolved_output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=PLOT_SAVEFIG_PAD,
    )
    fig.savefig(
        resolved_output_dir / f"{stem}.png",
        bbox_inches="tight",
        pad_inches=PLOT_SAVEFIG_PAD,
        dpi=300,
    )


def stacked_bar_visual_percentages(
    percentages: Iterable[float],
    *,
    minimum_segment_height: float = STACKED_BAR_MINIMUM_SEGMENT_HEIGHT,
) -> list[float]:
    """Return adjusted stacked-bar heights while preserving zero segments."""
    true_percentages = [
        max(0.0, float(percentage))
        for percentage in percentages
    ]
    nonzero_indexes = [
        index
        for index, percentage in enumerate(true_percentages)
        if percentage > 0.0
    ]
    if not nonzero_indexes:
        return [0.0 for _percentage in true_percentages]

    minimum_height = max(0.0, float(minimum_segment_height))
    if minimum_height <= 0.0:
        total = sum(true_percentages)
        if total <= 0.0:
            return [0.0 for _percentage in true_percentages]
        return [100.0 * percentage / total for percentage in true_percentages]

    if minimum_height * len(nonzero_indexes) >= 100.0:
        equal_height = 100.0 / len(nonzero_indexes)
        return [
            equal_height if index in nonzero_indexes else 0.0
            for index in range(len(true_percentages))
        ]

    visual_percentages = [0.0 for _percentage in true_percentages]
    remaining_indexes = list(nonzero_indexes)
    remaining_space = 100.0
    while remaining_indexes:
        remaining_total = sum(true_percentages[index] for index in remaining_indexes)
        if remaining_total <= 0.0:
            equal_height = remaining_space / len(remaining_indexes)
            for index in remaining_indexes:
                visual_percentages[index] = equal_height
            break

        scale = remaining_space / remaining_total
        below_minimum = [
            index
            for index in remaining_indexes
            if true_percentages[index] * scale < minimum_height
        ]
        if not below_minimum:
            for index in remaining_indexes:
                visual_percentages[index] = true_percentages[index] * scale
            break

        below_minimum_set = set(below_minimum)
        for index in below_minimum:
            visual_percentages[index] = minimum_height
        remaining_indexes = [
            index for index in remaining_indexes if index not in below_minimum_set
        ]
        remaining_space -= minimum_height * len(below_minimum)
        if remaining_space <= 0.0 and remaining_indexes:
            equal_height = 100.0 / len(nonzero_indexes)
            return [
                equal_height if index in nonzero_indexes else 0.0
                for index in range(len(true_percentages))
            ]

    visual_total = sum(visual_percentages)
    if visual_total > 0.0:
        visual_percentages = [
            100.0 * percentage / visual_total
            for percentage in visual_percentages
        ]
    return visual_percentages


def stacked_bar_visual_metadata(
    *,
    minimum_segment_height: float = STACKED_BAR_MINIMUM_SEGMENT_HEIGHT,
) -> dict[str, Any]:
    """Return plot JSON metadata for visually adjusted stacked bars."""
    return {
        "minimum_segment_height_applied": True,
        "minimum_segment_height_percentage": float(minimum_segment_height),
        "visual_percentages_are_adjusted": True,
    }


def add_stacked_bar_percentage_callouts(
    ax,
    x_value: float,
    segments: Iterable[tuple[float, float]],
    *,
    x_offset: float = 0.44,
    min_vertical_gap: float = 4.5,
    font_size: float = 5.0,
    bar_half_width: float = 0.4,
    edge_overlap: float = 0.04,
) -> None:
    """Add readable side callouts for small stacked-bar percentage segments."""
    labels = []
    for bottom, percentage in segments:
        if percentage <= 0.0:
            continue
        labels.append(
            {
                "center": float(bottom) + float(percentage) / 2.0,
                "text": f"{float(percentage):.1f}%",
            }
        )
    if not labels:
        return

    labels.sort(key=lambda item: item["center"])
    for index, label in enumerate(labels):
        label["y"] = min(max(float(label["center"]), 2.0), 98.0)
        if index > 0:
            previous_y = float(labels[index - 1]["y"])
            if float(label["y"]) - previous_y < min_vertical_gap:
                label["y"] = previous_y + min_vertical_gap

    overflow = float(labels[-1]["y"]) - 98.0
    if overflow > 0.0:
        shift = min(overflow, min(float(label["y"]) for label in labels) - 2.0)
        if shift > 0.0:
            for label in labels:
                label["y"] = float(label["y"]) - shift

    for label in labels:
        x_offset_value = float(x_offset)
        edge_sign = 1.0 if x_offset_value >= 0.0 else -1.0
        line_start_x = x_value + edge_sign * (
            float(bar_half_width) - float(edge_overlap)
        )
        ax.annotate(
            str(label["text"]),
            xy=(line_start_x, float(label["y"])),
            xytext=(x_value + x_offset_value, float(label["y"])),
            ha="right" if x_offset_value < 0.0 else "left",
            va="center",
            fontsize=font_size,
            color="black",
            annotation_clip=False,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.35,
                "color": "black",
                "shrinkA": 0,
                "shrinkB": 0,
            },
            clip_on=False,
        )
