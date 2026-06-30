"""Longitudinal analysis for refactoring metrics.

This companion pipeline consumes longitudinal values already accumulated by the
refactoring stream. It measures persistence of original PR refactoring effects
over future snapshots and writes line plots plus a compact JSON manifest.
"""

from __future__ import annotations

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
from longitudinal_analysis_utility import (
    build_longitudinal_results_from_payload,
    longitudinal_output_dir,
    longitudinal_output_path,
    write_longitudinal_json,
)
from plotting_utility import remove_plot_outputs


OUTPUT_FILENAME = "longitudinal_refactoring_results.json"
LONGITUDINAL_REFACTORING_METRICS = (
    "RefCount",
    "RefDensity",
    "RefDiversity",
    "RefMagLines",
    "RefRetentionRate",
    "RefZoneFutureTouchedLines",
    "FutureTouchingCommits",
)
LONGITUDINAL_REFACTORING_PLOT_METRICS = (
    "RefRetentionRate",
    "RefZoneFutureTouchedLines",
    "FutureTouchingCommits",
)
LONGITUDINAL_REFACTORING_GRID_SPECS = (
    {
        "stem": "refactoring_retention_zone_touching_commits_longitudinal_line_grid",
        "rows": 1,
        "columns": 3,
        "metrics": (
            "RefRetentionRate",
            "RefZoneFutureTouchedLines",
            "FutureTouchingCommits",
        ),
        "figure_height_inches": 1.5,
        "line_width": 0.75,
        "marker_size": 1.25,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "wspace": 0.18,
        "y_axis_label_fontsize": 7.0,
        "legend_max_columns": 10,
    },
    {
        "stem": "refactoring_retention_zone_longitudinal_line_grid",
        "rows": 1,
        "columns": 2,
        "metrics": ("RefRetentionRate", "RefZoneFutureTouchedLines"),
        "figure_height_inches": 1.25,
        "line_width": 0.75,
        "marker_size": 1.25,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "wspace": 0.18,
        "legend_max_columns": 10,
        "y_axis_label_overrides": {
            "RefZoneFutureTouchedLines": "Future Lines Touched\nper RefOp",
        },
    },
    {
        "stem": "refactoring_retention_touching_commits_longitudinal_line_grid",
        "rows": 1,
        "columns": 2,
        "metrics": ("RefRetentionRate", "FutureTouchingCommits"),
        "figure_height_inches": 1.75,
        "line_width": 0.75,
        "marker_size": 1.25,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "wspace": 0.18,
        "legend_max_columns": 10,
    },
    {
        "stem": "refactoring_zone_touching_commits_longitudinal_line_grid",
        "rows": 1,
        "columns": 2,
        "metrics": ("RefZoneFutureTouchedLines", "FutureTouchingCommits"),
        "figure_height_inches": 1.75,
        "line_width": 0.75,
        "marker_size": 1.25,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "wspace": 0.25,
        "legend_max_columns": 10,
    },
)
REMOVED_PLOT_OUTPUT_STEMS = longitudinal_line_plot_stems(
    (
        "RefCount",
        "RefDensity",
        "RefDiversity",
        "RefMagLines",
    )
) + ("attrition/longitudinal_attrition_by_cohort",)
PLOT_OUTPUT_STEMS = longitudinal_line_plot_stems(
    LONGITUDINAL_REFACTORING_PLOT_METRICS
)
PLOT_OUTPUT_STEMS = (
    *PLOT_OUTPUT_STEMS,
    *longitudinal_line_plot_grid_stems(LONGITUDINAL_REFACTORING_GRID_SPECS),
)


def plot_output_stems() -> tuple[str, ...]:
    """Return expected longitudinal refactoring plot stems."""
    return PLOT_OUTPUT_STEMS


def _main_json_payload(results: dict[str, Any]) -> dict[str, Any]:
    """Build the small JSON manifest for longitudinal refactoring outputs."""
    return {
        "eligible_pr_count": results.get("eligible_pr_count", 0),
        "timepoints": results.get("timepoints", []),
        "metrics": list((results.get("metrics") or {}).keys()),
        "plot_metrics": list(LONGITUDINAL_REFACTORING_PLOT_METRICS),
    }


def run_longitudinal_refactoring_analysis_from_payload(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    analysis_output_dir: Path | str,
    logger: Any | None = None,
    plot_only: bool = False,
) -> dict[str, Any]:
    """Write longitudinal refactoring JSON and line plots from streaming payloads."""
    results = build_longitudinal_results_from_payload(
        longitudinal_values,
        metrics=LONGITUDINAL_REFACTORING_METRICS,
    )
    output_dir = longitudinal_output_dir(analysis_output_dir, "refactoring")
    remove_plot_outputs(output_dir, REMOVED_PLOT_OUTPUT_STEMS)
    if logger is not None:
        logger.log("writing_refactoring_longitudinal_line_plots")
    write_longitudinal_line_plots(
        output_dir=output_dir,
        metrics=LONGITUDINAL_REFACTORING_PLOT_METRICS,
        results=results,
        x_labelpad=6.0,
        legend_y_offset_points=1.25,
        figure_height_inches=1.75,
        center_last_legend_row_by_metric={"FutureTouchingCommits": True},
    )
    if logger is not None:
        logger.log("writing_refactoring_longitudinal_line_plot_grids")
    write_longitudinal_line_plot_grids(
        output_dir=output_dir,
        grid_specs=LONGITUDINAL_REFACTORING_GRID_SPECS,
        results=results,
    )
    if not plot_only:
        write_longitudinal_json(
            longitudinal_output_path(
                analysis_output_dir,
                "refactoring",
                OUTPUT_FILENAME,
            ),
            _main_json_payload(results),
        )
    return results


def main() -> None:
    """Reject direct execution; this companion needs an in-memory payload."""
    raise SystemExit(
        "longitudinal_refactoring_pipeline.py is called from "
        "refactoring_analysis_pipeline.py with streaming payloads."
    )


if __name__ == "__main__":
    main()
