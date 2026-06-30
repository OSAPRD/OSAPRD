"""Longitudinal analysis for maintainability metrics.

This companion pipeline consumes longitudinal values already accumulated by the
maintainability stream. It writes line plots and a compact JSON manifest for
code-smell, duplication, complexity, Halstead, and maintainability-index trends.
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


OUTPUT_FILENAME = "longitudinal_maintainability_metrics_results.json"
LONGITUDINAL_MAINTAINABILITY_METRICS = (
    "SmellCount",
    "MI",
    "CC",
    "HV",
    "CCDensity",
    "HVDensity",
    "DuplicationDensity",
    "CommentDensity",
    "CodeSmellDensity",
    "NLOC",
    "KLOC",
)
LONGITUDINAL_MAINTAINABILITY_PLOT_METRICS = (
    "SmellCount",
    "MI",
    "CC",
    "HV",
    "CCDensity",
    "HVDensity",
    "DuplicationDensity",
    "CommentDensity",
    "CodeSmellDensity",
)
LONGITUDINAL_MAINTAINABILITY_GRID_SPECS = (
    {
        "stem": "maintainability_smell_count_density_longitudinal_line_grid",
        "rows": 1,
        "columns": 2,
        "metrics": ("SmellCount", "CodeSmellDensity"),
        "figure_height_inches": 2.0,
        "line_width": 0.75,
        "marker_size": 1.25,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "legend_max_columns": 20,
        "negative_visual_lower_limit": -5.0,
        "wspace": 0.25,
    },
    {
        "stem": "maintainability_quality_longitudinal_line_grid",
        "rows": 2,
        "columns": 3,
        "center_incomplete_final_row": True,
        "figure_height_inches": 3.5,
        "line_width": 0.75,
        "marker_size": 1.25,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "legend_max_columns": 20,
        "negative_visual_lower_limit": -5.0,
        "bottom_margin": 0.14,
        "hspace": 0.15,
        "wspace": 0.25,
        "metrics": (
            "CC",
            "HV",
            "DuplicationDensity",
            "CommentDensity",
            "MI",
        ),
    },
    {
        "stem": "maintainability_smell_quality_longitudinal_line_grid",
        "rows": 2,
        "columns": 4,
        "center_incomplete_final_row": True,
        "figure_height_inches": 3.5,
        "line_width": 0.75,
        "marker_size": 1.25,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "negative_visual_lower_limit": -5.0,
        "bottom_margin": 0.14,
        "hspace": 0.15,
        "wspace": 0.25,
        "final_row_wspace_multiplier": 2.0,
        "metrics": (
            "SmellCount",
            "CodeSmellDensity",
            "DuplicationDensity",
            "CommentDensity",
            "CC",
            "HV",
            "MI",
        ),
    },
    {
        "stem": "maintainability_smell_quality_normalized_longitudinal_line_grid",
        "rows": 2,
        "columns": 4,
        "center_incomplete_final_row": True,
        "figure_height_inches": 3.0,
        "line_width": 0.75,
        "marker_size": 1.25,
        "y_axis_label_fontsize": 6.5,
        "x_axis_label_y": -0.02,
        "legend_y_offset_points": 4.0,
        "legend_max_columns": 20,
        "negative_visual_lower_limit": -5.0,
        "bottom_margin": 0.14,
        "hspace": 0.2,
        "wspace": 0.28,
        "allow_zero_floor_when_positive": False,
        "final_row_wspace_multiplier": 2.0,
        "log_scale_metrics": ("HVDensity",),
        "metrics": (
            "SmellCount",
            "CodeSmellDensity",
            "DuplicationDensity",
            "CommentDensity",
            "CCDensity",
            "HVDensity",
            "MI",
        ),
    },
)
REMOVED_PLOT_OUTPUT_STEMS = longitudinal_line_plot_stems(
    (
        "SmellIntroCount",
        "SmellIntroRate",
        "SmellRegressionCount",
        "SmellRegressionRate",
        "SmellFixCount",
        "SmellFixRate",
        "SmellsFixed",
        "CodeSmellIntroRate",
        "CodeSmellFixRate",
    )
) + ("attrition/longitudinal_attrition_by_cohort",)
PLOT_OUTPUT_STEMS = longitudinal_line_plot_stems(
    LONGITUDINAL_MAINTAINABILITY_PLOT_METRICS
)
PLOT_OUTPUT_STEMS = (
    *PLOT_OUTPUT_STEMS,
    *longitudinal_line_plot_grid_stems(LONGITUDINAL_MAINTAINABILITY_GRID_SPECS),
)


def plot_output_stems() -> tuple[str, ...]:
    """Return expected longitudinal maintainability plot stems."""
    return PLOT_OUTPUT_STEMS


def _main_json_payload(results: dict[str, Any]) -> dict[str, Any]:
    """Build the small JSON manifest for longitudinal maintainability outputs."""
    return {
        "eligible_pr_count": results.get("eligible_pr_count", 0),
        "timepoints": results.get("timepoints", []),
        "metrics": list((results.get("metrics") or {}).keys()),
        "plot_metrics": list(LONGITUDINAL_MAINTAINABILITY_PLOT_METRICS),
    }


def run_longitudinal_maintainability_metrics_analysis_from_payload(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    analysis_output_dir: Path | str,
    analysis_kind: str = "maintainability",
    logger: Any | None = None,
    plot_only: bool = False,
) -> dict[str, Any]:
    """Write longitudinal maintainability JSON and plots from streaming payloads."""
    results = build_longitudinal_results_from_payload(
        longitudinal_values,
        metrics=LONGITUDINAL_MAINTAINABILITY_METRICS,
    )
    output_dir = longitudinal_output_dir(analysis_output_dir, analysis_kind)
    remove_plot_outputs(output_dir, REMOVED_PLOT_OUTPUT_STEMS)
    if logger is not None:
        logger.log("writing_maintainability_longitudinal_line_plots")
    write_longitudinal_line_plots(
        output_dir=output_dir,
        metrics=LONGITUDINAL_MAINTAINABILITY_PLOT_METRICS,
        results=results,
        x_labelpad=5.0,
        negative_visual_lower_limit=-5.0,
    )
    if logger is not None:
        logger.log("writing_maintainability_longitudinal_line_plot_grids")
    write_longitudinal_line_plot_grids(
        output_dir=output_dir,
        grid_specs=LONGITUDINAL_MAINTAINABILITY_GRID_SPECS,
        results=results,
    )
    if not plot_only:
        write_longitudinal_json(
            longitudinal_output_path(
                analysis_output_dir,
                analysis_kind,
                OUTPUT_FILENAME,
            ),
            _main_json_payload(results),
        )
    return results


def main() -> None:
    """Reject direct execution; this companion needs an in-memory payload."""
    raise SystemExit(
        "longitudinal_maintainability_metrics_pipeline.py is called from "
        "maintainability_analysis_pipeline.py with streaming payloads."
    )


if __name__ == "__main__":
    main()
