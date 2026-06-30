"""Characteristics analysis for maintainability metrics.

This companion pipeline is called by ``maintainability_analysis_pipeline`` after
streaming has produced compact metric and smell-count payloads. It writes
language, popularity, and optional topic-domain plots without reopening parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ANALYSIS_DIR / "config"
PLOTTER_DIR = ANALYSIS_DIR / "plotters"
UTILITY_DIR = ANALYSIS_DIR / "utility"
for candidate in (CONFIG_DIR, PLOTTER_DIR, UTILITY_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from characteristics_analysis_plotter import (
    characteristic_heatmap_stems,
    characteristic_composition_stacked_bar_stems,
    characteristic_median_ci_dotplot_stems,
    characteristic_plot_stems,
    write_characteristics_composition_stacked_bars_from_payload,
    write_extended_characteristics_heatmaps,
    write_characteristics_median_ci_dotplots_from_payload,
    write_characteristics_plots,
)
from characteristics_analysis_utility import (
    build_characteristics_results_from_payload,
    characteristics_output_dir,
    characteristics_output_path,
    write_characteristics_json,
)
from plotting_utility import remove_plot_outputs

OUTPUT_FILENAME = "characteristics_maintainability_metrics_results.json"
CHARACTERISTICS_MAINTAINABILITY_METRICS = (
    "CodeSmellDensityDelta",
    "DuplicationDensity",
    "CommentDensity",
    "CCDensity",
    "HVDensity",
    "MI",
)
BASE_WHERE_SQL = "TRUE"
PLOT_OUTPUT_STEMS = characteristic_plot_stems(
    metrics=CHARACTERISTICS_MAINTAINABILITY_METRICS,
    include_domain=True,
) + characteristic_composition_stacked_bar_stems(
    stem_prefix="smells",
    include_domain=True,
) + characteristic_median_ci_dotplot_stems(
    metric="SmellDensity",
    include_domain=True,
)


def plot_output_stems(*, include_domain: bool) -> tuple[str, ...]:
    """Return expected maintainability-characteristics plot stems."""
    return characteristic_plot_stems(
        metrics=CHARACTERISTICS_MAINTAINABILITY_METRICS,
        include_domain=include_domain,
    ) + characteristic_composition_stacked_bar_stems(
        stem_prefix="smells",
        include_domain=include_domain,
    ) + characteristic_median_ci_dotplot_stems(
        metric="SmellDensity",
        include_domain=include_domain,
    )


def _main_json_payload(results: dict[str, Any]) -> dict[str, Any]:
    """Build the small JSON manifest for maintainability characteristics outputs."""
    payload: dict[str, Any] = {
        "metrics": results.get("metrics", []),
        "language_scope": results.get("language_scope", []),
        "popularity_buckets": results.get("popularity_buckets", {}),
        "domain_enabled": bool(results.get("domain_enabled", False)),
        "dimensions": list((results.get("dimensions") or {}).keys()),
    }
    if payload["domain_enabled"]:
        payload["domain_scope"] = results.get("domain_scope", [])
    return payload


def run_characteristics_maintainability_metrics_analysis_from_payload(
    payload: dict[str, Any],
    *,
    analysis_output_dir: Path | str,
    analysis_kind: str = "maintainability",
    include_domain: bool,
    logger: Any | None = None,
    plot_only: bool = False,
) -> dict[str, Any]:
    """Write maintainability characteristics JSON and plots from streaming payloads."""
    characteristic_metric_values = payload.get("characteristic_metric_values", {})
    characteristic_counts_by_cohort = payload.get("characteristic_counts_by_cohort", {})
    results = build_characteristics_results_from_payload(
        characteristic_metric_values,
        metrics=CHARACTERISTICS_MAINTAINABILITY_METRICS,
        include_domain=include_domain,
    )
    output_dir = characteristics_output_dir(analysis_output_dir, analysis_kind)
    remove_plot_outputs(
        output_dir,
        characteristic_heatmap_stems(
            include_domain=True,
            stem_prefix="post_maintainability_",
        ),
    )
    if logger is not None:
        logger.log("writing_maintainability_characteristics_heatmaps")
    write_characteristics_plots(
        None,
        output_dir=output_dir,
        metrics=CHARACTERISTICS_MAINTAINABILITY_METRICS,
        results=results,
        include_domain=include_domain,
        base_where_sql=BASE_WHERE_SQL,
    )
    if logger is not None:
        logger.log("writing_maintainability_characteristics_stacked_bars")
    write_characteristics_composition_stacked_bars_from_payload(
        counts_by_dimension=characteristic_counts_by_cohort,
        output_dir=output_dir,
        stem_prefix="smells",
        y_axis_label="Percentage of Smells (%)",
        count_key="smell_count",
        total_key="total_smell_count",
        include_domain=include_domain,
    )
    if logger is not None:
        logger.log("writing_maintainability_characteristics_dotplots")
    write_characteristics_median_ci_dotplots_from_payload(
        characteristic_metric_values=characteristic_metric_values,
        output_dir=output_dir,
        metric="SmellDensity",
        include_domain=include_domain,
    )
    if logger is not None:
        logger.log("writing_maintainability_characteristics_extended_heatmaps")
    write_extended_characteristics_heatmaps(
        output_dir=output_dir,
        companion_output_dir=characteristics_output_dir(
            analysis_output_dir,
            "refactoring",
        ),
        primary_analysis_group="Maintainability",
        companion_analysis_group="Refactoring",
        include_domain=include_domain,
    )
    if not plot_only:
        write_characteristics_json(
            characteristics_output_path(
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
        "characteristics_maintainability_metrics_pipeline.py is called from "
        "maintainability_analysis_pipeline.py with streaming payloads."
    )


if __name__ == "__main__":
    main()
