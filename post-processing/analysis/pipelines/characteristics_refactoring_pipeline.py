"""Characteristics analysis for refactoring metrics.

This companion pipeline is called by ``refactoring_analysis_pipeline`` after the
streaming accumulator has built compact characteristic payloads. It writes
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
from refactoring_analysis_utility import REFACTORING_DISTRIBUTION_METRICS


OUTPUT_FILENAME = "characteristics_refactoring_results.json"
CHARACTERISTICS_REFACTORING_METRICS = REFACTORING_DISTRIBUTION_METRICS
BASE_WHERE_SQL = '"RefCount" > 0'
PLOT_OUTPUT_STEMS = characteristic_plot_stems(
    metrics=CHARACTERISTICS_REFACTORING_METRICS,
    include_domain=True,
) + characteristic_composition_stacked_bar_stems(
    stem_prefix="refops",
    include_domain=True,
) + characteristic_median_ci_dotplot_stems(
    metric="RefDensity",
    include_domain=True,
)


def plot_output_stems(*, include_domain: bool) -> tuple[str, ...]:
    """Return expected refactoring-characteristics plot stems."""
    return characteristic_plot_stems(
        metrics=CHARACTERISTICS_REFACTORING_METRICS,
        include_domain=include_domain,
    ) + characteristic_composition_stacked_bar_stems(
        stem_prefix="refops",
        include_domain=include_domain,
    ) + characteristic_median_ci_dotplot_stems(
        metric="RefDensity",
        include_domain=include_domain,
    )


def _main_json_payload(results: dict[str, Any]) -> dict[str, Any]:
    """Build the small JSON manifest for refactoring characteristics outputs."""
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


def run_characteristics_refactoring_analysis_from_payload(
    payload: dict[str, Any],
    *,
    analysis_output_dir: Path | str,
    include_domain: bool,
    logger: Any | None = None,
    plot_only: bool = False,
) -> dict[str, Any]:
    """Write refactoring characteristics JSON and plots from streaming payloads."""
    characteristic_metric_values = payload.get("characteristic_metric_values", {})
    characteristic_counts_by_cohort = payload.get("characteristic_counts_by_cohort", {})
    results = build_characteristics_results_from_payload(
        characteristic_metric_values,
        metrics=CHARACTERISTICS_REFACTORING_METRICS,
        include_domain=include_domain,
    )
    output_dir = characteristics_output_dir(analysis_output_dir, "refactoring")
    if logger is not None:
        logger.log("writing_refactoring_characteristics_heatmaps")
    write_characteristics_plots(
        None,
        output_dir=output_dir,
        metrics=CHARACTERISTICS_REFACTORING_METRICS,
        results=results,
        include_domain=include_domain,
        base_where_sql=BASE_WHERE_SQL,
    )
    if logger is not None:
        logger.log("writing_refactoring_characteristics_stacked_bars")
    write_characteristics_composition_stacked_bars_from_payload(
        counts_by_dimension=characteristic_counts_by_cohort,
        output_dir=output_dir,
        stem_prefix="refops",
        y_axis_label="Percentage of RefOps (%)",
        count_key="refop_count",
        total_key="total_refop_count",
        include_domain=include_domain,
    )
    if logger is not None:
        logger.log("writing_refactoring_characteristics_dotplots")
    write_characteristics_median_ci_dotplots_from_payload(
        characteristic_metric_values=characteristic_metric_values,
        output_dir=output_dir,
        metric="RefDensity",
        include_domain=include_domain,
    )
    if logger is not None:
        logger.log("writing_refactoring_characteristics_extended_heatmaps")
    write_extended_characteristics_heatmaps(
        output_dir=output_dir,
        companion_output_dir=characteristics_output_dir(
            analysis_output_dir,
            "maintainability",
        ),
        primary_analysis_group="Refactoring",
        companion_analysis_group="Maintainability",
        include_domain=include_domain,
    )
    if not plot_only:
        write_characteristics_json(
            characteristics_output_path(
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
        "characteristics_refactoring_pipeline.py is called from "
        "refactoring_analysis_pipeline.py with streaming payloads."
    )


if __name__ == "__main__":
    main()
