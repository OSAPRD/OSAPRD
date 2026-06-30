"""Single command line entrypoint for post-processing analysis.

The package keeps three independent analysis pipelines:

1. ``dataset``: dataset counts, balance checks, and topic/classification joins.
2. ``refactoring``: refactoring prevalence, characteristics, and longitudinal
   persistence analyses.
3. ``maintainability``: maintainability/code-smell metrics, characteristics,
   longitudinal trends, and optional Multimetric detail plots.

Use ``all`` to run the three pipelines in that order while preserving their
separate output directories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


ANALYSIS_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ANALYSIS_DIR / "config"
PIPELINE_DIR = ANALYSIS_DIR / "pipelines"
PLOTTER_DIR = ANALYSIS_DIR / "plotters"
UTILITY_DIR = ANALYSIS_DIR / "utility"

for candidate in (CONFIG_DIR, PIPELINE_DIR, PLOTTER_DIR, UTILITY_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from settings import (  # noqa: E402
    ANALYSIS_PIPELINES,
    MANTYLA_COUNT_SOURCES,
    MULTIMETRIC_SOURCES,
    MURPHY_HILL_COUNT_SOURCES,
    PIPELINE_ALL,
    PIPELINE_DATASET,
    PIPELINE_MAINTAINABILITY,
    PIPELINE_REFACTORING,
    AnalysisSettings,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI for reproducible analysis runs."""
    parser = argparse.ArgumentParser(
        description="Run local post-processing analysis pipelines.",
    )
    parser.add_argument(
        "pipeline",
        nargs="?",
        choices=ANALYSIS_PIPELINES,
        default=None,
        help="Pipeline to run. Use 'all' to run dataset, refactoring, and maintainability.",
    )
    parser.add_argument(
        "--curation-data-dir",
        type=Path,
        default=None,
        help="Root containing cohort curation parquet outputs.",
    )
    parser.add_argument(
        "--topic-classification-output-dir",
        type=Path,
        default=None,
        help="Root containing topic-classification output runs.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=None,
        help="Directory where analysis JSON and plots are written.",
    )
    parser.add_argument(
        "--excluded-agents",
        default=None,
        help="Comma-separated agent cohorts excluded from analysis.",
    )
    parser.add_argument(
        "--murphy-hill-count-source",
        choices=MURPHY_HILL_COUNT_SOURCES,
        default=None,
        help="Source for Murphy-Hill refactoring taxonomy counts.",
    )
    parser.add_argument(
        "--mantyla-count-source",
        choices=MANTYLA_COUNT_SOURCES,
        default=None,
        help="Source for Mantyla code-smell category counts.",
    )
    parser.add_argument(
        "--maintainability-require-refops",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Analyze maintainability only for PRs with mined refactoring operations.",
    )
    parser.add_argument(
        "--plot-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rewrite plots from compact payloads without extra plot-data exports.",
    )
    parser.add_argument(
        "--multimetric-source",
        choices=MULTIMETRIC_SOURCES,
        default=None,
        help=(
            "Multimetric detail source: input curated parquet, external legacy "
            "snapshot parquet folder, auto fallback, or off."
        ),
    )
    parser.add_argument(
        "--multimetric-output-dir",
        type=Path,
        default=None,
        help="Legacy external folder containing multimetric_snapshot_metrics.parquet files.",
    )
    return parser


def _load_pipeline_functions() -> dict[str, Callable[..., dict[str, Any]]]:
    """Import pipeline functions after settings have populated environment."""
    from data_analysis_pipeline import run_data_analysis_pipeline
    from maintainability_analysis_pipeline import run_maintainability_analysis_pipeline
    from refactoring_analysis_pipeline import run_refactoring_analysis_pipeline

    return {
        PIPELINE_DATASET: run_data_analysis_pipeline,
        PIPELINE_REFACTORING: run_refactoring_analysis_pipeline,
        PIPELINE_MAINTAINABILITY: run_maintainability_analysis_pipeline,
    }


def _pipeline_order(selected: str) -> tuple[str, ...]:
    """Return the concrete pipeline order for a selected command."""
    if selected == PIPELINE_ALL:
        return (
            PIPELINE_DATASET,
            PIPELINE_REFACTORING,
            PIPELINE_MAINTAINABILITY,
        )
    return (selected,)


def _run_pipeline(
    name: str,
    func: Callable[..., dict[str, Any]],
    settings: AnalysisSettings,
) -> dict[str, Any]:
    """Run one pipeline with the shared settings object."""
    common = {
        "curation_data_dir": settings.curation_data_dir,
        "analysis_output_dir": settings.analysis_output_dir,
        "excluded_agents": settings.excluded_agents,
    }
    if name == PIPELINE_DATASET:
        return func(
            **common,
            topic_classification_output_dir=settings.topic_classification_output_dir,
        )
    if name == PIPELINE_REFACTORING:
        return func(
            **common,
            topic_classification_output_dir=settings.topic_classification_output_dir,
            murphy_hill_count_source=settings.murphy_hill_count_source,
        )
    if name == PIPELINE_MAINTAINABILITY:
        return func(
            **common,
            topic_classification_output_dir=settings.topic_classification_output_dir,
            mantyla_count_source=settings.mantyla_count_source,
            require_refops=settings.maintainability_require_refops,
            multimetric_output_dir=settings.multimetric_output_dir,
            multimetric_source=settings.multimetric_source,
        )
    raise ValueError(f"Unknown analysis pipeline: {name}")


def run(settings: AnalysisSettings) -> dict[str, dict[str, Any]]:
    """Run the requested analysis pipeline or pipeline set."""
    settings.apply_to_environment()
    pipeline_functions = _load_pipeline_functions()
    outputs: dict[str, dict[str, Any]] = {}
    for pipeline in _pipeline_order(settings.pipeline):
        print(f"[post-processing/analysis] Running {pipeline} pipeline")
        outputs[pipeline] = _run_pipeline(
            pipeline,
            pipeline_functions[pipeline],
            settings,
        )
        print(f"[post-processing/analysis] Completed {pipeline} pipeline")
    return outputs


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run analysis."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = AnalysisSettings.from_cli(vars(args))
    run(settings)
    print(
        "[post-processing/analysis] Wrote analysis outputs to "
        f"{settings.analysis_output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
