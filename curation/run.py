"""CLI entry point for the single-pass curation stage.

This module intentionally stays thin: it translates command-line flags into one
`CurationSettings` object, applies those settings to the environment expected
by lower-level modules, then hands control to `run_curation`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extraction.config.agent_config import AGENT_RULES
from curation.config.settings import CurationSettings, SUPPORTED_INPUT_FORMATS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one curation run."""
    # Keep choices tied to AGENT_RULES so a new extraction agent becomes a
    # curation cohort without duplicating names in the CLI.
    cohorts = ["agentic", "human", *sorted(AGENT_RULES)]
    parser = argparse.ArgumentParser(
        description=(
            "Run curation from local extraction parquet to local curated outputs. "
            "Metrics use Multimetric, custom duplicated-lines density, and "
            "configured code-smell tools."
        )
    )
    parser.add_argument(
        "--cohort",
        choices=cohorts,
        help="Cohort to curate: human, agentic, or one configured agent.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        help=(
            "Local parquet root to scan. Pass multiple times for multiple roots; "
            "otherwise CURATION_LOCAL_DIRECTORIES or the default is used."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=sorted(SUPPORTED_INPUT_FORMATS),
        help="Local parquet layout to read.",
    )
    parser.add_argument(
        "--output-dir",
        help="Local output root for samples, snapshots, aggregates, and metadata.",
    )
    parser.add_argument(
        "--target-prs",
        type=int,
        help="Number of PRs to sample for the main curation set.",
    )
    parser.add_argument(
        "--longitudinal-prs",
        type=int,
        help="Number of sampled PRs to hydrate with future snapshots.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        # None means "not provided"; CurationSettings then applies env/default
        # precedence just like the other options.
        help="Skip PRs already recorded in processing progress.",
    )
    parser.add_argument(
        "--sample-history-dir",
        help="Optional directory of prior sampled PR identifiers to exclude.",
    )
    parser.add_argument(
        "--delete-snapshots-after-processing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Delete hydrated source snapshots after each PR is processed.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve settings and run the curation pipeline."""
    args = parse_args()
    settings = CurationSettings.from_overrides(
        cohort=args.cohort,
        input_dirs=args.input_dirs,
        input_format=args.input_format,
        output_dir=args.output_dir,
        target_prs=args.target_prs,
        longitudinal_prs=args.longitudinal_prs,
        resume=args.resume,
        sample_history_dir=args.sample_history_dir,
        delete_snapshots_after_processing=args.delete_snapshots_after_processing,
    )
    settings.apply_to_environment()

    # Delay importing the heavy pipeline until settings are written into env.
    # Some lower-level modules still expose config constants at import time.
    from curation.pipeline.curation_pipeline import run_curation

    print(
        "[curation.run] Starting curation "
        f"cohort={settings.cohort} input_format={settings.input_format} "
        f"target_prs={settings.target_prs} longitudinal_prs={settings.longitudinal_prs}"
    )
    run_curation(settings)


if __name__ == "__main__":
    main()
