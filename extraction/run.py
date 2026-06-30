"""
CLI entry point for the live-GitHub extraction stage.

This module intentionally stays thin: it translates command-line flags into one
`ExtractionSettings` object, then hands control to `ScraperManager`. Discovery,
enrichment, checkpointing, and storage are documented in their own modules so
the reproducibility entrypoint remains easy to audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support both recommended module execution (`python -m extraction.run`) and
# direct script execution (`python extraction/run.py`) from a local checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extraction.config.agent_config import AGENT_RULES
from extraction.config.settings import ExtractionSettings
from extraction.managers.scraper_manager import ScraperManager


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for one extraction run.

    The CLI exposes scrape targets, not data-source modes. Extraction always
    uses live GitHub scraping; `--target` selects whether to scrape human PRs,
    all configured agent PRs, or a single configured agent.
    """
    # Build the choices from AGENT_RULES so adding a new supported agent in
    # configuration automatically exposes it as `--target <agent>`.
    targets = ["agentic", "human", *sorted(AGENT_RULES)]
    parser = argparse.ArgumentParser(
        description="Run live GitHub PR discovery + enrichment and write local parquet outputs."
    )
    parser.add_argument(
        "--target",
        choices=targets,
        # Defaults are resolved by ExtractionSettings so CLI, environment, and
        # code paths all share the same precedence rules.
        help="Scrape target: human, agentic for all agents, or one configured agent.",
    )
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Discovery pagination cap per query/window.",
    )
    parser.add_argument(
        "--output-dir",
        help="Local output root for parquet, checkpoints, and manifests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of enriched PRs per parquet batch.",
    )
    parser.add_argument(
        "--use-graphql-enrichment",
        action=argparse.BooleanOptionalAction,
        default=None,
        # Keep the default as None here so settings can distinguish "flag not
        # provided" from an explicit --use/--no-use override.
        help="Enable or disable GraphQL enrichment where supported.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Resolve settings and run discovery plus enrichment.

    `ExtractionSettings.from_overrides` applies CLI > environment > default
    precedence and validates the target before any GitHub requests are made.
    """
    args = parse_args()
    settings = ExtractionSettings.from_overrides(
        target=args.target,
        start_date=args.start,
        end_date=args.end,
        max_pages=args.max_pages,
        local_output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_graphql_enrichment=args.use_graphql_enrichment,
    )
    print(f"[run] Starting scraper manager for target={settings.target}...")
    # The manager owns all stage behavior after settings are resolved: discovery
    # target expansion, enrichment mode, local storage, checkpoints, and the run
    # manifest.
    mgr = ScraperManager(settings=settings)
    print("[run] Running pipeline...")
    prs = mgr.run()
    print(f"[run] Completed. Enriched PRs: {len(prs)}")


if __name__ == "__main__":
    main()
