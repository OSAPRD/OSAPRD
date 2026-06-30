"""Resolved runtime settings for one live GitHub extraction run.

`ExtractionSettings` is the single object passed through orchestration,
discovery, enrichment, checkpoints, and storage. Values are resolved in this
order:

1. CLI overrides from `extraction.run`.
2. Environment variables.
3. Defaults from the relevant config modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from extraction.config.agent_config import AGENT_RULES
from extraction.config.storage_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCAL_OUTPUT_DIR,
    load_batch_size,
    load_local_output_dir,
)
from extraction.config.tokens_config import load_github_tokens

# Defaults
DEFAULT_TARGET = "agentic"
DEFAULT_START_DATE = "2025-05-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_MAX_PAGES = 10
DEFAULT_ENRICHMENT_USE_GRAPHQL = True
SCHEMA_VERSION = "extraction_full_pull_request_v1"

# Reserved non-specific-agent targets. Any other valid target must be an AGENT_RULES key.
RESERVED_TARGETS = {"agentic", "human"}


def _parse_bool(value: str | None, *, default: bool) -> bool:
    normalized = (value or "").strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_int(value: str | None, *, default: int, name: str) -> int:
    normalized = (value or "").strip()
    if not normalized:
        return int(default)
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def _env_first(*names: str) -> str | None:
    """Return the first non-empty environment variable from `names`."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return None


def validate_target(target: str) -> str:
    """Validate and normalize an extraction target.

    Valid targets are:
    - `agentic`: all configured agents.
    - `human`: human sampling.
    - one key from `AGENT_RULES`: a single-agent scrape.
    """
    normalized = (target or "").strip().lower()
    if not normalized:
        raise ValueError("Extraction target is required.")
    if normalized in RESERVED_TARGETS or normalized in AGENT_RULES:
        return normalized
    allowed = ", ".join(["agentic", "human", *sorted(AGENT_RULES)])
    raise ValueError(f"Unsupported extraction target {target!r}. Expected one of: {allowed}.")


@dataclass(frozen=True)
class ExtractionSettings:
    """Resolved runtime settings for one extraction run.

    Date fields are inclusive `YYYY-MM-DD` values. Discovery expands them to
    full UTC-day timestamps. `local_output_dir` contains data, checkpoints,
    duplicate manifests, and `extraction_run_manifest.json`.
    """

    target: str = DEFAULT_TARGET
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    max_pages: int = DEFAULT_MAX_PAGES
    local_output_dir: Path = DEFAULT_LOCAL_OUTPUT_DIR
    batch_size: int = DEFAULT_BATCH_SIZE
    use_graphql_enrichment: bool = DEFAULT_ENRICHMENT_USE_GRAPHQL
    github_tokens: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "ExtractionSettings":
        """Build settings from environment variables.

        Supported aliases are kept explicit so reproduction scripts can use
        short conventional names while the pipeline uses extraction-scoped
        names. Extraction does not read `SCRAPE_MODE`; use `TARGET` or
        `EXTRACTION_TARGET`.
        """
        target = validate_target(_env_first("EXTRACTION_TARGET", "TARGET") or DEFAULT_TARGET)
        return cls(
            target=target,
            start_date=_env_first("EXTRACTION_START_DATE", "START_DATE") or DEFAULT_START_DATE,
            end_date=_env_first("EXTRACTION_END_DATE", "END_DATE") or DEFAULT_END_DATE,
            max_pages=_parse_int(
                _env_first("EXTRACTION_MAX_PAGES", "MAX_PAGES"),
                default=DEFAULT_MAX_PAGES,
                name="EXTRACTION_MAX_PAGES",
            ),
            local_output_dir=load_local_output_dir(),
            batch_size=load_batch_size(),
            use_graphql_enrichment=_parse_bool(
                _env_first("EXTRACTION_ENRICHMENT_USE_GRAPHQL", "ENRICHMENT_USE_GRAPHQL"),
                default=DEFAULT_ENRICHMENT_USE_GRAPHQL,
            ),
            github_tokens=tuple(load_github_tokens()),
        )

    @classmethod
    def from_overrides(
        cls,
        *,
        target: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int | None = None,
        local_output_dir: Path | str | None = None,
        batch_size: int | None = None,
        use_graphql_enrichment: bool | None = None,
        github_tokens: Sequence[str] | None = None,
    ) -> "ExtractionSettings":
        """Build settings using environment defaults plus explicit overrides.

        This is the CLI boundary: pass parsed CLI values here and leave `None`
        for options that should fall back to environment/defaults.
        """
        base = cls.from_env()
        return cls(
            target=validate_target(target or base.target),
            start_date=start_date or base.start_date,
            end_date=end_date or base.end_date,
            max_pages=base.max_pages if max_pages is None else int(max_pages),
            local_output_dir=(
                base.local_output_dir if local_output_dir is None else Path(local_output_dir)
            ),
            batch_size=base.batch_size if batch_size is None else int(batch_size),
            use_graphql_enrichment=(
                base.use_graphql_enrichment
                if use_graphql_enrichment is None
                else bool(use_graphql_enrichment)
            ),
            github_tokens=tuple(github_tokens) if github_tokens is not None else base.github_tokens,
        )

    @property
    def is_human(self) -> bool:
        """Return True when this run uses human sampling."""
        return self.target == "human"

    @property
    def is_all_agents(self) -> bool:
        """Return True when this run scrapes all configured agents."""
        return self.target == "agentic"

    @property
    def resolved_agents(self) -> tuple[str, ...]:
        """Return the concrete agent list represented by the target."""
        if self.is_human:
            return ()
        if self.is_all_agents:
            return tuple(AGENT_RULES.keys())
        return (self.target,)
