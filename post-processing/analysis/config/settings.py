"""Typed settings for the post-processing analysis stage.

The analysis package is intentionally file-system only: it reads local curation
parquet and optional topic-classification outputs, then writes local JSON and
plot artifacts. Settings are resolved once by ``analysis/run.py`` using the same
precedence used by extraction and curation: CLI values override environment
variables, and environment variables override defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CURATION_DATA_DIR = ANALYSIS_DIR / "input" / "curation-data"
DEFAULT_TOPIC_CLASSIFICATION_OUTPUT_DIR = (
    ANALYSIS_DIR / "input" / "topic-classification"
)
DEFAULT_ANALYSIS_OUTPUT_DIR = ANALYSIS_DIR / "output"
DEFAULT_EXCLUDED_AGENTS = ("openhands", "junie", "codegen", "cosine")

PIPELINE_DATASET = "dataset"
PIPELINE_REFACTORING = "refactoring"
PIPELINE_MAINTAINABILITY = "maintainability"
PIPELINE_ALL = "all"
ANALYSIS_PIPELINES = (
    PIPELINE_DATASET,
    PIPELINE_REFACTORING,
    PIPELINE_MAINTAINABILITY,
    PIPELINE_ALL,
)

MULTIMETRIC_SOURCE_AUTO = "auto"
MULTIMETRIC_SOURCE_INPUT = "input"
MULTIMETRIC_SOURCE_EXTERNAL = "external"
MULTIMETRIC_SOURCE_OFF = "off"
MULTIMETRIC_SOURCES = (
    MULTIMETRIC_SOURCE_AUTO,
    MULTIMETRIC_SOURCE_INPUT,
    MULTIMETRIC_SOURCE_EXTERNAL,
    MULTIMETRIC_SOURCE_OFF,
)

MURPHY_HILL_COUNT_SOURCE_STORED = "stored"
MURPHY_HILL_COUNT_SOURCE_TAXONOMY = "taxonomy"
MURPHY_HILL_COUNT_SOURCES = (
    MURPHY_HILL_COUNT_SOURCE_STORED,
    MURPHY_HILL_COUNT_SOURCE_TAXONOMY,
)

MANTYLA_COUNT_SOURCE_STORED = "stored"
MANTYLA_COUNT_SOURCE_TAXONOMY = "taxonomy"
MANTYLA_COUNT_SOURCES = (
    MANTYLA_COUNT_SOURCE_STORED,
    MANTYLA_COUNT_SOURCE_TAXONOMY,
)


def _env_text(name: str, default: str | None = None) -> str | None:
    """Return a stripped environment value, treating blanks as missing."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_path(name: str, default: Path | None = None) -> Path | None:
    """Return an expanded ``Path`` from an environment variable."""
    value = _env_text(name)
    return Path(value).expanduser() if value else default


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with explicit validation."""
    value = _env_text(name)
    if value is None:
        return bool(default)
    return _parse_bool(value, name=name)


def _parse_bool(value: Any, *, name: str) -> bool:
    """Parse common truthy/falsy values for CLI and env settings."""
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def _parse_agent_list(value: str | None) -> tuple[str, ...]:
    """Return normalized, de-duplicated agent labels from CSV text."""
    if value is None:
        return DEFAULT_EXCLUDED_AGENTS
    agents: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        agent = part.strip().casefold()
        if not agent or agent in seen:
            continue
        seen.add(agent)
        agents.append(agent)
    return tuple(agents)


def _parse_choice(value: str | None, *, name: str, allowed: tuple[str, ...]) -> str:
    """Normalize and validate a string setting against an allow-list."""
    normalized = str(value or "").strip().casefold()
    if not normalized:
        normalized = allowed[0]
    if normalized not in allowed:
        joined = ", ".join(allowed)
        raise ValueError(f"{name} must be one of: {joined}")
    return normalized


@dataclass(frozen=True)
class AnalysisSettings:
    """Resolved settings shared by all three analysis pipelines."""

    pipeline: str = PIPELINE_ALL
    curation_data_dir: Path = DEFAULT_CURATION_DATA_DIR
    topic_classification_output_dir: Path = DEFAULT_TOPIC_CLASSIFICATION_OUTPUT_DIR
    analysis_output_dir: Path = DEFAULT_ANALYSIS_OUTPUT_DIR
    excluded_agents: tuple[str, ...] = DEFAULT_EXCLUDED_AGENTS
    murphy_hill_count_source: str = MURPHY_HILL_COUNT_SOURCE_TAXONOMY
    mantyla_count_source: str = MANTYLA_COUNT_SOURCE_TAXONOMY
    maintainability_require_refops: bool = False
    plot_mode: bool = False
    multimetric_source: str = MULTIMETRIC_SOURCE_AUTO
    multimetric_output_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "AnalysisSettings":
        """Resolve defaults from ``POST_PROCESSING_ANALYSIS_*`` variables."""
        return cls(
            pipeline=_parse_choice(
                _env_text("POST_PROCESSING_ANALYSIS_PIPELINE", PIPELINE_ALL),
                name="POST_PROCESSING_ANALYSIS_PIPELINE",
                allowed=ANALYSIS_PIPELINES,
            ),
            curation_data_dir=_env_path(
                "POST_PROCESSING_ANALYSIS_CURATION_DATA_DIR",
                DEFAULT_CURATION_DATA_DIR,
            )
            or DEFAULT_CURATION_DATA_DIR,
            topic_classification_output_dir=_env_path(
                "POST_PROCESSING_ANALYSIS_TOPIC_CLASSIFICATION_OUTPUT_DIR",
                DEFAULT_TOPIC_CLASSIFICATION_OUTPUT_DIR,
            )
            or DEFAULT_TOPIC_CLASSIFICATION_OUTPUT_DIR,
            analysis_output_dir=_env_path(
                "POST_PROCESSING_ANALYSIS_OUTPUT_DIR",
                DEFAULT_ANALYSIS_OUTPUT_DIR,
            )
            or DEFAULT_ANALYSIS_OUTPUT_DIR,
            excluded_agents=_parse_agent_list(
                _env_text(
                    "POST_PROCESSING_ANALYSIS_EXCLUDED_AGENTS",
                    ",".join(DEFAULT_EXCLUDED_AGENTS),
                )
            ),
            murphy_hill_count_source=_parse_choice(
                _env_text(
                    "POST_PROCESSING_ANALYSIS_MURPHY_HILL_COUNT_SOURCE",
                    MURPHY_HILL_COUNT_SOURCE_TAXONOMY,
                ),
                name="POST_PROCESSING_ANALYSIS_MURPHY_HILL_COUNT_SOURCE",
                allowed=MURPHY_HILL_COUNT_SOURCES,
            ),
            mantyla_count_source=_parse_choice(
                _env_text(
                    "POST_PROCESSING_ANALYSIS_MANTYLA_COUNT_SOURCE",
                    MANTYLA_COUNT_SOURCE_TAXONOMY,
                ),
                name="POST_PROCESSING_ANALYSIS_MANTYLA_COUNT_SOURCE",
                allowed=MANTYLA_COUNT_SOURCES,
            ),
            maintainability_require_refops=_env_bool(
                "POST_PROCESSING_ANALYSIS_MAINTAINABILITY_REQUIRE_REFOPS",
                False,
            ),
            plot_mode=_env_bool("POST_PROCESSING_ANALYSIS_PLOT_MODE", False),
            multimetric_source=_parse_choice(
                _env_text(
                    "POST_PROCESSING_ANALYSIS_MULTIMETRIC_SOURCE",
                    MULTIMETRIC_SOURCE_AUTO,
                ),
                name="POST_PROCESSING_ANALYSIS_MULTIMETRIC_SOURCE",
                allowed=MULTIMETRIC_SOURCES,
            ),
            multimetric_output_dir=_env_path(
                "POST_PROCESSING_ANALYSIS_MULTIMETRIC_OUTPUT_DIR"
            ),
        )

    @classmethod
    def from_cli(cls, args: Mapping[str, Any]) -> "AnalysisSettings":
        """Resolve settings from environment and sparse CLI overrides."""
        base = cls.from_env()
        return cls(
            pipeline=_parse_choice(
                args.get("pipeline") or base.pipeline,
                name="pipeline",
                allowed=ANALYSIS_PIPELINES,
            ),
            curation_data_dir=Path(
                args.get("curation_data_dir") or base.curation_data_dir
            ).expanduser(),
            topic_classification_output_dir=Path(
                args.get("topic_classification_output_dir")
                or base.topic_classification_output_dir
            ).expanduser(),
            analysis_output_dir=Path(
                args.get("analysis_output_dir") or base.analysis_output_dir
            ).expanduser(),
            excluded_agents=(
                _parse_agent_list(args.get("excluded_agents"))
                if args.get("excluded_agents") is not None
                else base.excluded_agents
            ),
            murphy_hill_count_source=_parse_choice(
                args.get("murphy_hill_count_source")
                or base.murphy_hill_count_source,
                name="murphy_hill_count_source",
                allowed=MURPHY_HILL_COUNT_SOURCES,
            ),
            mantyla_count_source=_parse_choice(
                args.get("mantyla_count_source") or base.mantyla_count_source,
                name="mantyla_count_source",
                allowed=MANTYLA_COUNT_SOURCES,
            ),
            maintainability_require_refops=(
                base.maintainability_require_refops
                if args.get("maintainability_require_refops") is None
                else bool(args.get("maintainability_require_refops"))
            ),
            plot_mode=(
                base.plot_mode
                if args.get("plot_mode") is None
                else bool(args.get("plot_mode"))
            ),
            multimetric_source=_parse_choice(
                args.get("multimetric_source") or base.multimetric_source,
                name="multimetric_source",
                allowed=MULTIMETRIC_SOURCES,
            ),
            multimetric_output_dir=(
                Path(args.get("multimetric_output_dir")).expanduser()
                if args.get("multimetric_output_dir") is not None
                else base.multimetric_output_dir
            ),
        )

    def apply_to_environment(self) -> None:
        """Populate env vars consumed by existing pipeline modules."""
        os.environ["POST_PROCESSING_ANALYSIS_PIPELINE"] = self.pipeline
        os.environ["POST_PROCESSING_ANALYSIS_CURATION_DATA_DIR"] = str(
            self.curation_data_dir
        )
        os.environ["POST_PROCESSING_ANALYSIS_TOPIC_CLASSIFICATION_OUTPUT_DIR"] = str(
            self.topic_classification_output_dir
        )
        os.environ["POST_PROCESSING_ANALYSIS_OUTPUT_DIR"] = str(
            self.analysis_output_dir
        )
        os.environ["POST_PROCESSING_ANALYSIS_EXCLUDED_AGENTS"] = ",".join(
            self.excluded_agents
        )
        os.environ["POST_PROCESSING_ANALYSIS_MURPHY_HILL_COUNT_SOURCE"] = (
            self.murphy_hill_count_source
        )
        os.environ["POST_PROCESSING_ANALYSIS_MANTYLA_COUNT_SOURCE"] = (
            self.mantyla_count_source
        )
        os.environ["POST_PROCESSING_ANALYSIS_MAINTAINABILITY_REQUIRE_REFOPS"] = (
            "1" if self.maintainability_require_refops else "0"
        )
        os.environ["POST_PROCESSING_ANALYSIS_PLOT_MODE"] = (
            "1" if self.plot_mode else "0"
        )
        os.environ["POST_PROCESSING_ANALYSIS_MULTIMETRIC_SOURCE"] = (
            self.multimetric_source
        )
        if self.multimetric_output_dir is not None:
            os.environ["POST_PROCESSING_ANALYSIS_MULTIMETRIC_OUTPUT_DIR"] = str(
                self.multimetric_output_dir
            )
