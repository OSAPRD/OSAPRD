"""Resolved runtime settings for one curation run.

Values are resolved in this order:

1. Explicit CLI overrides from :mod:`curation.run`.
2. Environment variables.
3. Defaults in this module.

The CLI applies the resolved settings to environment variables before importing
the heavier pipeline modules. That keeps legacy lower-level config imports
deterministic while giving the top-level run one auditable settings object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from extraction.config.agent_config import AGENT_RULES

DEFAULT_COHORT = "agentic"
DEFAULT_INPUT_FORMAT = "extractionpullrequests"
DEFAULT_INPUT_DIRS = (Path("/data/input"),)
DEFAULT_OUTPUT_DIR = Path("/data/output")
DEFAULT_TARGET_PRS = 50_000
DEFAULT_LONGITUDINAL_PRS = 5_000
DEFAULT_RESUME = True
DEFAULT_DELETE_SNAPSHOTS_AFTER_PROCESSING = False

SUPPORTED_INPUT_FORMATS = {
    "extractionpullrequests",
    "fullpullrequests_sharded",
    "curation_processed",
}
RESERVED_COHORTS = {"agentic", "human", "humans"}


def _env_first(*names: str) -> str | None:
    """Return the first non-empty environment variable from ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return None


def _parse_bool(value: str | None, *, default: bool) -> bool:
    """Parse a common boolean string with a clear validation error."""
    normalized = (value or "").strip().lower()
    if not normalized:
        return bool(default)
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_int(value: str | None, *, default: int, name: str) -> int:
    """Parse a non-negative integer from an environment or CLI value."""
    normalized = (value or "").strip()
    if not normalized:
        return int(default)
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}.")
    return parsed


def _split_paths(value: str | None) -> tuple[Path, ...]:
    """Split an OS-path-list value into paths."""
    if not value or not value.strip():
        return DEFAULT_INPUT_DIRS
    paths = [Path(part.strip()) for part in value.split(os.pathsep) if part.strip()]
    return tuple(paths) if paths else DEFAULT_INPUT_DIRS


def _coerce_input_dirs(value: Iterable[str | Path] | None) -> tuple[Path, ...] | None:
    """Normalize explicit CLI input directories without applying defaults."""
    if value is None:
        return None
    paths = [Path(part) for part in value if str(part).strip()]
    return tuple(paths)


def validate_cohort(cohort: str) -> str:
    """Validate and normalize a curation cohort.

    Supported values are ``human``/``humans``, ``agentic``, and every configured
    agent key from extraction's ``AGENT_RULES``.
    """
    normalized = (cohort or "").strip().lower()
    if not normalized:
        raise ValueError("Curation cohort is required.")
    if normalized in RESERVED_COHORTS or normalized in AGENT_RULES:
        return "human" if normalized == "humans" else normalized
    allowed = ", ".join(["agentic", "human", *sorted(AGENT_RULES)])
    raise ValueError(f"Unsupported curation cohort {cohort!r}. Expected one of: {allowed}.")


def validate_input_format(input_format: str) -> str:
    """Validate and normalize the local parquet input format."""
    normalized = (input_format or "").strip().lower()
    if normalized in SUPPORTED_INPUT_FORMATS:
        return normalized
    allowed = ", ".join(sorted(SUPPORTED_INPUT_FORMATS))
    raise ValueError(f"Unsupported curation input format {input_format!r}. Expected one of: {allowed}.")


def _load_github_tokens() -> tuple[str, ...]:
    """Load GitHub tokens using curation's token precedence."""
    from curation.config.tokens_config import load_github_tokens

    return tuple(load_github_tokens())


@dataclass(frozen=True)
class CurationSettings:
    """Resolved settings for one single-pass curation run."""

    cohort: str = DEFAULT_COHORT
    input_dirs: tuple[Path, ...] = DEFAULT_INPUT_DIRS
    input_format: str = DEFAULT_INPUT_FORMAT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    target_prs: int = DEFAULT_TARGET_PRS
    longitudinal_prs: int = DEFAULT_LONGITUDINAL_PRS
    resume: bool = DEFAULT_RESUME
    sample_history_dir: Path | None = None
    delete_snapshots_after_processing: bool = DEFAULT_DELETE_SNAPSHOTS_AFTER_PROCESSING
    github_tokens: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "CurationSettings":
        """Build settings from environment variables."""
        sample_history = _env_first("CURATION_SAMPLE_HISTORY_DIR", "SAMPLE_HISTORY_DIR")
        return cls(
            cohort=validate_cohort(_env_first("CURATION_COHORT", "COHORT") or DEFAULT_COHORT),
            input_dirs=_split_paths(
                _env_first("CURATION_INPUT_DIRS", "CURATION_INPUT_DIR", "CURATION_LOCAL_DIRECTORIES")
            ),
            input_format=validate_input_format(
                _env_first("CURATION_INPUT_FORMAT", "CURATION_LOCAL_DATA_FORMAT")
                or DEFAULT_INPUT_FORMAT
            ),
            output_dir=Path(_env_first("CURATION_OUTPUT_DIR", "LOCAL_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR),
            target_prs=_parse_int(
                _env_first("CURATION_TARGET_PRS", "TARGET_NO_PRS"),
                default=DEFAULT_TARGET_PRS,
                name="CURATION_TARGET_PRS",
            ),
            longitudinal_prs=_parse_int(
                _env_first("CURATION_LONGITUDINAL_PRS", "LONGITUDINAL_TARGET_NO_PRS"),
                default=DEFAULT_LONGITUDINAL_PRS,
                name="CURATION_LONGITUDINAL_PRS",
            ),
            resume=_parse_bool(
                _env_first("CURATION_RESUME", "RESUME_PROCESSING"),
                default=DEFAULT_RESUME,
            ),
            sample_history_dir=Path(sample_history) if sample_history else None,
            delete_snapshots_after_processing=_parse_bool(
                _env_first(
                    "CURATION_DELETE_SNAPSHOTS_AFTER_PROCESSING",
                    "DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING",
                ),
                default=DEFAULT_DELETE_SNAPSHOTS_AFTER_PROCESSING,
            ),
            github_tokens=_load_github_tokens(),
        )

    @classmethod
    def from_overrides(
        cls,
        *,
        cohort: str | None = None,
        input_dirs: Sequence[str | Path] | None = None,
        input_format: str | None = None,
        output_dir: str | Path | None = None,
        target_prs: int | None = None,
        longitudinal_prs: int | None = None,
        resume: bool | None = None,
        sample_history_dir: str | Path | None = None,
        delete_snapshots_after_processing: bool | None = None,
        github_tokens: Sequence[str] | None = None,
    ) -> "CurationSettings":
        """Build settings from env/defaults plus explicit CLI overrides."""
        base = cls.from_env()
        explicit_input_dirs = _coerce_input_dirs(input_dirs)
        return cls(
            cohort=validate_cohort(cohort or base.cohort),
            input_dirs=explicit_input_dirs if explicit_input_dirs is not None else base.input_dirs,
            input_format=validate_input_format(input_format or base.input_format),
            output_dir=Path(output_dir) if output_dir is not None else base.output_dir,
            target_prs=base.target_prs if target_prs is None else int(target_prs),
            longitudinal_prs=(
                base.longitudinal_prs if longitudinal_prs is None else int(longitudinal_prs)
            ),
            resume=base.resume if resume is None else bool(resume),
            sample_history_dir=(
                Path(sample_history_dir)
                if sample_history_dir is not None
                else base.sample_history_dir
            ),
            delete_snapshots_after_processing=(
                base.delete_snapshots_after_processing
                if delete_snapshots_after_processing is None
                else bool(delete_snapshots_after_processing)
            ),
            github_tokens=tuple(github_tokens) if github_tokens is not None else base.github_tokens,
        )

    def apply_to_environment(self) -> None:
        """Expose settings through env vars consumed by lower-level modules."""
        os.environ["COHORT"] = self.cohort
        os.environ["CURATION_COHORT"] = self.cohort
        os.environ["CURATION_LOCAL_DIRECTORIES"] = os.pathsep.join(
            str(path) for path in self.input_dirs
        )
        os.environ["CURATION_LOCAL_DATA_FORMAT"] = self.input_format
        os.environ["LOCAL_OUTPUT_DIR"] = str(self.output_dir)
        os.environ["TARGET_NO_PRS"] = str(self.target_prs)
        os.environ["LONGITUDINAL_TARGET_NO_PRS"] = str(self.longitudinal_prs)
        os.environ["RESUME_PROCESSING"] = "1" if self.resume else "0"
        os.environ["DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING"] = (
            "1" if self.delete_snapshots_after_processing else "0"
        )
        if self.sample_history_dir is not None:
            os.environ["CURATION_SAMPLE_HISTORY_DIR"] = str(self.sample_history_dir)
        else:
            os.environ.pop("CURATION_SAMPLE_HISTORY_DIR", None)
