"""Typed settings for extraction-data publishing.

Settings are resolved once by ``run.py`` with the same precedence used by the
other pipeline stages: CLI values override environment variables, and
environment variables override Docker-friendly defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from storage_config import (
    DEFAULT_DATA_SUBDIR,
    DEFAULT_MAX_FILES_PER_DIRECTORY,
    DEFAULT_OUTPUT_BATCH_SIZE,
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_SCHEMA_VERSION,
    DEFAULT_STATE_DB_FILENAME,
    DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_DELAY_SECONDS,
    DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_THRESHOLD,
    DEFAULT_UPLOAD_HOURLY_RATE_LIMIT_DELAY_SECONDS,
    DEFAULT_UPLOAD_LARGE_FOLDER_DIRECTORY_COOLDOWN_SECONDS,
    DEFAULT_UPLOAD_LARGE_FOLDER_NUM_WORKERS,
    DEFAULT_UPLOAD_MAX_RETRIES,
    DEFAULT_UPLOAD_RETRY_BASE_SECONDS,
    DEFAULT_UPLOAD_SHORT_TERM_RATE_LIMIT_WINDOW_SECONDS,
    load_output_dir,
    load_source_dirs,
    split_path_list,
)
from tokens_config import load_huggingface_token, redacted_token_state


COMMAND_PREPARE = "prepare"
COMMAND_UPLOAD = "upload"
COMMAND_ALL = "all"
COMMANDS = (COMMAND_PREPARE, COMMAND_UPLOAD, COMMAND_ALL)

# Hugging Face supports multiple repo types, but this stage publishes dataset
# artifacts. The option remains explicit so CLI/env manifests show the target.
DEFAULT_REPO_TYPE = "dataset"


def _env_text(name: str, default: str | None = None) -> str | None:
    """Return a stripped environment value, treating blanks as missing."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    """Parse a positive integer from the environment."""
    value = _env_text(name)
    if value is None:
        return int(default)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _env_float(name: str, default: float) -> float:
    """Parse a non-negative float from the environment."""
    value = _env_text(name)
    if value is None:
        return float(default)
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _parse_bool(value: Any, *, name: str) -> bool:
    """Parse common CLI/env boolean values with explicit errors."""
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""
    value = _env_text(name)
    return bool(default) if value is None else _parse_bool(value, name=name)


def _parse_choice(value: str | None, *, name: str, allowed: tuple[str, ...]) -> str:
    """Normalize and validate a string setting against an allow-list."""
    normalized = str(value or "").strip().casefold()
    if not normalized:
        normalized = allowed[0]
    if normalized not in allowed:
        joined = ", ".join(allowed)
        raise ValueError(f"{name} must be one of: {joined}")
    return normalized


def _coerce_source_dirs(value: Any, fallback: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return source roots from repeated CLI flags or OS-path-separated text."""
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        paths = tuple(Path(part).expanduser() for part in value if str(part).strip())
        return paths or fallback
    parsed = split_path_list(str(value))
    return parsed or fallback


@dataclass(frozen=True)
class UploadExtractionSettings:
    """Resolved settings for one prepare/upload command.

    The dataclass deliberately stores the Hugging Face token because the uploader
    needs it, but ``redacted_manifest_settings`` is the only settings snapshot
    written to disk.
    """

    command: str = COMMAND_ALL
    source_dirs: tuple[Path, ...] = load_source_dirs()
    output_dir: Path = load_output_dir()
    repo_id: str = ""
    repo_type: str = DEFAULT_REPO_TYPE
    hf_token: str = ""
    dry_run: bool = False
    data_subdir: str = DEFAULT_DATA_SUBDIR
    output_batch_size: int = DEFAULT_OUTPUT_BATCH_SIZE
    max_files_per_directory: int = DEFAULT_MAX_FILES_PER_DIRECTORY
    parquet_compression: str = DEFAULT_PARQUET_COMPRESSION
    state_db_filename: str = DEFAULT_STATE_DB_FILENAME
    schema_version: str = DEFAULT_SCHEMA_VERSION
    upload_max_retries: int = DEFAULT_UPLOAD_MAX_RETRIES
    upload_retry_base_seconds: float = DEFAULT_UPLOAD_RETRY_BASE_SECONDS
    upload_short_term_rate_limit_window_seconds: float = (
        DEFAULT_UPLOAD_SHORT_TERM_RATE_LIMIT_WINDOW_SECONDS
    )
    upload_hourly_rate_limit_delay_seconds: float = (
        DEFAULT_UPLOAD_HOURLY_RATE_LIMIT_DELAY_SECONDS
    )
    upload_consecutive_failure_threshold: int = (
        DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_THRESHOLD
    )
    upload_consecutive_failure_delay_seconds: float = (
        DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_DELAY_SECONDS
    )
    upload_large_folder_num_workers: int = DEFAULT_UPLOAD_LARGE_FOLDER_NUM_WORKERS
    upload_large_folder_directory_cooldown_seconds: float = (
        DEFAULT_UPLOAD_LARGE_FOLDER_DIRECTORY_COOLDOWN_SECONDS
    )

    @classmethod
    def from_env(cls) -> "UploadExtractionSettings":
        """Resolve settings from stage-specific environment variables.

        Accepted aliases are intentionally narrow: upload-extraction uses
        ``POST_PROCESSING_UPLOAD_EXTRACTION_*`` for stage settings, plus
        ``HF_DATASET_REPO_ID`` and Hugging Face token variables for upload
        credentials.
        """
        return cls(
            command=_parse_choice(
                _env_text("POST_PROCESSING_UPLOAD_EXTRACTION_COMMAND", COMMAND_ALL),
                name="POST_PROCESSING_UPLOAD_EXTRACTION_COMMAND",
                allowed=COMMANDS,
            ),
            source_dirs=load_source_dirs(),
            output_dir=load_output_dir(),
            repo_id=_env_text("POST_PROCESSING_UPLOAD_EXTRACTION_REPO_ID", "")
            or _env_text("HF_DATASET_REPO_ID", "")
            or "",
            repo_type=_env_text(
                "POST_PROCESSING_UPLOAD_EXTRACTION_REPO_TYPE",
                DEFAULT_REPO_TYPE,
            )
            or DEFAULT_REPO_TYPE,
            hf_token=load_huggingface_token(),
            dry_run=_env_bool("POST_PROCESSING_UPLOAD_EXTRACTION_DRY_RUN", False),
            data_subdir=_env_text(
                "POST_PROCESSING_UPLOAD_EXTRACTION_DATA_SUBDIR",
                DEFAULT_DATA_SUBDIR,
            )
            or DEFAULT_DATA_SUBDIR,
            output_batch_size=_env_int(
                "POST_PROCESSING_UPLOAD_EXTRACTION_OUTPUT_BATCH_SIZE",
                DEFAULT_OUTPUT_BATCH_SIZE,
            ),
            max_files_per_directory=_env_int(
                "POST_PROCESSING_UPLOAD_EXTRACTION_MAX_FILES_PER_DIRECTORY",
                DEFAULT_MAX_FILES_PER_DIRECTORY,
            ),
            parquet_compression=_env_text(
                "POST_PROCESSING_UPLOAD_EXTRACTION_PARQUET_COMPRESSION",
                DEFAULT_PARQUET_COMPRESSION,
            )
            or DEFAULT_PARQUET_COMPRESSION,
            state_db_filename=_env_text(
                "POST_PROCESSING_UPLOAD_EXTRACTION_STATE_DB_FILENAME",
                DEFAULT_STATE_DB_FILENAME,
            )
            or DEFAULT_STATE_DB_FILENAME,
            schema_version=_env_text(
                "POST_PROCESSING_UPLOAD_EXTRACTION_SCHEMA_VERSION",
                DEFAULT_SCHEMA_VERSION,
            )
            or DEFAULT_SCHEMA_VERSION,
            upload_max_retries=_env_int(
                "POST_PROCESSING_UPLOAD_EXTRACTION_UPLOAD_MAX_RETRIES",
                DEFAULT_UPLOAD_MAX_RETRIES,
            ),
            upload_retry_base_seconds=_env_float(
                "POST_PROCESSING_UPLOAD_EXTRACTION_UPLOAD_RETRY_BASE_SECONDS",
                DEFAULT_UPLOAD_RETRY_BASE_SECONDS,
            ),
            upload_short_term_rate_limit_window_seconds=_env_float(
                "POST_PROCESSING_UPLOAD_EXTRACTION_UPLOAD_SHORT_TERM_RATE_LIMIT_WINDOW_SECONDS",
                DEFAULT_UPLOAD_SHORT_TERM_RATE_LIMIT_WINDOW_SECONDS,
            ),
            upload_hourly_rate_limit_delay_seconds=_env_float(
                "POST_PROCESSING_UPLOAD_EXTRACTION_UPLOAD_HOURLY_RATE_LIMIT_DELAY_SECONDS",
                DEFAULT_UPLOAD_HOURLY_RATE_LIMIT_DELAY_SECONDS,
            ),
            upload_consecutive_failure_threshold=_env_int(
                "POST_PROCESSING_UPLOAD_EXTRACTION_UPLOAD_CONSECUTIVE_FAILURE_THRESHOLD",
                DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_THRESHOLD,
            ),
            upload_consecutive_failure_delay_seconds=_env_float(
                "POST_PROCESSING_UPLOAD_EXTRACTION_UPLOAD_CONSECUTIVE_FAILURE_DELAY_SECONDS",
                DEFAULT_UPLOAD_CONSECUTIVE_FAILURE_DELAY_SECONDS,
            ),
            upload_large_folder_num_workers=_env_int(
                "POST_PROCESSING_UPLOAD_EXTRACTION_UPLOAD_LARGE_FOLDER_NUM_WORKERS",
                DEFAULT_UPLOAD_LARGE_FOLDER_NUM_WORKERS,
            ),
            upload_large_folder_directory_cooldown_seconds=_env_float(
                "POST_PROCESSING_UPLOAD_EXTRACTION_UPLOAD_LARGE_FOLDER_DIRECTORY_COOLDOWN_SECONDS",
                DEFAULT_UPLOAD_LARGE_FOLDER_DIRECTORY_COOLDOWN_SECONDS,
            ),
        )

    @classmethod
    def from_cli(cls, args: Mapping[str, Any]) -> "UploadExtractionSettings":
        """Resolve settings from environment plus sparse CLI overrides.

        ``argparse`` passes ``None`` for omitted optional flags. This method
        keeps those environment/default values intact and only replaces values
        the caller explicitly supplied.
        """
        base = cls.from_env()
        return cls(
            command=_parse_choice(
                args.get("command") or base.command,
                name="command",
                allowed=COMMANDS,
            ),
            source_dirs=_coerce_source_dirs(args.get("source_dir"), base.source_dirs),
            output_dir=Path(args.get("output_dir") or base.output_dir).expanduser(),
            repo_id=str(args.get("repo_id") or base.repo_id or "").strip(),
            repo_type=str(args.get("repo_type") or base.repo_type or DEFAULT_REPO_TYPE).strip(),
            hf_token=str(args.get("hf_token") or base.hf_token or "").strip(),
            dry_run=(
                base.dry_run
                if args.get("dry_run") is None
                else bool(args.get("dry_run"))
            ),
            data_subdir=str(args.get("data_subdir") or base.data_subdir).strip(),
            output_batch_size=int(args.get("output_batch_size") or base.output_batch_size),
            max_files_per_directory=int(
                args.get("max_files_per_directory") or base.max_files_per_directory
            ),
            parquet_compression=str(
                args.get("parquet_compression") or base.parquet_compression
            ).strip(),
            state_db_filename=str(
                args.get("state_db_filename") or base.state_db_filename
            ).strip(),
            schema_version=str(args.get("schema_version") or base.schema_version).strip(),
            upload_max_retries=int(
                args.get("upload_max_retries") or base.upload_max_retries
            ),
            upload_retry_base_seconds=float(
                args.get("upload_retry_base_seconds")
                or base.upload_retry_base_seconds
            ),
            upload_short_term_rate_limit_window_seconds=float(
                args.get("upload_short_term_rate_limit_window_seconds")
                or base.upload_short_term_rate_limit_window_seconds
            ),
            upload_hourly_rate_limit_delay_seconds=float(
                args.get("upload_hourly_rate_limit_delay_seconds")
                or base.upload_hourly_rate_limit_delay_seconds
            ),
            upload_consecutive_failure_threshold=int(
                args.get("upload_consecutive_failure_threshold")
                or base.upload_consecutive_failure_threshold
            ),
            upload_consecutive_failure_delay_seconds=float(
                args.get("upload_consecutive_failure_delay_seconds")
                or base.upload_consecutive_failure_delay_seconds
            ),
            upload_large_folder_num_workers=int(
                args.get("upload_large_folder_num_workers")
                or base.upload_large_folder_num_workers
            ),
            upload_large_folder_directory_cooldown_seconds=float(
                args.get("upload_large_folder_directory_cooldown_seconds")
                or base.upload_large_folder_directory_cooldown_seconds
            ),
        )

    def redacted_manifest_settings(self) -> dict[str, object]:
        """Return settings safe to write to manifests.

        The token itself is never serialized. Manifests only record whether a
        token was configured and which environment-variable order was checked.
        """
        return {
            "command": self.command,
            "source_dirs": [str(path) for path in self.source_dirs],
            "output_dir": str(self.output_dir),
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "dry_run": self.dry_run,
            "data_subdir": self.data_subdir,
            "output_batch_size": self.output_batch_size,
            "max_files_per_directory": self.max_files_per_directory,
            "parquet_compression": self.parquet_compression,
            "state_db_filename": self.state_db_filename,
            "schema_version": self.schema_version,
            "huggingface_token": redacted_token_state(self.hf_token),
        }
