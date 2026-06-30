"""Shared config value parsing helpers for post-processing pipelines."""

from __future__ import annotations

import importlib.util
import os
from datetime import UTC, datetime
from pathlib import Path


TRUE_STRINGS = {"1", "true", "yes", "on"}


def positive_int(value: object, *, default: int = 1) -> int:
    """Return a positive integer parsed from a config value."""
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return max(1, int(default))
    return max(1, parsed)


def bool_config(value: object) -> bool:
    """Return whether a config value is one of the accepted true strings."""
    return str(value or "").strip().lower() in TRUE_STRINGS


def string_list(value: object) -> tuple[str, ...]:
    """Parse list-like or comma-separated config values into a string tuple."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def utc_timestamp_run_id(prefix: str) -> str:
    """Return a UTC timestamped run id using the existing post-processing format."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{str(prefix).strip('_')}_{timestamp}"


def tokens_from_file(
    path: Path,
    *,
    module_name: str = "post_processing_tokens_config",
) -> tuple[str, ...]:
    """Load a TOKENS sequence from a Python config file if it exists."""
    resolved = Path(path)
    if not resolved.exists():
        return ()
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        return ()
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return ()
    return string_list(getattr(module, "TOKENS", ()))


def github_tokens_from_sources(
    configured_tokens: object,
    token_config_path: Path,
    *,
    fallback_tokens: object = (),
) -> tuple[str, ...]:
    """Resolve GitHub tokens from explicit config, env, config file, then fallback."""
    configured = string_list(configured_tokens)
    if configured:
        return configured
    env_tokens = string_list(os.environ.get("GITHUB_TOKENS"))
    if env_tokens:
        return env_tokens
    env_token = string_list(os.environ.get("GITHUB_TOKEN"))
    if env_token:
        return env_token
    post_processing_tokens = tokens_from_file(token_config_path)
    if post_processing_tokens:
        return post_processing_tokens
    return string_list(fallback_tokens)


def extraction_config_tokens() -> tuple[str, ...]:
    """Load fallback GitHub tokens from the extraction package when available."""
    try:
        from extraction.config.tokens_config import TOKENS
    except Exception:
        return ()
    return string_list(TOKENS)


def resolve_github_tokens(
    configured_tokens: object,
    token_config_path: Path,
) -> tuple[str, ...]:
    """Resolve GitHub tokens using the standard post-processing fallback order."""
    return github_tokens_from_sources(
        configured_tokens,
        token_config_path,
        fallback_tokens=extraction_config_tokens(),
    )
