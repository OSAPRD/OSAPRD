"""Stable PR deduplication helpers shared by analysis pipelines."""

from __future__ import annotations

from typing import Any


def stable_numeric_id(value: Any) -> str | None:
    """Return a normalized numeric identifier string when the value is stable."""
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdecimal():
        return None
    return str(int(text))


def stable_pr_dedup_key(
    *,
    base_repository_id: Any,
    pull_request_number: Any,
) -> str | None:
    """Return the stable PR key ``<base_repository_id>#<pr_number>`` when possible."""
    repository_id = stable_numeric_id(base_repository_id)
    number = stable_numeric_id(pull_request_number)
    if not repository_id or not number:
        return None
    return f"{repository_id}#{number}"
