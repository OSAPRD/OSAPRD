"""Shared repository key normalization helpers."""

from __future__ import annotations

from typing import Any


def normalize_repository_key(owner: str, name: str) -> str:
    """Return normalized owner/name repository key."""
    return f"{str(owner).strip().lower()}/{str(name).strip().lower()}"


def safe_repository_key(owner: str, name: str) -> str:
    """Return curation-safe owner__name repository key."""
    return f"{str(owner).strip().replace('/', '_')}__{str(name).strip().replace('/', '_')}"


def repository_key_from_full_name(value: object) -> str | None:
    """Parse owner/name repository full name into a normalized key."""
    text = str(value or "").strip()
    if "/" not in text:
        return None
    owner, name = text.split("/", 1)
    if not owner.strip() or not name.strip():
        return None
    return normalize_repository_key(owner, name)


def repository_key_from_safe_key(value: object) -> str | None:
    """Parse owner__name safe repository key into a normalized key."""
    text = str(value or "").strip()
    if "__" not in text:
        return None
    owner, name = text.split("__", 1)
    if not owner.strip() or not name.strip():
        return None
    return normalize_repository_key(owner, name)


def repository_identity_key(repository_payload: dict[str, Any]) -> str | None:
    """Return a stable repository identity from a repository payload."""
    for field_name in ("id", "name_with_owner", "url", "ssh_url", "name"):
        value = repository_payload.get(field_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def stable_numeric_id(value: Any) -> str | None:
    """Return a normalized numeric id string when a value is a stable numeric id."""
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return str(int(text)) if text.isdecimal() else None


def repository_numeric_id_from_payload(repository_payload: dict[str, Any]) -> str | None:
    """Return the repository numeric id from common repository payload fields."""
    for field_name in ("id", "database_id", "databaseId", "repository_id"):
        resolved = stable_numeric_id(repository_payload.get(field_name))
        if resolved:
            return resolved
    return None
