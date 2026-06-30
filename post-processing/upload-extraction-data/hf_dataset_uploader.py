"""Prepare and publish extraction parquet as a public dataset package.

The active input contract is extraction's local output root, usually shaped as
``data/<cohort>/*.parquet``. The pipeline streams those raw PR rows, converts
them into the public entity tables used by downstream readers, writes local
parquet batches plus metadata, and optionally uploads the staged directory to a
Hugging Face dataset repository.

Resumability lives in ``UploadStateStore``. The SQLite database records source
batch completion, retained PR keys, global stable PR deduplication, and uploaded
repo paths so large runs can be resumed from the same staging directory.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Optional, get_args, get_origin, get_type_hints

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi

try:
    from curation.utility.loader import (
        coerce_pull_request_record,
        iter_source_parquet_files,
    )
except ModuleNotFoundError:
    from utility.loader import coerce_pull_request_record, iter_source_parquet_files
from extraction.dtos.dtos import FileChange, PullRequest, Repository

UTILITY_DIR = Path(__file__).resolve().parents[1] / "utility"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from repository_keys import repository_identity_key as _repository_identity_key


# ---------------------------------------------------------------------------
# Identity and source-row normalization
# ---------------------------------------------------------------------------

def _normalize_cohort(value: object) -> str | None:
    """Normalize a cohort/group label."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _cohort_authored_by_agent(cohort: str) -> bool:
    """Return True when the cohort should be treated as agent-authored."""
    normalized = _normalize_cohort(cohort)
    return normalized not in {None, "human", "humans"}


def _cohort_processing_priority(cohort: str) -> tuple[int, str]:
    """Sort agent cohorts before human cohorts for global duplicate retention."""
    normalized = _normalize_cohort(cohort) or ""
    return (0 if _cohort_authored_by_agent(normalized) else 1, normalized)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Return an attribute from a DTO or dict-like object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def stable_base_repository_id_from_snapshot_key(value: Any) -> Optional[str]:
    """Extract a stable numeric repository id from a repository snapshot key."""
    text = str(value or "").strip()
    marker = "::repository_snapshot::"
    if marker not in text:
        return None
    remainder = text.rsplit(marker, 1)[-1]
    repository_identity, separator, _role = remainder.rpartition("::")
    if not separator:
        return None
    repository_identity = repository_identity.strip()
    return repository_identity if repository_identity.isdecimal() else None


def _stable_pull_request_number(value: Any) -> Optional[str]:
    """Normalize a GitHub pull request number for stable deduplication."""
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
    base_repository_snapshot_key: Any,
    pull_request_number: Any,
) -> Optional[str]:
    """Return the global stable PR dedup key when stable inputs are present."""
    repository_id = stable_base_repository_id_from_snapshot_key(
        base_repository_snapshot_key
    )
    number = _stable_pull_request_number(pull_request_number)
    if not repository_id or not number:
        return None
    return f"{repository_id}#{number}"


def stable_pr_dedup_key_for_record(record: dict[str, Any]) -> Optional[str]:
    """Return the stable global PR dedup key for a standardized PR record."""
    return stable_pr_dedup_key(
        base_repository_snapshot_key=record.get("base_repository_snapshot_key"),
        pull_request_number=record.get("number"),
    )


def _pull_request_key_from_record(record: dict[str, Any]) -> Optional[str]:
    """Return a non-empty standardized pull_request_key from an entity row."""
    value = record.get("pull_request_key")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_timestamp_field_name(field_name: str) -> bool:
    """Return True for top-level GitHub datetime-like fields."""
    return field_name.endswith("_at")


def _parse_timestamp_value(value: Any) -> Optional[datetime]:
    """Parse a scalar timestamp value into a UTC datetime for Arrow."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def dedup_key_for_pr(pr: PullRequest) -> Optional[str]:
    """Return the curation-style deduplication key for a PR."""
    url = _get_attr(pr, "url")
    if url:
        return str(url)
    pr_id = _get_attr(pr, "id")
    if pr_id is not None:
        return str(pr_id)
    return None


def repository_keys_for_pr(
    pr: PullRequest,
    *,
    raw_payload: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Return stable keys for any repositories referenced by the PR."""
    dto_payload = pr.to_dict()
    seen: set[str] = set()
    keys: list[str] = []
    for full_field_name, peek_field_name in (
        ("base_repository_full", "base_repository"),
        ("head_repository_full", "head_repository"),
    ):
        repository_payload = _source_repository_payload(
            raw_payload=raw_payload,
            dto_payload=dto_payload,
            full_field_name=full_field_name,
            peek_field_name=peek_field_name,
        )
        if not repository_payload:
            continue
        key = _repository_identity_key(repository_payload)
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def languages_for_pr(pr: PullRequest) -> list[str]:
    """Return normalized programming languages associated with the PR."""
    values = _get_attr(pr, "file_languages") or []
    if not isinstance(values, list):
        values = list(values)
    languages: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        languages.append(normalized)
    return languages


# ---------------------------------------------------------------------------
# Public schema derivation
# ---------------------------------------------------------------------------

def _resolve_scalar_kind(annotation: Any) -> str:
    """Resolve a dataclass field annotation to a storage kind."""
    origin = get_origin(annotation)
    if origin is None:
        if annotation is str:
            return "string"
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        return "json"
    if origin in (list, tuple, dict):
        return "json"
    if origin is not None and str(origin).endswith("Union"):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _resolve_scalar_kind(args[0])
    return "json"


def _resolve_storage_kind(field_name: str, annotation: Any) -> str:
    """Resolve a top-level output field to its storage kind."""
    if _is_timestamp_field_name(field_name):
        return "timestamp"
    return _resolve_scalar_kind(annotation)


def _field_kinds_for_dataclass(cls: type[Any]) -> dict[str, str]:
    """Return storage kinds for every field on a DTO dataclass.

    Extraction DTOs use postponed annotations. Resolving type hints here keeps
    bool/int/timestamp fields typed correctly in Arrow instead of falling back
    to generic JSON strings.
    """
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}
    return {
        field.name: _resolve_storage_kind(field.name, type_hints.get(field.name, field.type))
        for field in fields(cls)
    }


PR_FIELD_KINDS: dict[str, str] = _field_kinds_for_dataclass(PullRequest)
FILE_FIELD_KINDS: dict[str, str] = _field_kinds_for_dataclass(FileChange)
REPOSITORY_FIELD_KINDS: dict[str, str] = _field_kinds_for_dataclass(Repository)

# ---------------------------------------------------------------------------
# Public entity layout
# ---------------------------------------------------------------------------

# Public entity folder names. Keep these stable unless downstream readers are
# migrated at the same time.
ENTITY_PULL_REQUESTS = "PullRequestRecords"
ENTITY_FULL_PULL_REQUESTS = "AggregatedPullRequests"
ENTITY_FILES = "FileChangeRecords"
ENTITY_REPOSITORY_SNAPSHOTS = "RepositoryRecords"
ENTITY_NAMES: tuple[str, ...] = (
    ENTITY_PULL_REQUESTS,
    ENTITY_FULL_PULL_REQUESTS,
    ENTITY_FILES,
    ENTITY_REPOSITORY_SNAPSHOTS,
)
ENTITY_FILE_PREFIXES: dict[str, str] = {
    ENTITY_PULL_REQUESTS: "pull_request_records_batch",
    ENTITY_FULL_PULL_REQUESTS: "aggregated_pull_requests_batch",
    ENTITY_FILES: "file_change_records_batch",
    ENTITY_REPOSITORY_SNAPSHOTS: "repository_records_batch",
}
ENTITY_CONFIG_SLUGS: dict[str, str] = {
    ENTITY_PULL_REQUESTS: "pull_request_records",
    ENTITY_FULL_PULL_REQUESTS: "aggregated_pull_requests",
    ENTITY_FILES: "file_change_records",
    ENTITY_REPOSITORY_SNAPSHOTS: "repository_records",
}
COHORT_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude",
    "codegen": "Codegen",
    "codex": "Codex",
    "copilot": "Copilot",
    "cosine": "Cosine",
    "cursor": "Cursor",
    "devin": "Devin",
    "jules": "Jules",
    "junie": "Junie",
    "openhands": "OpenHands",
}

# The upload dataset intentionally excludes heavy or duplicated nested fields
# from the flat PullRequestRecords table. AggregatedPullRequests keeps selected
# nested payloads for compatibility with existing readers.
PULL_REQUEST_TOP_LEVEL_EXCLUDED_FIELDS: set[str] = {
    "id",
    "comments",
    "reviews",
    "commits",
    "requested_reviewers",
    "active_lock_reason",
    "mergeable",
    "mergeable_state",
    "mergeable_method",
    "auto_merge",
    "label_count",
    "post_merge_file_snapshots",
    "timeline_items",
    "timeline_count",
    "files",
    "file_languages",
    "base_repository_full",
    "head_repository_full",
    "last_edited_at",
    "published_at",
    "discovered_agent",
}
FULL_PULL_REQUEST_INCLUDED_NESTED_FIELDS: set[str] = {
    "files",
    "base_repository_full",
    "head_repository_full",
}
FULL_PULL_REQUEST_TOP_LEVEL_EXCLUDED_FIELDS: set[str] = {
    field_name
    for field_name in PULL_REQUEST_TOP_LEVEL_EXCLUDED_FIELDS
    if field_name not in FULL_PULL_REQUEST_INCLUDED_NESTED_FIELDS
}

STANDARDIZED_NESTED_FIELD_EXCLUSIONS: dict[str, set[str]] = {
    "base_repository_full": {"readme_is_truncated"},
    "head_repository_full": {"readme_is_truncated"},
}

# Extraction can omit these fields from file payloads. Setting them explicitly to
# null keeps the Arrow schema stable across REST- and GraphQL-enriched rows.
STANDARDIZED_FILE_NULLABLE_FIELDS: tuple[str, ...] = (
    "patch",
    "sha",
    "raw_url",
    "contents_url",
    "previous_filename",
    "is_binary",
)

PULL_REQUEST_FIELD_KINDS: dict[str, str] = {
    field_name: kind
    for field_name, kind in PR_FIELD_KINDS.items()
    if field_name not in PULL_REQUEST_TOP_LEVEL_EXCLUDED_FIELDS
}
FULL_PULL_REQUEST_FIELD_KINDS: dict[str, str] = {
    field_name: kind
    for field_name, kind in PR_FIELD_KINDS.items()
    if field_name not in FULL_PULL_REQUEST_TOP_LEVEL_EXCLUDED_FIELDS
}
FILE_ENTITY_EXCLUDED_FIELDS: set[str] = {
    "base_content",
    "head_content",
    "is_truncated",
    "language",
    "status",
}
FILE_ENTITY_FIELD_KINDS: dict[str, str] = {
    field_name: kind
    for field_name, kind in FILE_FIELD_KINDS.items()
    if field_name not in FILE_ENTITY_EXCLUDED_FIELDS
}
REPOSITORY_ENTITY_EXCLUDED_FIELDS: set[str] = {
    "allow_merge_commit",
    "allow_squash_merge",
    "allow_rebase_merge",
    "security_policy_url",
    "archived_reason",
    "lock_reason",
    "readme_is_truncated",
    "domains",
    "repository_topics",
    "popularity_label",
}
REPOSITORY_ENTITY_FIELD_KINDS: dict[str, str] = {
    field_name: kind
    for field_name, kind in REPOSITORY_FIELD_KINDS.items()
    if field_name not in REPOSITORY_ENTITY_EXCLUDED_FIELDS
}


# ---------------------------------------------------------------------------
# JSON/scalar coercion helpers
# ---------------------------------------------------------------------------

def _nested_json_field_kinds(field_kinds: dict[str, str]) -> dict[str, str]:
    """Return field kinds for nested JSON payloads without timestamp coercion."""
    return {
        field_name: "string" if kind == "timestamp" else kind
        for field_name, kind in field_kinds.items()
    }


NESTED_FILE_ENTITY_FIELD_KINDS = _nested_json_field_kinds(FILE_ENTITY_FIELD_KINDS)
NESTED_REPOSITORY_ENTITY_FIELD_KINDS = _nested_json_field_kinds(
    REPOSITORY_ENTITY_FIELD_KINDS
)


def _json_ready(value: Any) -> Any:
    """Normalize a value into a JSON-serializable shape."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    if isinstance(value, dict):
        if not value:
            return None
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _serialize_json_field(value: Any) -> Optional[str]:
    """Serialize nested/list values to a stable JSON string."""
    normalized = _json_ready(value)
    if normalized is None:
        return None
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _jsonish_ready(value: Any) -> Any:
    """Normalize JSON-like values, decoding JSON strings when they hold nested data."""
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return _json_ready(value)


def _standardize_scalar(kind: str, value: Any) -> Any:
    """Convert one field value to the standardized dataset schema."""
    if value is None:
        return None
    if kind == "string":
        return str(value)
    if kind == "bool":
        return bool(value)
    if kind == "int":
        try:
            return int(value)
        except Exception:
            return None
    if kind == "float":
        try:
            return float(value)
        except Exception:
            return None
    if kind == "timestamp":
        return _parse_timestamp_value(value)
    return _serialize_json_field(value)


def _prune_nested_keys(value: Any, excluded_keys: set[str]) -> Any:
    """Drop excluded keys from nested dict/list payloads."""
    if isinstance(value, dict):
        return {
            str(key): _prune_nested_keys(item, excluded_keys)
            for key, item in value.items()
            if str(key) not in excluded_keys
        }
    if isinstance(value, list):
        return [_prune_nested_keys(item, excluded_keys) for item in value]
    if isinstance(value, tuple):
        return [_prune_nested_keys(item, excluded_keys) for item in value]
    return value


def _ensure_file_nullable_fields(value: Any) -> Any:
    """Ensure standardized file payloads retain nullable fields even when absent."""
    if isinstance(value, dict):
        normalized = {str(key): item for key, item in value.items()}
        for field_name in STANDARDIZED_FILE_NULLABLE_FIELDS:
            normalized.setdefault(field_name, None)
        return normalized
    if isinstance(value, list):
        return [_ensure_file_nullable_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_ensure_file_nullable_fields(item) for item in value]
    return value


def _sanitize_standardized_field_value(field_name: str, value: Any) -> Any:
    """Remove nested keys that should not be published in the upload dataset."""
    if value is None:
        return value
    if field_name == "files":
        value = _ensure_file_nullable_fields(value)
    excluded_keys = STANDARDIZED_NESTED_FIELD_EXCLUSIONS.get(field_name)
    return _prune_nested_keys(value, excluded_keys) if excluded_keys else value


def _source_payload_value(
    *,
    field_name: str,
    raw_payload: Optional[dict[str, Any]],
    dto_payload: dict[str, Any],
) -> Any:
    """Prefer the original row payload so nested source fields are preserved exactly."""
    if isinstance(raw_payload, dict) and field_name in raw_payload:
        return raw_payload.get(field_name)
    return dto_payload.get(field_name)


def _normalize_dict_payload(value: Any) -> Optional[dict[str, Any]]:
    """Normalize a nested payload into a plain dictionary when possible."""
    normalized = _jsonish_ready(value)
    if isinstance(normalized, dict):
        return {str(key): item for key, item in normalized.items()}
    return None


def _normalize_list_of_dict_payloads(value: Any) -> list[dict[str, Any]]:
    """Normalize a nested payload into a list of plain dictionaries."""
    normalized = _jsonish_ready(value)
    if isinstance(normalized, dict):
        return [{str(key): item for key, item in normalized.items()}]
    if isinstance(normalized, list):
        return [
            {str(key): item for key, item in item_value.items()}
            for item_value in normalized
            if isinstance(item_value, dict)
        ]
    return []


def _extra_fields_json(
    raw_payload: Optional[dict[str, Any]],
    *,
    known_field_names: set[str],
) -> Optional[str]:
    """Serialize unexpected source fields so nothing is silently dropped."""
    if not isinstance(raw_payload, dict):
        return None
    extras = {
        str(key): value
        for key, value in raw_payload.items()
        if str(key) not in known_field_names
    }
    return _serialize_json_field(extras)


# ---------------------------------------------------------------------------
# Raw extraction row -> public entity row conversion
# ---------------------------------------------------------------------------

def _standardize_payload_record(
    field_kinds: dict[str, str],
    payload: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Standardize one entity payload against the requested field kinds."""
    source_payload = payload if isinstance(payload, dict) else {}
    return {
        field_name: _standardize_scalar(kind, source_payload.get(field_name))
        for field_name, kind in field_kinds.items()
    }


def pull_request_key_for_record(pr: PullRequest, dedup_key: Optional[str]) -> str:
    """Return a stable relational key for one pull request row."""
    pr_id = _get_attr(pr, "id")
    if pr_id is not None and str(pr_id).strip():
        return str(pr_id)
    if dedup_key:
        return str(dedup_key)
    url = _get_attr(pr, "url")
    if url:
        return str(url)
    return "unknown_pull_request"


def _pull_request_id(pr: PullRequest) -> Optional[str]:
    """Return the raw pull request id when present."""
    value = _get_attr(pr, "id")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _repository_payload_signature(repository_payload: dict[str, Any]) -> str:
    """Return a comparable signature for repository snapshot deduping."""
    comparable = {
        key: value
        for key, value in repository_payload.items()
        if key not in {"pr_id", "role"}
    }
    normalized = _json_ready(comparable)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _repository_snapshot_key(
    *,
    pull_request_key: str,
    repository_payload: dict[str, Any],
    role: str,
) -> str:
    """Return a stable key for one repository snapshot row."""
    identity = _repository_identity_key(repository_payload) or role.lower()
    return f"{pull_request_key}::repository_snapshot::{identity}::{role}"


def _source_repository_payload(
    *,
    raw_payload: Optional[dict[str, Any]],
    dto_payload: dict[str, Any],
    full_field_name: str,
    peek_field_name: str,
) -> Optional[dict[str, Any]]:
    """Return the best available repository payload, preferring the full snapshot."""
    candidate = None
    if isinstance(raw_payload, dict) and raw_payload.get(full_field_name) is not None:
        candidate = raw_payload.get(full_field_name)
    elif dto_payload.get(full_field_name) is not None:
        candidate = dto_payload.get(full_field_name)
    elif isinstance(raw_payload, dict) and raw_payload.get(peek_field_name) is not None:
        candidate = raw_payload.get(peek_field_name)
    else:
        candidate = dto_payload.get(peek_field_name)
    candidate = _sanitize_standardized_field_value(full_field_name, candidate)
    return _normalize_dict_payload(candidate)


def standardize_pull_request_record(
    pr: PullRequest,
    *,
    raw_payload: Optional[dict[str, Any]],
    cohort: str,
    dedup_key: Optional[str],
    pull_request_key: str,
    base_repository_snapshot_key: Optional[str],
    head_repository_snapshot_key: Optional[str],
    schema_version: str,
    source_file: Path,
) -> dict[str, Any]:
    """Convert a PR DTO into a standardized pull-request row."""
    payload = pr.to_dict()
    standardized = _standardize_payload_record(
        PULL_REQUEST_FIELD_KINDS,
        {
            field_name: _sanitize_standardized_field_value(
                field_name,
                _source_payload_value(
                    field_name=field_name,
                    raw_payload=raw_payload,
                    dto_payload=payload,
                ),
            )
            for field_name in PULL_REQUEST_FIELD_KINDS
        },
    )
    standardized["pull_request_key"] = pull_request_key
    standardized["base_repository_snapshot_key"] = base_repository_snapshot_key
    standardized["head_repository_snapshot_key"] = head_repository_snapshot_key
    standardized["authored_by_agent"] = _cohort_authored_by_agent(cohort)
    standardized["cohort"] = cohort
    standardized["dedup_key"] = dedup_key
    standardized["source_parquet_file"] = str(source_file)
    return standardized


def _standardize_full_pull_request_nested_repository_payload(
    value: Any,
) -> Optional[str]:
    """Return one standardized nested repository payload as a JSON string."""
    payload = _normalize_dict_payload(value)
    if not payload:
        return None
    return _serialize_json_field(
        _standardize_payload_record(NESTED_REPOSITORY_ENTITY_FIELD_KINDS, payload)
    )


def _standardize_full_pull_request_nested_file_payloads(
    value: Any,
) -> Optional[str]:
    """Return standardized nested file payloads as a JSON string."""
    payloads = _normalize_list_of_dict_payloads(_ensure_file_nullable_fields(value))
    if not payloads:
        return None
    return _serialize_json_field(
        [
            _standardize_payload_record(NESTED_FILE_ENTITY_FIELD_KINDS, file_payload)
            for file_payload in payloads
        ]
    )


def standardize_full_pull_request_record(
    pr: PullRequest,
    *,
    raw_payload: Optional[dict[str, Any]],
    cohort: str,
    dedup_key: Optional[str],
    pull_request_key: str,
    base_repository_snapshot_key: Optional[str],
    head_repository_snapshot_key: Optional[str],
    schema_version: str,
    source_file: Path,
) -> dict[str, Any]:
    """Convert a PR DTO into a full standardized pull-request row with nested repo/file payloads."""
    payload = pr.to_dict()
    standardized: dict[str, Any] = {}
    for field_name, kind in FULL_PULL_REQUEST_FIELD_KINDS.items():
        if field_name == "files":
            standardized[field_name] = _standardize_full_pull_request_nested_file_payloads(
                _source_payload_value(
                    field_name=field_name,
                    raw_payload=raw_payload,
                    dto_payload=payload,
                )
            )
            continue
        if field_name in {"base_repository_full", "head_repository_full"}:
            standardized[field_name] = _standardize_full_pull_request_nested_repository_payload(
                _source_repository_payload(
                    raw_payload=raw_payload,
                    dto_payload=payload,
                    full_field_name=field_name,
                    peek_field_name=(
                        "base_repository"
                        if field_name == "base_repository_full"
                        else "head_repository"
                    ),
                )
            )
            continue
        standardized[field_name] = _standardize_scalar(
            kind,
            _sanitize_standardized_field_value(
                field_name,
                _source_payload_value(
                    field_name=field_name,
                    raw_payload=raw_payload,
                    dto_payload=payload,
                ),
            ),
        )
    standardized["pull_request_key"] = pull_request_key
    standardized["base_repository_snapshot_key"] = base_repository_snapshot_key
    standardized["head_repository_snapshot_key"] = head_repository_snapshot_key
    standardized["authored_by_agent"] = _cohort_authored_by_agent(cohort)
    standardized["cohort"] = cohort
    standardized["dedup_key"] = dedup_key
    standardized["source_parquet_file"] = str(source_file)
    return standardized


def standardize_file_records(
    pr: PullRequest,
    *,
    raw_payload: Optional[dict[str, Any]],
    cohort: str,
    pull_request_key: str,
    schema_version: str,
    source_file: Path,
) -> list[dict[str, Any]]:
    """Convert one PR payload into standardized file rows."""
    dto_payload = pr.to_dict()
    file_payloads = _normalize_list_of_dict_payloads(
        _ensure_file_nullable_fields(
            _source_payload_value(
                field_name="files",
                raw_payload=raw_payload,
                dto_payload=dto_payload,
            )
        )
    )
    pull_request_id = _pull_request_id(pr)
    records: list[dict[str, Any]] = []
    for file_index, file_payload in enumerate(file_payloads):
        standardized = _standardize_payload_record(FILE_ENTITY_FIELD_KINDS, file_payload)
        standardized["file_key"] = f"{pull_request_key}::file::{file_index:06d}"
        standardized["pull_request_key"] = pull_request_key
        standardized["pull_request_id"] = pull_request_id
        standardized["cohort"] = cohort
        standardized["file_index"] = file_index
        standardized["source_parquet_file"] = str(source_file)
        records.append(standardized)
    return records


def standardize_repository_snapshot_records(
    pr: PullRequest,
    *,
    raw_payload: Optional[dict[str, Any]],
    cohort: str,
    pull_request_key: str,
    schema_version: str,
    source_file: Path,
) -> tuple[list[dict[str, Any]], Optional[str], Optional[str]]:
    """Convert one PR payload into repository snapshot rows."""
    dto_payload = pr.to_dict()
    pull_request_id = _pull_request_id(pr)

    base_payload = _source_repository_payload(
        raw_payload=raw_payload,
        dto_payload=dto_payload,
        full_field_name="base_repository_full",
        peek_field_name="base_repository",
    )
    head_payload = _source_repository_payload(
        raw_payload=raw_payload,
        dto_payload=dto_payload,
        full_field_name="head_repository_full",
        peek_field_name="head_repository",
    )

    def _build_repository_record(
        repository_payload: dict[str, Any],
        *,
        role: str,
        referenced_as_base: bool,
        referenced_as_head: bool,
    ) -> tuple[dict[str, Any], str]:
        """Build one repository snapshot row and its relational snapshot key."""
        payload = dict(repository_payload)
        payload["pr_id"] = pull_request_id
        payload["role"] = role
        repository_snapshot_key = _repository_snapshot_key(
            pull_request_key=pull_request_key,
            repository_payload=payload,
            role=role,
        )
        standardized = _standardize_payload_record(REPOSITORY_ENTITY_FIELD_KINDS, payload)
        standardized["repository_snapshot_key"] = repository_snapshot_key
        standardized["pull_request_key"] = pull_request_key
        standardized["cohort"] = cohort
        standardized["referenced_as_base"] = referenced_as_base
        standardized["referenced_as_head"] = referenced_as_head
        standardized["source_parquet_file"] = str(source_file)
        return standardized, repository_snapshot_key

    if (
        base_payload
        and head_payload
        and _repository_identity_key(base_payload) == _repository_identity_key(head_payload)
        and _repository_payload_signature(base_payload)
        == _repository_payload_signature(head_payload)
    ):
        record, snapshot_key = _build_repository_record(
            base_payload,
            role="BASE_AND_HEAD",
            referenced_as_base=True,
            referenced_as_head=True,
        )
        return [record], snapshot_key, snapshot_key

    records: list[dict[str, Any]] = []
    base_snapshot_key: Optional[str] = None
    head_snapshot_key: Optional[str] = None
    if base_payload:
        record, base_snapshot_key = _build_repository_record(
            base_payload,
            role="BASE",
            referenced_as_base=True,
            referenced_as_head=False,
        )
        records.append(record)
    if head_payload:
        record, head_snapshot_key = _build_repository_record(
            head_payload,
            role="HEAD",
            referenced_as_base=False,
            referenced_as_head=True,
        )
        records.append(record)
    return records, base_snapshot_key, head_snapshot_key


def _arrow_field_for_kind(name: str, kind: str) -> pa.Field:
    """Return an Arrow field for one standardized storage kind."""
    if kind in {"string", "json"}:
        return pa.field(name, pa.string())
    if kind == "bool":
        return pa.field(name, pa.bool_())
    if kind == "int":
        return pa.field(name, pa.int64())
    if kind == "timestamp":
        return pa.field(name, pa.timestamp("ms", tz="UTC"))
    return pa.field(name, pa.float64())


def _build_entity_schema(
    field_kinds: dict[str, str],
    extra_fields: list[pa.Field],
) -> pa.Schema:
    """Build an Arrow schema from entity field kinds plus metadata fields."""
    return pa.schema(
        [
            *(_arrow_field_for_kind(name, kind) for name, kind in field_kinds.items()),
            *extra_fields,
        ]
    )


PULL_REQUEST_SCHEMA = _build_entity_schema(
    PULL_REQUEST_FIELD_KINDS,
    [
        pa.field("pull_request_key", pa.string()),
        pa.field("base_repository_snapshot_key", pa.string()),
        pa.field("head_repository_snapshot_key", pa.string()),
        pa.field("cohort", pa.string()),
    ],
)

FULL_PULL_REQUEST_SCHEMA = _build_entity_schema(
    FULL_PULL_REQUEST_FIELD_KINDS,
    [
        pa.field("pull_request_key", pa.string()),
        pa.field("base_repository_snapshot_key", pa.string()),
        pa.field("head_repository_snapshot_key", pa.string()),
        pa.field("cohort", pa.string()),
    ],
)

FILES_SCHEMA = _build_entity_schema(
    FILE_ENTITY_FIELD_KINDS,
    [
        pa.field("file_key", pa.string()),
        pa.field("pull_request_key", pa.string()),
        pa.field("pull_request_id", pa.string()),
        pa.field("file_index", pa.int64()),
    ],
)

REPOSITORY_SNAPSHOTS_SCHEMA = _build_entity_schema(
    REPOSITORY_ENTITY_FIELD_KINDS,
    [
        pa.field("repository_snapshot_key", pa.string()),
        pa.field("pull_request_key", pa.string()),
        pa.field("referenced_as_base", pa.bool_()),
        pa.field("referenced_as_head", pa.bool_()),
    ],
)

ENTITY_SCHEMAS: dict[str, pa.Schema] = {
    ENTITY_PULL_REQUESTS: PULL_REQUEST_SCHEMA,
    ENTITY_FULL_PULL_REQUESTS: FULL_PULL_REQUEST_SCHEMA,
    ENTITY_FILES: FILES_SCHEMA,
    ENTITY_REPOSITORY_SNAPSHOTS: REPOSITORY_SNAPSHOTS_SCHEMA,
}
ENTITY_PRIMARY_KEYS: dict[str, str] = {
    ENTITY_PULL_REQUESTS: "pull_request_key",
    ENTITY_FULL_PULL_REQUESTS: "pull_request_key",
    ENTITY_FILES: "file_key",
    ENTITY_REPOSITORY_SNAPSHOTS: "repository_snapshot_key",
}
ENTITY_FOREIGN_KEYS: dict[str, list[dict[str, str]]] = {
    ENTITY_PULL_REQUESTS: [
        {
            "column": "base_repository_snapshot_key",
            "references_entity": ENTITY_REPOSITORY_SNAPSHOTS,
            "references_column": "repository_snapshot_key",
        },
        {
            "column": "head_repository_snapshot_key",
            "references_entity": ENTITY_REPOSITORY_SNAPSHOTS,
            "references_column": "repository_snapshot_key",
        },
    ],
    ENTITY_FULL_PULL_REQUESTS: [
        {
            "column": "base_repository_snapshot_key",
            "references_entity": ENTITY_REPOSITORY_SNAPSHOTS,
            "references_column": "repository_snapshot_key",
        },
        {
            "column": "head_repository_snapshot_key",
            "references_entity": ENTITY_REPOSITORY_SNAPSHOTS,
            "references_column": "repository_snapshot_key",
        },
    ],
    ENTITY_FILES: [
        {
            "column": "pull_request_key",
            "references_entity": ENTITY_PULL_REQUESTS,
            "references_column": "pull_request_key",
        },
    ],
    ENTITY_REPOSITORY_SNAPSHOTS: [
        {
            "column": "pull_request_key",
            "references_entity": ENTITY_PULL_REQUESTS,
            "references_column": "pull_request_key",
        },
    ],
}


# ---------------------------------------------------------------------------
# Durable local state
# ---------------------------------------------------------------------------

class UploadStateStore:
    """SQLite-backed deduplication and upload state for large runs."""

    def __init__(self, db_path: Path) -> None:
        """Open the state database and create tables used for resumability."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")

        # seen_prs is cohort-scoped because the same PR URL may legitimately
        # appear in separate source roots before global stable deduplication.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_prs (
                cohort TEXT NOT NULL,
                dedup_key TEXT NOT NULL,
                first_seen_utc TEXT NOT NULL,
                PRIMARY KEY (cohort, dedup_key)
            )
            """
        )
        # Stable PR keys use repository id plus PR number. This makes agentic
        # and human duplicate resolution deterministic across source cohorts.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS global_seen_stable_prs (
                stable_pr_key TEXT NOT NULL PRIMARY KEY,
                cohort TEXT NOT NULL,
                source_row_key TEXT,
                first_seen_utc TEXT NOT NULL
            )
            """
        )
        # Retained PR keys drive relational filtering for file/repository rows:
        # child rows are written only when their parent PR survived dedup.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retained_pull_request_keys (
                pull_request_key TEXT NOT NULL PRIMARY KEY,
                cohort TEXT NOT NULL,
                stable_pr_key TEXT,
                retained_at_utc TEXT NOT NULL
            )
            """
        )
        # Uploaded repo paths are tracked separately from source progress so an
        # interrupted upload can resume without restaging parquet.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_files (
                repo_path TEXT NOT NULL PRIMARY KEY,
                local_path TEXT NOT NULL,
                uploaded_at_utc TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unique_repositories (
                cohort TEXT NOT NULL,
                repository_key TEXT NOT NULL,
                first_seen_utc TEXT NOT NULL,
                PRIMARY KEY (cohort, repository_key)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unique_languages (
                cohort TEXT NOT NULL,
                language TEXT NOT NULL,
                first_seen_utc TEXT NOT NULL,
                PRIMARY KEY (cohort, language)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retained_prs (
                source_row_key TEXT NOT NULL PRIMARY KEY,
                cohort TEXT NOT NULL,
                retained_at_utc TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_file_progress (
                source_path TEXT NOT NULL PRIMARY KEY,
                last_completed_batch_index INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at_utc TEXT NOT NULL
            )
            """
        )
        self._migrate_retained_prs_schema_if_needed()
        self.conn.commit()

    def _migrate_retained_prs_schema_if_needed(self) -> None:
        """Upgrade legacy retained_prs schemas to the row-keyed resume format."""
        columns = [
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(retained_prs)").fetchall()
        ]
        if "source_row_key" in columns:
            return
        legacy_rows = self.conn.execute(
            "SELECT retained_id, cohort, retained_at_utc FROM retained_prs"
        ).fetchall()
        self.conn.execute("ALTER TABLE retained_prs RENAME TO retained_prs_legacy")
        self.conn.execute(
            """
            CREATE TABLE retained_prs (
                source_row_key TEXT NOT NULL PRIMARY KEY,
                cohort TEXT NOT NULL,
                retained_at_utc TEXT NOT NULL
            )
            """
        )
        if legacy_rows:
            self.conn.executemany(
                """
                INSERT INTO retained_prs (source_row_key, cohort, retained_at_utc)
                VALUES (?, ?, ?)
                """,
                [
                    (f"legacy::{int(retained_id)}", str(cohort), str(retained_at_utc))
                    for retained_id, cohort, retained_at_utc in legacy_rows
                ],
            )
        self.conn.execute("DROP TABLE retained_prs_legacy")

    def record_pr_if_new(self, cohort: str, dedup_key: str) -> bool:
        """Return True when a PR key is new for the cohort."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO seen_prs (cohort, dedup_key, first_seen_utc)
            VALUES (?, ?, ?)
            """,
            (cohort, dedup_key, datetime.now(timezone.utc).isoformat()),
        )
        return cursor.rowcount > 0

    def record_stable_pr_if_new(
        self,
        *,
        stable_pr_key: str,
        cohort: str,
        source_row_key: Optional[str] = None,
    ) -> bool:
        """Return True when a stable PR key has not been seen globally."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO global_seen_stable_prs (
                stable_pr_key,
                cohort,
                source_row_key,
                first_seen_utc
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                stable_pr_key,
                cohort,
                source_row_key,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.rowcount > 0

    def record_retained_pull_request_key(
        self,
        *,
        pull_request_key: str,
        cohort: str,
        stable_pr_key: Optional[str],
    ) -> bool:
        """Persist a retained standardized pull_request_key."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO retained_pull_request_keys (
                pull_request_key,
                cohort,
                stable_pr_key,
                retained_at_utc
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                pull_request_key,
                cohort,
                stable_pr_key,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.rowcount > 0

    def is_retained_pull_request_key(self, pull_request_key: str) -> bool:
        """Return True when a standardized pull_request_key survived PR dedup."""
        row = self.conn.execute(
            """
            SELECT 1
            FROM retained_pull_request_keys
            WHERE pull_request_key = ?
            LIMIT 1
            """,
            (pull_request_key,),
        ).fetchone()
        return row is not None

    def stable_pr_key_for_retained_pull_request_key(
        self,
        pull_request_key: str,
    ) -> Optional[str]:
        """Return the stable PR key associated with a retained pull_request_key."""
        row = self.conn.execute(
            """
            SELECT stable_pr_key
            FROM retained_pull_request_keys
            WHERE pull_request_key = ?
            LIMIT 1
            """,
            (pull_request_key,),
        ).fetchone()
        if row is None:
            return None
        value = row[0]
        return str(value).strip() if value is not None and str(value).strip() else None

    def is_uploaded(self, repo_path: str) -> bool:
        """Return True when a repo target path has already been uploaded."""
        row = self.conn.execute(
            "SELECT 1 FROM uploaded_files WHERE repo_path = ? LIMIT 1",
            (repo_path,),
        ).fetchone()
        return row is not None

    def record_repository_if_new(self, cohort: str, repository_key: str) -> bool:
        """Return True when a repository is new for the cohort."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO unique_repositories (cohort, repository_key, first_seen_utc)
            VALUES (?, ?, ?)
            """,
            (cohort, repository_key, datetime.now(timezone.utc).isoformat()),
        )
        return cursor.rowcount > 0

    def record_language_if_new(self, cohort: str, language: str) -> bool:
        """Return True when a language is new for the cohort."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO unique_languages (cohort, language, first_seen_utc)
            VALUES (?, ?, ?)
            """,
            (cohort, language, datetime.now(timezone.utc).isoformat()),
        )
        return cursor.rowcount > 0

    def record_retained_pr_if_new(self, source_row_key: str, cohort: str) -> bool:
        """Return True when a retained source row has not been recorded before."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO retained_prs (source_row_key, cohort, retained_at_utc)
            VALUES (?, ?, ?)
            """,
            (source_row_key, cohort, datetime.now(timezone.utc).isoformat()),
        )
        return cursor.rowcount > 0

    def is_retained_pr_source_row(self, source_row_key: str) -> bool:
        """Return True when a source row was already retained."""
        row = self.conn.execute(
            """
            SELECT 1
            FROM retained_prs
            WHERE source_row_key = ?
            LIMIT 1
            """,
            (source_row_key,),
        ).fetchone()
        return row is not None

    def last_completed_batch_index(self, source_path: Path) -> int:
        """Return the last fully committed batch index for a source parquet file."""
        row = self.conn.execute(
            """
            SELECT last_completed_batch_index
            FROM source_file_progress
            WHERE source_path = ?
            LIMIT 1
            """,
            (str(source_path),),
        ).fetchone()
        if row is None:
            return 0
        return int(row[0] or 0)

    def is_source_file_completed(self, source_path: Path) -> bool:
        """Return True when a source parquet file has already been fully processed."""
        row = self.conn.execute(
            """
            SELECT completed
            FROM source_file_progress
            WHERE source_path = ?
            LIMIT 1
            """,
            (str(source_path),),
        ).fetchone()
        return bool(row and int(row[0] or 0))

    def mark_source_batch_completed(self, source_path: Path, batch_index: int) -> None:
        """Persist that one source batch was fully processed and committed."""
        self.conn.execute(
            """
            INSERT INTO source_file_progress (
                source_path,
                last_completed_batch_index,
                completed,
                updated_at_utc
            )
            VALUES (?, ?, 0, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                last_completed_batch_index = excluded.last_completed_batch_index,
                completed = 0,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                str(source_path),
                int(batch_index),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def mark_source_file_completed(self, source_path: Path, batch_index: int) -> None:
        """Persist that an entire source parquet file finished processing."""
        self.conn.execute(
            """
            INSERT INTO source_file_progress (
                source_path,
                last_completed_batch_index,
                completed,
                updated_at_utc
            )
            VALUES (?, ?, 1, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                last_completed_batch_index = excluded.last_completed_batch_index,
                completed = 1,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                str(source_path),
                int(batch_index),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def fetch_cohort_statistics(self) -> dict[str, dict[str, int]]:
        """Return per-cohort unique counts from sqlite state."""
        rows = self.conn.execute(
            """
            WITH cohorts AS (
                SELECT cohort FROM retained_prs
                UNION
                SELECT cohort FROM seen_prs
                UNION
                SELECT cohort FROM unique_repositories
                UNION
                SELECT cohort FROM unique_languages
            )
            SELECT
                cohorts.cohort,
                COALESCE(prs.pr_count, 0),
                COALESCE(repos.repository_count, 0),
                COALESCE(langs.language_count, 0)
            FROM cohorts
            LEFT JOIN (
                SELECT cohort, COUNT(*) AS pr_count
                FROM retained_prs
                GROUP BY cohort
            ) prs ON prs.cohort = cohorts.cohort
            LEFT JOIN (
                SELECT cohort, COUNT(*) AS repository_count
                FROM unique_repositories
                GROUP BY cohort
            ) repos ON repos.cohort = cohorts.cohort
            LEFT JOIN (
                SELECT cohort, COUNT(*) AS language_count
                FROM unique_languages
                GROUP BY cohort
            ) langs ON langs.cohort = cohorts.cohort
            ORDER BY cohorts.cohort
            """
        ).fetchall()
        return {
            str(cohort): {
                "pr_count": int(pr_count),
                "repository_count": int(repository_count),
                "language_count": int(language_count),
            }
            for cohort, pr_count, repository_count, language_count in rows
        }

    def fetch_total_statistics(self) -> dict[str, int]:
        """Return true global totals across all cohorts."""
        pr_count = self.conn.execute("SELECT COUNT(*) FROM retained_prs").fetchone()[0]
        repository_count = self.conn.execute(
            "SELECT COUNT(DISTINCT repository_key) FROM unique_repositories"
        ).fetchone()[0]
        language_count = self.conn.execute(
            "SELECT COUNT(DISTINCT language) FROM unique_languages"
        ).fetchone()[0]
        return {
            "pr_count": int(pr_count or 0),
            "repository_count": int(repository_count or 0),
            "language_count": int(language_count or 0),
        }

    def mark_uploaded(self, repo_path: str, local_path: Path) -> None:
        """Persist successful upload state."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO uploaded_files (repo_path, local_path, uploaded_at_utc)
            VALUES (?, ?, ?)
            """,
            (repo_path, str(local_path), datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the sqlite connection."""
        self.conn.close()

    def commit(self) -> None:
        """Commit pending sqlite work."""
        self.conn.commit()


# ---------------------------------------------------------------------------
# Streaming pipeline and Hugging Face upload orchestration
# ---------------------------------------------------------------------------

class HFDatasetUploadPipeline:
    """Stream parquet inputs, standardize them, and upload as a dataset."""

    def __init__(
        self,
        *,
        local_directories: list[str],
        target_huggingface_repo_id: str,
        huggingface_token: str,
        local_output_dir: Path,
        standardized_data_subdir: str,
        output_batch_size: int,
        max_files_per_directory: int,
        parquet_compression: str,
        upload_max_retries: int,
        upload_retry_base_seconds: float,
        upload_short_term_rate_limit_window_seconds: float,
        upload_hourly_rate_limit_delay_seconds: float,
        upload_consecutive_failure_threshold: int,
        upload_consecutive_failure_delay_seconds: float,
        upload_large_folder_num_workers: int,
        upload_large_folder_directory_cooldown_seconds: float,
        state_db_filename: str,
        standardized_schema_version: str,
    ) -> None:
        """Initialize local staging paths, retry policy, buffers, and run state."""
        self.local_directories = local_directories
        self.target_huggingface_repo_id = target_huggingface_repo_id
        self.huggingface_token = huggingface_token
        self.local_output_dir = local_output_dir
        self.standardized_data_subdir = standardized_data_subdir
        self.output_batch_size = max(1, int(output_batch_size))
        self.max_files_per_directory = max(1, int(max_files_per_directory))
        self.parquet_compression = parquet_compression
        self.upload_max_retries = max(1, int(upload_max_retries))
        self.upload_retry_base_seconds = max(1.0, float(upload_retry_base_seconds))
        self.upload_short_term_rate_limit_window_seconds = max(
            60.0, float(upload_short_term_rate_limit_window_seconds)
        )
        self.upload_hourly_rate_limit_delay_seconds = max(
            self.upload_short_term_rate_limit_window_seconds,
            float(upload_hourly_rate_limit_delay_seconds),
        )
        self.upload_consecutive_failure_threshold = max(
            1, int(upload_consecutive_failure_threshold)
        )
        self.upload_consecutive_failure_delay_seconds = max(
            1.0, float(upload_consecutive_failure_delay_seconds)
        )
        self.upload_large_folder_num_workers = max(1, int(upload_large_folder_num_workers))
        self.upload_large_folder_directory_cooldown_seconds = max(
            0.0, float(upload_large_folder_directory_cooldown_seconds)
        )
        self.standardized_schema_version = standardized_schema_version
        self.output_entity_names = ENTITY_NAMES

        self.data_root = self.local_output_dir / self.standardized_data_subdir
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.state_store = UploadStateStore(self.local_output_dir / state_db_filename)
        self.buffers: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.batch_counts: dict[tuple[str, str], int] = self._load_existing_batch_counts()
        self.summary = {
            "source": "raw_extraction",
            "parquet_files_discovered": 0,
            "records_loaded": 0,
            "records_standardized": 0,
            "pull_request_rows_standardized": 0,
            "full_pull_request_rows_standardized": 0,
            "file_rows_standardized": 0,
            "repository_snapshot_rows_standardized": 0,
            "duplicates_skipped": 0,
            "parse_failures": 0,
            "uploaded_files": 0,
            "cohorts": {},
        }
        self._consecutive_upload_failures = 0

    def close(self) -> None:
        """Close the durable state store."""
        self.state_store.close()

    def _is_excluded_source_parquet(self, parquet_path: Path) -> bool:
        """Return True when a discovered parquet file is part of this pipeline's output tree."""
        output_dir_name = self.local_output_dir.name.strip()
        standardized_subdir_name = self.standardized_data_subdir.strip()
        if not output_dir_name or not standardized_subdir_name:
            return False

        normalized_path = Path(parquet_path)
        for root in self.local_directories:
            root_path = Path(root)
            try:
                relative_parts = normalized_path.relative_to(root_path).parts
            except Exception:
                continue
            if len(relative_parts) < 2:
                continue
            if (
                relative_parts[0].lower() == output_dir_name.lower()
                and relative_parts[1].lower() == standardized_subdir_name.lower()
            ):
                return True
        return False

    def _is_hourly_rate_limit_error(self, exc: Exception) -> bool:
        """Return True when the exception text indicates an hourly HF cap."""
        text = str(exc).lower()
        return (
            "rate limit reached" in text
            and "reset hourly" in text
        ) or "free usage limit" in text or (
            "repository commits" in text
            and "per hour" in text
        ) or "retry this action in about 1 hour" in text

    def _is_repository_commit_hourly_quota_error(self, exc: Exception) -> bool:
        """Return True for HF's repository-commit hourly quota message."""
        text = str(exc).lower()
        return (
            "repository commits" in text
            and "per hour" in text
        ) or "retry this action in about 1 hour" in text

    def _is_short_term_rate_limit_error(self, exc: Exception) -> bool:
        """Return True for non-hourly 429-style Hugging Face throttling."""
        text = str(exc).lower()
        if self._is_hourly_rate_limit_error(exc):
            return False
        return "429" in text or "too many requests" in text

    def _is_transient_hf_server_error(self, exc: Exception) -> bool:
        """Return True for transient 5xx Hugging Face API failures."""
        text = str(exc).lower()
        transient_markers = (
            "500 server error",
            "502 server error",
            "503 server error",
            "504 server error",
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
        )
        return any(marker in text for marker in transient_markers)

    def _is_five_minute_quota_error(self, exc: Exception) -> bool:
        """Return True for HF's explicit 5-minute request quota error."""
        text = str(exc).lower()
        return (
            "quota of 1000 api requests per 5 minutes period" in text
            or (
                "5 minutes period" in text
                and ("rate limit" in text or "quota" in text)
            )
        )

    def _compute_retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        """Compute retry delay based on HF rate limit guidance."""
        if self._is_repository_commit_hourly_quota_error(exc):
            return max(30 * 60.0, self.upload_hourly_rate_limit_delay_seconds)
        if self._is_hourly_rate_limit_error(exc):
            return self.upload_hourly_rate_limit_delay_seconds
        if self._is_five_minute_quota_error(exc):
            return max(self.upload_short_term_rate_limit_window_seconds, 5 * 60.0)
        if self._is_transient_hf_server_error(exc):
            exponential = self.upload_retry_base_seconds * (2 ** max(0, attempt - 1))
            return min(exponential, self.upload_short_term_rate_limit_window_seconds)
        if self._is_short_term_rate_limit_error(exc):
            exponential = self.upload_retry_base_seconds * (2 ** max(0, attempt - 1))
            return min(exponential, self.upload_short_term_rate_limit_window_seconds)
        if self._consecutive_upload_failures >= self.upload_consecutive_failure_threshold:
            return self.upload_consecutive_failure_delay_seconds
        return self.upload_retry_base_seconds * (2 ** max(0, attempt - 1))

    def _buffer_key(self, cohort: str, entity_name: str) -> tuple[str, str]:
        """Return the internal buffer key for one cohort/entity pair."""
        return (cohort, entity_name)

    def _entity_output_dir(self, cohort: str, entity_name: str) -> Path:
        """Return the output directory for one cohort/entity pair."""
        return self.data_root / cohort / entity_name

    def _entity_shard_dir(self, cohort: str, entity_name: str, batch_idx: int) -> Path:
        """Return the shard directory for one cohort/entity batch."""
        shard_index = max(0, int(batch_idx) - 1) // self.max_files_per_directory
        return self._entity_output_dir(cohort, entity_name) / f"shard-{shard_index:04d}"

    def _load_existing_batch_counts(self) -> dict[tuple[str, str], int]:
        """Infer next batch indices from already written parquet files."""
        counts: dict[tuple[str, str], int] = {}
        if not self.data_root.exists():
            return counts
        for cohort_dir in self.data_root.iterdir():
            if not cohort_dir.is_dir():
                continue
            for entity_name in ENTITY_NAMES:
                entity_dir = cohort_dir / entity_name
                if not entity_dir.is_dir():
                    continue
                max_idx = 0
                pattern = f"{ENTITY_FILE_PREFIXES[entity_name]}-*.parquet"
                for parquet_path in entity_dir.rglob(pattern):
                    suffix = parquet_path.stem.rsplit("-", 1)[-1]
                    try:
                        max_idx = max(max_idx, int(suffix))
                    except ValueError:
                        continue
                counts[(cohort_dir.name, entity_name)] = max_idx
        return counts

    def _cohort_summary(self, cohort: str) -> dict[str, int]:
        """Return/create a mutable summary bucket for the cohort."""
        cohorts = self.summary["cohorts"]
        if cohort not in cohorts:
            cohorts[cohort] = {
                "loaded": 0,
                "retained": 0,
                "pull_request_rows": 0,
                "full_pull_request_rows": 0,
                "file_rows": 0,
                "repository_snapshot_rows": 0,
                "duplicates_skipped": 0,
                "written_files": 0,
            }
        return cohorts[cohort]

    def _statistics_markdown_lines(
        self,
        cohort_stats: dict[str, dict[str, int]],
        total_stats: dict[str, int],
    ) -> list[str]:
        """Render README markdown lines for cohort and total statistics."""
        lines = [
            "| Cohort | Number of PRs | Number of Merged PRs | Unique Repositories | Sum of Additions | Sum of Deletions |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        if cohort_stats:
            for cohort, stats in cohort_stats.items():
                pr_count = int(stats.get("pr_count", 0))
                merged_pr_count = int(stats.get("merged_pr_count", 0))
                repository_count = int(stats.get("repository_count", 0))
                additions_sum = int(stats.get("additions_sum", 0))
                deletions_sum = int(stats.get("deletions_sum", 0))
                display_name = COHORT_DISPLAY_NAMES.get(cohort, cohort)
                lines.append(
                    f"| {display_name} | {pr_count} | {merged_pr_count} | {repository_count} | {additions_sum} | {deletions_sum} |"
                )
        else:
            lines.append("| none | 0 | 0 | 0 | 0 | 0 |")
        lines.append(
            "| **Total** | **{pr_count}** | **{merged_pr_count}** | **{repository_count}** | **{additions_sum}** | **{deletions_sum}** |".format(
                pr_count=int(total_stats.get("pr_count", 0)),
                merged_pr_count=int(total_stats.get("merged_pr_count", 0)),
                repository_count=int(total_stats.get("repository_count", 0)),
                additions_sum=int(total_stats.get("additions_sum", 0)),
                deletions_sum=int(total_stats.get("deletions_sum", 0)),
            )
        )
        lines.append("")
        return lines

    def _pull_request_output_statistics(self) -> dict[str, dict[str, int]]:
        """Aggregate additions, deletions, and merged counts from PR parquet output."""
        stats: dict[str, dict[str, int]] = {}
        if not self.data_root.exists():
            return stats
        for cohort_dir in sorted(self.data_root.iterdir()):
            if not cohort_dir.is_dir():
                continue
            pr_dir = cohort_dir / ENTITY_PULL_REQUESTS
            if not pr_dir.is_dir():
                continue
            cohort_stats = {
                "additions_sum": 0,
                "deletions_sum": 0,
                "merged_pr_count": 0,
            }
            for parquet_path in sorted(pr_dir.rglob("*.parquet")):
                table = pq.read_table(
                    parquet_path,
                    columns=["additions", "deletions", "merged_at"],
                )
                additions_array = table.column("additions").combine_chunks()
                deletions_array = table.column("deletions").combine_chunks()
                merged_at_array = table.column("merged_at").combine_chunks()

                additions_sum = pc.sum(additions_array).as_py()
                deletions_sum = pc.sum(deletions_array).as_py()
                merged_mask = pc.is_valid(merged_at_array)
                merged_pr_count = pc.sum(pc.cast(merged_mask, pa.int64())).as_py()

                cohort_stats["additions_sum"] += int(additions_sum or 0)
                cohort_stats["deletions_sum"] += int(deletions_sum or 0)
                cohort_stats["merged_pr_count"] += int(merged_pr_count or 0)
            stats[cohort_dir.name] = cohort_stats
        return stats

    def _merge_statistics(
        self,
        cohort_stats: dict[str, dict[str, int]],
        total_stats: dict[str, int],
    ) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        """Merge state-store statistics with PR-output aggregate metrics."""
        pr_output_stats = self._pull_request_output_statistics()
        merged_cohort_keys = sorted(set(cohort_stats) | set(pr_output_stats))
        merged_cohort_stats: dict[str, dict[str, int]] = {}
        for cohort in merged_cohort_keys:
            merged_cohort_stats[cohort] = {
                **cohort_stats.get(cohort, {}),
                **pr_output_stats.get(cohort, {}),
            }
            merged_cohort_stats[cohort].setdefault("pr_count", 0)
            merged_cohort_stats[cohort].setdefault("repository_count", 0)
            merged_cohort_stats[cohort].setdefault("additions_sum", 0)
            merged_cohort_stats[cohort].setdefault("deletions_sum", 0)
            merged_cohort_stats[cohort].setdefault("merged_pr_count", 0)
        merged_total_stats = dict(total_stats)
        merged_total_stats["additions_sum"] = sum(
            int(stats.get("additions_sum", 0)) for stats in merged_cohort_stats.values()
        )
        merged_total_stats["deletions_sum"] = sum(
            int(stats.get("deletions_sum", 0)) for stats in merged_cohort_stats.values()
        )
        merged_total_stats["merged_pr_count"] = sum(
            int(stats.get("merged_pr_count", 0)) for stats in merged_cohort_stats.values()
        )
        return merged_cohort_stats, merged_total_stats

    def _cohort_display_name(self, cohort: str) -> str:
        """Return the display label for one cohort."""
        return COHORT_DISPLAY_NAMES.get(cohort, cohort)

    def _dataset_usage_lines(
        self,
        *,
        cohorts: list[str],
        config_date_suffix: str,
    ) -> list[str]:
        """Render README usage examples mirroring the public HF format."""
        if cohorts:
            example_cohort = cohorts[0]
        else:
            example_cohort = "claude"
        example_display = self._cohort_display_name(example_cohort)
        repo_id = self.target_huggingface_repo_id
        lines = [
            "## Dataset Usage",
            "",
            f"Example loading by configuration for **{example_display}**. The same applies for the other cohorts with configuration names derived from the cohort and entity names.",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
        ]
        variable_names = {
            ENTITY_PULL_REQUESTS: "pull_request_records",
            ENTITY_FULL_PULL_REQUESTS: "aggregated_pull_requests",
            ENTITY_FILES: "file_change_records",
            ENTITY_REPOSITORY_SNAPSHOTS: "repository_records",
        }
        for entity_name in self.output_entity_names:
            lines.append(
                f"{example_cohort}_{variable_names[entity_name]} = load_dataset("
                f"'{repo_id}', '{example_cohort}_{ENTITY_CONFIG_SLUGS[entity_name]}_{config_date_suffix}', split='train')"
            )
        lines.extend(
            [
                "```",
                "",
                f"Example loading by data directory for **{example_display}**.",
                "",
                "```python",
                "from datasets import load_dataset",
                "",
            ]
        )
        for entity_name in self.output_entity_names:
            lines.append(
                f"{example_cohort}_{variable_names[entity_name]} = load_dataset("
                f"'{repo_id}', data_dir='data/{example_cohort}/{entity_name}', split='train')"
            )
        lines.extend(["```", ""])
        return lines

    def _config_date_suffix(self) -> str:
        """Return the UTC date suffix used in Hugging Face config names."""
        return datetime.now(timezone.utc).strftime("%d-%m-%Y")

    def _write_dataset_card(self) -> Path:
        """Write a dataset README describing the standardized format."""
        cohorts = sorted(self.summary["cohorts"].keys())
        cohort_stats = self.state_store.fetch_cohort_statistics()
        total_stats = self.state_store.fetch_total_statistics()
        cohort_stats, total_stats = self._merge_statistics(cohort_stats, total_stats)
        readme_path = self.local_output_dir / "README.md"
        config_date_suffix = self._config_date_suffix()
        header_lines: list[str] = ["---"]
        if cohorts:
            header_lines.append("configs:")
            for cohort in cohorts:
                for entity_name in self.output_entity_names:
                    header_lines.extend(
                        [
                            f"- config_name: {cohort}_{ENTITY_CONFIG_SLUGS[entity_name]}_{config_date_suffix}",
                            "  data_files:",
                            "  - split: train",
                            f"    path: data/{cohort}/{entity_name}/**/*.parquet",
                        ]
                    )
        header_lines.append("---")
        header_lines.append("")
        cohort_display_names = [self._cohort_display_name(cohort) for cohort in cohorts]
        if cohort_display_names:
            cohort_summary_text = ", ".join(f"**{name}**" for name in cohort_display_names)
        else:
            cohort_summary_text = "the configured cohorts"
        overview_suffix = (
            "repository snapshots, modified files, and a full standardized pull-request export."
            if ENTITY_FULL_PULL_REQUESTS in self.output_entity_names
            else "repository state snapshots and modified-file records."
        )
        lines = [
            "# Post-Processed Pull Request Dataset",
            "",
            "## Dataset Overview",
            "",
            (
                "The dataset contains a total of **{pr_count}** Pull Requests from "
                "{cohorts}. It also includes additional activity metadata such as "
                "{overview_suffix} A summary of the dataset is presented below."
            ).format(
                pr_count=int(total_stats.get("pr_count", 0)),
                cohorts=cohort_summary_text,
                overview_suffix=overview_suffix,
            ),
            "",
        ]
        lines.extend(self._statistics_markdown_lines(cohort_stats, total_stats))
        lines.extend(["## Dataset Structure", ""])
        structure_lines = {
            ENTITY_PULL_REQUESTS: "- **PullRequestRecords**: records the content, state, author metadata, repository references, timestamps, and summary activity counts for a pull request.",
            ENTITY_FULL_PULL_REQUESTS: "- **AggregatedPullRequests**: stores the standardized pull request row together with standardized nested repository and file payloads for workflows that need a self-contained PR record.",
            ENTITY_REPOSITORY_SNAPSHOTS: "- **RepositoryRecords**: stores repository ownership, visibility, status flags, popularity metrics, programming languages, topics, licensing, timestamps, and descriptive information for the base/head repository context of a pull request.",
            ENTITY_FILES: "- **FileChangeRecords**: captures one changed file per row, including its path, additions, deletions, content URLs, and patch-level metadata when available.",
        }
        for entity_name in self.output_entity_names:
            lines.append(structure_lines[entity_name])
        lines.extend(["", "## Entity Keys", ""])
        key_lines = {
            ENTITY_PULL_REQUESTS: "- `PullRequestRecords.pull_request_key`: primary key for pull request rows.",
            ENTITY_FULL_PULL_REQUESTS: "- `AggregatedPullRequests.pull_request_key`: primary key for expanded pull request rows.",
            ENTITY_FILES: "- `FileChangeRecords.file_key`: primary key for changed-file rows; `FileChangeRecords.pull_request_key` joins to `PullRequestRecords.pull_request_key`.",
            ENTITY_REPOSITORY_SNAPSHOTS: "- `RepositoryRecords.repository_snapshot_key`: primary key for repository records; `RepositoryRecords.pull_request_key` joins to `PullRequestRecords.pull_request_key`.",
        }
        for entity_name in self.output_entity_names:
            lines.append(key_lines[entity_name])
        if ENTITY_REPOSITORY_SNAPSHOTS in self.output_entity_names:
            lines.append(
                "- `PullRequestRecords.base_repository_snapshot_key` and `PullRequestRecords.head_repository_snapshot_key` reference `RepositoryRecords.repository_snapshot_key`."
            )
        lines.append("")
        lines.extend(self._dataset_usage_lines(cohorts=cohorts, config_date_suffix=config_date_suffix))
        lines.extend(
            [
                "## Provenance",
                "",
                f"- Generated at `{datetime.now(timezone.utc).isoformat()}`",
                "- Source type: `local`",
                f"- Export schema manifest version: `{self.standardized_schema_version}`",
            ]
        )
        readme_path.write_text("\n".join(header_lines + lines) + "\n", encoding="utf-8")
        return readme_path

    def _write_schema_manifest(self) -> Path:
        """Write a local schema manifest for inspection and upload."""
        manifest = {
            "schema_version": self.standardized_schema_version,
            "entities": {
                entity_name: {
                    "primary_key": ENTITY_PRIMARY_KEYS.get(entity_name),
                    "foreign_keys": ENTITY_FOREIGN_KEYS.get(entity_name, []),
                    "columns": [
                        {"name": field.name, "type": str(field.type)}
                        for field in schema
                    ]
                }
                for entity_name, schema in ENTITY_SCHEMAS.items()
                if entity_name in self.output_entity_names
            },
        }
        path = self.local_output_dir / "schema_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _public_record_for_entity_schema(
        self,
        entity_name: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one record containing only public schema fields with typed values."""
        public_record: dict[str, Any] = {}
        for field in ENTITY_SCHEMAS[entity_name]:
            value = record.get(field.name)
            if pa.types.is_timestamp(field.type):
                value = _parse_timestamp_value(value)
            public_record[field.name] = value
        return public_record

    def _flush_entity_buffer(self, cohort: str, entity_name: str) -> Optional[Path]:
        """Flush one cohort/entity buffer to a standardized parquet batch."""
        buffer_key = self._buffer_key(cohort, entity_name)
        records = self.buffers.get(buffer_key) or []
        if not records:
            return None
        batch_idx = self.batch_counts.get(buffer_key, 0) + 1
        self.batch_counts[buffer_key] = batch_idx
        entity_dir = self._entity_shard_dir(cohort, entity_name, batch_idx)
        entity_dir.mkdir(parents=True, exist_ok=True)
        file_path = entity_dir / f"{ENTITY_FILE_PREFIXES[entity_name]}-{batch_idx:06d}.parquet"
        public_records = [
            self._public_record_for_entity_schema(entity_name, record)
            for record in records
        ]
        table = pa.Table.from_pylist(public_records, schema=ENTITY_SCHEMAS[entity_name])
        pq.write_table(table, file_path, compression=self.parquet_compression)
        self.buffers[buffer_key] = []
        cohort_summary = self._cohort_summary(cohort)
        cohort_summary["written_files"] += 1
        print(
            "[post-processing/upload-extraction-data] Wrote standardized parquet batch "
            f"{file_path} with {table.num_rows} rows."
        )
        return file_path

    def flush_all(self) -> list[Path]:
        """Flush all pending cohort buffers."""
        paths: list[Path] = []
        for cohort, entity_name in sorted(self.buffers.keys()):
            path = self._flush_entity_buffer(cohort, entity_name)
            if path is not None:
                paths.append(path)
        return paths

    def _standardized_repo_path(self, file_path: Path) -> str:
        """Map a local standardized parquet path to a dataset repo path."""
        relative = file_path.relative_to(self.local_output_dir).as_posix()
        return relative

    def _parquet_upload_batches(self) -> list[tuple[Path, list[Path]]]:
        """Group parquet outputs by their immediate containing directory for upload."""
        batches_by_directory: dict[Path, list[Path]] = defaultdict(list)
        for parquet_path in sorted(self.data_root.rglob("*.parquet")):
            batches_by_directory[parquet_path.parent].append(parquet_path)
        return [
            (directory, parquet_paths)
            for directory, parquet_paths in sorted(
                batches_by_directory.items(),
                key=lambda item: str(item[0]).lower(),
            )
            if parquet_paths
        ]

    def _parquet_upload_batches_for_cohort(
        self,
        cohort: str,
    ) -> list[tuple[Path, list[Path]]]:
        """Group parquet outputs for one cohort by their immediate containing directory."""
        normalized_cohort = _normalize_cohort(cohort)
        if not normalized_cohort:
            raise ValueError("Cohort must be a non-empty string.")
        cohort_root = self.data_root / normalized_cohort
        if not cohort_root.is_dir():
            available = sorted(
                path.name
                for path in self.data_root.iterdir()
                if path.is_dir()
            ) if self.data_root.is_dir() else []
            available_text = ", ".join(available) if available else "<none>"
            raise FileNotFoundError(
                f"Cohort directory not found under standardized data root: {cohort_root} "
                f"(available cohorts: {available_text})"
            )

        batches_by_directory: dict[Path, list[Path]] = defaultdict(list)
        for parquet_path in sorted(cohort_root.rglob("*.parquet")):
            batches_by_directory[parquet_path.parent].append(parquet_path)
        return [
            (directory, parquet_paths)
            for directory, parquet_paths in sorted(
                batches_by_directory.items(),
                key=lambda item: str(item[0]).lower(),
            )
            if parquet_paths
        ]

    def _matching_source_root(self, source_path: Path) -> Optional[Path]:
        """Return the configured local root that contains the source path."""
        resolved_source_path = source_path.resolve()
        matching_roots: list[Path] = []
        for root in self.local_directories:
            root_path = Path(root).resolve()
            try:
                resolved_source_path.relative_to(root_path)
                matching_roots.append(root_path)
            except ValueError:
                continue
        if not matching_roots:
            return None
        return max(matching_roots, key=lambda candidate: len(candidate.parts))

    def _infer_cohort_from_source_path(self, source_path: Path) -> str:
        """Infer cohort from ``data/<cohort>`` or the first directory below root."""
        root_path = self._matching_source_root(source_path)
        if root_path is not None:
            try:
                relative_parts = source_path.resolve().relative_to(root_path).parts
            except ValueError:
                relative_parts = ()
            if relative_parts:
                cohort_index = (
                    1
                    if len(relative_parts) > 1
                    and relative_parts[0].lower() == self.standardized_data_subdir.lower()
                    else 0
                )
                inferred = _normalize_cohort(relative_parts[cohort_index])
                if inferred:
                    return inferred
        return "unknown_cohort"

    def _cohort_directory_from_source_path(self, source_path: Path) -> Path:
        """Return the concrete cohort directory under the configured source root."""
        root_path = self._matching_source_root(source_path)
        if root_path is not None:
            try:
                relative_parts = source_path.resolve().relative_to(root_path).parts
            except ValueError:
                relative_parts = ()
            if relative_parts:
                cohort_index = (
                    1
                    if len(relative_parts) > 1
                    and relative_parts[0].lower() == self.standardized_data_subdir.lower()
                    else 0
                )
                return root_path.joinpath(*relative_parts[: cohort_index + 1])
        return source_path.parent

    def _log_upload_progress(self, completed: int, total: int, action: str) -> None:
        """Log x/total progress for Hugging Face uploads."""
        remaining = max(0, total - completed)
        print(
            "[post-processing/upload-extraction-data] HF upload progress "
            f"{completed}/{total} complete, {remaining} left: {action}"
        )

    def _source_row_key(self, source_file: Path, row_index: int) -> str:
        """Return a stable row-level key for resumable source processing."""
        return f"{source_file.resolve()}::{int(row_index)}"

    def _upload_with_retry(self, *, local_path: Path, repo_path: str) -> None:
        """Upload one file to Hugging Face with exponential retries and cooldown."""
        api = HfApi(token=self.huggingface_token)
        attempts = 0
        while True:
            attempts += 1
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=self.target_huggingface_repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload {repo_path}",
                )
                self._consecutive_upload_failures = 0
                self.state_store.mark_uploaded(repo_path, local_path)
                self.summary["uploaded_files"] += 1
                print(f"[post-processing/upload-extraction-data] Uploaded {repo_path}")
                return
            except Exception as exc:
                self._consecutive_upload_failures += 1
                print(
                    "[post-processing/upload-extraction-data] Upload failed "
                    f"(attempt={attempts}, consecutive_failures={self._consecutive_upload_failures}) "
                    f"for {repo_path}: {exc}"
                )
                if attempts >= self.upload_max_retries:
                    raise
                delay_seconds = self._compute_retry_delay_seconds(exc, attempts)
                if self._is_hourly_rate_limit_error(exc):
                    print(
                        "[post-processing/upload-extraction-data] Hourly Hugging Face rate limit detected; "
                        f"sleeping {delay_seconds} seconds before retry."
                    )
                elif self._is_five_minute_quota_error(exc):
                    print(
                        "[post-processing/upload-extraction-data] Hugging Face 5-minute API quota detected; "
                        f"sleeping {delay_seconds} seconds before retry."
                    )
                elif self._is_short_term_rate_limit_error(exc):
                    print(
                        "[post-processing/upload-extraction-data] Short-term Hugging Face rate limit detected; "
                        f"sleeping {delay_seconds} seconds before retry."
                    )
                elif self._consecutive_upload_failures >= self.upload_consecutive_failure_threshold:
                    print(
                        "[post-processing/upload-extraction-data] Consecutive upload failures reached "
                        f"{self._consecutive_upload_failures}; sleeping "
                        f"{delay_seconds} seconds before retry."
                    )
                time.sleep(delay_seconds)

    def _should_upload(self) -> bool:
        """Return True when Hugging Face upload configuration is usable."""
        token = (self.huggingface_token or "").strip()
        repo_id = (self.target_huggingface_repo_id or "").strip()
        return bool(token and repo_id and "your-" not in repo_id and token != "hf")

    def _upload_candidate_paths(self) -> list[Path]:
        """Return metadata and parquet files that should be published."""
        metadata_paths = [
            self._write_dataset_card(),
            self._write_schema_manifest(),
        ]
        run_manifest_path = self.local_output_dir / "upload_extraction_manifest.json"
        if run_manifest_path.exists():
            metadata_paths.append(run_manifest_path)
        parquet_paths = sorted(self.data_root.rglob("*.parquet"))
        return [*metadata_paths, *parquet_paths]

    def build_upload_plan(self) -> dict[str, Any]:
        """Build a local upload plan without contacting Hugging Face."""
        files: list[dict[str, Any]] = []
        total_bytes = 0
        for local_path in self._upload_candidate_paths():
            size_bytes = int(local_path.stat().st_size) if local_path.exists() else 0
            total_bytes += size_bytes
            files.append(
                {
                    "local_path": str(local_path),
                    "repo_path": self._standardized_repo_path(local_path),
                    "size_bytes": size_bytes,
                }
            )
        return {
            "target_huggingface_repo_id": self.target_huggingface_repo_id,
            "repo_type": "dataset",
            "local_output_dir": str(self.local_output_dir),
            "data_root": str(self.data_root),
            "file_count": len(files),
            "total_size_bytes": total_bytes,
            "files": files,
        }

    def write_upload_plan(self) -> Path:
        """Persist the current upload plan beside the staged parquet outputs."""
        plan_path = self.local_output_dir / "upload_extraction_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(self.build_upload_plan(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return plan_path

    def write_run_manifest(
        self,
        *,
        settings: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Write a safe manifest for the local preparation/upload run."""
        manifest_path = self.local_output_dir / "upload_extraction_manifest.json"
        payload = {
            "manifest_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "settings": settings or {},
            "entities": list(ENTITY_NAMES),
            "data_root": str(self.data_root),
            "summary": self.summary,
            "upload_plan": {
                key: value
                for key, value in self.build_upload_plan().items()
                if key != "files"
            },
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path

    def upload_outputs(self, *, dry_run: bool = False) -> None:
        """Upload standardized parquet files and dataset metadata to Hugging Face."""
        candidate_paths = self._upload_candidate_paths()
        metadata_paths = [
            path for path in candidate_paths if path.suffix.lower() != ".parquet"
        ]
        parquet_paths = sorted(self.data_root.rglob("*.parquet"))
        if dry_run:
            plan_path = self.write_upload_plan()
            print(
                "[post-processing/upload-extraction-data] Dry run enabled; "
                f"wrote upload plan to {plan_path}."
            )
            return
        if not self._should_upload():
            print("[post-processing/upload-extraction-data] Hugging Face upload skipped; token/repo not configured.")
            return

        api = HfApi(token=self.huggingface_token)
        api.create_repo(
            repo_id=self.target_huggingface_repo_id,
            repo_type="dataset",
            exist_ok=True,
        )
        print(
            "[post-processing/upload-extraction-data] Skipping remote pre-upload cleanup; "
            "existing remote data and metadata will be left in place."
        )

        parquet_upload_batches = self._parquet_upload_batches()
        total_upload_files = len(candidate_paths)
        self._log_upload_progress(
            0,
            total_upload_files,
            (
                "starting shard-scoped resumable uploads "
                f"from {self.local_output_dir} "
                f"(HF cache: {self.local_output_dir / '.cache' / '.huggingface'}, "
                f"workers={self.upload_large_folder_num_workers})"
            ),
        )
        completed_upload_files = 0
        for metadata_path in metadata_paths:
            repo_path = self._standardized_repo_path(metadata_path)
            self._upload_with_retry(local_path=metadata_path, repo_path=repo_path)
            completed_upload_files += 1
            self._log_upload_progress(
                completed_upload_files,
                total_upload_files,
                f"uploaded metadata file {repo_path}",
            )

        for index, (batch_directory, batch_parquet_paths) in enumerate(
            parquet_upload_batches,
            start=1,
        ):
            relative_directory = batch_directory.relative_to(self.local_output_dir).as_posix()
            allow_pattern = f"{relative_directory}/*"
            batch_file_count = len(batch_parquet_paths)
            self._upload_large_folder_pattern_with_retry(
                allow_patterns=[allow_pattern],
                progress_label=(
                    f"parquet directory {index}/{len(parquet_upload_batches)} "
                    f"({relative_directory})"
                ),
            )
            for local_path in batch_parquet_paths:
                repo_path = self._standardized_repo_path(local_path)
                self.state_store.mark_uploaded(repo_path, local_path)
            completed_upload_files += batch_file_count
            self._log_upload_progress(
                completed_upload_files,
                total_upload_files,
                f"completed upload for {relative_directory}",
            )
            if (
                index < len(parquet_upload_batches)
                and self.upload_large_folder_directory_cooldown_seconds > 0
            ):
                print(
                    "[post-processing/upload-extraction-data] Cooling down between shard uploads for "
                    f"{self.upload_large_folder_directory_cooldown_seconds} seconds."
                )
                time.sleep(self.upload_large_folder_directory_cooldown_seconds)

        self.summary["uploaded_files"] = total_upload_files
        self._log_upload_progress(
            total_upload_files,
            total_upload_files,
            "completed shard-scoped resumable uploads",
        )

    def prepare_outputs(self) -> dict[str, Any]:
        """Run raw extraction standardization and write local metadata."""
        self._run_raw_extraction_input()
        self.flush_all()
        self.state_store.commit()
        self._write_dataset_card()
        self._write_schema_manifest()
        return self.summary

    def _upload_large_folder_pattern_with_retry(
        self,
        *,
        allow_patterns: list[str],
        progress_label: str,
    ) -> None:
        """Upload a subset of the output tree with large-folder retries."""
        api = HfApi(token=self.huggingface_token)
        attempts = 0
        while True:
            attempts += 1
            try:
                api.upload_large_folder(
                    repo_id=self.target_huggingface_repo_id,
                    folder_path=str(self.local_output_dir),
                    repo_type="dataset",
                    allow_patterns=allow_patterns,
                    num_workers=self.upload_large_folder_num_workers,
                    print_report=True,
                    print_report_every=60,
                )
                self._consecutive_upload_failures = 0
                return
            except Exception as exc:
                self._consecutive_upload_failures += 1
                print(
                    "[post-processing/upload-extraction-data] Large-folder upload failed "
                    f"(attempt={attempts}, consecutive_failures={self._consecutive_upload_failures}, "
                    f"scope={progress_label}): {exc}"
                )
                if attempts >= self.upload_max_retries:
                    raise
                delay_seconds = self._compute_retry_delay_seconds(exc, attempts)
                if self._is_hourly_rate_limit_error(exc):
                    print(
                        "[post-processing/upload-extraction-data] Hourly Hugging Face rate limit detected during "
                        f"large-folder upload for {progress_label}; "
                        f"sleeping {delay_seconds} seconds before retry."
                    )
                elif self._is_five_minute_quota_error(exc):
                    print(
                        "[post-processing/upload-extraction-data] Hugging Face 5-minute API quota detected during "
                        f"large-folder upload for {progress_label}; "
                        f"sleeping {delay_seconds} seconds before retry."
                    )
                elif self._is_short_term_rate_limit_error(exc):
                    print(
                        "[post-processing/upload-extraction-data] Short-term Hugging Face rate limit detected during "
                        f"large-folder upload for {progress_label}; "
                        f"sleeping {delay_seconds} seconds before retry."
                    )
                elif self._consecutive_upload_failures >= self.upload_consecutive_failure_threshold:
                    print(
                        "[post-processing/upload-extraction-data] Consecutive large-folder upload failures reached "
                        f"{self._consecutive_upload_failures} for {progress_label}; "
                        f"sleeping {delay_seconds} seconds before retry."
                    )
                time.sleep(delay_seconds)

    def _iter_source_cohort_directories(self) -> dict[str, list[Path]]:
        """Discover source cohort directories across all local roots."""
        discovered_parquet_files = iter_source_parquet_files(
            source_type="local",
            local_directories=self.local_directories,
        )
        parquet_files = [
            parquet_path
            for parquet_path in discovered_parquet_files
            if not self._is_excluded_source_parquet(parquet_path)
        ]
        excluded_count = len(discovered_parquet_files) - len(parquet_files)
        self.summary["parquet_files_discovered"] = len(parquet_files)
        print(
            "[post-processing/upload-extraction-data] Discovered "
            f"{len(parquet_files)} source parquet files."
        )
        if excluded_count:
            print(
                "[post-processing/upload-extraction-data] Excluded "
                f"{excluded_count} parquet files from the standardized output subtree."
            )
        cohorts: dict[str, set[Path]] = defaultdict(set)
        for parquet_path in sorted(parquet_files):
            cohort = self._infer_cohort_from_source_path(parquet_path)
            cohort_dir = self._cohort_directory_from_source_path(parquet_path)
            cohorts[cohort].add(cohort_dir)
        return {
            cohort: sorted(cohort_dirs)
            for cohort, cohort_dirs in sorted(cohorts.items())
        }

    def _flush_and_commit(self, cohort: Optional[str] = None) -> None:
        """Persist buffered output rows and sqlite state."""
        if cohort is None:
            self.flush_all()
        else:
            for entity_name in ENTITY_NAMES:
                self._flush_entity_buffer(cohort, entity_name)
        self.state_store.commit()

    def _deduplicate_public_records(
        self,
        *,
        cohort: str,
        entity_name: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply global stable PR deduplication to public entity records."""
        processed_records: list[dict[str, Any]] = []
        if entity_name == ENTITY_PULL_REQUESTS:
            for record in records:
                stable_key = stable_pr_dedup_key_for_record(record)
                if stable_key:
                    if not self.state_store.record_stable_pr_if_new(
                        stable_pr_key=stable_key,
                        cohort=cohort,
                    ):
                        continue
                    record["dedup_key"] = stable_key
                pull_request_key = _pull_request_key_from_record(record)
                if pull_request_key:
                    self.state_store.record_retained_pull_request_key(
                        pull_request_key=pull_request_key,
                        cohort=cohort,
                        stable_pr_key=stable_key,
                    )
                processed_records.append(record)
            return processed_records

        for record in records:
            pull_request_key = _pull_request_key_from_record(record)
            if not pull_request_key:
                continue
            if not self.state_store.is_retained_pull_request_key(pull_request_key):
                continue
            if entity_name == ENTITY_FULL_PULL_REQUESTS:
                stable_key = self.state_store.stable_pr_key_for_retained_pull_request_key(
                    pull_request_key
                )
                if stable_key:
                    record["dedup_key"] = stable_key
            processed_records.append(record)
        return processed_records

    def _process_public_records(
        self,
        *,
        cohort: str,
        entity_name: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply deduplication before public entity records are written."""
        return self._deduplicate_public_records(
            cohort=cohort,
            entity_name=entity_name,
            records=records,
        )

    def _buffer_processed_standardized_records(
        self,
        cohort: str,
        entity_name: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Process standardized records, buffer them, and return the records written."""
        processed_records = self._process_public_records(
            cohort=cohort,
            entity_name=entity_name,
            records=records,
        )
        self._buffer_records(cohort, entity_name, processed_records)
        return processed_records

    def _buffer_records(
        self,
        cohort: str,
        entity_name: str,
        records: list[dict[str, Any]],
    ) -> None:
        """Append entity rows into the in-memory output buffer."""
        if not records:
            return
        buffer_key = self._buffer_key(cohort, entity_name)
        self.buffers.setdefault(buffer_key, []).extend(records)

    def _cohort_buffer_reached_flush_threshold(self, cohort: str) -> bool:
        """Return True when any cohort/entity buffer reached the flush threshold."""
        return any(
            len(self.buffers.get(self._buffer_key(cohort, entity_name), []))
            >= self.output_batch_size
            for entity_name in ENTITY_NAMES
        )

    def _process_parquet_file(self, parquet_path: Path) -> None:
        """Stream one parquet file and buffer standardized deduplicated rows."""
        if self.state_store.is_source_file_completed(parquet_path):
            print(f"[post-processing/upload-extraction-data] Skipping completed source file {parquet_path}")
            return

        cohort = self._infer_cohort_from_source_path(parquet_path)
        parquet_file = pq.ParquetFile(parquet_path)
        last_completed_batch_index = self.state_store.last_completed_batch_index(parquet_path)
        absolute_row_index = last_completed_batch_index * self.output_batch_size
        processed_batch_index = last_completed_batch_index
        print(f"[post-processing/upload-extraction-data] Reading {parquet_path}")
        for batch_index, batch in enumerate(
            parquet_file.iter_batches(batch_size=self.output_batch_size),
            start=1,
        ):
            if batch_index <= last_completed_batch_index:
                absolute_row_index += batch.num_rows
                continue
            rows = batch.to_pylist()
            for row in rows:
                source_row_key = self._source_row_key(parquet_path, absolute_row_index)
                absolute_row_index += 1
                self.summary["records_loaded"] += 1
                try:
                    pr = coerce_pull_request_record(row)
                except Exception as exc:
                    self.summary["parse_failures"] += 1
                    print(
                        "[post-processing/upload-extraction-data] Failed to parse row from "
                        f"{parquet_path}: {exc}"
                    )
                    continue
                cohort_summary = self._cohort_summary(cohort)
                cohort_summary["loaded"] += 1
                if self.state_store.is_retained_pr_source_row(source_row_key):
                    self.summary["duplicates_skipped"] += 1
                    cohort_summary["duplicates_skipped"] += 1
                    continue

                pull_request_key = pull_request_key_for_record(pr, None)
                repository_snapshot_records, base_repository_snapshot_key, head_repository_snapshot_key = (
                    standardize_repository_snapshot_records(
                        pr,
                        raw_payload=row if isinstance(row, dict) else None,
                        cohort=cohort,
                        pull_request_key=pull_request_key,
                        schema_version=self.standardized_schema_version,
                        source_file=parquet_path,
                    )
                )
                stable_dedup_key = stable_pr_dedup_key(
                    base_repository_snapshot_key=base_repository_snapshot_key,
                    pull_request_number=_get_attr(pr, "number"),
                )
                pull_request_record = standardize_pull_request_record(
                    pr,
                    raw_payload=row if isinstance(row, dict) else None,
                    cohort=cohort,
                    dedup_key=stable_dedup_key,
                    pull_request_key=pull_request_key,
                    base_repository_snapshot_key=base_repository_snapshot_key,
                    head_repository_snapshot_key=head_repository_snapshot_key,
                    schema_version=self.standardized_schema_version,
                    source_file=parquet_path,
                )
                full_pull_request_record = standardize_full_pull_request_record(
                    pr,
                    raw_payload=row if isinstance(row, dict) else None,
                    cohort=cohort,
                    dedup_key=stable_dedup_key,
                    pull_request_key=pull_request_key,
                    base_repository_snapshot_key=base_repository_snapshot_key,
                    head_repository_snapshot_key=head_repository_snapshot_key,
                    schema_version=self.standardized_schema_version,
                    source_file=parquet_path,
                )
                file_records = standardize_file_records(
                    pr,
                    raw_payload=row if isinstance(row, dict) else None,
                    cohort=cohort,
                    pull_request_key=pull_request_key,
                    schema_version=self.standardized_schema_version,
                    source_file=parquet_path,
                )
                processed_pull_request_records = self._buffer_processed_standardized_records(
                    cohort,
                    ENTITY_PULL_REQUESTS,
                    [pull_request_record],
                )
                if not processed_pull_request_records:
                    self.summary["duplicates_skipped"] += 1
                    cohort_summary["duplicates_skipped"] += 1
                    continue
                self.state_store.record_retained_pr_if_new(source_row_key, cohort)
                for repository_key in repository_keys_for_pr(
                    pr,
                    raw_payload=row if isinstance(row, dict) else None,
                ):
                    self.state_store.record_repository_if_new(cohort, repository_key)
                for language in languages_for_pr(pr):
                    self.state_store.record_language_if_new(cohort, language)
                processed_full_pull_request_records = self._buffer_processed_standardized_records(
                    cohort,
                    ENTITY_FULL_PULL_REQUESTS,
                    [full_pull_request_record],
                )
                processed_file_records = self._buffer_processed_standardized_records(
                    cohort,
                    ENTITY_FILES,
                    file_records,
                )
                processed_repository_snapshot_records = self._buffer_processed_standardized_records(
                    cohort,
                    ENTITY_REPOSITORY_SNAPSHOTS,
                    repository_snapshot_records,
                )
                self.summary["records_standardized"] += (
                    len(processed_pull_request_records)
                    + len(processed_full_pull_request_records)
                    + len(processed_file_records)
                    + len(processed_repository_snapshot_records)
                )
                self.summary["pull_request_rows_standardized"] += len(
                    processed_pull_request_records
                )
                self.summary["full_pull_request_rows_standardized"] += len(
                    processed_full_pull_request_records
                )
                self.summary["file_rows_standardized"] += len(processed_file_records)
                self.summary["repository_snapshot_rows_standardized"] += len(
                    processed_repository_snapshot_records
                )
                cohort_summary["retained"] += len(processed_pull_request_records)
                cohort_summary["pull_request_rows"] += len(processed_pull_request_records)
                cohort_summary["full_pull_request_rows"] += len(
                    processed_full_pull_request_records
                )
                cohort_summary["file_rows"] += len(processed_file_records)
                cohort_summary["repository_snapshot_rows"] += len(
                    processed_repository_snapshot_records
                )
                if self._cohort_buffer_reached_flush_threshold(cohort):
                    self._flush_and_commit(cohort)
            self.state_store.commit()
            self.state_store.mark_source_batch_completed(parquet_path, batch_index)
            self.state_store.commit()
            processed_batch_index = batch_index
        self._flush_and_commit()
        self.state_store.mark_source_file_completed(parquet_path, processed_batch_index)
        self.state_store.commit()

    def _process_cohort_directories(self, cohort: str, cohort_directories: list[Path]) -> None:
        """Process all source cohort directories for one cohort before moving on."""
        parquet_paths = [
            parquet_path
            for cohort_dir in cohort_directories
            for parquet_path in sorted(cohort_dir.rglob("*.parquet"))
        ]
        print(
            "[post-processing/upload-extraction-data] Processing cohort "
            f"{cohort} from {len(cohort_directories)} directories and {len(parquet_paths)} parquet files."
        )
        for parquet_path in parquet_paths:
            self._process_parquet_file(parquet_path)
        self._flush_and_commit(cohort)

    def _run_raw_extraction_input(self) -> None:
        """Run the current raw extraction parquet standardization flow."""
        cohort_directories_by_cohort = self._iter_source_cohort_directories()
        total_cohorts = len(cohort_directories_by_cohort)
        completed_cohorts = 0
        print(
            "[post-processing/upload-extraction-data] Cohort progress "
            f"{completed_cohorts}/{total_cohorts} complete, {total_cohorts} left."
        )
        for cohort, cohort_directories in sorted(
            cohort_directories_by_cohort.items(),
            key=lambda item: _cohort_processing_priority(item[0]),
        ):
            remaining_before = max(0, total_cohorts - completed_cohorts)
            print(
                "[post-processing/upload-extraction-data] Cohort progress "
                f"{completed_cohorts}/{total_cohorts} complete, "
                f"{remaining_before} left: processing {cohort}"
            )
            self._process_cohort_directories(cohort, cohort_directories)
            completed_cohorts += 1
            remaining_after = max(0, total_cohorts - completed_cohorts)
            print(
                "[post-processing/upload-extraction-data] Cohort progress "
                f"{completed_cohorts}/{total_cohorts} complete, "
                f"{remaining_after} left: completed {cohort}"
            )

    def run(self) -> dict[str, Any]:
        """Run the full configured processing + upload flow."""
        try:
            self.prepare_outputs()
            self.upload_outputs()
            return self.summary
        finally:
            self.close()
