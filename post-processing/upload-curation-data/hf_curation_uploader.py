"""Curation-specific parquet export and Hugging Face upload pipeline.

The active stage boundary is deliberately narrow: read local curation output,
normalize it into public parquet entity tables, write local manifests, and
optionally upload the prepared directory to a Hugging Face dataset repository.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

POST_PROCESSING_DIR = Path(__file__).resolve().parents[1]
ROOT = POST_PROCESSING_DIR.parent
UTILITY_DIR = POST_PROCESSING_DIR / "utility"
for candidate in (ROOT, UTILITY_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cohort_discovery import discover_cohort_dirs, safe_cohort_name
from curation.config.code_smell_standardization_config import canonicalize_code_smell_type
from curation.config.code_smell_taxonomy_config import classify_code_smell_taxonomy
from curation.config.refactoring_standardization_config import canonicalize_refactoring_type
from curation.config.refactoring_taxonomy_config import classify_refactoring_taxonomy
from curation_outputs import candidate_curation_roots, discover_aggregate_metric_paths
from json_io import read_json_object
from refactoring_outputs import (
    original_pr_refactoring_operations,
    resolve_curation_artifact_path,
)
from repository_keys import (
    normalize_repository_key,
    repository_key_from_full_name,
    repository_key_from_safe_key,
)


# Public entity names are the dataset contract. They are reused in directory
# layout, schema manifests, deterministic ids, and generated dataset-card YAML.
ENTITY_CURATION_PULL_REQUESTS = "CuratedPullRequests"
ENTITY_REFACTORING_RESULTS = "RefactoringResults"
ENTITY_MAINTAINABILITY_RESULTS = "MaintainabilityResults"
ENTITY_CODE_SMELL_RESULTS = "CodeSmellResults"
ENTITY_REPOSITORY_ADDITIONAL_METADATA = "RepositoryMetadata"
ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS = "RepositoryTopicClassifications"
ENTITY_SNAPSHOT_MANIFESTS = "SnapshotManifests"
ENTITY_SNAPSHOT_FILE_REFS = "SnapshotFileRefs"
ENTITY_SNAPSHOT_FILE_BLOBS = "SnapshotFileBlobs"

ENTITY_NAMES: tuple[str, ...] = (
    ENTITY_CURATION_PULL_REQUESTS,
    ENTITY_REFACTORING_RESULTS,
    ENTITY_MAINTAINABILITY_RESULTS,
    ENTITY_CODE_SMELL_RESULTS,
    ENTITY_REPOSITORY_ADDITIONAL_METADATA,
    ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS,
    ENTITY_SNAPSHOT_MANIFESTS,
    ENTITY_SNAPSHOT_FILE_REFS,
    ENTITY_SNAPSHOT_FILE_BLOBS,
)
ROOT_CURATION_ENTITY_NAMES: tuple[str, ...] = (
    ENTITY_REPOSITORY_ADDITIONAL_METADATA,
    ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS,
)
COHORT_CURATION_ENTITY_NAMES: tuple[str, ...] = tuple(
    entity_name for entity_name in ENTITY_NAMES if entity_name not in ROOT_CURATION_ENTITY_NAMES
)
ROOT_CURATION_SCOPE = "__root__"

ENTITY_FILE_PREFIXES: dict[str, str] = {
    ENTITY_CURATION_PULL_REQUESTS: "curated_pull_requests_batch",
    ENTITY_REFACTORING_RESULTS: "refactoring_results_batch",
    ENTITY_MAINTAINABILITY_RESULTS: "maintainability_results_batch",
    ENTITY_CODE_SMELL_RESULTS: "code_smell_results_batch",
    ENTITY_REPOSITORY_ADDITIONAL_METADATA: "repository_metadata_batch",
    ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS: "repository_topic_classifications_batch",
    ENTITY_SNAPSHOT_MANIFESTS: "snapshot_manifests_batch",
    ENTITY_SNAPSHOT_FILE_REFS: "snapshot_file_refs_batch",
    ENTITY_SNAPSHOT_FILE_BLOBS: "snapshot_file_blobs_batch",
}

ENTITY_CONFIG_SLUGS: dict[str, str] = {
    ENTITY_CURATION_PULL_REQUESTS: "curated_pull_requests",
    ENTITY_REFACTORING_RESULTS: "refactoring_results",
    ENTITY_MAINTAINABILITY_RESULTS: "maintainability_results",
    ENTITY_CODE_SMELL_RESULTS: "code_smell_results",
    ENTITY_REPOSITORY_ADDITIONAL_METADATA: "repository_metadata",
    ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS: "repository_topic_classifications",
    ENTITY_SNAPSHOT_MANIFESTS: "snapshot_manifests",
    ENTITY_SNAPSHOT_FILE_REFS: "snapshot_file_refs",
    ENTITY_SNAPSHOT_FILE_BLOBS: "snapshot_file_blobs",
}

ENTITY_PRIMARY_KEY_COLUMNS: dict[str, str] = {
    ENTITY_CURATION_PULL_REQUESTS: "curation_pull_request_record_id",
    ENTITY_REFACTORING_RESULTS: "refactoring_result_id",
    ENTITY_MAINTAINABILITY_RESULTS: "maintainability_result_id",
    ENTITY_CODE_SMELL_RESULTS: "code_smell_result_id",
    ENTITY_REPOSITORY_ADDITIONAL_METADATA: "repository_metadata_id",
    ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS: "repository_topic_classification_id",
    ENTITY_SNAPSHOT_MANIFESTS: "snapshot_manifest_id",
    ENTITY_SNAPSHOT_FILE_REFS: "snapshot_file_ref_id",
    ENTITY_SNAPSHOT_FILE_BLOBS: "snapshot_file_blob_id",
}
ENTITY_NATURAL_KEY_COLUMNS: dict[str, list[str]] = {
    ENTITY_CURATION_PULL_REQUESTS: ["base_repository_id", "pr_number"],
    ENTITY_REFACTORING_RESULTS: [
        "snapshot_manifest_id",
        "transition_label",
    ],
    ENTITY_MAINTAINABILITY_RESULTS: ["snapshot_manifest_id"],
    ENTITY_CODE_SMELL_RESULTS: ["snapshot_manifest_id"],
    ENTITY_REPOSITORY_ADDITIONAL_METADATA: ["repository_key"],
    ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS: ["repository_metadata_id"],
    ENTITY_SNAPSHOT_MANIFESTS: ["curation_pull_request_record_id", "snapshot_label"],
    ENTITY_SNAPSHOT_FILE_REFS: ["snapshot_manifest_id", "path"],
    ENTITY_SNAPSHOT_FILE_BLOBS: ["content_sha256"],
}
ENTITY_FOREIGN_KEYS: dict[str, list[dict[str, str]]] = {
    ENTITY_CURATION_PULL_REQUESTS: [
        {
            "column": "base_repository_metadata_id",
            "references_entity": ENTITY_REPOSITORY_ADDITIONAL_METADATA,
            "references_column": "repository_metadata_id",
        },
    ],
    ENTITY_REFACTORING_RESULTS: [
        {
            "column": "snapshot_manifest_id",
            "references_entity": ENTITY_SNAPSHOT_MANIFESTS,
            "references_column": "snapshot_manifest_id",
        },
    ],
    ENTITY_MAINTAINABILITY_RESULTS: [
        {
            "column": "snapshot_manifest_id",
            "references_entity": ENTITY_SNAPSHOT_MANIFESTS,
            "references_column": "snapshot_manifest_id",
        },
    ],
    ENTITY_CODE_SMELL_RESULTS: [
        {
            "column": "snapshot_manifest_id",
            "references_entity": ENTITY_SNAPSHOT_MANIFESTS,
            "references_column": "snapshot_manifest_id",
        },
    ],
    ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS: [
        {
            "column": "repository_metadata_id",
            "references_entity": ENTITY_REPOSITORY_ADDITIONAL_METADATA,
            "references_column": "repository_metadata_id",
        },
    ],
    ENTITY_SNAPSHOT_MANIFESTS: [
        {
            "column": "curation_pull_request_record_id",
            "references_entity": ENTITY_CURATION_PULL_REQUESTS,
            "references_column": "curation_pull_request_record_id",
        },
    ],
    ENTITY_SNAPSHOT_FILE_REFS: [
        {
            "column": "snapshot_manifest_id",
            "references_entity": ENTITY_SNAPSHOT_MANIFESTS,
            "references_column": "snapshot_manifest_id",
        },
        {
            "column": "snapshot_file_blob_id",
            "references_entity": ENTITY_SNAPSHOT_FILE_BLOBS,
            "references_column": "snapshot_file_blob_id",
        },
    ],
}

REFACTORING_RESULT_SOURCE_CURATION = "curation_aggregate"
REFACTORING_RESULT_SOURCE_LONGITUDINAL = "longitudinal_refactoring_analysis"
REFACTORING_INPUT_MODE_CURATION = "curation"
FUTURE_SNAPSHOT_LABELS: tuple[str, ...] = ("+3d", "+7d", "+31d", "+61d")
MANTYLA_CATEGORY_COUNT_COLUMNS: dict[str, str] = {
    "bloaters": "mantyla_bloaters_count",
    "object_orientation_abusers": "mantyla_object_orientation_abusers_count",
    "change_preventers": "mantyla_change_preventers_count",
    "dispensables": "mantyla_dispensables_count",
    "encapsulators": "mantyla_encapsulators_count",
    "couplers": "mantyla_couplers_count",
    "others": "mantyla_others_count",
    "unmapped": "mantyla_unmapped_count",
}
MURPHY_HILL_COUNT_COLUMNS: dict[str, str] = {
    "low": "murphyhill_low_count",
    "medium": "murphyhill_medium_count",
    "high": "murphyhill_high_count",
}
CURATION_DATASET_SUBDIR = "Curation"
README_SUMMARY_BEGIN = "<!-- BEGIN MOSAIC CURATION SUMMARY -->"
README_SUMMARY_END = "<!-- END MOSAIC CURATION SUMMARY -->"
README_TOPIC_SUMMARY_BEGIN = "<!-- BEGIN MOSAIC CURATION TOPIC SUMMARY -->"
README_TOPIC_SUMMARY_END = "<!-- END MOSAIC CURATION TOPIC SUMMARY -->"
README_LONGITUDINAL_REFACTORING_SUMMARY_BEGIN = (
    "<!-- BEGIN MOSAIC LONGITUDINAL REFACTORING SUMMARY -->"
)
README_LONGITUDINAL_REFACTORING_SUMMARY_END = "<!-- END MOSAIC LONGITUDINAL REFACTORING SUMMARY -->"
README_STRUCTURE_BEGIN = "<!-- BEGIN MOSAIC CURATION STRUCTURE -->"
README_STRUCTURE_END = "<!-- END MOSAIC CURATION STRUCTURE -->"
LONGITUDINAL_OPERATION_FIELDS: tuple[str, ...] = (
    "snapshot_label",
    "target_offset_days",
    "target_timestamp",
    "repository_observation_cutoff",
    "transition_start_commit",
    "snapshot_commit",
    "snapshot_result_status",
    "tool",
    "language",
    "operation_id",
    "commit_id",
    "raw_type",
    "standardized_type",
    "description",
    "murphy_hill_level",
    "taxonomy",
    "source_locations",
    "target_locations",
)


def _schema(fields: Iterable[tuple[str, pa.DataType]]) -> pa.Schema:
    """Build a PyArrow schema from compact ``(name, type)`` tuples."""
    return pa.schema([pa.field(name, data_type) for name, data_type in fields])


# Entity schemas are intentionally explicit. They define the public parquet
# columns and are also reflected into ``schema_manifest.json``.
CURATION_PULL_REQUESTS_SCHEMA = _schema(
    (
        ("curation_pull_request_record_id", pa.string()),
        ("pull_request_key", pa.string()),
        ("pr_number", pa.int64()),
        ("pr_url", pa.string()),
        ("source_run_id", pa.string()),
        ("cohort", pa.string()),
        ("author", pa.string()),
        ("authored_by_agent", pa.bool_()),
        ("author_agent", pa.string()),
        ("created_at", pa.timestamp("ms", tz="UTC")),
        ("merged_at", pa.timestamp("ms", tz="UTC")),
        ("closed_at", pa.timestamp("ms", tz="UTC")),
        ("base_repository_metadata_id", pa.string()),
        ("base_repository_id", pa.string()),
        ("base_repository_key", pa.string()),
        ("base_repository_url", pa.string()),
        ("head_repository_id", pa.string()),
        ("head_repository_key", pa.string()),
        ("head_repository_url", pa.string()),
        ("dominant_language", pa.string()),
        ("file_languages_json", pa.string()),
        ("sampling_language_bucket", pa.string()),
        ("sampling_time_bucket", pa.string()),
        ("sampling_popularity_bucket", pa.string()),
        ("longitudinal_selected", pa.bool_()),
        ("processing_status", pa.string()),
        ("additions", pa.int64()),
        ("deletions", pa.int64()),
        ("changed_files", pa.int64()),
    )
)

REFACTORING_RESULTS_SCHEMA = _schema(
    (
        ("refactoring_result_id", pa.string()),
        ("snapshot_manifest_id", pa.string()),
        ("transition_label", pa.string()),
        ("tool", pa.string()),
        ("result_status", pa.string()),
        ("skipped_tool_run", pa.bool_()),
        ("skip_reason", pa.string()),
        ("refop_retention_rate", pa.float64()),
        ("future_lines_touched", pa.int64()),
        ("touching_commit_count", pa.int64()),
        ("refop_count", pa.int64()),
        ("added_lines", pa.int64()),
        ("removed_lines", pa.int64()),
        ("magnitude_lines", pa.int64()),
        ("magnitude_files", pa.int64()),
        ("diversity", pa.int64()),
        ("murphyhill_low_count", pa.int64()),
        ("murphyhill_medium_count", pa.int64()),
        ("murphyhill_high_count", pa.int64()),
        ("operations_json", pa.string()),
        ("refactor_type_count_json", pa.string()),
        ("raw_refactoring_to_standardized_refactoring_json", pa.string()),
        ("standardized_refactoring_to_murphyhill_category_json", pa.string()),
    )
)

MAINTAINABILITY_RESULTS_SCHEMA = _schema(
    (
        ("maintainability_result_id", pa.string()),
        ("snapshot_manifest_id", pa.string()),
        ("loc", pa.float64()),
        ("cyclomatic_complexity", pa.float64()),
        ("maintainability_index", pa.float64()),
        ("duplicated_lines_density", pa.float64()),
        ("comment_ratio", pa.float64()),
        ("halstead_volume", pa.float64()),
        ("fan_out", pa.float64()),
    )
)

CODE_SMELL_RESULTS_SCHEMA = _schema(
    (
        ("code_smell_result_id", pa.string()),
        ("snapshot_manifest_id", pa.string()),
        ("result_status", pa.string()),
        ("smell_count", pa.int64()),
        ("mantyla_bloaters_count", pa.int64()),
        ("mantyla_object_orientation_abusers_count", pa.int64()),
        ("mantyla_change_preventers_count", pa.int64()),
        ("mantyla_dispensables_count", pa.int64()),
        ("mantyla_encapsulators_count", pa.int64()),
        ("mantyla_couplers_count", pa.int64()),
        ("mantyla_others_count", pa.int64()),
        ("mantyla_unmapped_count", pa.int64()),
        ("raw_smell_count_json", pa.string()),
        ("standardized_smell_type_count_json", pa.string()),
        ("raw_smell_to_standardized_smell_json", pa.string()),
        ("standardized_smell_to_mantyla_category_json", pa.string()),
        ("tools", pa.string()),
        ("skipped_tool_run", pa.bool_()),
        ("skip_reason", pa.string()),
    )
)

REPOSITORY_ADDITIONAL_METADATA_SCHEMA = _schema(
    (
        ("repository_metadata_id", pa.string()),
        ("repository_key", pa.string()),
        ("repository_owner", pa.string()),
        ("repository_name", pa.string()),
        ("popularity_label", pa.string()),
        ("stargazer_count", pa.int64()),
        ("labels", pa.string()),
        ("readme", pa.string()),
        ("readme_sha256", pa.string()),
        ("file_list_source_commit", pa.string()),
        ("file_list_generated_at", pa.timestamp("ms", tz="UTC")),
        ("file_count", pa.int64()),
        ("file_list", pa.string()),
    )
)

REPOSITORY_TOPIC_CLASSIFICATIONS_SCHEMA = _schema(
    (
        ("repository_topic_classification_id", pa.string()),
        ("repository_metadata_id", pa.string()),
        ("repository_id", pa.string()),
        ("source_ref", pa.string()),
        ("source_commit", pa.string()),
        ("predicted_topic_count", pa.int64()),
        ("topics_json", pa.string()),
        ("topic_1", pa.string()),
        ("topic_1_domain", pa.string()),
        ("topic_1_score", pa.float64()),
    )
)

SNAPSHOT_MANIFESTS_SCHEMA = _schema(
    (
        ("snapshot_manifest_id", pa.string()),
        ("curation_pull_request_record_id", pa.string()),
        ("snapshot_label", pa.string()),
        ("available", pa.bool_()),
        ("file_availability_status", pa.string()),
        ("missing_reason", pa.string()),
        ("snapshot_commit", pa.string()),
        ("repository_observation_cutoff", pa.timestamp("ms", tz="UTC")),
        ("target_offset_days", pa.int64()),
        ("target_timestamp", pa.timestamp("ms", tz="UTC")),
        ("missing_files_json", pa.string()),
        ("deleted_files_json", pa.string()),
        ("renamed_files_json", pa.string()),
        ("unknown_missing_files_json", pa.string()),
    )
)

SNAPSHOT_FILE_REFS_SCHEMA = _schema(
    (
        ("snapshot_file_ref_id", pa.string()),
        ("snapshot_manifest_id", pa.string()),
        ("snapshot_file_blob_id", pa.string()),
        ("path", pa.string()),
    )
)

SNAPSHOT_FILE_BLOBS_SCHEMA = _schema(
    (
        ("snapshot_file_blob_id", pa.string()),
        ("content_sha256", pa.string()),
        ("hash_prefix", pa.string()),
        ("content_size_bytes", pa.int64()),
        ("content_encoding", pa.string()),
        ("is_binary", pa.bool_()),
        ("content_text", pa.string()),
        ("content_bytes_base64", pa.string()),
    )
)

ENTITY_SCHEMAS: dict[str, pa.Schema] = {
    ENTITY_CURATION_PULL_REQUESTS: CURATION_PULL_REQUESTS_SCHEMA,
    ENTITY_REFACTORING_RESULTS: REFACTORING_RESULTS_SCHEMA,
    ENTITY_MAINTAINABILITY_RESULTS: MAINTAINABILITY_RESULTS_SCHEMA,
    ENTITY_CODE_SMELL_RESULTS: CODE_SMELL_RESULTS_SCHEMA,
    ENTITY_REPOSITORY_ADDITIONAL_METADATA: REPOSITORY_ADDITIONAL_METADATA_SCHEMA,
    ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS: REPOSITORY_TOPIC_CLASSIFICATIONS_SCHEMA,
    ENTITY_SNAPSHOT_MANIFESTS: SNAPSHOT_MANIFESTS_SCHEMA,
    ENTITY_SNAPSHOT_FILE_REFS: SNAPSHOT_FILE_REFS_SCHEMA,
    ENTITY_SNAPSHOT_FILE_BLOBS: SNAPSHOT_FILE_BLOBS_SCHEMA,
}


@dataclass(frozen=True)
class AggregateRef:
    """Pointer to the newest selected aggregate for one curated PR."""

    path: Path
    cohort: str
    source_run_id: str
    pull_request_key: str
    repository_key: str | None
    pr_number: int | None
    rank: tuple[str, float]


@dataclass(frozen=True)
class TopicOutputRef:
    """Pointer to a topic-classification output directory."""

    output_dir: Path
    repository_topics_path: Path
    manifest_path: Path | None
    generated_at_utc: str | None
    rank: tuple[float, float]


@dataclass(frozen=True)
class LongitudinalRefactoringOutputRef:
    """Pointer to a longitudinal refactoring analysis output directory."""

    output_dir: Path
    manifest_path: Path
    snapshot_results_path: Path
    operations_path: Path | None
    pr_summary_path: Path | None
    generated_at_utc: str | None
    input_mode: str
    cohorts: tuple[str, ...]
    rank: tuple[float, float]


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 text form."""
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    """Return dictionaries unchanged and treat all other payloads as empty."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Return lists unchanged and treat all other payloads as empty."""
    return value if isinstance(value, list) else []


def _json_dumps(value: Any) -> str | None:
    """Serialize nested payloads for parquet string columns."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _safe_str(value: Any) -> str | None:
    """Return stripped text or ``None`` for missing/blank values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _popularity_label(value: Any) -> str | None:
    """Normalize legacy popularity bucket names into reader-facing labels."""
    text = _safe_str(value)
    if not text:
        return None
    bucket_labels = {
        "pop0": "low",
        "pop1": "medium",
        "pop2": "high",
    }
    if text in bucket_labels:
        return bucket_labels[text]
    if text.endswith("_popularity"):
        stripped = text[: -len("_popularity")]
        return stripped or text
    return text


def _safe_int(value: Any) -> int | None:
    """Coerce integer-like values while preserving ``None`` for bad inputs."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """Coerce float-like values while preserving ``None`` for bad inputs."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    """Return the first value that can be represented as a float."""
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _future_metric_value(payload: dict[str, Any], field: str) -> Any:
    """Read a future metric value from current or legacy curation payloads."""
    if field in payload:
        return payload.get(field)
    metrics = _as_dict(payload.get("metrics"))
    metric = _as_dict(metrics.get(field))
    return metric.get("future")


def _safe_bool(value: Any) -> bool | None:
    """Parse common boolean representations from source JSON fields."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_timestamp_value(value: Any) -> datetime | None:
    """Parse timestamps into UTC datetimes for PyArrow timestamp columns."""
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


def _public_topic_prediction_records(
    predictions: list[dict[str, Any]],
    *,
    effective_top_k: int | None,
) -> list[dict[str, Any]]:
    """Return compact public topic prediction records for one repository."""
    records: list[dict[str, Any]] = []
    selected_predictions = predictions if effective_top_k is None else predictions[:effective_top_k]
    for index, prediction in enumerate(selected_predictions, start=1):
        records.append(
            {
                "rank": index,
                "topic": _safe_str(prediction.get("topic")),
                "topic_domain": _safe_str(
                    prediction.get("topic_domain") or prediction.get("topic_group")
                ),
                "score": _safe_float(prediction.get("score")),
                "raw_rank": _safe_int(prediction.get("raw_rank")),
            }
        )
    return records


def _sha256_bytes(content: bytes) -> str:
    """Return a SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(content).hexdigest()


def _sha256_text(value: str | None) -> str | None:
    """Return a SHA-256 digest for text, preserving ``None``."""
    if value is None:
        return None
    return _sha256_bytes(value.encode("utf-8"))


def _cohort_from_payload(payload: dict[str, Any], fallback: object) -> str:
    """Resolve a safe cohort name from payload metadata or a caller fallback."""
    return safe_cohort_name(payload.get("cohort") or fallback or "unknown") or "unknown"


def _repo_key_from_url(value: Any) -> str | None:
    """Extract ``owner/name`` from GitHub repository or pull-request URLs."""
    text = str(value or "").strip().rstrip("/")
    if "github.com/" not in text:
        return None
    suffix = text.split("github.com/", 1)[1]
    parts = [part for part in suffix.split("/") if part]
    if len(parts) < 2:
        return None
    return normalize_repository_key(parts[0], parts[1])


def _repo_key_from_repository_payload(payload: dict[str, Any]) -> str | None:
    """Resolve a canonical repository key from common repository JSON shapes."""
    for key in ("name_with_owner", "full_name", "repository_full_name"):
        resolved = repository_key_from_full_name(payload.get(key))
        if resolved:
            return resolved
    owner = payload.get("owner")
    if isinstance(owner, dict):
        owner = owner.get("login") or owner.get("name")
    name = payload.get("name")
    if owner and name:
        return normalize_repository_key(str(owner), str(name))
    return _repo_key_from_url(payload.get("url"))


def _repository_payload_field(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    field_name: str,
) -> str | None:
    """Read a repository field from preferred payload, then fallback payload."""
    return _safe_str(primary.get(field_name) or fallback.get(field_name))


# Repository and PR identity helpers normalize historical curation payload
# shapes into stable natural keys. These natural keys drive deterministic entity
# ids and prevent duplicate PR rows across repeated packaging runs.
def _stable_numeric_id(value: Any) -> str | None:
    """Return a stable decimal id string from int-like source values."""
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return str(int(text)) if text.isdecimal() else None


def _repository_numeric_id_from_payload(payload: dict[str, Any]) -> str | None:
    """Find a GitHub repository numeric id in common payload fields."""
    for key in ("id", "database_id", "databaseId", "repository_id"):
        resolved = _stable_numeric_id(payload.get(key))
        if resolved:
            return resolved
    return None


def _stable_base_repository_id_from_snapshot_key(value: Any) -> str | None:
    """Recover a base repository id embedded in legacy snapshot keys."""
    text = str(value or "").strip()
    marker = "::repository_snapshot::"
    if marker not in text:
        return None
    remainder = text.rsplit(marker, 1)[-1]
    repository_identity, separator, _role = remainder.rpartition("::")
    if not separator:
        return None
    return _stable_numeric_id(repository_identity)


def _stable_base_repository_id_from_aggregate(payload: dict[str, Any]) -> str | None:
    """Return the best stable base repository id from one aggregate payload."""
    pr = _as_dict(payload.get("pr"))
    for value in (
        pr.get("base_repository_id"),
        pr.get("base_repository_database_id"),
        pr.get("base_repository_databaseId"),
        payload.get("base_repository_id"),
    ):
        resolved = _stable_numeric_id(value)
        if resolved:
            return resolved
    for nested_key in ("base_repository_full", "base_repository"):
        resolved = _repository_numeric_id_from_payload(_as_dict(pr.get(nested_key)))
        if resolved:
            return resolved
    for key in ("base_repository_snapshot_key", "repository_snapshot_key"):
        resolved = _stable_base_repository_id_from_snapshot_key(pr.get(key) or payload.get(key))
        if resolved:
            return resolved
    return None


def _stable_pull_request_number(value: Any) -> str | None:
    """Return a stable PR number string from int-like source values."""
    return _stable_numeric_id(value)


def _pull_request_number_from_key(value: Any) -> str | None:
    """Extract a PR number from URL-like or key-like pull request strings."""
    text = str(value or "").strip()
    match = re.search(r"(?:#|/)pull/(\d+)(?:\b|$)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return _stable_pull_request_number(match.group(1))


def _stable_curation_pr_dedup_key(
    *,
    payload: dict[str, Any],
    pull_request_key: str,
    pr_number: Any,
) -> str | None:
    """Build the strongest available stable duplicate key for a curated PR."""
    repository_id = _stable_base_repository_id_from_aggregate(payload)
    if not repository_id:
        return None
    number = _stable_pull_request_number(pr_number) or _pull_request_number_from_key(
        pull_request_key
    )
    if number:
        return f"base-repository:{repository_id}::pull-number:{number}"
    normalized_pr_key = _safe_str(pull_request_key)
    if normalized_pr_key:
        return f"base-repository:{repository_id}::pull-request-key:{normalized_pr_key}"
    return None


def _author_label(pr: dict[str, Any]) -> str | None:
    """Return a compact author label from nested or flat PR payloads."""
    author = pr.get("author")
    if isinstance(author, dict):
        return _safe_str(
            author.get("login") or author.get("name") or author.get("id") or author.get("url")
        )
    return _safe_str(pr.get("author_login") or pr.get("user_login") or pr.get("login") or author)


def _repository_stargazer_count(
    repo_metadata: dict[str, Any],
    pr: dict[str, Any],
) -> int | None:
    """Return repository star count from curation metadata without treating zero as missing."""
    candidates: list[Any] = []
    for payload in (
        repo_metadata,
        _as_dict(pr.get("base_repository_full")),
        _as_dict(pr.get("base_repository")),
        _as_dict(pr.get("repository")),
        _as_dict(pr.get("head_repository_full")),
        _as_dict(pr.get("head_repository")),
    ):
        for key in ("stargazer_count", "stargazers_count", "star_count", "watchers_count"):
            if key in payload:
                candidates.append(payload.get(key))
    for value in candidates:
        count = _safe_int(value)
        if count is not None:
            return max(0, count)
    return None


def _repository_key_from_aggregate(payload: dict[str, Any]) -> str | None:
    """Resolve the canonical base repository key from a curation aggregate."""
    pr = _as_dict(payload.get("pr"))
    repo_metadata = _as_dict(payload.get("repository_metadata"))
    for value in (
        pr.get("repo_full_name"),
        repo_metadata.get("name_with_owner"),
        repo_metadata.get("full_name"),
    ):
        resolved = repository_key_from_full_name(value)
        if resolved:
            return resolved
    if pr.get("repository_owner") and pr.get("repository_name"):
        return normalize_repository_key(str(pr["repository_owner"]), str(pr["repository_name"]))
    for nested_key in (
        "base_repository_full",
        "base_repository",
        "head_repository_full",
        "head_repository",
    ):
        nested = _as_dict(pr.get(nested_key))
        resolved = _repo_key_from_repository_payload(nested)
        if resolved:
            return resolved
    resolved = _repo_key_from_repository_payload(repo_metadata)
    if resolved:
        return resolved
    return _repo_key_from_url(pr.get("url"))


def _owner_name_from_repo_key(repository_key: str | None) -> tuple[str | None, str | None]:
    """Split a canonical repository key into owner and repository name."""
    if not repository_key or "/" not in repository_key:
        return None, None
    owner, name = repository_key.split("/", 1)
    return owner or None, name or None


def _pull_request_key(payload: dict[str, Any]) -> str:
    """Return a stable pull request key from repository key and PR number."""
    pr = _as_dict(payload.get("pr"))
    repository_key = _repository_key_from_aggregate(payload)
    pr_number = _safe_int(pr.get("number") or pr.get("pr_number"))
    if repository_key and pr_number is not None:
        return f"{repository_key}#pull/{pr_number}"
    pr_url = _safe_str(pr.get("url") or pr.get("pr_url"))
    if pr_url:
        return pr_url.lower()
    pr_id = _safe_str(pr.get("id") or pr.get("pr_id"))
    if pr_id:
        return f"pr-id:{pr_id}"
    digest = _sha256_bytes(json.dumps(pr, sort_keys=True, default=str).encode("utf-8"))
    return f"unknown-pr:{digest}"


def _snapshot_key(pull_request_key: str, snapshot_label: str) -> str:
    """Return the stable key for a PR snapshot label."""
    return f"{pull_request_key}@{snapshot_label}"


def deterministic_entity_id(entity_name: str, *key_parts: Any) -> str:
    """Return a stable compact id for an entity natural key."""
    payload = "\0".join([entity_name, *(str(part or "") for part in key_parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _curation_pull_request_record_id(
    *,
    cohort: Any,
    pull_request_key: Any,
) -> str:
    """Return the primary id for a curated PR row."""
    return deterministic_entity_id(ENTITY_CURATION_PULL_REQUESTS, cohort, pull_request_key)


def _repository_metadata_id(repository_identity: Any) -> str | None:
    """Return the primary id for repository metadata identity text."""
    if not _safe_str(repository_identity):
        return None
    return deterministic_entity_id(ENTITY_REPOSITORY_ADDITIONAL_METADATA, repository_identity)


def _repository_metadata_identity_from_parts(
    *,
    stable_repository_id: Any = None,
    repository_key: Any = None,
) -> str | None:
    """Prefer stable numeric repository identity, then canonical repo key."""
    stable_id = _stable_numeric_id(stable_repository_id)
    if stable_id:
        return f"stable-base-repository-id:{stable_id}"
    key = _safe_str(repository_key)
    return f"repository-key:{key}" if key else None


def _repository_metadata_identity_from_row(row: dict[str, Any]) -> str | None:
    """Resolve repository metadata identity fields from one output row."""
    return _repository_metadata_identity_from_parts(
        stable_repository_id=row.get("base_repository_id")
        or row.get("repository_metadata_id_numeric")
        or row.get("stable_repository_id"),
        repository_key=row.get("repository_key") or row.get("base_repository_key"),
    )


def _repository_metadata_id_from_row(row: dict[str, Any]) -> str | None:
    """Return repository metadata primary id from a row when possible."""
    identity = _repository_metadata_identity_from_row(row)
    return _repository_metadata_id(identity) if identity else None


def _snapshot_manifest_id(
    *,
    cohort: Any,
    snapshot_key: Any,
) -> str | None:
    """Return the primary id for one snapshot manifest row."""
    if not _safe_str(snapshot_key):
        return None
    return deterministic_entity_id(ENTITY_SNAPSHOT_MANIFESTS, cohort, snapshot_key)


def _snapshot_file_blob_id(content_sha256: Any) -> str | None:
    """Return the primary id for a de-duplicated snapshot content blob."""
    if not _safe_str(content_sha256):
        return None
    return deterministic_entity_id(ENTITY_SNAPSHOT_FILE_BLOBS, content_sha256)


def _entity_primary_id(entity_name: str, row: dict[str, Any]) -> str | None:
    """Return the deterministic primary id for one curation entity row."""
    if entity_name == ENTITY_CURATION_PULL_REQUESTS:
        return _curation_pull_request_record_id(
            cohort=row.get("cohort"),
            pull_request_key=row.get("pull_request_key"),
        )
    if entity_name == ENTITY_REFACTORING_RESULTS:
        return deterministic_entity_id(
            entity_name,
            row.get("cohort"),
            row.get("result_source"),
            row.get("input_mode"),
            row.get("pull_request_key"),
            row.get("snapshot_label"),
            row.get("transition_label"),
        )
    if entity_name == ENTITY_MAINTAINABILITY_RESULTS:
        return deterministic_entity_id(entity_name, row.get("cohort"), row.get("snapshot_key"))
    if entity_name == ENTITY_CODE_SMELL_RESULTS:
        return deterministic_entity_id(entity_name, row.get("cohort"), row.get("snapshot_key"))
    if entity_name == ENTITY_REPOSITORY_ADDITIONAL_METADATA:
        return _repository_metadata_id_from_row(row)
    if entity_name == ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS:
        return deterministic_entity_id(entity_name, row.get("repository_key"))
    if entity_name == ENTITY_SNAPSHOT_MANIFESTS:
        return _snapshot_manifest_id(
            cohort=row.get("cohort"),
            snapshot_key=row.get("snapshot_key"),
        )
    if entity_name == ENTITY_SNAPSHOT_FILE_REFS:
        return deterministic_entity_id(
            entity_name,
            row.get("cohort"),
            row.get("snapshot_key"),
            row.get("path"),
        )
    if entity_name == ENTITY_SNAPSHOT_FILE_BLOBS:
        return _snapshot_file_blob_id(row.get("content_sha256"))
    return None


def _with_entity_ids(entity_name: str, row: dict[str, Any]) -> dict[str, Any]:
    """Return a row enriched with deterministic primary and foreign key ids."""
    enriched = dict(row)
    primary_key_column = ENTITY_PRIMARY_KEY_COLUMNS.get(entity_name)
    if primary_key_column:
        enriched[primary_key_column] = enriched.get(primary_key_column) or _entity_primary_id(
            entity_name,
            enriched,
        )

    if "curation_pull_request_record_id" in ENTITY_SCHEMAS[entity_name].names:
        enriched["curation_pull_request_record_id"] = (
            enriched.get("curation_pull_request_record_id")
            or _curation_pull_request_record_id(
                cohort=enriched.get("cohort"),
                pull_request_key=enriched.get("pull_request_key"),
            )
        )
    if "repository_metadata_id" in ENTITY_SCHEMAS[entity_name].names:
        enriched["repository_metadata_id"] = enriched.get(
            "repository_metadata_id"
        ) or _repository_metadata_id_from_row(enriched)
    if "base_repository_metadata_id" in ENTITY_SCHEMAS[entity_name].names:
        enriched["base_repository_metadata_id"] = enriched.get(
            "base_repository_metadata_id"
        ) or _repository_metadata_id_from_row(enriched)
    if "snapshot_manifest_id" in ENTITY_SCHEMAS[entity_name].names:
        enriched["snapshot_manifest_id"] = enriched.get(
            "snapshot_manifest_id"
        ) or _snapshot_manifest_id(
            cohort=enriched.get("cohort"),
            snapshot_key=enriched.get("snapshot_key"),
        )
    if "snapshot_file_blob_id" in ENTITY_SCHEMAS[entity_name].names:
        enriched["snapshot_file_blob_id"] = enriched.get(
            "snapshot_file_blob_id"
        ) or _snapshot_file_blob_id(enriched.get("content_sha256"))
    return enriched


def _curation_dataset_dir(cohort_root: Path) -> Path:
    """Return the canonical ``Curation`` subdirectory for a data root.

    Earlier local runs could produce case variants. When a matching directory
    exists with the wrong case, normalize it so downstream paths and dataset
    configs stay stable.
    """
    target = cohort_root / CURATION_DATASET_SUBDIR
    if not cohort_root.exists():
        return target
    matching_dirs = [
        child
        for child in cohort_root.iterdir()
        if child.is_dir() and child.name.lower() == CURATION_DATASET_SUBDIR.lower()
    ]
    for child in matching_dirs:
        if child.name == CURATION_DATASET_SUBDIR:
            return child
    if not matching_dirs:
        return target

    legacy = matching_dirs[0]
    temp = cohort_root / f".{CURATION_DATASET_SUBDIR}.casefix-{int(time.time() * 1000)}"
    while temp.exists():
        temp = cohort_root / f"{temp.name}-next"
    legacy.rename(temp)
    temp.rename(target)
    return target


def _pull_request_key_from_parts(
    *,
    repository_key: str | None,
    pr_number: int | None,
    pr_url: str | None,
    pr_id: str | None = None,
) -> str:
    """Build a pull request key from known pieces, falling back to a hash."""
    if repository_key and pr_number is not None:
        return f"{repository_key}#pull/{pr_number}"
    if pr_url:
        return pr_url.strip().lower()
    if pr_id:
        return f"pr-id:{pr_id}"
    seed = json.dumps(
        {
            "repository_key": repository_key,
            "pr_number": pr_number,
            "pr_url": pr_url,
            "pr_id": pr_id,
        },
        sort_keys=True,
        default=str,
    )
    return f"unknown-pr:{_sha256_bytes(seed.encode('utf-8'))}"


def _target_offset_days(snapshot_label: str) -> int | None:
    """Parse future snapshot labels like ``+31d`` into day offsets."""
    match = re.match(r"^\+(\d+)d$", str(snapshot_label or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _normalize_path(value: Any) -> str:
    """Normalize snapshot file paths into portable slash-separated form."""
    return str(value or "").replace("\\", "/").lstrip("/")


def _partition_component(value: Any) -> str:
    """Return a filesystem-safe parquet partition component."""
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.+=-]+", "_", text)
    return text.strip("_") or "unknown"


def _refactoring_results_partition(result_source: str, input_mode: str | None = None) -> str:
    """Return the partition path used for refactoring result provenance."""
    parts = [f"result_source={_partition_component(result_source)}"]
    if input_mode:
        parts.append(f"input_mode={_partition_component(input_mode)}")
    return "/".join(parts)


def _repository_key_from_topic_payload(payload: dict[str, Any]) -> str | None:
    """Resolve a repository key from topic-classification output rows."""
    repository_key = _safe_str(payload.get("repository_key"))
    if repository_key:
        resolved = repository_key_from_full_name(repository_key)
        if resolved:
            return resolved
        resolved = repository_key_from_safe_key(repository_key)
        if resolved:
            return resolved
        if "/" in repository_key:
            owner, name = repository_key.split("/", 1)
            return normalize_repository_key(owner, name)
    resolved = repository_key_from_full_name(payload.get("repository_full_name"))
    if resolved:
        return resolved
    owner = _safe_str(payload.get("repository_owner"))
    name = _safe_str(payload.get("repository_name"))
    if owner and name:
        return normalize_repository_key(owner, name)
    return None


def _parse_iso_timestamp_seconds(value: Any) -> float | None:
    """Parse ISO timestamps into seconds for output freshness ranking."""
    text = _safe_str(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _file_mtime(path: Path) -> float:
    """Return file modification time, or zero when the path is unavailable."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _topic_output_ref(output_dir: Path) -> TopicOutputRef | None:
    """Return a topic output reference when a directory has public JSONL rows."""
    output_dir = Path(output_dir)
    repository_topics_path = output_dir / "repository_topics.jsonl"
    if not repository_topics_path.exists():
        return None
    manifest_path = output_dir / "topic_classification_manifest.json"
    manifest = read_json_object(manifest_path) if manifest_path.exists() else {}
    manifest = manifest or {}
    generated_at_utc = _safe_str(manifest.get("generated_at_utc"))
    mtime = _file_mtime(output_dir)
    generated_rank = _parse_iso_timestamp_seconds(generated_at_utc)
    return TopicOutputRef(
        output_dir=output_dir,
        repository_topics_path=repository_topics_path,
        manifest_path=manifest_path if manifest_path.exists() else None,
        generated_at_utc=generated_at_utc,
        rank=(generated_rank if generated_rank is not None else mtime, mtime),
    )


def _topic_output_directories(root: Path) -> list[Path]:
    """Discover candidate directories containing ``repository_topics.jsonl``."""
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        if root.name == "repository_topics.jsonl":
            return [root.parent]
        return []
    candidate_dirs: dict[Path, None] = {}
    for path in root.rglob("repository_topics.jsonl"):
        candidate_dirs[path.parent.resolve()] = None
    return [
        Path(path)
        for path in candidate_dirs
    ]


def _discover_topic_output_refs(root: Path) -> list[TopicOutputRef]:
    """Return topic outputs ordered by generated timestamp and mtime."""
    root = Path(root)
    if not root.exists():
        return []
    candidate_dirs = sorted(_topic_output_directories(root))
    refs = [
        ref
        for ref in (_topic_output_ref(candidate) for candidate in candidate_dirs)
        if ref is not None
    ]
    return sorted(refs, key=lambda ref: (ref.rank, str(ref.output_dir).lower()))


def _longitudinal_output_cohorts(
    manifest: dict[str, Any],
    snapshot_results_path: Path,
) -> tuple[str, ...]:
    """Infer the cohorts covered by one longitudinal-refactoring output."""
    loader = _as_dict(_as_dict(manifest.get("statistics")).get("loader"))
    cohorts: set[str] = set()
    for key in ("cohort_counts", "selected_cohort_counts", "candidate_cohort_counts"):
        mapping = _as_dict(loader.get(key))
        for cohort in mapping:
            normalized = safe_cohort_name(cohort)
            if normalized:
                cohorts.add(normalized)
    if cohorts:
        return tuple(sorted(cohorts))
    for _line_number, payload in _read_jsonl(snapshot_results_path):
        cohort = _cohort_from_payload(payload, "")
        if cohort:
            cohorts.add(cohort)
    return tuple(sorted(cohorts))


def _longitudinal_output_ref(output_dir: Path) -> LongitudinalRefactoringOutputRef | None:
    """Return a longitudinal-refactoring output reference when files exist."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "longitudinal_refactoring_manifest.json"
    snapshot_results_path = output_dir / "longitudinal_refactoring_snapshot_results.jsonl"
    if not manifest_path.exists() or not snapshot_results_path.exists():
        return None
    manifest = read_json_object(
        manifest_path,
        description="longitudinal refactoring manifest JSON",
    )
    if manifest is None:
        return None
    config = _as_dict(manifest.get("config"))
    reporting = _as_dict(manifest.get("reporting"))
    statistics = _as_dict(manifest.get("statistics"))
    input_mode = _safe_str(
        config.get("input_mode") or reporting.get("input_mode") or statistics.get("input_mode")
    )
    if not input_mode:
        return None
    generated_at_utc = _safe_str(manifest.get("generated_at_utc"))
    mtime = _file_mtime(output_dir)
    generated_rank = _parse_iso_timestamp_seconds(generated_at_utc)
    operations_path = output_dir / "longitudinal_refactoring_operations.jsonl"
    pr_summary_path = output_dir / "longitudinal_refactoring_pr_summary.jsonl"
    return LongitudinalRefactoringOutputRef(
        output_dir=output_dir,
        manifest_path=manifest_path,
        snapshot_results_path=snapshot_results_path,
        operations_path=operations_path if operations_path.exists() else None,
        pr_summary_path=pr_summary_path if pr_summary_path.exists() else None,
        generated_at_utc=generated_at_utc,
        input_mode=input_mode,
        cohorts=_longitudinal_output_cohorts(manifest, snapshot_results_path),
        rank=(generated_rank if generated_rank is not None else mtime, mtime),
    )


def _discover_longitudinal_refactoring_output_refs(
    root: Path,
) -> list[LongitudinalRefactoringOutputRef]:
    """Return longitudinal outputs ordered by generated timestamp and mtime."""
    root = Path(root)
    if not root.exists():
        return []
    candidate_dirs: dict[Path, None] = {}
    for candidate in (
        root,
        root / "output",
    ):
        if (candidate / "longitudinal_refactoring_manifest.json").exists():
            candidate_dirs[candidate.resolve()] = None
    for path in root.glob("*/output/longitudinal_refactoring_manifest.json"):
        candidate_dirs[path.parent.resolve()] = None
    refs = [
        ref
        for ref in (_longitudinal_output_ref(candidate) for candidate in candidate_dirs)
        if ref is not None
    ]
    return sorted(refs, key=lambda ref: (ref.rank, str(ref.output_dir).lower()))


def _source_run_rank(path: Path) -> tuple[str, float]:
    """Return a sortable rank for choosing the newest source-run artifact."""
    timestamp_match = re.search(r"(\d{8}T\d{6}Z)", path.name)
    timestamp = timestamp_match.group(1) if timestamp_match else path.name
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return timestamp, mtime


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield valid JSON object rows from a JSONL file with line numbers."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield line_number, payload


def _discover_sampled_pr_jsonl_paths(cohort_dir: Path) -> list[Path]:
    """Find sampled-PR selection files for one curation cohort directory."""
    paths: dict[Path, None] = {}
    for root in candidate_curation_roots(cohort_dir):
        if not root.exists():
            continue
        for path in root.glob("sampled_prs_*.jsonl"):
            if path.name.endswith("_store.jsonl"):
                continue
            paths[path] = None
    return sorted(paths)


def _discover_longitudinal_pr_jsonl_paths(cohort_dir: Path) -> list[Path]:
    """Find longitudinal-PR selection files for one curation cohort directory."""
    paths: dict[Path, None] = {}
    for root in candidate_curation_roots(cohort_dir):
        if not root.exists():
            continue
        for path in root.glob("longitudinal_prs_*.jsonl"):
            if path.name.endswith("_store.jsonl"):
                continue
            paths[path] = None
    return sorted(paths)


def _discover_repository_file_list_paths(cohort_dir: Path) -> list[Path]:
    """Find hydrated repository file lists used for repository metadata rows."""
    paths: dict[Path, None] = {}
    for root in candidate_curation_roots(cohort_dir):
        snapshots_dir = root / "output" / "snapshots"
        if not snapshots_dir.exists():
            continue
        for path in snapshots_dir.glob("*/*/repository_file_list.json"):
            paths[path] = None
    return sorted(paths)


def _cohort_from_jsonl_path(path: Path, fallback: str) -> str:
    """Infer cohort name from selection JSONL filenames."""
    match = re.match(r"^(?:sampled|longitudinal)_prs_(.+?)\.jsonl$", path.name)
    if not match:
        return safe_cohort_name(fallback)
    cohort = match.group(1)
    for prefix in ("topup_relaxed_language_", "topup_"):
        if cohort.startswith(prefix):
            cohort = cohort[len(prefix) :]
    if cohort.endswith("_store"):
        cohort = cohort[: -len("_store")]
    return safe_cohort_name(cohort or fallback)


def _selection_payload_repository_key(payload: dict[str, Any]) -> str | None:
    """Resolve repository key from sampled/longitudinal selection rows."""
    direct = repository_key_from_full_name(payload.get("repo_full_name"))
    if direct:
        return direct
    original = _as_dict(payload.get("original_pr_payload"))
    aggregate_like = {
        "pr": {**original, **payload},
        "repository_metadata": _as_dict(original.get("base_repository_full")),
    }
    return _repository_key_from_aggregate(aggregate_like)


def _selection_pull_request_key(payload: dict[str, Any]) -> str:
    """Return a stable PR key for sampled/longitudinal selection rows."""
    repository_key = _selection_payload_repository_key(payload)
    pr_number = _safe_int(payload.get("pr_number") or payload.get("number"))
    if repository_key and pr_number is not None:
        return f"{repository_key}#pull/{pr_number}"
    pr_url = _safe_str(payload.get("pr_url") or payload.get("url"))
    if pr_url:
        return pr_url.lower()
    pr_id = _safe_str(payload.get("pr_id") or payload.get("id"))
    if pr_id:
        return f"pr-id:{pr_id}"
    return f"selection:{_sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode('utf-8'))}"


def _sampling_metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract sampler metadata fields from a selection payload."""
    original = _as_dict(payload.get("original_pr_payload"))
    metadata = dict(_as_dict(payload.get("sampling_metadata")))
    for key, value in payload.items():
        if key == "original_pr_payload":
            continue
        if (
            key.startswith("sampling_")
            or key.startswith("selected_")
            or key.startswith("time_bucket")
            or key
            in {
                "popularity_bucket",
                "target_languages",
                "time_bucket_granularity",
                "popularity_buckets",
                "soft_balance_alpha_time",
                "soft_balance_alpha_popularity",
            }
        ):
            metadata.setdefault(key, value)
    for key in ("primary_language", "pr_primary_language_effective"):
        if key in original:
            metadata.setdefault(key, original.get(key))
    return metadata


def _sampling_language_bucket(metadata: dict[str, Any]) -> str | None:
    """Return the language-balance bucket from sampler metadata."""
    return _safe_str(
        metadata.get("sampling_language_bucket")
        or metadata.get("selected_language_bucket")
        or metadata.get("primary_language")
        or metadata.get("pr_primary_language_effective")
    )


def _sampling_time_bucket(metadata: dict[str, Any]) -> str | None:
    """Return the time-balance bucket from sampler metadata."""
    return _safe_str(
        metadata.get("sampling_time_bucket")
        or metadata.get("time_bucket_sampling")
        or metadata.get("time_bucket_day")
    )


def _sampling_popularity_bucket(metadata: dict[str, Any]) -> str | None:
    """Return the popularity-balance bucket from sampler metadata."""
    return _safe_str(
        metadata.get("sampling_popularity_bucket") or metadata.get("popularity_bucket")
    )


def _snapshot_payloads(
    hydration: dict[str, Any],
    *,
    future_labels: Iterable[str] = (),
) -> Iterator[tuple[str, dict[str, Any], str]]:
    """Yield before, after, and future snapshot payloads from hydration JSON."""
    snapshots = _as_dict(hydration.get("snapshots"))
    for label in ("before", "after"):
        payload = _as_dict(snapshots.get(label))
        if payload:
            yield label, payload, "pr"
    future = _as_dict(snapshots.get("future"))
    labels = sorted({str(label) for label in future} | {str(label) for label in future_labels})
    for label in labels:
        payload = _as_dict(future.get(label))
        yield label, payload, "future"


def _future_snapshot_availability_records(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized availability rows for longitudinal future snapshots."""
    pr = _as_dict(aggregate.get("pr"))
    hydration = _as_dict(aggregate.get("hydration"))
    availability_by_label = _as_dict(hydration.get("future_snapshot_availability"))
    snapshots = _as_dict(hydration.get("snapshots"))
    future_snapshots = _as_dict(snapshots.get("future"))
    maintainability_summary = _as_dict(
        _as_dict(_as_dict(aggregate.get("metrics")).get("maintainability")).get("summary")
    )
    future_maint_metrics = _as_dict(
        maintainability_summary.get("maintainability_future_snapshot_metrics")
    )
    labels = sorted(
        set(FUTURE_SNAPSHOT_LABELS)
        | {str(label) for label in availability_by_label}
        | {str(label) for label in future_snapshots}
        | {str(label) for label in future_maint_metrics}
    )
    if not labels or (not pr.get("longitudinal_selected") and not availability_by_label):
        return []
    records: list[dict[str, Any]] = []
    for label in labels:
        availability = _as_dict(availability_by_label.get(label))
        snapshot = _as_dict(future_snapshots.get(label))
        metric = _as_dict(future_maint_metrics.get(label))
        explicit_available = availability.get("available")
        if explicit_available is None:
            explicit_available = snapshot.get("available")
        if explicit_available is None:
            explicit_available = metric.get("snapshot_available")
        if explicit_available is None:
            explicit_available = bool(snapshot.get("commit") or metric.get("snapshot_commit"))
        missing_reason = (
            availability.get("missing_reason")
            or snapshot.get("missing_reason")
            or metric.get("missing_reason")
        )
        if explicit_available is False and not missing_reason:
            missing_reason = "missing_snapshot_unknown"

        def pick(*keys: str) -> Any:
            for key in keys:
                for source in (availability, snapshot, metric):
                    value = source.get(key)
                    if value is not None:
                        return value
            return None

        records.append(
            {
                "snapshot_label": label,
                "available": bool(explicit_available),
                "missing_reason": missing_reason if explicit_available is False else None,
                "target_timestamp": pick("target_timestamp"),
                "repository_observation_cutoff": pick("repository_observation_cutoff"),
                "snapshot_commit": pick("snapshot_commit", "commit"),
                "file_availability_status": pick("file_availability_status"),
                "files_expected": pick("files_expected"),
                "files_copied": pick("files_copied"),
                "files_missing": pick("files_missing"),
                "missing_files": _as_list(pick("missing_files")),
                "deleted_files": _as_list(pick("deleted_files")),
                "renamed_files": _as_list(pick("renamed_files")),
                "unknown_missing_files": _as_list(pick("unknown_missing_files")),
            }
        )
    return records


def _count_mapping(value: Any) -> dict[str, int]:
    """Normalize count dictionaries and drop non-positive values."""
    mapping = _as_dict(value)
    normalized: dict[str, int] = {}
    for key, count_value in mapping.items():
        count = _safe_int(count_value) or 0
        if count <= 0:
            continue
        normalized[str(key)] = normalized.get(str(key), 0) + count
    return dict(sorted(normalized.items()))


def _standardized_refactoring_for_raw_refactoring(raw_refactoring: str) -> str | None:
    """Map a raw refactoring label to the active standardized taxonomy."""
    standardized = canonicalize_refactoring_type(raw_refactoring)
    standardized_label = str(standardized or "").strip()
    return standardized_label or None


def _current_refactoring_count_mapping(counts: dict[str, int]) -> dict[str, int]:
    """Aggregate raw refactoring counts by standardized refactoring label."""
    current_counts: dict[str, int] = defaultdict(int)
    for raw_label, count in counts.items():
        standardized = _standardized_refactoring_for_raw_refactoring(str(raw_label or ""))
        if standardized:
            current_counts[standardized] += int(count)
    return dict(sorted(current_counts.items()))


def _murphy_hill_category_for_standardized_refactoring(standardized_refactoring: str) -> str | None:
    """Return the Murphy-Hill level for a standardized refactoring label."""
    standardized_label = str(standardized_refactoring or "").strip()
    if not standardized_label:
        return None
    taxonomy = classify_refactoring_taxonomy(standardized_label)
    level = str(_as_dict(taxonomy).get("murphy_hill_level") or "").strip().lower()
    return level if level in MURPHY_HILL_COUNT_COLUMNS else None


def _operation_murphy_hill_level(operation: dict[str, Any]) -> str | None:
    """Return the Murphy-Hill level for one refactoring operation payload."""
    raw_type = str(operation.get("raw_type") or "").strip()
    stored_standardized = str(operation.get("standardized_type") or "").strip()
    standardized = (
        _standardized_refactoring_for_raw_refactoring(raw_type)
        if raw_type
        else _standardized_refactoring_for_raw_refactoring(stored_standardized)
    )
    return _murphy_hill_category_for_standardized_refactoring(standardized)


def _refactoring_mapping_fields(
    operations: list[dict[str, Any]],
    standardized_refactoring_counts: dict[str, int],
) -> dict[str, str]:
    """Build JSON mapping fields that document refactoring label conversion."""
    raw_to_standardized: dict[str, str | None] = {}
    standardized_to_murphy_hill: dict[str, str | None] = {}

    for operation_value in operations:
        operation = _as_dict(operation_value)
        raw_type = str(operation.get("raw_type") or "").strip()
        stored_standardized_type = str(operation.get("standardized_type") or "").strip()
        if raw_type:
            standardized_type = (
                _standardized_refactoring_for_raw_refactoring(raw_type)
                or _standardized_refactoring_for_raw_refactoring(stored_standardized_type)
                or ""
            )
        else:
            standardized_type = (
                _standardized_refactoring_for_raw_refactoring(stored_standardized_type)
                or ""
            )
        if not raw_type and standardized_type:
            raw_type = standardized_type
        if raw_type:
            existing = raw_to_standardized.get(raw_type)
            if existing is None:
                raw_to_standardized[raw_type] = standardized_type or None
        if standardized_type:
            murphy_hill_level = _operation_murphy_hill_level(operation)
            if murphy_hill_level is None:
                murphy_hill_level = _murphy_hill_category_for_standardized_refactoring(
                    standardized_type
                )
            existing = standardized_to_murphy_hill.get(standardized_type)
            if existing is None:
                standardized_to_murphy_hill[standardized_type] = murphy_hill_level

    has_operation_mapping = bool(raw_to_standardized)
    for standardized_type in sorted(standardized_refactoring_counts):
        standardized_label = (
            _standardized_refactoring_for_raw_refactoring(str(standardized_type or ""))
            or str(standardized_type or "").strip()
        )
        if not standardized_label:
            continue
        if not has_operation_mapping:
            raw_to_standardized[standardized_label] = standardized_label
        standardized_to_murphy_hill.setdefault(
            standardized_label,
            _murphy_hill_category_for_standardized_refactoring(standardized_label),
        )

    return {
        "raw_refactoring_to_standardized_refactoring_json": _json_dumps(
            dict(sorted(raw_to_standardized.items()))
        ),
        "standardized_refactoring_to_murphyhill_category_json": _json_dumps(
            dict(sorted(standardized_to_murphy_hill.items()))
        ),
    }


def _murphy_hill_count_columns(
    murphy_hill_counts: dict[str, int],
    standardized_refactoring_counts: dict[str, int],
) -> dict[str, int]:
    """Return fixed Murphy-Hill count columns for the public schema."""
    derived_counts: dict[str, int] = defaultdict(int)
    for standardized_refactoring, count in standardized_refactoring_counts.items():
        standardized_label = (
            _standardized_refactoring_for_raw_refactoring(str(standardized_refactoring or ""))
            or str(standardized_refactoring or "").strip()
        )
        level = _murphy_hill_category_for_standardized_refactoring(standardized_label)
        if level:
            derived_counts[level] += int(count)
    counts = dict(derived_counts) if derived_counts else dict(murphy_hill_counts)

    columns = {column: 0 for column in MURPHY_HILL_COUNT_COLUMNS.values()}
    for level, count in counts.items():
        normalized_level = str(level or "").strip().lower()
        column = MURPHY_HILL_COUNT_COLUMNS.get(normalized_level)
        if column:
            columns[column] += int(count)
    return columns


def _standardized_smell_for_raw_smell(raw_smell: str) -> str | None:
    """Map a raw smell label to the active standardized smell taxonomy."""
    raw_label = str(raw_smell or "").strip()
    if not raw_label:
        return None
    standardized = canonicalize_code_smell_type(
        rule_id=raw_label,
        category="maintainability",
        message=None,
        tool=None,
    )
    standardized_label = str(standardized or "").strip()
    return standardized_label or None


def _mantyla_category_for_standardized_smell(standardized_smell: str) -> str | None:
    """Return the Mantyla category for a standardized smell label."""
    standardized_label = str(standardized_smell or "").strip()
    if not standardized_label:
        return None
    taxonomy = classify_code_smell_taxonomy(
        rule_id=standardized_label,
        category="maintainability",
    )
    category = str(_as_dict(taxonomy).get("mantyla") or "").strip()
    return category or None


def _raw_to_standardized_smell_mapping(
    raw_smell_counts: dict[str, int],
    standardized_smell_counts: dict[str, int],
) -> dict[str, str | None]:
    """Build the raw-to-standardized smell mapping written to parquet JSON."""
    source_labels = raw_smell_counts if raw_smell_counts else standardized_smell_counts
    mapping: dict[str, str | None] = {}
    for raw_smell in sorted(source_labels):
        standardized = _standardized_smell_for_raw_smell(raw_smell)
        if standardized is None and not raw_smell_counts and raw_smell in standardized_smell_counts:
            standardized = raw_smell
        mapping[str(raw_smell)] = standardized
    return mapping


def _current_standardized_smell_count_mapping(
    raw_smell_counts: dict[str, int],
    standardized_smell_counts: dict[str, int],
) -> dict[str, int]:
    """Aggregate smell counts by standardized smell label."""
    source_counts = raw_smell_counts if raw_smell_counts else standardized_smell_counts
    current_counts: dict[str, int] = defaultdict(int)
    for raw_label, count in source_counts.items():
        standardized = _standardized_smell_for_raw_smell(str(raw_label or ""))
        if standardized is None and not raw_smell_counts:
            standardized = str(raw_label or "").strip()
        if standardized:
            current_counts[standardized] += int(count)
    return dict(sorted(current_counts.items()))


def _standardized_smell_to_mantyla_category_mapping(
    standardized_smell_counts: dict[str, int],
    raw_to_standardized_smells: dict[str, str | None],
) -> dict[str, str | None]:
    """Build the standardized-smell to Mantyla-category mapping."""
    standardized_labels = {
        str(label).strip() for label in standardized_smell_counts if str(label).strip()
    }
    standardized_labels.update(
        str(label).strip()
        for label in raw_to_standardized_smells.values()
        if str(label or "").strip()
    )
    mapping: dict[str, str | None] = {}
    for standardized_smell in sorted(standardized_labels):
        mapping[standardized_smell] = _mantyla_category_for_standardized_smell(standardized_smell)
    return mapping


def _normalized_mantyla_category_key(category: str) -> str:
    """Normalize Mantyla category labels into schema column keys."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(category or "").strip().lower()).strip("_")
    if normalized in {
        "",
        "unknown",
        "unclassified",
        "unmapped",
        "none",
        "null",
    }:
        return "unmapped"
    aliases = {
        "object_oriented_abusers": "object_orientation_abusers",
        "object_orientation_abuser": "object_orientation_abusers",
        "oo_abusers": "object_orientation_abusers",
        "other": "others",
    }
    return aliases.get(normalized, normalized)


def _mantyla_count_columns(
    mantyla_category_counts: dict[str, int],
    standardized_smell_counts: dict[str, int],
) -> dict[str, int]:
    """Return fixed Mantyla count columns for the public schema."""
    derived_counts: dict[str, int] = defaultdict(int)
    for standardized_smell, count in standardized_smell_counts.items():
        category = _mantyla_category_for_standardized_smell(standardized_smell)
        derived_counts[category or "unmapped"] += int(count)
    counts = dict(derived_counts) if derived_counts else dict(mantyla_category_counts)

    columns = {column: 0 for column in MANTYLA_CATEGORY_COUNT_COLUMNS.values()}
    for category, count in counts.items():
        normalized_category = _normalized_mantyla_category_key(category)
        column = MANTYLA_CATEGORY_COUNT_COLUMNS.get(
            normalized_category,
            MANTYLA_CATEGORY_COUNT_COLUMNS["unmapped"],
        )
        columns[column] += int(count)
    return columns


def _smell_snapshot_mappings(aggregate: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    """Return code-smell count mappings keyed by snapshot label."""
    summary = _as_dict(
        _as_dict(_as_dict(aggregate.get("metrics")).get("maintainability")).get("summary")
    )
    snapshots: dict[str, dict[str, dict[str, int]]] = {}
    if not summary:
        return snapshots
    snapshots["before"] = {
        "standardized_smell_type": _count_mapping(
            summary.get("smell_type_count_pre") or summary.get("smells_diversity_pre")
        ),
        "raw_smell_label": _count_mapping(summary.get("smells_diversity_pre")),
        "mantyla_category": _count_mapping(summary.get("smells_by_mantyla_pre")),
    }
    snapshots["after"] = {
        "standardized_smell_type": _count_mapping(
            summary.get("smell_type_count_post") or summary.get("smells_diversity_post")
        ),
        "raw_smell_label": _count_mapping(summary.get("smells_diversity_post")),
        "mantyla_category": _count_mapping(summary.get("smells_by_mantyla_post")),
    }
    future_metrics = _as_dict(summary.get("maintainability_future_snapshot_metrics"))
    for label, snapshot_payload_value in future_metrics.items():
        snapshot_summary = _as_dict(_as_dict(snapshot_payload_value).get("summary"))
        if not snapshot_summary:
            continue
        snapshots[str(label)] = {
            "standardized_smell_type": _count_mapping(snapshot_summary.get("smell_type_count")),
            "raw_smell_label": _count_mapping(snapshot_summary.get("smell_count_by_rule")),
            "mantyla_category": _count_mapping(snapshot_summary.get("smell_count_by_mantyla")),
        }
    return snapshots


def _total_smell_count_from_mappings(mappings: dict[str, dict[str, int]]) -> int:
    """Return one total from parallel smell count mappings without double-counting them."""
    totals = [sum(mapping.values()) for mapping in mappings.values() if mapping]
    return max(totals) if totals else 0


def _ordered_string_values(value: Any) -> list[str]:
    """Return de-duplicated string values from scalar or iterable fields."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        candidates = [value]
    values: list[str] = []
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def _maintainability_indicators(maintainability: dict[str, Any]) -> dict[str, Any]:
    """Return the nested maintainability indicator payload when present."""
    indicators = _as_dict(maintainability.get("maintainability_indicators"))
    return indicators if indicators else maintainability


def _maintainability_results_by_snapshot(
    maintainability: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index per-snapshot maintainability result payloads by snapshot label."""
    indicators = _maintainability_indicators(maintainability)
    results_by_snapshot: dict[str, dict[str, Any]] = {}
    for result in _as_list(indicators.get("results") or maintainability.get("results")):
        result_dict = _as_dict(result)
        label = str(result_dict.get("snapshot_label") or "").strip()
        if label:
            results_by_snapshot[label] = result_dict
    return results_by_snapshot


def _tool_names_from_runs(tool_runs: Any) -> list[str]:
    """Return unique tool names from code-smell tool-run payloads."""
    tools: list[str] = []
    for run in _as_list(tool_runs):
        tool = _safe_str(_as_dict(run).get("tool"))
        if tool and tool not in tools:
            tools.append(tool)
    return tools


def _code_smell_tools_field(
    maintainability: dict[str, Any],
    snapshot_result: dict[str, Any],
) -> str | None:
    """Return a compact public tool list for one code-smell result row."""
    indicators = _maintainability_indicators(maintainability)
    tools: list[str] = []
    for value in (
        indicators.get("selected_tools"),
        maintainability.get("selected_tools"),
        snapshot_result.get("selected_tools"),
    ):
        for tool in _ordered_string_values(value):
            if tool not in tools:
                tools.append(tool)
    for tool in _tool_names_from_runs(snapshot_result.get("tool_runs")):
        if tool not in tools:
            tools.append(tool)
    snapshot_tool = _safe_str(snapshot_result.get("tool"))
    if snapshot_tool and snapshot_tool not in tools:
        tools.append(snapshot_tool)
    return " & ".join(tools) if tools else None


def _code_smell_skip_fields(
    snapshot_result: dict[str, Any],
    future_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return skip status and reason from snapshot or future payload metadata."""
    skipped = _safe_bool(future_payload.get("skipped_tool_run"))
    if skipped is None:
        skipped = _safe_bool(snapshot_result.get("skipped_tool_run"))
    skip_reason = _safe_str(future_payload.get("skip_reason") or snapshot_result.get("skip_reason"))
    for run in _as_list(snapshot_result.get("tool_runs")):
        run_dict = _as_dict(run)
        if skipped is None:
            skipped = _safe_bool(run_dict.get("skipped_tool_run"))
        if skipped is None and str(run_dict.get("status") or "").strip().lower() == "skipped":
            skipped = True
        if not skip_reason:
            skip_reason = _safe_str(run_dict.get("skip_reason"))
        if skipped is not None and skip_reason:
            break
    return {
        "skipped_tool_run": skipped if skipped is not None else False,
        "skip_reason": skip_reason,
    }


def _coerce_row(row: dict[str, Any], schema: pa.Schema) -> dict[str, Any]:
    """Coerce a row into exactly the fields expected by one PyArrow schema."""
    coerced: dict[str, Any] = {}
    for field in schema:
        value = row.get(field.name)
        if pa.types.is_timestamp(field.type):
            value = _parse_timestamp_value(value)
        coerced[field.name] = value
    return coerced


class CurationUploadStateStore:
    """SQLite upload and blob de-duplication state."""

    def __init__(self, db_path: Path) -> None:
        """Open the state database and create resumability tables."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
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
            CREATE TABLE IF NOT EXISTS content_blobs (
                cohort TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                first_seen_utc TEXT NOT NULL,
                PRIMARY KEY (cohort, content_sha256)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS global_content_blobs (
                content_sha256 TEXT NOT NULL PRIMARY KEY,
                cohort TEXT NOT NULL,
                first_seen_utc TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def mark_uploaded(self, repo_path: str, local_path: Path) -> None:
        """Record one repo path as uploaded from the local staged file."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO uploaded_files (repo_path, local_path, uploaded_at_utc)
            VALUES (?, ?, ?)
            """,
            (repo_path, str(local_path), _now_iso()),
        )
        self.conn.commit()

    def record_blob_if_new(self, cohort: str, content_sha256: str) -> bool:
        """Return ``True`` only for the first globally seen content hash."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO global_content_blobs (
                content_sha256,
                cohort,
                first_seen_utc
            )
            VALUES (?, ?, ?)
            """,
            (content_sha256, cohort, _now_iso()),
        )
        return cursor.rowcount > 0

    def commit(self) -> None:
        """Commit pending SQLite changes."""
        self.conn.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()


class CurationParquetWriter:
    """Batch and shard curation rows into the requested HF layout."""

    def __init__(
        self,
        *,
        local_output_dir: Path,
        data_subdir: str,
        output_batch_size: int,
        max_files_per_directory: int,
        parquet_compression: str,
        blob_batch_bytes: int,
    ) -> None:
        """Initialize output layout, buffers, and existing batch counters."""
        self.local_output_dir = Path(local_output_dir)
        self.data_root = self.local_output_dir / data_subdir
        self.output_batch_size = max(1, int(output_batch_size))
        self.max_files_per_directory = max(1, int(max_files_per_directory))
        self.parquet_compression = parquet_compression
        self.blob_batch_bytes = max(1024 * 1024, int(blob_batch_bytes))
        self.buffers: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self.buffer_bytes: dict[tuple[str, str, str], int] = defaultdict(int)
        self.batch_counts = self._load_existing_batch_counts()
        self.written_paths: list[Path] = []
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _key(
        self, cohort: str, entity_name: str, partition: str | None = None
    ) -> tuple[str, str, str]:
        """Return the buffer key for cohort/root-scoped entity batches."""
        scope = ROOT_CURATION_SCOPE if entity_name in ROOT_CURATION_ENTITY_NAMES else cohort
        return (scope, entity_name, partition or "")

    def _entity_root(self, cohort: str, entity_name: str, partition: str | None = None) -> Path:
        """Return the output directory for one entity and optional partition."""
        if entity_name in ROOT_CURATION_ENTITY_NAMES:
            root = _curation_dataset_dir(self.data_root) / entity_name
        else:
            root = _curation_dataset_dir(self.data_root / cohort) / entity_name
        if entity_name == ENTITY_SNAPSHOT_FILE_BLOBS and partition:
            root = root / f"hash_prefix={partition}"
        elif partition:
            root = root.joinpath(*[part for part in str(partition).split("/") if part])
        return root

    def _shard_dir(
        self, cohort: str, entity_name: str, batch_idx: int, partition: str | None = None
    ) -> Path:
        """Return the shard directory for one batch index."""
        shard_index = max(0, int(batch_idx) - 1) // self.max_files_per_directory
        return self._entity_root(cohort, entity_name, partition) / f"shard-{shard_index:04d}"

    def _load_existing_batch_counts(self) -> dict[tuple[str, str, str], int]:
        """Detect existing batch numbers so repeated prepares append safely."""
        counts: dict[tuple[str, str, str], int] = {}
        if not self.data_root.exists():
            return counts
        root_curation_dir = _curation_dataset_dir(self.data_root)
        if root_curation_dir.is_dir():
            for entity_name in ROOT_CURATION_ENTITY_NAMES:
                entity_dir = root_curation_dir / entity_name
                if not entity_dir.is_dir():
                    continue
                pattern = f"{ENTITY_FILE_PREFIXES[entity_name]}-*.parquet"
                for parquet_path in entity_dir.rglob(pattern):
                    suffix = parquet_path.stem.rsplit("-", 1)[-1]
                    try:
                        batch_idx = int(suffix)
                    except ValueError:
                        continue
                    key = (ROOT_CURATION_SCOPE, entity_name, "")
                    counts[key] = max(counts.get(key, 0), batch_idx)
        for cohort_dir in self.data_root.iterdir():
            if cohort_dir.name == CURATION_DATASET_SUBDIR:
                continue
            curation_dir = _curation_dataset_dir(cohort_dir)
            if not curation_dir.is_dir():
                continue
            for entity_name in COHORT_CURATION_ENTITY_NAMES:
                entity_dir = curation_dir / entity_name
                if not entity_dir.is_dir():
                    continue
                pattern = f"{ENTITY_FILE_PREFIXES[entity_name]}-*.parquet"
                for parquet_path in entity_dir.rglob(pattern):
                    partition = ""
                    try:
                        relative_parts = parquet_path.relative_to(entity_dir).parts
                    except ValueError:
                        relative_parts = ()
                    if entity_name == ENTITY_SNAPSHOT_FILE_BLOBS and relative_parts:
                        first_part = relative_parts[0]
                        if first_part.startswith("hash_prefix="):
                            partition = first_part.split("=", 1)[1]
                    elif (
                        entity_name == ENTITY_REFACTORING_RESULTS
                        and len(relative_parts) > 2
                        and relative_parts[-2].startswith("shard-")
                    ):
                        partition = "/".join(relative_parts[:-2])
                    suffix = parquet_path.stem.rsplit("-", 1)[-1]
                    try:
                        batch_idx = int(suffix)
                    except ValueError:
                        continue
                    key = (cohort_dir.name, entity_name, partition)
                    counts[key] = max(counts.get(key, 0), batch_idx)
        return counts

    def append(
        self,
        cohort: str,
        entity_name: str,
        row: dict[str, Any],
        *,
        partition: str | None = None,
        row_bytes: int = 0,
    ) -> None:
        """Buffer one row and flush when row or blob-byte thresholds are met."""
        key = self._key(cohort, entity_name, partition)
        self.buffers[key].append(
            _coerce_row(_with_entity_ids(entity_name, row), ENTITY_SCHEMAS[entity_name])
        )
        self.buffer_bytes[key] += max(0, int(row_bytes))
        should_flush = len(self.buffers[key]) >= self.output_batch_size
        if entity_name == ENTITY_SNAPSHOT_FILE_BLOBS:
            should_flush = should_flush or self.buffer_bytes[key] >= self.blob_batch_bytes
        if should_flush:
            self.flush_key(key)

    def flush_key(self, key: tuple[str, str, str]) -> Path | None:
        """Flush one buffered entity/partition batch to parquet."""
        rows = self.buffers.get(key) or []
        if not rows:
            return None
        cohort, entity_name, partition = key
        batch_idx = self.batch_counts.get(key, 0) + 1
        self.batch_counts[key] = batch_idx
        shard_dir = self._shard_dir(cohort, entity_name, batch_idx, partition or None)
        shard_dir.mkdir(parents=True, exist_ok=True)
        file_path = shard_dir / f"{ENTITY_FILE_PREFIXES[entity_name]}-{batch_idx:06d}.parquet"
        table = pa.Table.from_pylist(rows, schema=ENTITY_SCHEMAS[entity_name])
        pq.write_table(table, file_path, compression=self.parquet_compression)
        self.buffers[key] = []
        self.buffer_bytes[key] = 0
        self.written_paths.append(file_path)
        print(
            "[post-processing/upload-curation-data] Wrote parquet batch "
            f"{file_path} with {table.num_rows} rows."
        )
        return file_path

    def flush_all(self) -> list[Path]:
        """Flush all buffered rows and return written parquet paths."""
        paths: list[Path] = []
        for key in sorted(list(self.buffers.keys())):
            path = self.flush_key(key)
            if path is not None:
                paths.append(path)
        return paths


class HFCurationUploadPipeline:
    """Export curation outputs into a local public dataset package."""

    def __init__(
        self,
        *,
        curation_outputs_dir: Path,
        curation_exclude_dirs: Iterable[str],
        topic_classification_outputs_dir: Path,
        topic_classification_top_k_topics: int,
        longitudinal_refactoring_outputs_dir: Path,
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
        curation_schema_version: str,
        blob_batch_bytes: int,
    ) -> None:
        """Initialize source roots, writer state, upload policy, and counters."""
        self.curation_outputs_dir = Path(curation_outputs_dir)
        self.curation_exclude_dirs = tuple(curation_exclude_dirs or ())
        self.topic_classification_outputs_dir = Path(topic_classification_outputs_dir)
        self.topic_classification_top_k_topics = min(
            5,
            max(1, int(topic_classification_top_k_topics or 5)),
        )
        self.longitudinal_refactoring_outputs_dir = Path(longitudinal_refactoring_outputs_dir)
        self.target_huggingface_repo_id = str(target_huggingface_repo_id or "")
        self.huggingface_token = str(huggingface_token or "")
        self.local_output_dir = Path(local_output_dir)
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
        self.curation_schema_version = curation_schema_version
        self.data_root = self.local_output_dir / self.standardized_data_subdir
        self.writer = CurationParquetWriter(
            local_output_dir=self.local_output_dir,
            data_subdir=self.standardized_data_subdir,
            output_batch_size=self.output_batch_size,
            max_files_per_directory=self.max_files_per_directory,
            parquet_compression=self.parquet_compression,
            blob_batch_bytes=blob_batch_bytes,
        )
        self.state_store = CurationUploadStateStore(self.local_output_dir / state_db_filename)
        self.summary: dict[str, Any] = {
            "aggregate_files_discovered": 0,
            "aggregate_files_selected": 0,
            "aggregate_parse_failures": 0,
            "aggregate_stable_pr_duplicate_files_skipped": 0,
            "aggregate_legacy_pr_dedup_keys_used": 0,
            "topic_output_directories_discovered": 0,
            "topic_outputs_selected": 0,
            "topic_output_parse_failures": 0,
            "repository_topic_classification_rows": 0,
            "repository_topic_duplicate_rows_skipped": 0,
            "longitudinal_refactoring_output_directories_discovered": 0,
            "longitudinal_refactoring_outputs_selected": 0,
            "longitudinal_refactoring_parse_failures": 0,
            "longitudinal_refactoring_rows": 0,
            "longitudinal_refactoring_operations": 0,
            "longitudinal_refactoring_duplicate_snapshot_rows_skipped": 0,
            "snapshot_file_refs": 0,
            "snapshot_file_blobs": 0,
            "snapshot_blob_duplicates_skipped": 0,
            "uploaded_files": 0,
            "cohorts": {},
        }
        self._cohort_repository_file_lists: dict[tuple[str, str], dict[str, Any]] = {}
        self._repository_file_lists_by_repo: dict[str, dict[str, Any]] = {}
        self._emitted_repository_metadata: set[str] = set()
        self._emitted_repository_topic_classifications: set[str] = set()
        self._sampling_metadata_by_pr: dict[tuple[str, str], dict[str, Any]] = {}
        self._consecutive_upload_failures = 0

    def _cohort_stats(self, cohort: str) -> dict[str, Any]:
        """Return mutable summary counters for one cohort."""
        cohorts = self.summary["cohorts"]
        if cohort not in cohorts:
            cohorts[cohort] = {
                "pr_count": 0,
                "repository_keys": set(),
                "additions_sum": 0,
                "deletions_sum": 0,
                "refactoring_pr_count": 0,
                "code_smell_pr_count": 0,
                "refactoring_operation_count": 0,
                "code_smell_count": 0,
                "future_snapshot_available_counts": {label: 0 for label in FUTURE_SNAPSHOT_LABELS},
                "future_snapshot_refactoring_operation_counts": {
                    label: 0 for label in FUTURE_SNAPSHOT_LABELS
                },
                "future_snapshot_code_smell_counts": {label: 0 for label in FUTURE_SNAPSHOT_LABELS},
                "topic_classified_repository_count": 0,
                "topic_primary_topic_count": 0,
                "longitudinal_refactoring_by_input_mode": defaultdict(
                    lambda: {
                        "pr_keys": set(),
                        "snapshot_results": 0,
                        "completed_snapshot_results": 0,
                        "prs_with_future_refactoring": set(),
                        "prs_with_observed_zero_future_refactoring": set(),
                        "refactoring_operations": 0,
                        "completed_by_label": defaultdict(int),
                    }
                ),
                "source_run_ids": set(),
                "entity_rows": defaultdict(int),
            }
        return cohorts[cohort]

    def _entity_row(self, cohort: str, entity_name: str) -> None:
        """Increment one cohort/entity row counter."""
        self._cohort_stats(cohort)["entity_rows"][entity_name] += 1

    def _append_entity(
        self,
        cohort: str,
        entity_name: str,
        row: dict[str, Any],
        *,
        partition: str | None = None,
        row_bytes: int = 0,
    ) -> None:
        """Append one entity row through the writer and update counters."""
        self.writer.append(cohort, entity_name, row, partition=partition, row_bytes=row_bytes)
        self._entity_row(cohort, entity_name)

    def _discover_aggregate_refs(self) -> list[AggregateRef]:
        """Discover newest curation aggregate files, de-duplicated by PR."""
        selected: dict[tuple[str, ...], AggregateRef] = {}
        cohort_dirs = discover_cohort_dirs(
            self.curation_outputs_dir,
            self.curation_exclude_dirs,
            log=lambda message: print(f"[post-processing/upload-curation-data] {message}"),
        )
        for cohort_dir in cohort_dirs:
            source_run_id = cohort_dir.name
            rank = _source_run_rank(cohort_dir)
            aggregate_paths = discover_aggregate_metric_paths(cohort_dir)
            self.summary["aggregate_files_discovered"] += len(aggregate_paths)
            for aggregate_path in aggregate_paths:
                payload = read_json_object(
                    aggregate_path,
                    description="curation aggregate metrics JSON",
                    log=lambda message: print(f"[post-processing/upload-curation-data] {message}"),
                )
                if payload is None:
                    self.summary["aggregate_parse_failures"] += 1
                    continue
                cohort = _cohort_from_payload(payload, cohort_dir.name)
                pull_request_key = _pull_request_key(payload)
                repository_key = _repository_key_from_aggregate(payload)
                pr_number = _safe_int(_as_dict(payload.get("pr")).get("number"))
                stable_dedup_key = _stable_curation_pr_dedup_key(
                    payload=payload,
                    pull_request_key=pull_request_key,
                    pr_number=pr_number,
                )
                if stable_dedup_key:
                    dedup_key = ("stable-base-repository-pr", stable_dedup_key)
                else:
                    dedup_key = ("legacy-cohort-pr", cohort, pull_request_key)
                    self.summary["aggregate_legacy_pr_dedup_keys_used"] += 1
                ref = AggregateRef(
                    path=aggregate_path,
                    cohort=cohort,
                    source_run_id=source_run_id,
                    pull_request_key=pull_request_key,
                    repository_key=repository_key,
                    pr_number=pr_number,
                    rank=rank,
                )
                existing = selected.get(dedup_key)
                if existing is None or ref.rank > existing.rank:
                    if existing is not None and stable_dedup_key:
                        self.summary["aggregate_stable_pr_duplicate_files_skipped"] += 1
                    selected[dedup_key] = ref
                elif stable_dedup_key:
                    self.summary["aggregate_stable_pr_duplicate_files_skipped"] += 1
        refs = sorted(
            selected.values(),
            key=lambda ref: (
                ref.cohort,
                ref.repository_key or "",
                ref.pr_number or -1,
                str(ref.path),
            ),
        )
        self.summary["aggregate_files_selected"] = len(refs)
        print(
            "[post-processing/upload-curation-data] Selected "
            f"{len(refs)} newest aggregate files from "
            f"{self.summary['aggregate_files_discovered']} discovered aggregate files."
        )
        return refs

    def _load_repository_file_lists(self) -> None:
        """Load hydrated repository file-list metadata for repository rows."""
        self._cohort_repository_file_lists = {}
        self._repository_file_lists_by_repo = {}
        for cohort_dir in discover_cohort_dirs(
            self.curation_outputs_dir, self.curation_exclude_dirs
        ):
            source_run_id = cohort_dir.name
            for path in _discover_repository_file_list_paths(cohort_dir):
                payload = read_json_object(path, description="repository file list JSON")
                if payload is None:
                    continue
                repository_key = None
                owner = _safe_str(payload.get("repository_owner"))
                name = _safe_str(payload.get("repository_name"))
                if owner and name:
                    repository_key = normalize_repository_key(owner, name)
                if repository_key is None:
                    try:
                        relative = path.relative_to(path.parents[2])
                        parts = relative.parts
                        if len(parts) >= 2:
                            repository_key = normalize_repository_key(parts[0], parts[1])
                    except Exception:
                        repository_key = None
                if repository_key is None:
                    continue
                # Cohort names come from file-list payload roots when available.
                cohort = source_run_id
                for root in candidate_curation_roots(cohort_dir):
                    try:
                        path.relative_to(root)
                    except ValueError:
                        continue
                    cohort = _cohort_from_jsonl_path(
                        next(
                            iter(root.glob("sampled_prs_*.jsonl")),
                            Path(f"sampled_prs_{root.name}.jsonl"),
                        ),
                        root.name,
                    )
                    break
                file_list_ref = {
                    "path": path,
                    "payload": payload,
                }
                self._cohort_repository_file_lists[(safe_cohort_name(cohort), repository_key)] = (
                    file_list_ref
                )
                self._repository_file_lists_by_repo[repository_key] = file_list_ref

    def _base_context(self, ref: AggregateRef, payload: dict[str, Any]) -> dict[str, Any]:
        """Return shared PR/repository fields used by cohort-scoped rows."""
        pr = _as_dict(payload.get("pr"))
        repository_key = ref.repository_key or _repository_key_from_aggregate(payload)
        owner, name = _owner_name_from_repo_key(repository_key)
        return {
            "cohort": ref.cohort,
            "source_run_id": ref.source_run_id,
            "pull_request_key": ref.pull_request_key,
            "repository_key": repository_key,
            "repository_owner": owner,
            "repository_name": name,
            "pr_number": _safe_int(pr.get("number")),
            "pr_url": _safe_str(pr.get("url")),
        }

    def _curation_pull_request_row(
        self, ref: AggregateRef, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the public ``CuratedPullRequests`` row for one aggregate."""
        pr = _as_dict(payload.get("pr"))
        ctx = self._base_context(ref, payload)
        sampling_metadata = dict(
            _as_dict(pr.get("sampling_metadata"))
            or self._sampling_metadata_by_pr.get((ref.cohort, ref.pull_request_key), {})
        )
        if not sampling_metadata:
            for key in (
                "sampling_language_bucket",
                "selected_language_bucket",
                "sampling_time_bucket",
                "time_bucket_sampling",
                "time_bucket_day",
                "sampling_popularity_bucket",
                "popularity_bucket",
            ):
                if key in pr:
                    sampling_metadata[key] = pr[key]
        base_repository_full = _as_dict(pr.get("base_repository_full"))
        base_repository = _as_dict(pr.get("base_repository"))
        head_repository_full = _as_dict(pr.get("head_repository_full"))
        head_repository = _as_dict(pr.get("head_repository"))
        base_repo_key = _repo_key_from_repository_payload(
            base_repository_full
        ) or _repo_key_from_repository_payload(base_repository)
        head_repo_key = _repo_key_from_repository_payload(
            head_repository_full
        ) or _repo_key_from_repository_payload(head_repository)
        return {
            **ctx,
            "pr_id": _safe_str(pr.get("id")),
            "author": _author_label(pr),
            "base_repository_key": base_repo_key,
            "base_repository_id": _repository_payload_field(
                base_repository_full, base_repository, "id"
            ),
            "base_repository_url": _repository_payload_field(
                base_repository_full, base_repository, "url"
            ),
            "head_repository_key": head_repo_key,
            "head_repository_id": _repository_payload_field(
                head_repository_full, head_repository, "id"
            ),
            "head_repository_url": _repository_payload_field(
                head_repository_full, head_repository, "url"
            ),
            "created_at": _safe_str(pr.get("created_at")),
            "merged_at": _safe_str(pr.get("merged_at")),
            "closed_at": _safe_str(pr.get("closed_at")),
            "dominant_language": _safe_str(
                pr.get("pr_primary_language_effective") or pr.get("primary_language_effective")
            ),
            "file_languages_json": _json_dumps(_as_list(pr.get("file_languages"))),
            "authored_by_agent": _safe_bool(pr.get("authored_by_agent")),
            "author_agent": _safe_str(pr.get("author_agent") or pr.get("discovered_agent")),
            "longitudinal_selected": bool(pr.get("longitudinal_selected")),
            "processing_status": _safe_str(_as_dict(payload.get("metrics")).get("status")),
            "sampling_language_bucket": _sampling_language_bucket(sampling_metadata),
            "sampling_time_bucket": _sampling_time_bucket(sampling_metadata),
            "sampling_popularity_bucket": _sampling_popularity_bucket(sampling_metadata),
            "additions": _safe_int(pr.get("additions")),
            "deletions": _safe_int(pr.get("deletions")),
            "changed_files": _safe_int(pr.get("changed_files")),
        }

    def _emit_repository_metadata(
        self,
        ref: AggregateRef,
        payload: dict[str, Any],
    ) -> None:
        """Emit root-scoped repository metadata once per stable repository."""
        repository_key = ref.repository_key or _repository_key_from_aggregate(payload)
        if not repository_key:
            return
        owner, name = _owner_name_from_repo_key(repository_key)
        repo_metadata = _as_dict(payload.get("repository_metadata"))
        pr = _as_dict(payload.get("pr"))
        stable_repository_id = _stable_base_repository_id_from_aggregate(payload)
        metadata_identity = _repository_metadata_identity_from_parts(
            stable_repository_id=stable_repository_id,
            repository_key=repository_key,
        )
        if not metadata_identity:
            return
        if metadata_identity in self._emitted_repository_metadata:
            return
        self._emitted_repository_metadata.add(metadata_identity)
        file_list_ref = self._cohort_repository_file_lists.get(
            (ref.cohort, repository_key)
        ) or self._repository_file_lists_by_repo.get(repository_key)
        file_list_payload = _as_dict(file_list_ref.get("payload")) if file_list_ref else {}
        readme_text = _safe_str(repo_metadata.get("readme"))
        labels = repo_metadata.get("repository_labels")
        if labels is None:
            labels = pr.get("labels")
        row = {
            "repository_metadata_id": _repository_metadata_id(metadata_identity),
            "source_run_id": ref.source_run_id,
            "repository_key": repository_key,
            "stable_repository_id": stable_repository_id,
            "repository_owner": owner,
            "repository_name": name,
            "popularity_label": _popularity_label(
                repo_metadata.get("popularity_label")
                or pr.get("popularity_label")
                or repo_metadata.get("popularity_bucket")
                or pr.get("popularity_bucket")
            ),
            "stargazer_count": _repository_stargazer_count(repo_metadata, pr),
            "labels": _json_dumps(labels),
            "readme": readme_text,
            "readme_sha256": _sha256_text(readme_text),
            "file_list_source_commit": _safe_str(file_list_payload.get("source_commit")),
            "file_list_generated_at": _safe_str(
                file_list_payload.get("generated_at_utc")
            ),
            "file_count": _safe_int(file_list_payload.get("file_count")),
            "file_list": _json_dumps(_as_list(file_list_payload.get("files"))),
        }
        self._append_entity(ref.cohort, ENTITY_REPOSITORY_ADDITIONAL_METADATA, row)

    def _emit_refactoring_results(self, ref: AggregateRef, payload: dict[str, Any]) -> None:
        """Emit original PR and curation-provided future refactoring summaries."""
        ctx = self._base_context(ref, payload)
        metrics = _as_dict(payload.get("metrics"))
        summary = _as_dict(_as_dict(metrics.get("refactoring")).get("summary"))
        snapshots = _as_dict(_as_dict(payload.get("hydration")).get("snapshots"))
        after_snapshot = _as_dict(snapshots.get("after"))
        operations = original_pr_refactoring_operations(payload, aggregate_path=ref.path)
        after_snapshot_available = bool(after_snapshot.get("commit") or after_snapshot.get("path"))
        curation_partition = _refactoring_results_partition(
            REFACTORING_RESULT_SOURCE_CURATION,
            REFACTORING_INPUT_MODE_CURATION,
        )
        if after_snapshot_available:
            stored_refactor_type_counts = _count_mapping(summary.get("refactor_type_count"))
            refactor_type_counts = (
                _current_refactoring_count_mapping(stored_refactor_type_counts)
                or stored_refactor_type_counts
            )
            murphy_hill_counts = _count_mapping(summary.get("refactor_murphyhill_count"))
            row = {
                **ctx,
                "result_source": REFACTORING_RESULT_SOURCE_CURATION,
                "input_mode": REFACTORING_INPUT_MODE_CURATION,
                "snapshot_label": "pr",
                "transition_label": "before_to_after",
                "snapshot_key": _snapshot_key(ref.pull_request_key, "after"),
                "snapshot_commit": _safe_str(after_snapshot.get("commit")),
                "result_status": _safe_str(metrics.get("status")),
                "refop_retention_rate": None,
                "future_lines_touched": None,
                "touching_commit_count": None,
                "refop_count": _safe_int(summary.get("refactor_count")) or len(operations),
                "added_lines": _safe_int(summary.get("refactor_added_lines")),
                "removed_lines": _safe_int(summary.get("refactor_removed_lines")),
                "magnitude_lines": _safe_int(summary.get("refactor_magnitude_lines")),
                "magnitude_files": _safe_int(summary.get("refactor_magnitude_files")),
                "diversity": _safe_int(summary.get("refactor_diversity")),
                "refactor_type_count_json": _json_dumps(refactor_type_counts),
                **_murphy_hill_count_columns(murphy_hill_counts, refactor_type_counts),
                **_refactoring_mapping_fields(operations, refactor_type_counts),
                "operations_json": _json_dumps(operations),
            }
            self._append_entity(
                ref.cohort,
                ENTITY_REFACTORING_RESULTS,
                row,
                partition=curation_partition,
            )

        future_metrics = _as_dict(summary.get("refactor_future_snapshot_metrics"))
        future_snapshots = _as_dict(snapshots.get("future"))
        for label, future_payload_value in sorted(future_metrics.items()):
            future_payload = _as_dict(future_payload_value)
            future_snapshot = _as_dict(future_snapshots.get(label))
            snapshot_available = (
                future_snapshot.get("available")
                if future_snapshot.get("available") is not None
                else bool(future_snapshot.get("commit"))
            )
            if not snapshot_available:
                continue
            future_operations = _as_list(future_payload.get("operations"))
            stored_refactor_type_counts = _count_mapping(future_payload.get("refactor_type_count"))
            refactor_type_counts = (
                _current_refactoring_count_mapping(stored_refactor_type_counts)
                or stored_refactor_type_counts
            )
            murphy_hill_counts = _count_mapping(future_payload.get("refactor_murphyhill_count"))
            future_row = {
                **ctx,
                "result_source": REFACTORING_RESULT_SOURCE_CURATION,
                "input_mode": REFACTORING_INPUT_MODE_CURATION,
                "snapshot_label": str(label),
                "transition_label": f"after_to_{label}",
                "snapshot_key": _snapshot_key(ref.pull_request_key, str(label)),
                "snapshot_commit": _safe_str(
                    future_snapshot.get("commit") or future_payload.get("snapshot_commit")
                ),
                "result_status": _safe_str(
                    future_payload.get("status") or future_payload.get("result_status")
                ),
                "refop_retention_rate": _safe_float(
                    _as_dict(future_payload.get("retention")).get("retention_rate")
                ),
                "future_lines_touched": (
                    _safe_int(
                        _as_dict(future_payload.get("future_impact")).get(
                            "touched_refactoring_zone_lines_count"
                        )
                    )
                    or _safe_int(
                        _as_dict(future_payload.get("future_impact")).get("touched_pr_lines_count")
                    )
                    or _safe_int(
                        _as_dict(future_payload.get("pr_changed_line_future_impact")).get(
                            "touched_pr_lines_count"
                        )
                    )
                ),
                "touching_commit_count": (
                    _safe_int(
                        _as_dict(future_payload.get("future_impact")).get(
                            "touching_commits_count"
                        )
                    )
                    or _safe_int(
                        _as_dict(future_payload.get("pr_changed_line_future_impact")).get(
                            "touching_commits_count"
                        )
                    )
                ),
                "refop_count": _safe_int(future_payload.get("refactor_count")),
                "added_lines": _safe_int(future_payload.get("refactor_added_lines")),
                "removed_lines": _safe_int(future_payload.get("refactor_removed_lines")),
                "magnitude_lines": _safe_int(
                    future_payload.get("refactor_magnitude_lines")
                ),
                "magnitude_files": _safe_int(
                    future_payload.get("refactor_magnitude_files")
                ),
                "diversity": _safe_int(future_payload.get("refactor_diversity")),
                "refactor_type_count_json": _json_dumps(refactor_type_counts),
                **_murphy_hill_count_columns(murphy_hill_counts, refactor_type_counts),
                **_refactoring_mapping_fields(future_operations, refactor_type_counts),
                "operations_json": _json_dumps(future_operations),
            }
            self._append_entity(
                ref.cohort,
                ENTITY_REFACTORING_RESULTS,
                future_row,
                partition=curation_partition,
            )

    def _emit_maintainability_results(self, ref: AggregateRef, payload: dict[str, Any]) -> None:
        """Emit Multimetric plus custom duplication values for each snapshot."""
        ctx = self._base_context(ref, payload)
        summary = _as_dict(
            _as_dict(_as_dict(payload.get("metrics")).get("maintainability")).get("summary")
        )
        snapshots = _as_dict(_as_dict(payload.get("hydration")).get("snapshots"))
        snapshot_measures = _as_dict(summary.get("snapshot_measures"))
        for label, field_suffix in (("before", "pre"), ("after", "post")):
            measures = _as_dict(snapshot_measures.get(label))
            snapshot = _as_dict(snapshots.get(label))
            snapshot_available = bool(snapshot.get("commit") or snapshot.get("path"))
            if not snapshot_available:
                continue
            row = {
                **ctx,
                "snapshot_label": label,
                "snapshot_key": _snapshot_key(ref.pull_request_key, label),
                "snapshot_commit": _safe_str(snapshot.get("commit")),
                "loc": _first_float(measures.get("loc"), measures.get("ncloc")),
                "cyclomatic_complexity": _first_float(
                    summary.get(f"cyclomatic_complexity_{field_suffix}")
                    or measures.get("cyclomatic_complexity"),
                    measures.get("complexity"),
                ),
                "maintainability_index": _first_float(
                    summary.get(f"maintainability_index_{field_suffix}")
                    or measures.get("maintainability_index"),
                ),
                "duplicated_lines_density": _first_float(
                    measures.get("duplicated_lines_density")
                ),
                "comment_ratio": _first_float(
                    measures.get("comment_ratio"),
                    measures.get("comment_lines_density"),
                ),
                "halstead_volume": _first_float(measures.get("halstead_volume")),
                "fan_out": _first_float(
                    measures.get("fan_out"),
                    measures.get("fanout_external"),
                    measures.get("fanout_internal"),
                ),
                "smell_count": _safe_int(summary.get(f"smells_total_{field_suffix}")),
                "measures_json": _json_dumps(measures),
                "summary_json": None,
            }
            self._append_entity(ref.cohort, ENTITY_MAINTAINABILITY_RESULTS, row)

        future_metrics = _as_dict(summary.get("maintainability_future_snapshot_metrics"))
        future_snapshots = _as_dict(snapshots.get("future"))
        for label, future_value in sorted(future_metrics.items()):
            future_payload = _as_dict(future_value)
            future_summary = _as_dict(future_payload.get("summary"))
            future_measures = _as_dict(future_payload.get("measures"))
            future_snapshot = _as_dict(future_snapshots.get(label))
            snapshot_available = (
                future_payload.get("available")
                if future_payload.get("available") is not None
                else bool(future_snapshot.get("commit"))
            )
            if not snapshot_available:
                continue
            smell_count = _safe_int(future_summary.get("smell_count"))
            row = {
                **ctx,
                "snapshot_label": str(label),
                "snapshot_key": _snapshot_key(ref.pull_request_key, str(label)),
                "snapshot_commit": _safe_str(
                    future_payload.get("snapshot_commit") or future_snapshot.get("commit")
                ),
                "loc": _first_float(
                    future_measures.get("loc"),
                    future_measures.get("ncloc"),
                    _future_metric_value(future_payload, "loc"),
                ),
                "cyclomatic_complexity": _first_float(
                    future_measures.get("cyclomatic_complexity"),
                    future_measures.get("complexity"),
                    _future_metric_value(future_payload, "cyclomatic_complexity"),
                ),
                "maintainability_index": _first_float(
                    future_measures.get("maintainability_index"),
                    _future_metric_value(future_payload, "maintainability_index"),
                ),
                "duplicated_lines_density": _first_float(
                    future_measures.get("duplicated_lines_density"),
                    _future_metric_value(future_payload, "duplicated_lines_density"),
                ),
                "comment_ratio": _first_float(
                    future_measures.get("comment_ratio"),
                    future_measures.get("comment_lines_density"),
                    _future_metric_value(future_payload, "comment_ratio"),
                ),
                "halstead_volume": _first_float(
                    future_measures.get("halstead_volume"),
                    _future_metric_value(future_payload, "halstead_volume"),
                ),
                "fan_out": _first_float(
                    future_measures.get("fan_out"),
                    future_measures.get("fanout_external"),
                    future_measures.get("fanout_internal"),
                    _future_metric_value(future_payload, "fanout_external"),
                    _future_metric_value(future_payload, "fanout_internal"),
                ),
                "smell_count": smell_count,
                "measures_json": _json_dumps(future_measures),
                "summary_json": _json_dumps(future_summary),
            }
            self._append_entity(ref.cohort, ENTITY_MAINTAINABILITY_RESULTS, row)

    def _emit_code_smell_results(self, ref: AggregateRef, payload: dict[str, Any]) -> None:
        """Emit code-smell summary rows derived from maintainability output."""
        ctx = self._base_context(ref, payload)
        maintainability = _as_dict(_as_dict(payload.get("metrics")).get("maintainability"))
        summary = _as_dict(maintainability.get("summary"))
        snapshots = _as_dict(_as_dict(payload.get("hydration")).get("snapshots"))
        future_metrics = _as_dict(summary.get("maintainability_future_snapshot_metrics"))
        future_snapshots = _as_dict(snapshots.get("future"))
        results_by_snapshot = _maintainability_results_by_snapshot(maintainability)
        for snapshot_label, mappings in sorted(_smell_snapshot_mappings(payload).items()):
            snapshot_key = _snapshot_key(ref.pull_request_key, snapshot_label)
            snapshot = _as_dict(snapshots.get(snapshot_label))
            snapshot_available = bool(snapshot.get("commit") or snapshot.get("path"))
            snapshot_commit = _safe_str(snapshot.get("commit"))
            result_status = _safe_str(maintainability.get("status"))
            snapshot_result = _as_dict(results_by_snapshot.get(snapshot_label))
            future_payload: dict[str, Any] = {}
            smell_count: int | None = None
            if snapshot_label in {"before", "after"}:
                field_suffix = "pre" if snapshot_label == "before" else "post"
                smell_count = _safe_int(summary.get(f"smells_total_{field_suffix}"))
            else:
                future_payload = _as_dict(future_metrics.get(snapshot_label))
                future_summary = _as_dict(future_payload.get("summary"))
                future_measures = _as_dict(future_payload.get("measures"))
                future_snapshot = _as_dict(future_snapshots.get(snapshot_label))
                future_available = _safe_bool(future_payload.get("available"))
                snapshot_available = (
                    future_available
                    if future_available is not None
                    else bool(future_snapshot.get("commit"))
                )
                snapshot_commit = _safe_str(
                    future_payload.get("snapshot_commit") or future_snapshot.get("commit")
                )
                result_status = _safe_str(
                    future_payload.get("status") or future_summary.get("status")
                )
                smell_count = _safe_int(future_summary.get("smell_count"))
                if smell_count is None:
                    smell_count = _safe_int(future_measures.get("code_smells"))
            if not snapshot_available:
                continue
            if smell_count is None:
                smell_count = _total_smell_count_from_mappings(mappings)
            raw_smell_counts = mappings.get("raw_smell_label") or {}
            stored_standardized_smell_counts = mappings.get("standardized_smell_type") or {}
            standardized_smell_counts = (
                _current_standardized_smell_count_mapping(
                    raw_smell_counts,
                    stored_standardized_smell_counts,
                )
                or stored_standardized_smell_counts
            )
            mantyla_category_counts = mappings.get("mantyla_category") or {}
            raw_to_standardized_smells = _raw_to_standardized_smell_mapping(
                raw_smell_counts,
                standardized_smell_counts,
            )
            standardized_to_mantyla_categories = _standardized_smell_to_mantyla_category_mapping(
                standardized_smell_counts,
                raw_to_standardized_smells,
            )
            row = {
                **ctx,
                "snapshot_label": snapshot_label,
                "snapshot_key": snapshot_key,
                "snapshot_commit": snapshot_commit,
                "tools": _code_smell_tools_field(maintainability, snapshot_result),
                "result_status": result_status,
                **_code_smell_skip_fields(snapshot_result, future_payload),
                "smell_count": smell_count,
                "raw_smell_count_json": _json_dumps(raw_smell_counts),
                "standardized_smell_type_count_json": _json_dumps(standardized_smell_counts),
                "raw_smell_to_standardized_smell_json": _json_dumps(raw_to_standardized_smells),
                "standardized_smell_to_mantyla_category_json": _json_dumps(
                    standardized_to_mantyla_categories
                ),
                **_mantyla_count_columns(mantyla_category_counts, standardized_smell_counts),
            }
            self._append_entity(ref.cohort, ENTITY_CODE_SMELL_RESULTS, row)

    def _iter_snapshot_files(self, snapshot_dir: Path) -> Iterator[Path]:
        """Yield source files from a materialized snapshot, excluding tool output."""
        ignored_dir_names = {".git", "__pycache__", "diff", "refactoring", "maintainability"}
        for path in sorted(Path(snapshot_dir).rglob("*")):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(snapshot_dir).parts
            if any(part in ignored_dir_names for part in relative_parts[:-1]):
                continue
            yield path

    def _content_payload(self, content: bytes) -> tuple[str | None, str | None, str, bool]:
        """Return UTF-8 text or base64 bytes for snapshot blob storage."""
        try:
            text = content.decode("utf-8")
            return text, None, "utf-8", False
        except UnicodeDecodeError:
            return None, base64.b64encode(content).decode("ascii"), "base64", True

    def _emit_snapshot_manifests_and_blobs(
        self, ref: AggregateRef, payload: dict[str, Any]
    ) -> None:
        """Emit snapshot manifests, file references, and de-duplicated blobs."""
        ctx = self._base_context(ref, payload)
        hydration = _as_dict(payload.get("hydration"))
        availability_by_label = {
            str(record.get("snapshot_label")): record
            for record in _future_snapshot_availability_records(payload)
            if str(record.get("snapshot_label") or "").strip()
        }
        for label, snapshot_value, snapshot_kind in _snapshot_payloads(
            hydration,
            future_labels=availability_by_label.keys(),
        ):
            snapshot = _as_dict(snapshot_value)
            availability = _as_dict(availability_by_label.get(label))
            snapshot_key = _snapshot_key(ref.pull_request_key, label)
            resolved_path = resolve_curation_artifact_path(ref.path, snapshot.get("path"))
            available = bool(snapshot.get("commit") or resolved_path)
            if snapshot.get("available") is not None:
                available = bool(snapshot.get("available"))
            if availability.get("available") is not None:
                available = bool(availability.get("available"))
            file_count = 0
            if available and resolved_path and resolved_path.exists():
                for file_path in self._iter_snapshot_files(resolved_path):
                    try:
                        content = file_path.read_bytes()
                    except OSError:
                        continue
                    file_count += 1
                    content_sha256 = _sha256_bytes(content)
                    content_text, content_bytes_base64, encoding, is_binary = self._content_payload(
                        content
                    )
                    relative_path = _normalize_path(file_path.relative_to(resolved_path).as_posix())
                    blob_new = self.state_store.record_blob_if_new(ref.cohort, content_sha256)
                    if blob_new:
                        blob_row = {
                            "cohort": ref.cohort,
                            "source_run_id": ref.source_run_id,
                            "content_sha256": content_sha256,
                            "hash_prefix": content_sha256[:2],
                            "content_size_bytes": len(content),
                            "content_encoding": encoding,
                            "content_text": content_text,
                            "content_bytes_base64": content_bytes_base64,
                            "is_binary": is_binary,
                        }
                        self._append_entity(
                            ref.cohort,
                            ENTITY_SNAPSHOT_FILE_BLOBS,
                            blob_row,
                            partition=content_sha256[:2],
                            row_bytes=len(content),
                        )
                        self.summary["snapshot_file_blobs"] += 1
                    else:
                        self.summary["snapshot_blob_duplicates_skipped"] += 1
                    ref_row = {
                        "cohort": ref.cohort,
                        "source_run_id": ref.source_run_id,
                        "snapshot_key": snapshot_key,
                        "snapshot_label": label,
                        "pull_request_key": ref.pull_request_key,
                        "repository_key": ref.repository_key,
                        "path": relative_path,
                        "content_sha256": content_sha256,
                        "content_size_bytes": len(content),
                        "content_encoding": encoding,
                        "is_binary": is_binary,
                    }
                    self._append_entity(ref.cohort, ENTITY_SNAPSHOT_FILE_REFS, ref_row)
                    self.summary["snapshot_file_refs"] += 1

            def pick_manifest_value(*keys: str) -> Any:
                for key in keys:
                    for source in (snapshot, availability):
                        value = source.get(key)
                        if value is not None:
                            return value
                return None

            missing_reason = _safe_str(
                snapshot.get("missing_reason") or availability.get("missing_reason")
            )
            if not available and snapshot_kind == "future" and not missing_reason:
                missing_reason = "missing_snapshot_unknown"
            manifest_row = {
                **ctx,
                "snapshot_label": label,
                "snapshot_key": snapshot_key,
                "available": available,
                "snapshot_commit": _safe_str(
                    snapshot.get("commit") or availability.get("snapshot_commit")
                ),
                "target_offset_days": _target_offset_days(label),
                "target_timestamp": _safe_str(
                    snapshot.get("target_timestamp") or availability.get("target_timestamp")
                ),
                "repository_observation_cutoff": _safe_str(
                    snapshot.get("repository_observation_cutoff")
                    or availability.get("repository_observation_cutoff")
                ),
                "file_availability_status": _safe_str(
                    pick_manifest_value("file_availability_status")
                ),
                "missing_reason": missing_reason,
                "missing_files_json": _json_dumps(_as_list(pick_manifest_value("missing_files"))),
                "deleted_files_json": _json_dumps(_as_list(pick_manifest_value("deleted_files"))),
                "renamed_files_json": _json_dumps(_as_list(pick_manifest_value("renamed_files"))),
                "unknown_missing_files_json": _json_dumps(
                    _as_list(pick_manifest_value("unknown_missing_files"))
                ),
            }
            self._append_entity(ref.cohort, ENTITY_SNAPSHOT_MANIFESTS, manifest_row)
        self.state_store.commit()

    def _process_aggregate(self, ref: AggregateRef) -> None:
        """Read one curation aggregate and emit all directly derived entities."""
        payload = read_json_object(
            ref.path,
            description="selected curation aggregate metrics JSON",
            log=lambda message: print(f"[post-processing/upload-curation-data] {message}"),
        )
        if payload is None:
            self.summary["aggregate_parse_failures"] += 1
            return
        row = self._curation_pull_request_row(ref, payload)
        self._append_entity(ref.cohort, ENTITY_CURATION_PULL_REQUESTS, row)
        stats = self._cohort_stats(ref.cohort)
        stats["pr_count"] += 1
        if ref.repository_key:
            stats["repository_keys"].add(ref.repository_key)
        stats["source_run_ids"].add(ref.source_run_id)
        stats["additions_sum"] += int(row.get("additions") or 0)
        stats["deletions_sum"] += int(row.get("deletions") or 0)

        refactor_summary = _as_dict(
            _as_dict(_as_dict(payload.get("metrics")).get("refactoring")).get("summary")
        )
        refactor_count = _safe_int(refactor_summary.get("refactor_count")) or 0
        stats["refactoring_operation_count"] += refactor_count
        if refactor_count > 0:
            stats["refactoring_pr_count"] += 1
        future_refactor_metrics = _as_dict(refactor_summary.get("refactor_future_snapshot_metrics"))
        for label, future_payload_value in future_refactor_metrics.items():
            label = str(label)
            if label not in stats["future_snapshot_refactoring_operation_counts"]:
                continue
            future_payload = _as_dict(future_payload_value)
            stats["future_snapshot_refactoring_operation_counts"][label] += (
                _safe_int(future_payload.get("refactor_count")) or 0
            )
        maintainability_summary = _as_dict(
            _as_dict(_as_dict(payload.get("metrics")).get("maintainability")).get("summary")
        )
        code_smell_count = _safe_int(maintainability_summary.get("smells_total_post")) or 0
        stats["code_smell_count"] += code_smell_count
        if code_smell_count > 0:
            stats["code_smell_pr_count"] += 1
        future_maintainability_metrics = _as_dict(
            maintainability_summary.get("maintainability_future_snapshot_metrics")
        )
        for label, future_payload_value in future_maintainability_metrics.items():
            label = str(label)
            if label not in stats["future_snapshot_code_smell_counts"]:
                continue
            future_payload = _as_dict(future_payload_value)
            future_summary = _as_dict(future_payload.get("summary"))
            future_measures = _as_dict(future_payload.get("measures"))
            stats["future_snapshot_code_smell_counts"][label] += (
                _safe_int(future_summary.get("smell_count"))
                or _safe_int(future_measures.get("code_smells"))
                or 0
            )
        for availability in _future_snapshot_availability_records(payload):
            label = str(availability.get("snapshot_label") or "")
            if label in stats["future_snapshot_available_counts"] and availability.get("available"):
                stats["future_snapshot_available_counts"][label] += 1

        self._emit_refactoring_results(ref, payload)
        self._emit_maintainability_results(ref, payload)
        self._emit_code_smell_results(ref, payload)
        self._emit_repository_metadata(ref, payload)
        self._emit_snapshot_manifests_and_blobs(ref, payload)

    def _load_sampling_metadata(self) -> None:
        """Index sampler metadata from sampled and longitudinal selection files."""
        cohort_dirs = discover_cohort_dirs(self.curation_outputs_dir, self.curation_exclude_dirs)
        seen: set[tuple[str, str]] = set()
        for cohort_dir in cohort_dirs:
            source_run_id = cohort_dir.name
            jsonl_paths = _discover_sampled_pr_jsonl_paths(
                cohort_dir
            ) + _discover_longitudinal_pr_jsonl_paths(cohort_dir)
            for jsonl_path in jsonl_paths:
                cohort = _cohort_from_jsonl_path(jsonl_path, source_run_id)
                for _line_number, payload in _read_jsonl(jsonl_path):
                    pull_request_key = _selection_pull_request_key(payload)
                    dedup_key = (cohort, pull_request_key)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    metadata = _sampling_metadata_from_payload(payload)
                    self._sampling_metadata_by_pr.setdefault((cohort, pull_request_key), metadata)

    def _iter_topic_payloads(
        self,
        path: Path,
        *,
        count_failures: bool = True,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield valid repository topic payloads from one topic output."""
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        if count_failures:
                            self.summary["topic_output_parse_failures"] += 1
                        continue
                    if isinstance(payload, dict):
                        yield line_number, payload
                    elif count_failures:
                        self.summary["topic_output_parse_failures"] += 1
        except OSError as exc:
            if count_failures:
                self.summary["topic_output_parse_failures"] += 1
            print(
                "[post-processing/upload-curation-data] Skipping unreadable topic output "
                f"{path}: {exc}"
            )

    def _discover_latest_topic_outputs_by_cohort(self) -> dict[str, TopicOutputRef]:
        """Select the newest topic output for each cohort."""
        refs = _discover_topic_output_refs(self.topic_classification_outputs_dir)
        self.summary["topic_output_directories_discovered"] = len(refs)
        selected: dict[str, TopicOutputRef] = {}
        for ref in refs:
            cohorts: set[str] = set()
            for _line_number, payload in self._iter_topic_payloads(
                ref.repository_topics_path,
                count_failures=False,
            ):
                cohort = _cohort_from_payload(payload, ref.output_dir.name)
                if cohort:
                    cohorts.add(cohort)
            for cohort in cohorts:
                current = selected.get(cohort)
                if current is None or ref.rank > current.rank:
                    selected[cohort] = ref
        return selected

    def _topic_predictions(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Return topic predictions sorted by model rank and original order."""
        predictions: list[tuple[int, int, dict[str, Any]]] = []
        for index, value in enumerate(_as_list(payload.get("predicted_topics")), start=1):
            if not isinstance(value, dict):
                continue
            rank = _safe_int(value.get("rank")) or index
            predictions.append((rank, index, value))
        return [value for _rank, _index, value in sorted(predictions)]

    def _repository_topic_classification_row(
        self,
        *,
        repository_key: str,
        payload: dict[str, Any],
        ref: TopicOutputRef,
    ) -> dict[str, Any]:
        """Build one root-scoped repository topic-classification row."""
        owner = _safe_str(payload.get("repository_owner"))
        name = _safe_str(payload.get("repository_name"))
        if not owner or not name:
            key_owner, key_name = _owner_name_from_repo_key(repository_key)
            owner = owner or key_owner
            name = name or key_name
        repository_full_name = _safe_str(payload.get("repository_full_name"))
        if repository_full_name is None and owner and name:
            repository_full_name = f"{owner}/{name}"
        predictions = self._topic_predictions(payload)
        source_top_k = _safe_int(payload.get("top_k"))
        threshold_policy = bool(payload.get("prediction_score_threshold")) or (
            _safe_str(payload.get("prediction_retention_policy"))
            == "all_above_threshold_after_filtering"
        )
        effective_top_k = None if threshold_policy else self.topic_classification_top_k_topics
        if source_top_k is not None and effective_top_k is not None:
            effective_top_k = min(effective_top_k, max(1, source_top_k))
        public_topic_predictions = _public_topic_prediction_records(
            predictions,
            effective_top_k=effective_top_k,
        )
        row: dict[str, Any] = {
            "repository_key": repository_key,
            "repository_owner": owner,
            "repository_name": name,
            "repository_full_name": repository_full_name,
            "repository_id": _safe_str(payload.get("repository_id")),
            "source_ref": _safe_str(payload.get("source_ref")),
            "source_commit": _safe_str(payload.get("source_commit")),
            "predicted_topic_count": len(predictions),
            "topics_json": (
                _json_dumps(public_topic_predictions) if public_topic_predictions else None
            ),
        }
        row["topic_1"] = None
        row["topic_1_domain"] = None
        row["topic_1_score"] = None
        if public_topic_predictions:
            first_prediction = public_topic_predictions[0]
            row["topic_1"] = first_prediction.get("topic")
            row["topic_1_domain"] = first_prediction.get("topic_domain")
            row["topic_1_score"] = first_prediction.get("score")
        return row

    def _emit_repository_topic_classifications(self) -> None:
        """Emit repository topic rows for repositories already present in curation."""
        selected_by_cohort = self._discover_latest_topic_outputs_by_cohort()
        eligible_cohorts = set(self.summary["cohorts"].keys())
        selected_count = 0
        for cohort, ref in sorted(selected_by_cohort.items()):
            if eligible_cohorts and cohort not in eligible_cohorts:
                continue
            selected_count += 1
            rows_by_repository: dict[str, dict[str, Any]] = {}
            duplicate_count = 0
            for _line_number, payload in self._iter_topic_payloads(ref.repository_topics_path):
                payload_cohort = _cohort_from_payload(payload, ref.output_dir.name)
                if payload_cohort != cohort:
                    continue
                repository_key = _repository_key_from_topic_payload(payload)
                if not repository_key:
                    continue
                if repository_key in self._emitted_repository_topic_classifications:
                    duplicate_count += 1
                    continue
                if repository_key in rows_by_repository:
                    duplicate_count += 1
                rows_by_repository[repository_key] = self._repository_topic_classification_row(
                    repository_key=repository_key,
                    payload=payload,
                    ref=ref,
                )
            for repository_key in sorted(rows_by_repository):
                row = rows_by_repository[repository_key]
                self._emitted_repository_topic_classifications.add(repository_key)
                self._append_entity(cohort, ENTITY_REPOSITORY_TOPIC_CLASSIFICATIONS, row)
                stats = self._cohort_stats(cohort)
                stats["topic_classified_repository_count"] += 1
                if row.get("topic_1"):
                    stats["topic_primary_topic_count"] += 1
                self.summary["repository_topic_classification_rows"] += 1
            self.summary["repository_topic_duplicate_rows_skipped"] += duplicate_count
            print(
                "[post-processing/upload-curation-data] Exported "
                f"{len(rows_by_repository)} repository topic classifications for {cohort} "
                f"from {ref.repository_topics_path}."
            )
        self.summary["topic_outputs_selected"] = selected_count

    def _iter_longitudinal_payloads(
        self,
        path: Path,
        *,
        count_failures: bool = True,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield valid longitudinal JSONL rows and count parse failures."""
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        if count_failures:
                            self.summary["longitudinal_refactoring_parse_failures"] += 1
                        continue
                    if isinstance(payload, dict):
                        yield line_number, payload
                    elif count_failures:
                        self.summary["longitudinal_refactoring_parse_failures"] += 1
        except OSError as exc:
            if count_failures:
                self.summary["longitudinal_refactoring_parse_failures"] += 1
            print(
                "[post-processing/upload-curation-data] Skipping unreadable "
                f"longitudinal refactoring output {path}: {exc}"
            )

    def _discover_latest_longitudinal_outputs(
        self,
    ) -> dict[tuple[str, str], LongitudinalRefactoringOutputRef]:
        """Select newest longitudinal-refactoring output per cohort/input mode."""
        refs = _discover_longitudinal_refactoring_output_refs(
            self.longitudinal_refactoring_outputs_dir
        )
        self.summary["longitudinal_refactoring_output_directories_discovered"] = len(refs)
        selected: dict[tuple[str, str], LongitudinalRefactoringOutputRef] = {}
        for ref in refs:
            for cohort in ref.cohorts:
                key = (cohort, ref.input_mode)
                current = selected.get(key)
                if current is None or ref.rank > current.rank:
                    selected[key] = ref
        return selected

    def _longitudinal_payload_identity(
        self,
        payload: dict[str, Any],
        *,
        fallback_input_mode: str,
    ) -> tuple[str, str, str, str, str | None] | None:
        """Return cohort/mode/PR/snapshot identity for longitudinal rows."""
        cohort = _cohort_from_payload(payload, "")
        input_mode = _safe_str(payload.get("input_mode")) or fallback_input_mode
        snapshot_label = _safe_str(payload.get("snapshot_label"))
        repository_key = repository_key_from_full_name(payload.get("repository_key"))
        if repository_key is None:
            repository_key = repository_key_from_safe_key(payload.get("repository_key"))
        if repository_key is None:
            owner = _safe_str(payload.get("repository_owner"))
            name = _safe_str(payload.get("repository_name"))
            if owner and name:
                repository_key = normalize_repository_key(owner, name)
        pr_number = _safe_int(payload.get("pr_number"))
        pr_url = _safe_str(payload.get("pr_url"))
        pull_request_key = _pull_request_key_from_parts(
            repository_key=repository_key,
            pr_number=pr_number,
            pr_url=pr_url,
        )
        if not cohort or not input_mode or not pull_request_key or not snapshot_label:
            return None
        return cohort, input_mode, pull_request_key, snapshot_label, repository_key

    def _longitudinal_pr_identity(
        self,
        payload: dict[str, Any],
        *,
        fallback_input_mode: str,
    ) -> tuple[str, str, str, str | None] | None:
        """Return cohort/mode/PR identity for longitudinal PR summaries."""
        identity = self._longitudinal_payload_identity(
            {**payload, "snapshot_label": payload.get("snapshot_label") or "pr"},
            fallback_input_mode=fallback_input_mode,
        )
        if identity is None:
            return None
        cohort, input_mode, pull_request_key, _snapshot_label, repository_key = identity
        return cohort, input_mode, pull_request_key, repository_key

    def _sanitize_longitudinal_operation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep only public longitudinal operation fields in stable order."""
        operation = {
            key: payload.get(key) for key in LONGITUDINAL_OPERATION_FIELDS if key in payload
        }
        operation["source_locations"] = _as_list(operation.get("source_locations"))
        operation["target_locations"] = _as_list(operation.get("target_locations"))
        taxonomy = operation.get("taxonomy")
        operation["taxonomy"] = taxonomy if isinstance(taxonomy, dict) else None
        return operation

    def _reset_longitudinal_temp_tables(self) -> None:
        """Reset temporary SQLite indexes for one longitudinal output file set."""
        conn = self.state_store.conn
        conn.execute("DROP TABLE IF EXISTS temp.longitudinal_snapshot_results")
        conn.execute("DROP TABLE IF EXISTS temp.longitudinal_operations")
        conn.execute("DROP TABLE IF EXISTS temp.longitudinal_pr_summaries")
        conn.execute(
            """
            CREATE TEMP TABLE longitudinal_snapshot_results (
                cohort TEXT NOT NULL,
                input_mode TEXT NOT NULL,
                pull_request_key TEXT NOT NULL,
                snapshot_label TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (cohort, input_mode, pull_request_key, snapshot_label)
            )
            """
        )
        conn.execute(
            """
            CREATE TEMP TABLE longitudinal_operations (
                cohort TEXT NOT NULL,
                input_mode TEXT NOT NULL,
                pull_request_key TEXT NOT NULL,
                snapshot_label TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                operation_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TEMP TABLE longitudinal_pr_summaries (
                cohort TEXT NOT NULL,
                input_mode TEXT NOT NULL,
                pull_request_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (cohort, input_mode, pull_request_key)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX longitudinal_operations_lookup
            ON longitudinal_operations (cohort, input_mode, pull_request_key, snapshot_label, line_number)
            """
        )
        conn.commit()

    def _index_longitudinal_output(
        self,
        ref: LongitudinalRefactoringOutputRef,
        *,
        selected_cohorts: set[str],
    ) -> int:
        """Index longitudinal rows into temp tables and return duplicate count."""
        self._reset_longitudinal_temp_tables()
        conn = self.state_store.conn
        duplicates = 0
        if ref.pr_summary_path is not None:
            for _line_number, payload in self._iter_longitudinal_payloads(ref.pr_summary_path):
                payload.setdefault("input_mode", ref.input_mode)
                identity = self._longitudinal_pr_identity(
                    payload,
                    fallback_input_mode=ref.input_mode,
                )
                if identity is None:
                    continue
                cohort, input_mode, pull_request_key, _repository_key = identity
                if cohort not in selected_cohorts or input_mode != ref.input_mode:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO longitudinal_pr_summaries
                    (cohort, input_mode, pull_request_key, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        cohort,
                        input_mode,
                        pull_request_key,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                    ),
                )

        if ref.operations_path is not None:
            for line_number, payload in self._iter_longitudinal_payloads(ref.operations_path):
                payload.setdefault("input_mode", ref.input_mode)
                identity = self._longitudinal_payload_identity(
                    payload,
                    fallback_input_mode=ref.input_mode,
                )
                if identity is None:
                    continue
                cohort, input_mode, pull_request_key, snapshot_label, _repository_key = identity
                if cohort not in selected_cohorts or input_mode != ref.input_mode:
                    continue
                operation = self._sanitize_longitudinal_operation(payload)
                conn.execute(
                    """
                    INSERT INTO longitudinal_operations
                    (cohort, input_mode, pull_request_key, snapshot_label, line_number, operation_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cohort,
                        input_mode,
                        pull_request_key,
                        snapshot_label,
                        line_number,
                        json.dumps(operation, ensure_ascii=False, sort_keys=True, default=str),
                    ),
                )

        for line_number, payload in self._iter_longitudinal_payloads(ref.snapshot_results_path):
            payload.setdefault("input_mode", ref.input_mode)
            identity = self._longitudinal_payload_identity(
                payload,
                fallback_input_mode=ref.input_mode,
            )
            if identity is None:
                continue
            cohort, input_mode, pull_request_key, snapshot_label, _repository_key = identity
            if cohort not in selected_cohorts or input_mode != ref.input_mode:
                continue
            exists = conn.execute(
                """
                SELECT 1 FROM longitudinal_snapshot_results
                WHERE cohort = ? AND input_mode = ? AND pull_request_key = ? AND snapshot_label = ?
                """,
                (cohort, input_mode, pull_request_key, snapshot_label),
            ).fetchone()
            if exists is not None:
                duplicates += 1
            conn.execute(
                """
                INSERT OR REPLACE INTO longitudinal_snapshot_results
                (cohort, input_mode, pull_request_key, snapshot_label, line_number, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cohort,
                    input_mode,
                    pull_request_key,
                    snapshot_label,
                    line_number,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                ),
            )
        conn.commit()
        return duplicates

    def _longitudinal_pr_summary(
        self,
        *,
        cohort: str,
        input_mode: str,
        pull_request_key: str,
    ) -> dict[str, Any]:
        """Return the longitudinal PR summary for one indexed PR."""
        row = self.state_store.conn.execute(
            """
            SELECT payload_json FROM longitudinal_pr_summaries
            WHERE cohort = ? AND input_mode = ? AND pull_request_key = ?
            """,
            (cohort, input_mode, pull_request_key),
        ).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(row[0])
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _longitudinal_operations(
        self,
        *,
        cohort: str,
        input_mode: str,
        pull_request_key: str,
        snapshot_label: str,
    ) -> list[dict[str, Any]]:
        """Return indexed longitudinal operations for one PR snapshot."""
        operations: list[dict[str, Any]] = []
        cursor = self.state_store.conn.execute(
            """
            SELECT operation_json FROM longitudinal_operations
            WHERE cohort = ? AND input_mode = ? AND pull_request_key = ? AND snapshot_label = ?
            ORDER BY line_number
            """,
            (cohort, input_mode, pull_request_key, snapshot_label),
        )
        for row in cursor:
            try:
                payload = json.loads(row[0])
            except Exception:
                continue
            if isinstance(payload, dict):
                operations.append(payload)
        return operations

    def _longitudinal_refactoring_result_row(
        self,
        snapshot_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build a public ``RefactoringResults`` row from longitudinal output."""
        identity = self._longitudinal_payload_identity(
            snapshot_payload,
            fallback_input_mode=_safe_str(snapshot_payload.get("input_mode")) or "unknown",
        )
        if identity is None:
            return None
        cohort, input_mode, pull_request_key, snapshot_label, repository_key = identity
        pr_summary = self._longitudinal_pr_summary(
            cohort=cohort,
            input_mode=input_mode,
            pull_request_key=pull_request_key,
        )
        future_metrics = _as_dict(pr_summary.get("future_snapshot_metrics"))
        snapshot_metrics = _as_dict(future_metrics.get(snapshot_label))
        operation_type_counts = _as_dict(
            _as_dict(pr_summary.get("operation_type_count_by_snapshot")).get(snapshot_label)
        )
        murphy_counts = _as_dict(
            _as_dict(pr_summary.get("murphy_hill_count_by_snapshot")).get(snapshot_label)
        )
        operation_count_by_snapshot = _as_dict(pr_summary.get("operation_count_by_snapshot"))
        operations = self._longitudinal_operations(
            cohort=cohort,
            input_mode=input_mode,
            pull_request_key=pull_request_key,
            snapshot_label=snapshot_label,
        )
        operation_count = (
            _safe_int(snapshot_payload.get("operation_count"))
            if snapshot_payload.get("operation_count") is not None
            else None
        )
        if operation_count is None:
            operation_count = _safe_int(operation_count_by_snapshot.get(snapshot_label))
        if operation_count is None:
            operation_count = len(operations)
        owner = _safe_str(snapshot_payload.get("repository_owner"))
        name = _safe_str(snapshot_payload.get("repository_name"))
        if not owner or not name:
            key_owner, key_name = _owner_name_from_repo_key(repository_key)
            owner = owner or key_owner
            name = name or key_name
        snapshot_available = _safe_bool(snapshot_payload.get("snapshot_available"))
        effective_available = _safe_bool(snapshot_payload.get("effective_snapshot_available"))
        if snapshot_available is None:
            snapshot_available = effective_available
        if snapshot_available is None:
            snapshot_available = bool(snapshot_payload.get("snapshot_commit"))
        result_status = _safe_str(snapshot_payload.get("status"))
        if result_status is None:
            result_status = _safe_str(
                _as_dict(pr_summary.get("future_snapshot_status_by_snapshot")).get(snapshot_label)
            )
        analysis_completed = _safe_bool(snapshot_payload.get("analysis_completed"))
        if analysis_completed is None:
            analysis_completed = str(result_status or "").strip().lower() in {
                "success",
                "partial_success",
                "completed",
            }
        if snapshot_available is False or effective_available is False or not analysis_completed:
            return None
        stored_refactor_type_counts = _count_mapping(
            operation_type_counts or snapshot_metrics.get("refactor_type_count")
        )
        refactor_type_counts = (
            _current_refactoring_count_mapping(stored_refactor_type_counts)
            or stored_refactor_type_counts
        )
        murphy_hill_counts = _count_mapping(
            murphy_counts or snapshot_metrics.get("refactor_murphyhill_count")
        )
        return {
            "cohort": cohort,
            "source_run_id": None,
            "result_source": REFACTORING_RESULT_SOURCE_LONGITUDINAL,
            "input_mode": input_mode,
            "pull_request_key": pull_request_key,
            "repository_key": repository_key,
            "repository_owner": owner,
            "repository_name": name,
            "pr_number": _safe_int(snapshot_payload.get("pr_number")),
            "pr_url": _safe_str(snapshot_payload.get("pr_url")),
            "snapshot_label": snapshot_label,
            "transition_label": f"after_to_{snapshot_label}",
            "snapshot_key": _snapshot_key(pull_request_key, snapshot_label),
            "snapshot_commit": _safe_str(snapshot_payload.get("snapshot_commit")),
            "result_status": result_status,
            "refop_retention_rate": _safe_float(
                _as_dict(snapshot_metrics.get("retention")).get("retention_rate")
            ),
            "future_lines_touched": (
                _safe_int(
                    _as_dict(snapshot_metrics.get("future_impact")).get(
                        "touched_refactoring_zone_lines_count"
                    )
                )
                or _safe_int(
                    _as_dict(snapshot_metrics.get("future_impact")).get("touched_pr_lines_count")
                )
                or _safe_int(
                    _as_dict(snapshot_metrics.get("pr_changed_line_future_impact")).get(
                        "touched_pr_lines_count"
                    )
                )
            ),
            "touching_commit_count": (
                _safe_int(
                    _as_dict(snapshot_metrics.get("future_impact")).get("touching_commits_count")
                )
                or _safe_int(
                    _as_dict(snapshot_metrics.get("pr_changed_line_future_impact")).get(
                        "touching_commits_count"
                    )
                )
            ),
            "refop_count": operation_count,
            "added_lines": _safe_int(snapshot_metrics.get("refactor_added_lines")),
            "removed_lines": _safe_int(snapshot_metrics.get("refactor_removed_lines")),
            "magnitude_lines": _safe_int(snapshot_metrics.get("refactor_magnitude_lines")),
            "magnitude_files": _safe_int(snapshot_metrics.get("refactor_magnitude_files")),
            "diversity": _safe_int(snapshot_metrics.get("refactor_diversity")),
            "refactor_type_count_json": _json_dumps(refactor_type_counts),
            **_murphy_hill_count_columns(murphy_hill_counts, refactor_type_counts),
            **_refactoring_mapping_fields(operations, refactor_type_counts),
            "operations_json": _json_dumps(operations),
            "tool": _safe_str(snapshot_payload.get("tool")),
            "skipped_tool_run": _safe_bool(snapshot_payload.get("skipped_tool_run")),
            "skip_reason": _safe_str(snapshot_payload.get("skip_reason")),
        }

    def _record_longitudinal_refactoring_stats(
        self,
        row: dict[str, Any],
        pr_summary: dict[str, Any],
    ) -> None:
        """Update cohort summary counters for one longitudinal row."""
        cohort = str(row.get("cohort") or "")
        input_mode = str(row.get("input_mode") or "unknown")
        pull_request_key = str(row.get("pull_request_key") or "")
        snapshot_label = str(row.get("snapshot_label") or "")
        stats = self._cohort_stats(cohort)
        by_mode = stats["longitudinal_refactoring_by_input_mode"][input_mode]
        if pull_request_key:
            by_mode["pr_keys"].add(pull_request_key)
        by_mode["snapshot_results"] += 1
        by_mode["completed_snapshot_results"] += 1
        if snapshot_label in FUTURE_SNAPSHOT_LABELS:
            by_mode["completed_by_label"][snapshot_label] += 1
        operation_count = int(row.get("refop_count") or 0)
        by_mode["refactoring_operations"] += operation_count
        if pr_summary.get("has_future_refactoring") or operation_count > 0:
            by_mode["prs_with_future_refactoring"].add(pull_request_key)
        if pr_summary.get("zero_future_refactoring_observed"):
            by_mode["prs_with_observed_zero_future_refactoring"].add(pull_request_key)
        self.summary["longitudinal_refactoring_rows"] += 1
        self.summary["longitudinal_refactoring_operations"] += operation_count

    def _emit_longitudinal_refactoring_results(self) -> None:
        """Emit optional longitudinal refactoring rows for selected curation PRs."""
        selected = self._discover_latest_longitudinal_outputs()
        eligible_cohorts = set(self.summary["cohorts"].keys())
        grouped_refs: dict[LongitudinalRefactoringOutputRef, set[str]] = defaultdict(set)
        for (cohort, _input_mode), ref in sorted(selected.items()):
            if eligible_cohorts and cohort not in eligible_cohorts:
                continue
            grouped_refs[ref].add(cohort)
        self.summary["longitudinal_refactoring_outputs_selected"] = len(selected)
        for ref, cohorts in sorted(
            grouped_refs.items(),
            key=lambda item: (item[0].input_mode, str(item[0].output_dir).lower()),
        ):
            duplicates = self._index_longitudinal_output(ref, selected_cohorts=cohorts)
            self.summary["longitudinal_refactoring_duplicate_snapshot_rows_skipped"] += duplicates
            partition = _refactoring_results_partition(
                REFACTORING_RESULT_SOURCE_LONGITUDINAL,
                ref.input_mode,
            )
            emitted = 0
            cursor = self.state_store.conn.execute(
                """
                SELECT payload_json FROM longitudinal_snapshot_results
                ORDER BY cohort, input_mode, pull_request_key, snapshot_label
                """
            )
            for (payload_json,) in cursor:
                try:
                    snapshot_payload = json.loads(payload_json)
                except Exception:
                    self.summary["longitudinal_refactoring_parse_failures"] += 1
                    continue
                if not isinstance(snapshot_payload, dict):
                    self.summary["longitudinal_refactoring_parse_failures"] += 1
                    continue
                row = self._longitudinal_refactoring_result_row(snapshot_payload)
                if row is None:
                    continue
                pr_summary = self._longitudinal_pr_summary(
                    cohort=str(row["cohort"]),
                    input_mode=str(row["input_mode"]),
                    pull_request_key=str(row["pull_request_key"]),
                )
                self._append_entity(
                    str(row["cohort"]),
                    ENTITY_REFACTORING_RESULTS,
                    row,
                    partition=partition,
                )
                self._record_longitudinal_refactoring_stats(row, pr_summary)
                emitted += 1
            print(
                "[post-processing/upload-curation-data] Exported "
                f"{emitted} longitudinal refactoring rows for input_mode={ref.input_mode} "
                f"from {ref.snapshot_results_path}."
            )

    def _write_cohort_metadata_files(self) -> list[Path]:
        """Write per-cohort metadata JSON files beside public parquet tables."""
        paths: list[Path] = []
        for cohort, stats in sorted(self.summary["cohorts"].items()):
            payload = {
                "schema_version": self.curation_schema_version,
                "generated_at_utc": _now_iso(),
                "cohort": cohort,
                "source_run_ids": sorted(str(value) for value in stats["source_run_ids"]),
                "entity_rows": dict(sorted(stats["entity_rows"].items())),
                "summary": self._serializable_cohort_stats(cohort, stats),
            }
            metadata_path = _curation_dataset_dir(self.data_root / cohort) / "metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            paths.append(metadata_path)
        return paths

    def _serializable_cohort_stats(self, cohort: str, stats: dict[str, Any]) -> dict[str, Any]:
        """Convert mutable set/defaultdict counters into JSON-safe summaries."""
        future_counts = stats.get("future_snapshot_available_counts") or {}
        future_refactoring_counts = stats.get("future_snapshot_refactoring_operation_counts") or {}
        future_code_smell_counts = stats.get("future_snapshot_code_smell_counts") or {}
        topic_repository_count = int(stats.get("topic_classified_repository_count", 0))
        topic_primary_count = int(stats.get("topic_primary_topic_count", 0))
        longitudinal: dict[str, Any] = {}
        for input_mode, raw_mode_stats in sorted(
            (stats.get("longitudinal_refactoring_by_input_mode") or {}).items()
        ):
            completed_by_label = raw_mode_stats.get("completed_by_label") or {}
            longitudinal[str(input_mode)] = {
                "prs_analyzed": len(raw_mode_stats.get("pr_keys") or set()),
                "snapshot_results": int(raw_mode_stats.get("snapshot_results", 0)),
                "completed_snapshot_results": int(
                    raw_mode_stats.get("completed_snapshot_results", 0)
                ),
                "prs_with_future_refactoring": len(
                    raw_mode_stats.get("prs_with_future_refactoring") or set()
                ),
                "prs_with_observed_zero_future_refactoring": len(
                    raw_mode_stats.get("prs_with_observed_zero_future_refactoring") or set()
                ),
                "refactoring_operations": int(raw_mode_stats.get("refactoring_operations", 0)),
                "completed_by_label": {
                    label: int(completed_by_label.get(label, 0)) for label in FUTURE_SNAPSHOT_LABELS
                },
            }
        return {
            "cohort": cohort,
            "pr_count": int(stats.get("pr_count", 0)),
            "unique_repositories": len(stats.get("repository_keys") or set()),
            "additions_sum": int(stats.get("additions_sum", 0)),
            "deletions_sum": int(stats.get("deletions_sum", 0)),
            "refactoring_pr_count": int(stats.get("refactoring_pr_count", 0)),
            "code_smell_pr_count": int(stats.get("code_smell_pr_count", 0)),
            "refactoring_operation_count": int(stats.get("refactoring_operation_count", 0)),
            "code_smell_count": int(stats.get("code_smell_count", 0)),
            "future_snapshot_available_counts": {
                label: int(future_counts.get(label, 0)) for label in FUTURE_SNAPSHOT_LABELS
            },
            "future_snapshot_refactoring_operation_counts": {
                label: int(future_refactoring_counts.get(label, 0))
                for label in FUTURE_SNAPSHOT_LABELS
            },
            "future_snapshot_code_smell_counts": {
                label: int(future_code_smell_counts.get(label, 0))
                for label in FUTURE_SNAPSHOT_LABELS
            },
            "topic_classified_repository_count": topic_repository_count,
            "topic_primary_topic_count": topic_primary_count,
            "topic_primary_topic_coverage": (
                topic_primary_count / topic_repository_count if topic_repository_count else 0.0
            ),
            "longitudinal_refactoring": longitudinal,
        }

    def export(self) -> dict[str, Any]:
        """Run local packaging from curation outputs into public parquet files."""
        self._load_repository_file_lists()
        refs = self._discover_aggregate_refs()
        self._load_sampling_metadata()
        for index, ref in enumerate(refs, start=1):
            if index % 1000 == 0:
                print(
                    "[post-processing/upload-curation-data] Processed "
                    f"{index}/{len(refs)} selected aggregate files."
                )
            self._process_aggregate(ref)
        self._emit_repository_topic_classifications()
        self._emit_longitudinal_refactoring_results()
        self.writer.flush_all()
        metadata_paths = [
            self._write_dataset_card(),
            self._write_schema_manifest(),
            *self._write_cohort_metadata_files(),
        ]
        self.summary["metadata_files"] = [str(path) for path in metadata_paths]
        self._finalize_summary()
        return self.summary

    def run(self) -> dict[str, Any]:
        """Run local export followed by upload using configured credentials."""
        self.export()
        self.upload_outputs()
        return self.summary

    def _finalize_summary(self) -> None:
        """Freeze cohort summary structures into JSON-serializable dictionaries."""
        finalized: dict[str, Any] = {}
        for cohort, stats in sorted(self.summary["cohorts"].items()):
            serializable = self._serializable_cohort_stats(cohort, stats)
            serializable["entity_rows"] = dict(sorted(stats["entity_rows"].items()))
            serializable["source_run_ids"] = sorted(str(value) for value in stats["source_run_ids"])
            finalized[cohort] = serializable
        self.summary["cohorts"] = finalized

    def _config_date_suffix(self) -> str:
        """Return a date suffix for generated Hugging Face dataset configs."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _remote_text_file(self, filename: str) -> str | None:
        """Read an existing local metadata file for in-place manifest updates.

        Preparation must be network-free even when upload credentials are set in
        the shell. The upload command is the only operation that contacts
        Hugging Face.
        """
        local_path = self.local_output_dir / filename
        if local_path.exists():
            return local_path.read_text(encoding="utf-8")
        return None

    def _split_front_matter(self, text: str) -> tuple[list[str], str]:
        """Split dataset-card YAML front matter from Markdown body text."""
        if not text.startswith("---"):
            return [], text
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return [], text
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return lines[1:idx], "\n".join(lines[idx + 1 :]).lstrip("\n")
        return [], text

    def _curation_config_blocks(self) -> list[list[str]]:
        """Build Hugging Face dataset config blocks for curation entities."""
        config_date_suffix = self._config_date_suffix()
        cohorts = sorted(self.summary["cohorts"].keys())
        blocks: list[list[str]] = []
        for entity_name in ROOT_CURATION_ENTITY_NAMES:
            entity_path = f"data/{CURATION_DATASET_SUBDIR}/{entity_name}/**/*.parquet"
            blocks.append(
                [
                    f"- config_name: curation_{ENTITY_CONFIG_SLUGS[entity_name]}_{config_date_suffix}",
                    "  data_files:",
                    "  - split: train",
                    f"    path: {entity_path}",
                ]
            )
        for cohort in cohorts:
            for entity_name in COHORT_CURATION_ENTITY_NAMES:
                entity_path = f"data/{cohort}/{CURATION_DATASET_SUBDIR}/{entity_name}/**/*.parquet"
                blocks.append(
                    [
                        f"- config_name: {cohort}_curation_{ENTITY_CONFIG_SLUGS[entity_name]}_{config_date_suffix}",
                        "  data_files:",
                        "  - split: train",
                        f"    path: {entity_path}",
                    ]
                )
        return blocks

    def _merge_front_matter_configs(self, yaml_lines: list[str]) -> list[str]:
        """Replace old curation dataset configs while preserving other configs."""
        lines: list[str] = []
        index = 0
        while index < len(yaml_lines):
            line = yaml_lines[index]
            if not line.startswith("- config_name:"):
                lines.append(line)
                index += 1
                continue
            block = [line]
            index += 1
            while index < len(yaml_lines) and not yaml_lines[index].startswith("- config_name:"):
                next_line = yaml_lines[index]
                if next_line and not next_line.startswith(" ") and not next_line.startswith("- "):
                    break
                block.append(next_line)
                index += 1
            block_text = "\n".join(block)
            if "_curation_" in block_text or re.search(
                r"path:\s*data/[^/]+/[cC]uration/", block_text
            ):
                continue
            lines.extend(block)
        if not any(line.strip() == "configs:" for line in lines):
            lines.append("configs:")
        for block in self._curation_config_blocks():
            lines.extend(block)
        return lines

    def _curation_summary_section(self) -> str:
        """Build the generated curation summary Markdown section."""
        lines = [
            README_SUMMARY_BEGIN,
            "### Curation Subset Summary",
            "",
            "| Cohort | Number of PRs | Unique Repositories | Sum of Additions | Sum of Deletions | PRs with Refactoring Operation | PRs with Code Smell | Refactoring Operations | Code Smells | PRs with +3d Snapshots | PRs with +7d Snapshots | PRs with +31d Snapshots | PRs with +61d Snapshots | Refactoring Ops +3d | Refactoring Ops +7d | Refactoring Ops +31d | Refactoring Ops +61d | Code Smells +3d | Code Smells +7d | Code Smells +31d | Code Smells +61d |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        totals = defaultdict(int)
        if not self.summary["cohorts"]:
            lines.append(
                "| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
            )
        for cohort, raw_stats in sorted(self.summary["cohorts"].items()):
            stats = (
                self._serializable_cohort_stats(cohort, raw_stats)
                if "repository_keys" in raw_stats
                else raw_stats
            )
            future_counts = stats.get("future_snapshot_available_counts") or {}
            future_refactoring_counts = (
                stats.get("future_snapshot_refactoring_operation_counts") or {}
            )
            future_code_smell_counts = stats.get("future_snapshot_code_smell_counts") or {}
            lines.append(
                "| {cohort} | {pr_count} | {unique_repositories} | {additions_sum} | {deletions_sum} | {refactoring_pr_count} | {code_smell_pr_count} | {refactoring_operation_count} | {code_smell_count} | {plus3} | {plus7} | {plus31} | {plus61} | {refactor_plus3} | {refactor_plus7} | {refactor_plus31} | {refactor_plus61} | {smell_plus3} | {smell_plus7} | {smell_plus31} | {smell_plus61} |".format(
                    cohort=cohort,
                    pr_count=stats.get("pr_count", 0),
                    unique_repositories=stats.get("unique_repositories", 0),
                    additions_sum=stats.get("additions_sum", 0),
                    deletions_sum=stats.get("deletions_sum", 0),
                    refactoring_pr_count=stats.get("refactoring_pr_count", 0),
                    code_smell_pr_count=stats.get("code_smell_pr_count", 0),
                    refactoring_operation_count=stats.get("refactoring_operation_count", 0),
                    code_smell_count=stats.get("code_smell_count", 0),
                    plus3=future_counts.get("+3d", 0),
                    plus7=future_counts.get("+7d", 0),
                    plus31=future_counts.get("+31d", 0),
                    plus61=future_counts.get("+61d", 0),
                    refactor_plus3=future_refactoring_counts.get("+3d", 0),
                    refactor_plus7=future_refactoring_counts.get("+7d", 0),
                    refactor_plus31=future_refactoring_counts.get("+31d", 0),
                    refactor_plus61=future_refactoring_counts.get("+61d", 0),
                    smell_plus3=future_code_smell_counts.get("+3d", 0),
                    smell_plus7=future_code_smell_counts.get("+7d", 0),
                    smell_plus31=future_code_smell_counts.get("+31d", 0),
                    smell_plus61=future_code_smell_counts.get("+61d", 0),
                )
            )
            for key in (
                "pr_count",
                "unique_repositories",
                "additions_sum",
                "deletions_sum",
                "refactoring_pr_count",
                "code_smell_pr_count",
                "refactoring_operation_count",
                "code_smell_count",
            ):
                totals[key] += int(stats.get(key, 0))
            for label in FUTURE_SNAPSHOT_LABELS:
                totals[label] += int(future_counts.get(label, 0))
                totals[f"refactor_{label}"] += int(future_refactoring_counts.get(label, 0))
                totals[f"smell_{label}"] += int(future_code_smell_counts.get(label, 0))
        if self.summary["cohorts"]:
            lines.append(
                "| **Total** | **{pr_count}** | **{unique_repositories}** | **{additions_sum}** | **{deletions_sum}** | **{refactoring_pr_count}** | **{code_smell_pr_count}** | **{refactoring_operation_count}** | **{code_smell_count}** | **{plus3}** | **{plus7}** | **{plus31}** | **{plus61}** | **{refactor_plus3}** | **{refactor_plus7}** | **{refactor_plus31}** | **{refactor_plus61}** | **{smell_plus3}** | **{smell_plus7}** | **{smell_plus31}** | **{smell_plus61}** |".format(
                    pr_count=totals["pr_count"],
                    unique_repositories=totals["unique_repositories"],
                    additions_sum=totals["additions_sum"],
                    deletions_sum=totals["deletions_sum"],
                    refactoring_pr_count=totals["refactoring_pr_count"],
                    code_smell_pr_count=totals["code_smell_pr_count"],
                    refactoring_operation_count=totals["refactoring_operation_count"],
                    code_smell_count=totals["code_smell_count"],
                    plus3=totals["+3d"],
                    plus7=totals["+7d"],
                    plus31=totals["+31d"],
                    plus61=totals["+61d"],
                    refactor_plus3=totals["refactor_+3d"],
                    refactor_plus7=totals["refactor_+7d"],
                    refactor_plus31=totals["refactor_+31d"],
                    refactor_plus61=totals["refactor_+61d"],
                    smell_plus3=totals["smell_+3d"],
                    smell_plus7=totals["smell_+7d"],
                    smell_plus31=totals["smell_+31d"],
                    smell_plus61=totals["smell_+61d"],
                )
            )
        lines.extend(["", README_SUMMARY_END])
        return "\n".join(lines)

    def _curation_structure_section(self) -> str:
        """Build the generated curation entity-layout Markdown section."""
        lines = [
            README_STRUCTURE_BEGIN,
            "### Curation Subset",
            "",
            f"Cohort-scoped curation entities live under `data/<cohort>/{CURATION_DATASET_SUBDIR}/`. Repository-level curation entities live once under `data/{CURATION_DATASET_SUBDIR}/` and are keyed by `repository_key`.",
            "",
            "- **CuratedPullRequests**: thin final curated PR membership/index rows with stable PR and repository join keys.",
            "- **RefactoringResults**: original PR, curation future-snapshot, and longitudinal refactoring summaries for analyzed snapshots, distinguished by `result_source` and optional `input_mode`, with fixed Murphy-Hill count columns, refactoring mapping JSONs, and `operations_json` embedded per result.",
            "- **MaintainabilityResults**: before, after, and future maintainability measures for available snapshots.",
            "- **CodeSmellResults**: one row per available PR snapshot with smell-tool metadata, skip metadata, smell totals, raw and standardized smell counts, raw-to-standardized and standardized-to-Mantyla maps, and fixed Mantyla category count columns.",
            f"- **RepositoryMetadata**: root-level curation-added repository metadata, README text/hash, and repository file lists with one row per `repository_key` under `data/{CURATION_DATASET_SUBDIR}/RepositoryMetadata/`.",
            f"- **RepositoryTopicClassifications**: root-level repository top-k topic predictions with one row per `repository_key` under `data/{CURATION_DATASET_SUBDIR}/RepositoryTopicClassifications/`.",
            "- **SnapshotManifests**: logical PR snapshot metadata and availability for `before`, `after`, and future checkpoints, including missing reasons and file availability diagnostics.",
            "- **SnapshotFileRefs**: snapshot path to deduplicated content hash references.",
            "- **SnapshotFileBlobs**: deduplicated text or base64 binary snapshot content partitioned by SHA-256 prefix.",
            "",
            README_STRUCTURE_END,
        ]
        return "\n".join(lines)

    def _remove_marked_section(self, text: str, begin: str, end: str) -> str:
        """Remove a generated Markdown section delimited by explicit markers."""
        pattern = re.compile(
            r"\n*" + re.escape(begin) + r".*?" + re.escape(end) + r"\n*",
            re.DOTALL,
        )
        return re.sub(r"\n{3,}", "\n\n", pattern.sub("\n\n", text)).strip()

    def _insert_curation_summary_section(self, text: str, section: str) -> str:
        """Insert the generated summary into the dataset-card body."""
        pattern = re.compile(
            re.escape(README_SUMMARY_BEGIN) + r".*?" + re.escape(README_SUMMARY_END),
            re.DOTALL,
        )
        body = pattern.sub("", text)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        for marker in ("## Dataset Summary", "## Dataset Overview"):
            index = body.find(marker)
            if index < 0:
                continue
            next_heading = body.find("\n## ", index + len(marker))
            if next_heading < 0:
                return body.rstrip() + "\n\n" + section + "\n"
            return (
                body[:next_heading].rstrip()
                + "\n\n"
                + section
                + "\n\n"
                + body[next_heading:].lstrip("\n")
            )
        marker = "## Dataset Structure"
        index = body.find(marker)
        dataset_summary = "## Dataset Summary\n\n" + section
        if index < 0:
            return body.rstrip() + "\n\n" + dataset_summary + "\n"
        return body[:index].rstrip() + "\n\n" + dataset_summary + "\n\n" + body[index:].lstrip("\n")

    def _insert_structure_section(self, text: str, section: str) -> str:
        """Insert or replace the generated curation structure section."""
        pattern = re.compile(
            re.escape(README_STRUCTURE_BEGIN) + r".*?" + re.escape(README_STRUCTURE_END),
            re.DOTALL,
        )
        if pattern.search(text):
            return pattern.sub(section, text)
        marker = "## Dataset Structure"
        index = text.find(marker)
        if index < 0:
            return text.rstrip() + "\n\n" + section + "\n"
        next_heading = text.find("\n## ", index + len(marker))
        if next_heading < 0:
            return text.rstrip() + "\n\n" + section + "\n"
        return (
            text[:next_heading].rstrip()
            + "\n\n"
            + section
            + "\n"
            + text[next_heading:].lstrip("\n")
        )

    def _write_dataset_card(self) -> Path:
        """Write the local dataset card with curation configs and summaries."""
        existing = self._remote_text_file("README.md")
        if existing is None:
            existing = "# Post-Processed Pull Request Dataset\n\n## Dataset Structure\n\n"
        yaml_lines, body = self._split_front_matter(existing)
        yaml_lines = self._merge_front_matter_configs(yaml_lines)
        body = self._insert_structure_section(body, self._curation_structure_section())
        body = self._insert_curation_summary_section(body, self._curation_summary_section())
        body = self._remove_marked_section(
            body, README_TOPIC_SUMMARY_BEGIN, README_TOPIC_SUMMARY_END
        )
        body = self._remove_marked_section(
            body,
            README_LONGITUDINAL_REFACTORING_SUMMARY_BEGIN,
            README_LONGITUDINAL_REFACTORING_SUMMARY_END,
        )
        readme = "---\n" + "\n".join(yaml_lines).rstrip() + "\n---\n\n" + body.strip() + "\n"
        path = self.local_output_dir / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readme, encoding="utf-8")
        return path

    def _curation_entity_manifest_path(self, entity_name: str) -> str:
        """Return the manifest path pattern for one curation entity."""
        root_prefix = (
            f"data/{CURATION_DATASET_SUBDIR}"
            if entity_name in ROOT_CURATION_ENTITY_NAMES
            else f"data/<cohort>/{CURATION_DATASET_SUBDIR}"
        )
        if entity_name == ENTITY_SNAPSHOT_FILE_BLOBS:
            return (
                f"{root_prefix}/{entity_name}/hash_prefix=<prefix>/shard-*/"
                f"{ENTITY_FILE_PREFIXES[entity_name]}-*.parquet"
            )
        if entity_name == ENTITY_REFACTORING_RESULTS:
            return (
                f"{root_prefix}/{entity_name}/**/"
                f"{ENTITY_FILE_PREFIXES[entity_name]}-*.parquet"
            )
        return (
            f"{root_prefix}/{entity_name}/shard-*/"
            f"{ENTITY_FILE_PREFIXES[entity_name]}-*.parquet"
        )

    def _write_schema_manifest(self) -> Path:
        """Write the local schema manifest for public curation entities."""
        existing = self._remote_text_file("schema_manifest.json")
        manifest: dict[str, Any]
        if existing:
            try:
                loaded = json.loads(existing)
                manifest = loaded if isinstance(loaded, dict) else {}
            except Exception:
                manifest = {}
        else:
            manifest = {}
        manifest["schema_version"] = self.curation_schema_version
        manifest["curation"] = {
            "schema_version": self.curation_schema_version,
            "layout": (
                f"Cohort-scoped entities: data/<cohort>/{CURATION_DATASET_SUBDIR}/<Entity>/**/*.parquet; "
                f"repository-level entities: data/{CURATION_DATASET_SUBDIR}/<Entity>/**/*.parquet"
            ),
            "entities": {
                entity_name: {
                    "path": self._curation_entity_manifest_path(entity_name),
                    "primary_key": ENTITY_PRIMARY_KEY_COLUMNS.get(entity_name),
                    "natural_key": ENTITY_NATURAL_KEY_COLUMNS.get(entity_name, []),
                    "foreign_keys": ENTITY_FOREIGN_KEYS.get(entity_name, []),
                    "columns": [
                        {"name": field.name, "type": str(field.type)}
                        for field in ENTITY_SCHEMAS[entity_name]
                    ],
                }
                for entity_name in ENTITY_NAMES
            },
        }
        path = self.local_output_dir / "schema_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    def _should_upload(self) -> bool:
        """Return whether repo id and token are both configured."""
        return bool(self.target_huggingface_repo_id and self.huggingface_token)

    def _standardized_repo_path(self, path: Path) -> str:
        """Map a local staged path to its Hugging Face repo path."""
        return Path(path).relative_to(self.local_output_dir).as_posix()

    def _root_curation_entity_paths(self) -> list[Path]:
        """Return root-scoped repository entity parquet paths."""
        root = _curation_dataset_dir(self.data_root)
        if not root.exists():
            return []
        paths: list[Path] = []
        for entity_name in ROOT_CURATION_ENTITY_NAMES:
            entity_root = root / entity_name
            if entity_root.is_dir():
                paths.extend(entity_root.rglob("*.parquet"))
        return sorted(paths)

    def _cohort_curation_entity_paths(self, cohort: str) -> list[Path]:
        """Return cohort-scoped curation entity parquet paths."""
        root = _curation_dataset_dir(self.data_root / cohort)
        if not root.exists():
            return []
        paths: list[Path] = []
        for entity_name in COHORT_CURATION_ENTITY_NAMES:
            entity_root = root / entity_name
            if entity_root.is_dir():
                paths.extend(entity_root.rglob("*.parquet"))
        return sorted(paths)

    def _parquet_upload_batches(self, cohort: str | None = None) -> list[tuple[Path, list[Path]]]:
        """Group uploadable parquet paths by directory."""
        batches: dict[Path, list[Path]] = defaultdict(list)
        root = _curation_dataset_dir(self.data_root / cohort) if cohort else self.data_root
        if not root.exists():
            return []
        parquet_paths = self._parquet_upload_paths(cohort)
        for parquet_path in parquet_paths:
            batches[parquet_path.parent].append(parquet_path)
        return sorted(batches.items(), key=lambda item: str(item[0]).lower())

    def _parquet_upload_paths(self, cohort: str | None = None) -> list[Path]:
        """Return uploadable parquet paths for all cohorts or one cohort."""
        root = _curation_dataset_dir(self.data_root / cohort) if cohort else self.data_root
        if not root.exists():
            return []
        if cohort:
            return sorted(
                self._root_curation_entity_paths() + self._cohort_curation_entity_paths(cohort)
            )
        paths: list[Path] = self._root_curation_entity_paths()
        for cohort_root in self.data_root.iterdir():
            if not cohort_root.is_dir():
                continue
            if cohort_root.name == CURATION_DATASET_SUBDIR:
                continue
            curation_root = _curation_dataset_dir(cohort_root)
            for entity_name in COHORT_CURATION_ENTITY_NAMES:
                entity_root = curation_root / entity_name
                if entity_root.is_dir():
                    paths.extend(entity_root.rglob("*.parquet"))
        return sorted(paths)

    def _upload_candidate_paths(self) -> list[Path]:
        """Return local files that can be published for this prepared package."""
        metadata_paths = [
            self.local_output_dir / "README.md",
            self.local_output_dir / "schema_manifest.json",
            self.local_output_dir / "upload_curation_manifest.json",
        ]
        metadata_paths.extend(
            sorted(self.data_root.glob(f"*/{CURATION_DATASET_SUBDIR}/metadata.json"))
        )
        return [
            *[path for path in metadata_paths if path.exists()],
            *self._parquet_upload_paths(),
        ]

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
        """Persist an upload plan beside the staged parquet outputs."""
        plan_path = self.local_output_dir / "upload_curation_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(self.build_upload_plan(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return plan_path

    def write_run_manifest(self, *, settings: dict[str, Any] | None = None) -> Path:
        """Write a safe manifest for local preparation and optional upload."""
        manifest_path = self.local_output_dir / "upload_curation_manifest.json"
        payload = {
            "manifest_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "settings": settings or {},
            "entities": list(ENTITY_NAMES),
            "maintainability_metrics": [
                "maintainability_index",
                "cyclomatic_complexity",
                "halstead_volume",
                "fan_out",
                "comment_ratio",
                "loc",
                "duplicated_lines_density",
            ],
            "data_root": str(self.data_root),
            "summary": self.summary,
            "upload_plan": {
                key: value
                for key, value in self.build_upload_plan().items()
                if key != "files"
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path

    def _root_curation_entity_allow_patterns(self) -> list[str]:
        """Return Hugging Face allow patterns for root-scoped curation entities."""
        return [
            f"{self.standardized_data_subdir}/{CURATION_DATASET_SUBDIR}/{entity_name}/**/*.parquet"
            for entity_name in ROOT_CURATION_ENTITY_NAMES
        ]

    def _curation_entity_allow_patterns(
        self,
        cohort: str | None = None,
        *,
        include_root_entities: bool = True,
    ) -> list[str]:
        """Return Hugging Face allow patterns for curation parquet uploads."""
        root_patterns = (
            self._root_curation_entity_allow_patterns() if include_root_entities else []
        )
        if cohort:
            relative_root = (
                _curation_dataset_dir(self.data_root / cohort)
                .relative_to(self.local_output_dir)
                .as_posix()
            )
            return root_patterns + [
                f"{relative_root}/{entity_name}/**/*.parquet"
                for entity_name in COHORT_CURATION_ENTITY_NAMES
            ]
        cohort_patterns = [
            f"{self.standardized_data_subdir}/*/{CURATION_DATASET_SUBDIR}/{entity_name}/**/*.parquet"
            for entity_name in COHORT_CURATION_ENTITY_NAMES
        ]
        return root_patterns + cohort_patterns

    def _is_hourly_rate_limit_error(self, exc: Exception) -> bool:
        """Detect Hugging Face hourly repository or usage quota errors."""
        text = str(exc).lower()
        return (
            ("rate limit reached" in text and "reset hourly" in text)
            or "free usage limit" in text
            or ("repository commits" in text and "per hour" in text)
            or "retry this action in about 1 hour" in text
        )

    def _is_short_term_rate_limit_error(self, exc: Exception) -> bool:
        """Detect short-term request throttling errors."""
        text = str(exc).lower()
        return not self._is_hourly_rate_limit_error(exc) and (
            "429" in text or "too many requests" in text
        )

    def _is_transient_hf_server_error(self, exc: Exception) -> bool:
        """Detect transient Hugging Face server-side failures."""
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "500 server error",
                "502 server error",
                "503 server error",
                "504 server error",
                "internal server error",
                "bad gateway",
                "service unavailable",
                "gateway timeout",
            )
        )

    def _is_five_minute_quota_error(self, exc: Exception) -> bool:
        """Detect five-minute quota messages from Hugging Face upload APIs."""
        text = str(exc).lower()
        return "5 minutes period" in text and ("rate limit" in text or "quota" in text)

    def _compute_retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        """Return the retry delay for a failed Hugging Face upload attempt."""
        if self._is_hourly_rate_limit_error(exc):
            return self.upload_hourly_rate_limit_delay_seconds
        if self._is_five_minute_quota_error(exc):
            return max(self.upload_short_term_rate_limit_window_seconds, 5 * 60.0)
        if self._is_transient_hf_server_error(exc) or self._is_short_term_rate_limit_error(exc):
            exponential = self.upload_retry_base_seconds * (2 ** max(0, attempt - 1))
            return min(exponential, self.upload_short_term_rate_limit_window_seconds)
        if self._consecutive_upload_failures >= self.upload_consecutive_failure_threshold:
            return self.upload_consecutive_failure_delay_seconds
        return self.upload_retry_base_seconds * (2 ** max(0, attempt - 1))

    def _upload_with_retry(self, *, local_path: Path, repo_path: str) -> None:
        """Upload one metadata file with retry and local state tracking."""
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
                )
                self._consecutive_upload_failures = 0
                self.state_store.mark_uploaded(repo_path, local_path)
                return
            except Exception as exc:
                self._consecutive_upload_failures += 1
                print(
                    "[post-processing/upload-curation-data] Metadata upload failed "
                    f"(attempt={attempts}, repo_path={repo_path}): {exc}"
                )
                if attempts >= self.upload_max_retries:
                    raise
                time.sleep(self._compute_retry_delay_seconds(exc, attempts))

    def _upload_large_folder_pattern_with_retry(
        self, *, allow_patterns: list[str], progress_label: str
    ) -> None:
        """Upload parquet files matching allow patterns with retry handling."""
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
                    "[post-processing/upload-curation-data] Large-folder upload failed "
                    f"(attempt={attempts}, scope={progress_label}): {exc}"
                )
                if attempts >= self.upload_max_retries:
                    raise
                time.sleep(self._compute_retry_delay_seconds(exc, attempts))

    def upload_outputs(self, *, dry_run: bool = False) -> None:
        """Upload standardized parquet files and metadata to Hugging Face."""
        if dry_run:
            plan_path = self.write_upload_plan()
            print(
                "[post-processing/upload-curation-data] Dry run; wrote upload plan "
                f"to {plan_path}."
            )
            return
        if not self._should_upload():
            print(
                "[post-processing/upload-curation-data] Hugging Face upload skipped; token/repo not configured."
            )
            return
        api = HfApi(token=self.huggingface_token)
        api.create_repo(
            repo_id=self.target_huggingface_repo_id,
            repo_type="dataset",
            exist_ok=True,
        )
        if self.data_root.exists():
            for cohort_root in self.data_root.iterdir():
                if cohort_root.is_dir():
                    _curation_dataset_dir(cohort_root)
        metadata_paths = [
            self.local_output_dir / "README.md",
            self.local_output_dir / "schema_manifest.json",
            self.local_output_dir / "upload_curation_manifest.json",
            *sorted(self.data_root.glob(f"*/{CURATION_DATASET_SUBDIR}/metadata.json")),
        ]
        parquet_paths = self._parquet_upload_paths()
        total_upload_files = len(metadata_paths) + len(parquet_paths)
        completed = 0
        for metadata_path in metadata_paths:
            if not metadata_path.exists():
                continue
            repo_path = self._standardized_repo_path(metadata_path)
            self._upload_with_retry(local_path=metadata_path, repo_path=repo_path)
            completed += 1
            print(
                "[post-processing/upload-curation-data] Uploaded metadata file "
                f"{completed}/{total_upload_files}: {repo_path}"
            )
        if parquet_paths:
            self._upload_large_folder_pattern_with_retry(
                allow_patterns=self._curation_entity_allow_patterns(),
                progress_label=f"all curation parquet files ({len(parquet_paths)} files)",
            )
            for parquet_path in parquet_paths:
                self.state_store.mark_uploaded(
                    self._standardized_repo_path(parquet_path), parquet_path
                )
            completed += len(parquet_paths)
            print(
                "[post-processing/upload-curation-data] Completed upload for all "
                f"curation parquet files ({completed}/{total_upload_files})."
            )
        self.summary["uploaded_files"] = completed

