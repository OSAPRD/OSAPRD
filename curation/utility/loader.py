"""Load local PR parquet rows for curation.

The active input format is extraction's local parquet output. The loader also
keeps read-only compatibility for older local layouts so existing local exports
can be curated without re-exporting them. All inputs are normalized into
extraction DTOs or plain row dictionaries before preprocessing.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, get_args, get_origin

import pandas as pd
import pyarrow.parquet as pq

try:
    from curation.config.storage_config import (
        CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE,
        CURATION_LOCAL_DATA_FORMAT,
        CURATION_REQUIRE_COMPLETE_SHARD_SET,
        LOCAL_DIRECTORIES,
    )
except ModuleNotFoundError:
    _STORAGE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "storage_config.py"
    _STORAGE_CONFIG_SPEC = importlib.util.spec_from_file_location(
        "flat_layout_curation_storage_config",
        _STORAGE_CONFIG_PATH,
    )
    if _STORAGE_CONFIG_SPEC is None or _STORAGE_CONFIG_SPEC.loader is None:
        raise ImportError(f"Unable to load storage config from {_STORAGE_CONFIG_PATH}")
    _STORAGE_CONFIG_MODULE = importlib.util.module_from_spec(_STORAGE_CONFIG_SPEC)
    _STORAGE_CONFIG_SPEC.loader.exec_module(_STORAGE_CONFIG_MODULE)
    CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE = (
        _STORAGE_CONFIG_MODULE.CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE
    )
    CURATION_LOCAL_DATA_FORMAT = _STORAGE_CONFIG_MODULE.CURATION_LOCAL_DATA_FORMAT
    CURATION_REQUIRE_COMPLETE_SHARD_SET = _STORAGE_CONFIG_MODULE.CURATION_REQUIRE_COMPLETE_SHARD_SET
    LOCAL_DIRECTORIES = _STORAGE_CONFIG_MODULE.LOCAL_DIRECTORIES

from extraction.dtos.dtos import (
    FileChange,
    Label,
    PullRequest,
    Repository,
    RepositoryPeek,
    UserPeek,
)


KNOWN_GROUPS = {
    "claude",
    "codegen",
    "codex",
    "copilot",
    "cosine",
    "cursor",
    "devin",
    "humans",
    "human",
    "jules",
    "junie",
    "openhands",
}

NESTED_JSON_FIELDS = {
    "author",
    "merged_by",
    "base_repository",
    "head_repository",
    "base_repository_full",
    "head_repository_full",
    "files",
    "labels",
    "requested_reviewers",
    "timeline_items",
    "comments",
    "reviews",
    "commits",
    "post_merge_file_snapshots",
}

CURATION_PROCESSED_JSON_FIELDS = NESTED_JSON_FIELDS | {
    "repository_metadata",
    "hydration",
    "metrics",
    "metrics_aggregate",
    "processing_timing",
}

REQUIRED_FULL_PULL_REQUEST_COLUMNS = {
    "id",
    "title",
    "url",
    "number",
    "body",
    "state",
    "created_at",
    "is_draft",
    "changed_files",
    "is_cross_repository",
    "locked",
    "is_in_merge_queue",
    "additions",
    "deletions",
    "author",
    "base_repository",
    "head_repository",
    "timeline_count",
    "files",
}

_FULLPR_PREFLIGHT_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCAL_SOURCE_TYPE = "local"


def _scalar_kind(annotation: Any) -> str:
    """Classify PullRequest dataclass fields for scalar normalization."""
    origin = get_origin(annotation)
    if origin is None:
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is str:
            return "str"
        return "other"
    if origin in (list, tuple, dict):
        return "other"
    if str(origin).endswith("Union"):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return _scalar_kind(non_none[0])
    return "other"


PR_FIELD_KINDS: Dict[str, str] = {
    field.name: _scalar_kind(field.type)
    for field in fields(PullRequest)
}
PULL_REQUEST_FIELD_NAMES: Set[str] = set(PR_FIELD_KINDS.keys())


def _normalize_group(group: str) -> str:
    """Normalize group names for filtering."""
    return (group or "").strip().lower()


def _group_from_path(path: Path) -> Optional[str]:
    """Extract group name from a file path like .../data/<group>/..."""
    parts = [p.lower() for p in path.parts]
    if "data" in parts:
        idx = len(parts) - 1 - parts[::-1].index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    for segment in parts:
        if segment in KNOWN_GROUPS:
            return segment
    return None


def _is_fullpullrequests_path(path: Path) -> bool:
    """Return True when a path belongs to the legacy FullPullRequests layout."""
    parts = [segment.lower() for segment in path.parts]
    return "fullpullrequests" in parts


def _filter_group(group: Optional[str], selector: Optional[str]) -> bool:
    """Return True if the group matches the selector."""
    if selector is None:
        return True
    selector_norm = _normalize_group(selector)
    group_norm = _normalize_group(group or "")
    if selector_norm in ("human", "humans"):
        return group_norm in ("human", "humans")
    if selector_norm == "agentic":
        return group_norm not in ("human", "humans")
    return group_norm == selector_norm


def _safe_json_loads(value: Any) -> Any:
    """Decode JSON strings while leaving already-materialized values alone."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _normalize_bool(value: Any) -> Optional[bool]:
    """Normalize bool-like parquet values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off", ""}:
            return False
    return None


def _normalize_int(value: Any) -> Optional[int]:
    """Normalize int-like parquet values."""
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _normalize_pull_request_scalars(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce scalar PullRequest fields to the DTO's expected Python types."""
    normalized = dict(payload)
    for key, kind in PR_FIELD_KINDS.items():
        if key not in normalized:
            continue
        value = normalized.get(key)
        if value is None:
            continue
        if kind == "int":
            normalized_int = _normalize_int(value)
            if normalized_int is not None:
                normalized[key] = normalized_int
        elif kind == "bool":
            normalized_bool = _normalize_bool(value)
            if normalized_bool is not None:
                normalized[key] = normalized_bool
        elif kind == "str":
            if isinstance(value, (dict, list, tuple)):
                continue
            normalized[key] = str(value)
    return normalized


def _normalize_row_for_local_data_format(
    row: Dict[str, Any],
    *,
    local_data_format: str,
) -> Dict[str, Any]:
    """Normalize one parquet row according to the configured local layout."""
    payload = dict(row)
    if local_data_format in {"fullpullrequests_sharded", "curation_processed"}:
        nested_fields = (
            CURATION_PROCESSED_JSON_FIELDS
            if local_data_format == "curation_processed"
            else NESTED_JSON_FIELDS
        )
        for field_name in nested_fields:
            if field_name in payload:
                payload[field_name] = _safe_json_loads(payload.get(field_name))
        if payload.get("label_count") is None:
            labels_value = _safe_json_loads(payload.get("labels"))
            if isinstance(labels_value, list):
                payload["label_count"] = len(labels_value)
            else:
                payload["label_count"] = 0
    return _normalize_pull_request_scalars(payload)


def _coerce_userpeek(value: Any) -> Optional[UserPeek]:
    """Best-effort conversion to UserPeek."""
    if value is None:
        return None
    value = _safe_json_loads(value)
    if isinstance(value, UserPeek):
        return value
    if isinstance(value, dict):
        return UserPeek(**value)
    return None


def _coerce_repopeek(value: Any) -> Optional[RepositoryPeek]:
    """Best-effort conversion to RepositoryPeek."""
    if value is None:
        return None
    value = _safe_json_loads(value)
    if isinstance(value, RepositoryPeek):
        return value
    if isinstance(value, dict):
        return RepositoryPeek(**value)
    return None


def _coerce_label(value: Any) -> Optional[Label]:
    """Best-effort conversion to Label."""
    if value is None:
        return None
    value = _safe_json_loads(value)
    if isinstance(value, Label):
        return value
    if isinstance(value, dict):
        return Label(**value)
    return None


def _coerce_filechange(value: Any) -> Optional[FileChange]:
    """Best-effort conversion to FileChange."""
    if value is None:
        return None
    value = _safe_json_loads(value)
    if isinstance(value, FileChange):
        return value
    if isinstance(value, dict):
        return FileChange(**value)
    return None


def _coerce_file_list(value: Any) -> Optional[List[FileChange]]:
    """Best-effort conversion to a list of FileChange."""
    if value is None:
        return None
    value = _safe_json_loads(value)
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = [value]
    elif hasattr(value, "tolist"):
        items = value.tolist()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        return None

    coerced = [fc for fc in (_coerce_filechange(x) for x in items) if fc]
    return coerced if coerced else None


def _coerce_pull_request(
    data: Dict[str, Any],
    *,
    local_data_format: str,
) -> PullRequest:
    """Convert a dict payload into a PullRequest DTO."""
    payload = _normalize_row_for_local_data_format(
        data,
        local_data_format=local_data_format,
    )
    payload = {
        key: value
        for key, value in payload.items()
        if key in PULL_REQUEST_FIELD_NAMES
    }

    payload["author"] = _coerce_userpeek(payload.get("author"))
    payload["base_repository"] = _coerce_repopeek(payload.get("base_repository"))
    payload["head_repository"] = _coerce_repopeek(payload.get("head_repository"))

    labels = payload.get("labels")
    labels = _safe_json_loads(labels)
    if isinstance(labels, list):
        payload["labels"] = [lbl for lbl in (_coerce_label(x) for x in labels) if lbl]

    files = _coerce_file_list(payload.get("files"))
    if files is not None:
        payload["files"] = files

    return PullRequest(**payload)


def _field_names(cls: Any) -> set[str]:
    """Return dataclass field names for a DTO."""
    return {f.name for f in fields(cls)}


def _sample_keys(rows: List[Dict[str, Any]], key: str) -> set[str]:
    """Extract keys from nested dict/list values for a field."""
    keys: set[str] = set()
    for row in rows:
        value = _safe_json_loads(row.get(key))
        if isinstance(value, dict):
            keys.update(value.keys())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    keys.update(item.keys())
        if keys:
            break
    return keys


def _log_schema_check(rows: List[Dict[str, Any]]) -> None:
    """Log a schema comparison between raw parquet rows and DTOs."""
    if not rows:
        return
    pr_fields = _field_names(PullRequest)
    row_keys: set[str] = set()
    for row in rows:
        row_keys.update(row.keys())
    missing = sorted(pr_fields - row_keys)
    extra = sorted(row_keys - pr_fields)
    if missing:
        print(f"[loader] Missing PR fields in parquet: {missing}")
    if extra:
        print(f"[loader] Extra fields in parquet not in DTO: {extra}")

    nested_checks = {
        "files": FileChange,
        "labels": Label,
        "base_repository_full": Repository,
        "head_repository_full": Repository,
        "author": UserPeek,
    }
    for field_name, dto in nested_checks.items():
        nested_keys = _sample_keys(rows, field_name)
        if not nested_keys:
            continue
        dto_fields = _field_names(dto)
        missing_nested = sorted(dto_fields - nested_keys)
        extra_nested = sorted(nested_keys - dto_fields)
        if missing_nested:
            print(f"[loader] Missing {field_name} fields in parquet: {missing_nested}")
        if extra_nested:
            print(f"[loader] Extra {field_name} fields in parquet not in DTO: {extra_nested}")


def _normalize_template_subpath(group: str, template: str) -> Path:
    """Render a configured legacy shard template into a relative path."""
    rendered = template.format(cohort=group)
    rendered = rendered.replace("\\", "/").strip("/")
    return Path(*[segment for segment in rendered.split("/") if segment])


def _iter_parquet_files_from_local_extraction(directories: Iterable[str]) -> List[Path]:
    """Collect all parquet files under the provided roots."""
    paths: List[Path] = []
    for root in directories:
        root_path = Path(root)
        print(f"[loader] Scanning local root: {root_path}")
        if not root_path.exists():
            print(f"[loader] Skipping missing root: {root_path}")
            continue
        paths.extend(root_path.rglob("*.parquet"))
    return sorted(paths)


def _collect_fullpullrequest_shards(
    directories: Iterable[str],
    *,
    group: Optional[str],
    template: str,
) -> List[Path]:
    """Collect parquet shards from the legacy FullPullRequests directory layout."""
    paths: List[Path] = []
    target_group = _normalize_group(group) if group else None

    for root in directories:
        root_path = Path(root)
        print(f"[loader] Scanning local root: {root_path}")
        if not root_path.exists():
            print(f"[loader] Skipping missing root: {root_path}")
            continue

        if target_group:
            cohort_dir = root_path / _normalize_template_subpath(target_group, template)
            if cohort_dir.exists():
                paths.extend(sorted(cohort_dir.rglob("*.parquet")))
                continue

        for candidate in root_path.rglob("*.parquet"):
            if not _is_fullpullrequests_path(candidate):
                continue
            if target_group and not _filter_group(_group_from_path(candidate), target_group):
                continue
            paths.append(candidate)

    return sorted(set(paths))


def _preflight_fullpullrequest_shards(
    parquet_files: Iterable[Path],
    *,
    group: Optional[str],
    strict: bool,
) -> Dict[str, Any]:
    """Validate legacy FullPullRequests shards before the main run starts."""
    files = sorted(parquet_files)
    manifest: Dict[str, Any] = {
        "group": group,
        "strict": bool(strict),
        "shard_count": len(files),
        "total_rows": 0,
        "invalid_shards": [],
        "missing_required_columns": {},
        "shards": [],
    }

    for path in files:
        shard_info: Dict[str, Any] = {
            "path": str(path),
            "size_bytes": 0,
            "row_count": 0,
            "columns": [],
        }
        try:
            stat = path.stat()
            shard_info["size_bytes"] = int(stat.st_size)
        except Exception:
            shard_info["size_bytes"] = 0

        try:
            parquet = pq.ParquetFile(path)
            schema_names = list(parquet.schema_arrow.names)
            shard_info["columns"] = schema_names
            shard_info["row_count"] = int(parquet.metadata.num_rows if parquet.metadata else 0)
            missing = sorted(REQUIRED_FULL_PULL_REQUEST_COLUMNS - set(schema_names))
            if missing:
                manifest["missing_required_columns"][str(path)] = missing
            manifest["total_rows"] += int(shard_info["row_count"])
        except Exception as exc:
            shard_info["error"] = str(exc)
            manifest["invalid_shards"].append({"path": str(path), "error": str(exc)})

        manifest["shards"].append(shard_info)

    print(
        "[loader] FullPullRequests shard preflight: "
        f"group={group}, shards={manifest['shard_count']}, total_rows={manifest['total_rows']}"
    )
    if manifest["invalid_shards"]:
        print(f"[loader] Invalid FullPullRequests shards: {len(manifest['invalid_shards'])}")
    if manifest["missing_required_columns"]:
        print(
            "[loader] FullPullRequests shards with missing required columns: "
            f"{len(manifest['missing_required_columns'])}"
        )

    if strict:
        if manifest["shard_count"] <= 0:
            raise RuntimeError(
                "No FullPullRequests shards discovered for cohort="
                f"{group!r}. Expected local shards at "
                "data/<cohort>/FullPullRequests/*.parquet."
            )
        if manifest["invalid_shards"]:
            first = manifest["invalid_shards"][0]
            raise RuntimeError(
                "FullPullRequests preflight failed due to unreadable shard: "
                f"{first.get('path')} ({first.get('error')})"
            )
        if manifest["missing_required_columns"]:
            sample_path, missing_cols = next(iter(manifest["missing_required_columns"].items()))
            raise RuntimeError(
                "FullPullRequests preflight failed due to missing required columns in "
                f"{sample_path}: {missing_cols}"
            )

    return manifest


def _fullpull_preflight_cache_key(
    *,
    directories: Iterable[str],
    group: Optional[str],
    template: str,
) -> str:
    """Return a stable key for cached legacy-shard preflight results."""
    roots = [str(Path(root)) for root in directories]
    roots.sort()
    normalized_group = _normalize_group(group or "") or "*"
    return "|".join([normalized_group, template, *roots])


def preflight_local_group_input(
    group: Optional[str],
    *,
    source_type: Optional[str] = None,
    local_directories: Optional[Iterable[str]] = None,
    local_data_format: Optional[str] = None,
    strict: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run a fail-fast local input preflight for one cohort/group."""
    resolved_source_type = (source_type or _LOCAL_SOURCE_TYPE).strip().lower()
    if resolved_source_type != _LOCAL_SOURCE_TYPE:
        raise ValueError(
            "Unsupported source_type={!r}. Curation loader only supports local input.".format(
                source_type
            )
        )

    resolved_format = (local_data_format or CURATION_LOCAL_DATA_FORMAT).strip().lower()
    resolved_local_directories = (
        list(local_directories) if local_directories is not None else list(LOCAL_DIRECTORIES)
    )
    strict_mode = CURATION_REQUIRE_COMPLETE_SHARD_SET if strict is None else bool(strict)

    if resolved_format != "fullpullrequests_sharded":
        return {
            "group": group,
            "source_type": resolved_source_type,
            "local_data_format": resolved_format,
            "shard_count": 0,
            "total_rows": 0,
            "skipped": True,
        }

    shards = _collect_fullpullrequest_shards(
        resolved_local_directories,
        group=group,
        template=CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE,
    )
    manifest = _preflight_fullpullrequest_shards(
        shards,
        group=group,
        strict=strict_mode,
    )
    manifest.update(
        {
            "source_type": resolved_source_type,
            "local_data_format": resolved_format,
            "skipped": False,
        }
    )
    cache_key = _fullpull_preflight_cache_key(
        directories=resolved_local_directories,
        group=group,
        template=CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE,
    )
    _FULLPR_PREFLIGHT_CACHE[cache_key] = dict(manifest)
    return manifest


def _resolve_parquet_files(
    *,
    group: Optional[str],
    source_type: Optional[str] = None,
    local_directories: Optional[Iterable[str]] = None,
    local_data_format: Optional[str] = None,
) -> Tuple[List[Path], bool, str]:
    """
    Resolve parquet files and whether path-based group filtering should be applied.

    Returns:
        (parquet_files, apply_group_file_filter, resolved_local_data_format)
    """
    resolved_source_type = (source_type or _LOCAL_SOURCE_TYPE).strip().lower()
    if resolved_source_type != _LOCAL_SOURCE_TYPE:
        raise ValueError(
            "Unsupported source_type={!r}. Curation loader only supports local input.".format(
                source_type
            )
        )
    resolved_local_directories = (
        list(local_directories) if local_directories is not None else list(LOCAL_DIRECTORIES)
    )
    resolved_local_data_format = (local_data_format or CURATION_LOCAL_DATA_FORMAT).strip().lower()
    if resolved_local_data_format == "fullpullrequests_sharded":
        shards = _collect_fullpullrequest_shards(
            resolved_local_directories,
            group=group,
            template=CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE,
        )
        cache_key = _fullpull_preflight_cache_key(
            directories=resolved_local_directories,
            group=group,
            template=CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE,
        )
        if cache_key not in _FULLPR_PREFLIGHT_CACHE:
            _FULLPR_PREFLIGHT_CACHE[cache_key] = _preflight_fullpullrequest_shards(
                shards,
                group=group,
                strict=bool(CURATION_REQUIRE_COMPLETE_SHARD_SET),
            )
        return (shards, False, resolved_local_data_format)

    return (
        _iter_parquet_files_from_local_extraction(resolved_local_directories),
        True,
        resolved_local_data_format,
    )


def iter_pr_rows(
    group: Optional[str] = None,
    *,
    source_type: Optional[str] = None,
    local_directories: Optional[Iterable[str]] = None,
    local_data_format: Optional[str] = None,
    batch_size: int = 256,
    columns: Optional[Iterable[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stream parquet rows as dictionaries.

    This iterator is intended for low-memory workflows (e.g. two-pass sampling).
    """
    parquet_files, apply_group_file_filter, resolved_local_data_format = _resolve_parquet_files(
        group=group,
        source_type=source_type,
        local_directories=local_directories,
        local_data_format=local_data_format,
    )
    print(
        "[loader] source_type={source}, group={group}, local_data_format={fmt}".format(
            source=_LOCAL_SOURCE_TYPE,
            group=group,
            fmt=resolved_local_data_format,
        )
    )
    print(f"[loader] Parquet files discovered: {len(parquet_files)}")

    selected_columns = [str(col) for col in (columns or []) if str(col).strip()]
    if selected_columns:
        print(f"[loader] Row projection columns: {selected_columns}")

    scanned = 0
    kept = 0
    for path in parquet_files:
        scanned += 1
        if apply_group_file_filter and not _filter_group(_group_from_path(path), group):
            continue
        kept += 1
        try:
            parquet = pq.ParquetFile(path)
            iter_kwargs: Dict[str, Any] = {"batch_size": max(1, int(batch_size))}
            if selected_columns:
                iter_kwargs["columns"] = selected_columns
            for batch in parquet.iter_batches(**iter_kwargs):
                for row in batch.to_pylist():
                    if isinstance(row, dict):
                        yield _normalize_row_for_local_data_format(
                            row,
                            local_data_format=resolved_local_data_format,
                        )
        except Exception:
            try:
                if selected_columns:
                    df = pd.read_parquet(path, columns=selected_columns)
                else:
                    df = pd.read_parquet(path)
                for row in df.to_dict(orient="records"):
                    if isinstance(row, dict):
                        yield _normalize_row_for_local_data_format(
                            row,
                            local_data_format=resolved_local_data_format,
                        )
            except Exception:
                continue
    print(f"[loader] Parquet files scanned: {scanned}, matched group: {kept}")


def iter_source_parquet_files(
    *,
    source_type: Optional[str] = None,
    local_directories: Optional[Iterable[str]] = None,
    local_data_format: Optional[str] = None,
) -> List[Path]:
    """Return parquet file paths for the configured source without loading records."""
    resolved_source_type = (source_type or _LOCAL_SOURCE_TYPE).strip().lower()
    if resolved_source_type != _LOCAL_SOURCE_TYPE:
        raise ValueError(
            "Unsupported source_type={!r}. Curation loader only supports local input.".format(
                source_type
            )
        )
    resolved_local_directories = (
        list(local_directories) if local_directories is not None else list(LOCAL_DIRECTORIES)
    )
    resolved_local_data_format = (local_data_format or CURATION_LOCAL_DATA_FORMAT).strip().lower()
    if resolved_local_data_format == "fullpullrequests_sharded":
        return _collect_fullpullrequest_shards(
            resolved_local_directories,
            group=None,
            template=CURATION_FULL_PULLREQUESTS_SUBDIR_TEMPLATE,
        )
    return _iter_parquet_files_from_local_extraction(resolved_local_directories)


def coerce_pull_request_record(
    data: Dict[str, Any],
    *,
    local_data_format: Optional[str] = None,
) -> PullRequest:
    """Public wrapper for normalizing one parquet row into a PullRequest DTO."""
    resolved_local_data_format = (local_data_format or CURATION_LOCAL_DATA_FORMAT).strip().lower()
    return _coerce_pull_request(data, local_data_format=resolved_local_data_format)


def load_prs(
    group: Optional[str] = None,
    *,
    source_type: Optional[str] = None,
    local_directories: Optional[Iterable[str]] = None,
    local_data_format: Optional[str] = None,
) -> List[PullRequest]:
    """
    Load PRs from local parquet files.

    Args:
        group: "human", "agentic", or a specific agent name (e.g. "claude").
        source_type: Optional compatibility parameter; only "local" is supported.
        local_directories: Optional override for local roots to scan.
        local_data_format: Optional local parquet data format override.

    Returns:
        List of PullRequest DTOs.
    """
    records: List[PullRequest] = []
    schema_rows: List[Dict[str, Any]] = []
    failed = 0
    resolved_local_data_format = (local_data_format or CURATION_LOCAL_DATA_FORMAT).strip().lower()
    for row in iter_pr_rows(
        group=group,
        source_type=source_type,
        local_directories=local_directories,
        local_data_format=resolved_local_data_format,
    ):
        if len(schema_rows) < 200:
            schema_rows.append(row)
        try:
            records.append(
                _coerce_pull_request(
                    row,
                    local_data_format=resolved_local_data_format,
                )
            )
        except Exception:
            failed += 1
    print(f"[loader] Total PR records loaded: {len(records)}")
    if failed:
        print(f"[loader] Failed to parse records: {failed}")
    _log_schema_check(schema_rows)
    return records


def load_prs_by_urls(
    urls: Iterable[str],
    group: Optional[str] = None,
    *,
    source_type: Optional[str] = None,
    local_directories: Optional[Iterable[str]] = None,
    local_data_format: Optional[str] = None,
    batch_size: int = 256,
) -> List[PullRequest]:
    """
    Stream all parquet rows and materialize only PRs whose URLs are requested.

    Only local parquet loading is supported.
    """
    ordered_urls = [str(url).strip() for url in urls if str(url).strip()]
    target_urls: Set[str] = set(ordered_urls)
    if not target_urls:
        return []
    found: Dict[str, PullRequest] = {}
    failed = 0
    resolved_local_data_format = (local_data_format or CURATION_LOCAL_DATA_FORMAT).strip().lower()
    for row in iter_pr_rows(
        group=group,
        source_type=source_type,
        local_directories=local_directories,
        local_data_format=resolved_local_data_format,
        batch_size=batch_size,
    ):
        url = str(row.get("url") or "").strip()
        if not url or url not in target_urls or url in found:
            continue
        try:
            found[url] = _coerce_pull_request(
                row,
                local_data_format=resolved_local_data_format,
            )
        except Exception:
            failed += 1
        if len(found) >= len(target_urls):
            break
    if failed:
        print(f"[loader] Failed to parse targeted records: {failed}")
    if len(found) < len(target_urls):
        missing = len(target_urls) - len(found)
        print(f"[loader] Targeted PR rows missing or unparsable: {missing}")
    ordered = [found[url] for url in ordered_urls if url in found]
    return ordered
