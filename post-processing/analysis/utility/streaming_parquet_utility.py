"""Streaming parquet helpers for low-memory analysis registration.

This module intentionally does not depend on DuckDB or pyarrow.dataset.  It
reads one parquet file at a time through ``pyarrow.parquet.ParquetFile`` and
exposes small row/extraction helpers that pipeline accumulators can reuse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from curation_parquet_utility import CohortParquetFiles
from stable_deduplication_utility import stable_pr_dedup_key
from topic_groups_utility import normalize_repository_key


DEFAULT_STREAMING_BATCH_SIZE = 512


@dataclass(frozen=True)
class SourceFileFingerprint:
    """Stable-enough source metadata for resumable streaming shards."""

    path: str
    size_bytes: int
    modified_time_ns: int


@dataclass(frozen=True)
class StreamingPrFacts:
    """Compact common PR facts extracted from one curation row."""

    cohort: str
    source_file: str
    source_row_number: int
    pr_id: str | None
    pr_url: str | None
    pr_number: str | None
    created_at: str | None
    language: str | None
    authorship_group: str
    agent_label: str | None
    longitudinal_selected: bool
    base_repository_id: str | None
    repository_key: str | None
    stargazer_count: int | None
    changed_files_count: int | None
    additions: int | None
    deletions: int | None
    changed_line_count: int | None
    stable_pr_key: str | None


def require_pyarrow_parquet():
    """Return ``pyarrow.parquet`` or raise a clear runtime error."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "pyarrow is required for streaming analysis. Install pyarrow or use "
            "the analysis Docker image with pyarrow included."
        ) from exc
    return pq


def source_file_fingerprint(path: Path | str) -> SourceFileFingerprint:
    """Return size/mtime metadata for a source parquet file."""
    resolved = Path(path)
    stat = resolved.stat()
    return SourceFileFingerprint(
        path=str(resolved),
        size_bytes=int(stat.st_size),
        modified_time_ns=int(stat.st_mtime_ns),
    )


def iter_parquet_rows(
    path: Path | str,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = DEFAULT_STREAMING_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield parquet rows as Python dictionaries from one file at a time."""
    pq = require_pyarrow_parquet()
    parquet_file = pq.ParquetFile(str(path))
    projected_columns = (
        _available_projection_columns(parquet_file.schema_arrow, columns)
        if columns is not None
        else None
    )
    for batch in parquet_file.iter_batches(
        batch_size=int(batch_size),
        columns=projected_columns,
    ):
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def iter_cohort_parquet_rows(
    cohort_inputs: Iterable[CohortParquetFiles],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = DEFAULT_STREAMING_BATCH_SIZE,
    progress_logger: Callable[..., Any] | None = None,
    progress_interval: int = 100,
) -> Iterator[tuple[str, Path, int, dict[str, Any]]]:
    """Yield ``(cohort, source_file, row_number, row)`` across cohort inputs."""
    for cohort_input in cohort_inputs:
        paths = tuple(cohort_input.paths)
        _log(
            progress_logger,
            "streaming_cohort_start",
            cohort=cohort_input.cohort,
            files=len(paths),
        )
        for file_index, path in enumerate(paths, start=1):
            _log(
                progress_logger,
                "streaming_file_start",
                cohort=cohort_input.cohort,
                file_index=file_index,
                files=len(paths),
                path=path,
            )
            row_count = 0
            for row_count, row in enumerate(
                iter_parquet_rows(path, columns=columns, batch_size=batch_size),
                start=1,
            ):
                yield cohort_input.cohort, Path(path), row_count, row
                if progress_interval > 0 and row_count % progress_interval == 0:
                    _log(
                        progress_logger,
                        "streaming_file_progress",
                        cohort=cohort_input.cohort,
                        file_index=file_index,
                        rows=row_count,
                    )
            _log(
                progress_logger,
                "streaming_file_done",
                cohort=cohort_input.cohort,
                file_index=file_index,
                rows=row_count,
            )
        _log(
            progress_logger,
            "streaming_cohort_done",
            cohort=cohort_input.cohort,
            files=len(paths),
        )


def extract_common_pr_facts(
    row: Mapping[str, Any],
    *,
    cohort: str,
    source_file: Path | str,
    source_row_number: int,
) -> StreamingPrFacts:
    """Extract compact PR facts shared by all streaming analysis pipelines."""
    repository = _repository_metadata(row)
    authored_by_agent = coerce_bool(row.get("authored_by_agent"), default=False)
    author_agent = coerce_str(row.get("author_agent"))
    discovered_agent = coerce_str(row.get("discovered_agent"))
    agent_label = author_agent or discovered_agent or (cohort if authored_by_agent else None)
    authorship_group = "agent" if authored_by_agent else "human"
    base_repository_id = coerce_str(repository.get("id"))
    repository_key = normalize_repository_key(repository.get("name_with_owner"))
    pr_number = coerce_str(row.get("number"))
    additions = coerce_int(row.get("additions"))
    deletions = coerce_int(row.get("deletions"))
    changed_line_count = (
        None
        if additions is None and deletions is None
        else int(additions or 0) + int(deletions or 0)
    )
    return StreamingPrFacts(
        cohort=str(cohort),
        source_file=str(source_file),
        source_row_number=int(source_row_number),
        pr_id=coerce_str(row.get("id")),
        pr_url=coerce_str(row.get("url")),
        pr_number=pr_number,
        created_at=coerce_str(row.get("created_at")),
        language=_normalize_language(row.get("pr_primary_language_effective")),
        authorship_group=authorship_group,
        agent_label=agent_label,
        longitudinal_selected=coerce_bool(row.get("longitudinal_selected"), default=False),
        base_repository_id=base_repository_id,
        repository_key=repository_key,
        stargazer_count=coerce_int(repository.get("stargazer_count")),
        changed_files_count=_changed_files_count(row),
        additions=additions,
        deletions=deletions,
        changed_line_count=changed_line_count,
        stable_pr_key=stable_pr_dedup_key(
            base_repository_id=base_repository_id,
            pull_request_number=pr_number,
        ),
    )


def nested_get(value: Any, *path: str, default: Any = None) -> Any:
    """Return a nested mapping value, tolerating missing or non-mapping nodes."""
    current = value
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current


def coerce_str(value: Any) -> str | None:
    """Return a stripped string, treating empty strings and NaN-ish values as null."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "nan"}:
        return None
    if text.endswith(".0") and text[:-2].isdecimal():
        return text[:-2]
    return text


def coerce_int(value: Any) -> int | None:
    """Return an integer where possible without treating booleans as numbers."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> float | None:
    """Return a finite float where possible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def coerce_bool(value: Any, *, default: bool | None = None) -> bool | None:
    """Return a boolean from common parquet scalar/string representations."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().casefold()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def json_mapping(value: Any) -> dict[str, Any]:
    """Return a dict from a dict/JSON string value; invalid values become empty."""
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _repository_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Read repository metadata from nested, JSON, or flattened parquet fields."""
    direct = json_mapping(row.get("repository_metadata"))
    if direct:
        return direct
    base = json_mapping(row.get("base_repository"))
    if base:
        return base
    full = json_mapping(row.get("base_repository_full"))
    if full:
        return full
    return {
        "id": row.get("repository_metadata_id") or row.get("base_repository_id"),
        "name_with_owner": (
            row.get("repository_metadata_name_with_owner")
            or row.get("repository_key")
        ),
        "stargazer_count": row.get("repository_metadata_stargazer_count")
        or row.get("stargazer_count"),
    }


def _changed_files_count(row: Mapping[str, Any]) -> int | None:
    """Return changed-file count across old and current curation schemas."""
    for candidate in (
        row.get("changed_files"),
        row.get("changed_files_count"),
        nested_get(row, "hydration", "diff_tracking", "pr", "changed_files_count"),
    ):
        parsed = coerce_int(candidate)
        if parsed is not None:
            return parsed
    return None


def _normalize_language(value: Any) -> str | None:
    """Normalize language labels for grouping while preserving missing values."""
    text = coerce_str(value)
    return text.casefold() if text else None


def _log(progress_logger: Callable[..., Any] | None, stage: str, **details: Any) -> None:
    """Emit a structured progress event when a logger is configured."""
    if progress_logger is None:
        return
    progress_logger(stage, **details)


def _available_projection_columns(
    schema: Any,
    columns: Sequence[str] | None,
) -> list[str]:
    """Return requested parquet projection columns present in this file schema."""
    if not columns:
        return []
    leaf_paths = set(_schema_leaf_paths(schema))
    available: list[str] = []
    for column in columns:
        column_path = str(column)
        prefix = f"{column_path}."
        if column_path in leaf_paths or any(path.startswith(prefix) for path in leaf_paths):
            available.append(column_path)
    return available


def _schema_leaf_paths(schema: Any, prefix: str = "") -> Iterator[str]:
    """Yield all available leaf and intermediate field paths in a schema."""
    for field in schema:
        yield from _field_paths(field, prefix=prefix)


def _field_paths(field: Any, *, prefix: str = "") -> Iterator[str]:
    """Yield available projection paths for a pyarrow field.

    ``pyarrow.parquet.ParquetFile.iter_batches(columns=...)`` accepts prefixes,
    so retaining intermediate paths lets callers request a whole struct/list
    while still filtering absent nested leaves safely.
    """
    name = str(field.name)
    path = f"{prefix}.{name}" if prefix else name
    yield path

    field_type = field.type
    num_fields = int(getattr(field_type, "num_fields", 0) or 0)
    if num_fields <= 0:
        return

    for index in range(num_fields):
        try:
            child = field_type.field(index)
        except (AttributeError, TypeError):
            return
        yield from _field_paths(child, prefix=path)
