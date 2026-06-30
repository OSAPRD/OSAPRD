"""Streaming repository input loader for topic-classification post-processing.

The classifier operates at repository grain, while curation outputs are stored
at PR grain. This loader builds lightweight repository references first and
only decodes file lists, metrics, and PR payloads when a repository is actually
classified.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

UTILITY_DIR = Path(__file__).resolve().parents[2] / "utility"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from cohort_discovery import discover_cohort_dirs
from json_io import iter_json_objects, iter_jsonl_offset_objects, read_json_object
from repository_keys import (
    normalize_repository_key,
    repository_numeric_id_from_payload,
    repository_key_from_full_name,
    repository_key_from_safe_key,
    safe_repository_key,
    stable_numeric_id,
)


LOG_PREFIX = "[post-processing/topic-classification]"


@dataclass(frozen=True)
class JsonlRecordRef:
    """Reference to one PR source record without retaining the decoded payload."""

    path: Path
    offset: int
    line_number: int
    source: str
    cohort: str
    repository_key: str
    repository_id: str | None
    pr_number: int | None
    pr_url: str | None
    record_format: str = "jsonl"


@dataclass(frozen=True)
class RepositoryInputRef:
    """Repository-level input references discovered from curation outputs."""

    cohort: str
    repository_owner: str
    repository_name: str
    repository_key: str
    repository_id: str | None
    repository_identity_key: str
    safe_repository_key: str
    file_list_path: Path | None
    cohorts: tuple[str, ...] = ()
    metrics_paths: tuple[Path, ...] = ()
    pr_record_refs: tuple[JsonlRecordRef, ...] = ()
    metadata_payloads: tuple[dict[str, Any], ...] = ()
    parquet_paths: tuple[Path, ...] = ()
    parquet_pr_count: int = 0
    pr_touched_files: tuple[str, ...] = ()
    has_file_list: bool = True
    input_format: str = "legacy-json"


@dataclass(frozen=True)
class RepositoryContext:
    """Loaded repository context plus lazy payload iterators for future classifiers."""

    ref: RepositoryInputRef
    file_list_metadata: dict[str, Any]
    repository_files: tuple[str, ...]

    def iter_pr_payloads(self) -> Iterator[dict[str, Any]]:
        jsonl_refs = tuple(
            ref for ref in self.ref.pr_record_refs if ref.record_format == "jsonl"
        )
        yield from iter_jsonl_payloads(jsonl_refs)

    def iter_metrics_payloads(self) -> Iterator[dict[str, Any]]:
        yield from iter_json_payloads(self.ref.metrics_paths)


@dataclass
class LoaderStats:
    """Counters describing input discovery, parsing, and repository filtering."""

    eligible_cohort_count: int = 0
    repository_count: int = 0
    metrics_path_count: int = 0
    pr_record_ref_count: int = 0
    pr_index_parse_failures: int = 0
    repositories_missing_metrics: int = 0
    repositories_missing_prs: int = 0
    file_list_parse_failures: int = 0
    metrics_parse_failures: int = 0
    pr_payload_parse_failures: int = 0
    repository_files_loaded: int = 0
    metrics_payloads_loaded: int = 0
    pr_payloads_loaded: int = 0
    repositories_filtered_by_longitudinal: int = 0
    repositories_deduplicated_by_identity: int = 0
    repositories_missing_file_lists: int = 0
    parquet_path_count: int = 0
    parquet_row_count: int = 0
    parquet_parse_failures: int = 0
    input_format: str = "legacy-json"
    cohort_repository_counts: dict[str, int] = field(default_factory=dict)

    @property
    def parse_failures(self) -> int:
        return (
            self.pr_index_parse_failures
            + self.file_list_parse_failures
            + self.metrics_parse_failures
            + self.pr_payload_parse_failures
            + self.parquet_parse_failures
        )


@dataclass(frozen=True)
class TopicInputIndex:
    """Lightweight repository index containing refs, not decoded repository data."""

    repository_refs: tuple[RepositoryInputRef, ...]
    eligible_cohort_count: int
    cohort_repository_counts: dict[str, int]
    metrics_path_count: int
    pr_record_ref_count: int
    pr_index_parse_failures: int
    repositories_filtered_by_longitudinal: int = 0
    repositories_deduplicated_by_identity: int = 0
    repositories_missing_file_lists: int = 0
    parquet_path_count: int = 0
    parquet_row_count: int = 0
    parquet_parse_failures: int = 0
    input_format: str = "legacy-json"

    def iter_repository_batches(self, batch_size: int) -> Iterator[tuple[RepositoryInputRef, ...]]:
        normalized_batch_size = max(1, int(batch_size))
        for index in range(0, len(self.repository_refs), normalized_batch_size):
            yield self.repository_refs[index : index + normalized_batch_size]


def log(message: str) -> None:
    """Print a topic-classification log line."""
    print(f"{LOG_PREFIX} {message}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _discover_metrics_paths(cohort_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    processed_data_dir = cohort_dir / "output" / "processed-data"
    if not processed_data_dir.exists():
        return grouped
    for metrics_dir in sorted(processed_data_dir.glob("*/metrics-json")):
        if not metrics_dir.is_dir():
            continue
        for repo_dir in sorted(path for path in metrics_dir.iterdir() if path.is_dir()):
            repository_key = repository_key_from_safe_key(repo_dir.name)
            if not repository_key:
                continue
            grouped[repository_key].extend(sorted(repo_dir.rglob("*.json")))
    return grouped


def _jsonl_source_name(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("sampled_prs_"):
        return "sampled"
    if name.startswith("longitudinal_prs_"):
        return "longitudinal"
    return "unknown"


def _pr_number_from_metrics_path(path: Path) -> int | None:
    match = re.match(r"^pr-(\d+)(?:__.*)?\.json$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _pr_number_from_pr_record(payload: dict[str, Any]) -> int | None:
    direct = _optional_int(payload.get("pr_number"))
    if direct is None:
        direct = _optional_int(payload.get("number"))
    if direct is not None:
        return direct
    original = payload.get("original_pr_payload")
    if isinstance(original, dict):
        return _optional_int(original.get("number"))
    return None


def _pr_url_from_pr_record(payload: dict[str, Any]) -> str | None:
    for key in ("pr_url", "url"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    original = payload.get("original_pr_payload")
    if isinstance(original, dict):
        value = original.get("url")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _coerce_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
    return None


def _repo_key_from_pr_record(payload: dict[str, Any]) -> str | None:
    direct = repository_key_from_full_name(payload.get("repo_full_name"))
    if direct:
        return direct

    for metadata_key in ("repository_metadata", "base_repository_full"):
        metadata = _coerce_json_object(payload.get(metadata_key))
        if not metadata:
            continue
        for key in ("name_with_owner", "full_name", "repository_full_name"):
            resolved = repository_key_from_full_name(metadata.get(key))
            if resolved:
                return resolved
        owner = metadata.get("owner")
        if isinstance(owner, dict):
            owner = owner.get("login") or owner.get("name")
        name = metadata.get("name")
        if owner and name:
            return normalize_repository_key(str(owner), str(name))
        url = str(metadata.get("url") or "").rstrip("/")
        if "github.com/" in url:
            owner_name = url.split("github.com/", 1)[1]
            resolved = repository_key_from_full_name(owner_name)
            if resolved:
                return resolved

    base_full = _coerce_json_object(payload.get("base_repository_full"))
    if isinstance(base_full, dict):
        for key in ("name_with_owner", "full_name", "repository_full_name"):
            resolved = repository_key_from_full_name(base_full.get(key))
            if resolved:
                return resolved
        owner = base_full.get("owner")
        if isinstance(owner, dict):
            owner = owner.get("login") or owner.get("name")
        name = base_full.get("name")
        if owner and name:
            return normalize_repository_key(str(owner), str(name))

    base_repository = _coerce_json_object(payload.get("base_repository"))
    if isinstance(base_repository, dict):
        resolved = repository_key_from_full_name(base_repository.get("name_with_owner"))
        if resolved:
            return resolved
        url = str(base_repository.get("url") or "").rstrip("/")
        if "github.com/" in url:
            owner_name = url.split("github.com/", 1)[1]
            resolved = repository_key_from_full_name(owner_name)
            if resolved:
                return resolved

    original = payload.get("original_pr_payload")
    if isinstance(original, dict):
        base_full = _coerce_json_object(original.get("base_repository_full"))
        if isinstance(base_full, dict):
            for key in ("name_with_owner", "full_name", "repository_full_name"):
                resolved = repository_key_from_full_name(base_full.get(key))
                if resolved:
                    return resolved
            owner = base_full.get("owner")
            if isinstance(owner, dict):
                owner = owner.get("login") or owner.get("name")
            name = base_full.get("name")
            if owner and name:
                return normalize_repository_key(str(owner), str(name))
    return None


def _repo_id_from_pr_record(payload: dict[str, Any]) -> str | None:
    direct = repository_numeric_id_from_payload(
        {
            "id": payload.get("base_repository_id"),
            "database_id": payload.get("base_repository_database_id"),
            "databaseId": payload.get("base_repository_databaseId"),
            "repository_id": payload.get("repository_id"),
        }
    )
    if direct:
        return direct

    for candidate in (
        _coerce_json_object(payload.get("base_repository_full")),
        _coerce_json_object(payload.get("base_repository")),
        _coerce_json_object(payload.get("repository_metadata")),
    ):
        if isinstance(candidate, dict):
            resolved = repository_numeric_id_from_payload(candidate)
            if resolved:
                return resolved

    original = payload.get("original_pr_payload")
    if isinstance(original, dict):
        original_direct = repository_numeric_id_from_payload(
            {
                "id": original.get("base_repository_id"),
                "database_id": original.get("base_repository_database_id"),
                "databaseId": original.get("base_repository_databaseId"),
                "repository_id": original.get("repository_id"),
            }
        )
        if original_direct:
            return original_direct
        for candidate in (
            _coerce_json_object(original.get("base_repository_full")),
            _coerce_json_object(original.get("base_repository")),
            _coerce_json_object(original.get("repository_metadata")),
        ):
            if isinstance(candidate, dict):
                resolved = repository_numeric_id_from_payload(candidate)
                if resolved:
                    return resolved
    return None


def _repository_id_from_metrics_payload(payload: dict[str, Any]) -> str | None:
    for candidate in (
        payload.get("repository_metadata"),
        payload.get("base_repository_full"),
        payload.get("base_repository"),
    ):
        if isinstance(candidate, dict):
            resolved = repository_numeric_id_from_payload(candidate)
            if resolved:
                return resolved

    pr_payload = payload.get("pr")
    if isinstance(pr_payload, dict):
        for candidate in (
            pr_payload.get("base_repository_full"),
            pr_payload.get("base_repository"),
            pr_payload.get("repository_metadata"),
        ):
            if isinstance(candidate, dict):
                resolved = repository_numeric_id_from_payload(candidate)
                if resolved:
                    return resolved
        resolved = _repo_id_from_pr_record(pr_payload)
        if resolved:
            return resolved

    return repository_numeric_id_from_payload(payload)


def _repository_id_from_metrics_paths(paths: Iterable[Path]) -> str | None:
    for path in sorted(paths, key=lambda value: str(value)):
        payload = read_json_object(path, description="metrics JSON", log=log)
        if payload is None:
            continue
        resolved = _repository_id_from_metrics_payload(payload)
        if resolved:
            return resolved
    return None


def _repository_identity_key(repository_id: str | None, repository_key: str) -> str:
    stable_id = stable_numeric_id(repository_id)
    if stable_id:
        return f"repository-id:{stable_id}"
    return f"repository-key:{repository_key}"


def _merge_repository_ref(
    existing: RepositoryInputRef,
    candidate: RepositoryInputRef,
) -> RepositoryInputRef:
    metrics_paths = tuple(
        sorted(
            {*(existing.metrics_paths), *(candidate.metrics_paths)},
            key=lambda path: str(path),
        )
    )
    pr_record_refs = tuple(
        sorted(
            {*(existing.pr_record_refs), *(candidate.pr_record_refs)},
            key=lambda ref: (str(ref.path), ref.offset, ref.line_number),
        )
    )
    cohorts = tuple(sorted({*existing.cohorts, *candidate.cohorts}))
    repository_id = existing.repository_id or candidate.repository_id
    metadata_payloads = _merge_metadata_payloads(
        existing.metadata_payloads,
        candidate.metadata_payloads,
    )
    parquet_paths = tuple(
        sorted(
            {*(existing.parquet_paths), *(candidate.parquet_paths)},
            key=lambda path: str(path),
        )
    )
    pr_touched_files = tuple(
        sorted(
            {*(existing.pr_touched_files), *(candidate.pr_touched_files)},
            key=lambda path: (path.lower(), path),
        )
    )
    file_list_path = existing.file_list_path or candidate.file_list_path
    return RepositoryInputRef(
        cohort=existing.cohort,
        repository_owner=existing.repository_owner,
        repository_name=existing.repository_name,
        repository_key=existing.repository_key,
        repository_id=repository_id,
        repository_identity_key=existing.repository_identity_key,
        safe_repository_key=existing.safe_repository_key,
        file_list_path=file_list_path,
        cohorts=cohorts,
        metrics_paths=metrics_paths,
        pr_record_refs=pr_record_refs,
        metadata_payloads=metadata_payloads,
        parquet_paths=parquet_paths,
        parquet_pr_count=existing.parquet_pr_count + candidate.parquet_pr_count,
        pr_touched_files=pr_touched_files,
        has_file_list=existing.has_file_list or candidate.has_file_list,
        input_format=existing.input_format,
    )


def _merge_metadata_payloads(
    first: tuple[dict[str, Any], ...],
    second: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in (*first, *second):
        key = json.dumps(payload, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        merged.append(payload)
    return tuple(merged)


@dataclass(frozen=True)
class PrJsonlIndex:
    """Index from repository keys to JSONL PR record references."""

    refs_by_repo: dict[str, list[JsonlRecordRef]]
    repository_ids_by_repo: dict[str, set[str]]
    longitudinal_pr_numbers_by_repo: dict[str, set[int]]
    longitudinal_pr_urls_by_repo: dict[str, set[str]]
    parse_failures: int


@dataclass(frozen=True)
class CohortPayloadDir:
    """Resolved curation payload directory for one logical cohort."""

    cohort: str
    path: Path


@dataclass
class _ParquetRepositoryAccumulator:
    cohort: str
    repository_owner: str
    repository_name: str
    repository_key: str
    repository_id: str | None
    repository_identity_key: str
    safe_repository_key: str
    file_list_path: Path | None
    has_file_list: bool
    cohorts: set[str] = field(default_factory=set)
    pr_record_refs: list[JsonlRecordRef] = field(default_factory=list)
    metadata_payloads: list[dict[str, Any]] = field(default_factory=list)
    metadata_keys: set[str] = field(default_factory=set)
    parquet_paths: set[Path] = field(default_factory=set)
    pr_touched_files: set[str] = field(default_factory=set)
    parquet_pr_count: int = 0

    def add(
        self,
        *,
        cohort: str,
        pr_ref: JsonlRecordRef,
        metadata_payloads: tuple[dict[str, Any], ...],
        parquet_path: Path | None,
        touched_files: tuple[str, ...],
        file_list_path: Path | None,
    ) -> None:
        self.cohorts.add(cohort)
        self.pr_record_refs.append(pr_ref)
        self.parquet_pr_count += 1
        if parquet_path is not None:
            self.parquet_paths.add(parquet_path)
        if file_list_path is not None and self.file_list_path is None:
            self.file_list_path = file_list_path
            self.has_file_list = True
            self.pr_touched_files.clear()
        if self.file_list_path is None:
            self.pr_touched_files.update(touched_files)
        for payload in metadata_payloads:
            key = json.dumps(payload, sort_keys=True, default=str)
            if key in self.metadata_keys:
                continue
            self.metadata_keys.add(key)
            self.metadata_payloads.append(payload)

    def to_ref(self) -> RepositoryInputRef:
        return RepositoryInputRef(
            cohort=self.cohort,
            repository_owner=self.repository_owner,
            repository_name=self.repository_name,
            repository_key=self.repository_key,
            repository_id=self.repository_id,
            repository_identity_key=self.repository_identity_key,
            safe_repository_key=self.safe_repository_key,
            file_list_path=self.file_list_path,
            cohorts=tuple(sorted(self.cohorts)),
            pr_record_refs=tuple(
                sorted(
                    self.pr_record_refs,
                    key=lambda ref: (str(ref.path), ref.line_number, ref.pr_number or 0),
                )
            ),
            metadata_payloads=tuple(self.metadata_payloads),
            parquet_paths=tuple(sorted(self.parquet_paths, key=lambda path: str(path))),
            parquet_pr_count=self.parquet_pr_count,
            pr_touched_files=tuple(
                sorted(self.pr_touched_files, key=lambda path: (path.lower(), path))
            ),
            has_file_list=self.has_file_list,
            input_format="parquet",
        )


def _has_curation_payload_markers(path: Path) -> bool:
    return (
        (path / "output" / "snapshots").exists()
        or (path / "output" / "processed-data").exists()
        or any(path.glob("sampled_prs_*.jsonl"))
        or any(path.glob("longitudinal_prs_*.jsonl"))
    )


def _infer_cohort_name(payload_dir: Path, run_dir: Path) -> str:
    for pattern in ("sampled_prs_*.jsonl", "longitudinal_prs_*.jsonl"):
        for path in sorted(payload_dir.glob(pattern)):
            match = re.match(r"^(?:sampled|longitudinal)_prs_(.+?)\.jsonl$", path.name)
            if match:
                cohort = match.group(1)
                if cohort.endswith("_store"):
                    cohort = cohort[: -len("_store")]
                if cohort:
                    return cohort

    run_name = run_dir.name
    timestamp_match = re.match(r"^(.+?)_\d{8}T\d{6}Z$", run_name)
    if timestamp_match:
        return timestamp_match.group(1)
    return run_name


def _resolve_cohort_payload_dirs(cohort_dir: Path) -> tuple[CohortPayloadDir, ...]:
    """Return payload dirs for both direct and archived curation-output layouts.

    Supported layouts:
    - ``<root>/<cohort>/output/...``
    - ``<root>/<run>/outputs/<cohort>/output/...``
    - ``<root>/<run>/output/output/...``
    """
    if _has_curation_payload_markers(cohort_dir):
        return (CohortPayloadDir(cohort=cohort_dir.name, path=cohort_dir),)

    single_output_dir = cohort_dir / "output"
    if single_output_dir.exists() and _has_curation_payload_markers(single_output_dir):
        return (
            CohortPayloadDir(
                cohort=_infer_cohort_name(single_output_dir, cohort_dir),
                path=single_output_dir,
            ),
        )

    outputs_dir = cohort_dir / "outputs"
    if not outputs_dir.exists():
        return ()
    payload_dirs = [
        CohortPayloadDir(cohort=path.name, path=path)
        for path in sorted(outputs_dir.iterdir())
        if path.is_dir() and _has_curation_payload_markers(path)
    ]
    return tuple(payload_dirs)


def _index_pr_jsonl_records(cohort: str, cohort_dir: Path) -> PrJsonlIndex:
    grouped: dict[str, list[JsonlRecordRef]] = defaultdict(list)
    repository_ids_by_repo: dict[str, set[str]] = defaultdict(set)
    longitudinal_pr_numbers_by_repo: dict[str, set[int]] = defaultdict(set)
    longitudinal_pr_urls_by_repo: dict[str, set[str]] = defaultdict(set)
    parse_failures = 0
    sampled_paths = [
        path
        for path in sorted(cohort_dir.glob("sampled_prs_*.jsonl"))
        if not path.name.endswith("_store.jsonl")
    ]
    jsonl_paths = sampled_paths + sorted(cohort_dir.glob("longitudinal_prs_*.jsonl"))
    for path in jsonl_paths:
        source = _jsonl_source_name(path)
        try:
            with path.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    line_number += 1
                    if not raw_line.strip():
                        continue
                    try:
                        payload = json.loads(raw_line.decode("utf-8"))
                    except Exception as exc:
                        parse_failures += 1
                        log(f"Skipping malformed PR JSONL record {path}:{line_number}: {exc}")
                        continue
                    if not isinstance(payload, dict):
                        parse_failures += 1
                        log(f"Skipping non-object PR JSONL record {path}:{line_number}")
                        continue
                    repository_key = _repo_key_from_pr_record(payload)
                    if not repository_key:
                        parse_failures += 1
                        log(f"Skipping PR JSONL record without repository key {path}:{line_number}")
                        continue
                    repository_id = _repo_id_from_pr_record(payload)
                    pr_number = _pr_number_from_pr_record(payload)
                    pr_url = _pr_url_from_pr_record(payload)
                    grouped[repository_key].append(
                        JsonlRecordRef(
                            path=path,
                            offset=offset,
                            line_number=line_number,
                            source=source,
                            cohort=cohort,
                            repository_key=repository_key,
                            repository_id=repository_id,
                            pr_number=pr_number,
                            pr_url=pr_url,
                        )
                    )
                    if repository_id:
                        repository_ids_by_repo[repository_key].add(repository_id)
                    if source == "longitudinal":
                        if pr_number is not None:
                            longitudinal_pr_numbers_by_repo[repository_key].add(pr_number)
                        if pr_url:
                            longitudinal_pr_urls_by_repo[repository_key].add(pr_url)
        except OSError as exc:
            parse_failures += 1
            log(f"Unable to read PR JSONL file {path}: {exc}")
    return PrJsonlIndex(
        refs_by_repo=grouped,
        repository_ids_by_repo=repository_ids_by_repo,
        longitudinal_pr_numbers_by_repo=longitudinal_pr_numbers_by_repo,
        longitudinal_pr_urls_by_repo=longitudinal_pr_urls_by_repo,
        parse_failures=parse_failures,
    )


def _choose_repository_id(
    repository_key: str,
    *,
    pr_index: PrJsonlIndex,
    metrics_paths: Iterable[Path],
) -> str | None:
    pr_repository_ids = sorted(
        pr_index.repository_ids_by_repo.get(repository_key, set()),
        key=lambda value: int(value),
    )
    if len(pr_repository_ids) == 1:
        return pr_repository_ids[0]
    if len(pr_repository_ids) > 1:
        log(
            "Multiple stable repository IDs found in PR JSONL for "
            f"{repository_key}; using {pr_repository_ids[0]} from {pr_repository_ids}."
        )
        return pr_repository_ids[0]
    return _repository_id_from_metrics_paths(metrics_paths)


def _filter_pr_refs_to_longitudinal(
    refs: Iterable[JsonlRecordRef],
    *,
    longitudinal_numbers: set[int],
    longitudinal_urls: set[str],
) -> tuple[JsonlRecordRef, ...]:
    return tuple(
        ref
        for ref in refs
        if ref.source == "longitudinal"
        or (ref.pr_number is not None and ref.pr_number in longitudinal_numbers)
        or (ref.pr_url is not None and ref.pr_url in longitudinal_urls)
    )


def _filter_metrics_paths_to_longitudinal(
    paths: Iterable[Path],
    *,
    longitudinal_numbers: set[int],
) -> tuple[Path, ...]:
    if not longitudinal_numbers:
        return ()
    return tuple(
        path
        for path in paths
        if (pr_number := _pr_number_from_metrics_path(path)) is not None
        and pr_number in longitudinal_numbers
    )


def _iter_repository_file_list_paths(cohort_dir: Path) -> Iterator[Path]:
    snapshots_dir = cohort_dir / "output" / "snapshots"
    if not snapshots_dir.exists():
        return
    yield from sorted(snapshots_dir.glob("*/*/repository_file_list.json"))


def _discover_processed_pr_parquet_paths(cohort_dir: Path) -> tuple[Path, ...]:
    processed_data_dir = cohort_dir / "output" / "processed-data"
    if not processed_data_dir.exists():
        return ()
    return tuple(sorted(processed_data_dir.glob("*/processed_pr_batch-*.parquet")))


def _index_repository_file_lists(cohort_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for file_list_path in _iter_repository_file_list_paths(cohort_dir):
        try:
            relative_parts = file_list_path.relative_to(
                cohort_dir / "output" / "snapshots"
            ).parts
        except ValueError:
            continue
        if len(relative_parts) < 3:
            continue
        repository_key = normalize_repository_key(relative_parts[0], relative_parts[1])
        indexed.setdefault(repository_key, file_list_path)
    return indexed


def _owner_name_from_repository_key(repository_key: str) -> tuple[str, str]:
    if "/" not in repository_key:
        return repository_key, ""
    owner, name = repository_key.split("/", 1)
    return owner, name


def _owner_name_from_metadata(
    repository_key: str,
    metadata_candidates: Iterable[dict[str, Any]],
) -> tuple[str, str]:
    for candidate in metadata_candidates:
        full_name = (
            candidate.get("name_with_owner")
            or candidate.get("full_name")
            or candidate.get("repository_full_name")
        )
        if full_name and "/" in str(full_name):
            owner, name = str(full_name).split("/", 1)
            if owner.strip() and name.strip():
                return owner.strip(), name.strip()
        owner = candidate.get("owner")
        if isinstance(owner, dict):
            owner = owner.get("login") or owner.get("name")
        name = candidate.get("name")
        if owner and name:
            return str(owner).strip(), str(name).strip()
    return _owner_name_from_repository_key(repository_key)


def _pr_touched_files_from_json(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        return ()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, list):
        return ()
    paths: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("filename") or item.get("new_path")
        if path is not None and str(path).strip():
            paths.append(str(path).strip())
    return paper_order_repository_files(paths)


def _metadata_payloads_from_parquet_row(
    repository_metadata: Any,
    base_repository_full: Any,
) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for value in (repository_metadata, base_repository_full):
        payload = _coerce_json_object(value)
        if payload:
            payloads.append(payload)
    return _merge_metadata_payloads((), tuple(payloads))


def _iter_parquet_pr_rows(
    paths: tuple[Path, ...],
    *,
    include_files: bool = False,
    filter_repository_ids: tuple[str, ...] = (),
    filter_repository_keys: tuple[str, ...] = (),
) -> Iterator[dict[str, Any]]:
    if not paths:
        return
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required for parquet topic-classification input. "
            "Install duckdb or set POST_PROCESSING_TOPIC_CLASSIFICATION_INPUT_FORMAT=legacy-json."
        ) from exc

    con = duckdb.connect(database=":memory:")
    try:
        selected_files_column = ",\n                files" if include_files else ""
        where_clause = ""
        parameters: list[Any] = [[str(path) for path in paths]]
        normalized_filter_repository_ids = tuple(
            str(value).strip() for value in filter_repository_ids if str(value).strip()
        )
        normalized_filter_repository_keys = tuple(
            str(value).strip().lower()
            for value in filter_repository_keys
            if str(value).strip()
        )
        if normalized_filter_repository_ids or normalized_filter_repository_keys:
            where_terms: list[str] = []
            if normalized_filter_repository_ids:
                where_terms.extend(
                    (
                        "repository_metadata.id IN ?",
                        "json_extract_string(base_repository_full, '$.id') IN ?",
                    )
                )
                parameters.extend(
                    (
                        list(normalized_filter_repository_ids),
                        list(normalized_filter_repository_ids),
                    )
                )
            if normalized_filter_repository_keys:
                where_terms.extend(
                    (
                        "lower(repository_metadata.name_with_owner) IN ?",
                        "lower(json_extract_string(base_repository_full, '$.name_with_owner')) IN ?",
                        "lower(json_extract_string(base_repository_full, '$.full_name')) IN ?",
                        "lower(json_extract_string(base_repository_full, '$.nameWithOwner')) IN ?",
                    )
                )
                parameters.extend(
                    (
                        list(normalized_filter_repository_keys),
                        list(normalized_filter_repository_keys),
                        list(normalized_filter_repository_keys),
                        list(normalized_filter_repository_keys),
                    )
                )
            where_clause = f"WHERE {' OR '.join(where_terms)}"
        query = """
            SELECT
                filename,
                row_number() OVER (PARTITION BY filename) AS parquet_row_number,
                number,
                url,
                longitudinal_selected,
                repository_metadata,
                base_repository_full,
                metrics_aggregate_path
                {selected_files_column}
            FROM read_parquet(?, filename=true, union_by_name=true)
            {where_clause}
        """.format(
            selected_files_column=selected_files_column,
            where_clause=where_clause,
        )
        cursor = con.execute(query, parameters)
        columns = [description[0] for description in cursor.description]
        while True:
            batch = cursor.fetchmany(4096)
            if not batch:
                break
            for row in batch:
                yield dict(zip(columns, row))
    finally:
        con.close()


def paper_order_repository_files(paths: Iterable[Any]) -> tuple[str, ...]:
    """Return repository paths in the paper's root-first traversal order.

    Existing curation outputs store files lexicographically. The paper processes
    files from the repository root, then one directory level deeper at a time.
    The original filesystem traversal order is not recoverable after curation,
    so this uses a stable breadth-first approximation: depth first, then path.
    """
    normalized_paths: list[str] = []
    for raw_path in paths:
        normalized = str(raw_path or "").replace("\\", "/").lstrip("/")
        parts = [part for part in normalized.split("/") if part]
        if parts:
            normalized_paths.append("/".join(parts))
    return tuple(
        sorted(
            normalized_paths,
            key=lambda path: (
                len([part for part in path.split("/") if part]),
                path.lower(),
                path,
            ),
        )
    )


class TopicClassificationInputLoader:
    """Discover and stream repository-level curation inputs."""

    def __init__(
        self,
        curation_outputs_dir: Path,
        *,
        exclude_dirs: Iterable[str] | None = None,
        longitudinal_only: bool = False,
        input_format: str = "parquet",
    ):
        self.curation_outputs_dir = Path(curation_outputs_dir)
        self.exclude_dirs = tuple(exclude_dirs or ())
        self.longitudinal_only = bool(longitudinal_only)
        normalized_input_format = str(input_format or "parquet").strip().lower()
        if normalized_input_format not in {"parquet", "legacy-json"}:
            raise ValueError(
                "Topic classification input format must be 'parquet' or 'legacy-json', "
                f"got {input_format!r}."
            )
        self.input_format = normalized_input_format

    def build_index(self) -> TopicInputIndex:
        if self.input_format == "legacy-json":
            return self._build_legacy_json_index()
        return self._build_parquet_index()

    def _build_legacy_json_index(self) -> TopicInputIndex:
        cohort_dirs = discover_cohort_dirs(
            self.curation_outputs_dir,
            self.exclude_dirs,
            log=log,
        )
        excluded_display = ", ".join(self.exclude_dirs) if self.exclude_dirs else "(none)"
        log(
            "Scanning "
            f"{self.curation_outputs_dir}/<cohort> and "
            f"{self.curation_outputs_dir}/<run>/outputs/<cohort> and "
            f"{self.curation_outputs_dir}/<run>/output for repository "
            f"file lists, PR JSONL, and metrics JSON files "
            f"(excluded={excluded_display}, longitudinal_only={self.longitudinal_only})."
        )

        repository_refs_by_identity: dict[str, RepositoryInputRef] = {}
        cohort_repository_counts: dict[str, int] = {}
        pr_index_parse_failures = 0
        repositories_filtered_by_longitudinal = 0
        repositories_deduplicated_by_identity = 0

        cohort_payload_dirs: list[CohortPayloadDir] = []
        for cohort_dir in cohort_dirs:
            resolved_payload_dirs = _resolve_cohort_payload_dirs(cohort_dir)
            if not resolved_payload_dirs:
                log(f"Skipping curation output without payload markers: {cohort_dir}")
            cohort_payload_dirs.extend(resolved_payload_dirs)

        for payload_dir in cohort_payload_dirs:
            cohort = payload_dir.cohort
            cohort_dir = payload_dir.path
            metrics_by_repo = _discover_metrics_paths(cohort_dir)
            pr_index = _index_pr_jsonl_records(cohort, cohort_dir)
            pr_index_parse_failures += pr_index.parse_failures

            for file_list_path in _iter_repository_file_list_paths(cohort_dir):
                try:
                    relative_parts = file_list_path.relative_to(
                        cohort_dir / "output" / "snapshots"
                    ).parts
                except ValueError:
                    continue
                if len(relative_parts) < 3:
                    continue
                owner, name = relative_parts[0], relative_parts[1]
                repository_key = normalize_repository_key(owner, name)
                pr_refs = tuple(pr_index.refs_by_repo.get(repository_key, ()))
                metrics_paths = tuple(metrics_by_repo.get(repository_key, ()))
                if self.longitudinal_only:
                    longitudinal_numbers = pr_index.longitudinal_pr_numbers_by_repo.get(
                        repository_key, set()
                    )
                    longitudinal_urls = pr_index.longitudinal_pr_urls_by_repo.get(
                        repository_key, set()
                    )
                    pr_refs = _filter_pr_refs_to_longitudinal(
                        pr_refs,
                        longitudinal_numbers=longitudinal_numbers,
                        longitudinal_urls=longitudinal_urls,
                    )
                    metrics_paths = _filter_metrics_paths_to_longitudinal(
                        metrics_paths,
                        longitudinal_numbers=longitudinal_numbers,
                    )
                    if not pr_refs:
                        repositories_filtered_by_longitudinal += 1
                        continue
                repository_id = _choose_repository_id(
                    repository_key,
                    pr_index=pr_index,
                    metrics_paths=metrics_paths,
                )
                repository_identity_key = _repository_identity_key(
                    repository_id,
                    repository_key,
                )
                candidate_ref = RepositoryInputRef(
                    cohort=cohort,
                    repository_owner=owner,
                    repository_name=name,
                    repository_key=repository_key,
                    repository_id=repository_id,
                    repository_identity_key=repository_identity_key,
                    safe_repository_key=safe_repository_key(owner, name),
                    file_list_path=file_list_path,
                    cohorts=(cohort,),
                    metrics_paths=metrics_paths,
                    pr_record_refs=pr_refs,
                )
                existing_ref = repository_refs_by_identity.get(repository_identity_key)
                if existing_ref is None:
                    repository_refs_by_identity[repository_identity_key] = candidate_ref
                else:
                    repository_refs_by_identity[repository_identity_key] = _merge_repository_ref(
                        existing_ref,
                        candidate_ref,
                    )
                    repositories_deduplicated_by_identity += 1
                cohort_repository_counts[cohort] = cohort_repository_counts.get(cohort, 0) + 1

        repository_refs = tuple(
            sorted(
                repository_refs_by_identity.values(),
                key=lambda ref: (ref.repository_identity_key, ref.cohort, ref.repository_key),
            )
        )
        metrics_path_count = sum(len(ref.metrics_paths) for ref in repository_refs)
        pr_record_ref_count = sum(len(ref.pr_record_refs) for ref in repository_refs)
        log(
            "Discovered "
            f"{len(repository_refs)} repositories, {metrics_path_count} metrics JSON paths, "
            f"{pr_record_ref_count} PR JSONL record refs "
            f"(repositories_filtered_by_longitudinal={repositories_filtered_by_longitudinal}, "
            f"repositories_deduplicated_by_identity={repositories_deduplicated_by_identity})."
        )
        return TopicInputIndex(
            repository_refs=repository_refs,
            eligible_cohort_count=len(cohort_payload_dirs),
            cohort_repository_counts=dict(sorted(cohort_repository_counts.items())),
            metrics_path_count=metrics_path_count,
            pr_record_ref_count=pr_record_ref_count,
            pr_index_parse_failures=pr_index_parse_failures,
            repositories_filtered_by_longitudinal=repositories_filtered_by_longitudinal,
            repositories_deduplicated_by_identity=repositories_deduplicated_by_identity,
            input_format="legacy-json",
        )

    def _build_parquet_index(self) -> TopicInputIndex:
        cohort_dirs = discover_cohort_dirs(
            self.curation_outputs_dir,
            self.exclude_dirs,
            log=log,
        )
        excluded_display = ", ".join(self.exclude_dirs) if self.exclude_dirs else "(none)"
        log(
            "Scanning "
            f"{self.curation_outputs_dir}/<cohort>/output/processed-data/*/"
            "processed_pr_batch-*.parquet and "
            f"{self.curation_outputs_dir}/<cohort>/output/snapshots/*/*/"
            "repository_file_list.json "
            f"(excluded={excluded_display}, longitudinal_only={self.longitudinal_only})."
        )

        accumulators_by_identity: dict[str, _ParquetRepositoryAccumulator] = {}
        cohort_identity_keys: dict[str, set[str]] = defaultdict(set)
        parquet_path_count = 0
        parquet_row_count = 0
        parquet_parse_failures = 0
        repositories_deduplicated_by_identity = 0
        all_identity_keys_seen: set[str] = set()
        retained_identity_keys: set[str] = set()
        eligible_cohort_count = 0

        for cohort_dir in cohort_dirs:
            cohort = cohort_dir.name
            parquet_paths = _discover_processed_pr_parquet_paths(cohort_dir)
            if not parquet_paths:
                log(f"Skipping cohort without processed PR parquet batches: {cohort_dir}")
                continue
            eligible_cohort_count += 1
            parquet_path_count += len(parquet_paths)
            file_lists_by_repo = _index_repository_file_lists(cohort_dir)
            cohort_missing_file_list_identity_keys: set[str] = set()

            try:
                row_iter = _iter_parquet_pr_rows(parquet_paths)
                for row in row_iter:
                    parquet_row_count += 1
                    repository_metadata = _coerce_json_object(row.get("repository_metadata"))
                    base_repository_full = _coerce_json_object(row.get("base_repository_full"))
                    metadata_payloads = _metadata_payloads_from_parquet_row(
                        repository_metadata,
                        base_repository_full,
                    )
                    payload = {
                        "repository_metadata": repository_metadata,
                        "base_repository_full": base_repository_full,
                        "number": row.get("number"),
                        "url": row.get("url"),
                    }
                    repository_key = _repo_key_from_pr_record(payload)
                    if not repository_key:
                        parquet_parse_failures += 1
                        log(
                            "Skipping parquet PR row without repository key "
                            f"{row.get('filename')}:{row.get('parquet_row_number')}"
                        )
                        continue
                    repository_id = _repo_id_from_pr_record(payload)
                    repository_identity_key = _repository_identity_key(
                        repository_id,
                        repository_key,
                    )
                    all_identity_keys_seen.add(repository_identity_key)
                    if self.longitudinal_only and not bool(row.get("longitudinal_selected")):
                        continue
                    retained_identity_keys.add(repository_identity_key)
                    cohort_identity_keys[cohort].add(repository_identity_key)

                    file_list_path = file_lists_by_repo.get(repository_key)
                    if file_list_path is None:
                        cohort_missing_file_list_identity_keys.add(repository_identity_key)
                    filename = str(row.get("filename") or "")
                    parquet_path = Path(filename) if filename else None
                    pr_ref = JsonlRecordRef(
                        path=parquet_path or Path(""),
                        offset=0,
                        line_number=_optional_int(row.get("parquet_row_number")) or 0,
                        source="parquet",
                        cohort=cohort,
                        repository_key=repository_key,
                        repository_id=repository_id,
                        pr_number=_optional_int(row.get("number")),
                        pr_url=str(row.get("url")).strip() if row.get("url") else None,
                        record_format="parquet",
                    )
                    accumulator = accumulators_by_identity.get(repository_identity_key)
                    if accumulator is None:
                        owner, name = _owner_name_from_metadata(
                            repository_key,
                            metadata_payloads,
                        )
                        accumulator = _ParquetRepositoryAccumulator(
                            cohort=cohort,
                            repository_owner=owner,
                            repository_name=name,
                            repository_key=repository_key,
                            repository_id=repository_id,
                            repository_identity_key=repository_identity_key,
                            safe_repository_key=safe_repository_key(owner, name),
                            file_list_path=file_list_path,
                            has_file_list=file_list_path is not None,
                        )
                        accumulators_by_identity[repository_identity_key] = accumulator
                    else:
                        repositories_deduplicated_by_identity += 1
                    accumulator.add(
                        cohort=cohort,
                        pr_ref=pr_ref,
                        metadata_payloads=metadata_payloads,
                        parquet_path=parquet_path,
                        touched_files=(),
                        file_list_path=file_list_path,
                    )
            except Exception as exc:
                parquet_parse_failures += 1
                log(f"Unable to read processed PR parquet batches for {cohort_dir}: {exc}")

            if cohort_missing_file_list_identity_keys:
                fallback_repository_ids = tuple(
                    identity_key.removeprefix("repository-id:")
                    for identity_key in sorted(cohort_missing_file_list_identity_keys)
                    if identity_key.startswith("repository-id:")
                )
                fallback_repository_keys = tuple(
                    accumulator.repository_key
                    for identity_key in sorted(cohort_missing_file_list_identity_keys)
                    if (accumulator := accumulators_by_identity.get(identity_key)) is not None
                )
                try:
                    row_iter = _iter_parquet_pr_rows(
                        parquet_paths,
                        include_files=True,
                        filter_repository_ids=fallback_repository_ids,
                        filter_repository_keys=fallback_repository_keys,
                    )
                    for row in row_iter:
                        repository_metadata = _coerce_json_object(row.get("repository_metadata"))
                        base_repository_full = _coerce_json_object(row.get("base_repository_full"))
                        payload = {
                            "repository_metadata": repository_metadata,
                            "base_repository_full": base_repository_full,
                        }
                        repository_key = _repo_key_from_pr_record(payload)
                        if not repository_key:
                            continue
                        repository_id = _repo_id_from_pr_record(payload)
                        repository_identity_key = _repository_identity_key(
                            repository_id,
                            repository_key,
                        )
                        if repository_identity_key not in cohort_missing_file_list_identity_keys:
                            continue
                        accumulator = accumulators_by_identity.get(repository_identity_key)
                        if accumulator is None or accumulator.has_file_list:
                            continue
                        accumulator.pr_touched_files.update(
                            _pr_touched_files_from_json(row.get("files"))
                        )
                except Exception as exc:
                    parquet_parse_failures += 1
                    log(
                        "Unable to read PR-touched file fallback from parquet batches "
                        f"for {cohort_dir}: {exc}"
                    )

        repository_refs = tuple(
            sorted(
                (accumulator.to_ref() for accumulator in accumulators_by_identity.values()),
                key=lambda ref: (ref.repository_identity_key, ref.cohort, ref.repository_key),
            )
        )
        repositories_missing_file_lists = sum(1 for ref in repository_refs if not ref.has_file_list)
        repositories_filtered_by_longitudinal = (
            len(all_identity_keys_seen - retained_identity_keys) if self.longitudinal_only else 0
        )
        pr_record_ref_count = sum(len(ref.pr_record_refs) for ref in repository_refs)
        cohort_repository_counts = {
            cohort: len(identity_keys)
            for cohort, identity_keys in sorted(cohort_identity_keys.items())
        }
        log(
            "Discovered "
            f"{len(repository_refs)} repositories, {parquet_path_count} parquet paths, "
            f"{parquet_row_count} parquet rows, {pr_record_ref_count} PR refs "
            f"(repositories_filtered_by_longitudinal={repositories_filtered_by_longitudinal}, "
            f"repositories_deduplicated_by_identity={repositories_deduplicated_by_identity}, "
            f"repositories_missing_file_lists={repositories_missing_file_lists})."
        )
        return TopicInputIndex(
            repository_refs=repository_refs,
            eligible_cohort_count=eligible_cohort_count,
            cohort_repository_counts=cohort_repository_counts,
            metrics_path_count=0,
            pr_record_ref_count=pr_record_ref_count,
            pr_index_parse_failures=0,
            repositories_filtered_by_longitudinal=repositories_filtered_by_longitudinal,
            repositories_deduplicated_by_identity=repositories_deduplicated_by_identity,
            repositories_missing_file_lists=repositories_missing_file_lists,
            parquet_path_count=parquet_path_count,
            parquet_row_count=parquet_row_count,
            parquet_parse_failures=parquet_parse_failures,
            input_format="parquet",
        )

    def load_repository_context(self, ref: RepositoryInputRef) -> RepositoryContext:
        """Load one repository's file-list metadata; PR/metrics remain lazy."""
        payload: dict[str, Any] = {}
        if ref.file_list_path is not None:
            loaded_payload = read_json_object(
                ref.file_list_path,
                description="repository file list JSON",
                log=log,
            )
            if loaded_payload is None:
                raise ValueError(f"repository file list is not a JSON object: {ref.file_list_path}")
            payload = loaded_payload
        files = payload.get("files")
        if isinstance(files, list):
            repository_files = paper_order_repository_files(files)
        else:
            repository_files = paper_order_repository_files(ref.pr_touched_files)
        return RepositoryContext(
            ref=ref,
            file_list_metadata=payload,
            repository_files=repository_files,
        )


def iter_json_payloads(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield JSON object payloads one at a time from file paths."""
    yield from iter_json_objects(paths, description="metrics JSON", log=log)


def iter_jsonl_payloads(record_refs: Iterable[JsonlRecordRef]) -> Iterator[dict[str, Any]]:
    """Yield JSONL payloads from stored byte offsets one record at a time."""
    yield from iter_jsonl_offset_objects(
        record_refs,
        description="PR JSONL payload",
        log=log,
    )
