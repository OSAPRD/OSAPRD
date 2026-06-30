"""Repository feature extraction for topic classification.

Feature extraction turns repository metadata, README/wiki text, and file paths
into the exact prepared text passed to the trained topic model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from topic_loader import RepositoryContext
from topic_preprocessing import SoftwareTagDataPreprocessor, get_default_preprocessor


@dataclass(frozen=True)
class RepositoryFeatures:
    """Final repository-level feature payload written beside predictions."""

    cohort: str
    repository_owner: str
    repository_name: str
    repository_full_name: str
    repository_key: str
    repository_id: str | None
    repository_identity_key: str
    cohorts: tuple[str, ...]
    source_ref: str | None
    source_commit: str | None
    inference_text: str
    observed_topics: tuple[str, ...]
    input_stats: dict[str, Any]
    metrics_payload_count: int
    pr_payload_count: int


@dataclass(frozen=True)
class RepositoryFeatureSources:
    """Raw candidate values collected before preprocessing resolves one value."""

    metadata_candidates: tuple[dict[str, Any], ...]
    owner: Any
    name: Any
    full_name: Any
    repository_id: Any
    source_ref: Any
    source_commit: Any
    description: str
    readme: str
    metadata_wiki: Any
    observed_topics: tuple[str, ...]
    metrics_payload_count: int
    pr_payload_count: int


def resolve_repository_feature_sources(context: RepositoryContext) -> RepositoryFeatureSources:
    """Resolve raw repository metadata inputs before text preprocessing."""
    metrics_payload_count = 0
    pr_payload_count = int(getattr(context.ref, "parquet_pr_count", 0) or 0)
    metadata_candidates: list[dict[str, Any]] = []

    metadata_candidates.extend(getattr(context.ref, "metadata_payloads", ()) or ())

    if isinstance(context.file_list_metadata, dict):
        metadata_candidates.append(context.file_list_metadata)

    for payload in context.iter_metrics_payloads():
        metrics_payload_count += 1
        metadata_candidates.extend(_metadata_candidates_from_metrics(payload))

    for payload in context.iter_pr_payloads():
        pr_payload_count += 1
        metadata_candidates.extend(_metadata_candidates_from_pr(payload))

    ref = context.ref
    owner = _first_non_empty(
        (candidate.get("repository_owner") for candidate in metadata_candidates),
        (_owner_value(candidate.get("owner")) for candidate in metadata_candidates),
        default=ref.repository_owner,
    )
    name = _first_non_empty(
        (candidate.get("repository_name") for candidate in metadata_candidates),
        (candidate.get("name") for candidate in metadata_candidates),
        default=ref.repository_name,
    )
    full_name = _first_non_empty(
        (candidate.get("name_with_owner") for candidate in metadata_candidates),
        (candidate.get("full_name") for candidate in metadata_candidates),
        (candidate.get("repository_full_name") for candidate in metadata_candidates),
        default=f"{owner}/{name}",
    )
    repository_id = _first_non_empty(
        (ref.repository_id,),
        (candidate.get("id") for candidate in metadata_candidates),
        (candidate.get("database_id") for candidate in metadata_candidates),
        (candidate.get("databaseId") for candidate in metadata_candidates),
        (candidate.get("repository_id") for candidate in metadata_candidates),
        default=None,
    )
    source_ref = _first_non_empty(
        (candidate.get("source_ref") for candidate in metadata_candidates),
        default=None,
    )
    source_commit = _first_non_empty(
        (candidate.get("source_commit") for candidate in metadata_candidates),
        default=None,
    )
    description = str(
        _first_non_empty(
            (candidate.get("description") for candidate in metadata_candidates),
            default="",
        )
        or ""
    )
    readme = str(
        _first_non_empty(
            (candidate.get("readme") for candidate in metadata_candidates),
            (candidate.get("readme_text") for candidate in metadata_candidates),
            default="",
        )
        or ""
    )
    metadata_wiki = _first_non_empty(
        (candidate.get("wiki") for candidate in metadata_candidates),
        (candidate.get("wiki_text") for candidate in metadata_candidates),
        (candidate.get("wiki_pages") for candidate in metadata_candidates),
        default="",
    )
    return RepositoryFeatureSources(
        metadata_candidates=tuple(metadata_candidates),
        owner=owner,
        name=name,
        full_name=full_name,
        repository_id=repository_id,
        source_ref=source_ref,
        source_commit=source_commit,
        description=description,
        readme=readme,
        metadata_wiki=metadata_wiki,
        observed_topics=_collect_observed_topics(metadata_candidates),
        metrics_payload_count=metrics_payload_count,
        pr_payload_count=pr_payload_count,
    )


def build_repository_features(
    context: RepositoryContext,
    *,
    readme_status: str = "input",
    wiki_text: str = "",
    wiki_status: str = "not_requested",
    preprocessor: SoftwareTagDataPreprocessor | None = None,
    resolved_sources: RepositoryFeatureSources | None = None,
) -> RepositoryFeatures:
    """Build classifier input text and stable repository identifiers."""
    sources = resolved_sources or resolve_repository_feature_sources(context)
    ref = context.ref
    owner = sources.owner
    name = sources.name
    full_name = sources.full_name
    repository_id = sources.repository_id
    source_ref = sources.source_ref
    source_commit = sources.source_commit
    description = sources.description
    readme = sources.readme
    effective_wiki_text = wiki_text or sources.metadata_wiki
    preprocessor = preprocessor or get_default_preprocessor()
    observed_topics = preprocessor.prepare_topics(sources.observed_topics)
    prepared_text = preprocessor.prepare_repository_text(
        name_parts=(owner, name),
        description=description,
        readme=readme,
        wiki=effective_wiki_text,
        file_paths=context.repository_files,
    )
    input_stats = {
        "file_count": len(context.repository_files),
        "metrics_payload_count": sources.metrics_payload_count,
        "pr_payload_count": sources.pr_payload_count,
        "repository_identity_key": ref.repository_identity_key,
        "cohorts": list(ref.cohorts or (ref.cohort,)),
        "input_format": getattr(ref, "input_format", None),
        "has_file_list": bool(getattr(ref, "has_file_list", True)),
        "file_list_path": str(ref.file_list_path) if ref.file_list_path is not None else None,
        "parquet_pr_count": int(getattr(ref, "parquet_pr_count", 0) or 0),
        "parquet_path_count": len(getattr(ref, "parquet_paths", ()) or ()),
        "has_description": prepared_text.token_counts["description"] > 0,
        "has_readme": prepared_text.token_counts["readme"] > 0,
        "readme_status": readme_status,
        "has_wiki": prepared_text.token_counts["wiki"] > 0,
        "wiki_status": wiki_status,
        "token_counts": prepared_text.token_counts,
        "data_preparation": prepared_text.data_preparation,
    }
    return RepositoryFeatures(
        cohort=ref.cohort,
        repository_owner=str(owner),
        repository_name=str(name),
        repository_full_name=str(full_name),
        repository_key=ref.repository_key,
        repository_id=str(repository_id) if repository_id is not None else None,
        repository_identity_key=ref.repository_identity_key,
        cohorts=ref.cohorts or (ref.cohort,),
        source_ref=str(source_ref) if source_ref is not None else None,
        source_commit=str(source_commit) if source_commit is not None else None,
        inference_text=prepared_text.text,
        observed_topics=observed_topics,
        input_stats=input_stats,
        metrics_payload_count=sources.metrics_payload_count,
        pr_payload_count=sources.pr_payload_count,
    )


def _metadata_candidates_from_metrics(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    repository_metadata = payload.get("repository_metadata")
    if isinstance(repository_metadata, dict):
        yield repository_metadata
    pr_payload = payload.get("pr")
    if isinstance(pr_payload, dict):
        base_repository = pr_payload.get("base_repository_full") or pr_payload.get(
            "base_repository"
        )
        if isinstance(base_repository, dict):
            yield base_repository


def _metadata_candidates_from_pr(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    base_repository = payload.get("base_repository_full") or payload.get("base_repository")
    if isinstance(base_repository, dict):
        yield base_repository
    original = payload.get("original_pr_payload")
    if isinstance(original, dict):
        base_repository = original.get("base_repository_full") or original.get(
            "base_repository"
        )
        if isinstance(base_repository, dict):
            yield base_repository


def _owner_value(owner: Any) -> Any:
    if isinstance(owner, dict):
        return owner.get("login") or owner.get("name")
    return owner


def _first_non_empty(*groups: Iterable[Any], default: Any = None) -> Any:
    for group in groups:
        for value in group:
            if value is None:
                continue
            if isinstance(value, (list, tuple, set, dict)):
                if value:
                    return value
                continue
            text = str(value).strip()
            if text and text.lower() not in {"none", "nan", "null"}:
                return value
    return default


def _collect_observed_topics(candidates: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    topics: list[str] = []
    for candidate in candidates:
        for key in ("repository_topics", "topics"):
            topics.extend(_topic_names(candidate.get(key)))
    deduped = sorted({topic for topic in topics if topic})
    return tuple(deduped)


def _topic_names(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        for part in re.split(r"[,;\s]+", value):
            text = part.strip()
            if text:
                yield text
        return
    if isinstance(value, dict):
        nodes = value.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                yield from _topic_names(node)
        topic = value.get("topic")
        if isinstance(topic, dict):
            yield from _topic_names(topic)
        name = value.get("name") or value.get("topic_name")
        if name:
            yield str(name).strip()
        return
    if isinstance(value, list):
        for item in value:
            yield from _topic_names(item)
