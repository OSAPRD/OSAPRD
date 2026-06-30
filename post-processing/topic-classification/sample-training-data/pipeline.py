"""Stratified sampling and train/test splitting for topic training data.

Stage 2b joins preprocessed repository records with extraction metadata, filters
the retained topic universe, samples deterministically by popularity and
creation-time strata, and writes model-ready train/test CSVs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from heapq import heappush, heapreplace
from pathlib import Path
from typing import Any, Iterable, Iterator


LOG_PREFIX = "[post-processing/topic-sample-training-data]"
SAMPLE_TRAINING_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_REPOS = 200_000
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_SEED = 42
DEFAULT_OUTPUT_SUBDIR = Path("sampled-training-data")
DEFAULT_PROGRESS_INTERVAL = 10_000
DEFAULT_FILTERED_TOPICS_PATH = SAMPLE_TRAINING_DATA_DIR / "filtered_topics.json"
POPULARITY_BINS = ("0", "1-18", "19+")
LABELS_COLUMN = "labels"
TEXT_COLUMN = "text"


@dataclass(frozen=True)
class SampleTrainingDataConfig:
    """Settings for the deterministic sampling and train/test split stage."""

    create_training_data_dir: Path
    topic_repo_extraction_dir: Path
    target_repos: int = DEFAULT_TARGET_REPOS
    train_fraction: float = DEFAULT_TRAIN_FRACTION
    seed: int = DEFAULT_SEED
    sample_name: str | None = None
    output_subdir: Path = DEFAULT_OUTPUT_SUBDIR
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL
    filtered_topics_path: Path | None = DEFAULT_FILTERED_TOPICS_PATH
    topic_domains_path: Path | None = None

    @classmethod
    def from_env(cls) -> "SampleTrainingDataConfig":
        create_dir = os.environ.get("POST_PROCESSING_TOPIC_SAMPLE_CREATE_TRAINING_DATA_DIR")
        extraction_dir = os.environ.get("POST_PROCESSING_TOPIC_SAMPLE_REPO_EXTRACTION_DIR")
        if not create_dir:
            raise ValueError(
                "Create-training-data input folder is required. Set "
                "POST_PROCESSING_TOPIC_SAMPLE_CREATE_TRAINING_DATA_DIR or pass "
                "--create-training-data-dir."
            )
        if not extraction_dir:
            raise ValueError(
                "Topic repository extraction input folder is required. Set "
                "POST_PROCESSING_TOPIC_SAMPLE_REPO_EXTRACTION_DIR or pass "
                "--topic-repo-extraction-dir."
            )
        return cls(
            create_training_data_dir=Path(create_dir).expanduser(),
            topic_repo_extraction_dir=Path(extraction_dir).expanduser(),
            target_repos=_env_int("POST_PROCESSING_TOPIC_SAMPLE_TARGET_REPOS", DEFAULT_TARGET_REPOS),
            train_fraction=_env_float(
                "POST_PROCESSING_TOPIC_SAMPLE_TRAIN_FRACTION",
                DEFAULT_TRAIN_FRACTION,
            ),
            seed=_env_int("POST_PROCESSING_TOPIC_SAMPLE_SEED", DEFAULT_SEED),
            sample_name=os.environ.get("POST_PROCESSING_TOPIC_SAMPLE_NAME") or None,
            output_subdir=Path(
                os.environ.get(
                    "POST_PROCESSING_TOPIC_SAMPLE_OUTPUT_SUBDIR",
                    str(DEFAULT_OUTPUT_SUBDIR),
                )
            ),
            progress_interval=_env_int(
                "POST_PROCESSING_TOPIC_SAMPLE_PROGRESS_INTERVAL",
                DEFAULT_PROGRESS_INTERVAL,
            ),
            filtered_topics_path=_env_optional_path(
                "POST_PROCESSING_TOPIC_SAMPLE_FILTERED_TOPICS_JSON",
                DEFAULT_FILTERED_TOPICS_PATH,
            ),
            topic_domains_path=_env_optional_path(
                "POST_PROCESSING_TOPIC_SAMPLE_TOPIC_DOMAINS_JSON",
                None,
            ),
        )


@dataclass(frozen=True)
class RepositoryMetadata:
    """Repository metadata needed for stratification and output manifests."""

    repository_identity_key: str | None
    repository_id: str | None
    repository_key: str | None
    repository_full_name: str | None
    stargazers_count: int
    created_at: str
    source: str


@dataclass(frozen=True)
class EligibleRecord:
    """Minimal record selected during quota counting and deterministic sampling."""

    identity_key: str
    stratum: str
    popularity_bin: str
    created_time_bin: str


@dataclass(frozen=True)
class TopicLabelFilter:
    """Topic-universe filter derived from filtered topics and domain mappings."""

    original_label_universe: list[str]
    retained_label_universe: list[str]
    filtered_topics: set[str]
    filtered_topics_present: list[str]
    topic_domain_topics: set[str]
    topic_domain_topics_present: list[str]
    topic_domain_topics_missing: list[str]
    retained_original_indices: list[int]
    original_label_universe_sha256: str
    retained_label_universe_sha256: str
    filtered_topics_path: Path | None
    topic_domains_path: Path | None


def log(message: str) -> None:
    """Print a flushed sampling-stage log line."""
    print(f"{LOG_PREFIX} {message}", flush=True)


def sample_training_data(config: SampleTrainingDataConfig) -> dict[str, Any]:
    """Create a stratified sample and write train/test CSV artifacts."""
    config = _normalized_config(config)
    create_dir = config.create_training_data_dir
    extraction_dir = config.topic_repo_extraction_dir
    paths = _resolve_paths(config)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Create-training-data input: {create_dir}")
    log(f"Topic repo extraction input: {extraction_dir}")
    log(f"Output directory: {output_dir}")
    log(
        "Config: "
        f"target_repos={config.target_repos}, "
        f"train_fraction={config.train_fraction}, "
        f"seed={config.seed}, "
        f"popularity_bins={', '.join(POPULARITY_BINS)}, "
        "created_time_bin=year-quarter, "
        f"filtered_topics_path={config.filtered_topics_path}, "
        f"topic_domains_path={config.topic_domains_path}, "
        f"progress_interval={config.progress_interval}"
    )

    label_filter = _build_label_filter(
        paths["label_universe"],
        config.filtered_topics_path,
        config.topic_domains_path,
    )
    log(
        "Topic label filtering: "
        f"original_labels={len(label_filter.original_label_universe)}, "
        f"filtered_topics={len(label_filter.filtered_topics)}, "
        f"filtered_present={len(label_filter.filtered_topics_present)}, "
        f"topic_domain_topics={len(label_filter.topic_domain_topics)}, "
        f"topic_domain_present={len(label_filter.topic_domain_topics_present)}, "
        f"topic_domain_missing={len(label_filter.topic_domain_topics_missing)}, "
        f"retained_labels={len(label_filter.retained_label_universe)}"
    )
    metadata_index, metadata_counts = _build_metadata_index(
        extraction_dir,
        progress_interval=config.progress_interval,
    )
    stratum_counts, eligibility_counts = _count_eligible_records(
        paths=paths,
        metadata_index=metadata_index,
        label_filter=label_filter,
        progress_interval=config.progress_interval,
    )
    eligible_total = sum(stratum_counts.values())
    if eligible_total < config.target_repos:
        raise ValueError(
            "Not enough eligible repositories for requested sample: "
            f"eligible={eligible_total}, target={config.target_repos}. "
            "Eligibility requires encoded text/labels plus stars and created_at "
            "from the topic repo extraction folder."
        )

    quotas = _allocate_proportional_quotas(stratum_counts, config.target_repos)
    selected_by_stratum = _select_records(
        paths=paths,
        metadata_index=metadata_index,
        quotas=quotas,
        seed=config.seed,
        label_filter=label_filter,
        progress_interval=config.progress_interval,
    )
    selected_keys = {
        identity_key
        for keys in selected_by_stratum.values()
        for identity_key in keys
    }
    split_assignment = _assign_train_test(
        selected_by_stratum,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    output_counts = _write_outputs(
        paths=paths,
        metadata_index=metadata_index,
        selected_keys=selected_keys,
        split_assignment=split_assignment,
        label_filter=label_filter,
        progress_interval=config.progress_interval,
    )
    _write_stratum_counts(
        paths["stratum_counts"],
        stratum_counts=stratum_counts,
        quotas=quotas,
        selected_by_stratum=selected_by_stratum,
        split_assignment=split_assignment,
    )

    manifest = {
        "schema_version": "topic_sample_train_test_manifest_v1",
        "created_at_utc": _utc_now_z(),
        "create_training_data_dir": str(create_dir),
        "topic_repo_extraction_dir": str(extraction_dir),
        "sample_name": output_dir.name,
        "target_repos": config.target_repos,
        "sampled_repos": output_counts["sampled_records"],
        "train_fraction": config.train_fraction,
        "seed": config.seed,
        "popularity_bins": list(POPULARITY_BINS),
        "created_time_bin": "repository_created_at_year_quarter",
        "original_label_universe_size": len(label_filter.original_label_universe),
        "original_label_universe_sha256": label_filter.original_label_universe_sha256,
        "label_universe_size": len(label_filter.retained_label_universe),
        "label_universe_sha256": label_filter.retained_label_universe_sha256,
        "filtered_topics_path": (
            str(label_filter.filtered_topics_path)
            if label_filter.filtered_topics_path is not None
            else None
        ),
        "filtered_topics_count": len(label_filter.filtered_topics),
        "filtered_topics_present_count": len(label_filter.filtered_topics_present),
        "filtered_topics_present": label_filter.filtered_topics_present,
        "topic_domains_path": (
            str(label_filter.topic_domains_path)
            if label_filter.topic_domains_path is not None
            else None
        ),
        "topic_domain_topics_count": len(label_filter.topic_domain_topics),
        "topic_domain_topics_present_count": len(
            label_filter.topic_domain_topics_present
        ),
        "topic_domain_topics_missing_count": len(
            label_filter.topic_domain_topics_missing
        ),
        "topic_domain_topics_missing": label_filter.topic_domain_topics_missing,
        "retained_label_universe": label_filter.retained_label_universe,
        "metadata_counts": dict(metadata_counts),
        "eligibility_counts": dict(eligibility_counts),
        "stratum_count": len(stratum_counts),
        "train_rows": output_counts["train_rows"],
        "test_rows": output_counts["test_rows"],
        "outputs": {
            "sampled_encoded_repository_records": str(paths["sampled_encoded_records"]),
            "sampled_topic_training_dataset_csv": str(paths["sampled_dataset_csv"]),
            "train_csv": str(paths["train_csv"]),
            "test_csv": str(paths["test_csv"]),
            "sampled_repository_metadata_csv": str(paths["sampled_metadata_csv"]),
            "stratum_counts_csv": str(paths["stratum_counts"]),
            "excluded_sampling_records": str(paths["excluded_records"]),
        },
        "input_hashes": {
            "encoded_repository_records": _sha256_file(paths["encoded_records"]),
            "topic_label_universe": _sha256_file(paths["label_universe"]),
            "filtered_topics": (
                _sha256_file(label_filter.filtered_topics_path)
                if label_filter.filtered_topics_path is not None
                and label_filter.filtered_topics_path.exists()
                else None
            ),
            "topic_domains": (
                _sha256_file(label_filter.topic_domains_path)
                if label_filter.topic_domains_path is not None
                and label_filter.topic_domains_path.exists()
                else None
            ),
            "raw_repository_search_candidates": (
                _sha256_file(paths["raw_candidates"])
                if paths["raw_candidates"].exists()
                else None
            ),
        },
    }
    _write_json(paths["manifest"], manifest)
    log(
        "Sampling complete: "
        f"sampled={output_counts['sampled_records']}, "
        f"train={output_counts['train_rows']}, "
        f"test={output_counts['test_rows']}, "
        f"eligible={eligible_total}, "
        f"excluded={eligibility_counts['excluded_records']}"
    )
    return manifest


def _build_metadata_index(
    extraction_dir: Path,
    *,
    progress_interval: int,
) -> tuple[dict[str, RepositoryMetadata], Counter[str]]:
    counts: Counter[str] = Counter()
    index: dict[str, RepositoryMetadata] = {}
    raw_candidates = extraction_dir / "raw" / "repository_search_candidates.jsonl"
    repository_records = extraction_dir / "repositories" / "repository_training_records.jsonl"
    log("Metadata indexing started.")
    for source_path, source_name in (
        (raw_candidates, "raw_repository_search_candidates"),
        (repository_records, "repository_training_records"),
    ):
        if not source_path.exists():
            counts[f"{source_name}_missing"] += 1
            continue
        for record in _iter_jsonl_records(source_path):
            counts[f"{source_name}_rows"] += 1
            metadata = _metadata_from_record(record, source=source_name)
            if metadata is None:
                counts[f"{source_name}_rows_without_complete_metadata"] += 1
            else:
                for key in _metadata_lookup_keys(metadata):
                    existing = index.get(key)
                    if existing is None or existing.source != "raw_repository_search_candidates":
                        index[key] = metadata
                counts[f"{source_name}_rows_indexed"] += 1
            processed = counts[f"{source_name}_rows"]
            if processed <= 10 or processed % progress_interval == 0:
                log(
                    "Metadata indexing progress: "
                    f"source={source_name}, rows={processed}, "
                    f"indexed={counts[f'{source_name}_rows_indexed']}, "
                    f"index_keys={len(index)}"
                )
    counts["metadata_index_keys"] = len(index)
    log(
        "Metadata indexing complete: "
        f"index_keys={len(index)}, "
        f"raw_rows={counts['raw_repository_search_candidates_rows']}, "
        f"repository_rows={counts['repository_training_records_rows']}"
    )
    return index, counts


def _count_eligible_records(
    *,
    paths: dict[str, Path],
    metadata_index: dict[str, RepositoryMetadata],
    label_filter: TopicLabelFilter,
    progress_interval: int,
) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    log("Eligibility count pass started.")
    with paths["excluded_records"].open("w", encoding="utf-8", newline="\n") as excluded:
        for record in _iter_jsonl_records(paths["encoded_records"]):
            counts["processed_records"] += 1
            decision = _eligible_record(record, metadata_index, label_filter)
            if isinstance(decision, str):
                counts["excluded_records"] += 1
                counts[f"excluded_{decision}"] += 1
                _write_jsonl(
                    excluded,
                    {
                        "repository_identity_key": _record_identity_key(record),
                        "repository_id": record.get("repository_id"),
                        "repository_key": record.get("repository_key"),
                        "repository_full_name": record.get("repository_full_name"),
                        "reason": decision,
                    },
                )
            else:
                counts["eligible_records"] += 1
                stratum_counts[decision.stratum] += 1
            processed = counts["processed_records"]
            if processed <= 10 or processed % progress_interval == 0:
                log(
                    "Eligibility count progress: "
                    f"processed={processed}, eligible={counts['eligible_records']}, "
                    f"excluded={counts['excluded_records']}, "
                    f"strata={len(stratum_counts)}"
                )
    log(
        "Eligibility count pass complete: "
        f"processed={counts['processed_records']}, "
        f"eligible={counts['eligible_records']}, "
        f"excluded={counts['excluded_records']}, "
        f"strata={len(stratum_counts)}"
    )
    return stratum_counts, counts


def _select_records(
    *,
    paths: dict[str, Path],
    metadata_index: dict[str, RepositoryMetadata],
    quotas: dict[str, int],
    seed: int,
    label_filter: TopicLabelFilter,
    progress_interval: int,
) -> dict[str, set[str]]:
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    log("Selection pass started.")
    for record in _iter_jsonl_records(paths["encoded_records"]):
        counts["processed_records"] += 1
        decision = _eligible_record(record, metadata_index, label_filter)
        if isinstance(decision, str):
            continue
        quota = quotas.get(decision.stratum, 0)
        if quota <= 0:
            continue
        priority = _stable_priority(seed, "sample", decision.identity_key)
        heap = heaps[decision.stratum]
        item = (-priority, decision.identity_key)
        if len(heap) < quota:
            heappush(heap, item)
            counts["candidate_records_selected_or_retained"] += 1
        elif priority < -heap[0][0]:
            heapreplace(heap, item)
            counts["candidate_records_selected_or_retained"] += 1
        processed = counts["processed_records"]
        if processed <= 10 or processed % progress_interval == 0:
            selected_count = sum(len(values) for values in heaps.values())
            log(
                "Selection progress: "
                f"processed={processed}, selected={selected_count}/{sum(quotas.values())}, "
                f"active_strata={len(heaps)}"
            )
    selected_by_stratum = {
        stratum: {identity_key for _, identity_key in heap}
        for stratum, heap in heaps.items()
    }
    selected_count = sum(len(values) for values in selected_by_stratum.values())
    log(
        "Selection pass complete: "
        f"selected={selected_count}/{sum(quotas.values())}, "
        f"strata={len(selected_by_stratum)}"
    )
    return selected_by_stratum


def _write_outputs(
    *,
    paths: dict[str, Path],
    metadata_index: dict[str, RepositoryMetadata],
    selected_keys: set[str],
    split_assignment: dict[str, str],
    label_filter: TopicLabelFilter,
    progress_interval: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    label_universe = label_filter.retained_label_universe
    csv_fieldnames = [*label_universe, LABELS_COLUMN, TEXT_COLUMN]
    metadata_fieldnames = [
        "repository_identity_key",
        "repository_id",
        "repository_full_name",
        "repository_key",
        "repository_stargazers_count",
        "repository_created_at",
        "popularity_bin",
        "created_time_bin",
        "sampling_stratum",
        "split",
    ]
    log("Output write pass started.")
    with (
        paths["sampled_encoded_records"].open("w", encoding="utf-8", newline="\n") as jsonl_out,
        paths["sampled_dataset_csv"].open("w", encoding="utf-8", newline="") as sampled_csv,
        paths["train_csv"].open("w", encoding="utf-8", newline="") as train_csv,
        paths["test_csv"].open("w", encoding="utf-8", newline="") as test_csv,
        paths["sampled_metadata_csv"].open("w", encoding="utf-8", newline="") as metadata_csv,
    ):
        sampled_writer = csv.DictWriter(sampled_csv, fieldnames=csv_fieldnames)
        train_writer = csv.DictWriter(train_csv, fieldnames=csv_fieldnames)
        test_writer = csv.DictWriter(test_csv, fieldnames=csv_fieldnames)
        metadata_writer = csv.DictWriter(metadata_csv, fieldnames=metadata_fieldnames)
        for writer in (sampled_writer, train_writer, test_writer, metadata_writer):
            writer.writeheader()

        for record in _iter_jsonl_records(paths["encoded_records"]):
            counts["processed_records"] += 1
            identity_key = _record_identity_key(record)
            if identity_key not in selected_keys:
                continue
            metadata = _lookup_metadata(record, metadata_index)
            if metadata is None:
                continue
            popularity_bin = _popularity_bin(metadata.stargazers_count)
            created_time_bin = _created_time_bin(metadata.created_at)
            stratum = _stratum(popularity_bin, created_time_bin)
            split = split_assignment[identity_key]
            label_vector = _filtered_label_vector(record, label_filter)
            prepared_topics = [
                label
                for index, label in enumerate(label_filter.retained_label_universe)
                if label_vector[index]
            ]
            filtered_removed_topics = _active_filtered_topics(record, label_filter)
            labels_text = "[" + " ".join(str(bit) for bit in label_vector) + "]"
            csv_payload = {
                label: label_vector[index]
                for index, label in enumerate(label_universe)
            }
            csv_payload[LABELS_COLUMN] = labels_text
            csv_payload[TEXT_COLUMN] = str(record.get("prepared_text") or "").strip()
            enriched = dict(record)
            enriched.update(
                {
                    "schema_version": "topic_training_sampled_repository_record_v1",
                    "prepared_topics": prepared_topics,
                    "label_vector": label_vector,
                    "label_indices": [
                        index for index, bit in enumerate(label_vector) if bit
                    ],
                    "label_count": sum(label_vector),
                    "label_universe_size": len(label_filter.retained_label_universe),
                    "label_universe_sha256": (
                        label_filter.retained_label_universe_sha256
                    ),
                    "original_label_universe_size": (
                        len(label_filter.original_label_universe)
                    ),
                    "original_label_universe_sha256": (
                        label_filter.original_label_universe_sha256
                    ),
                    "filtered_topics_removed": filtered_removed_topics,
                    "repository_stargazers_count": metadata.stargazers_count,
                    "repository_created_at": metadata.created_at,
                    "popularity_bin": popularity_bin,
                    "created_time_bin": created_time_bin,
                    "sampling_stratum": stratum,
                    "sample_split": split,
                    "sample_label_universe_sha256": (
                        label_filter.retained_label_universe_sha256
                    ),
                }
            )
            _write_jsonl(jsonl_out, enriched)
            sampled_writer.writerow(csv_payload)
            if split == "train":
                train_writer.writerow(csv_payload)
                counts["train_rows"] += 1
            else:
                test_writer.writerow(csv_payload)
                counts["test_rows"] += 1
            metadata_writer.writerow(
                {
                    "repository_identity_key": identity_key,
                    "repository_id": record.get("repository_id") or metadata.repository_id,
                    "repository_full_name": (
                        record.get("repository_full_name") or metadata.repository_full_name
                    ),
                    "repository_key": record.get("repository_key") or metadata.repository_key,
                    "repository_stargazers_count": metadata.stargazers_count,
                    "repository_created_at": metadata.created_at,
                    "popularity_bin": popularity_bin,
                    "created_time_bin": created_time_bin,
                    "sampling_stratum": stratum,
                    "split": split,
                }
            )
            counts["sampled_records"] += 1
            if counts["sampled_records"] <= 10 or counts["sampled_records"] % progress_interval == 0:
                log(
                    "Output write progress: "
                    f"processed={counts['processed_records']}, "
                    f"sampled={counts['sampled_records']}, "
                    f"train={counts['train_rows']}, test={counts['test_rows']}"
                )
    log(
        "Output write pass complete: "
        f"sampled={counts['sampled_records']}, "
        f"train={counts['train_rows']}, test={counts['test_rows']}"
    )
    return counts


def _allocate_proportional_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target > total:
        raise ValueError(f"target {target} exceeds available records {total}")
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, count in sorted(counts.items()):
        raw = (target * count) / total
        quota = min(int(math.floor(raw)), count)
        quotas[stratum] = quota
        remainders.append((raw - quota, stratum))
    remaining = target - sum(quotas.values())
    while remaining > 0:
        progressed = False
        for _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if remaining <= 0:
                break
            if quotas[stratum] >= counts[stratum]:
                continue
            quotas[stratum] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("Unable to allocate proportional sample quotas.")
    return quotas


def _assign_train_test(
    selected_by_stratum: dict[str, set[str]],
    *,
    train_fraction: float,
    seed: int,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for stratum, keys in selected_by_stratum.items():
        ordered = sorted(
            keys,
            key=lambda key: (_stable_priority(seed, "split", stratum, key), key),
        )
        count = len(ordered)
        if count == 1:
            train_count = 1
        else:
            train_count = int(math.floor(count * train_fraction))
            train_count = min(max(train_count, 1), count - 1)
        for index, key in enumerate(ordered):
            assignments[key] = "train" if index < train_count else "test"
    return assignments


def _eligible_record(
    record: dict[str, Any],
    metadata_index: dict[str, RepositoryMetadata],
    label_filter: TopicLabelFilter,
) -> EligibleRecord | str:
    identity_key = _record_identity_key(record)
    if not identity_key:
        return "missing_identity"
    text = str(record.get("prepared_text") or "").strip()
    if not text:
        return "empty_prepared_text"
    try:
        label_vector = _filtered_label_vector(record, label_filter)
    except ValueError:
        return "invalid_label_vector"
    if not any(label_vector):
        return "empty_label_vector_after_topic_filter"
    metadata = _lookup_metadata(record, metadata_index)
    if metadata is None:
        return "missing_metadata"
    popularity_bin = _popularity_bin(metadata.stargazers_count)
    created_time_bin = _created_time_bin(metadata.created_at)
    if created_time_bin is None:
        return "invalid_repository_created_at"
    return EligibleRecord(
        identity_key=identity_key,
        stratum=_stratum(popularity_bin, created_time_bin),
        popularity_bin=popularity_bin,
        created_time_bin=created_time_bin,
    )


def _metadata_from_record(record: dict[str, Any], *, source: str) -> RepositoryMetadata | None:
    stars = _optional_int(
        record.get("repository_stargazers_count")
        if record.get("repository_stargazers_count") is not None
        else record.get("stargazers_count")
    )
    created_at = _first_non_empty(
        record.get("repository_created_at"),
        record.get("created_at"),
    )
    if stars is None or not created_at:
        return None
    return RepositoryMetadata(
        repository_identity_key=_optional_str(record.get("repository_identity_key")),
        repository_id=_optional_str(record.get("repository_id")),
        repository_key=_normalized_repo_key(
            record.get("repository_key")
            or record.get("repository_full_name")
            or _join_owner_repo(record.get("repository_owner"), record.get("repository_name"))
        ),
        repository_full_name=_optional_str(
            record.get("repository_full_name")
            or _join_owner_repo(record.get("repository_owner"), record.get("repository_name"))
        ),
        stargazers_count=stars,
        created_at=str(created_at),
        source=source,
    )


def _lookup_metadata(
    record: dict[str, Any],
    metadata_index: dict[str, RepositoryMetadata],
) -> RepositoryMetadata | None:
    for key in _record_lookup_keys(record):
        metadata = metadata_index.get(key)
        if metadata is not None:
            return metadata
    return None


def _metadata_lookup_keys(metadata: RepositoryMetadata) -> set[str]:
    keys = set()
    if metadata.repository_identity_key:
        keys.add(metadata.repository_identity_key)
    if metadata.repository_id:
        keys.add(f"repository-id:{metadata.repository_id}")
        keys.add(metadata.repository_id)
    if metadata.repository_key:
        keys.add(f"repository-key:{metadata.repository_key}")
        keys.add(metadata.repository_key)
    return keys


def _record_lookup_keys(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    identity_key = _record_identity_key(record)
    if identity_key:
        keys.append(identity_key)
    repository_id = _optional_str(record.get("repository_id"))
    if repository_id:
        keys.extend([f"repository-id:{repository_id}", repository_id])
    repository_key = _normalized_repo_key(
        record.get("repository_key")
        or record.get("repository_full_name")
        or _join_owner_repo(record.get("repository_owner"), record.get("repository_name"))
    )
    if repository_key:
        keys.extend([f"repository-key:{repository_key}", repository_key])
    return list(dict.fromkeys(keys))


def _record_identity_key(record: dict[str, Any]) -> str:
    identity_key = _optional_str(record.get("repository_identity_key"))
    if identity_key:
        return identity_key
    repository_id = _optional_str(record.get("repository_id"))
    if repository_id:
        return f"repository-id:{repository_id}"
    repository_key = _normalized_repo_key(
        record.get("repository_key")
        or record.get("repository_full_name")
        or _join_owner_repo(record.get("repository_owner"), record.get("repository_name"))
    )
    return f"repository-key:{repository_key}" if repository_key else ""


def _filtered_label_vector(
    record: dict[str, Any],
    label_filter: TopicLabelFilter,
) -> list[int]:
    original_vector = _label_vector(record, len(label_filter.original_label_universe))
    return [
        original_vector[index]
        for index in label_filter.retained_original_indices
    ]


def _active_filtered_topics(
    record: dict[str, Any],
    label_filter: TopicLabelFilter,
) -> list[str]:
    original_vector = _label_vector(record, len(label_filter.original_label_universe))
    filtered_topics: list[str] = []
    for index, label in enumerate(label_filter.original_label_universe):
        if original_vector[index] and _normalize_topic_slug(label) in label_filter.filtered_topics:
            filtered_topics.append(label)
    return filtered_topics


def _label_vector(record: dict[str, Any], label_universe_size: int) -> list[int]:
    raw = record.get("label_vector")
    if isinstance(raw, list):
        vector = [1 if int(value) else 0 for value in raw]
    else:
        indices = [
            int(index)
            for index in (record.get("label_indices") or ())
            if str(index).strip()
        ]
        vector = [1 if index in set(indices) else 0 for index in range(label_universe_size)]
    if len(vector) != label_universe_size:
        raise ValueError(
            f"label vector width {len(vector)} does not match label universe {label_universe_size}"
        )
    return vector


def _popularity_bin(stars: int) -> str:
    if stars <= 0:
        return "0"
    if stars <= 18:
        return "1-18"
    return "19+"


def _created_time_bin(created_at: str) -> str | None:
    try:
        value = str(created_at).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    quarter = ((parsed.month - 1) // 3) + 1
    return f"{parsed.year}-Q{quarter}"


def _stratum(popularity_bin: str, created_time_bin: str) -> str:
    return f"stars:{popularity_bin}|created:{created_time_bin}"


def _write_stratum_counts(
    path: Path,
    *,
    stratum_counts: Counter[str],
    quotas: dict[str, int],
    selected_by_stratum: dict[str, set[str]],
    split_assignment: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sampling_stratum",
            "popularity_bin",
            "created_time_bin",
            "eligible_count",
            "sample_quota",
            "sampled_count",
            "train_count",
            "test_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stratum in sorted(stratum_counts):
            selected = selected_by_stratum.get(stratum, set())
            train_count = sum(1 for key in selected if split_assignment.get(key) == "train")
            test_count = sum(1 for key in selected if split_assignment.get(key) == "test")
            popularity_bin, created_time_bin = _parse_stratum(stratum)
            writer.writerow(
                {
                    "sampling_stratum": stratum,
                    "popularity_bin": popularity_bin,
                    "created_time_bin": created_time_bin,
                    "eligible_count": stratum_counts[stratum],
                    "sample_quota": quotas.get(stratum, 0),
                    "sampled_count": len(selected),
                    "train_count": train_count,
                    "test_count": test_count,
                }
            )


def _parse_stratum(stratum: str) -> tuple[str, str]:
    match = re.match(r"^stars:(.*)\|created:(.*)$", stratum)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def _resolve_paths(config: SampleTrainingDataConfig) -> dict[str, Path]:
    create_dir = config.create_training_data_dir
    extraction_dir = config.topic_repo_extraction_dir
    sample_name = config.sample_name or _default_sample_name(config)
    output_dir = create_dir / config.output_subdir / _safe_name(sample_name)
    paths = {
        "create_dir": create_dir,
        "extraction_dir": extraction_dir,
        "encoded_records": create_dir / "encoded_repository_records.jsonl",
        "label_universe": create_dir / "topic_label_universe.json",
        "label_encoding_manifest": create_dir / "label_encoding_manifest.json",
        "raw_candidates": extraction_dir / "raw" / "repository_search_candidates.jsonl",
        "repository_records": extraction_dir / "repositories" / "repository_training_records.jsonl",
        "output_dir": output_dir,
        "sampled_encoded_records": output_dir / "sampled_encoded_repository_records.jsonl",
        "sampled_dataset_csv": output_dir / "sampled_topic_training_dataset.csv",
        "train_csv": output_dir / "topics_train.csv",
        "test_csv": output_dir / "topics_test.csv",
        "sampled_metadata_csv": output_dir / "sampled_repository_metadata.csv",
        "stratum_counts": output_dir / "stratum_counts.csv",
        "excluded_records": output_dir / "excluded_sampling_records.jsonl",
        "manifest": output_dir / "sample_train_test_manifest.json",
    }
    for required in (
        paths["encoded_records"],
        paths["label_universe"],
        paths["label_encoding_manifest"],
        extraction_dir,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")
    return paths


def _normalized_config(config: SampleTrainingDataConfig) -> SampleTrainingDataConfig:
    target = int(config.target_repos)
    if target <= 0:
        raise ValueError(f"target_repos must be positive, got {target}")
    train_fraction = float(config.train_fraction)
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be greater than 0 and less than 1, got {train_fraction}")
    progress_interval = max(1, int(config.progress_interval))
    return SampleTrainingDataConfig(
        create_training_data_dir=Path(config.create_training_data_dir).expanduser(),
        topic_repo_extraction_dir=Path(config.topic_repo_extraction_dir).expanduser(),
        target_repos=target,
        train_fraction=train_fraction,
        seed=int(config.seed),
        sample_name=config.sample_name,
        output_subdir=Path(config.output_subdir),
        progress_interval=progress_interval,
        filtered_topics_path=(
            Path(config.filtered_topics_path).expanduser()
            if config.filtered_topics_path is not None
            else None
        ),
        topic_domains_path=(
            Path(config.topic_domains_path).expanduser()
            if config.topic_domains_path is not None
            else None
        ),
    )


def _default_sample_name(config: SampleTrainingDataConfig) -> str:
    percentage = int(round(float(config.train_fraction) * 100))
    return f"sample{int(config.target_repos)}_train{percentage}_seed{int(config.seed)}"


def _load_label_universe(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Label universe must be a JSON list: {path}")
    labels = [str(item).strip() for item in payload if str(item).strip()]
    if not labels:
        raise ValueError(f"Label universe is empty: {path}")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate labels in label universe: {duplicates[:10]}")
    return labels


def _build_label_filter(
    label_universe_path: Path,
    filtered_topics_path: Path | None,
    topic_domains_path: Path | None = None,
) -> TopicLabelFilter:
    original_labels = _load_label_universe(label_universe_path)
    filtered_topics = _load_filtered_topics(filtered_topics_path)
    topic_domain_topics = _load_topic_domain_topics(topic_domains_path)
    retained_indices = [
        index
        for index, label in enumerate(original_labels)
        if _normalize_topic_slug(label) not in filtered_topics
        and (
            not topic_domain_topics
            or _normalize_topic_slug(label) in topic_domain_topics
        )
    ]
    retained_labels = [original_labels[index] for index in retained_indices]
    if not retained_labels:
        raise ValueError(
            "All labels were removed by the filtered topics configuration: "
            f"{filtered_topics_path}"
        )
    original_label_set = {_normalize_topic_slug(label) for label in original_labels}
    filtered_present = sorted(filtered_topics & original_label_set)
    topic_domain_present = sorted(topic_domain_topics & original_label_set)
    topic_domain_missing = sorted(topic_domain_topics - original_label_set)
    return TopicLabelFilter(
        original_label_universe=original_labels,
        retained_label_universe=retained_labels,
        filtered_topics=filtered_topics,
        filtered_topics_present=filtered_present,
        topic_domain_topics=topic_domain_topics,
        topic_domain_topics_present=topic_domain_present,
        topic_domain_topics_missing=topic_domain_missing,
        retained_original_indices=retained_indices,
        original_label_universe_sha256=_sha256_lines(original_labels),
        retained_label_universe_sha256=_sha256_lines(retained_labels),
        filtered_topics_path=filtered_topics_path,
        topic_domains_path=topic_domains_path,
    )


def _load_filtered_topics(path: Path | None) -> set[str]:
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Filtered topics JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = []
        for key in ("filtered_topics", "topics", "programming_language_topics"):
            raw_values = payload.get(key)
            if isinstance(raw_values, list):
                values.extend(raw_values)
        categories = payload.get("categories")
        if isinstance(categories, dict):
            for category, raw_values in categories.items():
                if not isinstance(raw_values, list):
                    raise ValueError(
                        "Filtered topic category must be a JSON list: "
                        f"{path} category={category}"
                    )
                values.extend(raw_values)
    else:
        raise ValueError(f"Filtered topics JSON must be an object or list: {path}")
    if not isinstance(values, list):
        raise ValueError(f"Filtered topics value must be a JSON list: {path}")
    return {
        normalized
        for normalized in (_normalize_topic_slug(value) for value in values)
        if normalized
    }


def _load_topic_domain_topics(path: Path | None) -> set[str]:
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Topic domains JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    topic_domains = payload.get("topic_domains")
    if not isinstance(topic_domains, dict) or not topic_domains:
        raise ValueError(f"topic_domains JSON does not contain topic_domains: {path}")
    return {
        normalized
        for normalized in (_normalize_topic_slug(topic) for topic in topic_domains)
        if normalized
    }


def _normalize_topic_slug(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_priority(seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "big")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        text = _optional_str(value)
        if text:
            return text
    return None


def _normalized_repo_key(value: Any) -> str | None:
    text = str(value or "").strip().replace("\\", "/")
    if not text or "/" not in text:
        return None
    owner, repo = text.split("/", 1)
    owner = owner.strip().lower()
    repo = repo.strip().lower()
    return f"{owner}/{repo}" if owner and repo else None


def _join_owner_repo(owner: Any, repo: Any) -> str | None:
    owner_text = _optional_str(owner)
    repo_text = _optional_str(repo)
    if owner_text and repo_text:
        return f"{owner_text}/{repo_text}"
    return None


def _safe_name(value: str) -> str:
    text = str(value or "").strip() or "sample"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _utc_now_z() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)
