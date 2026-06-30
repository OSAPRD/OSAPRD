"""Offline preparation stage for collected topic-classifier training data.

Stage 2a reads extraction records, removes unusable repositories, applies the
topic/text preprocessing rules, counts corpus frequencies, and writes encoded
repository records consumed by the sampling stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import signal
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


CREATE_TRAINING_DATA_DIR = Path(__file__).resolve().parent
TOPIC_CLASSIFICATION_DIR = CREATE_TRAINING_DATA_DIR.parent
CLASSIFY_TOPICS_DIR = TOPIC_CLASSIFICATION_DIR / "classify-topics"
DATA_PREPARATION_DIR = TOPIC_CLASSIFICATION_DIR / "data-preparation"
if str(CLASSIFY_TOPICS_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFY_TOPICS_DIR))

from topic_preprocessing import (  # noqa: E402
    DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS,
    SoftwareTagDataPreprocessor,
)


LOG_PREFIX = "[post-processing/topic-create-training-data]"
DEFAULT_OUTPUT_SUBDIR = Path("training-data") / "create-training-data"
DEFAULT_MAX_NON_ENGLISH_RATIO = 0.5
DEFAULT_PROGRESS_INTERVAL = 1000
DEFAULT_MIN_TOPIC_REPOSITORIES = 100
DEFAULT_MIN_TEXT_TOKEN_FREQUENCY = 50
DEFAULT_MIN_FILE_NAME_TOKEN_FREQUENCY = 20
DEFAULT_IDENTIFIER_SPLITTER = "spiral"
DEFAULT_DATA_PREPARATION_ROOT = DATA_PREPARATION_DIR / "generated" / "github-topics"
DEFAULT_MAX_RECORDS = 0
DEFAULT_EXCLUDED_REPOSITORIES: tuple[str, ...] = ()
DEFAULT_EXCLUDED_REPOSITORIES_FILE: Path | None = None
DEFAULT_RESUME = True
DEFAULT_SOURCE_TIMEOUT_SECONDS = 60.0
SKIPPED_STATIC_LOW_FREQ_RULES = ("low_freq_topics.csv",)
COUNT_PASS_DIAGNOSTIC_RECORD_INTERVAL = 100
COUNT_PASS_HEARTBEAT_SECONDS = 60.0
COUNT_PASS_SLOW_SOURCE_SECONDS = 10.0
COUNT_PASS_CHECKPOINT_RECORD_INTERVAL = 10_000


@dataclass(frozen=True)
class CreateTrainingDataConfig:
    """Settings for filtering and preprocessing one extraction run."""

    input_run: Path
    output_subdir: Path = DEFAULT_OUTPUT_SUBDIR
    max_non_english_ratio: float = DEFAULT_MAX_NON_ENGLISH_RATIO
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL
    min_topic_repositories: int = DEFAULT_MIN_TOPIC_REPOSITORIES
    min_text_token_frequency: int = DEFAULT_MIN_TEXT_TOKEN_FREQUENCY
    min_file_name_token_frequency: int = DEFAULT_MIN_FILE_NAME_TOKEN_FREQUENCY
    identifier_splitter: str = DEFAULT_IDENTIFIER_SPLITTER
    data_preparation_root: Path = DEFAULT_DATA_PREPARATION_ROOT
    max_records: int = DEFAULT_MAX_RECORDS
    excluded_repositories: tuple[str, ...] = DEFAULT_EXCLUDED_REPOSITORIES
    excluded_repositories_file: Path | None = DEFAULT_EXCLUDED_REPOSITORIES_FILE
    resume: bool = DEFAULT_RESUME
    source_timeout_seconds: float = DEFAULT_SOURCE_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "CreateTrainingDataConfig":
        input_run = os.environ.get("POST_PROCESSING_TOPIC_CREATE_TRAINING_INPUT_RUN")
        if not input_run:
            raise ValueError(
                "Input run is required. Set POST_PROCESSING_TOPIC_CREATE_TRAINING_INPUT_RUN "
                "or pass --input-run."
            )
        return cls(
            input_run=Path(input_run).expanduser(),
            output_subdir=Path(
                os.environ.get(
                    "POST_PROCESSING_TOPIC_CREATE_TRAINING_OUTPUT_SUBDIR",
                    str(DEFAULT_OUTPUT_SUBDIR),
                )
            ),
            max_non_english_ratio=_env_float(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MAX_NON_ENGLISH_RATIO",
                DEFAULT_MAX_NON_ENGLISH_RATIO,
            ),
            progress_interval=_env_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_PROGRESS_INTERVAL",
                DEFAULT_PROGRESS_INTERVAL,
            ),
            min_topic_repositories=_env_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MIN_TOPIC_REPOSITORIES",
                DEFAULT_MIN_TOPIC_REPOSITORIES,
            ),
            min_text_token_frequency=_env_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MIN_TEXT_TOKEN_FREQUENCY",
                DEFAULT_MIN_TEXT_TOKEN_FREQUENCY,
            ),
            min_file_name_token_frequency=_env_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MIN_FILE_NAME_TOKEN_FREQUENCY",
                DEFAULT_MIN_FILE_NAME_TOKEN_FREQUENCY,
            ),
            identifier_splitter=os.environ.get(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_IDENTIFIER_SPLITTER",
                DEFAULT_IDENTIFIER_SPLITTER,
            ).strip()
            or DEFAULT_IDENTIFIER_SPLITTER,
            data_preparation_root=Path(
                os.environ.get(
                    "POST_PROCESSING_TOPIC_CREATE_TRAINING_DATA_PREPARATION_ROOT",
                    str(DEFAULT_DATA_PREPARATION_ROOT),
                )
            ).expanduser(),
            max_records=_env_int_allow_zero(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MAX_RECORDS",
                DEFAULT_MAX_RECORDS,
            ),
            excluded_repositories=_env_string_list(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_EXCLUDED_REPOSITORIES"
            ),
            excluded_repositories_file=_env_optional_path(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_EXCLUDED_REPOSITORIES_FILE"
            ),
            resume=_env_bool(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_RESUME",
                DEFAULT_RESUME,
            ),
            source_timeout_seconds=_env_float(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_SOURCE_TIMEOUT_SECONDS",
                DEFAULT_SOURCE_TIMEOUT_SECONDS,
            ),
        )


@dataclass(frozen=True)
class TextCharacterStats:
    """Alphabetic character counts used by the non-English text filter."""

    total_alpha_chars: int
    english_alpha_chars: int
    non_english_alpha_chars: int
    non_english_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_alpha_chars": self.total_alpha_chars,
            "english_alpha_chars": self.english_alpha_chars,
            "non_english_alpha_chars": self.non_english_alpha_chars,
            "non_english_ratio": self.non_english_ratio,
        }


@dataclass(frozen=True)
class FilterDecision:
    """Repository-level retention decision plus diagnostics for manifests."""

    retained: bool
    reason: str | None
    readme_path: Path | None
    readme_character_count: int
    text_character_stats: TextCharacterStats
    details: dict[str, Any]


@dataclass(frozen=True)
class RepositorySources:
    """Raw text/file sources assembled for one repository before preprocessing."""

    name_parts: tuple[str, ...]
    description: str
    readme: str
    wiki: str
    file_paths: tuple[str, ...]
    readme_path: Path | None
    wiki_path: Path | None
    file_list_path: Path | None
    source_statuses: dict[str, Any]


@dataclass(frozen=True)
class CountPassResult:
    """Corpus-level counts collected before label encoding."""

    raw_topic_counts: Counter[str]
    mapped_topic_counts: Counter[str]
    text_token_counts: Counter[str]
    file_name_token_counts: Counter[str]
    processed_records: int
    source_counts: Counter[str]
    resume_checkpoint_used: bool = False
    resume_complete_reused: bool = False
    resume_excluded_repository_configuration_changed: bool = False


@dataclass(frozen=True)
class LabelEncodingResult:
    """Manifest and counters returned by the final label-encoding pass."""

    manifest: dict[str, Any]
    counts: Counter[str]


class PreprocessingSourceTimeout(TimeoutError):
    """Raised when a single repository source exceeds the configured timeout."""

    def __init__(self, *, repo_label: str, source_name: str, timeout_seconds: float) -> None:
        self.repo_label = repo_label
        self.source_name = source_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Preprocessing timed out for {repo_label} source={source_name} "
            f"after {timeout_seconds:.1f}s"
        )


def _run_with_source_timeout(
    callback: Any,
    *,
    timeout_seconds: float,
    repo_label: str,
    source_name: str,
) -> Any:
    if timeout_seconds <= 0 or not _supports_signal_timeout():
        return callback()

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def handle_timeout(signum: int, frame: Any) -> None:
        raise PreprocessingSourceTimeout(
            repo_label=repo_label,
            source_name=source_name,
            timeout_seconds=float(timeout_seconds),
        )

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _supports_signal_timeout() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "ITIMER_REAL")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


def create_training_data(config: CreateTrainingDataConfig) -> dict[str, Any]:
    """Filter and preprocess an extraction run into encoded training records."""
    input_run = Path(config.input_run)
    output_dir = _resolve_output_dir(input_run, config.output_subdir)
    _validate_config(config)

    records_path = input_run / "repositories" / "repository_training_records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Repository training records not found: {records_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_dir)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if config.resume and _complete_create_training_outputs_exist(paths):
        manifest = _read_optional_json(paths["manifest"])
        if manifest is not None:
            log(
                "Resume enabled: complete create-training-data outputs already exist; "
                "returning existing manifest."
            )
            return manifest

    progress_interval = max(1, int(config.progress_interval))
    max_records = max(0, int(config.max_records))
    excluded_repository_keys = _load_excluded_repository_keys(config)
    source_manifest = _read_optional_json(input_run / "topic_training_extraction_manifest.json")
    log(f"Input run: {input_run}")
    log(f"Output directory: {output_dir}")
    log(
        "Config: "
        f"max_non_english_ratio={config.max_non_english_ratio}, "
        f"min_topic_repositories={config.min_topic_repositories}, "
        f"min_text_token_frequency={config.min_text_token_frequency}, "
        f"min_file_name_token_frequency={config.min_file_name_token_frequency}, "
        f"identifier_splitter={config.identifier_splitter}, "
        f"data_preparation_root={config.data_preparation_root}, "
        f"max_records={max_records or 'all'}, "
        f"excluded_repositories={len(excluded_repository_keys)}, "
        f"resume={config.resume}, "
        f"source_timeout_seconds={config.source_timeout_seconds}, "
        f"progress_interval={progress_interval}"
    )

    filter_counts = _get_or_create_filtered_records(
        input_run=input_run,
        records_path=records_path,
        paths=paths,
        max_non_english_ratio=float(config.max_non_english_ratio),
        progress_interval=progress_interval,
        max_records=max_records,
        excluded_repository_keys=excluded_repository_keys,
        resume=bool(config.resume),
    )

    topic_preprocessor = _make_preprocessor(
        config,
        allow_heuristic_file_splits=True,
    )
    count_result = _collect_preprocessing_counts(
        input_run=input_run,
        paths=paths,
        filtered_records_path=paths["filtered_records"],
        preprocessor=topic_preprocessor,
        progress_interval=progress_interval,
        excluded_repository_keys=excluded_repository_keys,
        resume=bool(config.resume),
        source_timeout_seconds=float(config.source_timeout_seconds),
    )

    final_labels = sorted(
        topic
        for topic, count in count_result.mapped_topic_counts.items()
        if count >= int(config.min_topic_repositories)
    )
    low_freq_topics = sorted(
        topic
        for topic, count in count_result.mapped_topic_counts.items()
        if count < int(config.min_topic_repositories)
    )
    preprocessing_artifacts = _build_preprocessing_artifacts(
        config=config,
        paths=paths,
        filtered_record_count=filter_counts["filtered_records"],
        label_universe=final_labels,
        text_token_counts=count_result.text_token_counts,
        file_name_token_counts=count_result.file_name_token_counts,
    )
    _write_preprocessing_artifacts(
        paths=paths,
        raw_topic_counts=count_result.raw_topic_counts,
        mapped_topic_counts=count_result.mapped_topic_counts,
        low_freq_topics=low_freq_topics,
        final_labels=final_labels,
        text_token_counts=count_result.text_token_counts,
        file_name_token_counts=count_result.file_name_token_counts,
        preprocessing_artifacts=preprocessing_artifacts,
    )
    log(
        "Vocabulary generated: "
        f"labels={len(final_labels)}, low_freq_topics={len(low_freq_topics)}, "
        f"allowed_text_tokens={len(preprocessing_artifacts['allowed_text_tokens'])}, "
        f"allowed_file_name_tokens={len(preprocessing_artifacts['allowed_file_name_tokens'])}"
    )

    final_preprocessor = _make_preprocessor(
        config,
        preprocessing_artifacts=preprocessing_artifacts,
    )
    final_counts = _write_preprocessed_records(
        input_run=input_run,
        paths=paths,
        topic_preprocessor=topic_preprocessor,
        text_preprocessor=final_preprocessor,
        label_universe=set(final_labels),
        progress_interval=progress_interval,
        excluded_repository_keys=excluded_repository_keys,
        resume=bool(config.resume),
        source_timeout_seconds=float(config.source_timeout_seconds),
    )
    encoding_result = _write_label_encoded_outputs(
        paths=paths,
        label_universe=final_labels,
    )

    total_excluded = (
        filter_counts["excluded_records"]
        + filter_counts["malformed_records"]
        + final_counts["excluded_no_final_label"]
        + final_counts["excluded_empty_prepared_text"]
        + final_counts["excluded_skipped_repository"]
        + final_counts["excluded_preprocessing_timeout"]
        + count_result.source_counts["preprocessing_timeout"]
    )
    manifest = {
        "schema_version": "topic_create_training_data_manifest_v2",
        "created_at_utc": _utc_now_z(),
        "input_run": str(input_run),
        "records_path": str(records_path),
        "output_dir": str(output_dir),
        "max_non_english_ratio": float(config.max_non_english_ratio),
        "progress_interval": progress_interval,
        "max_records": max_records,
        "is_validation_subset": max_records > 0,
        "resume": bool(config.resume),
        "source_timeout_seconds": float(config.source_timeout_seconds),
        "excluded_repository_skip_count": len(excluded_repository_keys),
        "excluded_repositories_file": (
            str(config.excluded_repositories_file)
            if config.excluded_repositories_file is not None
            else None
        ),
        "thresholds": {
            "min_topic_repositories": int(config.min_topic_repositories),
            "min_text_token_frequency": int(config.min_text_token_frequency),
            "min_file_name_token_frequency": int(config.min_file_name_token_frequency),
        },
        "data_preparation_root": str(config.data_preparation_root),
        "counts": {
            "input_records": int(filter_counts["input_records"]),
            "filtered_records": int(filter_counts["filtered_records"]),
            "preprocessed_records": int(final_counts["preprocessed_records"]),
            "retained_records": int(final_counts["preprocessed_records"]),
            "excluded_records": int(total_excluded),
            "excluded_filter_records": int(filter_counts["excluded_records"]),
            "excluded_no_readme": int(filter_counts["excluded_no_readme"]),
            "excluded_no_description": int(filter_counts["excluded_no_description"]),
            "excluded_non_english_ratio": int(filter_counts["excluded_non_english_ratio"]),
            "excluded_skipped_repository": int(filter_counts["excluded_skipped_repository"]),
            "final_excluded_skipped_repository": int(
                final_counts["excluded_skipped_repository"]
            ),
            "excluded_no_final_label": int(final_counts["excluded_no_final_label"]),
            "excluded_empty_prepared_text": int(final_counts["excluded_empty_prepared_text"]),
            "excluded_preprocessing_timeout": int(
                count_result.source_counts["preprocessing_timeout"]
                + final_counts["excluded_preprocessing_timeout"]
            ),
            "count_pass_excluded_preprocessing_timeout": int(
                count_result.source_counts["preprocessing_timeout"]
            ),
            "final_excluded_preprocessing_timeout": int(
                final_counts["excluded_preprocessing_timeout"]
            ),
            "malformed_records": int(filter_counts["malformed_records"]),
            "readme_path_failures": int(filter_counts["readme_path_failures"]),
            "raw_topic_count": len(count_result.raw_topic_counts),
            "mapped_topic_count": len(count_result.mapped_topic_counts),
            "low_freq_topic_count": len(low_freq_topics),
            "final_label_count": len(final_labels),
            "text_token_count": len(count_result.text_token_counts),
            "file_name_token_count": len(count_result.file_name_token_counts),
            "allowed_text_token_count": len(preprocessing_artifacts["allowed_text_tokens"]),
            "allowed_file_name_token_count": len(
                preprocessing_artifacts["allowed_file_name_tokens"]
            ),
            "encoded_records": int(encoding_result.counts["encoded_records"]),
            "label_vector_width": len(final_labels),
            "label_encoding_excluded_no_label": int(
                encoding_result.counts["excluded_no_label"]
            ),
            "label_encoding_excluded_empty_text": int(
                encoding_result.counts["excluded_empty_text"]
            ),
            "train_test_split_files_emitted": False,
            **{
                f"source_{key}": int(value)
                for key, value in sorted(count_result.source_counts.items())
            },
        },
        "source_extraction_manifest": source_manifest,
        "outputs": {name: str(path) for name, path in paths.items()},
        "preprocessing": final_preprocessor.manifest,
        "label_encoding": encoding_result.manifest,
        "resume_state": {
            "filtering_reused": bool(filter_counts["resume_reused_filtering"]),
            "count_checkpoint_used": bool(count_result.resume_checkpoint_used),
            "count_complete_reused": bool(count_result.resume_complete_reused),
            "count_checkpoint_excluded_repository_configuration_changed": bool(
                count_result.resume_excluded_repository_configuration_changed
            ),
        },
    }
    _write_json(paths["manifest"], manifest)
    log(
        "Create-training-data complete: "
        f"preprocessed={final_counts['preprocessed_records']}, "
        f"encoded={encoding_result.counts['encoded_records']}, "
        f"excluded={total_excluded}, labels={len(final_labels)}"
    )
    log(f"Retained repositories: {final_counts['preprocessed_records']}")
    return manifest


def _get_or_create_filtered_records(
    *,
    input_run: Path,
    records_path: Path,
    paths: dict[str, Path],
    max_non_english_ratio: float,
    progress_interval: int,
    max_records: int = 0,
    excluded_repository_keys: set[str] | None = None,
    resume: bool = True,
) -> Counter[str]:
    if resume and _filtering_outputs_exist(paths):
        counts = _load_filter_counts_from_outputs(paths)
        expected_input_records = _count_input_records(records_path, max_records=max_records)
        if counts["input_records"] >= expected_input_records:
            counts["resume_reused_filtering"] = 1
            log(
                "Resume enabled: reusing existing filtered records and skipping filtering: "
                f"filtered={counts['filtered_records']}, excluded={counts['excluded_records']}, "
                f"malformed={counts['malformed_records']}"
            )
            return counts
        log(
            "Resume enabled but existing filtered records are incomplete; rerunning filtering: "
            f"existing_input={counts['input_records']}, expected_input={expected_input_records}"
        )

    counts = _filter_repository_records(
        input_run=input_run,
        records_path=records_path,
        paths=paths,
        max_non_english_ratio=max_non_english_ratio,
        progress_interval=progress_interval,
        max_records=max_records,
        excluded_repository_keys=excluded_repository_keys,
    )
    counts["resume_reused_filtering"] = 0
    _write_json(
        paths["filtering_manifest"],
        {
            "schema_version": "topic_create_training_filtering_manifest_v1",
            "created_at_utc": _utc_now_z(),
            "records_path": str(records_path),
            "filtered_records": str(paths["filtered_records"]),
            "excluded_records": str(paths["excluded_records"]),
            "counts": {key: int(value) for key, value in sorted(counts.items())},
        },
    )
    return counts


def _filtering_outputs_exist(paths: dict[str, Path]) -> bool:
    return paths["filtered_records"].is_file() and paths["excluded_records"].is_file()


def _load_filter_counts_from_outputs(paths: dict[str, Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts["filtered_records"] = _count_jsonl_records(paths["filtered_records"])
    for record in _iter_jsonl_records(paths["excluded_records"]):
        if record.get("excluded_stage") != "filtering":
            continue
        reason = str(record.get("excluded_reason") or "").strip()
        if reason == "malformed_record":
            counts["malformed_records"] += 1
        else:
            counts["excluded_records"] += 1
            if reason:
                counts[f"excluded_{reason}"] += 1
        details = record.get("details")
        if isinstance(details, dict) and details.get("readme_path_resolution_failed"):
            counts["readme_path_failures"] += 1
    counts["input_records"] = (
        counts["filtered_records"] + counts["excluded_records"] + counts["malformed_records"]
    )
    return counts


def _count_input_records(records_path: Path, *, max_records: int = 0) -> int:
    count = 0
    for _item in _iter_jsonl_with_malformed(records_path):
        if max_records > 0 and count >= max_records:
            break
        count += 1
    return count


def _filter_repository_records(
    *,
    input_run: Path,
    records_path: Path,
    paths: dict[str, Path],
    max_non_english_ratio: float,
    progress_interval: int,
    max_records: int = 0,
    excluded_repository_keys: set[str] | None = None,
) -> Counter[str]:
    log("Filtering repositories.")
    counts: Counter[str] = Counter()
    excluded_repository_keys = excluded_repository_keys or set()
    with (
        paths["filtered_records"].open("w", encoding="utf-8", newline="\n") as filtered,
        paths["excluded_records"].open("w", encoding="utf-8", newline="\n") as excluded,
    ):
        for item in _iter_jsonl_with_malformed(records_path):
            if max_records > 0 and counts["input_records"] >= max_records:
                log(f"Filtering stopped at validation max_records={max_records}.")
                break
            counts["input_records"] += 1
            if item.get("malformed"):
                counts["malformed_records"] += 1
                _write_jsonl(
                    excluded,
                    {
                        "schema_version": "topic_create_training_excluded_record_v1",
                        "excluded_stage": "filtering",
                        "excluded_reason": "malformed_record",
                        "line_number": item.get("line_number"),
                        "details": {"raw_line_prefix": item.get("raw_line_prefix")},
                    },
                )
                _log_filter_progress(counts, progress_interval)
                continue

            record = item["record"]
            if _record_matches_excluded_repository(record, excluded_repository_keys):
                counts["excluded_records"] += 1
                counts["excluded_skipped_repository"] += 1
                _write_jsonl(
                    excluded,
                    {
                        "schema_version": "topic_create_training_excluded_record_v1",
                        "excluded_stage": "filtering",
                        "repository_identity_key": record.get("repository_identity_key"),
                        "repository_full_name": record.get("repository_full_name"),
                        "excluded_reason": "skipped_repository",
                        "details": {"matched_skip_list": True},
                        "text_character_stats": TextCharacterStats(0, 0, 0, 0.0).to_dict(),
                        "record": record,
                    },
                )
                _log_filter_progress(counts, progress_interval)
                continue

            decision = _filter_record(
                input_run=input_run,
                record=record,
                max_non_english_ratio=max_non_english_ratio,
            )
            if decision.retained:
                counts["filtered_records"] += 1
                output_record = dict(record)
                output_record["create_training_data_status"] = "filtered"
                output_record["readme_cache_path_resolved"] = (
                    str(decision.readme_path) if decision.readme_path is not None else None
                )
                output_record["readme_character_count"] = decision.readme_character_count
                output_record["text_character_stats"] = decision.text_character_stats.to_dict()
                _write_jsonl(filtered, output_record)
            else:
                counts["excluded_records"] += 1
                if decision.reason:
                    counts[f"excluded_{decision.reason}"] += 1
                if decision.details.get("readme_path_resolution_failed"):
                    counts["readme_path_failures"] += 1
                _write_jsonl(
                    excluded,
                    {
                        "schema_version": "topic_create_training_excluded_record_v1",
                        "excluded_stage": "filtering",
                        "repository_identity_key": record.get("repository_identity_key"),
                        "repository_full_name": record.get("repository_full_name"),
                        "excluded_reason": decision.reason,
                        "details": decision.details,
                        "text_character_stats": decision.text_character_stats.to_dict(),
                        "record": record,
                    },
                )
            _log_filter_progress(counts, progress_interval)

    log(
        "Filtering complete: "
        f"retained={counts['filtered_records']}, "
        f"excluded={counts['excluded_records'] + counts['malformed_records']}, "
        f"no_readme={counts['excluded_no_readme']}, "
        f"no_description={counts['excluded_no_description']}, "
        f"non_english_ratio={counts['excluded_non_english_ratio']}, "
        f"skipped_repository={counts['excluded_skipped_repository']}, "
        f"malformed={counts['malformed_records']}"
    )
    return counts


def _collect_preprocessing_counts(
    *,
    input_run: Path,
    paths: dict[str, Path],
    filtered_records_path: Path,
    preprocessor: SoftwareTagDataPreprocessor,
    progress_interval: int,
    excluded_repository_keys: set[str] | None = None,
    resume: bool = True,
    source_timeout_seconds: float = DEFAULT_SOURCE_TIMEOUT_SECONDS,
) -> CountPassResult:
    log("Preprocessing count pass started.")
    excluded_repository_keys = excluded_repository_keys or set()
    excluded_keys_sha256 = _sha256_lines(sorted(excluded_repository_keys))
    resume_checkpoint_used = False
    resume_complete_reused = False
    filtered_records_seen = 0
    checkpoint = (
        _load_count_checkpoint(
            paths["preprocessing_count_manifest"],
            excluded_keys_sha256=excluded_keys_sha256,
        )
        if resume
        else None
    )
    if checkpoint is not None and checkpoint.get("status") == "complete":
        log(
            "Resume enabled: reusing completed preprocessing count pass: "
            f"processed={checkpoint['processed_records']}, "
            f"filtered_records_seen={checkpoint['filtered_records_seen']}"
        )
        return CountPassResult(
            raw_topic_counts=_counter_from_json(checkpoint.get("raw_topic_counts")),
            mapped_topic_counts=_counter_from_json(checkpoint.get("mapped_topic_counts")),
            text_token_counts=_counter_from_json(checkpoint.get("text_token_counts")),
            file_name_token_counts=_counter_from_json(
                checkpoint.get("file_name_token_counts")
            ),
            processed_records=int(checkpoint.get("processed_records") or 0),
            source_counts=_counter_from_json(checkpoint.get("source_counts")),
            resume_checkpoint_used=False,
            resume_complete_reused=True,
            resume_excluded_repository_configuration_changed=bool(
                checkpoint.get("_excluded_repository_configuration_changed")
            ),
        )

    checkpoint = (
        _load_count_checkpoint(
            paths["preprocessing_count_checkpoint"],
            excluded_keys_sha256=excluded_keys_sha256,
        )
        if resume
        else None
    )
    if checkpoint is not None:
        raw_topic_counts = _counter_from_json(checkpoint.get("raw_topic_counts"))
        mapped_topic_counts = _counter_from_json(checkpoint.get("mapped_topic_counts"))
        text_token_counts = _counter_from_json(checkpoint.get("text_token_counts"))
        file_name_token_counts = _counter_from_json(checkpoint.get("file_name_token_counts"))
        source_counts = _counter_from_json(checkpoint.get("source_counts"))
        processed = int(checkpoint.get("processed_records") or 0)
        filtered_records_seen = int(checkpoint.get("filtered_records_seen") or 0)
        resume_checkpoint_used = True
        resume_excluded_repository_configuration_changed = bool(
            checkpoint.get("_excluded_repository_configuration_changed")
        )
        log(
            "Resume enabled: continuing preprocessing count pass from checkpoint: "
            f"processed={processed}, filtered_records_seen={filtered_records_seen}"
        )
    else:
        raw_topic_counts = Counter()
        mapped_topic_counts = Counter()
        text_token_counts = Counter()
        file_name_token_counts = Counter()
        source_counts = Counter()
        processed = 0
        resume_excluded_repository_configuration_changed = False

    count_started_at = perf_counter()
    last_heartbeat_at = count_started_at
    diagnostic_interval = max(
        1,
        min(progress_interval, COUNT_PASS_DIAGNOSTIC_RECORD_INTERVAL),
    )
    checkpoint_interval = max(progress_interval, COUNT_PASS_CHECKPOINT_RECORD_INTERVAL)

    def update_tokens(
        *,
        source_name: str,
        input_size: int,
        counter: Counter[str],
        build_tokens: Any,
        repo_label: str,
    ) -> None:
        source_started_at = perf_counter()
        tokens = _run_with_source_timeout(
            build_tokens,
            timeout_seconds=source_timeout_seconds,
            repo_label=repo_label,
            source_name=source_name,
        )
        counter.update(tokens)
        elapsed = perf_counter() - source_started_at
        if elapsed >= COUNT_PASS_SLOW_SOURCE_SECONDS:
            log(
                "Preprocessing count slow source: "
                f"processed={processed}, repo={repo_label}, source={source_name}, "
                f"seconds={elapsed:.1f}, input_size={input_size}, tokens={len(tokens)}"
            )

    for record_index, record in enumerate(_iter_jsonl_records(filtered_records_path), start=1):
        if record_index <= filtered_records_seen:
            continue
        filtered_records_seen = record_index
        if _record_matches_excluded_repository(record, excluded_repository_keys):
            source_counts["skipped_repository"] += 1
            if filtered_records_seen % checkpoint_interval == 0:
                _write_count_checkpoint(
                    paths["preprocessing_count_checkpoint"],
                    status="in_progress",
                    filtered_records_path=filtered_records_path,
                    filtered_records_seen=filtered_records_seen,
                    processed_records=processed,
                    raw_topic_counts=raw_topic_counts,
                    mapped_topic_counts=mapped_topic_counts,
                    text_token_counts=text_token_counts,
                    file_name_token_counts=file_name_token_counts,
                    source_counts=source_counts,
                    excluded_keys_sha256=excluded_keys_sha256,
                )
            continue
        processed += 1
        repo_label = _repository_log_label(record)
        log_record_diagnostic = (
            processed <= 10 or processed % diagnostic_interval == 0
        )
        if log_record_diagnostic:
            log(
                "Preprocessing count record started: "
                f"processed={processed}, repo={repo_label}"
            )
        raw_topics = _raw_topics(record)
        raw_topic_counts.update(raw_topics)
        mapped_topics = preprocessor.prepare_topics(raw_topics)
        mapped_topic_counts.update(mapped_topics)

        sources = _load_repository_sources(input_run, record)
        if log_record_diagnostic:
            log(
                "Preprocessing count record sources: "
                f"processed={processed}, repo={repo_label}, "
                f"description_chars={len(sources.description)}, "
                f"readme_chars={len(sources.readme)}, wiki_chars={len(sources.wiki)}, "
                f"file_paths={len(sources.file_paths)}"
            )
        try:
            update_tokens(
                source_name="description",
                input_size=len(sources.description),
                counter=text_token_counts,
                build_tokens=lambda: preprocessor.prepare_text_tokens(
                    sources.description,
                    limit=preprocessor.source_token_limits["description"],
                ),
                repo_label=repo_label,
            )
            update_tokens(
                source_name="readme",
                input_size=len(sources.readme),
                counter=text_token_counts,
                build_tokens=lambda: preprocessor.prepare_text_tokens(
                    sources.readme,
                    limit=preprocessor.source_token_limits["readme"],
                ),
                repo_label=repo_label,
            )
            update_tokens(
                source_name="wiki",
                input_size=len(sources.wiki),
                counter=text_token_counts,
                build_tokens=lambda: preprocessor.prepare_text_tokens(
                    sources.wiki,
                    limit=preprocessor.source_token_limits["wiki"],
                ),
                repo_label=repo_label,
            )
            update_tokens(
                source_name="project_name",
                input_size=sum(len(part) for part in sources.name_parts),
                counter=file_name_token_counts,
                build_tokens=lambda: preprocessor.prepare_project_name_tokens(
                    sources.name_parts,
                    limit=preprocessor.source_token_limits["name"],
                ),
                repo_label=repo_label,
            )
            update_tokens(
                source_name="file_paths",
                input_size=len(sources.file_paths),
                counter=file_name_token_counts,
                build_tokens=lambda: preprocessor.prepare_file_name_tokens(
                    sources.file_paths,
                    limit=preprocessor.source_token_limits["file_names"],
                ),
                repo_label=repo_label,
            )
        except PreprocessingSourceTimeout as exc:
            _subtract_counter(raw_topic_counts, raw_topics)
            _subtract_counter(mapped_topic_counts, mapped_topics)
            source_counts["preprocessing_timeout"] += 1
            log(
                "Preprocessing count timeout: "
                f"processed={processed}, repo={repo_label}, source={exc.source_name}, "
                f"timeout_seconds={exc.timeout_seconds:.1f}"
            )
            _append_final_exclusion(
                paths,
                record,
                "preprocessing_timeout",
                mapped_topics,
                details={
                    "pass": "count",
                    "source": exc.source_name,
                    "timeout_seconds": exc.timeout_seconds,
                },
            )
            if filtered_records_seen % checkpoint_interval == 0:
                _write_count_checkpoint(
                    paths["preprocessing_count_checkpoint"],
                    status="in_progress",
                    filtered_records_path=filtered_records_path,
                    filtered_records_seen=filtered_records_seen,
                    processed_records=processed,
                    raw_topic_counts=raw_topic_counts,
                    mapped_topic_counts=mapped_topic_counts,
                    text_token_counts=text_token_counts,
                    file_name_token_counts=file_name_token_counts,
                    source_counts=source_counts,
                    excluded_keys_sha256=excluded_keys_sha256,
                )
            continue
        _update_source_counts(source_counts, sources)
        if processed % progress_interval == 0:
            log(
                "Preprocessing count progress: "
                f"processed={processed}, mapped_topics={len(mapped_topic_counts)}, "
                f"text_tokens={len(text_token_counts)}, "
                f"file_name_tokens={len(file_name_token_counts)}"
            )
        now = perf_counter()
        if now - last_heartbeat_at >= COUNT_PASS_HEARTBEAT_SECONDS:
            elapsed = now - count_started_at
            records_per_hour = processed / max(elapsed, 0.001) * 3600.0
            log(
                "Preprocessing count heartbeat: "
                f"processed={processed}, repo={repo_label}, "
                f"elapsed_seconds={elapsed:.1f}, rate={records_per_hour:.1f} repos/hour, "
                f"mapped_topics={len(mapped_topic_counts)}, "
                f"text_tokens={len(text_token_counts)}, "
                f"file_name_tokens={len(file_name_token_counts)}"
            )
            last_heartbeat_at = now
        if filtered_records_seen % checkpoint_interval == 0:
            _write_count_checkpoint(
                paths["preprocessing_count_checkpoint"],
                status="in_progress",
                filtered_records_path=filtered_records_path,
                filtered_records_seen=filtered_records_seen,
                processed_records=processed,
                raw_topic_counts=raw_topic_counts,
                mapped_topic_counts=mapped_topic_counts,
                text_token_counts=text_token_counts,
                file_name_token_counts=file_name_token_counts,
                source_counts=source_counts,
                excluded_keys_sha256=excluded_keys_sha256,
            )

    log(
        "Preprocessing count pass complete: "
        f"processed={processed}, mapped_topics={len(mapped_topic_counts)}, "
        f"text_tokens={len(text_token_counts)}, file_name_tokens={len(file_name_token_counts)}"
    )
    _write_count_checkpoint(
        paths["preprocessing_count_checkpoint"],
        status="complete",
        filtered_records_path=filtered_records_path,
        filtered_records_seen=filtered_records_seen,
        processed_records=processed,
        raw_topic_counts=raw_topic_counts,
        mapped_topic_counts=mapped_topic_counts,
        text_token_counts=text_token_counts,
        file_name_token_counts=file_name_token_counts,
        source_counts=source_counts,
        excluded_keys_sha256=excluded_keys_sha256,
    )
    _write_count_checkpoint(
        paths["preprocessing_count_manifest"],
        status="complete",
        filtered_records_path=filtered_records_path,
        filtered_records_seen=filtered_records_seen,
        processed_records=processed,
        raw_topic_counts=raw_topic_counts,
        mapped_topic_counts=mapped_topic_counts,
        text_token_counts=text_token_counts,
        file_name_token_counts=file_name_token_counts,
        source_counts=source_counts,
        excluded_keys_sha256=excluded_keys_sha256,
    )
    return CountPassResult(
        raw_topic_counts=raw_topic_counts,
        mapped_topic_counts=mapped_topic_counts,
        text_token_counts=text_token_counts,
        file_name_token_counts=file_name_token_counts,
        processed_records=processed,
        source_counts=source_counts,
        resume_checkpoint_used=resume_checkpoint_used,
        resume_complete_reused=resume_complete_reused,
        resume_excluded_repository_configuration_changed=(
            resume_excluded_repository_configuration_changed
        ),
    )


def _write_preprocessed_records(
    *,
    input_run: Path,
    paths: dict[str, Path],
    topic_preprocessor: SoftwareTagDataPreprocessor,
    text_preprocessor: SoftwareTagDataPreprocessor,
    label_universe: set[str],
    progress_interval: int,
    excluded_repository_keys: set[str] | None = None,
    resume: bool = True,
    source_timeout_seconds: float = DEFAULT_SOURCE_TIMEOUT_SECONDS,
) -> Counter[str]:
    log("Final preprocessing pass started.")
    counts: Counter[str] = Counter()
    excluded_repository_keys = excluded_repository_keys or set()
    processed_identity_keys = (
        _load_final_preprocessing_identity_keys(paths) if resume else set()
    )
    if processed_identity_keys:
        counts["resume_skipped_existing_records"] = len(processed_identity_keys)
        log(
            "Resume enabled: final preprocessing will skip existing output records: "
            f"{len(processed_identity_keys)}"
        )
    prepared_mode = "a" if resume and paths["preprocessed_records"].exists() else "w"
    with (
        paths["preprocessed_records"].open(prepared_mode, encoding="utf-8", newline="\n") as prepared,
        paths["excluded_records"].open("a", encoding="utf-8", newline="\n") as excluded,
    ):
        for record in _iter_jsonl_records(paths["filtered_records"]):
            identity_keys = _repository_identity_keys_for_resume(record)
            if identity_keys & processed_identity_keys:
                counts["skipped_existing_records"] += 1
                continue
            if _record_matches_excluded_repository(record, excluded_repository_keys):
                counts["processed_filtered_records"] += 1
                counts["excluded_skipped_repository"] += 1
                _write_final_exclusion(excluded, record, "skipped_repository", [])
                processed_identity_keys.update(identity_keys)
                _log_final_progress(counts, progress_interval)
                continue
            counts["processed_filtered_records"] += 1
            mapped_topics = topic_preprocessor.prepare_topics(_raw_topics(record))
            final_topics = sorted(set(mapped_topics) & label_universe)
            if not final_topics:
                counts["excluded_no_final_label"] += 1
                _write_final_exclusion(excluded, record, "no_final_label", mapped_topics)
                processed_identity_keys.update(identity_keys)
                _log_final_progress(counts, progress_interval)
                continue

            sources = _load_repository_sources(input_run, record)
            try:
                prepared_text = _run_with_source_timeout(
                    lambda: text_preprocessor.prepare_repository_text(
                        name_parts=sources.name_parts,
                        description=sources.description,
                        readme=sources.readme,
                        wiki=sources.wiki,
                        file_paths=sources.file_paths,
                    ),
                    timeout_seconds=source_timeout_seconds,
                    repo_label=_repository_log_label(record),
                    source_name="repository_text",
                )
            except PreprocessingSourceTimeout as exc:
                counts["excluded_preprocessing_timeout"] += 1
                log(
                    "Final preprocessing timeout: "
                    f"processed={counts['processed_filtered_records']}, "
                    f"repo={_repository_log_label(record)}, source={exc.source_name}, "
                    f"timeout_seconds={exc.timeout_seconds:.1f}"
                )
                _write_final_exclusion(
                    excluded,
                    record,
                    "preprocessing_timeout",
                    mapped_topics,
                    details={
                        "pass": "final",
                        "source": exc.source_name,
                        "timeout_seconds": exc.timeout_seconds,
                    },
                )
                processed_identity_keys.update(identity_keys)
                _log_final_progress(counts, progress_interval)
                continue
            if not prepared_text.text.strip():
                counts["excluded_empty_prepared_text"] += 1
                _write_final_exclusion(excluded, record, "empty_prepared_text", mapped_topics)
                processed_identity_keys.update(identity_keys)
                _log_final_progress(counts, progress_interval)
                continue

            output_record = dict(record)
            output_record.update(
                {
                    "schema_version": "topic_training_preprocessed_repository_record_v1",
                    "data_preparation_status": "prepared",
                    "mapped_topics": list(mapped_topics),
                    "prepared_topics": final_topics,
                    "prepared_text": prepared_text.text,
                    "token_counts": prepared_text.token_counts,
                    "data_preparation": prepared_text.data_preparation,
                    "source_statuses": sources.source_statuses,
                    "wiki_cache_path_resolved": (
                        str(sources.wiki_path) if sources.wiki_path else None
                    ),
                    "file_list_path_resolved": (
                        str(sources.file_list_path) if sources.file_list_path else None
                    ),
                }
            )
            counts["preprocessed_records"] += 1
            _write_jsonl(prepared, output_record)
            processed_identity_keys.update(identity_keys)
            _log_final_progress(counts, progress_interval)

    counts["preprocessed_records"] = _count_jsonl_records(paths["preprocessed_records"])
    log(
        "Final preprocessing pass complete: "
        f"preprocessed={counts['preprocessed_records']}, "
        f"excluded_no_final_label={counts['excluded_no_final_label']}, "
        f"excluded_empty_prepared_text={counts['excluded_empty_prepared_text']}, "
        f"excluded_skipped_repository={counts['excluded_skipped_repository']}, "
        f"skipped_existing={counts['skipped_existing_records']}"
    )
    return counts


def _write_label_encoded_outputs(
    *,
    paths: dict[str, Path],
    label_universe: list[str],
) -> LabelEncodingResult:
    log("Label encoding pass started.")
    label_to_index = {label: index for index, label in enumerate(label_universe)}
    label_universe_sha256 = _sha256_lines(label_universe)
    counts: Counter[str] = Counter()
    csv_fieldnames = [*label_universe, "labels", "text"]
    with (
        paths["encoded_records"].open("w", encoding="utf-8", newline="\n") as encoded,
        paths["topic_training_dataset_csv"].open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_handle,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        for record in _iter_jsonl_records(paths["preprocessed_records"]):
            text = str(record.get("prepared_text") or "").strip()
            if not text:
                counts["excluded_empty_text"] += 1
                continue

            topics = sorted(
                {
                    str(topic).strip()
                    for topic in (record.get("prepared_topics") or ())
                    if str(topic).strip() in label_to_index
                }
            )
            if not topics:
                counts["excluded_no_label"] += 1
                continue

            active_topics = set(topics)
            label_indices = [
                index
                for index, label in enumerate(label_universe)
                if label in active_topics
            ]
            active_indices = set(label_indices)
            label_vector = [
                1 if index in active_indices else 0
                for index in range(len(label_universe))
            ]
            labels_text = "[" + " ".join(str(bit) for bit in label_vector) + "]"
            output_record = dict(record)
            output_record.update(
                {
                    "schema_version": "topic_training_encoded_repository_record_v1",
                    "label_vector": label_vector,
                    "label_indices": label_indices,
                    "label_count": len(label_indices),
                    "label_universe_size": len(label_universe),
                    "label_universe_sha256": label_universe_sha256,
                }
            )
            _write_jsonl(encoded, output_record)
            csv_payload = {
                label: label_vector[index]
                for index, label in enumerate(label_universe)
            }
            csv_payload["labels"] = labels_text
            csv_payload["text"] = text
            writer.writerow(csv_payload)
            counts["encoded_records"] += 1

    manifest = {
        "schema_version": "topic_label_encoding_manifest_v1",
        "created_at_utc": _utc_now_z(),
        "source_preprocessed_records": str(paths["preprocessed_records"]),
        "encoded_repository_records": str(paths["encoded_records"]),
        "topic_training_dataset_csv": str(paths["topic_training_dataset_csv"]),
        "label_universe_path": str(paths["topic_label_universe"]),
        "label_universe_size": len(label_universe),
        "label_universe_sha256": label_universe_sha256,
        "encoded_records": int(counts["encoded_records"]),
        "excluded_no_label": int(counts["excluded_no_label"]),
        "excluded_empty_text": int(counts["excluded_empty_text"]),
        "csv_columns": csv_fieldnames,
        "csv_label_columns": list(label_universe),
        "csv_labels_column_format": "[0 1 0]",
        "train_test_split_files_emitted": False,
    }
    _write_json(paths["label_encoding_manifest"], manifest)
    log(
        "Label encoding pass complete: "
        f"encoded={counts['encoded_records']}, "
        f"excluded_no_label={counts['excluded_no_label']}, "
        f"excluded_empty_text={counts['excluded_empty_text']}"
    )
    return LabelEncodingResult(manifest=manifest, counts=counts)


def _write_final_exclusion(
    handle: Any,
    record: dict[str, Any],
    reason: str,
    mapped_topics: tuple[str, ...],
    *,
    details: dict[str, Any] | None = None,
) -> None:
    _write_jsonl(
        handle,
        {
            "schema_version": "topic_create_training_excluded_record_v1",
            "excluded_stage": "preprocessing",
            "repository_identity_key": record.get("repository_identity_key"),
            "repository_full_name": record.get("repository_full_name"),
            "excluded_reason": reason,
            "mapped_topics": list(mapped_topics),
            "details": details or {},
            "record": record,
        },
    )


def _append_final_exclusion(
    paths: dict[str, Path],
    record: dict[str, Any],
    reason: str,
    mapped_topics: tuple[str, ...],
    *,
    details: dict[str, Any] | None = None,
) -> None:
    with paths["excluded_records"].open("a", encoding="utf-8", newline="\n") as handle:
        _write_final_exclusion(
            handle,
            record,
            reason,
            mapped_topics,
            details=details,
        )


def _make_preprocessor(
    config: CreateTrainingDataConfig,
    *,
    preprocessing_artifacts: dict[str, Any] | None = None,
    allow_heuristic_file_splits: bool = False,
) -> SoftwareTagDataPreprocessor:
    data_preparation_root = Path(config.data_preparation_root)
    return SoftwareTagDataPreprocessor(
        rules_dir=data_preparation_root / "rules",
        lists_dir=data_preparation_root / "lists",
        identifier_splitter_mode=config.identifier_splitter,
        preprocessing_artifacts=preprocessing_artifacts,
        allow_heuristic_file_splits=allow_heuristic_file_splits,
        skipped_topic_rule_names=SKIPPED_STATIC_LOW_FREQ_RULES,
    )


def _build_preprocessing_artifacts(
    *,
    config: CreateTrainingDataConfig,
    paths: dict[str, Path],
    filtered_record_count: int,
    label_universe: list[str],
    text_token_counts: Counter[str],
    file_name_token_counts: Counter[str],
) -> dict[str, Any]:
    allowed_text_tokens = sorted(
        token
        for token, count in text_token_counts.items()
        if count >= int(config.min_text_token_frequency)
    )
    allowed_file_name_tokens = sorted(
        token
        for token, count in file_name_token_counts.items()
        if count >= int(config.min_file_name_token_frequency)
    )
    allowed_runtime_tokens = sorted(set(allowed_text_tokens) | set(allowed_file_name_tokens))
    file_name_informative_split_tokens = sorted(
        token
        for token in DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS
        if token in set(allowed_file_name_tokens)
    )
    return {
        "schema_version": "topic_preprocessing_artifacts_v2",
        "source": "topic_create_training_data_dataset_local_raw_frequency_artifacts",
        "source_files": {
            "filtered_repository_records": str(paths["filtered_records"]),
            "text_token_counts": str(paths["text_token_counts"]),
            "file_name_token_counts": str(paths["file_name_token_counts"]),
            "topic_label_universe": str(paths["topic_label_universe"]),
            "low_freq_topics": str(paths["low_freq_topics"]),
        },
        "data_preparation_profile": _data_preparation_profile_manifest(
            config.data_preparation_root
        ),
        "source_row_counts": {
            "filtered_repository_records": int(filtered_record_count),
        },
        "label_universe": label_universe,
        "label_count": len(label_universe),
        "allowed_runtime_tokens": allowed_runtime_tokens,
        "allowed_runtime_token_count": len(allowed_runtime_tokens),
        "allowed_runtime_tokens_sha256": _sha256_lines(allowed_runtime_tokens),
        "allowed_text_tokens": allowed_text_tokens,
        "allowed_text_token_count": len(allowed_text_tokens),
        "allowed_text_tokens_sha256": _sha256_lines(allowed_text_tokens),
        "allowed_file_name_tokens": allowed_file_name_tokens,
        "allowed_file_name_token_count": len(allowed_file_name_tokens),
        "allowed_file_name_tokens_sha256": _sha256_lines(allowed_file_name_tokens),
        "file_name_informative_split_tokens": file_name_informative_split_tokens,
        "file_name_informative_split_token_count": len(file_name_informative_split_tokens),
        "file_name_informative_split_tokens_source": (
            "dataset_local_file_name_vocabulary_intersection_with_documented_examples"
        ),
        "source_token_policy": {
            "text_token_sources": ["description", "readme", "wiki"],
            "file_name_token_sources": [
                "repository_owner",
                "repository_name",
                "repository_file_paths",
            ],
            "project_name_tokens_use_file_name_vocabulary": True,
            "project_name_tokens_share_file_name_frequency_threshold": True,
            "vocabulary_count_pass_uses_final_source_token_limits": True,
            "vocabulary_count_source_token_limits": {
                "name": 10,
                "description": 50,
                "readme": 400,
                "wiki": 100,
                "file_names": 100,
            },
            "limited_source_text_window": {
                "enabled": True,
                "minimum_characters": 4096,
                "characters_per_requested_token": 200,
            },
            "file_name_generic_stop_tokens_source": "File names_confusing_tokens.txt",
            "file_name_informative_stop_token_exceptions": sorted(
                DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS
            ),
        },
        "frequency_policy": {
            "dataset_local_min_topic_repositories": int(config.min_topic_repositories),
            "paper_text_min_frequency": int(config.min_text_token_frequency),
            "paper_file_name_min_frequency": int(config.min_file_name_token_frequency),
            "project_name_tokens_use_file_name_frequency": True,
            "raw_text_token_counts_provided": False,
            "raw_file_name_token_counts_provided": False,
            "paper_equivalence": "dataset_local_final_document_frequency_artifacts",
            "runtime_filter": (
                "Filter description/README/wiki tokens to the dataset-local text-source "
                "vocabulary and project/source file-name tokens to the dataset-local "
                "file-name vocabulary."
            ),
        },
        "separate_source_frequency_filters_available": True,
        "raw_corpus_frequency_artifacts_available": True,
    }


def _write_preprocessing_artifacts(
    *,
    paths: dict[str, Path],
    raw_topic_counts: Counter[str],
    mapped_topic_counts: Counter[str],
    low_freq_topics: list[str],
    final_labels: list[str],
    text_token_counts: Counter[str],
    file_name_token_counts: Counter[str],
    preprocessing_artifacts: dict[str, Any],
) -> None:
    _write_json(paths["raw_topic_universe"], sorted(raw_topic_counts))
    _write_json(paths["topic_label_universe"], final_labels)
    _write_topic_counts_csv(paths["candidate_topics"], mapped_topic_counts, set(final_labels))
    _write_topic_counts_csv(paths["mapped_topic_counts"], mapped_topic_counts, set(final_labels))
    _write_low_freq_topics(paths["low_freq_topics"], low_freq_topics)
    _write_token_counts(paths["text_token_counts"], text_token_counts)
    _write_token_counts(paths["file_name_token_counts"], file_name_token_counts)
    _write_json(paths["preprocessing_artifacts"], preprocessing_artifacts)
    _write_json(
        paths["train_test_placeholder"],
        {
            "status": "placeholder",
            "intended_approach": (
                "After final formatting and label encoding, create rerunnable train/test "
                "splits from the encoded full dataset."
            ),
            "encoded_records_input": str(paths["encoded_records"]),
            "full_dataset_csv_input": str(paths["topic_training_dataset_csv"]),
            "train_test_csv_files_emitted": False,
        },
    )


def _load_repository_sources(input_run: Path, record: dict[str, Any]) -> RepositorySources:
    owner = str(record.get("repository_owner") or "").strip()
    name = str(record.get("repository_name") or "").strip()
    description = str(record.get("description") or "")
    readme_path = _resolve_readme_path(input_run, record)
    wiki_path = _resolve_wiki_path(input_run, record)
    file_list_path = _resolve_file_list_path(input_run, record)
    readme_text = _read_text(readme_path)
    wiki_text = _read_text(wiki_path)
    file_paths = tuple(_read_file_paths(file_list_path))
    return RepositorySources(
        name_parts=tuple(part for part in (owner, name) if part),
        description=description,
        readme=readme_text,
        wiki=wiki_text,
        file_paths=file_paths,
        readme_path=readme_path,
        wiki_path=wiki_path,
        file_list_path=file_list_path,
        source_statuses={
            "readme_status": record.get("readme_status"),
            "wiki_status": record.get("wiki_status"),
            "file_list_status": record.get("file_list_status"),
            "readme_loaded": bool(readme_text.strip()),
            "wiki_loaded": bool(wiki_text.strip()),
            "file_paths_loaded": len(file_paths),
        },
    )


def _filter_record(
    *,
    input_run: Path,
    record: dict[str, Any],
    max_non_english_ratio: float,
) -> FilterDecision:
    empty_stats = TextCharacterStats(0, 0, 0, 0.0)
    readme_status = str(record.get("readme_status") or "").strip().lower()
    if readme_status != "fetched":
        return FilterDecision(
            retained=False,
            reason="no_readme",
            readme_path=None,
            readme_character_count=0,
            text_character_stats=empty_stats,
            details={"readme_status": record.get("readme_status")},
        )

    readme_path = _resolve_readme_path(input_run, record)
    if readme_path is None:
        return FilterDecision(
            retained=False,
            reason="no_readme",
            readme_path=None,
            readme_character_count=0,
            text_character_stats=empty_stats,
            details={
                "readme_status": record.get("readme_status"),
                "readme_cache_path": record.get("readme_cache_path"),
                "readme_path_resolution_failed": True,
            },
        )

    readme_text = readme_path.read_text(encoding="utf-8", errors="replace")
    if not readme_text.strip():
        return FilterDecision(
            retained=False,
            reason="no_readme",
            readme_path=readme_path,
            readme_character_count=len(readme_text),
            text_character_stats=empty_stats,
            details={
                "readme_status": record.get("readme_status"),
                "readme_cache_path": str(readme_path),
                "readme_empty": True,
            },
        )

    description = str(record.get("description") or "").strip()
    if not description:
        return FilterDecision(
            retained=False,
            reason="no_description",
            readme_path=readme_path,
            readme_character_count=len(readme_text),
            text_character_stats=empty_stats,
            details={
                "readme_status": record.get("readme_status"),
                "readme_cache_path": str(readme_path),
                "description_empty": True,
            },
        )

    combined_text = f"{description}\n{readme_text}"
    stats = _text_character_stats(combined_text)
    if stats.non_english_ratio > max_non_english_ratio:
        return FilterDecision(
            retained=False,
            reason="non_english_ratio",
            readme_path=readme_path,
            readme_character_count=len(readme_text),
            text_character_stats=stats,
            details={
                "readme_status": record.get("readme_status"),
                "readme_cache_path": str(readme_path),
                "max_non_english_ratio": max_non_english_ratio,
            },
        )

    return FilterDecision(
        retained=True,
        reason=None,
        readme_path=readme_path,
        readme_character_count=len(readme_text),
        text_character_stats=stats,
        details={"readme_status": record.get("readme_status")},
    )


def _resolve_readme_path(input_run: Path, record: dict[str, Any]) -> Path | None:
    return _resolve_cache_path(
        input_run=input_run,
        record=record,
        path_key="readme_cache_path",
        marker="readme-cache",
        fallback_file_name="readme_text.txt",
    )


def _resolve_wiki_path(input_run: Path, record: dict[str, Any]) -> Path | None:
    return _resolve_cache_path(
        input_run=input_run,
        record=record,
        path_key="wiki_cache_path",
        marker="wiki-cache",
        fallback_file_name="wiki_text.txt",
    )


def _resolve_file_list_path(input_run: Path, record: dict[str, Any]) -> Path | None:
    candidates = _path_candidates(
        input_run=input_run,
        recorded=record.get("file_list_path"),
        marker="snapshots",
    )
    owner = record.get("repository_owner")
    repo = record.get("repository_name")
    if owner and repo:
        candidates.append(
            input_run
            / "snapshots"
            / _safe_path_part(owner)
            / _safe_path_part(repo)
            / "repository_file_list.json"
        )
    return _first_existing_file(candidates)


def _resolve_cache_path(
    *,
    input_run: Path,
    record: dict[str, Any],
    path_key: str,
    marker: str,
    fallback_file_name: str,
) -> Path | None:
    candidates = _path_candidates(
        input_run=input_run,
        recorded=record.get(path_key),
        marker=marker,
    )
    owner = record.get("repository_owner")
    repo = record.get("repository_name")
    if owner and repo:
        candidates.append(
            input_run
            / marker
            / f"{_safe_path_part(owner)}__{_safe_path_part(repo)}"
            / fallback_file_name
        )
    return _first_existing_file(candidates)


def _path_candidates(input_run: Path, recorded: Any, marker: str) -> list[Path]:
    candidates: list[Path] = []
    if recorded:
        recorded_path = Path(str(recorded))
        candidates.append(recorded_path)
        if not recorded_path.is_absolute():
            candidates.append(input_run / recorded_path)
        suffix = _suffix_from_marker(str(recorded), marker)
        if suffix:
            candidates.append(input_run / marker / suffix)
    return candidates


def _first_existing_file(candidates: list[Path]) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _suffix_from_marker(path_text: str, marker: str) -> Path | None:
    normalized = path_text.replace("\\", "/")
    marker_text = f"/{marker}/"
    if marker_text in normalized:
        suffix = normalized.split(marker_text, maxsplit=1)[1]
        return Path(*[part for part in suffix.split("/") if part])
    if normalized.startswith(f"{marker}/"):
        suffix = normalized[len(marker) + 1 :]
        return Path(*[part for part in suffix.split("/") if part])
    return None


def _read_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_file_paths(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return []
    return [str(file) for file in files if str(file).strip()]


def _text_character_stats(text: str) -> TextCharacterStats:
    total = 0
    english = 0
    for char in str(text or ""):
        if not char.isalpha():
            continue
        total += 1
        if ("A" <= char <= "Z") or ("a" <= char <= "z"):
            english += 1
    non_english = max(0, total - english)
    ratio = (non_english / total) if total else 0.0
    return TextCharacterStats(
        total_alpha_chars=total,
        english_alpha_chars=english,
        non_english_alpha_chars=non_english,
        non_english_ratio=ratio,
    )


def _iter_jsonl_with_malformed(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                yield {
                    "malformed": True,
                    "line_number": line_number,
                    "raw_line_prefix": text[:200],
                }
                continue
            if not isinstance(payload, dict):
                yield {
                    "malformed": True,
                    "line_number": line_number,
                    "raw_line_prefix": text[:200],
                }
                continue
            yield {"malformed": False, "line_number": line_number, "record": payload}


def _iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _complete_create_training_outputs_exist(paths: dict[str, Path]) -> bool:
    required = (
        "manifest",
        "encoded_records",
        "topic_training_dataset_csv",
        "label_encoding_manifest",
        "preprocessed_records",
        "topic_label_universe",
        "preprocessing_artifacts",
    )
    return all(paths[name].is_file() for name in required)


def _raw_topics(record: dict[str, Any]) -> list[str]:
    values = record.get("raw_topics") or record.get("raw_repository_topics") or []
    if isinstance(values, str):
        values = re.split(r"[,;\s]+", values)
    if not isinstance(values, list):
        return []
    return sorted({str(topic).strip() for topic in values if str(topic).strip()})


def _load_excluded_repository_keys(config: CreateTrainingDataConfig) -> set[str]:
    raw_values = list(config.excluded_repositories or ())
    if config.excluded_repositories_file is not None:
        path = Path(config.excluded_repositories_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Excluded repositories file not found: {path}")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            raw_values.append(text)

    keys: set[str] = set()
    for raw_value in raw_values:
        keys.update(_repository_skip_key_variants(raw_value))
    return keys


def _record_matches_excluded_repository(
    record: dict[str, Any],
    excluded_repository_keys: set[str],
) -> bool:
    if not excluded_repository_keys:
        return False
    for raw_value in _repository_identity_values(record):
        if _repository_skip_key_variants(raw_value) & excluded_repository_keys:
            return True
    return False


def _load_final_preprocessing_identity_keys(paths: dict[str, Path]) -> set[str]:
    keys: set[str] = set()
    for record in _iter_jsonl_records(paths["preprocessed_records"]):
        keys.update(_repository_identity_keys_for_resume(record))
    for exclusion in _iter_jsonl_records(paths["excluded_records"]):
        if exclusion.get("excluded_stage") != "preprocessing":
            continue
        record = exclusion.get("record")
        if isinstance(record, dict):
            keys.update(_repository_identity_keys_for_resume(record))
        else:
            keys.update(_repository_identity_keys_for_resume(exclusion))
    return keys


def _repository_identity_keys_for_resume(record: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for raw_value in _repository_identity_values(record):
        keys.update(_repository_skip_key_variants(raw_value))
    return keys


def _repository_identity_values(record: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in (
        "repository_identity_key",
        "repository_full_name",
        "repository_key",
        "repository_id",
    ):
        value = str(record.get(field) or "").strip()
        if value:
            values.add(value)
    owner = str(record.get("repository_owner") or "").strip()
    name = str(record.get("repository_name") or "").strip()
    if owner and name:
        values.add(f"{owner}/{name}")
        values.add(f"repository-key:{owner}/{name}")
    return values


def _repository_skip_key_variants(raw_value: Any) -> set[str]:
    text = str(raw_value or "").strip()
    if not text:
        return set()
    lower = text.lower()
    values = {text, lower}
    for prefix in ("repository-key:", "repository-id:"):
        if lower.startswith(prefix):
            suffix = text[len(prefix) :].strip()
            if suffix:
                values.add(suffix)
                values.add(suffix.lower())
    if "/" in text and not lower.startswith("repository-key:"):
        values.add(f"repository-key:{text}")
        values.add(f"repository-key:{lower}")
    if text.isdigit() and not lower.startswith("repository-id:"):
        values.add(f"repository-id:{text}")
    return {value.strip().lower() for value in values if value and value.strip()}


def _repository_log_label(record: dict[str, Any]) -> str:
    full_name = str(record.get("repository_full_name") or "").strip()
    if full_name:
        return full_name
    owner = str(record.get("repository_owner") or "").strip()
    name = str(record.get("repository_name") or "").strip()
    if owner and name:
        return f"{owner}/{name}"
    repository_id = str(record.get("repository_id") or "").strip()
    if repository_id:
        return f"repository_id={repository_id}"
    identity_key = str(record.get("repository_identity_key") or "").strip()
    return identity_key or "unknown"


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "filtered_records": output_dir / "filtered_repository_records.jsonl",
        "excluded_records": output_dir / "excluded_repository_records.jsonl",
        "filtering_manifest": output_dir / "filtering_manifest.json",
        "preprocessing_count_checkpoint": output_dir / "preprocessing_count_checkpoint.json",
        "preprocessing_count_manifest": output_dir / "preprocessing_count_manifest.json",
        "preprocessed_records": output_dir / "preprocessed_repository_records.jsonl",
        "candidate_topics": output_dir / "candidate_topics.csv",
        "raw_topic_universe": output_dir / "raw_topic_universe.json",
        "mapped_topic_counts": output_dir / "mapped_topic_counts.csv",
        "low_freq_topics": output_dir / "low_freq_topics.csv",
        "topic_label_universe": output_dir / "topic_label_universe.json",
        "text_token_counts": output_dir / "text_token_counts.csv",
        "file_name_token_counts": output_dir / "file_name_token_counts.csv",
        "preprocessing_artifacts": output_dir / "preprocessing_artifacts.json",
        "encoded_records": output_dir / "encoded_repository_records.jsonl",
        "topic_training_dataset_csv": output_dir / "topic_training_dataset.csv",
        "label_encoding_manifest": output_dir / "label_encoding_manifest.json",
        "manifest": output_dir / "create_training_data_manifest.json",
        "train_test_placeholder": output_dir / "train_test_placeholder.json",
    }


def _write_topic_counts_csv(
    path: Path,
    counts: Counter[str],
    retained_labels: set[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["topic", "repository_count", "retained_label"],
        )
        writer.writeheader()
        for topic in sorted(counts):
            writer.writerow(
                {
                    "topic": topic,
                    "repository_count": counts[topic],
                    "retained_label": topic in retained_labels,
                }
            )


def _write_low_freq_topics(path: Path, low_freq_topics: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for index, topic in enumerate(low_freq_topics):
            writer.writerow([index, topic, "-1", ""])


def _write_token_counts(path: Path, counts: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "count"])
        writer.writeheader()
        for token in sorted(counts):
            writer.writerow({"token": token, "count": counts[token]})


def _write_json(path: Path, payload: Any) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
        newline="\n",
    )
    with temp_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n")
    temp_path.replace(path)


def _write_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, default=str))
    handle.write("\n")


def _write_count_checkpoint(
    path: Path,
    *,
    status: str,
    filtered_records_path: Path,
    filtered_records_seen: int,
    processed_records: int,
    raw_topic_counts: Counter[str],
    mapped_topic_counts: Counter[str],
    text_token_counts: Counter[str],
    file_name_token_counts: Counter[str],
    source_counts: Counter[str],
    excluded_keys_sha256: str,
) -> None:
    _write_json(
        path,
        {
            "schema_version": "topic_create_training_count_checkpoint_v1",
            "status": status,
            "updated_at_utc": _utc_now_z(),
            "filtered_records_path": str(filtered_records_path),
            "filtered_records_seen": int(filtered_records_seen),
            "processed_records": int(processed_records),
            "excluded_repository_keys_sha256": excluded_keys_sha256,
            "raw_topic_counts": _counter_to_json(raw_topic_counts),
            "mapped_topic_counts": _counter_to_json(mapped_topic_counts),
            "text_token_counts": _counter_to_json(text_token_counts),
            "file_name_token_counts": _counter_to_json(file_name_token_counts),
            "source_counts": _counter_to_json(source_counts),
        },
    )


def _load_count_checkpoint(
    path: Path,
    *,
    excluded_keys_sha256: str,
) -> dict[str, Any] | None:
    payload = _read_optional_json(path)
    if payload is None:
        return None
    if payload.get("schema_version") != "topic_create_training_count_checkpoint_v1":
        return None
    if str(payload.get("excluded_repository_keys_sha256") or "") != excluded_keys_sha256:
        payload["_excluded_repository_configuration_changed"] = True
        log(
            "Resume checkpoint excluded repository configuration changed; resuming anyway: "
            f"{path}. Already-counted repositories are not subtracted from count artifacts, "
            "but the current skip-list applies to remaining records and final preprocessing."
        )
    else:
        payload["_excluded_repository_configuration_changed"] = False
    return payload


def _counter_to_json(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _counter_from_json(payload: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(payload, dict):
        return counter
    for key, value in payload.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count:
            counter[str(key)] = count
    return counter


def _subtract_counter(counter: Counter[str], values: tuple[str, ...] | list[str]) -> None:
    counter.subtract(values)
    for key in list(counter):
        if counter[key] <= 0:
            del counter[key]


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _update_source_counts(counter: Counter[str], sources: RepositorySources) -> None:
    if sources.name_parts:
        counter["project_names_loaded"] += 1
    if sources.readme.strip():
        counter["readme_loaded"] += 1
    if sources.wiki.strip():
        counter["wiki_loaded"] += 1
    if sources.file_paths:
        counter["file_list_loaded"] += 1
    if sources.wiki_path is None:
        counter["wiki_missing"] += 1
    if sources.file_list_path is None:
        counter["file_list_missing"] += 1


def _log_filter_progress(counts: Counter[str], progress_interval: int) -> None:
    processed = counts["input_records"]
    if processed <= 0 or processed % progress_interval != 0:
        return
    excluded = counts["excluded_records"] + counts["malformed_records"]
    log(
        "Filtering progress: "
        f"processed={processed}, retained={counts['filtered_records']}, "
        f"excluded={excluded}, no_readme={counts['excluded_no_readme']}, "
        f"no_description={counts['excluded_no_description']}, "
        f"non_english_ratio={counts['excluded_non_english_ratio']}, "
        f"skipped_repository={counts['excluded_skipped_repository']}, "
        f"malformed={counts['malformed_records']}"
    )


def _log_final_progress(counts: Counter[str], progress_interval: int) -> None:
    processed = counts["processed_filtered_records"]
    if processed <= 0 or processed % progress_interval != 0:
        return
    excluded = counts["excluded_no_final_label"] + counts["excluded_empty_prepared_text"]
    log(
        "Final preprocessing progress: "
        f"processed={processed}, retained={counts['preprocessed_records']}, "
        f"excluded={excluded}, no_label={counts['excluded_no_final_label']}, "
        f"empty_text={counts['excluded_empty_prepared_text']}"
    )


def _resolve_output_dir(input_run: Path, output_subdir: Path) -> Path:
    output = Path(output_subdir)
    return output if output.is_absolute() else input_run / output


def _validate_config(config: CreateTrainingDataConfig) -> None:
    ratio = float(config.max_non_english_ratio)
    if ratio < 0 or ratio > 1:
        raise ValueError(
            f"max_non_english_ratio must be between 0 and 1, got {ratio}"
        )
    source_timeout_seconds = float(config.source_timeout_seconds)
    if source_timeout_seconds < 0:
        raise ValueError(
            "source_timeout_seconds must be >= 0. Use 0 to disable source timeouts; "
            f"got {source_timeout_seconds}"
        )
    data_preparation_root = Path(config.data_preparation_root)
    rules_dir = data_preparation_root / "rules"
    lists_dir = data_preparation_root / "lists"
    if not rules_dir.is_dir() or not lists_dir.is_dir():
        raise FileNotFoundError(
            "Topic data-preparation profile is missing. Expected "
            f"{rules_dir} and {lists_dir}. Run "
            "`python post-processing/topic-classification/data-preparation/"
            "generate_github_topic_rules.py` to create the default GitHub-topics "
            "profile, or set POST_PROCESSING_TOPIC_CREATE_TRAINING_DATA_PREPARATION_ROOT "
            "to an existing profile root such as "
            "post-processing/topic-classification/data-preparation."
        )


def _safe_path_part(value: Any) -> str:
    text = str(value or "").strip() or "unknown"
    text = text.replace("\\", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_preparation_profile_manifest(root: Path) -> dict[str, Any]:
    root = Path(root)
    rules_dir = root / "rules"
    lists_dir = root / "lists"
    rule_files = sorted(path for path in rules_dir.iterdir() if path.is_file())
    list_files = sorted(path for path in lists_dir.iterdir() if path.is_file())
    return {
        "root": str(root),
        "rules_dir": str(rules_dir),
        "lists_dir": str(lists_dir),
        "rule_file_count": len(rule_files),
        "list_file_count": len(list_files),
        "rule_file_checksums": {
            path.name: _sha256_file(path) for path in rule_files
        },
        "list_file_checksums": {
            path.name: _sha256_file(path) for path in list_files
        },
        "skipped_topic_rule_names": list(SKIPPED_STATIC_LOW_FREQ_RULES),
    }


def _utc_now_z() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


def _env_int_allow_zero(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_string_list(name: str) -> tuple[str, ...]:
    value = os.environ.get(name, "")
    if not value:
        return ()
    parts = re.split(r"[\n,]+", value)
    return tuple(part.strip() for part in parts if part.strip())


def _env_optional_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def log(message: str) -> None:
    """Print a flushed create-training-data log line."""
    print(f"{LOG_PREFIX} {message}", flush=True)
