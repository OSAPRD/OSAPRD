"""Train and export the repository topic-classification model bundle.

Stage 3 consumes sampled `topics_train.csv` and `topics_test.csv` files,
validates the retained label universe, trains the TF-IDF plus one-vs-rest
logistic-regression model, and writes a complete inference bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


LOG_PREFIX = "[post-processing/topic-classification]"
TEXT_COLUMN = "text"
LABELS_COLUMN = "labels"
DEFAULT_MODEL_NAME = "topic_model"
DEFAULT_TOP_K = 5
DEFAULT_MAX_FEATURES = 20000
DEFAULT_NGRAM_RANGE = (1, 2)
PREPROCESSING_ARTIFACT_SCHEMA_VERSION = "topic_preprocessing_artifacts_v2"
RUNTIME_PREPROCESSING_TOKEN_RE = re.compile(r"^[a-z]{2,}$")
PAPER_TEXT_MIN_FREQUENCY = 50
PAPER_FILE_NAME_MIN_FREQUENCY = 20
DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS = (
    "api",
    "app",
    "bot",
    "client",
    "config",
    "core",
    "demo",
    "extension",
    "framework",
    "git",
    "github",
    "json",
    "lib",
    "library",
    "manager",
    "module",
    "plugin",
    "server",
    "service",
    "tool",
    "ui",
    "web",
)
ORIGINAL_RECOMMENDER_MODEL = "lr"
ORIGINAL_RECOMMENDER_TOKENIZER = "tfidf"
ORIGINAL_RECOMMENDER_METHOD = "ovr"
ORIGINAL_RECOMMENDER_FEATURE_MODE = 0
ORIGINAL_RECOMMENDER_TUNING_MODE = 0
ORIGINAL_RECOMMENDER_CLASS_WEIGHTS = "balanced"
ORIGINAL_RECOMMENDER_K_LIST = (1, 2, 3, 5, 8, 10)
ORIGINAL_RECOMMENDER_THRESHOLDS = (0.25, 0.5)
PAPER_TOP_K_VALUES = ORIGINAL_RECOMMENDER_K_LIST
ORIGINAL_RECOMMENDER_REPOSITORY_URL = (
    "https://github.com/MalihehIzadi/SoftwareTagRecommender"
)
ORIGINAL_RECOMMENDER_TRAINING_SCRIPT_URL = (
    "https://github.com/MalihehIzadi/SoftwareTagRecommender"
    "/blob/master/machine_learning/traditional_classifiers.py"
)

TOPIC_CLASSIFICATION_DIR = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore")

DEFAULT_TRAIN_CSV = (
    TOPIC_CLASSIFICATION_DIR
    / "training-data"
    / "topics220_repos152k_train.csv"
)
DEFAULT_TEST_CSV = (
    TOPIC_CLASSIFICATION_DIR
    / "training-data"
    / "topics220_repos152K_test.csv"
)
DEFAULT_OUTPUT_DIR = TOPIC_CLASSIFICATION_DIR / "model-output"
DEFAULT_RAW_FREQUENCY_ARTIFACT_DIR = TOPIC_CLASSIFICATION_DIR / "data-preparation" / "raw-frequency"
DEFAULT_RAW_TEXT_TOKEN_COUNTS = DEFAULT_RAW_FREQUENCY_ARTIFACT_DIR / "text_token_counts.csv"
DEFAULT_RAW_FILE_NAME_TOKEN_COUNTS = (
    DEFAULT_RAW_FREQUENCY_ARTIFACT_DIR / "file_name_token_counts.csv"
)
SOURCE_COMPATIBILITY_NOTES = [
    {
        "source_issue": "traditional_classifiers.py references stats.hmean without importing stats.",
        "wrapper_behavior": (
            "Use the equivalent per-row harmonic mean calculation so the intended "
            "F1@k metric is runnable."
        ),
    },
    {
        "source_issue": "traditional_classifiers.py returns undefined s2/s3/s4 values from prf_at_k.",
        "wrapper_behavior": (
            "Do not reproduce the broken return values; export the defined R/P/F/S1/S5 "
            "metrics used by the script output."
        ),
    },
    {
        "source_issue": "traditional_classifiers.py pickles only the classifier model.",
        "wrapper_behavior": (
            "Save a complete inference bundle containing vectorizer, classifier, "
            "ordered topic labels, metrics, and metadata."
        ),
    },
]
TRAINING_DATA_PREPARATION_METADATA = {
    "training_text_preprocessing": "none",
    "released_csv_assumption": (
        "The train/test CSV text column is treated as already prepared by "
        "the paper's data-preparation process."
    ),
    "runtime_repository_preprocessing": (
        "Repository classification uses classify-topics/topic_preprocessing.py, but the "
        "released training CSV text is not reprocessed during model training."
    ),
    "runtime_token_vocabulary": (
        "Training exports a model-bundle preprocessing artifact for runtime token "
        "filtering. If raw source-specific token-count artifacts are supplied, "
        "the artifact uses the paper's separate thresholds: 50 for text tokens "
        "and 20 for file-name tokens. Otherwise it falls back to the released "
        "prepared train/test CSV text vocabulary and records that this is not "
        "raw-corpus equivalent."
    ),
}


@dataclass(frozen=True)
class TrainingDependencies:
    """Imported training libraries passed around as one explicit dependency bag."""

    np: Any
    pd: Any
    joblib: Any
    sklearn: Any
    LogisticRegression: Any
    OneVsRestClassifier: Any
    TfidfVectorizer: Any
    label_ranking_average_precision_score: Any


@dataclass(frozen=True)
class ArtifactPaths:
    """Output paths for the model bundle and companion metadata files."""

    bundle: Path
    labels: Path
    labels_alphabetical_json: Path
    labels_alphabetical_txt: Path
    preprocessing_artifacts: Path
    metrics: Path
    manifest: Path


def log(message: str) -> None:
    """Print a flushed model-training log line."""
    print(f"{LOG_PREFIX} {message}", flush=True)


def _load_training_dependencies() -> TrainingDependencies:
    try:
        import numpy as np
        if int(str(np.__version__).split(".", maxsplit=1)[0]) >= 2:
            raise RuntimeError(
                "NumPy 2.x is installed, but this training entry point is pinned "
                "to numpy<2 for compatibility with pandas/scikit-learn builds used "
                f"by this environment. Current NumPy version: {np.__version__}"
            )

        import joblib
        import pandas as pd
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import label_ranking_average_precision_score
        from sklearn.multiclass import OneVsRestClassifier
    except Exception as exc:  # pragma: no cover - exercised by environment failures.
        raise RuntimeError(
            "Unable to import topic-model training dependencies. "
            "Install the post-processing requirements first, including numpy<2, "
            "pandas, scikit-learn, and joblib. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return TrainingDependencies(
        np=np,
        pd=pd,
        joblib=joblib,
        sklearn=sklearn,
        LogisticRegression=LogisticRegression,
        OneVsRestClassifier=OneVsRestClassifier,
        TfidfVectorizer=TfidfVectorizer,
        label_ranking_average_precision_score=label_ranking_average_precision_score,
    )


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc


def _topic_labels_from_header(header: Sequence[str], path: Path) -> list[str]:
    if len(header) < 3:
        raise ValueError(f"CSV header has too few columns: {path}")
    trailing = list(header[-2:])
    if trailing != [LABELS_COLUMN, TEXT_COLUMN]:
        raise ValueError(
            "Expected CSV header to end with "
            f"{LABELS_COLUMN!r}, {TEXT_COLUMN!r}; got {trailing!r} in {path}"
        )
    topic_labels = list(header[:-2])
    if not topic_labels:
        raise ValueError(f"No topic label columns found in CSV header: {path}")
    duplicates = sorted({label for label in topic_labels if topic_labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate topic label columns in {path}: {duplicates}")
    return topic_labels


def _validate_headers(train_csv: Path, test_csv: Path) -> list[str]:
    train_header = _read_csv_header(train_csv)
    test_header = _read_csv_header(test_csv)
    if train_header != test_header:
        raise ValueError(
            "Train/test CSV headers differ. The model requires identical topic label "
            "columns in the same order."
        )
    return _topic_labels_from_header(train_header, train_csv)


def _sha256_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_topic_slug(value: Any) -> str:
    return str(value or "").strip().lower()


def _load_sample_manifest_labels(path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = _load_json(path)
    labels = [
        str(label).strip()
        for label in payload.get("retained_label_universe") or []
        if str(label).strip()
    ]
    if not labels:
        raise ValueError(
            "Sample manifest does not contain retained_label_universe: "
            f"{path}"
        )
    return labels, payload


def _load_filtered_topics(path: Path) -> set[str]:
    payload = _load_json(path)
    topics: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str):
            normalized = _normalize_topic_slug(value)
            if normalized:
                topics.add(normalized)
        elif isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            for item in value.values():
                add(item)

    if isinstance(payload, dict):
        for key in ("filtered_topics", "topics", "programming_language_topics"):
            add(payload.get(key))
        add(payload.get("categories"))
    elif isinstance(payload, list):
        add(payload)
    else:
        raise ValueError(f"Unsupported filtered topics JSON shape: {path}")
    return topics


def _load_topic_domain_topics(path: Path) -> set[str]:
    payload = _load_json(path)
    topic_domains = payload.get("topic_domains")
    if not isinstance(topic_domains, dict) or not topic_domains:
        raise ValueError(f"topic_domains JSON does not contain topic_domains: {path}")
    return {
        _normalize_topic_slug(topic)
        for topic in topic_domains
        if _normalize_topic_slug(topic)
    }


def _validate_training_label_guards(
    *,
    topic_labels: Sequence[str],
    sample_manifest: Path | None = None,
    filtered_topics_json: Path | None = None,
    topic_domains_json: Path | None = None,
    expected_label_count: int | None = None,
    require_exact_domain_labels: bool = False,
) -> dict[str, Any]:
    labels = [str(label).strip() for label in topic_labels if str(label).strip()]
    if len(labels) != len(topic_labels):
        raise ValueError("Topic label guard received empty topic label values.")
    label_set = {_normalize_topic_slug(label) for label in labels}
    if len(label_set) != len(labels):
        raise ValueError("Topic label guard received duplicate normalized labels.")

    guard: dict[str, Any] = {
        "enabled": any(
            value is not None
            for value in (
                sample_manifest,
                filtered_topics_json,
                topic_domains_json,
                expected_label_count,
            )
        )
        or require_exact_domain_labels,
        "label_count": len(labels),
        "label_universe_sha256": _sha256_lines(labels),
        "expected_label_count": expected_label_count,
        "require_exact_domain_labels": require_exact_domain_labels,
        "sample_manifest": str(sample_manifest) if sample_manifest is not None else None,
        "filtered_topics_json": (
            str(filtered_topics_json) if filtered_topics_json is not None else None
        ),
        "topic_domains_json": str(topic_domains_json) if topic_domains_json else None,
    }

    if expected_label_count is not None and int(expected_label_count) > 0:
        expected = int(expected_label_count)
        if len(labels) != expected:
            raise ValueError(
                "Training label count does not match the expected retained topic "
                f"count: labels={len(labels)}, expected={expected}."
            )

    if sample_manifest is not None:
        retained_labels, sample_payload = _load_sample_manifest_labels(sample_manifest)
        if labels != retained_labels:
            missing = sorted(set(retained_labels) - set(labels))
            extra = sorted(set(labels) - set(retained_labels))
            raise ValueError(
                "Training CSV label columns do not exactly match the sampled "
                "retained label universe. "
                f"csv_count={len(labels)}, retained_count={len(retained_labels)}, "
                f"missing={missing[:20]}, extra={extra[:20]}."
            )
        manifest_sha = str(
            sample_payload.get("retained_label_universe_sha256")
            or sample_payload.get("label_universe_sha256")
            or ""
        ).strip()
        computed_sha = _sha256_lines(labels)
        if manifest_sha and manifest_sha != computed_sha:
            raise ValueError(
                "Training CSV label columns match the sample manifest labels, but "
                "the retained label universe checksum differs. "
                f"manifest={manifest_sha}, computed={computed_sha}."
            )
        guard.update(
            {
                "sample_manifest_label_count": len(retained_labels),
                "sample_manifest_label_universe_sha256": manifest_sha or computed_sha,
                "sample_manifest_sha256": _sha256_file(sample_manifest),
            }
        )

    if filtered_topics_json is not None:
        filtered_topics = _load_filtered_topics(filtered_topics_json)
        leaks = sorted(label_set & filtered_topics)
        if leaks:
            raise ValueError(
                "Filtered topics are present in training labels. "
                f"count={len(leaks)}, examples={leaks[:20]}."
            )
        guard.update(
            {
                "filtered_topics_count": len(filtered_topics),
                "filtered_topics_leak_count": 0,
                "filtered_topics_sha256": _sha256_file(filtered_topics_json),
            }
        )

    if topic_domains_json is not None:
        domain_topics = _load_topic_domain_topics(topic_domains_json)
        missing_domain_topics = sorted(label_set - domain_topics)
        if missing_domain_topics:
            raise ValueError(
                "Training labels are missing from topic_domains.json. "
                f"count={len(missing_domain_topics)}, "
                f"examples={missing_domain_topics[:20]}."
            )
        extra_domain_topics = sorted(domain_topics - label_set)
        if require_exact_domain_labels and extra_domain_topics:
            raise ValueError(
                "Training labels do not exactly cover the retained domain topic "
                "mapping. "
                f"csv_count={len(labels)}, domain_topic_count={len(domain_topics)}, "
                f"missing_from_csv={extra_domain_topics[:20]}."
            )
        guard.update(
            {
                "domain_topic_count": len(domain_topics),
                "labels_missing_domain_mapping_count": 0,
                "domain_topics_missing_from_training_count": len(extra_domain_topics),
                "topic_domains_sha256": _sha256_file(topic_domains_json),
            }
        )

    return guard


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(output_dir: Path, model_name: str) -> ArtifactPaths:
    normalized_name = model_name.strip() or DEFAULT_MODEL_NAME
    if normalized_name == DEFAULT_MODEL_NAME:
        return ArtifactPaths(
            bundle=output_dir / "topic_model_bundle.joblib",
            labels=output_dir / "topic_labels.json",
            labels_alphabetical_json=output_dir / "topic_labels_alphabetical.json",
            labels_alphabetical_txt=output_dir / "topic_labels_alphabetical.txt",
            preprocessing_artifacts=output_dir / "topic_preprocessing_artifacts.json",
            metrics=output_dir / "topic_model_metrics.json",
            manifest=output_dir / "topic_model_manifest.json",
        )
    return ArtifactPaths(
        bundle=output_dir / f"{normalized_name}_bundle.joblib",
        labels=output_dir / f"{normalized_name}_labels.json",
        labels_alphabetical_json=output_dir / f"{normalized_name}_labels_alphabetical.json",
        labels_alphabetical_txt=output_dir / f"{normalized_name}_labels_alphabetical.txt",
        preprocessing_artifacts=output_dir / f"{normalized_name}_preprocessing_artifacts.json",
        metrics=output_dir / f"{normalized_name}_metrics.json",
        manifest=output_dir / f"{normalized_name}_manifest.json",
    )


def _run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{str(prefix).strip('_')}_{timestamp}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(f"{value}\n")


def _alphabetical_topics(topic_labels: Sequence[str]) -> list[str]:
    return sorted(str(label) for label in topic_labels)


def _update_prepared_text_token_counts(
    token_counts: Counter[str],
    texts: Sequence[Any],
) -> None:
    for text in texts:
        for token in str(text).split():
            normalized = token.strip().lower()
            if RUNTIME_PREPROCESSING_TOKEN_RE.fullmatch(normalized):
                token_counts[normalized] += 1


def _normalize_frequency_token(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if not RUNTIME_PREPROCESSING_TOKEN_RE.fullmatch(token):
        return None
    return token


def _parse_frequency_count(value: Any) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except Exception:
        return 1
    return max(0, parsed)


def _update_token_count(token_counts: Counter[str], token: Any, count: Any = 1) -> None:
    normalized = _normalize_frequency_token(token)
    if normalized is None:
        return
    parsed_count = _parse_frequency_count(count)
    if parsed_count > 0:
        token_counts[normalized] += parsed_count


def _load_json_token_counts(path: Path) -> Counter[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    token_counts: Counter[str] = Counter()
    if isinstance(payload, dict):
        for token, count in payload.items():
            _update_token_count(token_counts, token, count)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                token = (
                    item.get("token")
                    or item.get("term")
                    or item.get("word")
                    or item.get("name")
                )
                count = item.get("count", item.get("frequency", item.get("freq", 1)))
                _update_token_count(token_counts, token, count)
            elif isinstance(item, (list, tuple)) and item:
                count = item[1] if len(item) > 1 else 1
                _update_token_count(token_counts, item[0], count)
            else:
                _update_token_count(token_counts, item, 1)
    else:
        raise ValueError(f"Unsupported JSON token-count artifact shape: {path}")
    return token_counts


def _load_tabular_token_counts(path: Path) -> Counter[str]:
    token_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(handle, dialect))
    if not rows:
        return token_counts

    header = [cell.strip().lower() for cell in rows[0]]
    token_columns = {"token", "term", "word", "name"}
    count_columns = {"count", "frequency", "freq"}
    has_header = bool(set(header) & token_columns)
    token_index = 0
    count_index: int | None = 1
    data_rows = rows
    if has_header:
        data_rows = rows[1:]
        for index, column in enumerate(header):
            if column in token_columns:
                token_index = index
                break
        count_index = None
        for index, column in enumerate(header):
            if column in count_columns:
                count_index = index
                break

    for row in data_rows:
        if not row or token_index >= len(row):
            continue
        count = row[count_index] if count_index is not None and count_index < len(row) else 1
        _update_token_count(token_counts, row[token_index], count)
    return token_counts


def _load_token_count_artifact(path: Path, *, label: str) -> Counter[str]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} token-count artifact not found: {resolved}")
    if resolved.suffix.lower() == ".json":
        token_counts = _load_json_token_counts(resolved)
    else:
        token_counts = _load_tabular_token_counts(resolved)
    if not token_counts:
        raise ValueError(f"{label} token-count artifact did not contain any valid tokens: {resolved}")
    return token_counts


def _resolve_raw_frequency_paths(
    *,
    raw_text_token_counts: Path | None,
    raw_file_name_token_counts: Path | None,
) -> tuple[Path | None, Path | None]:
    text_path = raw_text_token_counts
    file_path = raw_file_name_token_counts
    if text_path is None and DEFAULT_RAW_TEXT_TOKEN_COUNTS.exists():
        text_path = DEFAULT_RAW_TEXT_TOKEN_COUNTS
    if file_path is None and DEFAULT_RAW_FILE_NAME_TOKEN_COUNTS.exists():
        file_path = DEFAULT_RAW_FILE_NAME_TOKEN_COUNTS
    if (text_path is None) != (file_path is None):
        raise ValueError(
            "Raw corpus frequency filtering requires both --raw-text-token-counts "
            "and --raw-file-name-token-counts. Provide both artifacts or neither."
        )
    return text_path, file_path


def _load_data_preparation_profile_from_raw_artifacts(
    *,
    raw_text_token_counts: Path | None,
    raw_file_name_token_counts: Path | None,
) -> dict[str, Any] | None:
    if raw_text_token_counts is None or raw_file_name_token_counts is None:
        return None
    text_parent = raw_text_token_counts.resolve().parent
    file_parent = raw_file_name_token_counts.resolve().parent
    if text_parent != file_parent:
        return None
    artifact_path = text_parent / "preprocessing_artifacts.json"
    if not artifact_path.exists():
        return None
    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
    except Exception:
        return None
    profile = artifact.get("data_preparation_profile")
    if isinstance(profile, dict):
        return profile
    return None


def _build_preprocessing_artifacts(
    *,
    prepared_text_token_counts: Counter[str],
    train_csv: Path,
    test_csv: Path,
    train_row_count: int,
    test_row_count: int,
    raw_text_token_counts: Counter[str] | None = None,
    raw_file_name_token_counts: Counter[str] | None = None,
    raw_text_token_counts_path: Path | None = None,
    raw_file_name_token_counts_path: Path | None = None,
    data_preparation_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    using_raw_frequency_artifacts = (
        raw_text_token_counts is not None and raw_file_name_token_counts is not None
    )
    text_token_counts = (
        raw_text_token_counts if raw_text_token_counts is not None else prepared_text_token_counts
    )
    file_name_token_counts = (
        raw_file_name_token_counts
        if raw_file_name_token_counts is not None
        else prepared_text_token_counts
    )
    allowed_text_tokens = sorted(
        token
        for token, count in text_token_counts.items()
        if count >= PAPER_TEXT_MIN_FREQUENCY
    )
    allowed_file_name_tokens = sorted(
        token
        for token, count in file_name_token_counts.items()
        if count >= PAPER_FILE_NAME_MIN_FREQUENCY
    )
    file_name_informative_split_tokens = sorted(
        token
        for token in DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS
        if token in allowed_file_name_tokens
    )
    source_files = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }
    source_checksums = {
        "train_csv_sha256": _sha256_file(train_csv),
        "test_csv_sha256": _sha256_file(test_csv),
    }
    if using_raw_frequency_artifacts:
        assert raw_text_token_counts_path is not None
        assert raw_file_name_token_counts_path is not None
        source_files.update(
            {
                "raw_text_token_counts": str(raw_text_token_counts_path),
                "raw_file_name_token_counts": str(raw_file_name_token_counts_path),
            }
        )
        source_checksums.update(
            {
                "raw_text_token_counts_sha256": _sha256_file(raw_text_token_counts_path),
                "raw_file_name_token_counts_sha256": _sha256_file(
                    raw_file_name_token_counts_path
                ),
            }
        )
    source = (
        "raw_source_specific_frequency_artifacts_with_paper_thresholds"
        if using_raw_frequency_artifacts
        else "released_train_test_csv_prepared_text_with_paper_frequency_thresholds"
    )
    paper_equivalence = (
        "raw_source_frequency_artifact_equivalent"
        if using_raw_frequency_artifacts
        else "threshold_aligned_not_raw_corpus_equivalent"
    )
    runtime_filter = (
        "Filter repository runtime text tokens to the raw text-source vocabulary "
        "observed at least 50 times, and file-name tokens to the raw file-name "
        "vocabulary observed at least 20 times."
        if using_raw_frequency_artifacts
        else (
            "Filter repository runtime tokens to the vocabulary observed in the "
            "released prepared train/test CSV text column after applying the "
            "paper's text/file-name frequency thresholds."
        )
    )
    compatibility_note = (
        "Raw source-specific token frequency artifacts were provided. Runtime "
        "classification can therefore apply separate paper-threshold filters for "
        "description/README/wiki text and repository file names."
        if using_raw_frequency_artifacts
        else (
            "The vendored data-preparation release used here provides prepared "
            "train/test text but not separate raw source-specific token frequency "
            "artifacts for description/README/wiki versus file names. This artifact "
            "therefore preserves the prepared-corpus runtime vocabulary available "
            "from the released CSVs, with paper thresholds applied."
        )
    )
    allowed_tokens = sorted(set(allowed_text_tokens) | set(allowed_file_name_tokens))
    digest = hashlib.sha256()
    for token in allowed_tokens:
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    text_digest = hashlib.sha256()
    for token in allowed_text_tokens:
        text_digest.update(token.encode("utf-8"))
        text_digest.update(b"\n")
    file_digest = hashlib.sha256()
    for token in allowed_file_name_tokens:
        file_digest.update(token.encode("utf-8"))
        file_digest.update(b"\n")
    return {
        "schema_version": PREPROCESSING_ARTIFACT_SCHEMA_VERSION,
        "source": source,
        "source_files": source_files,
        "source_checksums": source_checksums,
        "data_preparation_profile": data_preparation_profile,
        "source_row_counts": {
            "train_row_count": train_row_count,
            "test_row_count": test_row_count,
        },
        "allowed_runtime_tokens": allowed_tokens,
        "allowed_runtime_token_count": len(allowed_tokens),
        "allowed_runtime_tokens_sha256": digest.hexdigest(),
        "allowed_text_tokens": allowed_text_tokens,
        "allowed_text_token_count": len(allowed_text_tokens),
        "allowed_text_tokens_sha256": text_digest.hexdigest(),
        "allowed_file_name_tokens": allowed_file_name_tokens,
        "allowed_file_name_token_count": len(allowed_file_name_tokens),
        "allowed_file_name_tokens_sha256": file_digest.hexdigest(),
        "file_name_informative_split_tokens": file_name_informative_split_tokens,
        "file_name_informative_split_token_count": len(file_name_informative_split_tokens),
        "file_name_informative_split_tokens_source": (
            "prepared_train_test_csv_intersection_with_documented_paper_examples"
        ),
        "frequency_policy": {
            "paper_text_min_frequency": PAPER_TEXT_MIN_FREQUENCY,
            "paper_file_name_min_frequency": PAPER_FILE_NAME_MIN_FREQUENCY,
            "released_csv_text_is_already_prepared": True,
            "raw_text_token_counts_provided": raw_text_token_counts is not None,
            "raw_file_name_token_counts_provided": raw_file_name_token_counts is not None,
            "runtime_token_shape": RUNTIME_PREPROCESSING_TOKEN_RE.pattern,
            "paper_equivalence": paper_equivalence,
            "runtime_filter": runtime_filter,
        },
        "separate_source_frequency_filters_available": using_raw_frequency_artifacts,
        "raw_corpus_frequency_artifacts_available": using_raw_frequency_artifacts,
        "compatibility_note": compatibility_note,
    }


def _load_dataset(
    deps: TrainingDependencies,
    path: Path,
    topic_labels: Sequence[str],
) -> tuple[Any, Any, list[str]]:
    """Load a CSV using the same dataframe shaping as traditional_classifiers.py."""
    frame = deps.pd.read_csv(path)
    if LABELS_COLUMN not in frame.columns:
        raise ValueError(f"CSV is missing {LABELS_COLUMN!r} column: {path}")
    if TEXT_COLUMN not in frame.columns:
        raise ValueError(f"CSV is missing {TEXT_COLUMN!r} column: {path}")

    # The reference script used df.columns.difference([text_col]), which can reorder
    # labels. Preserve the validated CSV label order so the exported bundle cannot
    # drift from the sampled retained label universe.
    frame = frame.drop(columns=[LABELS_COLUMN])
    x_frame = frame[[TEXT_COLUMN]].copy()
    y_frame = frame[list(topic_labels)]
    label_columns = [str(column) for column in y_frame.columns]
    if set(label_columns) != set(topic_labels):
        missing = sorted(set(topic_labels) - set(label_columns))
        extra = sorted(set(label_columns) - set(topic_labels))
        raise ValueError(
            "CSV label columns differ from the validated header labels. "
            f"Missing={missing[:10]}, extra={extra[:10]}, path={path}"
        )

    x_frame[TEXT_COLUMN] = x_frame[TEXT_COLUMN].astype(str)
    texts = x_frame[TEXT_COLUMN].values.astype("U")
    labels = y_frame.to_numpy(dtype=deps.np.int8, copy=True)
    return texts, labels, label_columns


def _build_original_tfidf_vectorizer(
    deps: TrainingDependencies,
    *,
    max_features: int,
) -> Any:
    return deps.TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",
        ngram_range=DEFAULT_NGRAM_RANGE,
        max_features=max(1, int(max_features)),
    )


def _build_original_multilabel_classifier(deps: TrainingDependencies) -> Any:
    estimator = deps.LogisticRegression(
        n_jobs=-1,
        class_weight=ORIGINAL_RECOMMENDER_CLASS_WEIGHTS,
    )
    return deps.OneVsRestClassifier(estimator)


def _metric_percent_view(metrics: dict[str, Any]) -> dict[str, float]:
    metric_prefixes = ("P@", "R@", "F1@", "S@", "S1@", "S5@")
    percent_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "LRAP" or key.startswith(metric_prefixes):
            if isinstance(value, float):
                percent_metrics[key] = round(value * 100, 2)
    return percent_metrics


def _paper_top_k_values(label_count: int, requested_top_k: int) -> list[int]:
    top_k_values = set(PAPER_TOP_K_VALUES)
    top_k_values.add(min(max(1, int(requested_top_k)), label_count))
    return [k for k in sorted(top_k_values) if k <= label_count]


def _original_recommender_prf_at_k(
    deps: TrainingDependencies,
    y_original: Any,
    y_pred_probab: Any,
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Mirror traditional_classifiers.py prf_at_k, with runnable edge-case guards."""
    np = deps.np
    y_org_array = y_original
    org_label_count_vec = np.sum(y_org_array, axis=1)
    repo_5_tags = int(len(np.where(org_label_count_vec >= 5)[0]))
    results: dict[str, Any] = {
        "repositories_with_at_least_5_labels": repo_5_tags,
    }

    for k in k_values:
        org_label_count = np.sum(y_org_array, axis=1).tolist()
        top_ind = np.argpartition(y_pred_probab, -1 * k, axis=1)[:, -1 * k:]
        pred_in_org = y_org_array[np.arange(y_org_array.shape[0])[:, None], top_ind]
        common_topk = np.sum(pred_in_org, axis=1)
        recall: list[float] = []
        precision: list[float] = []
        success1 = 0
        success5 = 0
        for index, value in enumerate(common_topk):
            denominator = min(k, org_label_count[index])
            recall.append(float(value / denominator) if denominator else 0.0)
            precision.append(float(value / k))
            if value >= 1:
                success1 += 1
            if value >= 5:
                success5 += 1

        precision_array = np.asarray(precision, dtype=np.float64)
        recall_array = np.asarray(recall, dtype=np.float64)
        f1 = np.divide(
            2 * precision_array * recall_array,
            precision_array + recall_array,
            out=np.zeros_like(precision_array, dtype=np.float64),
            where=(precision_array + recall_array) > 0,
        )
        results[f"R@{k}"] = float(np.mean(recall_array))
        results[f"P@{k}"] = float(np.mean(precision_array))
        results[f"F1@{k}"] = float(np.mean(f1))
        results[f"S@{k}"] = float(success1 / len(y_original))
        results[f"S1@{k}"] = results[f"S@{k}"]
        results[f"S5@{k}"] = float(success5 / repo_5_tags) if repo_5_tags else 0.0

    return results


def _original_threshold_metrics(
    deps: TrainingDependencies,
    y_original: Any,
    y_pred_probab: Any,
) -> dict[str, Any]:
    """Mirror the reference script's threshold eval_results block."""
    np = deps.np
    metrics = deps.sklearn.metrics

    def success_rate(y_pred: Any) -> float:
        common = 0
        for index in range(0, y_pred.shape[0]):
            if sum(y_original[index] * y_pred[index]) > 0:
                common += 1
        return common / y_pred.shape[0]

    def coverage(y_pred: Any) -> float:
        selected_per_label = y_pred.sum(axis=0)
        covered = np.count_nonzero(selected_per_label > 0)
        return covered / y_original.shape[1]

    result: dict[str, Any] = {}
    for threshold in ORIGINAL_RECOMMENDER_THRESHOLDS:
        y_pred = np.where(y_pred_probab > threshold, 1, 0)
        threshold_result: dict[str, str] = {
            "Success_Rate": f"{success_rate(y_pred) * 100:.2f}",
            "Coverage": f"{coverage(y_pred) * 100:.2f}",
        }
        metric_calls = {
            "LRL": lambda: metrics.label_ranking_loss(y_original, y_pred),
            "F1_micro": lambda: metrics.f1_score(y_original, y_pred, average="micro"),
            "F1_macro": lambda: metrics.f1_score(y_original, y_pred, average="macro"),
            "F1_weighted": lambda: metrics.f1_score(y_original, y_pred, average="weighted"),
            "F1_samples": lambda: metrics.f1_score(y_original, y_pred, average="samples"),
            "P_micro": lambda: metrics.precision_score(y_original, y_pred, average="micro"),
            "P_macro": lambda: metrics.precision_score(y_original, y_pred, average="macro"),
            "P_weighted": lambda: metrics.precision_score(y_original, y_pred, average="weighted"),
            "P_samples": lambda: metrics.precision_score(y_original, y_pred, average="samples"),
            "R_micro": lambda: metrics.recall_score(y_original, y_pred, average="micro"),
            "R_macro": lambda: metrics.recall_score(y_original, y_pred, average="macro"),
            "R_weighted": lambda: metrics.recall_score(y_original, y_pred, average="weighted"),
            "R_samples": lambda: metrics.recall_score(y_original, y_pred, average="samples"),
            "Hamming_loss": lambda: metrics.hamming_loss(y_original, y_pred),
            "Exact_match_ratio": lambda: metrics.accuracy_score(y_original, y_pred),
            "AUC_micro": lambda: metrics.roc_auc_score(y_original, y_pred, average="micro"),
            "AUC_macro": lambda: metrics.roc_auc_score(y_original, y_pred, average="macro"),
            "AUC_wighted": lambda: metrics.roc_auc_score(y_original, y_pred, average="weighted"),
            "Coverage_err": lambda: metrics.coverage_error(y_original, y_pred),
            "Avg_P_score_micro": lambda: metrics.average_precision_score(
                y_original,
                y_pred,
                average="micro",
            ),
            "Avg_P_score_macro": lambda: metrics.average_precision_score(
                y_original,
                y_pred,
                average="macro",
            ),
        }
        for name, metric_call in metric_calls.items():
            try:
                threshold_result[name] = f"{metric_call() * 100:.2f}"
            except Exception as exc:
                threshold_result[name] = f"ERROR: {type(exc).__name__}: {exc}"
        result[str(threshold)] = threshold_result
    return result


def _evaluate_paper_metrics(
    deps: TrainingDependencies,
    y_true: Any,
    y_scores: Any,
    top_k: int,
) -> dict[str, Any]:
    label_count = int(y_true.shape[1])
    normalized_top_k = min(max(1, int(top_k)), label_count)
    lrap = deps.label_ranking_average_precision_score(y_true, y_scores)
    top_k_values = _paper_top_k_values(label_count, normalized_top_k)
    prf_metrics = _original_recommender_prf_at_k(
        deps,
        y_true,
        y_scores,
        k_values=top_k_values,
    )

    metrics: dict[str, Any] = {
        "metric_family": "paper_top_k_recommendation_metrics",
        "metric_scale": "0_to_1",
        "primary_top_k": normalized_top_k,
        "top_k_values": top_k_values,
        "repositories_with_at_least_5_labels": prf_metrics[
            "repositories_with_at_least_5_labels"
        ],
        "LRAP": float(lrap),
        "paper_reported_metric_order": [
            f"R@{normalized_top_k}",
            f"P@{normalized_top_k}",
            f"F1@{normalized_top_k}",
            f"S@{normalized_top_k}",
            "LRAP",
        ],
        "original_recommender_metric_aliases": {
            "S1@k": "S@k",
            "S5@k": (
                "Fraction of repositories with at least five true labels where at "
                "least five top-k recommendations are true labels."
            ),
        },
        "original_recommender_percent_strings": {},
        "source_compatibility_notes": SOURCE_COMPATIBILITY_NOTES,
        "definitions": {
            "precision_at_k": "Mean fraction of top-k predicted topics present in true labels.",
            "recall_at_k": (
                "Mean fraction of a repository's true topics recovered in the top-k "
                "recommendation list."
            ),
            "success_at_k": "Fraction of repositories with at least one true topic in top-k.",
            "success_5_at_k": (
                "Original recommender S5@k: among repositories with at least five "
                "true labels, the fraction with five or more true labels in top-k."
            ),
            "f1_at_k": "Mean per-repository harmonic mean of P@k and R@k.",
            "lrap": "sklearn.metrics.label_ranking_average_precision_score.",
        },
    }
    metrics.update(prf_metrics)

    metrics["paper_percent_metrics"] = _metric_percent_view(metrics)
    metrics["original_recommender_percent_strings"] = {
        key: f"{value:.2f}"
        for key, value in metrics["paper_percent_metrics"].items()
    }
    metrics["original_threshold_metrics"] = _original_threshold_metrics(
        deps,
        y_true,
        y_scores,
    )
    return metrics


def _dependency_versions(deps: TrainingDependencies) -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": str(deps.np.__version__),
        "pandas": str(deps.pd.__version__),
        "scikit_learn": str(deps.sklearn.__version__),
        "joblib": str(getattr(deps.joblib, "__version__", "unknown")),
    }


def train_topic_model(
    *,
    train_csv: Path,
    test_csv: Path,
    output_dir: Path,
    model_name: str,
    top_k: int,
    max_features: int,
    raw_text_token_counts: Path | None = None,
    raw_file_name_token_counts: Path | None = None,
    require_raw_frequency_artifacts: bool = False,
    sample_manifest: Path | None = None,
    filtered_topics_json: Path | None = None,
    topic_domains_json: Path | None = None,
    expected_label_count: int | None = None,
    require_exact_domain_labels: bool = False,
) -> dict[str, Any]:
    """Train the topic model and write the reusable inference bundle.

    The train/test CSV text is treated as already prepared. Runtime
    preprocessing artifacts are still exported so later classification can apply
    the same token filtering policy to newly loaded repositories.
    """
    train_csv = train_csv.resolve()
    test_csv = test_csv.resolve()
    output_dir = output_dir.resolve()
    sample_manifest = sample_manifest.resolve() if sample_manifest is not None else None
    filtered_topics_json = (
        filtered_topics_json.resolve() if filtered_topics_json is not None else None
    )
    topic_domains_json = (
        topic_domains_json.resolve() if topic_domains_json is not None else None
    )
    raw_text_token_counts, raw_file_name_token_counts = _resolve_raw_frequency_paths(
        raw_text_token_counts=raw_text_token_counts,
        raw_file_name_token_counts=raw_file_name_token_counts,
    )
    if require_raw_frequency_artifacts and (
        raw_text_token_counts is None or raw_file_name_token_counts is None
    ):
        raise FileNotFoundError(
            "Raw corpus frequency artifacts are required for paper-equivalent "
            "runtime filtering, but they were not found. Provide "
            "--raw-text-token-counts and --raw-file-name-token-counts, or place "
            f"them at {DEFAULT_RAW_TEXT_TOKEN_COUNTS} and "
            f"{DEFAULT_RAW_FILE_NAME_TOKEN_COUNTS}."
        )
    run_id = _run_id("topic_model_training")
    run_root = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_dir, model_name)
    deps = _load_training_dependencies()

    log(f"Train CSV: {train_csv}")
    log(f"Test CSV: {test_csv}")
    log(f"Run id: {run_id}")
    log(f"Output directory: {output_dir}")
    if raw_text_token_counts is not None and raw_file_name_token_counts is not None:
        raw_text_token_counts = raw_text_token_counts.resolve()
        raw_file_name_token_counts = raw_file_name_token_counts.resolve()
        log(f"Raw text token counts: {raw_text_token_counts}")
        log(f"Raw file-name token counts: {raw_file_name_token_counts}")
    else:
        log(
            "Raw source-specific frequency artifacts: not provided; "
            "falling back to prepared train/test CSV token vocabulary because "
            "raw artifact enforcement was explicitly disabled."
        )
    header_topic_labels = _validate_headers(train_csv, test_csv)
    log(f"Topic label count: {len(header_topic_labels)}")
    label_guard = _validate_training_label_guards(
        topic_labels=header_topic_labels,
        sample_manifest=sample_manifest,
        filtered_topics_json=filtered_topics_json,
        topic_domains_json=topic_domains_json,
        expected_label_count=expected_label_count,
        require_exact_domain_labels=require_exact_domain_labels,
    )
    if label_guard["enabled"]:
        log(
            "Training label guard: "
            f"labels={label_guard['label_count']}, "
            f"expected={label_guard.get('expected_label_count') or 'not-set'}, "
            f"sample_manifest={bool(sample_manifest)}, "
            f"filtered_topics={label_guard.get('filtered_topics_count', 'not-set')}, "
            f"domain_topics={label_guard.get('domain_topic_count', 'not-set')}, "
            f"exact_domain_labels={require_exact_domain_labels}"
        )
    prepared_text_token_counts: Counter[str] = Counter()
    raw_text_token_count_values: Counter[str] | None = None
    raw_file_name_token_count_values: Counter[str] | None = None
    if raw_text_token_counts is not None and raw_file_name_token_counts is not None:
        raw_text_token_count_values = _load_token_count_artifact(
            raw_text_token_counts,
            label="Raw text",
        )
        raw_file_name_token_count_values = _load_token_count_artifact(
            raw_file_name_token_counts,
            label="Raw file-name",
        )
    data_preparation_profile = _load_data_preparation_profile_from_raw_artifacts(
        raw_text_token_counts=raw_text_token_counts,
        raw_file_name_token_counts=raw_file_name_token_counts,
    )
    if data_preparation_profile:
        log(
            "Data-preparation profile metadata: "
            f"{data_preparation_profile.get('root')}"
        )

    log("Loading training data")
    x_train_texts, y_train, topic_labels = _load_dataset(
        deps,
        train_csv,
        header_topic_labels,
    )
    train_row_count = int(y_train.shape[0])
    model_label_guard = _validate_training_label_guards(
        topic_labels=topic_labels,
        sample_manifest=sample_manifest,
        filtered_topics_json=filtered_topics_json,
        topic_domains_json=topic_domains_json,
        expected_label_count=expected_label_count,
        require_exact_domain_labels=require_exact_domain_labels,
    )
    log(f"Training rows: {train_row_count}")
    _update_prepared_text_token_counts(prepared_text_token_counts, x_train_texts)

    vectorizer = _build_original_tfidf_vectorizer(
        deps,
        max_features=max_features,
    )
    log("Fitting TF-IDF vectorizer")
    x_train = vectorizer.fit_transform(x_train_texts)
    del x_train_texts

    classifier = _build_original_multilabel_classifier(deps)
    log("Training OneVsRest LogisticRegression classifier")
    classifier.fit(x_train, y_train)
    del x_train
    del y_train

    log("Loading test data")
    x_test_texts, y_test, test_topic_labels = _load_dataset(
        deps,
        test_csv,
        header_topic_labels,
    )
    if test_topic_labels != topic_labels:
        raise ValueError(
            "Train/test label column order differs after applying the original "
            "pandas columns.difference([text_col]) ordering."
        )
    test_row_count = int(y_test.shape[0])
    log(f"Test rows: {test_row_count}")
    _update_prepared_text_token_counts(prepared_text_token_counts, x_test_texts)

    log("Transforming test data")
    x_test = vectorizer.transform(x_test_texts)
    del x_test_texts

    log("Scoring test data")
    y_scores = classifier.predict_proba(x_test)
    del x_test

    log("Evaluating paper-style top-k metrics")
    metrics = _evaluate_paper_metrics(deps, y_test, y_scores, top_k)
    del y_test
    del y_scores

    preprocessing_artifacts = _build_preprocessing_artifacts(
        prepared_text_token_counts=prepared_text_token_counts,
        train_csv=train_csv,
        test_csv=test_csv,
        train_row_count=train_row_count,
        test_row_count=test_row_count,
        raw_text_token_counts=raw_text_token_count_values,
        raw_file_name_token_counts=raw_file_name_token_count_values,
        raw_text_token_counts_path=raw_text_token_counts,
        raw_file_name_token_counts_path=raw_file_name_token_counts,
        data_preparation_profile=data_preparation_profile,
    )
    del prepared_text_token_counts

    created_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    training_config = {
        "run_id": run_id,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "algorithm": "tfidf_one_vs_rest_logistic_regression",
        "text_column": TEXT_COLUMN,
        "labels_column": LABELS_COLUMN,
        "primary_top_k": metrics["primary_top_k"],
        "top_k_values": metrics["top_k_values"],
        "tfidf": {
            "stop_words": "english",
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "analyzer": "word",
            "token_pattern": r"\w{2,}",
            "ngram_range": list(DEFAULT_NGRAM_RANGE),
            "max_features": max(1, int(max_features)),
        },
        "classifier": {
            "wrapper": "OneVsRestClassifier",
            "estimator": "LogisticRegression",
            "n_jobs": -1,
            "class_weight": ORIGINAL_RECOMMENDER_CLASS_WEIGHTS,
        },
        "runtime_frequency_filtering": {
            "require_raw_frequency_artifacts": bool(require_raw_frequency_artifacts),
            "raw_text_token_counts": (
                str(raw_text_token_counts) if raw_text_token_counts is not None else None
            ),
            "raw_file_name_token_counts": (
                str(raw_file_name_token_counts)
                if raw_file_name_token_counts is not None
                else None
            ),
            "paper_equivalence": preprocessing_artifacts["frequency_policy"][
                "paper_equivalence"
            ],
        },
        "label_guard": model_label_guard,
        "data_preparation": TRAINING_DATA_PREPARATION_METADATA,
        "source_compatibility_notes": SOURCE_COMPATIBILITY_NOTES,
        "original_recommender_reference": {
            "repository_url": ORIGINAL_RECOMMENDER_REPOSITORY_URL,
            "training_script_url": ORIGINAL_RECOMMENDER_TRAINING_SCRIPT_URL,
            "source_file_available": False,
            "source_file_note": (
                "Only data-preparation artifacts are vendored in this repository; "
                "the original machine-learning script is referenced from the "
                "upstream GitHub repository."
            ),
            "model": ORIGINAL_RECOMMENDER_MODEL,
            "tokenizer": ORIGINAL_RECOMMENDER_TOKENIZER,
            "method": ORIGINAL_RECOMMENDER_METHOD,
            "feature_mode": ORIGINAL_RECOMMENDER_FEATURE_MODE,
            "tuning_mode": ORIGINAL_RECOMMENDER_TUNING_MODE,
            "class_weights": ORIGINAL_RECOMMENDER_CLASS_WEIGHTS,
            "k_list": list(ORIGINAL_RECOMMENDER_K_LIST),
            "thresholds": list(ORIGINAL_RECOMMENDER_THRESHOLDS),
            "dataframe_shaping": [
                "pd.read_csv(path)",
                "df.drop(columns=['labels'])",
                "X = df[[text_col]]",
                "y = df[validated_topic_label_columns]",
                "X[text_col].values.astype('U')",
            ],
            "label_order_note": (
                "This wrapper preserves validated CSV topic-label order instead of "
                "using pandas columns.difference so saved model labels cannot drift "
                "from the retained sampled label universe."
            ),
        },
        "paper_alignment": {
            "dataset": (
                "Uses the paper/recommender train-test CSVs with preprocessed "
                "repository text in the text column and one binary column per topic. "
                "The text column is not reprocessed before vectorization, matching "
                "the original traditional_classifiers.py training path."
            ),
            "input_text_sources": [
                "project_name",
                "description",
                "readme",
                "wiki",
                "file_names",
            ],
            "evaluation_metrics": [
                "R@1",
                "P@1",
                "F1@1",
                "S@1",
                "R@2",
                "P@2",
                "F1@2",
                "S@2",
                "R@3",
                "P@3",
                "F1@3",
                "S@3",
                "R@5",
                "P@5",
                "F1@5",
                "S@5",
                "R@8",
                "P@8",
                "F1@8",
                "S@8",
                "R@10",
                "P@10",
                "F1@10",
                "S@10",
                "R@k",
                "P@k",
                "F1@k",
                "S@k",
                "LRAP",
            ],
            "non_paper_metrics_excluded": [
                "exact_match_accuracy",
                "hamming_accuracy",
            ],
        },
    }
    source_files = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }
    source_checksums = {
        "train_csv_sha256": _sha256_file(train_csv),
        "test_csv_sha256": _sha256_file(test_csv),
    }
    if raw_text_token_counts is not None and raw_file_name_token_counts is not None:
        source_files.update(
            {
                "raw_text_token_counts": str(raw_text_token_counts),
                "raw_file_name_token_counts": str(raw_file_name_token_counts),
            }
        )
        source_checksums.update(
            {
                "raw_text_token_counts_sha256": _sha256_file(raw_text_token_counts),
                "raw_file_name_token_counts_sha256": _sha256_file(
                    raw_file_name_token_counts
                ),
            }
        )
    row_counts = {
        "train_row_count": train_row_count,
        "test_row_count": test_row_count,
        "label_count": len(topic_labels),
    }
    dependency_versions = _dependency_versions(deps)
    bundle_metadata = {
        "schema_version": "topic_model_bundle_v2",
        "created_at_utc": created_at_utc,
        "run_id": run_id,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "source_files": source_files,
        "source_checksums": source_checksums,
        "row_counts": row_counts,
        "training_config": training_config,
        "evaluation_metrics": metrics,
        "preprocessing_artifacts_summary": {
            "schema_version": preprocessing_artifacts["schema_version"],
            "source": preprocessing_artifacts["source"],
            "allowed_runtime_token_count": preprocessing_artifacts[
                "allowed_runtime_token_count"
            ],
            "allowed_runtime_tokens_sha256": preprocessing_artifacts[
                "allowed_runtime_tokens_sha256"
            ],
            "allowed_text_token_count": preprocessing_artifacts[
                "allowed_text_token_count"
            ],
            "allowed_text_tokens_sha256": preprocessing_artifacts[
                "allowed_text_tokens_sha256"
            ],
            "allowed_file_name_token_count": preprocessing_artifacts[
                "allowed_file_name_token_count"
            ],
            "allowed_file_name_tokens_sha256": preprocessing_artifacts[
                "allowed_file_name_tokens_sha256"
            ],
            "file_name_informative_split_token_count": preprocessing_artifacts[
                "file_name_informative_split_token_count"
            ],
            "file_name_informative_split_tokens_source": preprocessing_artifacts[
                "file_name_informative_split_tokens_source"
            ],
            "frequency_policy": preprocessing_artifacts["frequency_policy"],
            "separate_source_frequency_filters_available": preprocessing_artifacts[
                "separate_source_frequency_filters_available"
            ],
            "raw_corpus_frequency_artifacts_available": preprocessing_artifacts[
                "raw_corpus_frequency_artifacts_available"
            ],
            "data_preparation_profile": preprocessing_artifacts.get(
                "data_preparation_profile"
            ),
        },
        "dependency_versions": dependency_versions,
    }
    bundle = {
        "schema_version": "topic_model_bundle_v2",
        "vectorizer": vectorizer,
        "classifier": classifier,
        "topic_labels": topic_labels,
        "preprocessing_artifacts": preprocessing_artifacts,
        "metadata": bundle_metadata,
    }

    paths.bundle.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing model bundle: {paths.bundle}")
    deps.joblib.dump(bundle, paths.bundle, compress=3)
    log(f"Writing topic labels: {paths.labels}")
    _write_json(paths.labels, topic_labels)
    alphabetical_topics = _alphabetical_topics(topic_labels)
    log(f"Writing alphabetical topic labels: {paths.labels_alphabetical_json}")
    _write_json(paths.labels_alphabetical_json, alphabetical_topics)
    log(f"Writing alphabetical topic labels text: {paths.labels_alphabetical_txt}")
    _write_text_lines(paths.labels_alphabetical_txt, alphabetical_topics)
    log(f"Writing preprocessing artifacts: {paths.preprocessing_artifacts}")
    _write_json(paths.preprocessing_artifacts, preprocessing_artifacts)
    metrics_payload = {
        "schema_version": "topic_model_metrics_v1",
        "created_at_utc": created_at_utc,
        "row_counts": row_counts,
        "evaluation_metrics": metrics,
    }
    log(f"Writing metrics: {paths.metrics}")
    _write_json(paths.metrics, metrics_payload)

    artifact_paths = {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "bundle": str(paths.bundle),
        "labels": str(paths.labels),
        "labels_alphabetical_json": str(paths.labels_alphabetical_json),
        "labels_alphabetical_txt": str(paths.labels_alphabetical_txt),
        "preprocessing_artifacts": str(paths.preprocessing_artifacts),
        "metrics": str(paths.metrics),
        "manifest": str(paths.manifest),
    }
    artifact_sizes = {
        "bundle_bytes": paths.bundle.stat().st_size,
        "labels_bytes": paths.labels.stat().st_size,
        "labels_alphabetical_json_bytes": paths.labels_alphabetical_json.stat().st_size,
        "labels_alphabetical_txt_bytes": paths.labels_alphabetical_txt.stat().st_size,
        "preprocessing_artifacts_bytes": paths.preprocessing_artifacts.stat().st_size,
        "metrics_bytes": paths.metrics.stat().st_size,
    }
    artifact_checksums = {
        "bundle_sha256": _sha256_file(paths.bundle),
        "labels_sha256": _sha256_file(paths.labels),
        "labels_alphabetical_json_sha256": _sha256_file(paths.labels_alphabetical_json),
        "labels_alphabetical_txt_sha256": _sha256_file(paths.labels_alphabetical_txt),
        "preprocessing_artifacts_sha256": _sha256_file(paths.preprocessing_artifacts),
        "metrics_sha256": _sha256_file(paths.metrics),
    }
    manifest = {
        "schema_version": "topic_model_manifest_v1",
        "created_at_utc": created_at_utc,
        "run_id": run_id,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "source_files": source_files,
        "source_checksums": source_checksums,
        "artifact_paths": artifact_paths,
        "artifact_sizes": artifact_sizes,
        "artifact_checksums": artifact_checksums,
        "row_counts": row_counts,
        "training_config": training_config,
        "evaluation_metrics": metrics,
        "preprocessing_artifacts_summary": bundle_metadata[
            "preprocessing_artifacts_summary"
        ],
        "label_guard": model_label_guard,
        "dependency_versions": dependency_versions,
    }
    log(f"Writing manifest: {paths.manifest}")
    _write_json(paths.manifest, manifest)

    log(
        "Evaluation: "
        + ", ".join(
            f"{key}={value:.4f}"
            for key, value in metrics.items()
            if isinstance(value, float)
        )
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone training command arguments."""
    parser = argparse.ArgumentParser(
        description="Train and export the repository topic-classification model."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help=f"Training CSV path. Default: {DEFAULT_TRAIN_CSV}",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help=f"Test CSV path. Default: {DEFAULT_TEST_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for exported model artifacts. Default: "
            f"{DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model name stored in metadata. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-k cutoff for evaluation metrics. Default: {DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help=f"Maximum TF-IDF feature count. Default: {DEFAULT_MAX_FEATURES}",
    )
    parser.add_argument(
        "--raw-text-token-counts",
        type=Path,
        default=None,
        help=(
            "Optional raw corpus token-count artifact for description/README/wiki "
            "text. Supports JSON, CSV, TSV, or one-token-per-line text. Default: "
            f"use {DEFAULT_RAW_TEXT_TOKEN_COUNTS} if it exists."
        ),
    )
    parser.add_argument(
        "--raw-file-name-token-counts",
        type=Path,
        default=None,
        help=(
            "Optional raw corpus token-count artifact for repository file names. "
            "Supports JSON, CSV, TSV, or one-token-per-line text. Default: use "
            f"{DEFAULT_RAW_FILE_NAME_TOKEN_COUNTS} if it exists."
        ),
    )
    parser.add_argument(
        "--require-raw-frequency-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require raw text and file-name token-count artifacts for "
            "paper-equivalent runtime frequency filtering. Disabled by default "
            "because the upstream release does not include these artifacts; "
            "default training uses the prepared train/test CSV vocabulary and "
            "records that it is not raw-corpus equivalent."
        ),
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help=(
            "Optional sampled-training-data manifest. When provided, train/test "
            "CSV label columns must exactly match retained_label_universe."
        ),
    )
    parser.add_argument(
        "--filtered-topics-json",
        type=Path,
        default=None,
        help=(
            "Optional filtered-topics JSON. When provided, training fails if any "
            "filtered topic is present as a label column."
        ),
    )
    parser.add_argument(
        "--topic-domains-json",
        type=Path,
        default=None,
        help=(
            "Optional topic domain mapping JSON. When provided, every training "
            "label must be present in topic_domains."
        ),
    )
    parser.add_argument(
        "--expected-label-count",
        type=int,
        default=None,
        help=(
            "Optional exact retained topic count guard, for example 384. "
            "Training fails if the CSV label count differs."
        ),
    )
    parser.add_argument(
        "--require-exact-domain-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require the training label set to exactly match topic_domains keys, "
            "not just be a subset."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run standalone model training and return a process exit code."""
    args = parse_args(argv)
    try:
        train_topic_model(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            model_name=args.model_name,
            top_k=args.top_k,
            max_features=args.max_features,
            raw_text_token_counts=args.raw_text_token_counts,
            raw_file_name_token_counts=args.raw_file_name_token_counts,
            require_raw_frequency_artifacts=args.require_raw_frequency_artifacts,
            sample_manifest=args.sample_manifest,
            filtered_topics_json=args.filtered_topics_json,
            topic_domains_json=args.topic_domains_json,
            expected_label_count=args.expected_label_count,
            require_exact_domain_labels=args.require_exact_domain_labels,
        )
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1
    return 0
