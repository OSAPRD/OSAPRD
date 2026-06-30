"""Repository topic-classification streaming entrypoint.

Stage 4 scans curation outputs, loads repository context lazily, optionally
enriches missing README/wiki text, builds model features, applies the trained
topic model, and writes streaming JSONL outputs plus a manifest.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
POST_PROCESSING_DIR = Path(__file__).resolve().parents[2]
UTILITY_DIR = POST_PROCESSING_DIR / "utility"
TOPIC_DIR = Path(__file__).resolve().parent
DEFAULT_TOPIC_DOMAIN_MAPPING_PATH = TOPIC_DIR / "topic_domains.json"
POST_PROCESSING_TOKENS_CONFIG_PATH = POST_PROCESSING_DIR / "config" / "tokens_config.py"
PREDICTION_RETENTION_POLICY = "all_above_threshold_after_filtering"
EFFECTIVE_INCLUDE_RAW_PREDICTIONS = False
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))
if str(TOPIC_DIR) not in sys.path:
    sys.path.insert(0, str(TOPIC_DIR))

from topic_loader import LoaderStats, TopicClassificationInputLoader, log
from config_loader import load_topic_classification_config
from config_values import (
    bool_config,
    positive_int,
    resolve_github_tokens,
    string_list,
    utc_timestamp_run_id,
)
from topic_classifier import TopicPredictionResult, load_topic_model_bundle
from topic_domains import load_topic_domain_mapping
from topic_features import (
    RepositoryFeatures,
    RepositoryFeatureSources,
    build_repository_features,
    resolve_repository_feature_sources,
)
from topic_outputs import TopicClassificationOutputWriter
from topic_preprocessing import get_preprocessor_for_model
from readme_enrichment import ReadmeFetchResult, TopicReadmeEnricher
from wiki_enrichment import TopicWikiEnricher


@dataclass(frozen=True)
class RepositoryClassificationResult:
    """Pickleable repository classification result returned by a worker."""

    ref: Any
    status: str
    stage: str | None = None
    message: str | None = None
    repository_file_count: int = 0
    metrics_payload_count: int = 0
    pr_payload_count: int = 0
    readme_status: str | None = None
    wiki_status: str | None = None
    filter_source_stats: dict[str, Any] | None = None
    features: RepositoryFeatures | None = None
    prediction_result: TopicPredictionResult | None = None


_WIKI_STAT_KEYS = (
    "cache_hit",
    "fetched",
    "missing",
    "not_requested",
    "auth_failed",
    "rate_limited",
    "fetch_failed",
)
_README_STAT_KEYS = (
    "input",
    "cache_hit",
    "fetched",
    "missing",
    "auth_failed",
    "rate_limited",
    "fetch_failed",
    "decode_failed",
)
_WORKER_LOADER: TopicClassificationInputLoader | None = None
_WORKER_README_ENRICHER: TopicReadmeEnricher | None = None
_WORKER_WIKI_ENRICHER: TopicWikiEnricher | None = None
_WORKER_CLASSIFIER: Any | None = None
_WORKER_PREPROCESSOR: Any | None = None
_WORKER_PREDICTION_SCORE_THRESHOLD = 0.7
_WORKER_EXCLUDED_TOPICS: tuple[str, ...] = ()
_WORKER_TOPIC_DOMAIN_MAP: dict[str, str] = {}
_WORKER_INCLUDE_RAW_PREDICTIONS = False


def _repository_failure(ref: Any, *, stage: str, message: str) -> dict[str, str]:
    return {
        "cohort": str(getattr(ref, "cohort", "")),
        "repository_owner": str(getattr(ref, "repository_owner", "")),
        "repository_name": str(getattr(ref, "repository_name", "")),
        "repository_key": str(getattr(ref, "repository_key", "")),
        "repository_id": str(getattr(ref, "repository_id", "") or ""),
        "repository_identity_key": str(getattr(ref, "repository_identity_key", "") or ""),
        "safe_repository_key": str(getattr(ref, "safe_repository_key", "")),
        "stage": stage,
        "message": message,
        "file_list_path": str(getattr(ref, "file_list_path", "")),
    }


def _empty_wiki_stats() -> dict[str, int]:
    return {key: 0 for key in _WIKI_STAT_KEYS}


def _empty_readme_stats() -> dict[str, int]:
    return {key: 0 for key in _README_STAT_KEYS}


def _runtime_repository_filter(
    *,
    ref: Any,
    context: Any,
    sources: RepositoryFeatureSources,
    readme_result: ReadmeFetchResult | None = None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Return a runtime exclusion reason before expensive wiki/model work."""
    description = str(sources.description or "")
    readme = str(sources.readme or "")
    source_stats = {
        "repository_identity_key": str(getattr(ref, "repository_identity_key", "") or ""),
        "description_character_count": len(description),
        "description_present": bool(description.strip()),
        "readme_character_count": len(readme),
        "readme_present": bool(readme.strip()),
        "readme_status": readme_result.status if readme_result is not None else "input",
        "readme_cache_path": readme_result.cache_path if readme_result is not None else None,
        "readme_metadata_path": (
            readme_result.metadata_path if readme_result is not None else None
        ),
        "readme_fetch_notes": readme_result.notes if readme_result is not None else None,
        "metadata_candidate_count": len(sources.metadata_candidates),
        "file_count": len(getattr(context, "repository_files", ()) or ()),
        "has_file_list": bool(getattr(ref, "has_file_list", True)),
        "file_list_path": str(getattr(ref, "file_list_path", "") or ""),
        "parquet_pr_count": int(getattr(ref, "parquet_pr_count", 0) or 0),
    }
    if not readme.strip():
        return "no_readme", "repository README text is empty or missing", source_stats
    return None, None, source_stats


def _initialize_topic_classification_worker(
    model_bundle_path: Path,
    prediction_score_threshold: float,
    excluded_topics: tuple[str, ...],
    topic_domain_map: dict[str, str],
    include_raw_predictions: bool,
    enable_live_wiki_fetch: bool,
    wiki_cache_dir: Path,
    readme_cache_dir: Path,
    github_tokens: tuple[str, ...],
    identifier_splitter_mode: str,
    allow_heuristic_file_splits: bool,
    data_preparation_root: Path,
) -> None:
    """Initialize one process-local classifier and preprocessing stack."""
    global _WORKER_CLASSIFIER
    global _WORKER_EXCLUDED_TOPICS
    global _WORKER_INCLUDE_RAW_PREDICTIONS
    global _WORKER_LOADER
    global _WORKER_PREPROCESSOR
    global _WORKER_README_ENRICHER
    global _WORKER_PREDICTION_SCORE_THRESHOLD
    global _WORKER_TOPIC_DOMAIN_MAP
    global _WORKER_WIKI_ENRICHER

    _WORKER_CLASSIFIER = load_topic_model_bundle(Path(model_bundle_path))
    _WORKER_PREPROCESSOR = get_preprocessor_for_model(
        _WORKER_CLASSIFIER.preprocessing_artifacts,
        identifier_splitter_mode=identifier_splitter_mode,
        allow_heuristic_file_splits=allow_heuristic_file_splits,
        data_preparation_root=Path(data_preparation_root),
    )
    _WORKER_WIKI_ENRICHER = TopicWikiEnricher(
        cache_dir=Path(wiki_cache_dir),
        enable_live_fetch=enable_live_wiki_fetch,
        tokens=github_tokens,
        log=log,
    )
    _WORKER_README_ENRICHER = TopicReadmeEnricher(
        cache_dir=Path(readme_cache_dir),
        tokens=github_tokens,
        log=log,
    )
    _WORKER_LOADER = TopicClassificationInputLoader(Path("."))
    _WORKER_PREDICTION_SCORE_THRESHOLD = float(prediction_score_threshold)
    _WORKER_EXCLUDED_TOPICS = tuple(excluded_topics)
    _WORKER_TOPIC_DOMAIN_MAP = dict(topic_domain_map)
    _WORKER_INCLUDE_RAW_PREDICTIONS = include_raw_predictions


def _classify_repository(
    ref: Any,
    *,
    loader: TopicClassificationInputLoader,
    readme_enricher: TopicReadmeEnricher,
    wiki_enricher: TopicWikiEnricher,
    classifier: Any,
    preprocessor: Any,
    prediction_score_threshold: float,
    excluded_topics: tuple[str, ...],
    topic_domain_map: dict[str, str],
    include_raw_predictions: bool,
) -> RepositoryClassificationResult:
    try:
        context = loader.load_repository_context(ref)
    except Exception as exc:
        return RepositoryClassificationResult(
            ref=ref,
            status="failed",
            stage="file_list",
            message=str(exc),
        )

    repository_file_count = len(context.repository_files)
    try:
        resolved_sources = resolve_repository_feature_sources(context)
    except Exception as exc:
        return RepositoryClassificationResult(
            ref=ref,
            status="failed",
            stage="feature_extraction",
            message=str(exc),
            repository_file_count=repository_file_count,
        )

    readme_result: ReadmeFetchResult | None = None
    if resolved_sources.readme.strip():
        readme_enricher.record_input_readme()
        readme_result = ReadmeFetchResult(status="input", text=resolved_sources.readme)
    else:
        try:
            readme_result = readme_enricher.get_readme_text(
                str(resolved_sources.owner),
                str(resolved_sources.name),
                ref=str(resolved_sources.source_ref or "").strip() or None,
            )
            if readme_result.text.strip():
                resolved_sources = replace(resolved_sources, readme=readme_result.text)
        except Exception as exc:
            readme_result = ReadmeFetchResult(status="fetch_failed", notes=str(exc))

    filter_reason, filter_message, filter_source_stats = _runtime_repository_filter(
        ref=ref,
        context=context,
        sources=resolved_sources,
        readme_result=readme_result,
    )
    if filter_reason is not None:
        return RepositoryClassificationResult(
            ref=ref,
            status="filtered",
            stage=filter_reason,
            message=filter_message,
            repository_file_count=repository_file_count,
            metrics_payload_count=resolved_sources.metrics_payload_count,
            pr_payload_count=resolved_sources.pr_payload_count,
            readme_status=readme_result.status if readme_result is not None else None,
            filter_source_stats=filter_source_stats,
        )

    wiki_status: str | None = None
    try:
        wiki_result = wiki_enricher.get_wiki_text(
            ref.repository_owner,
            ref.repository_name,
        )
        wiki_status = wiki_result.status
    except Exception as exc:
        return RepositoryClassificationResult(
            ref=ref,
            status="failed",
            stage="wiki_enrichment",
            message=str(exc),
            repository_file_count=repository_file_count,
            metrics_payload_count=resolved_sources.metrics_payload_count,
            pr_payload_count=resolved_sources.pr_payload_count,
            readme_status=readme_result.status if readme_result is not None else None,
        )

    try:
        features = build_repository_features(
            context,
            readme_status=readme_result.status if readme_result is not None else "input",
            wiki_text=wiki_result.text,
            wiki_status=wiki_result.status,
            preprocessor=preprocessor,
            resolved_sources=resolved_sources,
        )
    except Exception as exc:
        return RepositoryClassificationResult(
            ref=ref,
            status="failed",
            stage="feature_extraction",
            message=str(exc),
            repository_file_count=repository_file_count,
            readme_status=readme_result.status if readme_result is not None else None,
            wiki_status=wiki_status,
        )

    try:
        prediction_result = classifier.predict(
            features.inference_text,
            score_threshold=prediction_score_threshold,
            excluded_topics=excluded_topics,
            retained_topics=topic_domain_map.keys(),
            include_raw_predictions=include_raw_predictions,
        ).with_topic_domains(topic_domain_map)
    except Exception as exc:
        return RepositoryClassificationResult(
            ref=ref,
            status="failed",
            stage="classification",
            message=str(exc),
            repository_file_count=repository_file_count,
            metrics_payload_count=features.metrics_payload_count,
            pr_payload_count=features.pr_payload_count,
            readme_status=readme_result.status if readme_result is not None else None,
            wiki_status=wiki_status,
        )

    return RepositoryClassificationResult(
        ref=ref,
        status="classified",
        repository_file_count=repository_file_count,
        metrics_payload_count=features.metrics_payload_count,
        pr_payload_count=features.pr_payload_count,
        readme_status=readme_result.status if readme_result is not None else None,
        wiki_status=wiki_status,
        features=features,
        prediction_result=prediction_result,
    )


def _classify_repository_worker(ref: Any) -> RepositoryClassificationResult:
    if (
        _WORKER_LOADER is None
        or _WORKER_WIKI_ENRICHER is None
        or _WORKER_README_ENRICHER is None
        or _WORKER_CLASSIFIER is None
        or _WORKER_PREPROCESSOR is None
    ):
        raise RuntimeError("Topic-classification worker was not initialized.")
    return _classify_repository(
        ref,
        loader=_WORKER_LOADER,
        readme_enricher=_WORKER_README_ENRICHER,
        wiki_enricher=_WORKER_WIKI_ENRICHER,
        classifier=_WORKER_CLASSIFIER,
        preprocessor=_WORKER_PREPROCESSOR,
        prediction_score_threshold=_WORKER_PREDICTION_SCORE_THRESHOLD,
        excluded_topics=_WORKER_EXCLUDED_TOPICS,
        topic_domain_map=_WORKER_TOPIC_DOMAIN_MAP,
        include_raw_predictions=_WORKER_INCLUDE_RAW_PREDICTIONS,
    )


def _iter_parallel_classification_results(
    repository_refs: tuple[Any, ...],
    *,
    workers: int,
    model_bundle_path: Path,
    prediction_score_threshold: float,
    excluded_topics: tuple[str, ...],
    topic_domain_map: dict[str, str],
    include_raw_predictions: bool,
    enable_live_wiki_fetch: bool,
    wiki_cache_dir: Path,
    readme_cache_dir: Path,
    github_tokens: tuple[str, ...],
    identifier_splitter_mode: str,
    allow_heuristic_file_splits: bool,
    data_preparation_root: Path,
) -> Any:
    max_workers = max(1, int(workers))
    max_pending = max_workers * 4
    refs_iter = iter(repository_refs)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_initialize_topic_classification_worker,
        initargs=(
            Path(model_bundle_path),
            prediction_score_threshold,
            tuple(excluded_topics),
            dict(topic_domain_map),
            include_raw_predictions,
            enable_live_wiki_fetch,
            Path(wiki_cache_dir),
            Path(readme_cache_dir),
            tuple(github_tokens),
            identifier_splitter_mode,
            allow_heuristic_file_splits,
            Path(data_preparation_root),
        ),
    ) as executor:
        futures: dict[Any, Any] = {}

        def submit_until_full() -> None:
            while len(futures) < max_pending:
                try:
                    ref = next(refs_iter)
                except StopIteration:
                    return
                futures[executor.submit(_classify_repository_worker, ref)] = ref

        submit_until_full()
        while futures:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                ref = futures.pop(future)
                try:
                    yield future.result()
                except Exception as exc:
                    yield RepositoryClassificationResult(
                        ref=ref,
                        status="failed",
                        stage="worker",
                        message=f"{type(exc).__name__}: {exc}",
                    )
            submit_until_full()


def _apply_classification_result(
    result: RepositoryClassificationResult,
    *,
    writer: TopicClassificationOutputWriter,
    classifier_info: dict[str, Any],
    prediction_score_threshold: float,
    prediction_retention_policy: str,
    include_raw_predictions: bool,
    stats: LoaderStats,
    readme_stats: Counter[str],
    wiki_stats: Counter[str],
    repository_failures: list[dict[str, str]],
) -> dict[str, int]:
    updates = {
        "classified_repositories": 0,
        "classification_failures": 0,
        "feature_extraction_failures": 0,
        "wiki_enrichment_failures": 0,
        "worker_failures": 0,
        "repository_pr_map_rows": 0,
        "filtered_repositories": 0,
        "filtered_no_readme": 0,
        "filtered_no_description": 0,
    }
    stats.repository_files_loaded += result.repository_file_count
    stats.pr_payloads_loaded += result.pr_payload_count
    stats.metrics_payloads_loaded += result.metrics_payload_count
    if result.readme_status:
        readme_stats[result.readme_status] += 1
    if result.wiki_status:
        wiki_stats[result.wiki_status] += 1

    if result.status == "classified":
        if result.features is None or result.prediction_result is None:
            raise RuntimeError("Classified repository result is missing prediction payload.")
        writer.write_repository_topics(
            features=result.features,
            prediction_result=result.prediction_result,
            classifier_info=classifier_info,
            top_k=None,
            prediction_score_threshold=prediction_score_threshold,
            prediction_retention_policy=prediction_retention_policy,
            include_raw_predictions=include_raw_predictions,
        )
        updates["repository_pr_map_rows"] += writer.write_repository_pr_map(result.ref)
        updates["classified_repositories"] += 1
        return updates

    if result.status == "filtered":
        reason = result.stage or "runtime_filter"
        writer.write_filtered_repository(
            ref=result.ref,
            reason=reason,
            message=result.message or "",
            repository_file_count=result.repository_file_count,
            metrics_payload_count=result.metrics_payload_count,
            pr_payload_count=result.pr_payload_count,
            source_stats=result.filter_source_stats or {},
        )
        updates["filtered_repositories"] += 1
        if reason == "no_readme":
            updates["filtered_no_readme"] += 1
        elif reason == "no_description":
            updates["filtered_no_description"] += 1
        return updates

    stage = result.stage or "unknown"
    message = result.message or "unknown failure"
    repository_failures.append(
        _repository_failure(
            result.ref,
            stage=stage,
            message=message,
        )
    )
    if stage == "file_list":
        stats.file_list_parse_failures += 1
        log(f"Skipping repository file list {result.ref.file_list_path}: {message}")
    elif stage == "feature_extraction":
        updates["feature_extraction_failures"] += 1
        log(
            f"Skipping repository {result.ref.repository_key}; "
            f"feature extraction failed: {message}"
        )
    elif stage == "classification":
        updates["classification_failures"] += 1
        log(
            f"Skipping repository {result.ref.repository_key}; "
            f"classification failed: {message}"
        )
    elif stage == "wiki_enrichment":
        updates["wiki_enrichment_failures"] += 1
        log(
            f"Skipping repository {result.ref.repository_key}; "
            f"wiki enrichment failed: {message}"
        )
    else:
        updates["worker_failures"] += 1
        log(
            f"Skipping repository {result.ref.repository_key}; "
            f"{stage} failed: {message}"
        )
    return updates


def _topic_classification_run_paths(runs_root: Path, run_id: str) -> tuple[Path, Path]:
    """Return a fresh run root and its standardized output directory."""
    runs_root = Path(runs_root)
    base_run_root = runs_root / run_id
    run_root = base_run_root
    suffix = 1
    while run_root.exists():
        run_root = Path(f"{base_run_root}_{suffix}")
        suffix += 1
    return run_root, run_root / "output"


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
    if not rules_dir.is_dir() or not lists_dir.is_dir():
        raise FileNotFoundError(
            "Topic data-preparation profile is missing. Expected "
            f"{rules_dir} and {lists_dir}. Run "
            "`python post-processing/topic-classification/data-preparation/"
            "generate_github_topic_rules.py` or set "
            "POST_PROCESSING_TOPIC_CLASSIFICATION_DATA_PREPARATION_ROOT."
        )

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
        "skipped_topic_rule_names": ["low_freq_topics.csv"],
    }


def _preprocessing_artifact_summary(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": artifacts.get("schema_version"),
        "source": artifacts.get("source"),
        "allowed_runtime_token_count": artifacts.get("allowed_runtime_token_count"),
        "allowed_runtime_tokens_sha256": artifacts.get("allowed_runtime_tokens_sha256"),
        "allowed_text_token_count": artifacts.get("allowed_text_token_count"),
        "allowed_text_tokens_sha256": artifacts.get("allowed_text_tokens_sha256"),
        "allowed_file_name_token_count": artifacts.get("allowed_file_name_token_count"),
        "allowed_file_name_tokens_sha256": artifacts.get("allowed_file_name_tokens_sha256"),
        "raw_corpus_frequency_artifacts_available": artifacts.get(
            "raw_corpus_frequency_artifacts_available"
        ),
        "separate_source_frequency_filters_available": artifacts.get(
            "separate_source_frequency_filters_available"
        ),
        "frequency_policy": artifacts.get("frequency_policy"),
        "source_token_policy": artifacts.get("source_token_policy"),
        "data_preparation_profile": artifacts.get("data_preparation_profile"),
    }


def _merge_topic_lists(*topic_lists: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for topic_list in topic_lists:
        for topic in topic_list:
            text = str(topic).strip()
            key = text.lower()
            if text and key not in seen:
                merged.append(text)
                seen.add(key)
    return tuple(merged)


def _resolve_prediction_output_policy(topic_config: Any) -> dict[str, Any]:
    configured_top_k_topics = positive_int(
        getattr(topic_config, "TOPIC_CLASSIFICATION_TOP_K_TOPICS", 5),
        default=5,
    )
    try:
        prediction_score_threshold = float(
            getattr(topic_config, "TOPIC_CLASSIFICATION_PREDICTION_SCORE_THRESHOLD", 0.7)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TOPIC_CLASSIFICATION_PREDICTION_SCORE_THRESHOLD must be a float."
        ) from exc
    if prediction_score_threshold < 0.0 or prediction_score_threshold > 1.0:
        raise ValueError(
            "TOPIC_CLASSIFICATION_PREDICTION_SCORE_THRESHOLD must be between 0.0 and 1.0."
        )
    configured_include_raw_predictions = bool_config(
        getattr(topic_config, "TOPIC_CLASSIFICATION_OUTPUT_RAW_PREDICTIONS", False)
    )
    return {
        "prediction_retention_policy": PREDICTION_RETENTION_POLICY,
        "configured_top_k_topics": configured_top_k_topics,
        "top_k_topics": None,
        "prediction_score_threshold": prediction_score_threshold,
        "configured_include_raw_predictions": configured_include_raw_predictions,
        "include_raw_predictions": EFFECTIVE_INCLUDE_RAW_PREDICTIONS,
    }


def run_topic_classification_pipeline() -> dict[str, Any]:
    """Classify repositories in curation outputs using a trained topic model bundle."""
    topic_config = load_topic_classification_config(
        module_name="post_processing_topic_classification_config",
    )
    curation_outputs_dir = Path(getattr(topic_config, "CURATION_OUTPUTS_DIR"))
    exclude_dirs = tuple(getattr(topic_config, "CURATION_EXCLUDE_DIRS", ()))
    batch_size = positive_int(getattr(topic_config, "TOPIC_CLASSIFICATION_BATCH_SIZE", 10))
    classification_workers = positive_int(
        getattr(topic_config, "TOPIC_CLASSIFICATION_WORKERS", 1),
        default=1,
    )
    input_format = (
        str(getattr(topic_config, "TOPIC_CLASSIFICATION_INPUT_FORMAT", "parquet"))
        .strip()
        .lower()
        or "parquet"
    )
    longitudinal_only = bool(
        getattr(topic_config, "TOPIC_CLASSIFICATION_LONGITUDINAL_ONLY", False)
    )
    model_bundle_path = Path(getattr(topic_config, "TOPIC_CLASSIFICATION_MODEL_BUNDLE_PATH"))
    runs_root = Path(getattr(topic_config, "TOPIC_CLASSIFICATION_RUNS_DIR"))
    configured_run_id = str(getattr(topic_config, "TOPIC_CLASSIFICATION_RUN_ID", "")).strip()
    prediction_policy = _resolve_prediction_output_policy(topic_config)
    configured_top_k_topics = int(prediction_policy["configured_top_k_topics"])
    top_k_topics = prediction_policy["top_k_topics"]
    prediction_score_threshold = float(prediction_policy["prediction_score_threshold"])
    excluded_topics = string_list(
        getattr(topic_config, "TOPIC_CLASSIFICATION_EXCLUDED_TOPICS", ())
    )
    configured_include_raw_predictions = bool(prediction_policy["configured_include_raw_predictions"])
    include_raw_predictions = bool(prediction_policy["include_raw_predictions"])
    enable_live_wiki_fetch = bool_config(
        getattr(topic_config, "TOPIC_CLASSIFICATION_ENABLE_LIVE_WIKI_FETCH", True)
    )
    identifier_splitter_mode = (
        str(getattr(topic_config, "TOPIC_CLASSIFICATION_IDENTIFIER_SPLITTER", "spiral"))
        .strip()
        .lower()
        or "spiral"
    )
    require_preprocessing_artifacts = bool_config(
        getattr(
            topic_config,
            "TOPIC_CLASSIFICATION_REQUIRE_MODEL_PREPROCESSING_ARTIFACTS",
            True,
        )
    )
    require_raw_frequency_artifacts = bool_config(
        getattr(
            topic_config,
            "TOPIC_CLASSIFICATION_REQUIRE_RAW_FREQUENCY_ARTIFACTS",
            True,
        )
    )
    allow_heuristic_file_splits = bool_config(
        getattr(
            topic_config,
            "TOPIC_CLASSIFICATION_ALLOW_HEURISTIC_FILE_SPLITS",
            False,
        )
    )
    data_preparation_root = Path(
        getattr(
            topic_config,
            "TOPIC_CLASSIFICATION_DATA_PREPARATION_ROOT",
            TOPIC_DIR.parent / "data-preparation" / "generated" / "github-topics",
        )
    )
    data_preparation_profile = _data_preparation_profile_manifest(data_preparation_root)
    topic_domain_mapping_path = Path(
        getattr(
            topic_config,
            "TOPIC_CLASSIFICATION_TOPIC_DOMAIN_MAPPING_PATH",
            DEFAULT_TOPIC_DOMAIN_MAPPING_PATH,
        )
    )
    wiki_cache_dir = Path(getattr(topic_config, "TOPIC_CLASSIFICATION_WIKI_CACHE_DIR"))
    readme_cache_dir = Path(
        getattr(
            topic_config,
            "TOPIC_CLASSIFICATION_README_CACHE_DIR",
            wiki_cache_dir / "readme-cache",
        )
    )
    github_tokens = resolve_github_tokens(
        (),
        POST_PROCESSING_TOKENS_CONFIG_PATH,
    )
    run_id = configured_run_id or utc_timestamp_run_id("topic_classification")
    run_root, output_dir = _topic_classification_run_paths(runs_root, run_id)

    log(f"Input root: {curation_outputs_dir}")
    log(f"Batch size: {batch_size}")
    log(f"Worker processes: {classification_workers}")
    log(f"Input format: {input_format}")
    log(f"Longitudinal-only PR filter: {longitudinal_only}")
    log(f"Model bundle: {model_bundle_path}")
    log(f"Runs root: {runs_root}")
    log(f"Run root: {run_root}")
    log(f"Output directory: {output_dir}")
    log(
        "Prediction policy: "
        f"{PREDICTION_RETENTION_POLICY}, "
        f"score_threshold>{prediction_score_threshold} "
        f"(configured_top_k={configured_top_k_topics} ignored)"
    )
    log(f"Excluded topics: {len(excluded_topics)}")
    log(
        "Raw predictions enabled: "
        f"{include_raw_predictions} effective "
        f"(configured={configured_include_raw_predictions})"
    )
    log(f"Live wiki fetch enabled: {enable_live_wiki_fetch}")
    log("Live README fetch enabled: True (required for missing README inputs)")
    log(f"Identifier splitter: {identifier_splitter_mode}")
    log(f"Require model preprocessing artifacts: {require_preprocessing_artifacts}")
    log(f"Require raw frequency artifacts: {require_raw_frequency_artifacts}")
    log(f"Heuristic file-name splits enabled: {allow_heuristic_file_splits}")
    log(f"Data-preparation profile: {data_preparation_root}")
    log(f"Topic domain mapping: {topic_domain_mapping_path}")

    classifier = load_topic_model_bundle(model_bundle_path)
    topic_domain_mapping = load_topic_domain_mapping(
        topic_domain_mapping_path,
        topic_labels=classifier.topic_labels,
    )
    filtered_out_topics = topic_domain_mapping.filtered_topics_present
    filtered_out_topics_manifest = {
        "enabled": True,
        "path": topic_domain_mapping.manifest["filtered_topics_path"],
        "source": "sample-training-data/filtered_topics.json",
        "filter_order": "score_all_topics_then_filter_out_topics_then_filter_to_domain_mapped_topics_then_apply_threshold",
        "topic_count": len(filtered_out_topics),
        "filtered_topic_count": topic_domain_mapping.filtered_topic_count,
        "topics": list(filtered_out_topics),
    }
    topic_domain_map = topic_domain_mapping.topic_domains
    topic_domains_manifest = topic_domain_mapping.manifest
    effective_excluded_topics = _merge_topic_lists(excluded_topics, filtered_out_topics)
    log(
        "Topic domains: "
        f"{len(topic_domains_manifest['domains'])} domains covering "
        f"{topic_domains_manifest['topic_domain_count']} retained topics"
    )
    log(
        "Model labels mapped to domains: "
        f"{topic_domains_manifest['model_label_domain_mapped_count']}/"
        f"{topic_domains_manifest['model_label_count']} "
        f"(unmapped ignored={topic_domains_manifest['model_label_unmapped_count']})"
    )
    log(
        "Filtered-out topic filter: "
        f"{filtered_out_topics_manifest['topic_count']} catalog topics excluded before thresholding"
    )
    log(f"Effective excluded topics: {len(effective_excluded_topics)}")
    if require_preprocessing_artifacts and not classifier.preprocessing_artifacts:
        raise RuntimeError(
            "Topic model bundle does not contain preprocessing artifacts. "
            "Regenerate the model with `python post-processing/topic-classification/run.py train` "
            "so runtime classification "
            "can apply the prepared-corpus token vocabulary."
        )
    if require_raw_frequency_artifacts and not classifier.preprocessing_artifacts.get(
        "raw_corpus_frequency_artifacts_available"
    ):
        raise RuntimeError(
            "Topic model bundle does not contain raw source-specific frequency "
            "artifacts. Regenerate the model with "
            "`python post-processing/topic-classification/run.py train` using "
            "--raw-text-token-counts, --raw-file-name-token-counts, and "
            "--require-raw-frequency-artifacts."
        )
    preprocessing_artifact_summary = _preprocessing_artifact_summary(
        dict(classifier.preprocessing_artifacts or {})
    )
    log(
        "Model preprocessing artifacts: "
        f"schema={preprocessing_artifact_summary['schema_version']}, "
        f"text_tokens={preprocessing_artifact_summary['allowed_text_token_count']}, "
        f"file_name_tokens={preprocessing_artifact_summary['allowed_file_name_token_count']}, "
        "raw_frequency_artifacts="
        f"{preprocessing_artifact_summary['raw_corpus_frequency_artifacts_available']}"
    )
    data_preprocessor = get_preprocessor_for_model(
        classifier.preprocessing_artifacts,
        identifier_splitter_mode=identifier_splitter_mode,
        allow_heuristic_file_splits=allow_heuristic_file_splits,
        data_preparation_root=data_preparation_root,
    )
    readme_enricher = TopicReadmeEnricher(
        cache_dir=readme_cache_dir,
        tokens=github_tokens,
        log=log,
    )
    wiki_enricher = TopicWikiEnricher(
        cache_dir=wiki_cache_dir,
        enable_live_fetch=enable_live_wiki_fetch,
        tokens=github_tokens,
        log=log,
    )
    loader = TopicClassificationInputLoader(
        curation_outputs_dir,
        exclude_dirs=exclude_dirs,
        longitudinal_only=longitudinal_only,
        input_format=input_format,
    )
    index = loader.build_index()
    batch_count = ceil(len(index.repository_refs) / batch_size) if index.repository_refs else 0
    stats = LoaderStats(
        eligible_cohort_count=index.eligible_cohort_count,
        repository_count=len(index.repository_refs),
        metrics_path_count=index.metrics_path_count,
        pr_record_ref_count=index.pr_record_ref_count,
        pr_index_parse_failures=index.pr_index_parse_failures,
        repositories_filtered_by_longitudinal=index.repositories_filtered_by_longitudinal,
        repositories_deduplicated_by_identity=index.repositories_deduplicated_by_identity,
        repositories_missing_file_lists=index.repositories_missing_file_lists,
        parquet_path_count=index.parquet_path_count,
        parquet_row_count=index.parquet_row_count,
        parquet_parse_failures=index.parquet_parse_failures,
        input_format=index.input_format,
        cohort_repository_counts=dict(index.cohort_repository_counts),
    )

    log(f"Eligible cohorts: {stats.eligible_cohort_count}")
    log(f"Repositories: {stats.repository_count}")
    log(f"Repositories deduplicated by identity: {stats.repositories_deduplicated_by_identity}")
    log(f"Repositories missing file lists: {stats.repositories_missing_file_lists}")
    log(f"Batches: {batch_count}")
    for cohort, count in stats.cohort_repository_counts.items():
        log(f"Cohort {cohort}: {count} repositories")

    classified_repositories = 0
    classification_failures = 0
    feature_extraction_failures = 0
    wiki_enrichment_failures = 0
    worker_failures = 0
    repository_pr_map_rows = 0
    filtered_repositories = 0
    filtered_no_readme = 0
    filtered_no_description = 0
    repository_failures: list[dict[str, str]] = []
    readme_stats = Counter(_empty_readme_stats())
    wiki_stats = Counter(_empty_wiki_stats())
    for ref in index.repository_refs:
        if stats.input_format == "legacy-json" and not ref.metrics_paths:
            stats.repositories_missing_metrics += 1
        if not ref.pr_record_refs:
            stats.repositories_missing_prs += 1

    with TopicClassificationOutputWriter(output_dir, run_id=run_id) as writer:
        if classification_workers == 1:
            for batch_number, batch in enumerate(
                index.iter_repository_batches(batch_size),
                start=1,
            ):
                log(f"Classifying batch {batch_number}/{batch_count} ({len(batch)} repositories)")
                for ref in batch:
                    result = _classify_repository(
                        ref,
                        loader=loader,
                        readme_enricher=readme_enricher,
                        wiki_enricher=wiki_enricher,
                        classifier=classifier,
                        preprocessor=data_preprocessor,
                        prediction_score_threshold=prediction_score_threshold,
                        excluded_topics=effective_excluded_topics,
                        topic_domain_map=topic_domain_map,
                        include_raw_predictions=include_raw_predictions,
                    )
                    updates = _apply_classification_result(
                        result,
                        writer=writer,
                        classifier_info=classifier.model_info,
                        prediction_score_threshold=prediction_score_threshold,
                        prediction_retention_policy=PREDICTION_RETENTION_POLICY,
                        include_raw_predictions=include_raw_predictions,
                        stats=stats,
                        readme_stats=readme_stats,
                        wiki_stats=wiki_stats,
                        repository_failures=repository_failures,
                    )
                    classified_repositories += updates["classified_repositories"]
                    classification_failures += updates["classification_failures"]
                    feature_extraction_failures += updates["feature_extraction_failures"]
                    wiki_enrichment_failures += updates["wiki_enrichment_failures"]
                    worker_failures += updates["worker_failures"]
                    repository_pr_map_rows += updates["repository_pr_map_rows"]
                    filtered_repositories += updates["filtered_repositories"]
                    filtered_no_readme += updates["filtered_no_readme"]
                    filtered_no_description += updates["filtered_no_description"]
        else:
            processed_repositories = 0
            progress_interval = max(1000, batch_size, classification_workers * 10)
            log(
                "Classifying repositories with "
                f"{classification_workers} worker processes "
                f"(progress interval={progress_interval})."
            )
            for result in _iter_parallel_classification_results(
                index.repository_refs,
                workers=classification_workers,
                model_bundle_path=model_bundle_path,
                prediction_score_threshold=prediction_score_threshold,
                excluded_topics=effective_excluded_topics,
                topic_domain_map=topic_domain_map,
                include_raw_predictions=include_raw_predictions,
                enable_live_wiki_fetch=enable_live_wiki_fetch,
                wiki_cache_dir=wiki_cache_dir,
                readme_cache_dir=readme_cache_dir,
                github_tokens=github_tokens,
                identifier_splitter_mode=identifier_splitter_mode,
                allow_heuristic_file_splits=allow_heuristic_file_splits,
                data_preparation_root=data_preparation_root,
            ):
                updates = _apply_classification_result(
                    result,
                    writer=writer,
                    classifier_info=classifier.model_info,
                    prediction_score_threshold=prediction_score_threshold,
                    prediction_retention_policy=PREDICTION_RETENTION_POLICY,
                    include_raw_predictions=include_raw_predictions,
                    stats=stats,
                    readme_stats=readme_stats,
                    wiki_stats=wiki_stats,
                    repository_failures=repository_failures,
                )
                classified_repositories += updates["classified_repositories"]
                classification_failures += updates["classification_failures"]
                feature_extraction_failures += updates["feature_extraction_failures"]
                wiki_enrichment_failures += updates["wiki_enrichment_failures"]
                worker_failures += updates["worker_failures"]
                repository_pr_map_rows += updates["repository_pr_map_rows"]
                filtered_repositories += updates["filtered_repositories"]
                filtered_no_readme += updates["filtered_no_readme"]
                filtered_no_description += updates["filtered_no_description"]
                processed_repositories += 1
                if (
                    processed_repositories % progress_interval == 0
                    or processed_repositories == stats.repository_count
                ):
                    log(
                        "Parallel classification progress: "
                        f"{processed_repositories}/{stats.repository_count} processed, "
                        f"{classified_repositories} classified, "
                        f"{filtered_repositories} filtered, "
                        f"{len(repository_failures)} failures"
                    )

        output_paths = writer.output_paths
        summary = {
            "input_root": str(curation_outputs_dir),
            "input_format": stats.input_format,
            "batch_size": batch_size,
            "classification_workers": classification_workers,
            "longitudinal_only": longitudinal_only,
            "run_id": run_id,
            "model_bundle_path": str(model_bundle_path),
            "runs_root": str(runs_root),
            "run_root": str(run_root),
            "output_dir": str(output_dir),
            "output_paths": output_paths,
            "top_k_topics": top_k_topics,
            "configured_top_k_topics": configured_top_k_topics,
            "prediction_score_threshold": prediction_score_threshold,
            "prediction_retention_policy": PREDICTION_RETENTION_POLICY,
            "filtered_out_topics": filtered_out_topics_manifest,
            "topic_domains": topic_domains_manifest,
            "topic_domain_mapping_path": str(topic_domain_mapping_path),
            "topic_domain_catalog_topic_count": topic_domains_manifest["catalog_topic_count"],
            "topic_domain_filtered_topic_count": topic_domains_manifest["filtered_topic_count"],
            "topic_domain_retained_topic_count": topic_domains_manifest["retained_topic_count"],
            "configured_excluded_topics": list(excluded_topics),
            "excluded_topics": list(effective_excluded_topics),
            "include_raw_predictions": include_raw_predictions,
            "configured_include_raw_predictions": configured_include_raw_predictions,
            "enable_live_wiki_fetch": enable_live_wiki_fetch,
            "enable_live_readme_fetch": True,
            "readme_cache_dir": str(readme_cache_dir),
            "identifier_splitter": identifier_splitter_mode,
            "require_preprocessing_artifacts": require_preprocessing_artifacts,
            "require_raw_frequency_artifacts": require_raw_frequency_artifacts,
            "allow_heuristic_file_splits": allow_heuristic_file_splits,
            "data_preparation_profile": data_preparation_profile,
            "model_preprocessing_artifacts": preprocessing_artifact_summary,
            "eligible_cohort_count": stats.eligible_cohort_count,
            "repository_count": stats.repository_count,
            "batch_count": batch_count,
            "cohort_repository_counts": stats.cohort_repository_counts,
            "repository_files_loaded": stats.repository_files_loaded,
            "pr_record_refs": stats.pr_record_ref_count,
            "pr_records_loaded": stats.pr_payloads_loaded,
            "metrics_json_paths": stats.metrics_path_count,
            "metrics_json_records_loaded": stats.metrics_payloads_loaded,
            "parquet_paths": stats.parquet_path_count,
            "parquet_rows": stats.parquet_row_count,
            "classified_repositories": classified_repositories,
            "filtered_repositories": filtered_repositories,
            "filtered_no_readme": filtered_no_readme,
            "filtered_no_description": filtered_no_description,
            "repository_failure_count": len(repository_failures),
            "repository_failures": repository_failures,
            "classification_failures": classification_failures,
            "feature_extraction_failures": feature_extraction_failures,
            "wiki_enrichment_failures": wiki_enrichment_failures,
            "worker_failures": worker_failures,
            "repository_pr_map_rows": repository_pr_map_rows,
            "repositories_missing_metrics": stats.repositories_missing_metrics,
            "repositories_missing_prs": stats.repositories_missing_prs,
            "repositories_missing_file_lists": stats.repositories_missing_file_lists,
            "repositories_filtered_by_longitudinal": stats.repositories_filtered_by_longitudinal,
            "repositories_deduplicated_by_identity": stats.repositories_deduplicated_by_identity,
            "parse_failures": stats.parse_failures,
            "pr_index_parse_failures": stats.pr_index_parse_failures,
            "parquet_parse_failures": stats.parquet_parse_failures,
            "file_list_parse_failures": stats.file_list_parse_failures,
            "metrics_parse_failures": stats.metrics_parse_failures,
            "pr_payload_parse_failures": stats.pr_payload_parse_failures,
            "readme_stats": dict(readme_stats),
            "wiki_stats": dict(wiki_stats),
            "classifier": classifier.model_info,
        }
        writer.write_manifest(
            {
                "config": {
                    "input_root": str(curation_outputs_dir),
                    "input_format": stats.input_format,
                    "batch_size": batch_size,
                    "classification_workers": classification_workers,
                    "longitudinal_only": longitudinal_only,
                    "model_bundle_path": str(model_bundle_path),
                    "runs_root": str(runs_root),
                    "run_root": str(run_root),
                    "output_dir": str(output_dir),
                    "top_k_topics": top_k_topics,
                    "configured_top_k_topics": configured_top_k_topics,
                    "prediction_score_threshold": prediction_score_threshold,
                    "prediction_retention_policy": PREDICTION_RETENTION_POLICY,
                    "filtered_out_topics": filtered_out_topics_manifest,
                    "topic_domains": topic_domains_manifest,
                    "topic_domain_mapping_path": str(topic_domain_mapping_path),
                    "configured_excluded_topics": list(excluded_topics),
                    "excluded_topics": list(effective_excluded_topics),
                    "include_raw_predictions": include_raw_predictions,
                    "configured_include_raw_predictions": configured_include_raw_predictions,
                    "enable_live_wiki_fetch": enable_live_wiki_fetch,
                    "enable_live_readme_fetch": True,
                    "identifier_splitter": identifier_splitter_mode,
                    "require_preprocessing_artifacts": require_preprocessing_artifacts,
                    "require_raw_frequency_artifacts": require_raw_frequency_artifacts,
                    "allow_heuristic_file_splits": allow_heuristic_file_splits,
                    "data_preparation_root": str(data_preparation_root),
                    "data_preparation_profile": data_preparation_profile,
                    "model_preprocessing_artifacts": preprocessing_artifact_summary,
                    "readme_cache_dir": str(readme_cache_dir),
                    "wiki_cache_dir": str(wiki_cache_dir),
                    "github_token_count": len(github_tokens),
                },
                "classifier": classifier.model_info,
                "data_preparation": data_preprocessor.manifest,
                "statistics": summary,
                "output_paths": output_paths,
            }
        )

    log(f"Loaded repository files: {stats.repository_files_loaded}")
    log(f"PR records loaded: {stats.pr_payloads_loaded}/{stats.pr_record_ref_count}")
    log(f"Parquet rows indexed: {stats.parquet_row_count}/{stats.parquet_path_count} files")
    log(
        "Metrics JSON records loaded: "
        f"{stats.metrics_payloads_loaded}/{stats.metrics_path_count}"
    )
    log(
        "Missing inputs: "
        f"metrics={stats.repositories_missing_metrics}, prs={stats.repositories_missing_prs}, "
        f"file_lists={stats.repositories_missing_file_lists}"
    )
    log(
        "Parse failures: "
        f"total={stats.parse_failures}, pr_index={stats.pr_index_parse_failures}, "
        f"parquet={stats.parquet_parse_failures}, "
        f"file_lists={stats.file_list_parse_failures}, "
        f"metrics={stats.metrics_parse_failures}, pr_payloads={stats.pr_payload_parse_failures}"
    )
    log(
        "Classification: "
        f"classified={classified_repositories}, "
        f"filtered={filtered_repositories}, "
        f"no_readme={filtered_no_readme}, "
        f"no_description={filtered_no_description}, "
        f"feature_failures={feature_extraction_failures}, "
        f"classification_failures={classification_failures}, "
        f"wiki_failures={wiki_enrichment_failures}, "
        f"worker_failures={worker_failures}"
    )
    log(f"README stats: {dict(readme_stats)}")
    if repository_failures:
        log(f"Repository failures ({len(repository_failures)}):")
        for failure in repository_failures:
            log(
                "- "
                f"{failure['cohort']} {failure['repository_key']} "
                f"stage={failure['stage']}: {failure['message']}"
            )
    else:
        log("Repository failures: none")
    log(f"Output manifest: {summary['output_paths']['manifest']}")
    return summary
