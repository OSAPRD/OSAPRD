"""Four-stage command line interface for repository topic classification.

The implementation keeps the existing extraction, preparation, sampling,
training, and classification modules intact, but exposes only the stable stage
commands needed to run the pipeline end to end:

1. extract: collect GitHub repositories with topics.
2. prepare: filter, preprocess, and sample the training set.
3. train: train and export the topic model bundle.
4. classify: apply the trained model to curated repository outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


TOPIC_CLASSIFICATION_DIR = Path(__file__).resolve().parent
POST_PROCESSING_DIR = TOPIC_CLASSIFICATION_DIR.parent
REPO_ROOT = POST_PROCESSING_DIR.parent


def _add_import_path(path: Path) -> None:
    """Prepend an import path once, preserving stage-local imports."""
    path_text = str(Path(path))
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _load_stage_module(stage: str, filename: str, module_name: str) -> ModuleType:
    """Load a stage module from a directory whose name is not import-safe."""
    stage_dir = TOPIC_CLASSIFICATION_DIR / stage
    for path in (
        REPO_ROOT,
        POST_PROCESSING_DIR / "utility",
        TOPIC_CLASSIFICATION_DIR / "classify-topics",
        stage_dir,
    ):
        _add_import_path(path)
    module_path = stage_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _env_text(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_path(name: str, default: Path | None = None) -> Path | None:
    value = _env_text(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = _env_text(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _env_nonnegative_int(name: str, default: int) -> int:
    return max(0, _env_int(name, default))


def _env_float(name: str, default: float) -> float:
    value = _env_text(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_text(name)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.replace("\n", ",").split(",") if part.strip())


def _set_env_if_present(name: str, value: Any) -> None:
    """Set an environment override for config modules loaded later."""
    if value is not None:
        os.environ[name] = str(value)


def _add_extract_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Extract topic-training repositories from GitHub.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--target-repos", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--time-bucket-sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Spread a precise target across the date window.",
    )
    parser.add_argument(
        "--live-wiki-fetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fetch and cache repository wiki text during extraction.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse existing checkpoints and records when present.",
    )
    parser.set_defaults(func=run_extract)


def _add_prepare_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "prepare",
        aliases=("filter-sample",),
        help="Filter, preprocess, and sample extracted topic-training data.",
    )
    parser.add_argument("--input-run", type=Path, default=None)
    parser.add_argument("--output-subdir", type=Path, default=None)
    parser.add_argument("--sample-output-subdir", type=Path, default=None)
    parser.add_argument("--target-repos", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--max-non-english-ratio", type=float, default=None)
    parser.add_argument("--min-topic-repositories", type=int, default=None)
    parser.add_argument("--min-text-token-frequency", type=int, default=None)
    parser.add_argument("--min-file-name-token-frequency", type=int, default=None)
    parser.add_argument("--identifier-splitter", default=None)
    parser.add_argument("--data-preparation-root", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--excluded-repositories", default=None)
    parser.add_argument("--excluded-repositories-file", type=Path, default=None)
    parser.add_argument("--source-timeout-seconds", type=float, default=None)
    parser.add_argument("--filtered-topics-json", type=Path, default=None)
    parser.add_argument("--topic-domains-json", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse completed preparation outputs and checkpoints when possible.",
    )
    parser.set_defaults(func=run_prepare)


def _add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Train and export the topic model.")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--raw-text-token-counts", type=Path, default=None)
    parser.add_argument("--raw-file-name-token-counts", type=Path, default=None)
    parser.add_argument(
        "--require-raw-frequency-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--filtered-topics-json", type=Path, default=None)
    parser.add_argument("--topic-domains-json", type=Path, default=None)
    parser.add_argument("--expected-label-count", type=int, default=None)
    parser.add_argument(
        "--require-exact-domain-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.set_defaults(func=run_train)


def _add_classify_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "classify",
        help="Classify curated repositories with a trained topic model.",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--input-format", choices=("parquet", "json"), default=None)
    parser.add_argument("--model-bundle", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--top-k-topics", type=int, default=None)
    parser.add_argument("--prediction-score-threshold", type=float, default=None)
    parser.add_argument("--topic-domains-json", type=Path, default=None)
    parser.add_argument("--data-preparation-root", type=Path, default=None)
    parser.add_argument("--wiki-cache-dir", type=Path, default=None)
    parser.add_argument("--readme-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--longitudinal-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--live-wiki-fetch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.set_defaults(func=run_classify)


def build_parser() -> argparse.ArgumentParser:
    """Build the single public CLI for all topic-classification stages."""
    parser = argparse.ArgumentParser(
        description="Run the repository topic-classification pipeline.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    _add_extract_parser(subparsers)
    _add_prepare_parser(subparsers)
    _add_train_parser(subparsers)
    _add_classify_parser(subparsers)
    return parser


def run_extract(args: argparse.Namespace) -> int:
    """Run stage 1: collect topic-labeled repositories from GitHub."""
    module = _load_stage_module(
        "extract-training-data",
        "pipeline.py",
        "topic_training_extract_pipeline",
    )
    defaults = module.TopicTrainingExtractionConfig.from_env()
    target_repos = (
        args.target_repos
        if args.target_repos is not None
        else defaults.target_repos
    )
    config = module.TopicTrainingExtractionConfig(
        start_date=args.start_date or defaults.start_date,
        end_date=args.end_date or defaults.end_date,
        target_repos=max(0, int(target_repos)),
        enable_live_wiki_fetch=(
            defaults.enable_live_wiki_fetch
            if args.live_wiki_fetch is None
            else bool(args.live_wiki_fetch)
        ),
        resume=defaults.resume if args.resume is None else bool(args.resume),
        output_dir=args.output_dir or defaults.output_dir,
        max_pages=args.max_pages or defaults.max_pages,
        workers=args.workers if args.workers is not None else defaults.workers,
        progress_interval=(
            args.progress_interval
            if args.progress_interval is not None
            else defaults.progress_interval
        ),
        time_bucket_sampling=(
            defaults.time_bucket_sampling
            if args.time_bucket_sampling is None
            else bool(args.time_bucket_sampling)
        ),
        run_id=args.run_id or defaults.run_id,
    )
    module.TopicTrainingExtractor(config).run()
    return 0


def run_prepare(args: argparse.Namespace) -> int:
    """Run stage 2: filter, preprocess, sample, and split training records."""
    create_module = _load_stage_module(
        "create-training-data",
        "pipeline.py",
        "topic_create_training_data_pipeline",
    )
    sample_module = _load_stage_module(
        "sample-training-data",
        "pipeline.py",
        "topic_sample_training_data_pipeline",
    )
    input_run = args.input_run or _env_path("POST_PROCESSING_TOPIC_PREPARE_INPUT_RUN")
    input_run = input_run or _env_path("POST_PROCESSING_TOPIC_CREATE_TRAINING_INPUT_RUN")
    if input_run is None:
        raise ValueError(
            "Pass --input-run or set POST_PROCESSING_TOPIC_PREPARE_INPUT_RUN."
        )

    create_config = create_module.CreateTrainingDataConfig(
        input_run=Path(input_run),
        output_subdir=args.output_subdir
        or _env_path(
            "POST_PROCESSING_TOPIC_PREPARE_OUTPUT_SUBDIR",
            create_module.DEFAULT_OUTPUT_SUBDIR,
        ),
        max_non_english_ratio=(
            args.max_non_english_ratio
            if args.max_non_english_ratio is not None
            else _env_float(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MAX_NON_ENGLISH_RATIO",
                create_module.DEFAULT_MAX_NON_ENGLISH_RATIO,
            )
        ),
        progress_interval=(
            args.progress_interval
            if args.progress_interval is not None
            else _env_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_PROGRESS_INTERVAL",
                create_module.DEFAULT_PROGRESS_INTERVAL,
            )
        ),
        min_topic_repositories=(
            args.min_topic_repositories
            if args.min_topic_repositories is not None
            else _env_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MIN_TOPIC_REPOSITORIES",
                create_module.DEFAULT_MIN_TOPIC_REPOSITORIES,
            )
        ),
        min_text_token_frequency=(
            args.min_text_token_frequency
            if args.min_text_token_frequency is not None
            else _env_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MIN_TEXT_TOKEN_FREQUENCY",
                create_module.DEFAULT_MIN_TEXT_TOKEN_FREQUENCY,
            )
        ),
        min_file_name_token_frequency=(
            args.min_file_name_token_frequency
            if args.min_file_name_token_frequency is not None
            else _env_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MIN_FILE_NAME_TOKEN_FREQUENCY",
                create_module.DEFAULT_MIN_FILE_NAME_TOKEN_FREQUENCY,
            )
        ),
        identifier_splitter=args.identifier_splitter
        or _env_text(
            "POST_PROCESSING_TOPIC_CREATE_TRAINING_IDENTIFIER_SPLITTER",
            create_module.DEFAULT_IDENTIFIER_SPLITTER,
        ),
        data_preparation_root=args.data_preparation_root
        or _env_path(
            "POST_PROCESSING_TOPIC_CREATE_TRAINING_DATA_PREPARATION_ROOT",
            create_module.DEFAULT_DATA_PREPARATION_ROOT,
        ),
        max_records=(
            args.max_records
            if args.max_records is not None
            else _env_nonnegative_int(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_MAX_RECORDS",
                create_module.DEFAULT_MAX_RECORDS,
            )
        ),
        excluded_repositories=(
            _split_csv(args.excluded_repositories)
            if args.excluded_repositories is not None
            else _split_csv(
                os.environ.get(
                    "POST_PROCESSING_TOPIC_CREATE_TRAINING_EXCLUDED_REPOSITORIES",
                    "",
                )
            )
        ),
        excluded_repositories_file=args.excluded_repositories_file
        or _env_path("POST_PROCESSING_TOPIC_CREATE_TRAINING_EXCLUDED_REPOSITORIES_FILE"),
        resume=(
            bool(args.resume)
            if args.resume is not None
            else _env_bool(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_RESUME",
                create_module.DEFAULT_RESUME,
            )
        ),
        source_timeout_seconds=(
            args.source_timeout_seconds
            if args.source_timeout_seconds is not None
            else _env_float(
                "POST_PROCESSING_TOPIC_CREATE_TRAINING_SOURCE_TIMEOUT_SECONDS",
                create_module.DEFAULT_SOURCE_TIMEOUT_SECONDS,
            )
        ),
    )
    create_manifest = create_module.create_training_data(create_config)
    create_output_dir = Path(create_manifest["output_dir"])

    sample_config = sample_module.SampleTrainingDataConfig(
        create_training_data_dir=create_output_dir,
        topic_repo_extraction_dir=Path(input_run),
        target_repos=(
            args.target_repos
            if args.target_repos is not None
            else _env_int(
                "POST_PROCESSING_TOPIC_SAMPLE_TARGET_REPOS",
                sample_module.DEFAULT_TARGET_REPOS,
            )
        ),
        train_fraction=(
            args.train_fraction
            if args.train_fraction is not None
            else _env_float(
                "POST_PROCESSING_TOPIC_SAMPLE_TRAIN_FRACTION",
                sample_module.DEFAULT_TRAIN_FRACTION,
            )
        ),
        seed=(
            args.seed
            if args.seed is not None
            else _env_int("POST_PROCESSING_TOPIC_SAMPLE_SEED", sample_module.DEFAULT_SEED)
        ),
        sample_name=args.sample_name or _env_text("POST_PROCESSING_TOPIC_SAMPLE_NAME"),
        output_subdir=args.sample_output_subdir
        or _env_path(
            "POST_PROCESSING_TOPIC_SAMPLE_OUTPUT_SUBDIR",
            sample_module.DEFAULT_OUTPUT_SUBDIR,
        ),
        progress_interval=(
            args.progress_interval
            if args.progress_interval is not None
            else _env_int(
                "POST_PROCESSING_TOPIC_SAMPLE_PROGRESS_INTERVAL",
                sample_module.DEFAULT_PROGRESS_INTERVAL,
            )
        ),
        filtered_topics_path=args.filtered_topics_json
        or _env_path(
            "POST_PROCESSING_TOPIC_SAMPLE_FILTERED_TOPICS_JSON",
            sample_module.DEFAULT_FILTERED_TOPICS_PATH,
        ),
        topic_domains_path=args.topic_domains_json
        or _env_path("POST_PROCESSING_TOPIC_SAMPLE_TOPIC_DOMAINS_JSON"),
    )
    sample_module.sample_training_data(sample_config)
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Run stage 3: train and export the topic model bundle."""
    module = _load_stage_module("train-model", "train_model.py", "topic_train_model")
    module.train_topic_model(
        train_csv=args.train_csv or module.DEFAULT_TRAIN_CSV,
        test_csv=args.test_csv or module.DEFAULT_TEST_CSV,
        output_dir=args.output_dir or module.DEFAULT_OUTPUT_DIR,
        model_name=args.model_name or module.DEFAULT_MODEL_NAME,
        top_k=args.top_k if args.top_k is not None else module.DEFAULT_TOP_K,
        max_features=(
            args.max_features
            if args.max_features is not None
            else module.DEFAULT_MAX_FEATURES
        ),
        raw_text_token_counts=args.raw_text_token_counts,
        raw_file_name_token_counts=args.raw_file_name_token_counts,
        require_raw_frequency_artifacts=(
            bool(args.require_raw_frequency_artifacts)
            if args.require_raw_frequency_artifacts is not None
            else False
        ),
        sample_manifest=args.sample_manifest,
        filtered_topics_json=args.filtered_topics_json,
        topic_domains_json=args.topic_domains_json,
        expected_label_count=args.expected_label_count,
        require_exact_domain_labels=(
            bool(args.require_exact_domain_labels)
            if args.require_exact_domain_labels is not None
            else False
        ),
    )
    return 0


def run_classify(args: argparse.Namespace) -> int:
    """Run stage 4: classify curated repositories with an exported model."""
    _set_env_if_present("POST_PROCESSING_CURATION_OUTPUTS_DIR", args.input_dir)
    _set_env_if_present("POST_PROCESSING_TOPIC_CLASSIFICATION_INPUT_FORMAT", args.input_format)
    _set_env_if_present(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_MODEL_BUNDLE_PATH",
        args.model_bundle,
    )
    _set_env_if_present("POST_PROCESSING_TOPIC_CLASSIFICATION_RUNS_DIR", args.runs_dir)
    _set_env_if_present("POST_PROCESSING_TOPIC_CLASSIFICATION_RUN_ID", args.run_id)
    _set_env_if_present("POST_PROCESSING_TOPIC_CLASSIFICATION_BATCH_SIZE", args.batch_size)
    _set_env_if_present("POST_PROCESSING_TOPIC_CLASSIFICATION_WORKERS", args.workers)
    _set_env_if_present(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_TOP_K_TOPICS",
        args.top_k_topics,
    )
    _set_env_if_present(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_PREDICTION_SCORE_THRESHOLD",
        args.prediction_score_threshold,
    )
    _set_env_if_present(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_TOPIC_DOMAIN_MAPPING_PATH",
        args.topic_domains_json,
    )
    _set_env_if_present(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_DATA_PREPARATION_ROOT",
        args.data_preparation_root,
    )
    _set_env_if_present("POST_PROCESSING_TOPIC_CLASSIFICATION_WIKI_CACHE_DIR", args.wiki_cache_dir)
    _set_env_if_present(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_README_CACHE_DIR",
        args.readme_cache_dir,
    )
    if args.longitudinal_only is not None:
        os.environ["POST_PROCESSING_TOPIC_CLASSIFICATION_LONGITUDINAL_ONLY"] = (
            "true" if args.longitudinal_only else "false"
        )
    if args.live_wiki_fetch is not None:
        os.environ["POST_PROCESSING_TOPIC_CLASSIFICATION_ENABLE_LIVE_WIKI_FETCH"] = (
            "true" if args.live_wiki_fetch else "false"
        )

    module = _load_stage_module(
        "classify-topics",
        "topic_classification_pipeline.py",
        "topic_classification_pipeline_entry",
    )
    module.run_topic_classification_pipeline()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and dispatch to the selected stage."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"[post-processing/topic-classification] ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
