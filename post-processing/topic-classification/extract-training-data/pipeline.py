"""Pipeline for extracting topic-classifier training data.

Stage 1 searches public GitHub repositories with topics, slices large date
windows to stay below GitHub search caps, enriches each candidate with README,
wiki, and file-list artifacts, and writes a restartable run directory.
"""

from __future__ import annotations

import calendar
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
import math
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from common import (
    DEFAULT_OUTPUT_DIR,
    append_jsonl,
    env_bool,
    env_int,
    env_nonnegative_int,
    env_path,
    env_text,
    iter_jsonl,
    json_default,
    load_post_processing_github_tokens,
    log,
    normalize_repository_key,
    repository_identity_key,
    safe_path_part,
    utc_now_z,
    utc_timestamp_run_id,
    write_json,
)
from github_training_client import GitHubTopicTrainingClient, RepositorySearchResult
from wiki_enrichment import TopicWikiEnricher


SEARCH_PER_PAGE = 100
GITHUB_SEARCH_RESULT_CAP = 1000
TOPICS_QUERY = "topics:>0"
STARS_QUERY: str | None = None


@dataclass(frozen=True)
class TopicTrainingExtractionConfig:
    """Runtime settings for one GitHub topic-training extraction run."""

    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    target_repos: int = 0
    enable_live_wiki_fetch: bool = True
    resume: bool = True
    output_dir: Path = DEFAULT_OUTPUT_DIR
    max_pages: int = 10
    workers: int = 8
    progress_interval: int = 25
    time_bucket_sampling: bool = True
    run_id: str | None = None

    @classmethod
    def from_env(cls) -> "TopicTrainingExtractionConfig":
        return cls(
            start_date=env_text("POST_PROCESSING_TOPIC_TRAINING_START_DATE", "2024-01-01"),
            end_date=env_text("POST_PROCESSING_TOPIC_TRAINING_END_DATE", "2025-12-31"),
            target_repos=env_nonnegative_int("POST_PROCESSING_TOPIC_TRAINING_TARGET_REPOS", 0),
            enable_live_wiki_fetch=env_bool(
                "POST_PROCESSING_TOPIC_TRAINING_ENABLE_LIVE_WIKI_FETCH",
                True,
            ),
            resume=env_bool("POST_PROCESSING_TOPIC_TRAINING_RESUME", True),
            output_dir=env_path("POST_PROCESSING_TOPIC_TRAINING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
            max_pages=env_int("POST_PROCESSING_TOPIC_TRAINING_MAX_PAGES", 10),
            workers=env_int("POST_PROCESSING_TOPIC_TRAINING_WORKERS", 8),
            progress_interval=env_int("POST_PROCESSING_TOPIC_TRAINING_PROGRESS_INTERVAL", 25),
            time_bucket_sampling=env_bool(
                "POST_PROCESSING_TOPIC_TRAINING_TIME_BUCKET_SAMPLING",
                True,
            ),
            run_id=os.environ.get("POST_PROCESSING_TOPIC_TRAINING_RUN_ID") or None,
        )


@dataclass
class RepositoryAccumulator:
    """Aggregate repeated search hits for the same repository identity."""

    owner: str
    repo: str
    repository_id: str | None
    repository_key: str
    repository_identity_key: str
    metadata_candidates: list[dict[str, Any]] = field(default_factory=list)
    source_searches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SearchDateBucket:
    """GitHub `created:` search window that can split from month to day/hour."""

    start: datetime
    end: datetime
    granularity: str

    @property
    def query_value(self) -> str:
        if self.granularity == "hour":
            return f"{_github_datetime(self.start)}..{_github_datetime(self.end)}"
        return f"{self.start.date().isoformat()}..{self.end.date().isoformat()}"

    @property
    def can_split(self) -> bool:
        return self.granularity in {"month", "day"}

    def split(self) -> tuple["SearchDateBucket", ...]:
        if self.granularity == "month":
            buckets: list[SearchDateBucket] = []
            current = self.start.date()
            end_day = self.end.date()
            while current <= end_day:
                buckets.append(
                    SearchDateBucket(
                        datetime.combine(current, time(0, 0, 0)),
                        datetime.combine(current, time(23, 59, 59)),
                        "day",
                    )
                )
                current += timedelta(days=1)
            return tuple(buckets)
        if self.granularity == "day":
            day = self.start.date()
            return tuple(
                SearchDateBucket(
                    datetime.combine(day, time(hour, 0, 0)),
                    datetime.combine(day, time(hour, 59, 59)),
                    "hour",
                )
                for hour in range(24)
            )
        return ()


class TopicTrainingExtractor:
    """Orchestrate discovery, enrichment, checkpointing, and manifest writing."""

    def __init__(
        self,
        config: TopicTrainingExtractionConfig,
        *,
        tokens: tuple[str, ...] | None = None,
        discovery: Any | None = None,
        enrichment: Any | None = None,
        github_client: GitHubTopicTrainingClient | None = None,
        wiki_enricher: TopicWikiEnricher | None = None,
        preprocessor: Any | None = None,
    ) -> None:
        self.config = config
        self.tokens = tokens if tokens is not None else load_post_processing_github_tokens()
        self.discovery = discovery
        self.enrichment = enrichment
        self.github_client = github_client
        self.wiki_enricher = wiki_enricher
        self.preprocessor = preprocessor

    def run(self) -> dict[str, Any]:
        run_started = perf_counter()
        run_started_at_utc = utc_now_z()
        if not self.tokens:
            raise RuntimeError(
                "No GitHub tokens found in post-processing/config/tokens_config.py."
            )
        _validate_date_window(self.config.start_date, self.config.end_date)
        run_id = self.config.run_id or utc_timestamp_run_id("topic_training_extraction")
        run_dir = Path(self.config.output_dir) / run_id
        paths = _run_paths(run_dir)
        if not self.config.resume and _has_existing_run_outputs(paths):
            raise RuntimeError(
                f"Run directory already has extraction outputs: {run_dir}. "
                "Use resume=True or choose a new run id/output directory."
            )
        for path in paths.values():
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)

        target_repos = max(0, int(self.config.target_repos))
        max_pages = max(1, int(self.config.max_pages))
        workers = max(1, int(self.config.workers))
        progress_interval = max(1, int(self.config.progress_interval))
        time_bucket_sampling = bool(self.config.time_bucket_sampling) and target_repos > 0
        log(f"Run directory: {run_dir}")
        log(
            "Config: mode=general-repository-search, "
            f"created={self.config.start_date}..{self.config.end_date}, "
            f"target_repos={target_repos or 'all'}, max_pages={max_pages}, "
            f"{TOPICS_QUERY}, stars=unfiltered, "
            f"live_wiki_fetch={self.config.enable_live_wiki_fetch}, "
            f"workers={workers}, time_bucket_sampling={time_bucket_sampling}, "
            f"time_bucket_sampling_granularity=day, "
            f"resume={self.config.resume}"
        )

        github_client = self.github_client or GitHubTopicTrainingClient(self.tokens)
        search_started = perf_counter()
        repositories, search_counts, searched_buckets, capped_hourly_buckets = (
            self._collect_repositories(
                github_client=github_client,
                paths=paths,
                target_repos=target_repos,
                max_pages=max_pages,
                resume=self.config.resume,
            )
        )
        search_elapsed = max(0.001, perf_counter() - search_started)
        log(f"Unique candidate repositories: {len(repositories)}")

        raw_topics_counter: Counter[str] = Counter()
        raw_topic_universe: set[str] = set()
        completed_records = (
            _load_repository_records(paths["repository_records"]) if self.config.resume else {}
        )
        initial_completed_record_count = len(completed_records)
        for record in completed_records.values():
            _update_topic_counters_from_record(
                record,
                raw_topics_counter,
                raw_topic_universe,
            )
        artifact_counts, artifact_performance = self._fetch_repository_artifacts(
            repositories=repositories,
            paths=paths,
            completed_records=completed_records,
            raw_topics_counter=raw_topics_counter,
            raw_topic_universe=raw_topic_universe,
            workers=workers,
            progress_interval=progress_interval,
        )
        repository_records = artifact_counts["repository_records"]
        skipped_existing_records = artifact_counts["skipped_existing_records"]
        artifact_failures = artifact_counts["artifact_failures"]
        total_elapsed = max(0.001, perf_counter() - run_started)
        new_repository_records = max(0, repository_records - initial_completed_record_count)
        total_new_records_per_hour = new_repository_records / total_elapsed * 3600.0
        candidate_repositories_per_hour = (
            search_counts.get("unique_candidate_repositories", 0) / search_elapsed * 3600.0
        )
        performance = {
            **artifact_performance,
            "search_elapsed_seconds": round(search_elapsed, 3),
            "search_candidate_repositories_per_hour": round(
                candidate_repositories_per_hour,
                3,
            ),
            "total_elapsed_seconds": round(total_elapsed, 3),
            "total_new_records_per_hour": round(total_new_records_per_hour, 3),
            "projected_200k_hours_at_total_rate": (
                round(200000 / total_new_records_per_hour, 3)
                if total_new_records_per_hour > 0
                else None
            ),
            "projected_200k_days_at_total_rate": (
                round(200000 / total_new_records_per_hour / 24, 3)
                if total_new_records_per_hour > 0
                else None
            ),
        }

        raw_topics = sorted(raw_topic_universe)
        write_json(paths["raw_topic_universe"], raw_topics)
        _write_candidate_topics(
            paths["candidate_topics"],
            raw_topics_counter,
        )
        manifest = {
            "schema_version": "topic_training_extraction_manifest_v4",
            "mode": "general-repository-search",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "started_at_utc": run_started_at_utc,
            "created_at_utc": utc_now_z(),
            "created_start_date": self.config.start_date,
            "created_end_date": self.config.end_date,
            "stars_query": STARS_QUERY,
            "topics_query": TOPICS_QUERY,
            "enable_live_wiki_fetch": self.config.enable_live_wiki_fetch,
            "target_repos": target_repos,
            "resume": self.config.resume,
            "config": {
                "mode": "general-repository-search",
                "created_start_date": self.config.start_date,
                "created_end_date": self.config.end_date,
                "target_repos": target_repos,
                "max_pages": max_pages,
                "workers": workers,
                "progress_interval": progress_interval,
                "time_bucket_sampling": time_bucket_sampling,
                "time_bucket_sampling_granularity": "day",
                "token_source": "post-processing/config/tokens_config.py",
                "token_count": len(self.tokens),
                "stars_query": STARS_QUERY,
                "topics_query": TOPICS_QUERY,
                "enable_live_wiki_fetch": self.config.enable_live_wiki_fetch,
                "resume": self.config.resume,
                "data_preparation": "deferred",
            },
            "counts": {
                "raw_topic_count": len(raw_topics),
                "searched_buckets": len(searched_buckets),
                "capped_hourly_buckets": len(capped_hourly_buckets),
                "repository_search_pages": search_counts.get("search_pages", 0),
                "repository_search_failures": search_counts.get("search_failures", 0),
                "repository_search_candidate_rows": search_counts.get("candidate_rows", 0),
                "unique_candidate_repositories": search_counts.get(
                    "unique_candidate_repositories",
                    0,
                ),
                "resumed_candidate_repositories": search_counts.get(
                    "resumed_candidate_repositories",
                    0,
                ),
                "resumed_search_pages": search_counts.get("resumed_search_pages", 0),
                "skipped_completed_search_pages": search_counts.get(
                    "skipped_completed_search_pages",
                    0,
                ),
                "unique_repositories": len(repositories),
                "repository_records": repository_records,
                "resumed_repository_records": initial_completed_record_count,
                "skipped_existing_repository_records": skipped_existing_records,
                "artifact_failures": artifact_failures,
                "artifact_workers": artifact_performance["workers"],
                "artifact_new_records": artifact_performance["new_records"],
            },
            "performance": performance,
            "searched_buckets": searched_buckets,
            "capped_hourly_buckets": capped_hourly_buckets,
            "outputs": {key: str(value) for key, value in paths.items()},
        }
        write_json(paths["manifest"], manifest)
        log(f"Extraction complete: {paths['manifest']}")
        return manifest

    def _fetch_repository_artifacts(
        self,
        *,
        repositories: dict[str, RepositoryAccumulator],
        paths: dict[str, Path],
        completed_records: dict[str, dict[str, Any]],
        raw_topics_counter: Counter[str],
        raw_topic_universe: set[str],
        workers: int,
        progress_interval: int,
    ) -> tuple[dict[str, int], dict[str, Any]]:
        artifact_started = perf_counter()
        repository_records = len(completed_records)
        skipped_existing_records = 0
        artifact_failures = 0
        pending: list[RepositoryAccumulator] = []
        total_repositories = len(repositories)

        for accumulator in repositories.values():
            if accumulator.repository_identity_key in completed_records:
                skipped_existing_records += 1
            else:
                pending.append(accumulator)

        if skipped_existing_records:
            log(
                f"Repository artifact resume: {skipped_existing_records} existing records "
                f"will be skipped, {len(pending)} records pending."
            )

        effective_workers = max(1, min(max(1, int(workers)), max(1, len(pending))))
        use_parallel = (
            len(pending) > 1
            and effective_workers > 1
            and self.github_client is None
            and self.wiki_enricher is None
        )
        if not use_parallel:
            effective_workers = 1

        log(
            f"Repository artifact workers: {effective_workers} "
            f"(configured={max(1, int(workers))}, pending={len(pending)})"
        )

        processed_pending = 0

        def record_success(record: dict[str, Any]) -> None:
            nonlocal repository_records
            append_jsonl(paths["repository_records"], record)
            repository_records += 1
            completed_records[str(record["repository_identity_key"])] = record
            _update_topic_counters_from_record(
                record,
                raw_topics_counter,
                raw_topic_universe,
            )

        def emit_progress(force: bool = False) -> None:
            if not force and processed_pending % progress_interval != 0:
                return
            elapsed = max(0.001, perf_counter() - artifact_started)
            records_per_hour = processed_pending / elapsed * 3600.0
            log(
                f"Repository artifact progress: {processed_pending}/{len(pending)} pending, "
                f"total={repository_records}/{total_repositories}, "
                f"skipped={skipped_existing_records}, failures={artifact_failures}, "
                f"rate={records_per_hour:.1f} repos/hour"
            )

        if use_parallel:
            worker_state = threading.local()
            worker_indexes = count()
            worker_index_lock = threading.Lock()

            def worker_clients() -> tuple[GitHubTopicTrainingClient, TopicWikiEnricher]:
                github = getattr(worker_state, "github_client", None)
                wiki = getattr(worker_state, "wiki_enricher", None)
                if github is None or wiki is None:
                    with worker_index_lock:
                        worker_index = next(worker_indexes)
                    worker_tokens = _rotate_tokens(self.tokens, worker_index)
                    github = GitHubTopicTrainingClient(worker_tokens)
                    wiki = TopicWikiEnricher(
                        cache_dir=paths["wiki_cache"],
                        enable_live_fetch=self.config.enable_live_wiki_fetch,
                        tokens=worker_tokens,
                        log=log,
                    )
                    worker_state.github_client = github
                    worker_state.wiki_enricher = wiki
                return github, wiki

            def build_in_worker(accumulator: RepositoryAccumulator) -> dict[str, Any]:
                github, wiki = worker_clients()
                return self._build_repository_record(
                    accumulator,
                    github_client=github,
                    wiki_enricher=wiki,
                    snapshots_root=paths["snapshots"],
                    readme_cache_root=paths["readme_cache"],
                )

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(build_in_worker, accumulator): accumulator
                    for accumulator in pending
                }
                for future in as_completed(futures):
                    accumulator = futures[future]
                    try:
                        record_success(future.result())
                    except Exception as exc:
                        artifact_failures += 1
                        append_jsonl(
                            paths["errors"],
                            _error_payload("repository_artifacts", accumulator, exc),
                        )
                    processed_pending += 1
                    emit_progress(force=processed_pending == len(pending))
        else:
            github_client = self.github_client or GitHubTopicTrainingClient(self.tokens)
            wiki_enricher = self.wiki_enricher or TopicWikiEnricher(
                cache_dir=paths["wiki_cache"],
                enable_live_fetch=self.config.enable_live_wiki_fetch,
                tokens=self.tokens,
                log=log,
            )
            for accumulator in pending:
                try:
                    record = self._build_repository_record(
                        accumulator,
                        github_client=github_client,
                        wiki_enricher=wiki_enricher,
                        snapshots_root=paths["snapshots"],
                        readme_cache_root=paths["readme_cache"],
                    )
                    record_success(record)
                except Exception as exc:
                    artifact_failures += 1
                    append_jsonl(
                        paths["errors"],
                        _error_payload("repository_artifacts", accumulator, exc),
                    )
                processed_pending += 1
                emit_progress(force=processed_pending == len(pending))

        elapsed = max(0.001, perf_counter() - artifact_started)
        successful_new_records = max(0, processed_pending - artifact_failures)
        new_records_per_hour = successful_new_records / elapsed * 3600.0
        processed_per_hour = processed_pending / elapsed * 3600.0
        performance = {
            "workers": effective_workers,
            "configured_workers": max(1, int(workers)),
            "pending_repositories": len(pending),
            "new_records": successful_new_records,
            "artifact_failures": artifact_failures,
            "artifact_elapsed_seconds": round(elapsed, 3),
            "artifact_new_records_per_hour": round(new_records_per_hour, 3),
            "artifact_processed_per_hour": round(processed_per_hour, 3),
            "projected_200k_hours_at_artifact_rate": (
                round(200000 / new_records_per_hour, 3)
                if new_records_per_hour > 0
                else None
            ),
            "projected_200k_days_at_artifact_rate": (
                round(200000 / new_records_per_hour / 24, 3)
                if new_records_per_hour > 0
                else None
            ),
        }
        return (
            {
                "repository_records": repository_records,
                "skipped_existing_records": skipped_existing_records,
                "artifact_failures": artifact_failures,
            },
            performance,
        )

    def _collect_repositories(
        self,
        *,
        github_client: Any,
        paths: dict[str, Path],
        target_repos: int,
        max_pages: int,
        resume: bool,
    ) -> tuple[
        dict[str, RepositoryAccumulator],
        dict[str, int],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        repositories = (
            _load_repository_accumulators_from_candidates(
                paths["repository_search_candidates"]
            )
            if resume
            else {}
        )
        seen_candidates: set[str] = set(repositories)
        completed_search_pages = (
            _load_search_page_checkpoints(paths["repository_search_pages"])
            if resume
            else {}
        )
        counts: Counter[str] = Counter()
        counts["resumed_candidate_repositories"] = len(repositories)
        counts["resumed_search_pages"] = len(completed_search_pages)
        searched_buckets: list[dict[str, Any]] = []
        capped_hourly_buckets: list[dict[str, Any]] = []
        if target_repos > 0 and len(repositories) >= target_repos:
            return repositories, dict(counts), searched_buckets, capped_hourly_buckets
        if target_repos > 0 and self.config.time_bucket_sampling:
            self._collect_repositories_time_sampled(
                github_client=github_client,
                paths=paths,
                repositories=repositories,
                seen_candidates=seen_candidates,
                completed_search_pages=completed_search_pages,
                counts=counts,
                searched_buckets=searched_buckets,
                capped_hourly_buckets=capped_hourly_buckets,
                target_repos=target_repos,
                max_pages=max_pages,
            )
        else:
            self._collect_repositories_sequential(
                github_client=github_client,
                paths=paths,
                repositories=repositories,
                seen_candidates=seen_candidates,
                completed_search_pages=completed_search_pages,
                counts=counts,
                searched_buckets=searched_buckets,
                capped_hourly_buckets=capped_hourly_buckets,
                target_repos=target_repos,
                max_pages=max_pages,
            )
        return repositories, dict(counts), searched_buckets, capped_hourly_buckets

    def _collect_repositories_sequential(
        self,
        *,
        github_client: Any,
        paths: dict[str, Path],
        repositories: dict[str, RepositoryAccumulator],
        seen_candidates: set[str],
        completed_search_pages: dict[str, dict[str, Any]],
        counts: Counter[str],
        searched_buckets: list[dict[str, Any]],
        capped_hourly_buckets: list[dict[str, Any]],
        target_repos: int,
        max_pages: int,
    ) -> None:
        for bucket, result in self._iter_repository_search_results(
            github_client,
            max_pages=max_pages,
            searched_buckets=searched_buckets,
            capped_hourly_buckets=capped_hourly_buckets,
            completed_search_pages=completed_search_pages,
        ):
            _added, stop, processed_all = self._consume_repository_search_result(
                bucket=bucket,
                result=result,
                paths=paths,
                repositories=repositories,
                seen_candidates=seen_candidates,
                completed_search_pages=completed_search_pages,
                counts=counts,
                target_repos=target_repos,
                max_new_repositories=None,
            )
            if processed_all:
                self._checkpoint_repository_search_page(
                    bucket=bucket,
                    result=result,
                    paths=paths,
                    completed_search_pages=completed_search_pages,
                )
            if stop:
                return

    def _collect_repositories_time_sampled(
        self,
        *,
        github_client: Any,
        paths: dict[str, Path],
        repositories: dict[str, RepositoryAccumulator],
        seen_candidates: set[str],
        completed_search_pages: dict[str, dict[str, Any]],
        counts: Counter[str],
        searched_buckets: list[dict[str, Any]],
        capped_hourly_buckets: list[dict[str, Any]],
        target_repos: int,
        max_pages: int,
    ) -> None:
        daily_buckets = _daily_buckets(self.config.start_date, self.config.end_date)
        remaining_target = max(0, target_repos - len(repositories))
        quotas = _allocate_bucket_quotas(remaining_target, len(daily_buckets))
        carry = 0
        counts["time_bucket_sampling_days"] = len(daily_buckets)
        active_daily_buckets = 0
        for bucket, base_quota in zip(daily_buckets, quotas):
            bucket_quota = base_quota + carry
            if bucket_quota <= 0:
                continue
            active_daily_buckets += 1
            before = len(repositories)
            self._collect_repositories_for_bucket_quota(
                github_client=github_client,
                bucket=bucket,
                quota=bucket_quota,
                paths=paths,
                repositories=repositories,
                seen_candidates=seen_candidates,
                completed_search_pages=completed_search_pages,
                counts=counts,
                searched_buckets=searched_buckets,
                capped_hourly_buckets=capped_hourly_buckets,
                target_repos=target_repos,
                max_pages=max_pages,
            )
            added = len(repositories) - before
            carry = max(0, bucket_quota - added)
            if (
                active_daily_buckets == 1
                or active_daily_buckets % 25 == 0
                or target_repos > 0
                and len(repositories) >= target_repos
            ):
                log(
                    f"Repository search daily sampling: {active_daily_buckets} "
                    f"active days, latest={bucket.query_value}, "
                    f"quota={bucket_quota}, added={added}, "
                    f"total={len(repositories)}/{target_repos}, carry={carry}"
                )
            if target_repos > 0 and len(repositories) >= target_repos:
                counts["time_bucket_sampling_active_days"] = active_daily_buckets
                return

        counts["time_bucket_sampling_active_days"] = active_daily_buckets
        if target_repos > 0 and len(repositories) < target_repos:
            counts["time_bucket_sampling_fill_passes"] += 1
            self._collect_repositories_sequential(
                github_client=github_client,
                paths=paths,
                repositories=repositories,
                seen_candidates=seen_candidates,
                completed_search_pages=completed_search_pages,
                counts=counts,
                searched_buckets=searched_buckets,
                capped_hourly_buckets=capped_hourly_buckets,
                target_repos=target_repos,
                max_pages=max_pages,
            )

    def _collect_repositories_for_bucket_quota(
        self,
        *,
        github_client: Any,
        bucket: SearchDateBucket,
        quota: int,
        paths: dict[str, Path],
        repositories: dict[str, RepositoryAccumulator],
        seen_candidates: set[str],
        completed_search_pages: dict[str, dict[str, Any]],
        counts: Counter[str],
        searched_buckets: list[dict[str, Any]],
        capped_hourly_buckets: list[dict[str, Any]],
        target_repos: int,
        max_pages: int,
    ) -> int:
        if quota <= 0 or (target_repos > 0 and len(repositories) >= target_repos):
            return 0
        before_bucket = len(repositories)
        first_key = _search_page_key(bucket, 1)
        first = (
            _search_result_from_checkpoint(completed_search_pages[first_key])
            if first_key in completed_search_pages
            else github_client.search_repositories(
                created_bucket=bucket.query_value,
                topics_query=TOPICS_QUERY,
                stars_query=STARS_QUERY,
                page=1,
                per_page=SEARCH_PER_PAGE,
            )
        )
        searched_buckets.append(
            _search_bucket_payload(
                bucket,
                first,
                resume_checkpoint=first_key in completed_search_pages,
            )
        )
        should_split_capped_bucket = (
            first.status == "fetched"
            and first.total_count >= GITHUB_SEARCH_RESULT_CAP
            and quota > GITHUB_SEARCH_RESULT_CAP
        )
        if should_split_capped_bucket:
            if bucket.can_split:
                children = bucket.split()
                child_quotas = _allocate_bucket_quotas(quota, len(children))
                carry = 0
                for child, child_base_quota in zip(children, child_quotas):
                    child_quota = child_base_quota + carry
                    if child_quota <= 0:
                        continue
                    before_child = len(repositories)
                    self._collect_repositories_for_bucket_quota(
                        github_client=github_client,
                        bucket=child,
                        quota=child_quota,
                        paths=paths,
                        repositories=repositories,
                        seen_candidates=seen_candidates,
                        completed_search_pages=completed_search_pages,
                        counts=counts,
                        searched_buckets=searched_buckets,
                        capped_hourly_buckets=capped_hourly_buckets,
                        target_repos=target_repos,
                        max_pages=max_pages,
                    )
                    child_added = len(repositories) - before_child
                    carry = max(0, child_quota - child_added)
                    if target_repos > 0 and len(repositories) >= target_repos:
                        break
                return len(repositories) - before_bucket
            capped_hourly_buckets.append(_capped_hourly_bucket_payload(bucket, first))

        added, stop, processed_all = self._consume_repository_search_result(
            bucket=bucket,
            result=first,
            paths=paths,
            repositories=repositories,
            seen_candidates=seen_candidates,
            completed_search_pages=completed_search_pages,
            counts=counts,
            target_repos=target_repos,
            max_new_repositories=quota,
        )
        if processed_all:
            self._checkpoint_repository_search_page(
                bucket=bucket,
                result=first,
                paths=paths,
                completed_search_pages=completed_search_pages,
            )
        if first.status != "fetched" or stop or added >= quota:
            return len(repositories) - before_bucket

        page_count = min(
            max(1, int(max_pages)),
            _search_page_count(first.total_count, first.per_page),
            GITHUB_SEARCH_RESULT_CAP // max(1, int(first.per_page)),
        )
        for page in range(2, page_count + 1):
            page_key = _search_page_key(bucket, page)
            result = (
                _search_result_from_checkpoint(completed_search_pages[page_key])
                if page_key in completed_search_pages
                else github_client.search_repositories(
                    created_bucket=bucket.query_value,
                    topics_query=TOPICS_QUERY,
                    stars_query=STARS_QUERY,
                    page=page,
                    per_page=first.per_page,
                )
            )
            remaining_quota = quota - (len(repositories) - before_bucket)
            if remaining_quota <= 0:
                break
            _added, stop, processed_all = self._consume_repository_search_result(
                bucket=bucket,
                result=result,
                paths=paths,
                repositories=repositories,
                seen_candidates=seen_candidates,
                completed_search_pages=completed_search_pages,
                counts=counts,
                target_repos=target_repos,
                max_new_repositories=remaining_quota,
            )
            if processed_all:
                self._checkpoint_repository_search_page(
                    bucket=bucket,
                    result=result,
                    paths=paths,
                    completed_search_pages=completed_search_pages,
                )
            if stop:
                break
        return len(repositories) - before_bucket

    def _consume_repository_search_result(
        self,
        *,
        bucket: SearchDateBucket,
        result: RepositorySearchResult,
        paths: dict[str, Path],
        repositories: dict[str, RepositoryAccumulator],
        seen_candidates: set[str],
        completed_search_pages: dict[str, dict[str, Any]],
        counts: Counter[str],
        target_repos: int,
        max_new_repositories: int | None,
    ) -> tuple[int, bool, bool]:
        page_key = _search_page_key(bucket, result.page)
        was_checkpointed = page_key in completed_search_pages
        counts["search_pages"] += 1
        if was_checkpointed:
            counts["skipped_completed_search_pages"] += 1
        if result.status != "fetched":
            counts["search_failures"] += 1
            return 0, False, True

        before = len(repositories)
        item_count = len(result.items)
        for item_index, item in enumerate(result.items, start=1):
            counts["candidate_rows"] += 1
            owner, repo, repo_id, metadata = _repository_from_repo_payload(item)
            if not owner or not repo:
                counts["invalid_candidates"] += 1
                continue
            identity_key = repository_identity_key(repo_id, owner, repo)
            if identity_key not in seen_candidates:
                seen_candidates.add(identity_key)
                counts["unique_candidate_repositories"] += 1
                append_jsonl(
                    paths["repository_search_candidates"],
                    _repository_search_candidate_payload(
                        bucket=bucket,
                        result=result,
                        owner=owner,
                        repo=repo,
                        repo_id=repo_id,
                        metadata=metadata,
                        identity_key=identity_key,
                    ),
                )
            accumulator = repositories.get(identity_key)
            if accumulator is None:
                accumulator = RepositoryAccumulator(
                    owner=owner,
                    repo=repo,
                    repository_id=repo_id,
                    repository_key=normalize_repository_key(owner, repo),
                    repository_identity_key=identity_key,
                )
                repositories[identity_key] = accumulator
            _record_repository_search_match(
                accumulator,
                bucket=bucket,
                result=result,
                metadata=metadata,
            )
            added = len(repositories) - before
            if target_repos > 0 and len(repositories) >= target_repos:
                return added, True, item_index == item_count
            if max_new_repositories is not None and added >= max_new_repositories:
                return added, True, item_index == item_count
        return len(repositories) - before, False, True

    def _checkpoint_repository_search_page(
        self,
        *,
        bucket: SearchDateBucket,
        result: RepositorySearchResult,
        paths: dict[str, Path],
        completed_search_pages: dict[str, dict[str, Any]],
    ) -> None:
        if result.status != "fetched":
            return
        page_key = _search_page_key(bucket, result.page)
        if page_key in completed_search_pages:
            return
        checkpoint = _search_page_checkpoint_payload(bucket, result)
        append_jsonl(paths["repository_search_pages"], checkpoint)
        completed_search_pages[page_key] = checkpoint

    def _iter_repository_search_results(
        self,
        github_client: Any,
        *,
        max_pages: int,
        searched_buckets: list[dict[str, Any]] | None = None,
        capped_hourly_buckets: list[dict[str, Any]] | None = None,
        completed_search_pages: dict[str, dict[str, Any]] | None = None,
    ) -> Iterable[tuple[SearchDateBucket, RepositorySearchResult]]:
        searched_buckets = searched_buckets if searched_buckets is not None else []
        capped_hourly_buckets = (
            capped_hourly_buckets if capped_hourly_buckets is not None else []
        )
        completed_search_pages = completed_search_pages or {}
        for bucket in _monthly_buckets(self.config.start_date, self.config.end_date):
            yield from self._iter_repository_search_bucket(
                github_client,
                bucket=bucket,
                max_pages=max_pages,
                searched_buckets=searched_buckets,
                capped_hourly_buckets=capped_hourly_buckets,
                completed_search_pages=completed_search_pages,
            )

    def _iter_repository_search_bucket(
        self,
        github_client: Any,
        *,
        bucket: SearchDateBucket,
        max_pages: int,
        searched_buckets: list[dict[str, Any]] | None = None,
        capped_hourly_buckets: list[dict[str, Any]] | None = None,
        completed_search_pages: dict[str, dict[str, Any]] | None = None,
    ) -> Iterable[tuple[SearchDateBucket, RepositorySearchResult]]:
        searched_buckets = searched_buckets if searched_buckets is not None else []
        capped_hourly_buckets = (
            capped_hourly_buckets if capped_hourly_buckets is not None else []
        )
        completed_search_pages = completed_search_pages or {}
        first_key = _search_page_key(bucket, 1)
        first = (
            _search_result_from_checkpoint(completed_search_pages[first_key])
            if first_key in completed_search_pages
            else github_client.search_repositories(
                created_bucket=bucket.query_value,
                topics_query=TOPICS_QUERY,
                stars_query=STARS_QUERY,
                page=1,
                per_page=SEARCH_PER_PAGE,
            )
        )
        searched_buckets.append(
            _search_bucket_payload(
                bucket,
                first,
                resume_checkpoint=first_key in completed_search_pages,
            )
        )
        if first.status == "fetched" and first.total_count >= GITHUB_SEARCH_RESULT_CAP:
            if bucket.can_split:
                for child in bucket.split():
                    yield from self._iter_repository_search_bucket(
                        github_client,
                        bucket=child,
                        max_pages=max_pages,
                        searched_buckets=searched_buckets,
                        capped_hourly_buckets=capped_hourly_buckets,
                        completed_search_pages=completed_search_pages,
                    )
                return
            capped_hourly_buckets.append(_capped_hourly_bucket_payload(bucket, first))
        yield bucket, first
        if first.status != "fetched":
            return
        page_count = min(
            max(1, int(max_pages)),
            _search_page_count(first.total_count, first.per_page),
            GITHUB_SEARCH_RESULT_CAP // max(1, int(first.per_page)),
        )
        for page in range(2, page_count + 1):
            page_key = _search_page_key(bucket, page)
            if page_key in completed_search_pages:
                yield bucket, _search_result_from_checkpoint(completed_search_pages[page_key])
                continue
            yield bucket, github_client.search_repositories(
                created_bucket=bucket.query_value,
                topics_query=TOPICS_QUERY,
                stars_query=STARS_QUERY,
                page=page,
                per_page=first.per_page,
            )

    def _build_repository_record(
        self,
        accumulator: RepositoryAccumulator,
        *,
        github_client: Any,
        wiki_enricher: Any,
        snapshots_root: Path,
        readme_cache_root: Path,
    ) -> dict[str, Any]:
        owner = accumulator.owner
        repo = accumulator.repo
        repo_payload = github_client.get_repo(owner, repo)
        metadata = _merge_metadata(accumulator.metadata_candidates, repo_payload)
        default_branch = _first_non_empty(metadata.get("default_branch"), "HEAD")
        description = str(metadata.get("description") or "")
        raw_repository_topics = _topic_names(
            metadata.get("topics") or metadata.get("repository_topics")
        )
        readme = github_client.get_readme_text(owner, repo, ref=default_branch)
        readme_cache_path, readme_cache_metadata_path = _write_readme_cache(
            cache_root=readme_cache_root,
            owner=owner,
            repo=repo,
            ref=default_branch,
            readme=readme,
        )
        file_list = github_client.list_repository_files(
            owner,
            repo,
            default_branch=default_branch,
        )
        file_list_path = _write_repository_file_list(
            snapshots_root=snapshots_root,
            owner=owner,
            repo=repo,
            ref=file_list.source_ref or default_branch,
            commit=file_list.source_commit,
            files=list(file_list.files),
        )
        wiki = wiki_enricher.get_wiki_text(owner, repo)
        return {
            "schema_version": "topic_training_repository_record_v4",
            "repository_identity_key": accumulator.repository_identity_key,
            "repository_id": _plain_repository_id(accumulator.repository_id),
            "repository_owner": owner,
            "repository_name": repo,
            "repository_full_name": f"{owner}/{repo}",
            "repository_key": accumulator.repository_key,
            "repository_url": metadata.get("html_url") or metadata.get("url"),
            "repository_stargazers_count": _optional_int(metadata.get("stargazers_count")),
            "default_branch": default_branch,
            "description": description,
            "raw_topics": raw_repository_topics,
            "data_preparation_status": "deferred",
            "readme_status": readme.status,
            "readme_cache_path": str(readme_cache_path),
            "readme_cache_metadata_path": str(readme_cache_metadata_path),
            "wiki_status": wiki.status,
            "file_list_status": file_list.status,
            "file_count": len(file_list.files),
            "file_list_path": str(file_list_path),
            "wiki_cache_path": getattr(wiki, "cache_path", None),
            "wiki_cache_metadata_path": getattr(wiki, "metadata_path", None),
            "source_searches": accumulator.source_searches,
            "extracted_at_utc": utc_now_z(),
        }


def _run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "run_dir": run_dir,
        "raw_dir": run_dir / "raw",
        "repositories_dir": run_dir / "repositories",
        "snapshots": run_dir / "snapshots",
        "readme_cache": run_dir / "readme-cache",
        "wiki_cache": run_dir / "wiki-cache",
        "labels_dir": run_dir / "labels",
        "checkpoints_dir": run_dir / "checkpoints",
        "repository_search_candidates": run_dir
        / "raw"
        / "repository_search_candidates.jsonl",
        "repository_search_pages": run_dir
        / "checkpoints"
        / "repository_search_pages.jsonl",
        "repository_records": run_dir / "repositories" / "repository_training_records.jsonl",
        "raw_topic_universe": run_dir / "labels" / "raw_topic_universe.json",
        "candidate_topics": run_dir / "labels" / "candidate_topics.csv",
        "manifest": run_dir / "topic_training_extraction_manifest.json",
        "errors": run_dir / "run_errors.jsonl",
    }


def _has_existing_run_outputs(paths: dict[str, Path]) -> bool:
    output_keys = (
        "repository_search_candidates",
        "repository_search_pages",
        "repository_records",
        "raw_topic_universe",
        "candidate_topics",
        "manifest",
        "errors",
    )
    return any(paths[key].exists() for key in output_keys)


def _repository_from_repo_payload(
    payload: dict[str, Any],
) -> tuple[str, str, str | None, dict[str, Any]]:
    full_name = str(payload.get("full_name") or payload.get("name_with_owner") or "")
    owner = ""
    repo = ""
    if "/" in full_name:
        owner, repo = full_name.split("/", 1)
    owner_data = payload.get("owner")
    if not owner and isinstance(owner_data, dict):
        owner = str(owner_data.get("login") or "")
    repo = repo or str(payload.get("name") or "")
    repo_id = payload.get("id") or payload.get("database_id")
    return owner, repo, str(repo_id) if repo_id is not None else None, dict(payload)


def _load_repository_accumulators_from_candidates(
    path: Path,
) -> dict[str, RepositoryAccumulator]:
    repositories: dict[str, RepositoryAccumulator] = {}
    for payload in iter_jsonl(path):
        owner = str(payload.get("repository_owner") or "").strip()
        repo = str(payload.get("repository_name") or "").strip()
        repo_id = payload.get("repository_id")
        identity_key = str(payload.get("repository_identity_key") or "").strip()
        if not owner or not repo:
            continue
        if not identity_key:
            identity_key = repository_identity_key(repo_id, owner, repo)
        accumulator = repositories.get(identity_key)
        if accumulator is None:
            accumulator = RepositoryAccumulator(
                owner=owner,
                repo=repo,
                repository_id=str(repo_id) if repo_id is not None else None,
                repository_key=normalize_repository_key(owner, repo),
                repository_identity_key=identity_key,
            )
            repositories[identity_key] = accumulator
        metadata = _metadata_from_candidate_payload(payload)
        if metadata:
            accumulator.metadata_candidates.append(metadata)
        source = {
            "created_bucket": payload.get("created_bucket"),
            "bucket_granularity": payload.get("bucket_granularity"),
            "repository_search_query": payload.get("repository_search_query"),
            "search_page": payload.get("search_page"),
        }
        if source not in accumulator.source_searches:
            accumulator.source_searches.append(source)
    return repositories


def _metadata_from_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    owner = str(payload.get("repository_owner") or "").strip()
    repo = str(payload.get("repository_name") or "").strip()
    repo_id = payload.get("repository_id")
    metadata: dict[str, Any] = {
        "id": repo_id,
        "name": repo,
        "full_name": payload.get("repository_full_name") or f"{owner}/{repo}",
        "owner": {"login": owner} if owner else None,
        "html_url": payload.get("repository_url"),
        "topics": payload.get("raw_repository_topics") or [],
        "stargazers_count": payload.get("repository_stargazers_count"),
        "created_at": payload.get("repository_created_at"),
        "pushed_at": payload.get("repository_pushed_at"),
        "updated_at": payload.get("repository_updated_at"),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _load_repository_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for payload in iter_jsonl(path):
        identity_key = str(payload.get("repository_identity_key") or "").strip()
        if identity_key:
            records[identity_key] = payload
    return records


def _update_topic_counters_from_record(
    record: dict[str, Any],
    raw_counter: Counter[str],
    raw_topic_universe: set[str],
) -> None:
    raw_topics = [
        str(topic)
        for topic in (record.get("raw_topics") or record.get("raw_repository_topics") or [])
    ]
    raw_counter.update(raw_topics)
    raw_topic_universe.update(topic for topic in raw_topics if topic)


def _record_repository_search_match(
    accumulator: RepositoryAccumulator,
    *,
    bucket: SearchDateBucket,
    result: RepositorySearchResult,
    metadata: dict[str, Any],
) -> None:
    source = {
        "created_bucket": bucket.query_value,
        "bucket_granularity": bucket.granularity,
        "repository_search_query": result.query,
        "search_page": result.page,
    }
    if source not in accumulator.source_searches:
        accumulator.source_searches.append(source)
    if metadata:
        accumulator.metadata_candidates.append(metadata)


def _repository_search_candidate_payload(
    *,
    bucket: SearchDateBucket,
    result: RepositorySearchResult,
    owner: str,
    repo: str,
    repo_id: str | None,
    metadata: dict[str, Any],
    identity_key: str,
) -> dict[str, Any]:
    return {
        "schema_version": "topic_training_repository_search_candidate_v1",
        "created_bucket": bucket.query_value,
        "bucket_granularity": bucket.granularity,
        "repository_search_query": result.query,
        "search_page": result.page,
        "search_total_count": result.total_count,
        "search_incomplete_results": result.incomplete_results,
        "repository_identity_key": identity_key,
        "repository_id": repo_id,
        "repository_owner": owner,
        "repository_name": repo,
        "repository_full_name": f"{owner}/{repo}",
        "repository_key": normalize_repository_key(owner, repo),
        "repository_url": metadata.get("html_url") or metadata.get("url"),
        "raw_repository_topics": _topic_names(metadata.get("topics")),
        "repository_stargazers_count": metadata.get("stargazers_count"),
        "repository_created_at": metadata.get("created_at"),
        "repository_pushed_at": metadata.get("pushed_at"),
        "repository_updated_at": metadata.get("updated_at"),
        "observed_at_utc": utc_now_z(),
    }


def _search_bucket_payload(
    bucket: SearchDateBucket,
    result: RepositorySearchResult,
    *,
    resume_checkpoint: bool = False,
) -> dict[str, Any]:
    return {
        "created_bucket": bucket.query_value,
        "bucket_granularity": bucket.granularity,
        "repository_search_query": result.query,
        "search_page": result.page,
        "status": result.status,
        "total_count": result.total_count,
        "incomplete_results": result.incomplete_results,
        "resume_checkpoint": resume_checkpoint,
        "observed_at_utc": utc_now_z(),
    }


def _capped_hourly_bucket_payload(
    bucket: SearchDateBucket,
    result: RepositorySearchResult,
) -> dict[str, Any]:
    payload = _search_bucket_payload(bucket, result)
    payload["under_collected_reason"] = "github_search_1000_result_window"
    return payload


def _search_page_key(bucket: SearchDateBucket, page: int) -> str:
    return f"{bucket.granularity}|{bucket.query_value}|{int(page)}"


def _load_search_page_checkpoints(path: Path) -> dict[str, dict[str, Any]]:
    checkpoints: dict[str, dict[str, Any]] = {}
    for payload in iter_jsonl(path):
        key = str(payload.get("search_page_key") or "").strip()
        if key:
            checkpoints[key] = payload
    return checkpoints


def _search_page_checkpoint_payload(
    bucket: SearchDateBucket,
    result: RepositorySearchResult,
) -> dict[str, Any]:
    return {
        "schema_version": "topic_training_repository_search_page_checkpoint_v1",
        "search_page_key": _search_page_key(bucket, result.page),
        "created_bucket": bucket.query_value,
        "bucket_granularity": bucket.granularity,
        "repository_search_query": result.query,
        "status": result.status,
        "search_page": result.page,
        "per_page": result.per_page,
        "total_count": result.total_count,
        "incomplete_results": result.incomplete_results,
        "completed_at_utc": utc_now_z(),
    }


def _search_result_from_checkpoint(payload: dict[str, Any]) -> RepositorySearchResult:
    return RepositorySearchResult(
        status=str(payload.get("status") or "fetched"),
        query=str(payload.get("repository_search_query") or ""),
        page=_int_value(payload.get("search_page"), 1),
        per_page=_int_value(payload.get("per_page"), SEARCH_PER_PAGE),
        total_count=_int_value(payload.get("total_count"), 0),
        incomplete_results=bool(payload.get("incomplete_results")),
        items=(),
    )


def _merge_metadata(
    candidates: Iterable[dict[str, Any]],
    repo_payload: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for candidate in candidates:
        merged.update({key: value for key, value in candidate.items() if value not in (None, "")})
    merged.update({key: value for key, value in repo_payload.items() if value not in (None, "")})
    if "repository_topics" in merged and "topics" not in merged:
        merged["topics"] = merged["repository_topics"]
    return merged


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _topic_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, dict):
        nodes = value.get("nodes")
        if isinstance(nodes, list):
            return _topic_names(nodes)
        name = value.get("name") or value.get("topic_name")
        return [str(name).strip()] if name else []
    if isinstance(value, list):
        topics: list[str] = []
        for item in value:
            topics.extend(_topic_names(item))
        return sorted({topic for topic in topics if topic})
    return []


def _plain_repository_id(value: Any) -> str | None:
    if value is None:
        return None
    repository_id = str(value).strip()
    if not repository_id:
        return None
    return repository_id.removeprefix("repository-id:")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_readme_cache(
    *,
    cache_root: Path,
    owner: str,
    repo: str,
    ref: str,
    readme: Any,
) -> tuple[Path, Path]:
    repo_root = cache_root / _safe_repo_cache_key(owner, repo)
    text_path = repo_root / "readme_text.txt"
    metadata_path = repo_root / "readme_metadata.json"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(str(getattr(readme, "text", "") or ""), encoding="utf-8", newline="\n")
    write_json(
        metadata_path,
        {
            "schema_version": "topic_training_readme_cache_v1",
            "repository_owner": owner,
            "repository_name": repo,
            "source_ref": ref,
            "status": getattr(readme, "status", None),
            "notes": getattr(readme, "notes", None),
            "text_path": str(text_path),
            "cached_at_utc": utc_now_z(),
        },
    )
    return text_path, metadata_path


def _safe_repo_cache_key(owner: str, repo: str) -> str:
    return f"{safe_path_part(owner)}__{safe_path_part(repo)}"


def _write_repository_file_list(
    *,
    snapshots_root: Path,
    owner: str,
    repo: str,
    ref: str,
    commit: str | None,
    files: list[str],
) -> Path:
    repo_root = snapshots_root / safe_path_part(owner) / safe_path_part(repo)
    path = repo_root / "repository_file_list.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "repository_owner": owner,
            "repository_name": repo,
            "source_ref": ref,
            "source_commit": commit,
            "generated_at_utc": utc_now_z(),
            "file_count": len(files),
            "files": files,
        },
    )
    return path


def _write_candidate_topics(
    path: Path,
    raw_counter: Counter[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["topic", "raw_count"],
        )
        writer.writeheader()
        for topic in sorted(raw_counter):
            writer.writerow(
                {
                    "topic": topic,
                    "raw_count": raw_counter.get(topic, 0),
                }
            )


def _monthly_buckets(start_date: str, end_date: str) -> list[SearchDateBucket]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise ValueError(f"Start date is after end date: {start_date} > {end_date}")
    buckets: list[SearchDateBucket] = []
    current = start
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        bucket_end = min(month_end, end)
        buckets.append(
            SearchDateBucket(
                datetime.combine(current, time(0, 0, 0)),
                datetime.combine(bucket_end, time(23, 59, 59)),
                "month",
            )
        )
        current = bucket_end + timedelta(days=1)
    return buckets


def _daily_buckets(start_date: str, end_date: str) -> list[SearchDateBucket]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise ValueError(f"Start date is after end date: {start_date} > {end_date}")
    buckets: list[SearchDateBucket] = []
    current = start
    while current <= end:
        buckets.append(
            SearchDateBucket(
                datetime.combine(current, time(0, 0, 0)),
                datetime.combine(current, time(23, 59, 59)),
                "day",
            )
        )
        current += timedelta(days=1)
    return buckets


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _github_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _validate_date_window(start_date: str, end_date: str) -> None:
    _monthly_buckets(start_date, end_date)


def _search_page_count(total_count: int, per_page: int) -> int:
    if total_count <= 0:
        return 1
    return max(1, int(math.ceil(total_count / max(1, per_page))))


def _allocate_bucket_quotas(total: int, bucket_count: int) -> list[int]:
    total = max(0, int(total))
    bucket_count = max(0, int(bucket_count))
    if total <= 0 or bucket_count <= 0:
        return [0 for _ in range(bucket_count)]
    base = total // bucket_count
    remainder = total % bucket_count
    quotas = [base for _ in range(bucket_count)]
    for index in _spread_indices(remainder, bucket_count):
        quotas[index] += 1
    return quotas


def _spread_indices(count_value: int, bucket_count: int) -> list[int]:
    if count_value <= 0 or bucket_count <= 0:
        return []
    if count_value >= bucket_count:
        return list(range(bucket_count))
    indexes: list[int] = []
    used: set[int] = set()
    for value in range(count_value):
        index = int((value + 0.5) * bucket_count / count_value)
        index = min(bucket_count - 1, max(0, index))
        while index in used and index + 1 < bucket_count:
            index += 1
        while index in used and index > 0:
            index -= 1
        used.add(index)
        indexes.append(index)
    return sorted(indexes)


def _rotate_tokens(tokens: Iterable[str], offset: int) -> tuple[str, ...]:
    token_list = tuple(str(token).strip() for token in tokens if str(token).strip())
    if not token_list:
        return ()
    offset = int(offset) % len(token_list)
    return token_list[offset:] + token_list[:offset]


def _error_payload(stage: str, item: Any, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": "topic_training_error_v1",
        "stage": stage,
        "item": json_default(item),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "created_at_utc": utc_now_z(),
    }
