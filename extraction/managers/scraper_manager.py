"""Top-level orchestration for live GitHub PR extraction."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import List

from extraction.config.human_config import ALLOWED_PR_LANGUAGES, TARGET_HUMAN_PRS_PER_LANGUAGE
from extraction.config.settings import ExtractionSettings, SCHEMA_VERSION
from extraction.dtos.dtos import PullRequest
from extraction.filters.agent_filter import AgentFilter
from extraction.filters.human_filter import HumanFilter
from extraction.samplers.human_sampler import HumanSampler
from extraction.scrapers.discovery_scraper import DiscoveryScraper
from extraction.scrapers.enrichment_scraper import EnrichmentScraper, SkipPRError
from extraction.utility.checkpoint_handler import CheckpointHandler
from extraction.utility.storage_handler import StorageHandler


class ScraperManager:
    """
    Coordinate one extraction run from discovery through local persistence.

    Discovery is always live GitHub scraping. Outputs are local parquet batches grouped by
    agent name for agentic targets and under ``humans`` for human sampling.
    """

    def __init__(
        self,
        settings: ExtractionSettings,
    ) -> None:
        """Initialize scrapers, filters, storage, and checkpoints for one run."""
        if not settings.github_tokens:
            raise RuntimeError("No GitHub tokens configured. Set GITHUB_TOKENS or GITHUB_TOKEN.")

        # Copy resolved settings onto attributes for readability in the long
        # orchestration flow below. The original settings object is still passed
        # to downstream components as the source of truth.
        self.settings = settings
        self.target = settings.target
        self.start_iso = settings.start_date
        self.end_iso = settings.end_date
        self.max_pages = settings.max_pages
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_label = f"{self.start_iso}_to_{self.end_iso}_{self.target}_{timestamp}"

        # Checkpoints live under the output root so a run can be resumed by
        # reusing the same local directory.
        token_list = list(settings.github_tokens)
        checkpoints_dir = settings.local_output_dir / "checkpoints"
        self.discovery = DiscoveryScraper(
            tokens=token_list,
            checkpoint_path=checkpoints_dir / "discovery_checkpoint.json",
            settings=settings,
        )
        self.enrichment = EnrichmentScraper(
            tokens=token_list,
            settings=settings,
        )
        self.human_filter = HumanFilter()
        self.agent_filter = AgentFilter()

        # Human sampling is balanced by language after enrichment. The sampler
        # target is total PRs, so derive it from per-language target * languages.
        self.human_languages = [lang.lower() for lang in ALLOWED_PR_LANGUAGES if lang]
        per_language_target = max(0, int(TARGET_HUMAN_PRS_PER_LANGUAGE))
        total_target = per_language_target * max(1, len(self.human_languages))
        self.human_target_per_language = per_language_target
        self.human_sampler = HumanSampler(
            discovery=self.discovery,
            target_total=total_target,
        )
        self.storage = StorageHandler(
            local_dir=settings.local_output_dir,
            run_label=self.run_label,
            batch_size=settings.batch_size,
            settings=settings,
        )
        self.enrichment_checkpoint = CheckpointHandler(
            checkpoints_dir / "enrichment_checkpoint.json"
        )
        self.persisted_counts_by_group: Counter[str] = Counter()

    def _resolve_group(self, pr: PullRequest, human: bool) -> str:
        """Return the local output group for an enriched PR."""
        if human:
            return "humans"
        if pr.discovered_agent:
            if not pr.author_agent:
                pr.author_agent = pr.discovered_agent
            return pr.discovered_agent
        agent = self.agent_filter.identify_agent(pr)
        if agent:
            if not pr.author_agent:
                pr.author_agent = agent
            return agent
        return "unknown_agent"

    def _base_name(self) -> str:
        """Return the parquet filename prefix for this target."""
        return f"{self.target}_pr"

    def _persist_enriched(self, pr: PullRequest, group: str) -> None:
        """Persist one enriched PR and track per-group counts for the run manifest."""
        if self.storage.persist_one(pr, base_name=self._base_name(), group=group):
            self.persisted_counts_by_group[group] += 1

    def _write_run_manifest(self, enriched_count: int) -> None:
        """Write a local extraction run manifest with operational metadata."""
        manifest_path = self.settings.local_output_dir / "extraction_run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "target": self.target,
            "resolved_agent_list": list(self.settings.resolved_agents),
            "start_date": self.start_iso,
            "end_date": self.end_iso,
            "max_pages": self.max_pages,
            "batch_size": self.settings.batch_size,
            "use_graphql_enrichment": self.settings.use_graphql_enrichment,
            "run_label": self.run_label,
            "output_groups": sorted(self.persisted_counts_by_group),
            "counts": {
                "enriched_prs": enriched_count,
                "persisted_by_group": dict(sorted(self.persisted_counts_by_group.items())),
                "persisted_total": sum(self.persisted_counts_by_group.values()),
            },
            "token_count": len(self.settings.github_tokens),
            "token_count_redacted": len(self.settings.github_tokens),
            "written_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[manager] Wrote extraction run manifest: {manifest_path}")

    def run(self) -> List[PullRequest]:
        """Run discovery, enrichment, filtering, local persistence, and manifesting."""
        human = self.settings.is_human
        print(
            f"[manager] Target={self.target}, start={self.start_iso}, "
            f"end={self.end_iso}, max_pages={self.max_pages}"
        )

        print("[manager] Starting discovery...")
        if human:
            # Human discovery is time-stratified and pre-filters obvious agentic
            # candidates before enrichment. Post-enrichment filters below can
            # still reject candidates when richer metadata exposes agent signals
            # or disallowed languages.
            sample_result = self.human_sampler.sample(
                start_date=self.start_iso,
                end_date=self.end_iso,
                max_pages=self.max_pages,
                seen_urls=set(self.storage.seen_ids()),
            )
            discovered = list(sample_result.get("primary", []))
            quota_by_hour = sample_result.get("quota_by_hour", {})
            resume_by_hour = sample_result.get("resume_by_hour", {})
            quota_by_hour_lang: dict[str, dict[str, int]] = {}
            if quota_by_hour and self.human_languages:
                lang_count = len(self.human_languages)
                for hour_key, quota in quota_by_hour.items():
                    # Split each hourly quota across configured languages. Any
                    # remainder is assigned deterministically by config order.
                    base = quota // lang_count
                    remainder = quota % lang_count
                    quota_by_hour_lang[hour_key] = {}
                    for idx, lang in enumerate(self.human_languages):
                        quota_by_hour_lang[hour_key][lang] = base + (1 if idx < remainder else 0)
        else:
            # Agentic discovery handles both all-agent and single-agent targets;
            # the target has already been validated by ExtractionSettings.
            discovered = self.discovery.discover_prs(
                start_iso=self.start_iso,
                end_iso=self.end_iso,
                target=self.target,
                max_pages=self.max_pages,
                seen_urls=set(self.storage.seen_ids()),
            )
            quota_by_hour = {}
            quota_by_hour_lang = {}
            resume_by_hour = {}
        print(f"[manager] Discovery complete. Found {len(discovered)} candidates.")

        enriched: List[PullRequest] = []
        total = len(discovered)
        start_index = 0
        ckpt = self.enrichment_checkpoint.load()
        # Enrichment checkpoints are scoped by target/date so changing the run
        # target or window starts a fresh pass even if the output directory is
        # reused.
        if (
            ckpt.get("target") == self.target
            and ckpt.get("start") == self.start_iso
            and ckpt.get("end") == self.end_iso
        ):
            start_index = int(ckpt.get("next_index", 0))
            if start_index:
                print(f"[manager] Resuming enrichment from index {start_index}/{total}.")
        kept_by_hour_lang: dict[str, dict[str, int]] = {}
        kept_by_lang: dict[str, int] = {lang: 0 for lang in self.human_languages}
        try:
            for idx, pr in enumerate(discovered, start=1):
                try:
                    if idx <= start_index:
                        continue
                    if self.storage.is_recorded_any(pr.url, getattr(pr, "database_id", None)):
                        print(f"[manager] Skipping already stored PR {pr.url}")
                        continue
                    print(f"[manager] Enriching PR {idx}/{total} {pr.url} ...")
                    # PRs found by explicit agent clauses do not need the extra
                    # first-commit author classification call during enrichment.
                    skip_first_commit = (not human) and bool(pr.discovered_agent)
                    enriched_pr = self.enrichment.enrich_pr(
                        pr,
                        skip_first_commit=skip_first_commit,
                    )
                    print(f"[manager] Enriched PR {idx}/{total} {enriched_pr.url}")
                    # Human target filters are intentionally applied after
                    # enrichment because file languages and first-commit signals
                    # are not available in discovery search results.
                    if human and self.human_filter.is_agentic(enriched_pr):
                        print(f"[manager] Skipping agentic PR in human target: {enriched_pr.url}")
                        continue
                    if human and not self.human_filter.is_language_allowed(enriched_pr):
                        print(
                            f"[manager] Skipping PR due to file language filter: {enriched_pr.url}"
                        )
                        continue
                    if human:
                        primary_lang = self.human_filter.select_primary_language(enriched_pr)
                        if not primary_lang:
                            print(
                                f"[manager] Skipping PR with no primary language: {enriched_pr.url}"
                            )
                            continue
                        if kept_by_lang.get(primary_lang, 0) >= self.human_target_per_language:
                            print(f"[manager] Skipping PR due to language quota: {enriched_pr.url}")
                            continue
                        if pr.sampled_hour and quota_by_hour:
                            hour_lang_quota = quota_by_hour_lang.get(pr.sampled_hour, {})
                            if hour_lang_quota.get(primary_lang, 0) <= kept_by_hour_lang.get(
                                pr.sampled_hour, {}
                            ).get(primary_lang, 0):
                                print(
                                    f"[manager] Skipping PR due to hourly language quota: {enriched_pr.url}"
                                )
                                continue
                    group = self._resolve_group(enriched_pr, human=human)
                    enriched.append(enriched_pr)
                    self._persist_enriched(enriched_pr, group)
                    if human:
                        kept_by_lang[primary_lang] = kept_by_lang.get(primary_lang, 0) + 1
                        if pr.sampled_hour:
                            kept_by_hour_lang.setdefault(pr.sampled_hour, {})
                            kept_by_hour_lang[pr.sampled_hour][primary_lang] = (
                                kept_by_hour_lang[pr.sampled_hour].get(primary_lang, 0) + 1
                            )
                except SkipPRError as exc:
                    print(f"[manager] Skipping PR {pr.url}: {exc}")
                except Exception as exc:  # pragma: no cover - keep discovery going
                    print(f"Failed to enrich PR {pr.url}: {exc}")
                finally:
                    # Save progress after each candidate, including skipped
                    # candidates, so retries do not repeatedly process the same
                    # problematic PR.
                    self.enrichment_checkpoint.save(
                        {
                            "target": self.target,
                            "start": self.start_iso,
                            "end": self.end_iso,
                            "next_index": idx,
                        }
                    )
        except Exception as exc:
            print(f"[manager] Fatal error during enrichment: {exc}")
            # Flush any buffered successes before surfacing the fatal error.
            self.storage.flush_local()
            self._write_run_manifest(len(enriched))
            raise

        if human and quota_by_hour:
            # Top-up only applies to human runs. It uses the sampler's resume
            # state to continue within hours where post-enrichment filters
            # caused language/hour quotas to fall short.
            total_kept = len(enriched)
            total_target = self.human_target_per_language * max(1, len(self.human_languages))
            missing_total = max(0, total_target - total_kept)
            if missing_total:
                print(f"[manager] Human top-up needed: missing={missing_total}")
            topped_up = 0
            for hour_key, _quota in quota_by_hour.items():
                resume_state = resume_by_hour.get(hour_key)
                if not resume_state:
                    continue
                hour_lang_quota = quota_by_hour_lang.get(hour_key, {})
                missing_by_lang = {
                    lang: max(
                        0,
                        hour_lang_quota.get(lang, 0)
                        - kept_by_hour_lang.get(hour_key, {}).get(lang, 0),
                    )
                    for lang in self.human_languages
                }
                missing = sum(missing_by_lang.values())
                if missing <= 0:
                    continue
                hour_start = datetime.strptime(hour_key, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
                hour_end = hour_start + timedelta(hours=1) - timedelta(seconds=1)
                start_iso = hour_start.isoformat().replace("+00:00", "Z")
                end_iso = hour_end.isoformat().replace("+00:00", "Z")

                while missing > 0 and missing_total > 0:
                    additional, resume_state = self.discovery.discover_prs_between_limited(
                        start_iso=start_iso,
                        end_iso=end_iso,
                        max_pages=self.max_pages,
                        max_needed=missing,
                        seen_urls=set(self.storage.seen_ids()),
                        resume_state=resume_state,
                    )
                    if not additional:
                        break
                    if resume_state is None:
                        resume_by_hour[hour_key] = {}
                    else:
                        resume_by_hour[hour_key] = resume_state
                    filtered = self.human_sampler.filter_human_candidates(additional)
                    for pr in filtered:
                        if missing <= 0 or missing_total <= 0:
                            break
                        pr.sampled_hour = hour_key
                        if self.storage.is_recorded_any(pr.url, getattr(pr, "database_id", None)):
                            continue
                        try:
                            enriched_pr = self.enrichment.enrich_pr(pr, skip_first_commit=False)
                            # Top-up candidates must pass the same
                            # post-enrichment checks as primary candidates.
                            if self.human_filter.is_agentic(enriched_pr):
                                continue
                            if not self.human_filter.is_language_allowed(enriched_pr):
                                continue
                            primary_lang = self.human_filter.select_primary_language(enriched_pr)
                            if not primary_lang:
                                continue
                            if kept_by_lang.get(primary_lang, 0) >= self.human_target_per_language:
                                continue
                            if hour_lang_quota.get(primary_lang, 0) <= kept_by_hour_lang.get(
                                hour_key, {}
                            ).get(primary_lang, 0):
                                continue
                            group = self._resolve_group(enriched_pr, human=True)
                            enriched.append(enriched_pr)
                            self._persist_enriched(enriched_pr, group)
                            kept_by_lang[primary_lang] = kept_by_lang.get(primary_lang, 0) + 1
                            kept_by_hour_lang.setdefault(hour_key, {})
                            kept_by_hour_lang[hour_key][primary_lang] = (
                                kept_by_hour_lang[hour_key].get(primary_lang, 0) + 1
                            )
                            missing -= 1
                            missing_total -= 1
                            topped_up += 1
                        except SkipPRError as exc:
                            print(f"[manager] Skipping PR {pr.url}: {exc}")
                        except Exception as exc:
                            print(f"Failed to enrich PR {pr.url}: {exc}")
                    if resume_state is None:
                        break
            if topped_up:
                print(f"[manager] Human top-up added {topped_up} PRs.")
            else:
                print("[manager] Human top-up added 0 PRs.")

        print("[manager] Enrichment complete. PRs buffered and flushed to Parquet.")
        # Local parquet and run manifest are the final artifacts for extraction.
        self.storage.flush_local()
        self._write_run_manifest(len(enriched))
        self.enrichment_checkpoint.clear()

        return enriched
