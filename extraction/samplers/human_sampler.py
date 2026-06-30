"""Time-stratified sampling for human-target PR discovery."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from extraction.config.agent_config import AGENT_RULES
from extraction.config.human_config import EXTRA_EXCLUDE_CLAUSES, TARGET_HUMAN_PRS_PER_LANGUAGE
from extraction.dtos.dtos import PullRequestTest
from extraction.scrapers.discovery_scraper import DiscoveryScraper


class HumanSampler:
    """Sample candidate human PRs using per-hour quotas and agentic exclusions."""

    def __init__(
        self,
        discovery: DiscoveryScraper,
        target_total: Optional[int] = None,
    ) -> None:
        """Initialize the sampler with a discovery client and target size."""
        self.discovery = discovery
        self.target_total = int(target_total or TARGET_HUMAN_PRS_PER_LANGUAGE)

    def _hour_windows(self, start_date: str, end_date: str) -> List[tuple[datetime, datetime]]:
        """Return inclusive hourly UTC windows between inclusive date bounds."""
        start_day = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_day = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_dt = start_day.replace(hour=0, minute=0, second=0)
        end_dt = end_day.replace(hour=23, minute=59, second=59)
        windows: List[tuple[datetime, datetime]] = []
        current = start_dt
        while current <= end_dt:
            hour_end = min(current + timedelta(hours=1) - timedelta(seconds=1), end_dt)
            windows.append((current, hour_end))
            current += timedelta(hours=1)
        return windows

    def _body_markers_from_clause(self, clause: str) -> list[str]:
        """Extract simple body markers from a supported search fragment."""
        without_scope = (
            clause.replace("in:body", "")
            .replace("in:comments", "")
            .replace("in:commits", "")
        )
        parts = re.split(r"\s+OR\s+", without_scope, flags=re.IGNORECASE)
        markers: list[str] = []
        for part in parts:
            marker = part.strip().strip("()").strip().strip('"').strip().lower()
            if marker and not marker.startswith(("author:", "head:")):
                markers.append(marker)
        return markers

    def _matches_search_fragment(
        self,
        clause: str,
        *,
        author_login: str,
        head_ref: str,
        body_lower: str,
    ) -> bool:
        """Return True when a discovery candidate matches a supported fragment."""
        normalized = clause.strip().strip("()")
        if normalized.startswith("author:"):
            target = normalized.split("author:", 1)[1].strip().strip('"').lower()
            return bool(target) and author_login.lower() == target
        if normalized.startswith("head:"):
            prefix = normalized.split("head:", 1)[1].strip().rstrip("/").lower()
            return bool(prefix) and head_ref.startswith(prefix)
        return any(marker in body_lower for marker in self._body_markers_from_clause(clause))

    def _is_agentic_candidate(self, pr: PullRequestTest) -> bool:
        """Return True if a discovery candidate should be excluded from human data."""
        author_login = (pr.author.login if pr.author else "") or ""
        author_type = (pr.author.typename if pr.author else "") or ""
        head_ref = (pr.head_ref_name or "").lower()
        body_lower = (pr.body or "").lower()

        # Exclude every GitHub bot identity from human sampling, even if the bot
        # is not one of the thesis agent targets.
        if author_login.lower().endswith("[bot]") or author_type.lower() == "bot":
            return True

        for clause_list in AGENT_RULES.values():
            for clause in clause_list:
                if self._matches_search_fragment(
                    clause,
                    author_login=author_login,
                    head_ref=head_ref,
                    body_lower=body_lower,
                ):
                    return True

        for excl in EXTRA_EXCLUDE_CLAUSES:
            if self._matches_search_fragment(
                excl,
                author_login=author_login,
                head_ref=head_ref,
                body_lower=body_lower,
            ):
                return True

        return False

    def filter_human_candidates(self, prs: List[PullRequestTest]) -> List[PullRequestTest]:
        """Return candidates that do not match pre-enrichment agentic/bot exclusions."""
        return [pr for pr in prs if not self._is_agentic_candidate(pr)]

    def sample(
        self,
        start_date: str,
        end_date: str,
        max_pages: int,
        seen_urls: Optional[Set[str]] = None,
    ) -> Dict[str, object]:
        """Sample human candidates across hourly windows."""
        seen_urls = seen_urls or set()
        windows = self._hour_windows(start_date, end_date)
        if not windows:
            return {"primary": [], "quota_by_hour": {}, "resume_by_hour": {}}
        total_hours = len(windows)
        base_quota = self.target_total // total_hours
        remainder = self.target_total % total_hours

        primary: List[PullRequestTest] = []
        quota_by_hour: Dict[str, int] = {}
        resume_by_hour: Dict[str, Dict[str, Optional[str]]] = {}

        print(
            f"[human_sampler] Sampling {start_date}..{end_date}: "
            f"hours={total_hours}, target={self.target_total}, "
            f"base_quota={base_quota}, remainder={remainder}"
        )
        for idx, (hour_start, hour_end) in enumerate(windows):
            # Distribute the remainder over earlier hours for deterministic
            # quota assignment across retries.
            quota = base_quota + (1 if idx < remainder else 0)
            if quota == 0:
                continue
            discovery_quota = quota
            hour_key = hour_start.strftime("%Y-%m-%dT%H")
            quota_by_hour[hour_key] = quota
            start_iso = hour_start.isoformat().replace("+00:00", "Z")
            end_iso = hour_end.isoformat().replace("+00:00", "Z")
            print(
                f"[human_sampler] Hour {hour_key}: quota={quota}, "
                f"discover={discovery_quota}, window={start_iso}..{end_iso}"
            )
            filtered: List[PullRequestTest] = []
            resume_state: Optional[Dict[str, Optional[str]]] = None
            while len(filtered) < discovery_quota:
                needed = discovery_quota - len(filtered)
                hourly, resume_state = self.discovery.discover_prs_between_limited(
                    start_iso=start_iso,
                    end_iso=end_iso,
                    max_pages=max_pages,
                    max_needed=needed,
                    seen_urls=seen_urls,
                    resume_state=resume_state,
                )
                if not hourly:
                    break
                print(f"[human_sampler] Hour {hour_key}: discovered={len(hourly)}")
                chunk_filtered = self.filter_human_candidates(hourly)
                print(
                    f"[human_sampler] Hour {hour_key}: after_agentic_filter={len(chunk_filtered)}"
                )
                if not chunk_filtered:
                    if resume_state is None:
                        break
                    continue
                for pr in chunk_filtered:
                    pr.sampled_hour = hour_key
                filtered.extend(chunk_filtered)
                if resume_state is None:
                    break
            if not filtered:
                continue
            chosen_primary = filtered[:discovery_quota]
            primary.extend(chosen_primary)
            if resume_state:
                resume_by_hour[hour_key] = resume_state
            print(f"[human_sampler] Hour {hour_key}: candidates={len(chosen_primary)}")

        print(f"[human_sampler] Completed sampling: primary_total={len(primary)}")
        return {
            "primary": primary,
            "quota_by_hour": quota_by_hour,
            "resume_by_hour": resume_by_hour,
        }
