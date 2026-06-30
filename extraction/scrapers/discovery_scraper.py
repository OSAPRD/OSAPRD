"""Discover candidate pull requests from live GitHub APIs.

Discovery is intentionally lightweight: it finds candidate PR URLs and minimal
metadata, then hands those candidates to enrichment. The heavy PR/repository
payloads are not fetched here.

The main path uses GitHub GraphQL issue search. Because GitHub search caps a
query at roughly 1,000 results, the scraper advances or slices time windows when
it approaches that cap. A REST commit-search backfill supplements GraphQL for
agents that are more reliably identified by first-commit author.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import requests

from extraction.config.agent_config import FIRST_COMMIT_AUTHORS
from extraction.config.settings import ExtractionSettings
from extraction.config.storage_config import LOCAL_OUTPUT_DIR
from extraction.dtos.dtos import PullRequestTest, UserPeek
from extraction.filters.agent_filter import AgentFilter
from extraction.utility.checkpoint_handler import CheckpointHandler
from extraction.utility.token_manager import TokenManager

# Preview media types required by GitHub's commit search and "pulls for commit"
# REST endpoints.
COMMIT_SEARCH_ACCEPT = "application/vnd.github.cloak-preview"
PULLS_FOR_COMMIT_ACCEPT = "application/vnd.github.groot-preview+json"
GRAPHQL_URL = "https://api.github.com/graphql"

# When GraphQL resource limits are hit, discovery halves time windows until this
# minimum slice size. Below that, it retries a few times before advancing by one
# second to keep the run moving.
MIN_RESOURCE_SLICE_SECONDS = 60 * 60
MAX_RESOURCE_RETRIES = 5


class ResourceLimitError(RuntimeError):
    """Raised when GitHub GraphQL resource limits require window slicing."""

    def __init__(self, errors: List[dict]) -> None:
        super().__init__("GraphQL resource limits exceeded.")
        self.errors = errors


class DiscoveryScraper:
    """Discover candidate PRs with time slicing, pagination, and checkpointing."""

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        base_url: str = "https://api.github.com",
        agent_filter: Optional[AgentFilter] = None,
        checkpoint_path: Optional[Path] = None,
        settings: Optional[ExtractionSettings] = None,
    ) -> None:
        """Initialize GitHub clients, filters, token rotation, and checkpoints."""
        if settings is not None and tokens is None:
            tokens = list(settings.github_tokens)
        token_list = [token.strip() for token in (tokens or []) if token and token.strip()]
        if not token_list:
            raise RuntimeError("No GitHub tokens provided.")
        self.token_manager = TokenManager(token_list)
        self.base_url = base_url.rstrip("/")
        self.graphql_url = GRAPHQL_URL
        self.agent_filter = agent_filter or AgentFilter()
        self.session = requests.Session()
        # A caller can provide an explicit checkpoint path. Production runs store
        # discovery checkpoints under the configured local output dir.
        local_output_dir = (
            settings.local_output_dir if settings is not None else Path(LOCAL_OUTPUT_DIR)
        )
        ckpt_path = checkpoint_path or (
            local_output_dir / "checkpoints" / "discovery_checkpoint.json"
        )
        self.checkpoint = CheckpointHandler(ckpt_path)

    def _headers_graphql(self) -> dict:
        """Build headers for GraphQL requests using the current token."""
        token = self.token_manager.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

    def _get_rate_limit(self) -> dict:
        """Fetch current GraphQL rate limit information."""
        query = """
        query {
          rateLimit {
            limit
            cost
            remaining
            resetAt
          }
        }
        """
        resp = self.session.post(
            self.graphql_url,
            headers=self._headers_graphql(),
            json={"query": query},
            timeout=15,
        )
        resp.raise_for_status()
        return (resp.json().get("data") or {}).get("rateLimit", {}) or {}

    def _update_rate_limit_info(self, rate_limit: dict) -> None:
        """Update token manager state from GraphQL rate limit data."""
        remaining = rate_limit.get("remaining")
        reset_at = rate_limit.get("resetAt")
        if remaining is None or reset_at is None:
            return
        try:
            reset_ts = int(datetime.fromisoformat(reset_at.replace("Z", "+00:00")).timestamp())
            self.token_manager.update_limit(int(remaining), reset_ts)
        except Exception:
            return

    def _handle_rate_limit(self) -> None:
        """Rotate tokens or sleep until the next reset when exhausted."""
        try:
            self.token_manager.rotate_token()
            time.sleep(2)
            return
        except RuntimeError:
            # Fall back to waiting on the shortest reset if all tokens are exhausted.
            reset_times = [t for t in self.token_manager.get_all_reset_times() if t]
            if not reset_times:
                time.sleep(60)
                return
            now = time.time()
            shortest_reset = min(reset_times)
            if shortest_reset < now:
                self.token_manager.update_index(
                    self.token_manager.get_all_reset_times().index(shortest_reset)
                )
                return
            wait_time = shortest_reset - now + 30
            print(f"[discovery] All tokens exhausted, waiting {wait_time / 60:.1f} minutes.")
            time.sleep(wait_time)

    def _headers_commit_search(self) -> dict:
        """Build headers for commit search preview API."""
        token = self.token_manager.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": COMMIT_SEARCH_ACCEPT,
        }

    def _headers_commit_pulls(self) -> dict:
        """Build headers for the pulls-for-commit preview API."""
        token = self.token_manager.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": PULLS_FOR_COMMIT_ACCEPT,
        }

    def _headers_commits_list(self) -> dict:
        """Build headers for PR commits listing."""
        token = self.token_manager.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

    def _is_invalid_token_response(self, resp: requests.Response) -> bool:
        """Return True if a response indicates invalid credentials."""
        if resp.status_code not in (401, 403):
            return False
        text = (resp.text or "").lower()
        return (
            "bad credentials" in text
            or "requires authentication" in text
            or "invalid token" in text
        )

    def _expand_date_bounds(self, start_date: str, end_date: str) -> Tuple[str, str]:
        """Return full-range ISO timestamps (UTC) for the provided dates."""
        start_day = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_day = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_dt = start_day.replace(hour=0, minute=0, second=0)
        end_dt = end_day.replace(hour=23, minute=59, second=59)
        return (
            start_dt.isoformat().replace("+00:00", "Z"),
            end_dt.isoformat().replace("+00:00", "Z"),
        )

    def _advance_window_start(self, current_start: str, last_created_at: Optional[str]) -> str:
        """Advance the window start to the last createdAt (or +1s if unchanged)."""
        if not last_created_at:
            return current_start
        try:
            current_dt = datetime.fromisoformat(current_start.replace("Z", "+00:00"))
            last_dt = datetime.fromisoformat(last_created_at.replace("Z", "+00:00"))
        except Exception:
            return current_start
        if last_dt <= current_dt:
            last_dt = current_dt + timedelta(seconds=1)
        return last_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _parse_iso(self, value: str) -> datetime:
        """Parse ISO timestamps into timezone-aware datetimes."""
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

    def _midpoint_iso(self, start_iso: str, end_iso: str) -> str:
        """Return the midpoint ISO timestamp between start and end (UTC)."""
        start_dt = self._parse_iso(start_iso)
        end_dt = self._parse_iso(end_iso)
        midpoint = start_dt + (end_dt - start_dt) / 2
        if midpoint <= start_dt:
            midpoint = start_dt + timedelta(seconds=1)
        return midpoint.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _build_graphql_search_query(self) -> str:
        """Return GraphQL query text for PR search."""
        return """
        query($queryString: String!, $first: Int!, $after: String) {
          search(type: ISSUE, query: $queryString, first: $first, after: $after) {
            issueCount
            pageInfo {
              endCursor
              hasNextPage
            }
            nodes {
              ... on PullRequest {
                id
                databaseId
                title
                url
                bodyText
                createdAt
                mergedAt
                closedAt
                isDraft
                headRefName
                author {
                  login
                  url
                  __typename
                }
              }
            }
          }
        }
        """

    def _agent_clause_pairs(
        self, agent_names: Optional[Sequence[str]] = None
    ) -> List[Tuple[Optional[str], str]]:
        """Return (agent, clause) pairs for agentic discovery."""
        pairs: List[Tuple[Optional[str], str]] = []
        selected = {name.lower() for name in agent_names} if agent_names is not None else None
        for agent_name, clauses in self.agent_filter.filter_rules.items():
            if selected is not None and agent_name.lower() not in selected:
                continue
            for clause in clauses:
                pairs.append((agent_name, clause))
        return pairs

    def _request_graphql(self, query: str, variables: Dict[str, Any]) -> dict:
        """
        Send a GraphQL request with retry, token rotation, and rate-limit handling.

        GitHub can return recoverable internal errors for large search windows.
        This method retries those for up to one hour. GraphQL resource-limit
        errors are raised to the caller so the caller can shrink the time window.
        """
        backoff = 1
        max_retry_seconds = 60 * 60
        retry_start = time.time()
        while True:
            try:
                resp = self.session.post(
                    self.graphql_url,
                    headers=self._headers_graphql(),
                    json={"query": query, "variables": variables},
                    timeout=30,
                )
                if self._is_invalid_token_response(resp):
                    self.token_manager.invalidate_current()
                    backoff = 1
                    continue
                if resp.status_code == 403 and "rate limit" in resp.text.lower():
                    self._handle_rate_limit()
                    backoff = 1
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    rate_limited = False
                    resource_limited = False
                    retryable_internal = False
                    for err in data["errors"]:
                        msg = (err.get("message") or "").lower()
                        if (
                            "something went wrong while executing your query" in msg
                            or "please include" in msg
                        ):
                            retryable_internal = True
                            break
                        if (
                            err.get("type") == "RESOURCE_LIMITS_EXCEEDED"
                            or "resource limits" in msg
                        ):
                            resource_limited = True
                            break
                        if "bad credentials" in msg or "requires authentication" in msg:
                            self.token_manager.invalidate_current()
                            rate_limited = True
                            break
                        if err.get("type") in ("RATE_LIMITED", "RATE_LIMIT"):
                            rate_limited = True
                            rate_limit = self._get_rate_limit()
                            self._update_rate_limit_info(rate_limit)
                            self._handle_rate_limit()
                            break
                    if retryable_internal:
                        print(f"[discovery] GraphQL internal error payload: {data['errors']}")
                        elapsed = time.time() - retry_start
                        if elapsed >= max_retry_seconds:
                            raise RuntimeError(
                                f"GraphQL request failed after waiting {elapsed / 60:.1f} minutes."
                            )
                        print(
                            "[discovery] GraphQL internal error, retrying in "
                            f"{backoff}s (elapsed {elapsed / 60:.1f}m)..."
                        )
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue
                    if rate_limited:
                        backoff = 1
                        continue
                    if resource_limited:
                        raise ResourceLimitError(data["errors"])
                    raise RuntimeError(f"GraphQL errors: {data['errors']}")
                rate_limit = self._get_rate_limit()
                self._update_rate_limit_info(rate_limit)
                return data.get("data", {})
            except requests.RequestException as exc:
                elapsed = time.time() - retry_start
                if elapsed >= max_retry_seconds:
                    raise RuntimeError(
                        f"GraphQL request failed after waiting {elapsed / 60:.1f} minutes."
                    ) from exc
                print(
                    "[discovery] GraphQL request failed, retrying in "
                    f"{backoff}s (elapsed {elapsed / 60:.1f}m)..."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _request_search_commits(self, query: str, page: int, per_page: int = 100) -> dict:
        """Query the REST commit-search endpoint with pagination and retries."""
        url = f"{self.base_url}/search/commits"
        retries = 0
        backoff = 2
        while True:
            try:
                resp = self.session.get(
                    url,
                    headers=self._headers_commit_search(),
                    params={"q": query, "per_page": per_page, "page": page},
                    timeout=15,
                )
                if self._is_invalid_token_response(resp):
                    self.token_manager.invalidate_current()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                if resp.status_code == 403 and "rate limit" in resp.text.lower():
                    self.token_manager.rotate_token()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                resp.raise_for_status()
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset = resp.headers.get("X-RateLimit-Reset")
                if remaining is not None and reset is not None:
                    try:
                        self.token_manager.update_limit(int(remaining), int(reset))
                    except ValueError:
                        pass
                return resp.json()
            except requests.RequestException:
                retries += 1
                if retries > 5:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _parse_pr_url(self, pr_url: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Parse owner/repo/number from a GitHub PR URL."""
        if not pr_url:
            return None, None, None
        parts = pr_url.rstrip("/").split("/")
        if len(parts) < 5:
            return None, None, None
        try:
            owner, repo, number = parts[-4], parts[-3], int(parts[-1])
            return owner, repo, number
        except (ValueError, IndexError):
            return None, None, None

    def _is_seen(self, seen: Set[str], url: Optional[str], database_id: Optional[str]) -> bool:
        """Return True if the URL or database id is already recorded."""
        if url and url in seen:
            return True
        if database_id and database_id in seen:
            return True
        return False

    def _mark_seen(self, seen: Set[str], url: Optional[str], database_id: Optional[str]) -> None:
        """Track the URL and database id as seen."""
        if url:
            seen.add(url)
        if database_id:
            seen.add(database_id)

    def _request_first_commit_author(self, owner: str, repo: str, number: int) -> Optional[str]:
        """Return the first commit author login/name for a PR, if available."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{number}/commits"
        retries = 0
        backoff = 2
        while True:
            try:
                resp = self.session.get(
                    url,
                    headers=self._headers_commits_list(),
                    params={"per_page": 1, "page": 1},
                    timeout=15,
                )
                if self._is_invalid_token_response(resp):
                    self.token_manager.invalidate_current()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                if resp.status_code == 403 and "rate limit" in resp.text.lower():
                    self.token_manager.rotate_token()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                resp.raise_for_status()
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset = resp.headers.get("X-RateLimit-Reset")
                if remaining is not None and reset is not None:
                    try:
                        self.token_manager.update_limit(int(remaining), int(reset))
                    except ValueError:
                        pass
                data = resp.json()
                if not isinstance(data, list) or not data:
                    return None
                commit = data[0]
                return (commit.get("author") or {}).get("login") or (
                    commit.get("commit") or {}
                ).get("author", {}).get("name")
            except requests.RequestException:
                retries += 1
                if retries > 5:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _parse_pr_item(self, item: dict, discovered_agent: Optional[str] = None) -> PullRequestTest:
        """Convert a REST PR item into the lightweight discovery DTO."""
        user = item.get("user") or {}
        head_ref = (item.get("head") or {}).get("ref")
        author = UserPeek(
            id=str(user.get("id")) if user.get("id") is not None else None,
            login=user.get("login"),
            url=user.get("html_url"),
            name=None,
            typename=user.get("type"),
        )
        return PullRequestTest(
            id=str(item.get("id")),
            database_id=str(item.get("id")) if item.get("id") is not None else None,
            title=item.get("title") or "",
            author=author,
            url=item.get("html_url") or "",
            body=item.get("body") or "",
            created_at=item.get("created_at") or "",
            is_draft=bool(item.get("draft", False)),
            additions=0,
            deletions=0,
            changed_files=0,
            commits=0,
            comments=item.get("comments") or 0,
            reviews=0,
            merged_at=item.get("merged_at"),
            closed_at=item.get("closed_at"),
            head_ref_name=head_ref,
            discovered_agent=discovered_agent,
        )

    def _pulls_for_commit(self, owner: str, repo: str, sha: str) -> List[dict]:
        """List PRs associated with a commit SHA using GitHub's preview endpoint."""
        url = f"{self.base_url}/repos/{owner}/{repo}/commits/{sha}/pulls"
        retries = 0
        backoff = 2
        while True:
            try:
                resp = self.session.get(
                    url,
                    headers=self._headers_commit_pulls(),
                    timeout=15,
                )
                if self._is_invalid_token_response(resp):
                    self.token_manager.invalidate_current()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                if resp.status_code == 403 and "rate limit" in resp.text.lower():
                    self.token_manager.rotate_token()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                resp.raise_for_status()
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset = resp.headers.get("X-RateLimit-Reset")
                if remaining is not None and reset is not None:
                    try:
                        self.token_manager.update_limit(int(remaining), int(reset))
                    except ValueError:
                        pass
                data = resp.json()
                return data if isinstance(data, list) else []
            except requests.RequestException:
                retries += 1
                if retries > 5:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _discover_prs_from_commits(
        self,
        author: str,
        discovered_agent: Optional[str],
        start_iso: str,
        end_iso: str,
        seen_urls: set,
        max_pages: int,
        start_page: int = 1,
        checkpoint_target: str = "agentic",
        on_discovered: Optional[Callable[[PullRequestTest], None]] = None,
        on_page_complete: Optional[Callable[[str], None]] = None,
    ) -> List[PullRequestTest]:
        """
        Discover PRs by searching commits authored by a known agent account.

        This backfill catches PRs whose PR author/body/branch does not identify
        the agent, but whose first commit author does. It verifies first-commit
        authorship before accepting a PR so later commits by the agent do not
        incorrectly classify human-authored PRs.
        """
        results: List[PullRequestTest] = []
        agent_label = discovered_agent or author
        query = f"author:{author} author-date:{start_iso}..{end_iso}"
        print(f"[discovery] Commit search query: {query}")
        for page in range(start_page, max_pages + 1 if max_pages else start_page + 1000000):
            if max_pages and page > max_pages:
                self.checkpoint.save(
                    {
                        "phase": "commit",
                        "target": checkpoint_target,
                        "start_date": start_iso,
                        "end_date": end_iso,
                        "commit_page": page,
                    }
                )
                return results
            print(f"[discovery] Fetching commit page {page}...")
            data = self._request_search_commits(query, page=page)
            items = data.get("items", [])
            if not items:
                print("[discovery] No commit items returned; stopping commit search.")
                break
            for commit in items:
                repo_url = (commit.get("repository") or {}).get("html_url") or ""
                parts = repo_url.rstrip("/").split("/")
                if len(parts) < 2:
                    continue
                owner, repo = parts[-2], parts[-1]
                sha = commit.get("sha")
                if not sha:
                    continue
                pulls = self._pulls_for_commit(owner, repo, sha)
                for pr_item in pulls:
                    pr_url = pr_item.get("html_url") or ""
                    owner, repo, number = self._parse_pr_url(pr_url)
                    if owner and repo and number:
                        # Commit search returns any commit by the author. The
                        # thesis extraction backfill only wants PRs where the
                        # first commit identifies the agent.
                        first_author = self._request_first_commit_author(owner, repo, number)
                        if first_author and first_author.lower() != author.lower():
                            continue
                    pr_obj = self._parse_pr_item(pr_item, discovered_agent=agent_label)
                    if not self._is_seen(seen_urls, pr_obj.url, pr_obj.database_id):
                        self._mark_seen(seen_urls, pr_obj.url, pr_obj.database_id)
                        results.append(pr_obj)
                        if on_discovered:
                            on_discovered(pr_obj)
            if on_page_complete:
                on_page_complete(agent_label)
            if len(items) < 100:
                print("[discovery] Last commit page reached (less than 100 items).")
                break
            # Persist commit pagination progress for resume.
            self.checkpoint.save(
                {
                    "phase": "commit",
                    "target": checkpoint_target,
                    "start_date": start_iso,
                    "end_date": end_iso,
                    "commit_page": page + 1,
                }
            )
        return results

    def discover_prs_between_limited(
        self,
        start_iso: str,
        end_iso: str,
        max_pages: int,
        max_needed: int,
        seen_urls: Optional[Set[str]] = None,
        resume_state: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[List[PullRequestTest], Optional[Dict[str, Optional[str]]]]:
        """Discover PRs within a precise ISO time window (UTC) with an early-stop limit."""
        results: List[PullRequestTest] = []
        seen_urls = seen_urls or set()
        graphql_query = self._build_graphql_search_query()

        cursor = (resume_state or {}).get("cursor")
        active_start = (resume_state or {}).get("start_iso") or start_iso
        active_end = (resume_state or {}).get("end_iso") or end_iso
        last_created_at = (resume_state or {}).get("last_created_at")
        page = 0
        window_count = 0
        resource_backoff = 2
        resource_retries = 0

        while True:
            query_string = f"is:pr created:{active_start}..{active_end} sort:created-asc"
            print(f"[discovery] Human GraphQL query: {query_string}")
            page += 1
            if max_pages and page > max_pages:
                break
            try:
                data = self._request_graphql(
                    graphql_query, {"queryString": query_string, "first": 100, "after": cursor}
                )
            except ResourceLimitError:
                start_dt = self._parse_iso(active_start)
                end_dt = self._parse_iso(active_end)
                window_seconds = (end_dt - start_dt).total_seconds()
                if window_seconds <= MIN_RESOURCE_SLICE_SECONDS:
                    resource_retries += 1
                    if resource_retries > MAX_RESOURCE_RETRIES:
                        print(
                            "[discovery] Resource limit persists at minimum slice; skipping ahead by 1s."
                        )
                        active_start = self._advance_window_start(active_start, active_start)
                        cursor = None
                        window_count = 0
                        page = 0
                        resource_retries = 0
                        continue
                    print(f"[discovery] Resource limit hit; backing off {resource_backoff}s.")
                    time.sleep(resource_backoff)
                    resource_backoff = min(resource_backoff * 2, 60)
                    continue
                active_end = self._midpoint_iso(active_start, active_end)
                print(
                    f"[discovery] Resource limit hit; slicing window to {active_start}..{active_end}"
                )
                cursor = None
                window_count = 0
                page = 0
                resource_retries = 0
                continue
            search = data.get("search") or {}
            nodes = search.get("nodes") or []
            page_info = search.get("pageInfo") or {}
            end_cursor = page_info.get("endCursor")
            has_next = page_info.get("hasNextPage")

            if not nodes:
                break

            for node in nodes:
                if not node:
                    continue
                last_created_at = node.get("createdAt") or last_created_at
                pr = PullRequestTest(
                    id=str(node.get("id")),
                    database_id=(
                        str(node.get("databaseId")) if node.get("databaseId") is not None else None
                    ),
                    title=node.get("title") or "",
                    author=UserPeek(
                        login=((node.get("author") or {}).get("login")),
                        url=((node.get("author") or {}).get("url")),
                        typename=((node.get("author") or {}).get("__typename")),
                    ),
                    url=node.get("url") or "",
                    body=node.get("bodyText") or "",
                    created_at=node.get("createdAt") or "",
                    is_draft=bool(node.get("isDraft", False)),
                    additions=0,
                    deletions=0,
                    changed_files=0,
                    commits=0,
                    comments=0,
                    reviews=0,
                    merged_at=node.get("mergedAt"),
                    closed_at=node.get("closedAt"),
                    head_ref_name=node.get("headRefName"),
                    discovered_agent=None,
                )
                if not self._is_seen(seen_urls, pr.url, pr.database_id):
                    self._mark_seen(seen_urls, pr.url, pr.database_id)
                    results.append(pr)

            window_count += len(nodes)
            cursor = end_cursor

            if max_needed and len(results) >= max_needed:
                # Return enough cursor/window state for the manager to continue
                # this hour if enrichment later rejects some candidates.
                return results, {
                    "start_iso": active_start,
                    "end_iso": active_end,
                    "cursor": cursor,
                    "last_created_at": last_created_at,
                }

            if not has_next:
                cursor = None
                window_count = 0
                if self._parse_iso(active_end) < self._parse_iso(end_iso):
                    active_start = self._advance_window_start(active_end, active_end)
                    active_end = end_iso
                    page = 0
                    continue
                break

            if window_count + 200 >= 1000:
                # See `discover_prs_between`: keep each search below GitHub's
                # result cap by advancing to the last observed creation time.
                active_start = self._advance_window_start(active_start, last_created_at)
                if active_start >= active_end:
                    break
                cursor = None
                window_count = 0
                page = 0
                continue

        return results, None

    def discover_prs(
        self,
        start_iso: str,
        end_iso: str,
        target: str = "agentic",
        max_pages: int = 10,
        seen_urls: Optional[Set[str]] = None,
    ) -> List[PullRequestTest]:
        """Discover PRs within a date window for one extraction target."""
        normalized_target = (target or "agentic").strip().lower()
        human = normalized_target == "human"
        if human:
            selected_agents: Optional[tuple[str, ...]] = ()
        elif normalized_target == "agentic":
            selected_agents = None
        else:
            selected_agents = (normalized_target,)

        all_results: List[PullRequestTest] = []
        seen_urls = seen_urls or set()
        total_discovered = 0
        per_agent_counts: Dict[str, int] = defaultdict(int)

        def _log_discovery(pr_obj: PullRequestTest) -> None:
            nonlocal total_discovered
            label = pr_obj.discovered_agent or ("human" if human else "unknown")
            total_discovered += 1
            per_agent_counts[label] += 1

        def _log_summary(label: str, context: str) -> None:
            print(
                f"[discovery] {context}: {total_discovered} total; "
                f"{label}={per_agent_counts.get(label, 0)}"
            )

        checkpoint = self.checkpoint.load()
        # Checkpoints are target/date scoped. Reusing the same output directory
        # for a different target or window should start discovery from scratch.
        if (
            checkpoint.get("start_date") != start_iso
            or checkpoint.get("end_date") != end_iso
            or checkpoint.get("target", normalized_target) != normalized_target
        ):
            checkpoint = {}

        phase = checkpoint.get("phase", "graphql")
        clause_index = int(checkpoint.get("clause_index", 0))
        cursor_after = checkpoint.get("cursor")
        current_start = checkpoint.get("current_start")
        current_end = checkpoint.get("current_end")
        count_in_window = int(checkpoint.get("count_in_window", 0))

        window_start, window_end = self._expand_date_bounds(start_iso, end_iso)
        # Human discovery uses a broad no-agent query. Agentic discovery uses
        # one search clause at a time so each agent/fragment can be checkpointed.
        clause_pairs = [(None, "")] if human else self._agent_clause_pairs(selected_agents)
        if not human and not clause_pairs:
            raise ValueError(f"No discovery clauses configured for target={normalized_target!r}.")
        graphql_query = self._build_graphql_search_query()

        if phase == "graphql":
            if checkpoint:
                print(
                    "[discovery] Resuming from checkpoint: "
                    f"clause={clause_index}, start={current_start}, cursor={cursor_after}, count={count_in_window}"
                )
            for ci in range(clause_index, len(clause_pairs)):
                discovered_agent, clause = clause_pairs[ci]
                label = discovered_agent or ("human" if human else "unknown")
                active_start = (
                    current_start if ci == clause_index and current_start else window_start
                )
                active_end = current_end if ci == clause_index and current_end else window_end
                cursor = cursor_after if ci == clause_index else None
                window_count = count_in_window if ci == clause_index else 0
                page = 0
                resource_backoff = 2
                resource_retries = 0

                while True:
                    query_string = f"is:pr created:{active_start}..{active_end} sort:created-asc"
                    if clause:
                        query_string = (
                            f"is:pr {clause} created:{active_start}..{active_end} sort:created-asc"
                        )
                    print(f"[discovery] GraphQL query: {query_string}")
                    page += 1
                    if max_pages and page > max_pages:
                        print("[discovery] Max pages reached for this clause; checkpointing.")
                        self.checkpoint.save(
                            {
                                "phase": "graphql",
                                "target": normalized_target,
                                "start_date": start_iso,
                                "end_date": end_iso,
                                "clause_index": ci,
                                "cursor": cursor,
                                "current_start": active_start,
                                "current_end": active_end,
                                "count_in_window": window_count,
                            }
                        )
                        return all_results

                    try:
                        data = self._request_graphql(
                            graphql_query,
                            {"queryString": query_string, "first": 100, "after": cursor},
                        )
                    except ResourceLimitError:
                        # Large or expensive GraphQL searches are retried with
                        # smaller time windows rather than failing the whole
                        # extraction run.
                        start_dt = self._parse_iso(active_start)
                        end_dt = self._parse_iso(active_end)
                        window_seconds = (end_dt - start_dt).total_seconds()
                        if window_seconds <= MIN_RESOURCE_SLICE_SECONDS:
                            resource_retries += 1
                            if resource_retries > MAX_RESOURCE_RETRIES:
                                print(
                                    "[discovery] Resource limit persists at minimum slice; skipping ahead by 1s."
                                )
                                active_start = self._advance_window_start(
                                    active_start, active_start
                                )
                                cursor = None
                                window_count = 0
                                page = 0
                                resource_retries = 0
                                self.checkpoint.save(
                                    {
                                        "phase": "graphql",
                                        "target": normalized_target,
                                        "start_date": start_iso,
                                        "end_date": end_iso,
                                        "clause_index": ci,
                                        "cursor": cursor,
                                        "current_start": active_start,
                                        "current_end": active_end,
                                        "count_in_window": window_count,
                                    }
                                )
                                continue
                            print(
                                f"[discovery] Resource limit hit; backing off {resource_backoff}s."
                            )
                            time.sleep(resource_backoff)
                            resource_backoff = min(resource_backoff * 2, 60)
                            continue
                        active_end = self._midpoint_iso(active_start, active_end)
                        print(
                            f"[discovery] Resource limit hit; slicing window to {active_start}..{active_end}"
                        )
                        cursor = None
                        window_count = 0
                        page = 0
                        resource_retries = 0
                        self.checkpoint.save(
                            {
                                "phase": "graphql",
                                "target": normalized_target,
                                "start_date": start_iso,
                                "end_date": end_iso,
                                "clause_index": ci,
                                "cursor": cursor,
                                "current_start": active_start,
                                "current_end": active_end,
                                "count_in_window": window_count,
                            }
                        )
                        continue
                    search = data.get("search") or {}
                    nodes = search.get("nodes") or []
                    page_info = search.get("pageInfo") or {}
                    end_cursor = page_info.get("endCursor")
                    has_next = page_info.get("hasNextPage")

                    if not nodes:
                        break

                    last_created_at = None
                    for node in nodes:
                        if not node:
                            continue
                        last_created_at = node.get("createdAt") or last_created_at
                        pr = PullRequestTest(
                            id=str(node.get("id")),
                            database_id=(
                                str(node.get("databaseId"))
                                if node.get("databaseId") is not None
                                else None
                            ),
                            title=node.get("title") or "",
                            author=UserPeek(
                                login=((node.get("author") or {}).get("login")),
                                url=((node.get("author") or {}).get("url")),
                                typename=((node.get("author") or {}).get("__typename")),
                            ),
                            url=node.get("url") or "",
                            body=node.get("bodyText") or "",
                            created_at=node.get("createdAt") or "",
                            is_draft=bool(node.get("isDraft", False)),
                            additions=0,
                            deletions=0,
                            changed_files=0,
                            commits=0,
                            comments=0,
                            reviews=0,
                            merged_at=node.get("mergedAt"),
                            closed_at=node.get("closedAt"),
                            discovered_agent=discovered_agent,
                        )
                        if not self._is_seen(seen_urls, pr.url, pr.database_id):
                            self._mark_seen(seen_urls, pr.url, pr.database_id)
                            all_results.append(pr)
                            _log_discovery(pr)

                    window_count += len(nodes)
                    cursor = end_cursor
                    _log_summary(label, "Query complete")
                    self.checkpoint.save(
                        {
                            "phase": "graphql",
                            "target": normalized_target,
                            "start_date": start_iso,
                            "end_date": end_iso,
                            "clause_index": ci,
                            "cursor": cursor,
                            "current_start": active_start,
                            "current_end": active_end,
                            "count_in_window": window_count,
                        }
                    )

                    if not has_next:
                        cursor = None
                        window_count = 0
                        if self._parse_iso(active_end) < self._parse_iso(window_end):
                            # Continue from a sliced sub-window back toward the
                            # original full date window.
                            active_start = self._advance_window_start(active_end, active_end)
                            active_end = window_end
                            self.checkpoint.save(
                                {
                                    "phase": "graphql",
                                    "target": normalized_target,
                                    "start_date": start_iso,
                                    "end_date": end_iso,
                                    "clause_index": ci,
                                    "cursor": cursor,
                                    "current_start": active_start,
                                    "current_end": active_end,
                                    "count_in_window": window_count,
                                }
                            )
                            page = 0
                            continue
                        break

                    if window_count + 200 >= 1000:
                        # Avoid GitHub's search cap by advancing the lower
                        # bound and restarting pagination for this clause.
                        active_start = self._advance_window_start(active_start, last_created_at)
                        if active_start >= active_end:
                            break
                        cursor = None
                        window_count = 0
                        _log_summary(label, "Window advanced")
                        self.checkpoint.save(
                            {
                                "phase": "graphql",
                                "target": normalized_target,
                                "start_date": start_iso,
                                "end_date": end_iso,
                                "clause_index": ci,
                                "cursor": cursor,
                                "current_start": active_start,
                                "current_end": active_end,
                                "count_in_window": window_count,
                            }
                        )
                        page = 0
                        continue
                cursor_after = None
                current_start = None
                current_end = None
                count_in_window = 0

            phase = "commit" if not human else "done"
            self.checkpoint.save(
                {
                    "phase": phase,
                    "target": normalized_target,
                    "start_date": start_iso,
                    "end_date": end_iso,
                    "clause_index": 0,
                    "cursor": None,
                    "current_start": window_start,
                    "count_in_window": 0,
                    "commit_page": 1,
                }
            )

        # Additional commit-based discovery for first-commit-authored PRs.
        if not human:
            commit_page = int(checkpoint.get("commit_page", 1)) if phase == "commit" else 1
            if normalized_target == "agentic":
                commit_author_items = FIRST_COMMIT_AUTHORS.items()
            else:
                # Single-agent runs only execute matching commit backfill.
                commit_author_items = (
                    [(normalized_target, FIRST_COMMIT_AUTHORS[normalized_target])]
                    if normalized_target in FIRST_COMMIT_AUTHORS
                    else []
                )
            for agent_name, authors in commit_author_items:
                for author in authors:
                    all_results.extend(
                        self._discover_prs_from_commits(
                            author=author,
                            discovered_agent=agent_name,
                            start_iso=start_iso,
                            end_iso=end_iso,
                            seen_urls=seen_urls,
                            max_pages=max_pages,
                            start_page=commit_page,
                            checkpoint_target=normalized_target,
                            on_discovered=_log_discovery,
                            on_page_complete=lambda label: _log_summary(
                                label, "Commit page complete"
                            ),
                        )
                    )
        self.checkpoint.clear()
        if per_agent_counts:
            print(f"[discovery] Totals: {total_discovered} total.")
            for label, count in sorted(per_agent_counts.items()):
                print(f"[discovery] Total for {label}: {count}")
        return all_results
