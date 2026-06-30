"""Enrich discovered PR candidates into extraction output rows.

Enrichment takes lightweight `PullRequestTest` records from discovery and
hydrates them into the parquet-facing `PullRequest` DTO. It fetches PR core
fields, changed files, repository metadata, README text where available, and
first-commit agent signals.

This module normalizes the two supported API modes:
- GraphQL enrichment: fewer requests for core PR fields and file lists, but no
  patches/raw file URLs.
- REST enrichment: fallback mode with REST file payloads and some REST-only
  merge fields.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from extraction.config.agent_config import FIRST_COMMIT_AUTHORS
from extraction.config.settings import ExtractionSettings
from extraction.dtos.dtos import (
    FileChange,
    Label,
    LicenseInfo,
    PullRequest,
    PullRequestTest,
    Repository,
    RepositoryPeek,
    UserPeek,
)
from extraction.utility.language_labeller import infer_language
from extraction.utility.token_manager import TokenManager

GRAPHQL_URL = "https://api.github.com/graphql"


class SkipPRError(RuntimeError):
    """Raised when a PR should be skipped without failing the whole run."""


class EnrichmentScraper:
    """
    Enrich discovered PRs with GitHub metadata and file lists.

    The output schema intentionally excludes heavyweight timeline details such
    as full comments, reviews, commit lists, and file contents. Those fields stay
    present as empty/optional DTO slots for schema compatibility.
    """

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        base_url: str = "https://api.github.com",
        use_graphql: Optional[bool] = None,
        settings: Optional[ExtractionSettings] = None,
    ) -> None:
        """Initialize API clients, token rotation, and enrichment mode."""
        if settings is not None and tokens is None:
            tokens = list(settings.github_tokens)
        token_list = [token.strip() for token in (tokens or []) if token and token.strip()]
        if not token_list:
            raise RuntimeError("No GitHub tokens provided.")
        self.token_manager = TokenManager(token_list)
        self.base_url = base_url.rstrip("/")
        if use_graphql is None:
            use_graphql = (
                settings.use_graphql_enrichment if settings is not None else True
            )
        # GraphQL is the default because it fetches core PR fields and files in
        # one paginated query. REST remains available as a deterministic fallback.
        self.use_graphql = bool(use_graphql)
        self.session = requests.Session()
        self._repo_cache: Dict[str, dict] = {}

    def _headers(self) -> dict:
        """Build standard REST headers using the current token."""
        token = self.token_manager.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

    def _headers_graphql(self) -> dict:
        """Build headers for GraphQL requests using the current token."""
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

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """Issue a REST GET request with retries, token rotation, and rate-limit handling."""
        retries = 0
        backoff = 2
        while True:
            try:
                resp = self.session.get(url, headers=self._headers(), params=params, timeout=20)
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

    def _request_graphql(self, query: str, variables: dict) -> dict:
        """
        Issue a GraphQL request with retries and rate-limit handling.

        PR-specific GraphQL resource-limit errors are converted to `SkipPRError`
        so the manager can continue with the rest of the run.
        """
        retries = 0
        backoff = 2
        while True:
            try:
                resp = self.session.post(
                    GRAPHQL_URL,
                    headers=self._headers_graphql(),
                    json={"query": query, "variables": variables},
                    timeout=20,
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
                data = resp.json()
                if "errors" in data:
                    for err in data["errors"]:
                        msg = (err.get("message") or "").lower()
                        if (
                            err.get("type") == "RESOURCE_LIMITS_EXCEEDED"
                            or "resource limits" in msg
                        ):
                            raise SkipPRError("GraphQL resource limits exceeded for PR.")
                    raise RuntimeError(f"GraphQL errors: {data['errors']}")
                return data.get("data") or {}
            except requests.RequestException:
                retries += 1
                if retries > 5:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _paged_get(self, url: str, per_page: int = 100) -> List[dict]:
        """Return all pages from a REST list endpoint."""
        page = 1
        results: List[dict] = []
        while True:
            data = self._get(url, params={"per_page": per_page, "page": page})
            if not isinstance(data, list) or not data:
                break
            results.extend(data)
            if len(data) < per_page:
                break
            page += 1
        return results

    def _parse_repo_info(self, pr_url: str) -> Tuple[str, str, int]:
        """Extract owner, repo, and PR number from a canonical GitHub PR URL."""
        # Expecting URLs like https://github.com/{owner}/{repo}/pull/{number}
        parts = pr_url.rstrip("/").split("/")
        if len(parts) < 5:
            raise ValueError(f"Unexpected PR URL format: {pr_url}")
        owner, repo, number_str = parts[-4], parts[-3], parts[-1]
        return owner, repo, int(number_str)

    def _utcnow(self) -> datetime:
        """Return current UTC time."""
        return datetime.now(timezone.utc)

    def _fetch_repo(self, owner: str, repo: str) -> dict:
        """Fetch repository metadata from REST, cached per owner/repo."""
        if not owner or not repo:
            return {}
        cache_key = f"{owner}/{repo}"
        if cache_key in self._repo_cache:
            return self._repo_cache[cache_key]
        data = self._get(f"{self.base_url}/repos/{owner}/{repo}")
        if isinstance(data, dict):
            self._repo_cache[cache_key] = data
        return data

    def _fetch_pr_graphql(self, owner: str, repo: str, number: int) -> dict:
        """Fetch PR core data and all changed-file nodes using GraphQL pagination."""
        query = """
        query($owner: String!, $repo: String!, $number: Int!, $first: Int!, $after: String) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $number) {
              id
              number
              title
              url
              bodyText
              state
              createdAt
              updatedAt
              closedAt
              mergedAt
              isDraft
              locked
              authorAssociation
              mergeable
              mergeCommit { oid }
              additions
              deletions
              changedFiles
              comments { totalCount }
              commits { totalCount }
              baseRefName
              headRefName
              baseRefOid
              headRefOid
              author { login url __typename }
              mergedBy {
                __typename
                ... on User { login url }
                ... on Bot { login url }
                ... on Organization { login url }
                ... on Mannequin { login url }
              }
              baseRepository { id name url owner { login } }
              headRepository { id name url owner { login } }
              files(first: $first, after: $after) {
                pageInfo { hasNextPage endCursor }
                nodes {
                  path
                  additions
                  deletions
                  changeType
                }
              }
            }
          }
        }
        """
        files: List[dict] = []
        cursor = None
        pr_node: Optional[dict] = None
        while True:
            # Only the `files` connection is paginated here. The PR node itself
            # is copied from the first page and augmented with a flat files list.
            data = self._request_graphql(
                query,
                {
                    "owner": owner,
                    "repo": repo,
                    "number": number,
                    "first": 100,
                    "after": cursor,
                },
            )
            repo_node = data.get("repository") or {}
            pr = repo_node.get("pullRequest")
            if not pr:
                return {}
            if pr_node is None:
                pr_node = pr
            files_block = pr.get("files") or {}
            nodes = files_block.get("nodes") or []
            files.extend(nodes)
            page_info = files_block.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
        pr_node["files_nodes"] = files
        return pr_node

    def _fetch_first_commit(self, owner: str, repo: str, number: int) -> Optional[dict]:
        """Fetch the first commit entry for a PR, returning None on lookup failure."""
        try:
            data = self._get(
                f"{self.base_url}/repos/{owner}/{repo}/pulls/{number}/commits",
                params={"per_page": 1, "page": 1},
            )
        except Exception:
            return None
        if isinstance(data, list) and data:
            return data[0]
        return None

    def _build_repository(self, repo_data: dict, role: str, pr_id: str) -> Repository:
        """
        Build a Repository DTO from a REST repository payload.

        GitHub REST does not expose every field represented in the DTO. Missing
        fields are filled with conservative defaults or `None` to keep the
        parquet schema stable.
        """
        owner_data = repo_data.get("owner") or {}
        owner_login = owner_data.get("login", "")
        repo_name = repo_data.get("name", "")
        owner = UserPeek(
            id=str(owner_data.get("id")) if owner_data.get("id") is not None else None,
            login=owner_login,
            url=owner_data.get("html_url"),
            name=None,
            typename=owner_data.get("type"),
        )
        license_info = repo_data.get("license") or {}
        topics = repo_data.get("topics") or []
        primary_language = repo_data.get("language")
        watchers_val = repo_data.get("watchers_count")
        if watchers_val is None:
            watchers_val = repo_data.get("subscribers_count") or 0
        return Repository(
            id=str(repo_data.get("id")),
            pr_id=pr_id,
            role=role,
            name=repo_name or "",
            name_with_owner=repo_data.get("full_name") or "",
            url=repo_data.get("html_url") or "",
            ssh_url=repo_data.get("ssh_url") or "",
            stargazer_count=repo_data.get("stargazers_count") or 0,
            is_fork=bool(repo_data.get("fork")),
            is_archived=bool(repo_data.get("archived")),
            is_disabled=bool(repo_data.get("disabled")),
            is_empty=False,  # not provided; assume False
            is_in_organization=(owner_data.get("type") == "Organization"),
            is_locked=bool(
                repo_data.get("archived")
            ),  # proxy; GitHub REST doesn't expose lock flag directly
            is_private=bool(repo_data.get("private")),
            is_mirror=bool(repo_data.get("mirror_url")),
            is_template=bool(repo_data.get("is_template")),
            is_user_configuration_repository=False,
            fork_count=repo_data.get("forks_count") or 0,
            forking_allowed=bool(repo_data.get("allow_forking")),
            created_at=repo_data.get("created_at") or "",
            visibility=repo_data.get("visibility")
            or ("private" if repo_data.get("private") else "public"),
            owner=owner,
            topics_count=len(topics),
            languages=[],
            language_count=0,
            watchers=watchers_val,
            license_info=license_info.get("name"),
            default_branch=repo_data.get("default_branch"),
            license=(
                LicenseInfo(
                    key=license_info.get("key"),
                    name=license_info.get("name"),
                    spdx_id=license_info.get("spdx_id"),
                    url=license_info.get("url"),
                )
                if license_info
                else None
            ),
            size_kb=repo_data.get("size"),
            open_issues_count=repo_data.get("open_issues_count"),
            subscribers_count=repo_data.get("subscribers_count"),
            allow_merge_commit=repo_data.get("allow_merge_commit"),
            allow_squash_merge=repo_data.get("allow_squash_merge"),
            allow_rebase_merge=repo_data.get("allow_rebase_merge"),
            has_issues=repo_data.get("has_issues"),
            has_projects=repo_data.get("has_projects"),
            has_wiki=repo_data.get("has_wiki"),
            homepage_url=repo_data.get("homepage"),
            topics=topics,
            network_count=repo_data.get("network_count"),
            security_policy_url=repo_data.get("security_policy_url"),
            archived_reason=repo_data.get("archived_reason"),
            forks_count=repo_data.get("forks_count"),
            license_name=license_info.get("name"),
            domains=None,
            repository_topics=topics,
            primary_language=primary_language,
            lock_reason=None,
            pushed_at=repo_data.get("pushed_at"),
            updated_at=repo_data.get("updated_at"),
            archived_at=None,
            description=repo_data.get("description"),
            popularity_label=None,
            readme=None,
            readme_is_truncated=None,
        )

    def enrich_pr(self, pr: PullRequestTest, *, skip_first_commit: bool = False) -> PullRequest:
        """
        Populate a `PullRequest` DTO from one discovery candidate.

        `skip_first_commit` should be true for PRs already discovered through an
        explicit agent clause; this avoids one extra REST request per PR. Human
        candidates should not skip it because first-commit authorship is one of
        the post-enrichment exclusion signals.
        """
        owner, repo, number = self._parse_repo_info(pr.url)

        if self.use_graphql:
            # GraphQL provides core PR fields and changed files in one paginated
            # query, but omits REST-only file details such as patch/raw URLs.
            pr_data = self._fetch_pr_graphql(owner, repo, number)
            if not pr_data:
                raise SkipPRError("PR not found in GraphQL response.")
            files_data = pr_data.get("files_nodes") or []
            base_repo_owner = ((pr_data.get("baseRepository") or {}).get("owner") or {}).get(
                "login", ""
            )
            base_repo_name = (pr_data.get("baseRepository") or {}).get("name", "")
            head_repo_owner = ((pr_data.get("headRepository") or {}).get("owner") or {}).get(
                "login", ""
            )
            head_repo_name = (pr_data.get("headRepository") or {}).get("name", "")
            print(f"[enrich] PR {pr.url}: fetched core data (GraphQL)")
        else:
            # REST fallback fetches PR details and file list separately. This
            # mode supplies patch/raw file fields that GraphQL does not expose.
            pr_data = self._get(f"{self.base_url}/repos/{owner}/{repo}/pulls/{number}")
            files_data = self._paged_get(
                f"{self.base_url}/repos/{owner}/{repo}/pulls/{number}/files"
            )
            base_repo_owner = (
                (pr_data.get("base") or {}).get("repo", {}).get("owner", {}).get("login", "")
            )
            base_repo_name = (pr_data.get("base") or {}).get("repo", {}).get("name", "")
            head_repo_owner = (
                (pr_data.get("head") or {}).get("repo", {}).get("owner", {}).get("login", "")
            )
            head_repo_name = (pr_data.get("head") or {}).get("repo", {}).get("name", "")
            print(f"[enrich] PR {pr.url}: fetched core data (PR, files, repos)")

        base_repo_data = self._fetch_repo(base_repo_owner, base_repo_name)
        head_repo_data = self._fetch_repo(head_repo_owner, head_repo_name)

        # Keep lightweight repo peeks on the root PR even when full repository
        # metadata cannot be fetched.
        base_repo_peek = RepositoryPeek(
            id=str(
                (pr_data.get("baseRepository") or {}).get("id")
                if self.use_graphql
                else (pr_data.get("base") or {}).get("repo", {}).get("id")
            ),
            name=(
                (pr_data.get("baseRepository") or {}).get("name")
                if self.use_graphql
                else (pr_data.get("base") or {}).get("repo", {}).get("name")
            ),
            url=(
                (pr_data.get("baseRepository") or {}).get("url")
                if self.use_graphql
                else (pr_data.get("base") or {}).get("repo", {}).get("html_url")
            ),
        )
        head_repo_peek = RepositoryPeek(
            id=str(
                (pr_data.get("headRepository") or {}).get("id")
                if self.use_graphql
                else (pr_data.get("head") or {}).get("repo", {}).get("id")
            ),
            name=(
                (pr_data.get("headRepository") or {}).get("name")
                if self.use_graphql
                else (pr_data.get("head") or {}).get("repo", {}).get("name")
            ),
            url=(
                (pr_data.get("headRepository") or {}).get("url")
                if self.use_graphql
                else (pr_data.get("head") or {}).get("repo", {}).get("html_url")
            ),
        )
        base_repo_full = (
            self._build_repository(base_repo_data, "BASE", str(pr_data.get("id")))
            if base_repo_data
            else None
        )
        head_repo_full = (
            self._build_repository(head_repo_data, "HEAD", str(pr_data.get("id")))
            if head_repo_data
            else None
        )

        file_changes = []
        graphql_head_oid = pr_data.get("headRefOid") if self.use_graphql else None
        if files_data:
            for item in files_data:
                # Normalize REST and GraphQL file payloads into the same
                # FileChange DTO. Fields unavailable in the selected mode remain
                # None.
                path = item.get("filename", "")
                status = item.get("status", "")
                if self.use_graphql:
                    path = item.get("path", "")
                    status = (item.get("changeType") or "").lower()
                graph_ql_blob_url = None
                if self.use_graphql and graphql_head_oid and path:
                    graph_ql_blob_url = (
                        f"https://github.com/{owner}/{repo}/blob/{graphql_head_oid}/{path}"
                    )
                file_changes.append(
                    FileChange(
                        additions=item.get("additions", 0),
                        deletions=item.get("deletions", 0),
                        path=path,
                        change_type=status,
                        language=infer_language(path),
                        patch=item.get("patch") if not self.use_graphql else None,
                        sha=item.get("sha") if not self.use_graphql else None,
                        blob_url=(
                            item.get("blob_url")
                            if not self.use_graphql
                            else graph_ql_blob_url
                        ),
                        raw_url=item.get("raw_url") if not self.use_graphql else None,
                        contents_url=(
                            item.get("contents_url") if not self.use_graphql else None
                        ),
                        previous_filename=(
                            item.get("previous_filename")
                            if not self.use_graphql
                            else None
                        ),
                        status=status,
                        base_content=None,
                        head_content=None,
                        is_binary=None,
                        is_truncated=None,
                    )
                )
        file_languages = sorted({fc.language for fc in file_changes if fc.language})

        # Normalize REST `user` and GraphQL `author` actor payloads into the
        # lightweight UserPeek stored on the root PR.
        author_data = pr_data.get("user") or {}
        if self.use_graphql:
            author_data = pr_data.get("author") or {}
        author = UserPeek(
            id=str(author_data.get("id")) if author_data.get("id") is not None else None,
            login=author_data.get("login"),
            url=(
                author_data.get("html_url")
                if not self.use_graphql
                else author_data.get("url")
            ),
            name=None,
            typename=(
                author_data.get("type")
                if not self.use_graphql
                else author_data.get("__typename")
            ),
        )

        is_cross_repo = (
            (pr_data.get("baseRepository") or {}).get("id")
            != (pr_data.get("headRepository") or {}).get("id")
            if self.use_graphql
            else (pr_data.get("base") or {}).get("repo", {}).get("id")
            != (pr_data.get("head") or {}).get("repo", {}).get("id")
        )

        labels: List[Label] = []
        assignees: List[UserPeek] = []
        comments: List = []
        reviews: List = []
        commits: List = []
        first_commit_author_login = None
        first_commit_agent: Optional[str] = None
        if skip_first_commit:
            # Discovery already identified the agent, so reuse that label and
            # avoid an extra first-commit lookup.
            if pr.discovered_agent:
                first_commit_agent = pr.discovered_agent
        else:
            # Human and ambiguous agentic PRs need first-commit lookup because
            # it can reveal agent authorship absent from author/body/branch.
            first_commit_item = self._fetch_first_commit(owner, repo, number)
            if first_commit_item:
                commit_sha = first_commit_item.get("sha")
                commit_detail = (
                    self._get(f"{self.base_url}/repos/{owner}/{repo}/commits/{commit_sha}")
                    if commit_sha
                    else {}
                )
                first_commit_author_login = (
                    (commit_detail.get("author") or {}).get("login")
                    or (commit_detail.get("commit") or {}).get("author", {}).get("name")
                    or (first_commit_item.get("author") or {}).get("login")
                    or (first_commit_item.get("commit") or {}).get("author", {}).get("name")
                )
            if first_commit_author_login:
                normalized = first_commit_author_login.lower()
                for agent_name, authors in FIRST_COMMIT_AUTHORS.items():
                    if any(normalized == author.lower() for author in authors):
                        first_commit_agent = agent_name
                        break

        mergeable_state_val = pr_data.get("mergeable_state")
        mergeable_method_val = pr_data.get("merge_method")
        auto_merge_val = pr_data.get("auto_merge")
        merged_by_data = pr_data.get("merged_by") or {}
        requested_reviewers_data = pr_data.get("requested_reviewers", [])
        if self.use_graphql:
            # These REST-only fields are not part of the GraphQL PR query used
            # here, so keep them unset rather than mixing API semantics.
            mergeable_state_val = None
            mergeable_method_val = None
            auto_merge_val = None
            merged_by_data = pr_data.get("mergedBy") or {}
            requested_reviewers_data = []

        pull_request = PullRequest(
            id=str(pr_data.get("id")),
            title=pr_data.get("title") or "",
            url=(
                pr_data.get("html_url")
                if not self.use_graphql
                else (pr_data.get("url") or "")
            ),
            number=pr_data.get("number") or 0,
            body=(
                pr_data.get("body")
                if not self.use_graphql
                else (pr_data.get("bodyText") or "")
            ),
            state=pr_data.get("state") or "",
            created_at=(
                pr_data.get("created_at")
                if not self.use_graphql
                else (pr_data.get("createdAt") or "")
            ),
            is_draft=(
                bool(pr_data.get("draft", False))
                if not self.use_graphql
                else bool(pr_data.get("isDraft", False))
            ),
            changed_files=(
                pr_data.get("changed_files")
                if not self.use_graphql
                else pr_data.get("changedFiles") or len(file_changes)
            ),
            is_cross_repository=bool(is_cross_repo),
            locked=bool(pr_data.get("locked", False)),
            is_in_merge_queue=False,
            additions=pr_data.get("additions") or 0,
            deletions=pr_data.get("deletions") or 0,
            author=author,
            label_count=len(labels),
            base_repository=base_repo_peek,
            head_repository=head_repo_peek,
            timeline_count=0,
            merged_at=(
                pr_data.get("merged_at") if not self.use_graphql else pr_data.get("mergedAt")
            ),
            closed_at=(
                pr_data.get("closed_at") if not self.use_graphql else pr_data.get("closedAt")
            ),
            updated_at=(
                pr_data.get("updated_at")
                if not self.use_graphql
                else pr_data.get("updatedAt")
            ),
            comments_count=(
                pr_data.get("comments")
                if not self.use_graphql
                else (pr_data.get("comments") or {}).get("totalCount")
            ),
            commits_count=(
                pr_data.get("commits")
                if not self.use_graphql
                else (pr_data.get("commits") or {}).get("totalCount")
            ),
            files=file_changes,
            file_languages=file_languages,
            reviews_count=len(reviews),
            assignees_count=len(assignees),
            closing_issues_count=0,
            author_association=(
                pr_data.get("author_association")
                if not self.use_graphql
                else pr_data.get("authorAssociation")
            ),
            labels=labels,
            active_lock_reason=pr_data.get("active_lock_reason"),
            mergeable=pr_data.get("mergeable"),
            merge_commit_sha=(
                pr_data.get("merge_commit_sha")
                if not self.use_graphql
                else ((pr_data.get("mergeCommit") or {}).get("oid"))
            ),
            merged_by=(
                UserPeek(
                    id=(
                        str(merged_by_data.get("id"))
                        if merged_by_data.get("id") is not None
                        else None
                    ),
                    login=merged_by_data.get("login"),
                    url=(
                        merged_by_data.get("html_url")
                        if not self.use_graphql
                        else merged_by_data.get("url")
                    ),
                    name=None,
                    typename=None if not self.use_graphql else merged_by_data.get("__typename"),
                )
                if merged_by_data
                else None
            ),
            requested_reviewers=[
                UserPeek(
                    id=str(u.get("id")) if u.get("id") is not None else None,
                    login=u.get("login"),
                    url=u.get("html_url") if not self.use_graphql else u.get("url"),
                    name=None,
                    typename=None if not self.use_graphql else u.get("__typename"),
                )
                for u in requested_reviewers_data
            ],
            mergeable_state=mergeable_state_val,
            mergeable_method=mergeable_method_val,
            auto_merge=auto_merge_val,
            authored_by_agent=first_commit_agent is not None,
            author_agent=first_commit_agent,
            discovered_agent=pr.discovered_agent,
            base_commit_sha=(
                (pr_data.get("base") or {}).get("sha")
                if not self.use_graphql
                else pr_data.get("baseRefOid")
            ),
            head_commit_sha=(
                (pr_data.get("head") or {}).get("sha")
                if not self.use_graphql
                else pr_data.get("headRefOid")
            ),
            compare_url=(
                f"https://api.github.com/repos/{owner}/{repo}/compare/{(pr_data.get('base') or {}).get('sha')}...{(pr_data.get('head') or {}).get('sha')}"
                if not self.use_graphql
                else f"https://api.github.com/repos/{owner}/{repo}/compare/{pr_data.get('baseRefOid')}...{pr_data.get('headRefOid')}"
            ),
            scraped_at=self._utcnow().isoformat().replace("+00:00", "Z"),
            post_merge_file_snapshots=[],
            head_ref_name=(
                (pr_data.get("head") or {}).get("ref")
                if not self.use_graphql
                else pr_data.get("headRefName")
            ),
            head_ref_oid=(
                (pr_data.get("head") or {}).get("sha")
                if not self.use_graphql
                else pr_data.get("headRefOid")
            ),
            base_ref_name=(
                (pr_data.get("base") or {}).get("ref")
                if not self.use_graphql
                else pr_data.get("baseRefName")
            ),
            base_ref_oid=(
                (pr_data.get("base") or {}).get("sha")
                if not self.use_graphql
                else pr_data.get("baseRefOid")
            ),
            base_repository_full=base_repo_full,
            head_repository_full=head_repo_full,
            comments=comments,
            reviews=reviews,
            commits=commits,
        )

        # Timeline collections are intentionally empty in this extraction stage.
        # The fields remain on the DTO for schema compatibility with earlier
        # experiments and downstream code that expects them to exist.
        pull_request.timeline_items = []
        pull_request.assignees_count = len(assignees)

        return pull_request
