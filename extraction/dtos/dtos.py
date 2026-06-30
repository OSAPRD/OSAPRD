"""Data transfer objects for extraction discovery, enrichment, and storage.

These dataclasses are the extraction stage's schema boundary. Discovery returns
`PullRequestTest` candidates, enrichment expands them into `PullRequest`, and
`StorageHandler` serializes the resulting object graph to parquet-compatible
dictionaries.

Adding, renaming, or changing a field here changes the extraction parquet row
contract and should be paired with a `SCHEMA_VERSION` review in
`extraction.config.settings`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Union


class Serializable:
    """Mixin for dataclasses that can serialize to plain dictionaries."""

    def to_dict(self) -> dict:
        """Return a recursive dataclass-to-dict conversion for parquet export."""
        return asdict(self)

@dataclass
class OrganizationSummary(Serializable):
    """Minimal organization identity embedded in user or enterprise payloads."""

    id: str
    login: str
    url: str
    name: Optional[str] = None


@dataclass
class Domain(Serializable):
    """Verified domain entry associated with an organization."""

    id: str
    domain: str


@dataclass
class UserPeek(Serializable):
    """Minimal actor identity used throughout PR, issue, and repo payloads."""

    id: Optional[str] = None
    login: Optional[str] = None
    url: Optional[str] = None
    name: Optional[str] = None
    typename: Optional[str] = None


@dataclass
class User(Serializable):
    """Expanded GitHub user profile with activity and account metadata."""

    id: str
    login: str
    name: str
    email: str
    url: str
    typename: str
    created_at: str
    is_employee: bool
    is_hireable: bool
    followers: int
    following: int
    organization_count: int
    pull_requests: int
    repositories: int
    repositories_contributed_to: int
    commit_comments: int
    issues: int
    sponsors: int
    sponsoring: int
    watching: int
    bio: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    updated_at: Optional[str] = None
    organizations: Optional[List[OrganizationSummary]] = None
    avatar_url: Optional[str] = None
    website_url: Optional[str] = None
    twitter_username: Optional[str] = None
    bio_html: Optional[str] = None
    contributions: Optional[dict] = None


@dataclass
class Bot(Serializable):
    """GitHub bot account identity."""

    id: str
    login: str
    url: str
    typename: str
    created_at: str
    updated_at: Optional[str] = None


@dataclass
class Mannequin(Serializable):
    """Placeholder identity returned by GitHub for transferred/deleted users."""

    id: str
    login: str
    url: str
    typename: str
    name: str
    email: str
    created_at: str
    updated_at: Optional[str] = None
    claimant: Optional[UserPeek] = None


@dataclass
class Organization(Serializable):
    """Expanded organization profile with counts, domains, and contact fields."""

    id: str
    login: str
    url: str
    typename: str
    name: str
    email: str
    created_at: str
    is_verified: bool
    mannequins: int
    domain_count: int
    enterprise_owners_count: int
    repositories: int
    sponsors: int
    sponsoring: int
    teams: int
    updated_at: Optional[str] = None
    archived_at: Optional[str] = None
    description: Optional[str] = None
    enterprise_owners: Optional[List[UserPeek]] = None
    domains: Optional[List[Domain]] = None
    location: Optional[str] = None
    avatar_url: Optional[str] = None
    website_url: Optional[str] = None
    twitter_username: Optional[str] = None
    billing_email: Optional[str] = None


@dataclass
class Enterprise(Serializable):
    """GitHub Enterprise account summary embedded in enterprise user records."""

    id: str
    name: str
    members: int
    organizations: int
    description: Optional[str] = None
    location: Optional[str] = None


@dataclass
class EnterpriseUserAccount(Serializable):
    """Enterprise-managed user account and related organizations."""

    id: str
    login: str
    url: str
    typename: str
    name: str
    created_at: str
    organization_count: int
    enterprise: Optional[Enterprise] = None
    updated_at: Optional[str] = None
    organizations: Optional[List[OrganizationSummary]] = None


@dataclass
class Label(Serializable):
    """Issue or pull-request label definition."""

    name: str
    description: Optional[str] = None


@dataclass
class LicenseInfo(Serializable):
    """Repository license metadata normalized across REST/GraphQL responses."""

    key: Optional[str] = None
    name: Optional[str] = None
    spdx_id: Optional[str] = None
    url: Optional[str] = None


@dataclass
class IssueType(Serializable):
    """GitHub issue type classification."""

    name: str
    description: Optional[str] = None


@dataclass
class Issue(Serializable):
    """Issue associated with a PR, including counts and participants."""

    id: str
    pr_id: str
    url: str
    title: str
    body: str
    created_at: str
    locked: bool
    number: int
    state: str
    tracked_issues_count: int
    label_count: int
    last_edited_at: Optional[str] = None
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    issue_type: Optional[IssueType] = None
    labels: Optional[List[Label]] = None
    state_reason: Optional[str] = None
    author: Optional[UserPeek] = None
    pr_ids: Optional[List[str]] = None
    prs_closing_issue: Optional[int] = None
    assignees: Optional[List[UserPeek]] = None
    closed_by: Optional[UserPeek] = None
    reactions_count: Optional[int] = None
    comments_count: Optional[int] = None


@dataclass
class Comment(Serializable):
    """Pull-request comment with author, timestamps, and reactions."""

    id: str
    pr_id: str
    url: str
    body: str
    created_at: str
    is_minimized: bool
    minimized_reason: Optional[str] = None
    last_edited_at: Optional[str] = None
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    author: Optional[UserPeek] = None
    reactions_count: Optional[int] = None
    author_association: Optional[str] = None


@dataclass
class Review(Serializable):
    """Pull-request review event and reviewer metadata."""

    id: str
    pr_id: str
    url: str
    body: str
    created_at: str
    is_minimized: bool
    state: str
    updated_at: Optional[str] = None
    last_edited_at: Optional[str] = None
    published_at: Optional[str] = None
    submitted_at: Optional[str] = None
    minimized_reason: Optional[str] = None
    author: Optional[UserPeek] = None
    reactions_count: Optional[int] = None
    author_association: Optional[str] = None


@dataclass
class Committer(Serializable):
    """Commit author or committer identity as recorded in Git."""

    name: str
    email: str


@dataclass
class Commit(Serializable):
    """Commit metadata within a PR timeline."""

    id: str
    sha: str
    pr_id: str
    url: str
    committed_date: str
    additions: int
    deletions: int
    authored_date: str
    message_body: str
    message_headline: str
    author_count: int
    committer: Optional[Committer] = None
    changed_files: Optional[int] = None
    authors: Optional[List[Committer]] = None
    parents: Optional[List[str]] = None
    verification: Optional[dict] = None
    message: Optional[str] = None
    authored_by_agent: Optional[bool] = None
    author_agent: Optional[str] = None


@dataclass
class FileChange(Serializable):
    """Per-file diff metadata for a pull request.

    GraphQL enrichment supplies paths, change type, and additions/deletions.
    REST-only fields such as `patch`, `raw_url`, and `contents_url` can be
    absent when GraphQL enrichment is enabled.
    """

    additions: int
    deletions: int
    path: str
    change_type: str
    language: Optional[str] = None
    patch: Optional[str] = None
    status: Optional[str] = None
    sha: Optional[str] = None
    blob_url: Optional[str] = None
    raw_url: Optional[str] = None
    contents_url: Optional[str] = None
    previous_filename: Optional[str] = None
    is_binary: Optional[bool] = None
    is_truncated: Optional[bool] = None
    base_content: Optional[str] = None
    head_content: Optional[str] = None


@dataclass
class RepositoryPeek(Serializable):
    """Minimal repository identity embedded in the root PR payload."""

    id: str
    name: str
    url: str


@dataclass
class PullRequest(Serializable):
    """Fully enriched pull request row written to parquet."""

    id: str
    title: str
    url: str
    number: int
    body: str
    state: str
    created_at: str
    is_draft: bool
    changed_files: int
    is_cross_repository: bool
    locked: bool
    is_in_merge_queue: bool
    additions: int
    deletions: int
    author: Union[User, Bot, Mannequin, Organization, EnterpriseUserAccount, UserPeek, None]
    label_count: int
    base_repository: RepositoryPeek
    head_repository: RepositoryPeek
    timeline_count: int
    merged_at: Optional[str] = None
    closed_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_edited_at: Optional[str] = None
    published_at: Optional[str] = None
    review_decision: Optional[str] = None
    head_ref_name: Optional[str] = None
    head_ref_oid: Optional[str] = None
    timeline_items: Optional[List[str]] = None
    base_ref_name: Optional[str] = None
    base_ref_oid: Optional[str] = None
    comments_count: Optional[int] = None
    reviews_count: Optional[int] = None
    commits_count: Optional[int] = None
    files: Optional[List[FileChange]] = None
    file_languages: Optional[List[str]] = None
    assignees_count: Optional[int] = None
    closing_issues_count: Optional[int] = None
    author_association: Optional[str] = None
    labels: Optional[List[Label]] = None
    active_lock_reason: Optional[str] = None
    mergeable: Optional[str] = None
    merge_commit_sha: Optional[str] = None
    merged_by: Optional[UserPeek] = None
    requested_reviewers: Optional[List[UserPeek]] = None
    mergeable_state: Optional[str] = None
    mergeable_method: Optional[str] = None
    auto_merge: Optional[bool] = None
    authored_by_agent: Optional[bool] = None
    author_agent: Optional[str] = None
    discovered_agent: Optional[str] = None
    base_commit_sha: Optional[str] = None
    head_commit_sha: Optional[str] = None
    compare_url: Optional[str] = None
    scraped_at: Optional[str] = None

    # Optional nested payloads. They may be empty for runs that only collect the
    # current extraction schema's required PR/file/repository metadata.
    post_merge_file_snapshots: Optional[List["FileSnapshot"]] = None
    base_repository_full: Optional["Repository"] = None
    head_repository_full: Optional["Repository"] = None
    comments: Optional[List[Comment]] = None
    reviews: Optional[List[Review]] = None
    commits: Optional[List[Commit]] = None


@dataclass
class Repository(Serializable):
    """Repository snapshot for either the base or head side of a PR.

    `role` is expected to be `base` or `head`. Optional text-heavy fields such
    as README remain available in the schema but are not populated by the
    current extraction stage.
    """

    id: str
    pr_id: str
    role: str  # base or head
    name: str
    name_with_owner: str
    url: str
    ssh_url: str
    stargazer_count: int
    is_fork: bool
    is_archived: bool
    is_disabled: bool
    is_empty: bool
    is_in_organization: bool
    is_locked: bool
    is_private: bool
    is_mirror: bool
    is_template: bool
    is_user_configuration_repository: bool
    fork_count: int
    forking_allowed: bool
    created_at: str
    visibility: str
    owner: UserPeek
    topics_count: int
    languages: List[str]
    language_count: int
    watchers: int
    license_info: Optional[str] = None
    default_branch: Optional[str] = None
    license: Optional[LicenseInfo] = None
    size_kb: Optional[int] = None
    open_issues_count: Optional[int] = None
    subscribers_count: Optional[int] = None
    allow_merge_commit: Optional[bool] = None
    allow_squash_merge: Optional[bool] = None
    allow_rebase_merge: Optional[bool] = None
    has_issues: Optional[bool] = None
    has_projects: Optional[bool] = None
    has_wiki: Optional[bool] = None
    homepage_url: Optional[str] = None
    topics: Optional[List[str]] = None
    network_count: Optional[int] = None
    security_policy_url: Optional[str] = None
    archived_reason: Optional[str] = None
    forks_count: Optional[int] = None
    license_name: Optional[str] = None
    domains: Optional[List[Domain]] = None
    repository_topics: Optional[List[str]] = None
    primary_language: Optional[str] = None
    lock_reason: Optional[str] = None
    pushed_at: Optional[str] = None
    updated_at: Optional[str] = None
    archived_at: Optional[str] = None
    description: Optional[str] = None
    popularity_label: Optional[str] = None
    readme: Optional[str] = None
    readme_is_truncated: Optional[bool] = None


@dataclass
class PullRequestTest(Serializable):
    """Discovery-stage pull request candidate before full enrichment.

    This DTO is intentionally smaller than `PullRequest` so discovery can page
    quickly through GitHub search results. `database_id` and `url` are used for
    duplicate detection before enrichment.
    """

    id: str
    title: str
    author: UserPeek
    url: str
    body: str
    created_at: str
    is_draft: bool
    additions: int
    deletions: int
    changed_files: int
    commits: int
    comments: int
    reviews: int
    merged_at: Optional[str] = None
    closed_at: Optional[str] = None
    head_ref_name: Optional[str] = None
    sampled_hour: Optional[str] = None
    authored_by_agent: Optional[bool] = None
    author_agent: Optional[str] = None
    discovered_agent: Optional[str] = None
    database_id: Optional[str] = None


@dataclass
class FileSnapshot(Serializable):
    """Stored file state at a specific time offset after merge.

    This is optional longitudinal context. The current extraction stage does not
    fetch post-merge snapshots by default, but the type remains part of the
    schema for downstream reproducibility.
    """

    path: str
    sha: Optional[str] = None
    status: Optional[str] = None
    additions: Optional[int] = None
    deletions: Optional[int] = None
    patch: Optional[str] = None
    content: Optional[str] = None
    blob_url: Optional[str] = None
    raw_url: Optional[str] = None
    contents_url: Optional[str] = None
    previous_filename: Optional[str] = None
    is_binary: Optional[bool] = None
    is_truncated: Optional[bool] = None
    captured_at: Optional[str] = None
    offset_days_from_merge: Optional[int] = None
