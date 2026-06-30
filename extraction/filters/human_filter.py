"""Post-enrichment filters for human-target PRs.

The human target is sampled before enrichment, then checked again after
enrichment because additional evidence becomes available: normalized actor type,
file languages, first-commit agent flags, and enriched PR body/head metadata.

This module is intentionally conservative. If a PR matches known agent signals
or a bot identity, it is removed from the human sample.
"""

from __future__ import annotations

import re
from typing import Optional

from extraction.config.agent_config import AGENT_RULES
from extraction.config.human_config import ALLOWED_PR_LANGUAGES, EXTRA_EXCLUDE_CLAUSES
from extraction.dtos.dtos import PullRequest


class HumanFilter:
    """Post-enrichment checks for human-sampled pull requests."""

    def __init__(self) -> None:
        """Load configured agent rules, extra exclusions, and language targets."""
        self.agent_rules = AGENT_RULES
        self.extra_excludes = EXTRA_EXCLUDE_CLAUSES
        # Preserve config order for deterministic per-language quota assignment.
        self.allowed_language_order = [lang.lower() for lang in ALLOWED_PR_LANGUAGES if lang]
        self.allowed_languages = set(self.allowed_language_order)

    def _body_markers_from_clause(self, clause: str) -> list[str]:
        """Extract simple text markers from a GitHub search fragment."""
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
        """Return True when an enriched PR matches a supported search fragment."""
        normalized = clause.strip().strip("()")
        if normalized.startswith("author:"):
            target = normalized.split("author:", 1)[1].strip().strip('"').lower()
            return bool(target) and author_login.lower() == target
        if normalized.startswith("head:"):
            prefix = normalized.split("head:", 1)[1].strip().rstrip("/").lower()
            return bool(prefix) and head_ref.startswith(prefix)
        return any(marker in body_lower for marker in self._body_markers_from_clause(clause))

    def is_language_allowed(self, pr_obj: PullRequest) -> bool:
        """Return True if the enriched PR contains at least one allowed language."""
        if not self.allowed_languages:
            return True
        languages = pr_obj.file_languages or []
        normalized = {lang.lower() for lang in languages if lang}
        if not normalized:
            return False
        return any(lang in normalized for lang in self.allowed_languages)

    def select_primary_language(self, pr_obj: PullRequest) -> Optional[str]:
        """Select the language bucket used for human balancing."""
        if not self.allowed_language_order:
            return None
        languages = pr_obj.file_languages or []
        normalized = {lang.lower() for lang in languages if lang}
        for lang in self.allowed_language_order:
            if lang in normalized:
                return lang
        return None

    def is_agentic(self, pr_obj: PullRequest) -> bool:
        """Return True if the PR matches an agent signal or bot identity."""
        # First-commit enrichment can identify agent authorship even when the PR
        # author, branch, and body did not match discovery-time clauses.
        if pr_obj.authored_by_agent:
            return True

        author_login = (pr_obj.author.login if pr_obj.author else "") or ""
        author_type = (pr_obj.author.typename if pr_obj.author else "") or ""
        head_ref = (pr_obj.head_ref_name or "").lower()
        body_lower = (pr_obj.body or "").lower()

        # Treat all GitHub bot identities as non-human, even when they are not
        # listed in AGENT_RULES. This keeps the human target strictly human.
        if author_login.lower().endswith("[bot]") or author_type.lower() == "bot":
            return True

        # Apply the configured first-class agent rules against enriched fields.
        for clause_list in self.agent_rules.values():
            for clause in clause_list:
                if self._matches_search_fragment(
                    clause,
                    author_login=author_login,
                    head_ref=head_ref,
                    body_lower=body_lower,
                ):
                    return True

        # Extra exclusions are configured as GitHub search fragments. At this
        # post-enrichment stage, the supported subset is author/head/body-like
        # matching. More complex GitHub search syntax should be converted to one
        # of those forms before being added to human_config.
        for excl in self.extra_excludes:
            if self._matches_search_fragment(
                excl,
                author_login=author_login,
                head_ref=head_ref,
                body_lower=body_lower,
            ):
                return True

        return False
