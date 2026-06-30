"""Agent-target search clause and local classification helpers.

`AgentFilter` serves two related purposes:

1. Build GitHub search clauses for discovery.
2. Re-identify an agent after enrichment from local PR fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from extraction.config.agent_config import AGENT_RULES


class AgentFilter:
    """Stores agent rules and applies local agent classification."""

    def __init__(
        self,
        filter_rules: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Initialize filter rules used by discovery and local classification."""
        self.filter_rules = filter_rules or dict(AGENT_RULES)

    def identify_agent(self, pr_obj: Any) -> Optional[str]:
        """Return the first matching agent name for an enriched PR object."""
        author_login = ""
        head_ref = ""
        body_text = ""
        comment_texts: List[str] = []
        commit_texts: List[str] = []
        if pr_obj is not None:
            # Use getattr throughout because some callers pass lightweight
            # stand-ins rather than concrete DTO instances.
            author_login = getattr(getattr(pr_obj, "author", None), "login", "") or ""
            head_ref = (getattr(pr_obj, "head_ref_name", "") or "").lower()
            body_text = (getattr(pr_obj, "body", "") or "").lower()
            comments = getattr(pr_obj, "comments", None) or []
            for comment in comments:
                text = getattr(comment, "body", "") or ""
                if text:
                    comment_texts.append(text.lower())
            commits = getattr(pr_obj, "commits", None) or []
            for commit in commits:
                msg = getattr(commit, "message", "") or ""
                headline = getattr(commit, "message_headline", "") or ""
                body = getattr(commit, "message_body", "") or ""
                combined = "\n".join([m for m in (headline, body, msg) if m])
                if combined:
                    commit_texts.append(combined.lower())

        if getattr(pr_obj, "authored_by_agent", False):
            author_agent = getattr(pr_obj, "author_agent", "") or ""
            # First-commit enrichment stores the OpenHands login rather than the
            # canonical output group name.
            if author_agent == "openhands-agent":
                return "openhands"
            if author_agent in self.filter_rules:
                return author_agent

        for agent_name, clause_list in self.filter_rules.items():
            for clause in clause_list:
                # These branches mirror the supported AGENT_RULES search syntax.
                if clause.startswith("author:"):
                    target = clause.split("author:", 1)[1].strip().strip('"').lower()
                    if author_login.lower() == target:
                        return agent_name
                elif clause.startswith("head:"):
                    prefix = clause.split("head:", 1)[1].strip().rstrip("/").lower()
                    if prefix and head_ref.startswith(prefix):
                        return agent_name
                elif "in:body" in clause:
                    marker = clause.replace("in:body", "").strip().strip('"').lower()
                    if marker and marker in body_text:
                        return agent_name
                elif "in:comments" in clause:
                    marker = clause.replace("in:comments", "").strip().strip('"').lower()
                    if marker and any(marker in text for text in comment_texts):
                        return agent_name
                elif "in:commits" in clause:
                    marker = clause.replace("in:commits", "").strip().strip('"').lower()
                    if marker and any(marker in text for text in commit_texts):
                        return agent_name

        return None
