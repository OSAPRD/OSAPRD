"""GitHub token configuration for curation hydration.

Hydration can make many GitHub API calls while resolving commits and future
snapshots. A comma- or path-separator-delimited ``GITHUB_TOKENS`` value is used
first so the token manager can rotate across credentials. ``GITHUB_TOKEN`` is a
single-token fallback.
"""

from __future__ import annotations

import os

PRIMARY_TOKENS_ENV = "GITHUB_TOKENS"
FALLBACK_TOKEN_ENV = "GITHUB_TOKEN"


def split_github_tokens(value: str) -> list[str]:
    """Split a multi-token environment value into individual tokens."""
    tokens: list[str] = []
    for chunk in value.replace(os.pathsep, ",").split(","):
        token = chunk.strip()
        if token:
            tokens.append(token)
    return tokens


def load_github_tokens() -> list[str]:
    """Load GitHub tokens from environment variables only."""
    raw_tokens = os.environ.get(PRIMARY_TOKENS_ENV, "").strip()
    if raw_tokens:
        return split_github_tokens(raw_tokens)
    raw_token = os.environ.get(FALLBACK_TOKEN_ENV, "").strip()
    return [raw_token] if raw_token else []


TOKENS: list[str] = load_github_tokens()
