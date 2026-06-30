"""
GitHub token configuration for live extraction.
"""

from __future__ import annotations

import os

PRIMARY_TOKENS_ENV = "GITHUB_TOKENS"
FALLBACK_TOKEN_ENV = "GITHUB_TOKEN"


def split_github_tokens(value: str) -> list[str]:
    """
    Split a multi-token environment value into individual tokens.
    """
    tokens: list[str] = []
    for chunk in value.replace(os.pathsep, ",").split(","):
        token = chunk.strip()
        if token:
            tokens.append(token)
    return tokens


def load_github_tokens() -> list[str]:
    """
    Load GitHub tokens from environment variables only.

    `GITHUB_TOKENS` takes precedence over `GITHUB_TOKEN` so a run
    can rotate across multiple tokens when both are present.
    """
    raw_tokens = os.environ.get(PRIMARY_TOKENS_ENV, "").strip()
    if raw_tokens:
        return split_github_tokens(raw_tokens)
    raw_token = os.environ.get(FALLBACK_TOKEN_ENV, "").strip()
    return [raw_token] if raw_token else []
