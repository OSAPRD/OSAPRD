"""Hugging Face token loading for extraction-data upload.

Tokens are never hardcoded. Set one of the supported environment variables
before running an upload command:

``HF_TOKEN``, ``HUGGINGFACE_HUB_TOKEN``, or ``HUGGING_FACE_HUB_TOKEN``.
"""

from __future__ import annotations

import os


TOKEN_ENV_NAMES = (
    # HF_TOKEN is the current Hugging Face Hub convention and should be used for
    # new runs. The two aliases are accepted for compatibility with older shells
    # and automation snippets.
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def load_huggingface_token() -> str:
    """Return the first non-empty Hugging Face token from the environment."""
    for name in TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return ""


def redacted_token_state(token: str) -> dict[str, object]:
    """Return token presence metadata without writing the token itself."""
    return {
        "configured": bool((token or "").strip()),
        "source_env_order": TOKEN_ENV_NAMES,
    }
