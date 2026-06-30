"""Hugging Face token loading for curation-data upload.

Tokens are read from environment variables only. Set ``HF_TOKEN`` for new runs;
the two Hugging Face Hub aliases are accepted for compatibility.
"""

from __future__ import annotations

import os


TOKEN_ENV_NAMES = (
    # HF_TOKEN is the current Hugging Face Hub convention. The aliases are kept
    # so older shells and automation snippets do not need source edits.
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def load_huggingface_token() -> str:
    """Return the first configured Hugging Face token, or an empty string."""
    for name in TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return ""


def redacted_token_state(token: str) -> dict[str, object]:
    """Return token metadata that is safe to persist in run manifests."""
    return {
        "configured": bool((token or "").strip()),
        "source_env_order": TOKEN_ENV_NAMES,
    }
