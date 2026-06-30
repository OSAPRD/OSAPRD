"""Shared helpers for topic-classifier training data extraction.

The extraction stage writes durable JSON/JSONL artifacts and manifests. These
helpers keep path setup, token loading, run identifiers, and repository keys
consistent across the GitHub client and extraction pipeline.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


EXTRACT_TRAINING_DATA_DIR = Path(__file__).resolve().parent
TOPIC_CLASSIFICATION_DIR = EXTRACT_TRAINING_DATA_DIR.parent
POST_PROCESSING_DIR = TOPIC_CLASSIFICATION_DIR.parent
REPO_ROOT = POST_PROCESSING_DIR.parent
UTILITY_DIR = POST_PROCESSING_DIR / "utility"
CLASSIFY_TOPICS_DIR = TOPIC_CLASSIFICATION_DIR / "classify-topics"
DEFAULT_OUTPUT_DIR = EXTRACT_TRAINING_DATA_DIR / "output"
DEFAULT_TOKENS_CONFIG_PATH = POST_PROCESSING_DIR / "config" / "tokens_config.py"
DEFAULT_TRAINING_CSV = (
    TOPIC_CLASSIFICATION_DIR / "training-data" / "topics220_repos152k_train.csv"
)

for path in (REPO_ROOT, UTILITY_DIR, CLASSIFY_TOPICS_DIR, EXTRACT_TRAINING_DATA_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from config_values import bool_config, github_tokens_from_sources, positive_int  # noqa: E402


def log(message: str) -> None:
    """Print a flushed extraction log line with a stable prefix."""
    print(f"[post-processing/topic-training-extract] {message}", flush=True)


def utc_now_z() -> str:
    """Return the current UTC time in manifest-friendly ISO format."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_timestamp_run_id(prefix: str) -> str:
    """Build a timestamped run id used for extraction output folders."""
    return f"{prefix.strip('_')}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def json_default(value: Any) -> Any:
    """JSON fallback serializer for dataclasses, paths, and DTO-like objects."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so interrupted runs do not leave partial manifests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
        newline="\n",
    )
    with temp_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n")
    temp_path.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON object to a newline-delimited artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default))
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield valid JSON objects from a JSONL artifact, skipping malformed lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                log(f"Skipping malformed JSONL line in {path}.")
                continue
            if isinstance(payload, dict):
                yield payload


def normalize_repository_key(owner: Any, repo: Any) -> str:
    """Normalize owner/name into the lowercase repository key used in joins."""
    owner_text = str(owner or "").strip().lower()
    repo_text = str(repo or "").strip().lower()
    return f"{owner_text}/{repo_text}" if owner_text and repo_text else ""


def safe_path_part(value: Any) -> str:
    """Return a filesystem-safe path segment for repository-owned artifacts."""
    text = str(value or "").strip() or "unknown"
    text = text.replace("\\", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def repository_identity_key(repository_id: Any, owner: Any, repo: Any) -> str:
    """Prefer GitHub numeric ids while retaining an owner/name fallback key."""
    repository_id_text = str(repository_id or "").strip()
    if repository_id_text and repository_id_text.lower() not in {"none", "nan", "null"}:
        return f"repository-id:{repository_id_text}"
    return f"repository-key:{normalize_repository_key(owner, repo)}"


def load_post_processing_github_tokens(
    token_config_path: Path = DEFAULT_TOKENS_CONFIG_PATH,
) -> tuple[str, ...]:
    """Load only post-processing GitHub tokens, without extraction fallback tokens."""
    return github_tokens_from_sources(
        (),
        Path(token_config_path),
        fallback_tokens=(),
    )


def env_path(name: str, default: Path) -> Path:
    """Read a path-valued environment variable with expansion."""
    return Path(os.environ.get(name, str(default))).expanduser()


def env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable."""
    return positive_int(os.environ.get(name), default=default)


def env_nonnegative_int(name: str, default: int) -> int:
    """Read a non-negative integer environment variable."""
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(0, int(default))


def env_float(name: str, default: float) -> float:
    """Read a float environment variable with a safe default."""
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable using the shared truthy set."""
    if name not in os.environ:
        return bool(default)
    return bool_config(os.environ.get(name))


def env_text(name: str, default: str) -> str:
    """Read a stripped text environment variable."""
    return str(os.environ.get(name, default)).strip()
