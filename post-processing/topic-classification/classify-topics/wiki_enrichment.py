"""Optional GitHub wiki text enrichment with cache-first behavior.

GitHub exposes repository wikis as separate git repositories. This helper first
checks a local cache and then, when enabled, attempts a shallow wiki clone with
token rotation.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

UTILITY_DIR = Path(__file__).resolve().parents[2] / "utility"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))

from config_values import resolve_github_tokens


TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".mdown",
    ".mkd",
    ".rst",
    ".txt",
    ".wiki",
    ".mediawiki",
    ".textile",
    ".creole",
}


@dataclass(frozen=True)
class WikiFetchResult:
    """Wiki lookup status plus text/cache metadata for manifests."""

    status: str
    text: str = ""
    cache_path: str | None = None
    metadata_path: str | None = None
    notes: str | None = None


class _WikiTokenRotator:
    """Small local token rotator for live wiki git clone attempts."""

    def __init__(self, tokens: Iterable[str]) -> None:
        self.tokens = tuple(str(token).strip() for token in tokens if str(token).strip())
        if not self.tokens:
            raise ValueError("At least one token is required.")
        self.current_index = 0
        self.invalid_indexes: set[int] = set()

    def get_token(self) -> str:
        for _ in self.tokens:
            if self.current_index not in self.invalid_indexes:
                return self.tokens[self.current_index]
            self.rotate_token()
        raise RuntimeError("All tokens are invalid.")

    def rotate_token(self) -> None:
        if len(self.invalid_indexes) >= len(self.tokens):
            raise RuntimeError("All tokens are invalid.")
        for _ in self.tokens:
            self.current_index = (self.current_index + 1) % len(self.tokens)
            if self.current_index not in self.invalid_indexes:
                return
        raise RuntimeError("All tokens are invalid.")

    def invalidate_current(self) -> None:
        self.invalid_indexes.add(self.current_index)
        self.rotate_token()


class TopicWikiEnricher:
    """Cache-first wiki text provider for topic classification."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        enable_live_fetch: bool = False,
        tokens: Iterable[str] | None = None,
        log: Callable[[str], None] | None = None,
        git_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.enable_live_fetch = bool(enable_live_fetch)
        self.tokens = tuple(str(token).strip() for token in (tokens or ()) if str(token).strip())
        self.log = log
        self.git_runner = git_runner or subprocess.run
        self._token_manager: Any | None = None
        self.stats = {
            "cache_hit": 0,
            "fetched": 0,
            "missing": 0,
            "not_requested": 0,
            "auth_failed": 0,
            "rate_limited": 0,
            "fetch_failed": 0,
        }

    def get_wiki_text(self, owner: str, repo: str) -> WikiFetchResult:
        """Return wiki text from cache or optional live git clone."""
        cache_text_path = self._cache_text_path(owner, repo)
        cache_metadata_path = self._cache_metadata_path(owner, repo)
        if cache_text_path.exists():
            text = cache_text_path.read_text(encoding="utf-8", errors="replace")
            self.stats["cache_hit"] += 1
            return WikiFetchResult(
                status="cache_hit",
                text=text,
                cache_path=str(cache_text_path),
                metadata_path=str(cache_metadata_path),
            )
        if not self.enable_live_fetch:
            self.stats["not_requested"] += 1
            return WikiFetchResult(
                status="not_requested",
                cache_path=str(cache_text_path),
                metadata_path=str(cache_metadata_path),
            )

        tokens = self._resolve_tokens()
        attempt_count = (len(tokens) if tokens else 0) + 1
        last_status = "fetch_failed"
        last_notes: str | None = None
        for attempt in range(1, attempt_count + 1):
            token = self._current_token(tokens)
            url = self._wiki_clone_url(owner, repo, token=token)
            with tempfile.TemporaryDirectory(prefix="topic-wiki-") as temp_dir:
                clone_dir = Path(temp_dir) / "wiki"
                result = self._run_git_clone(url, clone_dir)
                if result.returncode == 0:
                    text = _extract_text_from_wiki_clone(clone_dir)
                    status = "fetched" if text.strip() else "missing"
                    self._write_cache(
                        cache_text_path,
                        cache_metadata_path,
                        text=text,
                        status=status,
                        notes=None,
                    )
                    self.stats[status] += 1
                    return WikiFetchResult(
                        status=status,
                        text=text,
                        cache_path=str(cache_text_path),
                        metadata_path=str(cache_metadata_path),
                    )

                message = _combined_process_output(result).lower()
                last_notes = _redact_token(_combined_process_output(result))
                if _looks_missing(message):
                    last_status = "missing"
                    self._write_cache(
                        cache_text_path,
                        cache_metadata_path,
                        text="",
                        status=last_status,
                        notes=last_notes,
                    )
                    self.stats[last_status] += 1
                    return WikiFetchResult(
                        status=last_status,
                        cache_path=str(cache_text_path),
                        metadata_path=str(cache_metadata_path),
                        notes=last_notes,
                    )
                if _looks_auth_failure(message) and token is not None:
                    last_status = "auth_failed"
                    self._invalidate_current_token()
                    continue
                if _looks_rate_limited(message) and token is not None:
                    last_status = "rate_limited"
                    self._rotate_token()
                    continue
                if token is not None:
                    self._rotate_token()
                    continue
                break

        self.stats[last_status] = self.stats.get(last_status, 0) + 1
        self._emit(
            f"Wiki fetch failed for {owner}/{repo} after {attempt_count} attempts "
            f"(status={last_status})."
        )
        return WikiFetchResult(
            status=last_status,
            cache_path=str(cache_text_path),
            metadata_path=str(cache_metadata_path),
            notes=last_notes,
        )

    def _run_git_clone(self, url: str, clone_dir: Path) -> Any:
        return self.git_runner(
            ["git", "clone", "--depth", "1", url, str(clone_dir)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _resolve_tokens(self) -> tuple[str, ...]:
        return resolve_github_tokens(
            self.tokens,
            Path(__file__).resolve().parents[2] / "config" / "tokens_config.py",
        )

    def _token_manager_instance(self) -> Any | None:
        tokens = self._resolve_tokens()
        if not tokens:
            return None
        if self._token_manager is None:
            self._token_manager = _WikiTokenRotator(tokens)
        return self._token_manager

    def _current_token(self, tokens: tuple[str, ...]) -> str | None:
        if not tokens:
            return None
        manager = self._token_manager_instance()
        if manager is None:
            return None
        try:
            return manager.get_token()
        except RuntimeError:
            return None

    def _rotate_token(self) -> None:
        manager = self._token_manager_instance()
        if manager is None:
            return
        try:
            manager.rotate_token()
        except RuntimeError:
            return

    def _invalidate_current_token(self) -> None:
        manager = self._token_manager_instance()
        if manager is None:
            return
        try:
            manager.invalidate_current()
        except RuntimeError:
            return

    def _cache_text_path(self, owner: str, repo: str) -> Path:
        return self.cache_dir / _safe_repo_key(owner, repo) / "wiki_text.txt"

    def _cache_metadata_path(self, owner: str, repo: str) -> Path:
        return self.cache_dir / _safe_repo_key(owner, repo) / "wiki_metadata.json"

    def _write_cache(
        self,
        text_path: Path,
        metadata_path: Path,
        *,
        text: str,
        status: str,
        notes: str | None,
    ) -> None:
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8", newline="\n")
        metadata = {
            "schema_version": "topic_wiki_cache_v1",
            "status": status,
            "cached_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "text_path": str(text_path),
            "notes": notes,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _wiki_clone_url(self, owner: str, repo: str, *, token: str | None) -> str:
        if token:
            return f"https://x-access-token:{token}@github.com/{owner}/{repo}.wiki.git"
        return f"https://github.com/{owner}/{repo}.wiki.git"

    def _emit(self, message: str) -> None:
        if self.log is not None:
            self.log(message)


def _safe_repo_key(owner: str, repo: str) -> str:
    return f"{str(owner).strip().replace('/', '_')}__{str(repo).strip().replace('/', '_')}"


def _extract_text_from_wiki_clone(clone_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(clone_dir.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(clone_dir)
        parts.append(f"# {rel}\n{content.strip()}")
    return "\n\n".join(part for part in parts if part.strip())


def _combined_process_output(result: Any) -> str:
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return f"{stdout}\n{stderr}".strip()


def _looks_missing(message: str) -> bool:
    return any(
        marker in message
        for marker in (
            "repository not found",
            "not found",
            "does not exist",
            "could not read from remote repository",
            "could not read username",
        )
    )


def _looks_auth_failure(message: str) -> bool:
    return any(
        marker in message
        for marker in (
            "authentication",
            "bad credentials",
            "access denied",
            "permission denied",
            "401",
            "403",
        )
    )


def _looks_rate_limited(message: str) -> bool:
    return any(
        marker in message
        for marker in (
            "rate limit",
            "secondary rate",
            "too many requests",
            "abuse detection",
            "429",
        )
    )


def _redact_token(value: str) -> str:
    if "x-access-token:" not in value:
        return value
    return re.sub(r"x-access-token:[^@\s]+@", "x-access-token:<redacted>@", value)
