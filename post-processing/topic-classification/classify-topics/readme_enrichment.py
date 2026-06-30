"""Cache-first GitHub README enrichment for runtime topic classification.

Curation outputs often contain README text, but not always. The classifier uses
this helper to fill missing README text from GitHub, cache the result, and avoid
re-fetching the same repository across runs.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class ReadmeFetchResult:
    """README lookup status plus text/cache metadata for manifests."""

    status: str
    text: str = ""
    cache_path: str | None = None
    metadata_path: str | None = None
    notes: str | None = None


@dataclass
class _TokenState:
    token: str
    remaining: int | None = None
    reset: int | None = None


class _ReadmeTokenRotator:
    """Minimal GitHub token rotator for README REST API requests."""

    def __init__(self, tokens: Iterable[str]) -> None:
        self.tokens = [_TokenState(token=str(token).strip()) for token in tokens if str(token).strip()]
        self.current_index = 0
        self.invalid_tokens: set[str] = set()

    def get_token(self) -> str | None:
        if not self.tokens:
            return None
        for _ in self.tokens:
            token_state = self.tokens[self.current_index]
            if token_state.token in self.invalid_tokens:
                self.rotate_token()
                continue
            now = time.time()
            if token_state.remaining == 0 and token_state.reset and now < token_state.reset:
                self.rotate_token()
                continue
            return token_state.token
        return None

    def rotate_token(self) -> None:
        if not self.tokens:
            return
        for _ in self.tokens:
            self.current_index = (self.current_index + 1) % len(self.tokens)
            token_state = self.tokens[self.current_index]
            if token_state.token in self.invalid_tokens:
                continue
            now = time.time()
            if token_state.remaining is None or token_state.remaining > 0:
                return
            if token_state.reset and now >= token_state.reset:
                return

    def invalidate_current(self) -> None:
        if not self.tokens:
            return
        self.invalid_tokens.add(self.tokens[self.current_index].token)
        self.rotate_token()

    def update_limit(self, remaining: int, reset_timestamp: int) -> None:
        if not self.tokens:
            return
        token_state = self.tokens[self.current_index]
        token_state.remaining = int(remaining)
        token_state.reset = int(reset_timestamp)


class TopicReadmeEnricher:
    """Cache-first README text provider backed by the GitHub repository API."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        tokens: Iterable[str] | None = None,
        log: Callable[[str], None] | None = None,
        session: requests.Session | None = None,
        base_url: str = "https://api.github.com",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.tokens = tuple(str(token).strip() for token in (tokens or ()) if str(token).strip())
        self.log = log
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.token_rotator = _ReadmeTokenRotator(self.tokens)
        self.stats = {
            "input": 0,
            "cache_hit": 0,
            "fetched": 0,
            "missing": 0,
            "auth_failed": 0,
            "rate_limited": 0,
            "fetch_failed": 0,
            "decode_failed": 0,
        }

    def get_readme_text(self, owner: str, repo: str, *, ref: str | None = None) -> ReadmeFetchResult:
        cache_text_path = self._cache_text_path(owner, repo)
        cache_metadata_path = self._cache_metadata_path(owner, repo)
        if cache_text_path.exists():
            text = cache_text_path.read_text(encoding="utf-8", errors="replace")
            self.stats["cache_hit"] += 1
            return ReadmeFetchResult(
                status="cache_hit" if text.strip() else "missing",
                text=text,
                cache_path=str(cache_text_path),
                metadata_path=str(cache_metadata_path),
            )

        if not self.tokens:
            self.stats["fetch_failed"] += 1
            return ReadmeFetchResult(
                status="fetch_failed",
                cache_path=str(cache_text_path),
                metadata_path=str(cache_metadata_path),
                notes="no GitHub tokens configured",
            )

        attempt_count = max(1, len(self.tokens))
        last_status = "fetch_failed"
        last_notes: str | None = None
        for _attempt in range(1, attempt_count + 1):
            token = self.token_rotator.get_token()
            if not token:
                break
            try:
                response = self.session.get(
                    f"{self.base_url}/repos/{quote(owner)}/{quote(repo)}/readme",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github+json",
                    },
                    params={"ref": ref} if ref else None,
                    timeout=30,
                )
            except requests.RequestException as exc:
                last_status = "fetch_failed"
                last_notes = str(exc)
                self.token_rotator.rotate_token()
                continue

            self._update_rate_limit(response)
            if response.status_code == 404:
                self._write_cache(
                    cache_text_path,
                    cache_metadata_path,
                    text="",
                    status="missing",
                    notes=None,
                )
                self.stats["missing"] += 1
                return ReadmeFetchResult(
                    status="missing",
                    cache_path=str(cache_text_path),
                    metadata_path=str(cache_metadata_path),
                )
            message = response.text or ""
            if response.status_code in (401, 403) and _looks_invalid_token(message):
                last_status = "auth_failed"
                last_notes = _safe_notes(message)
                self.token_rotator.invalidate_current()
                continue
            if response.status_code in (403, 429) and _looks_rate_limited(message):
                last_status = "rate_limited"
                last_notes = _safe_notes(message)
                self.token_rotator.rotate_token()
                continue
            if response.status_code >= 400:
                last_status = "fetch_failed"
                last_notes = _safe_notes(message)
                self.token_rotator.rotate_token()
                continue

            payload = _safe_json(response)
            result = self._decode_readme_payload(payload)
            if result.status == "fetched":
                self._write_cache(
                    cache_text_path,
                    cache_metadata_path,
                    text=result.text,
                    status=result.status,
                    notes=result.notes,
                )
                self.stats["fetched"] += 1
                return ReadmeFetchResult(
                    status="fetched",
                    text=result.text,
                    cache_path=str(cache_text_path),
                    metadata_path=str(cache_metadata_path),
                    notes=result.notes,
                )
            last_status = result.status
            last_notes = result.notes
            self._write_cache(
                cache_text_path,
                cache_metadata_path,
                text="",
                status=result.status,
                notes=result.notes,
            )
            self.stats[result.status] = self.stats.get(result.status, 0) + 1
            return ReadmeFetchResult(
                status=result.status,
                cache_path=str(cache_text_path),
                metadata_path=str(cache_metadata_path),
                notes=result.notes,
            )

        self.stats[last_status] = self.stats.get(last_status, 0) + 1
        self._emit(f"README fetch failed for {owner}/{repo} (status={last_status}).")
        return ReadmeFetchResult(
            status=last_status,
            cache_path=str(cache_text_path),
            metadata_path=str(cache_metadata_path),
            notes=last_notes,
        )

    def record_input_readme(self) -> None:
        self.stats["input"] += 1

    def _decode_readme_payload(self, payload: Any) -> ReadmeFetchResult:
        if not isinstance(payload, dict):
            return ReadmeFetchResult(status="fetch_failed", notes="unexpected README payload")
        content = payload.get("content")
        encoding = str(payload.get("encoding") or "").lower()
        if isinstance(content, str) and encoding == "base64":
            try:
                text = base64.b64decode(content, validate=False).decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception as exc:
                return ReadmeFetchResult(status="decode_failed", notes=str(exc))
            if text.strip():
                return ReadmeFetchResult(status="fetched", text=text)
        return ReadmeFetchResult(status="missing", notes="README content absent")

    def _cache_text_path(self, owner: str, repo: str) -> Path:
        return self.cache_dir / _safe_repo_key(owner, repo) / "readme_text.txt"

    def _cache_metadata_path(self, owner: str, repo: str) -> Path:
        return self.cache_dir / _safe_repo_key(owner, repo) / "readme_metadata.json"

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
            "schema_version": "topic_readme_cache_v1",
            "status": status,
            "cached_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "text_path": str(text_path),
            "notes": notes,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _update_rate_limit(self, response: requests.Response) -> None:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining is None or reset is None:
            return
        try:
            self.token_rotator.update_limit(int(remaining), int(reset))
        except ValueError:
            return

    def _emit(self, message: str) -> None:
        if self.log is not None:
            self.log(message)


def _safe_repo_key(owner: str, repo: str) -> str:
    return f"{str(owner).strip().replace('/', '_')}__{str(repo).strip().replace('/', '_')}"


def _safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {}


def _looks_invalid_token(message: str) -> bool:
    normalized = str(message or "").lower()
    return any(
        marker in normalized
        for marker in (
            "bad credentials",
            "requires authentication",
            "invalid token",
            "access denied",
        )
    )


def _looks_rate_limited(message: str) -> bool:
    normalized = str(message or "").lower()
    return any(
        marker in normalized
        for marker in (
            "rate limit",
            "secondary rate",
            "too many requests",
            "abuse detection",
        )
    )


def _safe_notes(value: str) -> str:
    return str(value or "").replace("\n", " ")[:500]
