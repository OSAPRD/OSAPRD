"""GitHub API helpers for topic-training artifact extraction.

This module is intentionally small and REST-focused: repository search,
repository metadata, README text, file lists, and wiki text are the only GitHub
surfaces the training-data extraction stage needs.
"""

from __future__ import annotations

import base64
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from common import log


@dataclass
class _TokenInfo:
    token: str
    remaining: int | None = None
    reset: int | None = None


class _TokenManager:
    """Local token rotator for the post-processing-only Docker image."""

    def __init__(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("At least one token is required.")
        self.tokens_info = [_TokenInfo(token=token) for token in tokens]
        self.current_index = 0
        self.invalid_tokens: set[str] = set()

    def get_token(self) -> str:
        for _ in self.tokens_info:
            token_info = self.tokens_info[self.current_index]
            if token_info.token in self.invalid_tokens:
                self.rotate_token()
                continue
            now = time.time()
            if token_info.remaining == 0 and token_info.reset and now < token_info.reset:
                self.rotate_token()
                continue
            return token_info.token
        raise RuntimeError("All tokens are invalid or exhausted.")

    def rotate_token(self) -> None:
        for _ in self.tokens_info:
            self.current_index = (self.current_index + 1) % len(self.tokens_info)
            token_info = self.tokens_info[self.current_index]
            if token_info.token in self.invalid_tokens:
                continue
            now = time.time()
            if (token_info.remaining is None or token_info.remaining > 0) or (
                token_info.reset and int(now) >= token_info.reset
            ):
                return
        raise RuntimeError("All tokens are exhausted until reset.")

    def invalidate_current(self) -> None:
        self.invalid_tokens.add(self.tokens_info[self.current_index].token)
        if len(self.invalid_tokens) >= len(self.tokens_info):
            raise RuntimeError("All tokens are invalid.")
        self.rotate_token()

    def update_limit(self, remaining: int, reset_timestamp: int) -> None:
        token_info = self.tokens_info[self.current_index]
        token_info.remaining = int(remaining)
        token_info.reset = int(reset_timestamp)


@dataclass(frozen=True)
class FileListResult:
    """Repository file-list fetch result plus the ref/commit used."""

    status: str
    files: tuple[str, ...]
    source_ref: str | None = None
    source_commit: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class TextFetchResult:
    """Text fetch result for README or wiki-style sources."""

    status: str
    text: str = ""
    notes: str | None = None


@dataclass(frozen=True)
class RepositorySearchResult:
    """One GitHub repository-search page and its query metadata."""

    status: str
    query: str
    page: int
    per_page: int
    total_count: int
    incomplete_results: bool
    items: tuple[dict[str, Any], ...]


class GitHubTopicTrainingClient:
    """Small GitHub client using extraction-style token rotation and retries."""

    def __init__(
        self,
        tokens: list[str] | tuple[str, ...],
        *,
        base_url: str = "https://api.github.com",
        session: requests.Session | None = None,
    ) -> None:
        token_list = [str(token).strip() for token in tokens if str(token).strip()]
        if not token_list:
            raise RuntimeError("No GitHub tokens configured for topic-training extraction.")
        self.token_manager = _TokenManager(token_list)
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        token = self.token_manager.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

    @staticmethod
    def _is_invalid_token_response(resp: requests.Response) -> bool:
        if resp.status_code not in (401, 403):
            return False
        text = (resp.text or "").lower()
        return (
            "bad credentials" in text
            or "requires authentication" in text
            or "invalid token" in text
        )

    def request_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        allow_statuses: tuple[int, ...] = (),
    ) -> tuple[int, Any, requests.Response]:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        retries = 0
        backoff = 2
        while True:
            try:
                resp = self.session.get(url, headers=self._headers(), params=params, timeout=30)
                if self._is_invalid_token_response(resp):
                    self.token_manager.invalidate_current()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                if resp.status_code == 403 and "rate limit" in (resp.text or "").lower():
                    self.token_manager.rotate_token()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset = resp.headers.get("X-RateLimit-Reset")
                if remaining is not None and reset is not None:
                    try:
                        self.token_manager.update_limit(int(remaining), int(reset))
                    except ValueError:
                        pass
                if resp.status_code in allow_statuses:
                    return resp.status_code, _safe_json(resp), resp
                resp.raise_for_status()
                return resp.status_code, _safe_json(resp), resp
            except requests.RequestException:
                retries += 1
                if retries > 5:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        _, payload, _ = self.request_json(f"/repos/{quote(owner)}/{quote(repo)}")
        return payload if isinstance(payload, dict) else {}

    def search_repositories(
        self,
        *,
        created_bucket: str,
        topics_query: str = "topics:>0",
        stars_query: str | None = None,
        page: int = 1,
        per_page: int = 100,
    ) -> RepositorySearchResult:
        query_parts = [
            topics_query,
            str(stars_query or "").strip(),
            f"created:{created_bucket}",
            "fork:false",
            "archived:false",
            "is:public",
        ]
        query = " ".join(part for part in query_parts if part)
        status, payload, _ = self.request_json(
            "/search/repositories",
            params={
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
            allow_statuses=(422,),
        )
        if status == 422 or not isinstance(payload, dict):
            return RepositorySearchResult(
                status="invalid_query" if status == 422 else "fetch_failed",
                query=query,
                page=int(page),
                per_page=int(per_page),
                total_count=0,
                incomplete_results=False,
                items=(),
            )
        items = tuple(item for item in payload.get("items", ()) if isinstance(item, dict))
        return RepositorySearchResult(
            status="fetched",
            query=query,
            page=int(page),
            per_page=int(per_page),
            total_count=int(payload.get("total_count") or 0),
            incomplete_results=bool(payload.get("incomplete_results")),
            items=items,
        )

    def get_readme_text(self, owner: str, repo: str, ref: str | None = None) -> TextFetchResult:
        params = {"ref": ref} if ref else None
        status, payload, _ = self.request_json(
            f"/repos/{quote(owner)}/{quote(repo)}/readme",
            params=params,
            allow_statuses=(404,),
        )
        if status == 404:
            return TextFetchResult(status="missing")
        if not isinstance(payload, dict):
            return TextFetchResult(status="fetch_failed", notes="unexpected README payload")
        content = payload.get("content")
        encoding = str(payload.get("encoding") or "").lower()
        if isinstance(content, str) and encoding == "base64":
            try:
                text = base64.b64decode(content, validate=False).decode(
                    "utf-8",
                    errors="replace",
                )
                return TextFetchResult(status="fetched", text=text)
            except Exception as exc:
                return TextFetchResult(status="decode_failed", notes=str(exc))
        return TextFetchResult(status="missing", notes="README content absent")

    def list_repository_files(
        self,
        owner: str,
        repo: str,
        *,
        default_branch: str | None,
        clone_fallback: bool = True,
    ) -> FileListResult:
        ref = default_branch or "HEAD"
        status, payload, _ = self.request_json(
            f"/repos/{quote(owner)}/{quote(repo)}/git/trees/{quote(ref, safe='')}",
            params={"recursive": "1"},
            allow_statuses=(404, 409, 422),
        )
        if status in (404, 409, 422):
            if clone_fallback:
                return self._list_files_with_git_clone(owner, repo)
            return FileListResult(status="missing", files=(), source_ref=ref)
        if not isinstance(payload, dict):
            return FileListResult(status="fetch_failed", files=(), source_ref=ref)
        tree = payload.get("tree")
        truncated = bool(payload.get("truncated"))
        sha = payload.get("sha")
        if isinstance(tree, list) and not truncated:
            files = tuple(
                sorted(
                    {
                        str(item.get("path") or "").replace("\\", "/").lstrip("/")
                        for item in tree
                        if isinstance(item, dict)
                        and item.get("type") == "blob"
                        and str(item.get("path") or "").strip()
                    }
                )
            )
            return FileListResult(
                status="fetched",
                files=files,
                source_ref=ref,
                source_commit=str(sha) if sha else None,
            )
        if truncated and clone_fallback:
            return self._list_files_with_git_clone(owner, repo)
        return FileListResult(
            status="truncated",
            files=(),
            source_ref=ref,
            source_commit=str(sha) if sha else None,
            notes="GitHub tree response was truncated.",
        )

    def _list_files_with_git_clone(self, owner: str, repo: str) -> FileListResult:
        token: str | None
        try:
            token = self.token_manager.get_token()
        except Exception:
            token = None
        if token:
            url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"
        else:
            url = f"https://github.com/{owner}/{repo}.git"
        with tempfile.TemporaryDirectory(prefix="topic-training-repo-") as temp_dir:
            clone_dir = Path(temp_dir) / "repo"
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--no-checkout",
                    url,
                    str(clone_dir),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                notes = _redact_token(f"{result.stdout}\n{result.stderr}".strip())
                log(f"Repository file-list clone failed for {owner}/{repo}: {notes[:180]}")
                return FileListResult(status="clone_failed", files=(), notes=notes)
            tree = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "HEAD"],
                cwd=clone_dir,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if tree.returncode != 0:
                notes = _redact_token(f"{tree.stdout}\n{tree.stderr}".strip())
                return FileListResult(status="clone_failed", files=(), notes=notes)
            files = tuple(
                sorted(
                    {
                        line.strip().replace("\\", "/").lstrip("/")
                        for line in (tree.stdout or "").splitlines()
                        if line.strip()
                    }
                )
            )
            return FileListResult(status="fetched_via_clone", files=files, source_ref="HEAD")


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return {}


def _redact_token(value: str) -> str:
    return re.sub(r"x-access-token:[^@\s]+@", "x-access-token:<redacted>@", value)
