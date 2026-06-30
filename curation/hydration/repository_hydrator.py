"""Git-backed repository hydration helpers.

`RepositoryHydrator` owns cloning/fetching, commit resolution, worktree export,
and diff/name-status queries. Higher-level PR hydration uses these primitives
to materialize only the files required for metrics instead of copying entire
repositories for every snapshot.
"""

from __future__ import annotations

import os
import signal
import shutil
import subprocess
import tarfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class GitCommandError(RuntimeError):
    """Raised when a git command fails."""


def _float_env(name: str, default: float) -> float:
    """Read a float environment variable with a forgiving default fallback."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


GIT_ARCHIVE_TIMEOUT_SECONDS = _float_env("GIT_ARCHIVE_TIMEOUT_SECONDS", 300.0)


def _git_env() -> dict[str, str]:
    """Return git environment settings used for all repository commands."""
    env = os.environ.copy()
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    return env


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate a git subprocess and its process group when possible."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _run_git(
    args: list[str],
    cwd: Optional[Path] = None,
    capture_output: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Run a git command and raise :class:`GitCommandError` on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=False,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
        env=_git_env(),
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise GitCommandError(f"git {' '.join(args)} failed: {stderr}")
    return result


def _parse_git_iso_timestamp(value: str) -> Optional[datetime]:
    """Parse git/GitHub timestamps into UTC datetimes."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class RepositoryHydrator:
    """Manage one local repository clone and file exports for curation."""

    owner: str
    name: str
    clone_root: Path
    default_branch: Optional[str] = None
    github_token: Optional[str] = None

    def __post_init__(self) -> None:
        """Derive local clone/worktree paths and the authenticated remote URL."""
        safe_owner = self.owner.replace("/", "_")
        safe_name = self.name.replace("/", "_")
        self.repo_dir = self.clone_root / f"{safe_owner}__{safe_name}"
        self.worktree_dir = self.clone_root / f"{safe_owner}__{safe_name}__worktree"
        self.repo_url = self._build_repo_url()

    def set_token(self, token: Optional[str]) -> None:
        """Update the token used for git operations."""
        self.github_token = token
        self.repo_url = self._build_repo_url()

    def prepare(self) -> None:
        """Clone or update the repository for processing."""
        self.clone_root.mkdir(parents=True, exist_ok=True)
        if (self.repo_dir / ".git").exists():
            if self.github_token:
                self._set_remote_url(self.repo_url)
            _run_git(["fetch", "--all", "--prune"], cwd=self.repo_dir, capture_output=True)
        else:
            if self.repo_dir.exists():
                shutil.rmtree(self.repo_dir, ignore_errors=True)
            _run_git(["clone", self.repo_url, str(self.repo_dir)], capture_output=True)

    def cleanup(self) -> None:
        """Remove the local clone after processing is complete."""
        if self.worktree_dir.exists():
            if self.repo_dir.exists():
                try:
                    _run_git(
                        ["worktree", "remove", "--force", str(self.worktree_dir)],
                        cwd=self.repo_dir,
                        capture_output=True,
                    )
                except GitCommandError:
                    pass
            shutil.rmtree(self.worktree_dir, ignore_errors=True)
        if self.repo_dir.exists():

            def _onerror(func, path, _exc):
                try:
                    Path(path).chmod(0o666)
                except Exception:
                    pass
                try:
                    func(path)
                except FileNotFoundError:
                    return

            shutil.rmtree(self.repo_dir, onerror=_onerror)

    def resolve_default_branch(self) -> str:
        """Determine the default branch for the repository."""
        if self.default_branch:
            if self._branch_exists(self.default_branch):
                return self.default_branch
        try:
            ref = _run_git(
                ["symbolic-ref", "refs/remotes/origin/HEAD"],
                cwd=self.repo_dir,
                capture_output=True,
            ).stdout.strip()
            if ref.startswith("refs/remotes/origin/"):
                branch = ref.split("/", 3)[-1]
                if self._branch_exists(branch):
                    return branch
        except GitCommandError:
            pass
        for candidate in ("main", "master"):
            if self._branch_exists(candidate):
                return candidate
        return "main"

    def _branch_exists(self, branch: str) -> bool:
        try:
            _run_git(
                ["show-ref", "--verify", f"refs/remotes/origin/{branch}"],
                cwd=self.repo_dir,
                capture_output=True,
            )
            return True
        except GitCommandError:
            return False

    def _build_repo_url(self) -> str:
        """Build the GitHub remote URL, embedding the active token when set."""
        if self.github_token:
            return f"https://x-access-token:{self.github_token}@github.com/{self.owner}/{self.name}.git"
        return f"https://github.com/{self.owner}/{self.name}.git"

    def _set_remote_url(self, url: str) -> None:
        """Best-effort update of the origin URL after token rotation."""
        try:
            _run_git(["remote", "set-url", "origin", url], cwd=self.repo_dir, capture_output=True)
        except GitCommandError:
            pass

    def commit_after(self, iso_timestamp: str, branch: str) -> Optional[str]:
        """Return the earliest commit after the given timestamp on the branch."""
        try:
            result = _run_git(
                ["rev-list", "--after", iso_timestamp, "--reverse", "-n", "1", f"origin/{branch}"],
                cwd=self.repo_dir,
                capture_output=True,
            )
            sha = result.stdout.strip()
            return sha or None
        except GitCommandError:
            return None

    def commits_after(self, iso_timestamps: list[str], branch: str) -> dict[str, Optional[str]]:
        """Return the earliest commit after each timestamp using one streamed git walk."""
        parsed_targets = [
            (str(timestamp), _parse_git_iso_timestamp(str(timestamp)))
            for timestamp in iso_timestamps
            if str(timestamp).strip()
        ]
        results: dict[str, Optional[str]] = {
            timestamp: None for timestamp, _parsed in parsed_targets
        }
        sortable_targets = [
            (timestamp, parsed) for timestamp, parsed in parsed_targets if parsed is not None
        ]
        if not sortable_targets:
            return results

        earliest = min(parsed for _timestamp, parsed in sortable_targets)
        args = [
            "log",
            f"--after={earliest.isoformat()}",
            "--reverse",
            "--format=%H%x00%cI",
            f"origin/{branch}",
        ]
        try:
            proc = subprocess.Popen(
                ["git", *args],
                cwd=str(self.repo_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            for timestamp, _target_dt in sortable_targets:
                results[timestamp] = self.commit_after(timestamp, branch)
            return results
        completed_by_stream = False
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if not line or "\x00" not in line:
                    continue
                sha, commit_timestamp = line.split("\x00", 1)
                commit_dt = _parse_git_iso_timestamp(commit_timestamp)
                if commit_dt is None:
                    continue
                for target_timestamp, target_dt in sortable_targets:
                    if results.get(target_timestamp) is None and commit_dt > target_dt:
                        results[target_timestamp] = sha.strip() or None
                if all(results.get(timestamp) is not None for timestamp, _dt in sortable_targets):
                    completed_by_stream = True
                    proc.terminate()
                    break
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        return_code = proc.wait()
        if return_code not in (0, -15) and not completed_by_stream:
            # Preserve correctness on unexpected git failures by falling back to
            # the older per-target query for unresolved timestamps.
            for timestamp, _target_dt in sortable_targets:
                if results.get(timestamp) is None:
                    results[timestamp] = self.commit_after(timestamp, branch)
            if all(results.get(timestamp) is None for timestamp, _dt in sortable_targets):
                _ = stderr
        return results

    def is_ancestor(self, ancestor_sha: str, descendant_sha: str) -> Optional[bool]:
        """Return whether ancestor_sha is in descendant_sha's ancestry, or None on git errors."""
        ancestor = (ancestor_sha or "").strip()
        descendant = (descendant_sha or "").strip()
        if not ancestor or not descendant:
            return None
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=str(self.repo_dir),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        return None

    def latest_commit_timestamp(self, branch: str) -> Optional[str]:
        """Return the latest observed commit timestamp on the remote branch."""
        try:
            result = _run_git(
                ["log", "-1", "--format=%cI", f"origin/{branch}"],
                cwd=self.repo_dir,
                capture_output=True,
            )
            timestamp = result.stdout.strip()
            return timestamp or None
        except GitCommandError:
            return None

    def parent_commit(self, sha: str) -> Optional[str]:
        """Return the first parent commit for a merge commit."""
        try:
            result = _run_git(["rev-parse", f"{sha}^1"], cwd=self.repo_dir, capture_output=True)
            parent = result.stdout.strip()
            return parent or None
        except GitCommandError:
            return None

    def has_commit(self, sha: Optional[str]) -> bool:
        """Return whether the given commit object exists locally."""
        if not sha:
            return False
        try:
            _run_git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=self.repo_dir, capture_output=True)
            return True
        except GitCommandError:
            return False

    def ensure_commit(self, sha: Optional[str], pr_number: Optional[int] = None) -> bool:
        """Best-effort fetch when a commit is missing locally."""
        if not sha:
            return False
        if self.has_commit(sha):
            return True
        fetch_attempts: list[list[str]] = [["fetch", "--no-tags", "origin", sha]]
        if pr_number is not None:
            pr_ref = str(pr_number)
            fetch_attempts.extend(
                [
                    ["fetch", "--no-tags", "origin", f"pull/{pr_ref}/head"],
                    ["fetch", "--no-tags", "origin", f"pull/{pr_ref}/merge"],
                ]
            )
        fetch_attempts.append(["fetch", "--all", "--prune"])
        for args in fetch_attempts:
            try:
                _run_git(args, cwd=self.repo_dir, capture_output=True)
            except GitCommandError:
                continue
            if self.has_commit(sha):
                return True
        return self.has_commit(sha)

    def prefetch_pull_refs(self, pr_numbers: list[int], chunk_size: int = 50) -> int:
        """Best-effort batch fetch of GitHub pull-request refs for a repository."""
        normalized: set[int] = set()
        for number in pr_numbers:
            try:
                value = int(number)
            except (TypeError, ValueError):
                continue
            if value > 0:
                normalized.add(value)
        normalized_numbers = sorted(normalized)
        if not normalized_numbers:
            return 0
        chunk_size = max(1, int(chunk_size))

        def _fetch_refspecs(refspecs: list[str]) -> int:
            fetched = 0
            for index in range(0, len(refspecs), chunk_size):
                chunk = refspecs[index : index + chunk_size]
                try:
                    _run_git(["fetch", "--no-tags", "origin", *chunk], cwd=self.repo_dir, capture_output=True)
                    fetched += len(chunk)
                    continue
                except GitCommandError:
                    pass
                for refspec in chunk:
                    try:
                        _run_git(["fetch", "--no-tags", "origin", refspec], cwd=self.repo_dir, capture_output=True)
                        fetched += 1
                    except GitCommandError:
                        continue
            return fetched

        head_refspecs = [
            f"+refs/pull/{number}/head:refs/remotes/origin/pull/{number}/head"
            for number in normalized_numbers
        ]
        merge_refspecs = [
            f"+refs/pull/{number}/merge:refs/remotes/origin/pull/{number}/merge"
            for number in normalized_numbers
        ]
        return _fetch_refspecs(head_refspecs) + _fetch_refspecs(merge_refspecs)

    def ensure_worktree(self, sha: str) -> Path:
        """Ensure a single worktree exists and is checked out to the requested commit."""
        def _clear_stale_worktree_lock() -> None:
            git_dir = self.repo_dir / ".git"
            if not git_dir.exists() or not git_dir.is_dir():
                return
            worktree_meta = git_dir / "worktrees" / self.worktree_dir.name
            lock_path = worktree_meta / "index.lock"
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError:
                    pass

        self.worktree_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.worktree_dir.exists() and not self.worktree_dir.is_dir():
            try:
                self.worktree_dir.unlink()
            except OSError:
                pass
        if self.worktree_dir.exists() and (self.worktree_dir / ".git").exists():
            try:
                current = _run_git(["rev-parse", "HEAD"], cwd=self.worktree_dir, capture_output=True)
                if (current.stdout or "").strip() == sha:
                    return self.worktree_dir
                _run_git(["checkout", "--detach", sha], cwd=self.worktree_dir, capture_output=True)
                return self.worktree_dir
            except GitCommandError:
                try:
                    _run_git(
                        ["worktree", "remove", "--force", str(self.worktree_dir)],
                        cwd=self.repo_dir,
                        capture_output=True,
                    )
                except GitCommandError:
                    pass
                shutil.rmtree(self.worktree_dir, ignore_errors=True)
        if self.worktree_dir.exists():
            shutil.rmtree(self.worktree_dir, ignore_errors=True)
        try:
            _run_git(["worktree", "prune"], cwd=self.repo_dir, capture_output=True)
        except GitCommandError:
            pass
        _clear_stale_worktree_lock()
        try:
            _run_git(
                ["worktree", "add", "--detach", str(self.worktree_dir), sha],
                cwd=self.repo_dir,
                capture_output=True,
            )
        except GitCommandError as exc:
            if "index.lock" not in str(exc).lower():
                raise
            _clear_stale_worktree_lock()
            _run_git(
                ["worktree", "add", "--detach", str(self.worktree_dir), sha],
                cwd=self.repo_dir,
                capture_output=True,
            )
        return self.worktree_dir

    def files_existing_at_commit(self, sha: str, file_paths: list[str]) -> list[str]:
        """Return requested paths that exist as files at a commit."""
        normalized_paths = sorted(
            {
                str(path).strip().replace("\\", "/").lstrip("/")
                for path in (file_paths or [])
                if str(path).strip()
            }
        )
        if not sha or not normalized_paths:
            return []
        try:
            result = _run_git(
                ["ls-tree", "-r", "--name-only", sha, "--", *normalized_paths],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
            )
        except GitCommandError:
            return []
        requested = set(normalized_paths)
        existing: list[str] = []
        for line in (result.stdout or "").splitlines():
            normalized = line.strip().replace("\\", "/").lstrip("/")
            if normalized in requested:
                existing.append(normalized)
        return sorted(set(existing))

    def _export_files_with_archive(
        self,
        sha: str,
        file_paths: list[str],
        destination: Path,
    ) -> list[str]:
        """Export selected files through ``git archive`` without a full checkout copy."""
        command = [
            "git",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.smudge=cat",
            "-c",
            "filter.lfs.required=false",
            "archive",
            "--format=tar",
            sha,
            "--",
            *file_paths,
        ]
        popen_kwargs = {
            "cwd": str(self.repo_dir),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": False,
            "env": _git_env(),
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(command, **popen_kwargs)
        timed_out = {"value": False}

        def _on_timeout() -> None:
            timed_out["value"] = True
            _kill_process_tree(proc)

        timer: threading.Timer | None = None
        if GIT_ARCHIVE_TIMEOUT_SECONDS > 0:
            timer = threading.Timer(GIT_ARCHIVE_TIMEOUT_SECONDS, _on_timeout)
            timer.daemon = True
            timer.start()
        copied: list[str] = []
        try:
            assert proc.stdout is not None
            with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
                requested = set(file_paths)
                for member in archive:
                    normalized = member.name.replace("\\", "/").lstrip("/")
                    if (
                        not member.isfile()
                        or normalized not in requested
                        or normalized.startswith("../")
                        or "/../" in normalized
                    ):
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    target = destination / normalized
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    copied.append(normalized)
        except Exception:
            if proc.poll() is None:
                _kill_process_tree(proc)
            proc.wait()
            if timed_out["value"]:
                raise GitCommandError(
                    f"git archive timed out after {GIT_ARCHIVE_TIMEOUT_SECONDS:.1f}s"
                )
            raise
        finally:
            if timer is not None:
                timer.cancel()
            if proc.stdout is not None:
                proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        return_code = proc.wait()
        if timed_out["value"]:
            raise GitCommandError(f"git archive timed out after {GIT_ARCHIVE_TIMEOUT_SECONDS:.1f}s")
        if return_code != 0:
            raise GitCommandError(f"git archive failed: {stderr.strip()}")
        return sorted(set(copied))

    def export_files(self, sha: str, file_paths: list[str], destination: Path) -> list[str]:
        """Export only specific files for the given commit into destination."""
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        existing_paths = self.files_existing_at_commit(sha, file_paths)
        if not existing_paths:
            return []
        try:
            copied = self._export_files_with_archive(sha, existing_paths, destination)
            if copied:
                return copied
        except (GitCommandError, OSError, tarfile.TarError):
            shutil.rmtree(destination, ignore_errors=True)
            destination.mkdir(parents=True, exist_ok=True)
        worktree = self.ensure_worktree(sha)
        copied: list[str] = []
        for rel_path in existing_paths:
            normalized = rel_path.lstrip("/").replace("\\", "/")
            if not normalized:
                continue
            src = worktree / normalized
            if not src.exists() or not src.is_file():
                continue
            dest = destination / normalized
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(normalized)
        return copied

    def read_readme_at_commit(self, sha: str) -> tuple[Optional[str], Optional[str]]:
        """Read README content at a commit without checking out a worktree."""
        if not sha:
            return None, None
        candidates = [
            "README.md",
            "README.rst",
            "README.txt",
            "README",
        ]
        try:
            root_files = _run_git(
                ["ls-tree", "--name-only", sha],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
            )
            existing_root_files = [
                line.strip()
                for line in (root_files.stdout or "").splitlines()
                if line.strip()
            ]
        except GitCommandError:
            existing_root_files = []
        for candidate in candidates:
            if candidate in existing_root_files:
                try:
                    result = _run_git(
                        ["show", f"{sha}:{candidate}"],
                        cwd=self.repo_dir,
                        capture_output=True,
                        text=True,
                    )
                    return candidate, result.stdout
                except GitCommandError:
                    continue
        for candidate in existing_root_files:
            if candidate.upper().startswith("README"):
                try:
                    result = _run_git(
                        ["show", f"{sha}:{candidate}"],
                        cwd=self.repo_dir,
                        capture_output=True,
                        text=True,
                    )
                    return candidate, result.stdout
                except GitCommandError:
                    continue
        return None, None

    def path_exists_at_commit(self, sha: str, path: str) -> bool:
        """Return whether a tracked path exists as a file at the given commit."""
        normalized = str(path).replace("\\", "/").lstrip("/")
        if not sha or not normalized:
            return False
        try:
            _run_git(
                ["cat-file", "-e", f"{sha}:{normalized}"],
                cwd=self.repo_dir,
                capture_output=True,
            )
            return True
        except GitCommandError:
            return False

    def path_statuses_between(
        self,
        start_sha: str,
        end_sha: str,
        paths: list[str],
    ) -> dict[str, dict[str, object]]:
        """Return deletion/rename status for tracked paths between two commits."""
        normalized_paths = sorted(
            {
                str(path).strip().replace("\\", "/").lstrip("/")
                for path in (paths or [])
                if str(path).strip()
            }
        )
        if not start_sha or not end_sha or not normalized_paths:
            return {}
        args = [
            "diff",
            "--name-status",
            "-M",
            start_sha,
            end_sha,
            "--",
            *normalized_paths,
        ]
        statuses: dict[str, dict[str, object]] = {}

        def _parse_name_status(output: str) -> None:
            for raw_line in (output or "").splitlines():
                parts = raw_line.strip().split("\t")
                if len(parts) < 2:
                    continue
                status = parts[0].strip()
                status_code = status[:1].upper()
                if status_code == "R" and len(parts) >= 3:
                    old_path = parts[1].replace("\\", "/").lstrip("/")
                    new_path = parts[2].replace("\\", "/").lstrip("/")
                    if old_path in normalized_paths:
                        statuses[old_path] = {
                            "status": "renamed",
                            "raw_status": status,
                            "old_path": old_path,
                            "new_path": new_path,
                        }
                elif status_code == "D":
                    old_path = parts[1].replace("\\", "/").lstrip("/")
                    if old_path in normalized_paths:
                        statuses[old_path] = {
                            "status": "deleted",
                            "raw_status": status,
                            "old_path": old_path,
                            "new_path": None,
                        }
                else:
                    changed_path = parts[1].replace("\\", "/").lstrip("/")
                    if changed_path in normalized_paths:
                        statuses[changed_path] = {
                            "status": status_code.lower() or "changed",
                            "raw_status": status,
                            "old_path": changed_path,
                            "new_path": parts[2].replace("\\", "/").lstrip("/") if len(parts) >= 3 else None,
                        }

        try:
            result = _run_git(args, cwd=self.repo_dir, capture_output=True, text=True)
            _parse_name_status(result.stdout or "")
        except GitCommandError:
            return {}

        if len(statuses) < len(normalized_paths):
            try:
                full_result = _run_git(
                    ["diff", "--name-status", "-M", start_sha, end_sha],
                    cwd=self.repo_dir,
                    capture_output=True,
                    text=True,
                )
                _parse_name_status(full_result.stdout or "")
            except GitCommandError:
                pass
        return statuses

    def diff_text_between(
        self,
        start_sha: str,
        end_sha: str,
        *,
        paths: Optional[list[str]] = None,
        unified: int = 0,
    ) -> str:
        """Return a unified diff between two commits for selected paths.

        PR hydration stores these small diffs as changed-line maps for
        longitudinal persistence tracking. The caller may pass an empty path
        list to request the full diff; path values are normalized to Git's
        repository-relative slash format before invoking ``git diff``.
        """
        start = str(start_sha or "").strip()
        end = str(end_sha or "").strip()
        if not start or not end:
            return ""
        normalized_paths = sorted(
            {
                str(path).strip().replace("\\", "/").lstrip("/")
                for path in (paths or [])
                if str(path).strip()
            }
        )
        try:
            normalized_unified = max(0, int(unified))
        except (TypeError, ValueError):
            normalized_unified = 0
        args = [
            "diff",
            f"--unified={normalized_unified}",
            "-M",
            start,
            end,
        ]
        if normalized_paths:
            args.extend(["--", *normalized_paths])
        try:
            result = _run_git(args, cwd=self.repo_dir, capture_output=True, text=True)
        except GitCommandError:
            return ""
        return result.stdout or ""

    def list_repository_files(self, ref: Optional[str] = None) -> list[str]:
        """Return all tracked file paths for the provided ref."""
        target_ref = (ref or "HEAD").strip() or "HEAD"
        result = _run_git(
            ["ls-tree", "-r", "--name-only", target_ref],
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
        )
        files: list[str] = []
        for line in (result.stdout or "").splitlines():
            item = line.strip().replace("\\", "/").lstrip("/")
            if item:
                files.append(item)
        return sorted(set(files))

    def resolve_ref_sha(self, ref: str) -> Optional[str]:
        """Resolve a git ref to its commit SHA."""
        target_ref = (ref or "").strip()
        if not target_ref:
            return None
        try:
            result = _run_git(["rev-parse", target_ref], cwd=self.repo_dir, capture_output=True)
            sha = (result.stdout or "").strip()
            return sha or None
        except GitCommandError:
            return None

    def commits_touching_paths_between(
        self,
        start_sha: str,
        end_sha: str,
        paths: list[str],
    ) -> Optional[list[dict[str, object]]]:
        """
        Return commits in (start_sha, end_sha] that touched any provided path.

        Each item contains:
        - sha: commit SHA
        - touched_paths: sorted list of matched paths touched in that commit

        Returns None when git cannot answer the query; callers should not treat
        that as equivalent to zero matching commits.
        """
        normalized_paths = sorted(
            {
                str(path).strip().replace("\\", "/").lstrip("/")
                for path in (paths or [])
                if str(path).strip()
            }
        )
        if not start_sha or not end_sha or not normalized_paths:
            return []

        args = [
            "log",
            "--reverse",
            "--pretty=format:__COMMIT__%H",
            "--name-only",
            f"{start_sha}..{end_sha}",
            "--",
            *normalized_paths,
        ]
        try:
            result = _run_git(args, cwd=self.repo_dir, capture_output=True, text=True)
        except GitCommandError:
            return None

        path_set = set(normalized_paths)
        events: list[dict[str, object]] = []
        current_sha: Optional[str] = None
        current_touched: set[str] = set()

        def _flush_current() -> None:
            nonlocal current_sha, current_touched
            if current_sha and current_touched:
                events.append(
                    {
                        "sha": current_sha,
                        "touched_paths": sorted(current_touched),
                    }
                )
            current_sha = None
            current_touched = set()

        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("__COMMIT__"):
                _flush_current()
                current_sha = line[len("__COMMIT__") :].strip() or None
                continue
            normalized = line.replace("\\", "/").lstrip("/")
            if normalized in path_set:
                current_touched.add(normalized)
        _flush_current()
        return events
