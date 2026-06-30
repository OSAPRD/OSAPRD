"""File-backed subprocess output helpers for curation tool invocations."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

DEFAULT_OUTPUT_PARSE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class FileBackedProcessResult:
    """Small subprocess result that keeps stdout/stderr on disk."""

    command: List[str]
    cwd: Optional[str]
    returncode: Optional[int]
    timed_out: bool
    elapsed_seconds: float
    stdout_path: str
    stderr_path: str
    stdout_preview: str
    stderr_preview: str
    stdout_tail: str
    stderr_tail: str
    size_limit_exceeded: bool = False
    termination_reason: Optional[str] = None
    size_limit_bytes: Optional[int] = None
    size_guard_bytes: Optional[int] = None
    size_guard_path: Optional[str] = None


def _safe_prefix(label: str) -> str:
    """Return a filesystem-safe prefix for stdout/stderr log files."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label or "tool")).strip("._-")
    return sanitized[:80] or "tool"


def _read_bytes_head(path: Path, max_bytes: int) -> bytes:
    """Read a bounded prefix from a binary file."""
    with path.open("rb") as handle:
        return handle.read(max(0, max_bytes))


def _read_bytes_tail(path: Path, max_bytes: int) -> bytes:
    """Read a bounded suffix from a binary file."""
    if max_bytes <= 0:
        return b""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read(max_bytes)


def _decode_output(data: bytes, *, encoding: str = "utf-8") -> str:
    """Decode subprocess output with replacement for invalid bytes."""
    return data.decode(encoding, errors="replace")


def read_text_head(path: Optional[str | Path], *, max_bytes: int = 4096) -> str:
    """Read a bounded prefix from a text output file."""

    if not path:
        return ""
    output_path = Path(path)
    if not output_path.exists() or not output_path.is_file():
        return ""
    text = _decode_output(_read_bytes_head(output_path, max_bytes))
    if output_path.stat().st_size > max_bytes:
        text += f"...[truncated {output_path.stat().st_size - max_bytes} bytes]"
    return text


def read_text_tail(path: Optional[str | Path], *, max_bytes: int = 4096) -> str:
    """Read a bounded suffix from a text output file."""

    if not path:
        return ""
    output_path = Path(path)
    if not output_path.exists() or not output_path.is_file():
        return ""
    size = output_path.stat().st_size
    text = _decode_output(_read_bytes_tail(output_path, max_bytes))
    if size > max_bytes:
        text = f"...[truncated {size - max_bytes} bytes]" + text
    return text


def read_text_if_within_limit(
    path: Optional[str | Path],
    *,
    max_bytes: int = DEFAULT_OUTPUT_PARSE_BYTES,
) -> Optional[str]:
    """Read an output file only when it is below the explicit memory guardrail."""

    if not path:
        return ""
    output_path = Path(path)
    if not output_path.exists() or not output_path.is_file():
        return ""
    if output_path.stat().st_size > max_bytes:
        return None
    return _decode_output(output_path.read_bytes())


def _path_size_bytes(path: Path, *, stop_after_bytes: Optional[int] = None) -> int:
    """Return recursive path size, optionally stopping after a guard threshold."""
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            if current.is_symlink():
                continue
            if current.is_file():
                total += current.stat().st_size
            elif current.is_dir():
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_file(follow_symlinks=False):
                                total += entry.stat(follow_symlinks=False).st_size
                            elif entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                        except OSError:
                            continue
            if stop_after_bytes is not None and total > stop_after_bytes:
                return total
        except OSError:
            continue
    return total


def _size_guard_delta(
    paths: Sequence[Path],
    baseline_sizes: dict[str, int],
    limit_bytes: int,
) -> tuple[bool, int, Optional[str]]:
    """Return whether monitored paths grew beyond an allowed byte delta."""
    total_delta = 0
    exceeded_path: Optional[str] = None
    for path in paths:
        key = str(path)
        current = _path_size_bytes(
            path,
            stop_after_bytes=baseline_sizes.get(key, 0) + max(0, limit_bytes - total_delta),
        )
        delta = max(0, current - baseline_sizes.get(key, 0))
        total_delta += delta
        if total_delta > limit_bytes:
            exceeded_path = key
            return True, total_delta, exceeded_path
    return False, total_delta, exceeded_path


def _terminate_process_tree(
    process: subprocess.Popen[bytes], *, grace_seconds: float = 5.0
) -> None:
    """Terminate a subprocess and descendants on Windows and POSIX."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        try:
            process.kill()
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    try:
        process.wait(timeout=grace_seconds)
    except Exception:
        pass


def run_command_file_backed(
    command: Sequence[str],
    *,
    cwd: Optional[str | Path],
    output_dir: str | Path,
    label: str,
    timeout_seconds: Optional[float] = None,
    preview_bytes: int = 4096,
    size_limit_paths: Optional[Sequence[str | Path]] = None,
    size_limit_bytes: Optional[int] = None,
    size_check_interval_seconds: float = 5.0,
) -> FileBackedProcessResult:
    """Run a command with stdout/stderr redirected to temp files.

    The caller gets only bounded previews in memory. Full output remains on disk
    until the containing snapshot/artifact directory is cleaned up.
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(label)
    stdout_fd, stdout_name = tempfile.mkstemp(
        prefix=f"{prefix}.stdout.",
        suffix=".log",
        dir=str(output_root),
    )
    stderr_fd, stderr_name = tempfile.mkstemp(
        prefix=f"{prefix}.stderr.",
        suffix=".log",
        dir=str(output_root),
    )
    stdout_path = Path(stdout_name)
    stderr_path = Path(stderr_name)
    cwd_text = str(cwd) if cwd is not None else None
    started = time.monotonic()
    returncode: Optional[int] = None
    timed_out = False
    size_limit_exceeded = False
    termination_reason: Optional[str] = None
    size_guard_bytes: Optional[int] = None
    size_guard_path: Optional[str] = None
    guard_paths = [Path(path) for path in (size_limit_paths or [])]
    size_limit = int(size_limit_bytes) if size_limit_bytes is not None else None
    baseline_sizes = (
        {str(path): _path_size_bytes(path) for path in guard_paths}
        if guard_paths and size_limit is not None and size_limit > 0
        else {}
    )
    try:
        with (
            os.fdopen(stdout_fd, "wb") as stdout_handle,
            os.fdopen(stderr_fd, "wb") as stderr_handle,
        ):
            creationflags = 0
            start_new_session = os.name != "nt"
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            process = subprocess.Popen(
                list(command),
                cwd=cwd_text,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=start_new_session,
                creationflags=creationflags,
            )
            deadline = (
                started + float(timeout_seconds)
                if timeout_seconds is not None and timeout_seconds > 0
                else None
            )
            next_size_check = started
            interval = max(0.25, float(size_check_interval_seconds or 5.0))
            while True:
                polled = process.poll()
                if polled is not None:
                    returncode = int(polled)
                    break
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    timed_out = True
                    termination_reason = f"timeout after {float(timeout_seconds):.1f}s"
                    _terminate_process_tree(process)
                    returncode = process.returncode
                    break
                if (
                    guard_paths
                    and size_limit is not None
                    and size_limit > 0
                    and now >= next_size_check
                ):
                    exceeded, observed, path = _size_guard_delta(
                        guard_paths,
                        baseline_sizes,
                        size_limit,
                    )
                    size_guard_bytes = observed
                    size_guard_path = path
                    if exceeded:
                        size_limit_exceeded = True
                        termination_reason = (
                            f"size limit exceeded after generating {observed} bytes "
                            f"(limit {size_limit} bytes)"
                        )
                        _terminate_process_tree(process)
                        returncode = process.returncode
                        break
                    next_size_check = now + interval
                sleep_for = 0.25
                if deadline is not None:
                    sleep_for = min(sleep_for, max(0.01, deadline - now))
                time.sleep(sleep_for)
            if termination_reason:
                stderr_handle.write(
                    f"\n[subprocess_output] terminated: {termination_reason}\n".encode(
                        "utf-8",
                        errors="replace",
                    )
                )
                stderr_handle.flush()
    except Exception:
        # File descriptors are owned by the context managers above once opened.
        raise
    elapsed = time.monotonic() - started
    return FileBackedProcessResult(
        command=[str(part) for part in command],
        cwd=cwd_text,
        returncode=returncode,
        timed_out=timed_out,
        elapsed_seconds=elapsed,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_preview=read_text_head(stdout_path, max_bytes=preview_bytes),
        stderr_preview=read_text_head(stderr_path, max_bytes=preview_bytes),
        stdout_tail=read_text_tail(stdout_path, max_bytes=preview_bytes),
        stderr_tail=read_text_tail(stderr_path, max_bytes=preview_bytes),
        size_limit_exceeded=size_limit_exceeded,
        termination_reason=termination_reason,
        size_limit_bytes=size_limit,
        size_guard_bytes=size_guard_bytes,
        size_guard_path=size_guard_path,
    )
