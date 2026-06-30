"""Context-local progress labels for long-running PR processing.

Worker threads set the current PR index before invoking hydration and metrics.
Nested modules can then prefix logs without passing progress counters through
every internal function signature.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional, Tuple

_CURRENT_PR_INDEX: ContextVar[Optional[int]] = ContextVar("current_pr_index", default=None)
_TOTAL_PRS: ContextVar[Optional[int]] = ContextVar("total_prs", default=None)


def set_current_pr_progress(current_index: int, total_prs: int) -> None:
    """Set the current worker-local PR progress marker."""
    _CURRENT_PR_INDEX.set(int(current_index))
    _TOTAL_PRS.set(int(total_prs))


def clear_current_pr_progress() -> None:
    """Clear the worker-local PR progress marker."""
    _CURRENT_PR_INDEX.set(None)
    _TOTAL_PRS.set(None)


def get_current_pr_progress() -> Optional[Tuple[int, int]]:
    """Return ``(current, total)`` for the active worker, if available."""
    current_index = _CURRENT_PR_INDEX.get()
    total_prs = _TOTAL_PRS.get()
    if current_index is None or total_prs is None or total_prs <= 0:
        return None
    return current_index, total_prs


def progress_suffix() -> str:
    """Return the short progress prefix used in logs."""
    progress = get_current_pr_progress()
    if not progress:
        return ""
    current_index, total_prs = progress
    return f"PR {current_index}/{total_prs}"


def with_pr_progress(message: str) -> str:
    """Prefix a log message with current PR progress when known."""
    prefix = progress_suffix()
    if not prefix:
        return message
    return f"{prefix} {message}"
