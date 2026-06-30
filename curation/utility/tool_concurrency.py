"""Process-wide semaphores for heavyweight external tools.

Repository workers run concurrently, but refactoring miners and source metric
tools can be CPU- and IO-heavy. These semaphores cap local tool fan-out without
changing the higher-level repository worker count.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, Semaphore
from typing import Dict, Iterator, Tuple

_LOCK = Lock()
_SEMAPHORES: Dict[Tuple[str, int], Semaphore] = {}


def _get_semaphore(name: str, limit: int) -> Semaphore:
    """Return the shared semaphore for one named tool/limit pair."""
    key = (name, limit)
    with _LOCK:
        sem = _SEMAPHORES.get(key)
        if sem is None:
            sem = Semaphore(limit)
            _SEMAPHORES[key] = sem
        return sem


@contextmanager
def tool_slot(name: str, limit: int) -> Iterator[None]:
    """Process-wide concurrency gate for heavy tools inside one container."""
    if limit <= 0:
        yield
        return
    sem = _get_semaphore(name, limit)
    sem.acquire()
    try:
        yield
    finally:
        sem.release()
