"""Shared JSON and JSONL object readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def _emit(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def read_json_object(
    path: Path,
    *,
    description: str = "JSON",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Read a JSON object from a file, returning None for malformed/non-object input."""
    resolved_path = Path(path)
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _emit(log, f"Skipping malformed {description} {resolved_path}: {exc}")
        return None
    if not isinstance(payload, dict):
        _emit(log, f"Skipping non-object {description} {resolved_path}")
        return None
    return payload


def iter_json_objects(
    paths: Iterable[Path],
    *,
    description: str = "JSON",
    log: Callable[[str], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield JSON object payloads one at a time from file paths."""
    for path in paths:
        payload = read_json_object(path, description=description, log=log)
        if payload is not None:
            yield payload


def iter_jsonl_offset_objects(
    record_refs: Iterable[object],
    *,
    description: str = "JSONL payload",
    log: Callable[[str], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield JSONL objects from refs with path, offset, and line_number attributes."""
    for record_ref in record_refs:
        path = Path(getattr(record_ref, "path"))
        line_number = int(getattr(record_ref, "line_number", 0) or 0)
        try:
            with path.open("rb") as handle:
                handle.seek(int(getattr(record_ref, "offset")))
                raw_line = handle.readline()
            payload = json.loads(raw_line.decode("utf-8"))
        except Exception as exc:
            _emit(
                log,
                f"Skipping malformed {description} {path}:{line_number}: {exc}",
            )
            continue
        if not isinstance(payload, dict):
            _emit(log, f"Skipping non-object {description} {path}:{line_number}")
            continue
        yield payload
