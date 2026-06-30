"""Custom duplicated-lines-density metrics.

The implementation is deterministic and language-aware enough for the four
curation languages. It counts non-comment source lines, normalizes whitespace,
and treats repeated normalized lines as duplicates.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from curation.metrics.multimetric_runner import discover_source_files

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".cc": "c++",
    ".cpp": "c++",
    ".cxx": "c++",
    ".hpp": "c++",
    ".hh": "c++",
    ".hxx": "c++",
}


def _language_for_path(path: Path) -> str | None:
    """Return the supported language token for a source file path."""
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower())


def _scan_c_like_line(line: str, in_block: bool) -> Tuple[bool, bool, bool]:
    """Return ``(has_code, has_comment, in_block_comment)`` for C-like syntax."""
    code_on_line = False
    comment_on_line = False
    quote: Optional[str] = None
    escaped = False
    i = 0
    length = len(line)
    while i < length:
        ch = line[i]
        nxt = line[i + 1] if i + 1 < length else ""
        if in_block:
            comment_on_line = True
            if ch == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == "/" and nxt == "/":
            comment_on_line = True
            break
        if ch == "/" and nxt == "*":
            comment_on_line = True
            in_block = True
            i += 2
            continue
        if not ch.isspace():
            code_on_line = True
        i += 1
    return code_on_line, comment_on_line, in_block


def _scan_python_line(line: str, in_triple: Optional[str]) -> Tuple[bool, bool, Optional[str]]:
    """Return ``(has_code, has_comment, triple_quote_delimiter)`` for Python."""
    code_on_line = False
    comment_on_line = False
    stripped = line.strip()
    if in_triple:
        comment_on_line = True
        end = line.find(in_triple)
        if end == -1:
            return code_on_line, comment_on_line, in_triple
        remainder = line[end + 3 :]
        extra_code, extra_comment, next_triple = _scan_python_line(remainder, None)
        return extra_code, (comment_on_line or extra_comment), next_triple

    docstring_match = re.match(r"^[rRuUbBfF]{0,2}(\"\"\"|''')", stripped)
    if docstring_match:
        delimiter = docstring_match.group(1)
        comment_on_line = True
        if stripped.count(delimiter) < 2:
            return False, True, delimiter
        return False, True, None

    quote: Optional[str] = None
    escaped = False
    i = 0
    length = len(line)
    while i < length:
        ch = line[i]
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == "#":
            comment_on_line = True
            break
        if not ch.isspace():
            code_on_line = True
        i += 1
    return code_on_line, comment_on_line, None


def compute_snapshot_duplication(snapshot_path: Path | None) -> Dict[str, float | int | str | None]:
    """Compute duplicated-lines density for one hydrated snapshot.

    The denominator is non-comment lines of code. Duplicate lines are counted as
    repeated normalized code lines beyond their first occurrence.
    """
    if snapshot_path is None or not snapshot_path.exists() or not snapshot_path.is_dir():
        return {
            "custom_duplication_status": "missing_snapshot",
            "custom_duplication_ncloc": 0,
            "duplicated_lines": 0,
            "duplicated_lines_density": 0.0,
        }

    ncloc = 0
    normalized_lines: Dict[str, int] = {}
    files_analyzed = 0
    for abs_path in discover_source_files(snapshot_path):
        language = _language_for_path(abs_path)
        if not language:
            continue
        files_analyzed += 1
        in_c_block_comment = False
        in_py_triple: Optional[str] = None
        for raw_line in abs_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if language == "python":
                code_on_line, _, in_py_triple = _scan_python_line(raw_line, in_py_triple)
            else:
                code_on_line, _, in_c_block_comment = _scan_c_like_line(
                    raw_line,
                    in_c_block_comment,
                )
            if not code_on_line:
                continue
            # Normalize spacing only; identifiers and literals remain intact so
            # the metric is deterministic and easy to audit.
            normalized = re.sub(r"\s+", " ", stripped)
            normalized_lines[normalized] = normalized_lines.get(normalized, 0) + 1
            ncloc += 1

    duplicated_lines = sum(count - 1 for count in normalized_lines.values() if count > 1)
    density = 0.0 if ncloc <= 0 else (float(duplicated_lines) / float(ncloc)) * 100.0
    return {
        "custom_duplication_status": "success",
        "custom_duplication_ncloc": int(ncloc),
        "custom_duplication_files_analyzed": int(files_analyzed),
        "duplicated_lines": int(duplicated_lines),
        "duplicated_lines_density": float(density),
    }
