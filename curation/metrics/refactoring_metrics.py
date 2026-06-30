"""Refactoring mining metrics for processed PRs.

The active curation pipeline mines refactorings only for the original PR
before/after comparison. Future snapshots are used to track persistence of the
original refactoring zones; they are not passed back through refactoring tools.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from curation.config.refactoring_config import (
    FUTURE_REFACTORING_SNAPSHOT_LABELS,
    REFACTORING_MINER_CONFIG,
    REFACTORING_MINER_PP_CAP_TIMEOUT_SECONDS,
    REFACTORING_MINER_PP_CONFIG,
    REFFDIFF_CONFIG,
    RefactoringToolConfig,
)
from curation.config.refactoring_standardization_config import (
    canonicalize_refactoring_type,
    is_non_refactoring_relationship_type,
)
from curation.config.run_config import (
    CPP_TOOL_CONCURRENCY_LIMIT,
    EXTERNAL_TOOL_CONCURRENCY_LIMIT,
)
from curation.config.refactoring_taxonomy_config import (
    classify_refactoring_taxonomy,
)
from curation.hydration.repository_hydrator import RepositoryHydrator
from curation.pipeline.progress_context import with_pr_progress
from curation.utility.language_selection import dominant_pr_language
from curation.utility.subprocess_output import (
    DEFAULT_OUTPUT_PARSE_BYTES,
    read_text_if_within_limit,
    run_command_file_backed,
)
from curation.utility.tool_concurrency import tool_slot

LOG_PREFIX = "[refactoring_metrics]"
logger = logging.getLogger(__name__)

_CURATION_ROOT = Path(__file__).resolve().parents[1]
_RMPP_RUNTIME_DIR = _CURATION_ROOT / "tools" / "runtime" / "refactoringminerpp"
_RMPP_BIN = _RMPP_RUNTIME_DIR / "bin" / "RefactoringMiner"
_RMPP_CAP_BIN = _RMPP_RUNTIME_DIR / "bin" / "cap"
_CPP_EXTENSIONS = {
    ".c",
    ".cc",
    ".cp",
    ".cpp",
    ".cxx",
    ".c++",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".h++",
}


def _run_external_tool_file_backed(command: List[str], **kwargs: Any):
    """Run an external analyzer under the shared tool concurrency gate."""
    with tool_slot("external_tools", max(0, int(EXTERNAL_TOOL_CONCURRENCY_LIMIT))):
        return run_command_file_backed(command, **kwargs)


def _get_pr_number(pr: Any) -> Any:
    """Return the PR number from either a DTO-like object or a dict."""
    return getattr(pr, "number", None) if not isinstance(pr, dict) else pr.get("number")


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from either a DTO-like object or a dict."""
    return getattr(obj, name, default) if not isinstance(obj, dict) else obj.get(name, default)


def _normalize_language(language: Optional[str]) -> Optional[str]:
    """Normalize language labels used for refactoring-tool selection."""
    if not language:
        return None
    normalized = str(language).strip().lower()
    aliases = {
        "javascript": "javascript",
        "js": "javascript",
        "python": "python",
        "java": "java",
        "c++": "c++",
        "cpp": "c++",
    }
    return aliases.get(normalized, normalized)


def _select_pr_language(pr: Any) -> Optional[str]:
    """Choose dominant PR language for tool selection."""
    effective = _normalize_language(_get_attr(pr, "pr_primary_language_effective"))
    if effective in {"java", "python", "javascript", "c++"}:
        return effective
    primary = _normalize_language(_get_attr(pr, "primary_language"))
    if primary in {"java", "python", "javascript", "c++"}:
        return primary
    return dominant_pr_language(
        pr,
        supported_languages=("java", "python", "javascript", "c++"),
        tie_break_priority=("c++", "java", "javascript", "python"),
    )


def _log_info(message: str) -> None:
    """Emit high-level progress for refactoring mining."""
    logger.info(message)
    print(with_pr_progress(f"{LOG_PREFIX} {message}"))


def _log_error(message: str) -> None:
    """Emit failure-level progress for refactoring mining."""
    logger.error(message)
    print(with_pr_progress(f"{LOG_PREFIX} ERROR: {message}"))


def _is_executable_file(path: Path) -> bool:
    """Return True when a path is an executable file."""
    return path.is_file() and os.access(path, os.X_OK)


def _ensure_refactoringminerpp_runtime() -> None:
    """
    Ensure RefactoringMiner++ exists at tools/runtime/refactoringminerpp.

    Runtime bootstrap is intentionally disabled: Docker/image setup must preinstall
    RM++ so C++ runs do not incur first-PR install latency.
    """
    if not _RMPP_BIN.exists():
        raise RuntimeError(
            f"RefactoringMiner++ runtime not found at {_RMPP_BIN}. "
            "Build/update the Docker image to preinstall RM++."
        )
    if not _is_executable_file(_RMPP_BIN):
        try:
            _RMPP_BIN.chmod(_RMPP_BIN.stat().st_mode | 0o755)
        except Exception as exc:
            raise RuntimeError(
                f"RefactoringMiner++ binary exists but is not executable: {_RMPP_BIN} ({exc})"
            ) from exc
        if not _is_executable_file(_RMPP_BIN):
            raise RuntimeError(
                f"RefactoringMiner++ binary exists but remains non-executable: {_RMPP_BIN}"
            )
    if not _RMPP_CAP_BIN.exists():
        raise RuntimeError(
            f"RefactoringMiner++ CAP binary not found at {_RMPP_CAP_BIN}. "
            "Build/update the Docker image to preinstall CAP."
        )
    if not _is_executable_file(_RMPP_CAP_BIN):
        try:
            _RMPP_CAP_BIN.chmod(_RMPP_CAP_BIN.stat().st_mode | 0o755)
        except Exception as exc:
            raise RuntimeError(
                f"RefactoringMiner++ CAP binary exists but is not executable: {_RMPP_CAP_BIN} ({exc})"
            ) from exc
    if not _is_executable_file(_RMPP_CAP_BIN):
        raise RuntimeError(
            f"RefactoringMiner++ CAP binary exists but remains non-executable: {_RMPP_CAP_BIN}"
        )


def _is_cpp_path(path: Optional[str]) -> bool:
    if not path:
        return False
    return Path(str(path)).suffix.lower() in _CPP_EXTENSIONS


def _extract_cpp_file_pairs(
    pr: Any,
    *,
    use_previous_filenames: bool = True,
) -> List[Dict[str, str]]:
    """
    Build before/after C++ file pairs from PR file metadata.

    Uses `previous_filename -> path` for PR renames and `path -> path` otherwise.
    Future snapshot comparisons are post-merge transitions, so they compare the
    PR-after file path against the same file path in the future snapshot.
    """
    files = _get_attr(pr, "files") or []
    pairs: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in files:
        path = _get_attr(item, "path") or _get_attr(item, "filename") or _get_attr(item, "file")
        prev = _get_attr(item, "previous_filename")
        after_path = str(path) if path else ""
        before_path = (
            str(prev)
            if use_previous_filenames and prev
            else (str(path) if path else "")
        )
        if not _is_cpp_path(before_path) and not _is_cpp_path(after_path):
            continue
        if not before_path or not after_path:
            continue
        key = (before_path, after_path)
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"before_path": before_path, "after_path": after_path})
    return pairs


def _extract_rmpp_text_operations(
    stdout: str,
    *,
    tool_name: str,
    snapshot_label: str,
    snapshot_commit: Optional[str],
    language: Optional[str],
) -> List[Dict[str, Any]]:
    """Parse textual RM++ `-cpp` output when JSON output is unavailable."""
    if not stdout:
        return []

    operations: List[Dict[str, Any]] = []
    in_ref_section = False
    current_type: Optional[str] = None
    current_desc: Optional[str] = None

    def _flush() -> None:
        nonlocal current_type, current_desc
        if not current_type and not current_desc:
            return
        operations.append(
            _standardized_operation(
                tool_name=tool_name,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                language=language,
                operation_index=len(operations),
                raw_type=current_type,
                description=current_desc,
                source_locations=[],
                target_locations=[],
                raw_operation={
                    "type": current_type,
                    "description": current_desc,
                    "source": "rmpp_stdout",
                },
                commit_id=snapshot_commit,
            )
        )
        current_type = None
        current_desc = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if "detected refactorings" in lower:
            in_ref_section = True
            continue
        if not in_ref_section:
            continue
        if "detected functionality changes" in lower or lower.startswith("summary"):
            break
        if not line:
            continue
        if line.endswith(")") and line[:-1].isdigit():
            _flush()
            continue
        if line.lower().startswith("the following refactorings were detected"):
            continue
        if line.lower().startswith("the following lines of the current version"):
            continue
        if line.lower().startswith("lines changed in total"):
            continue
        if line.startswith("="):
            continue
        if line.isdigit():
            continue
        if current_type is None:
            if "\t" in line:
                left, _, right = line.partition("\t")
                current_type = left.strip() or None
                current_desc = right.strip() or line
            elif "  " in line:
                left, right = line.split("  ", 1)
                current_type = left.strip() or None
                current_desc = right.strip() or line
            else:
                current_type = line
                current_desc = line
            continue
        current_desc = f"{current_desc or ''} {line}".strip()

    _flush()
    return operations


def _compact_text(value: Any, *, max_chars: int = 1200) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated {len(text) - max_chars} chars]"


def _compact_command(command: Optional[List[str]]) -> Optional[str]:
    if not command:
        return None
    return _compact_text(" ".join(str(part) for part in command), max_chars=2000)


def _strip_known_j2v8_warnings(stderr_text: str) -> str:
    """Remove known non-fatal J2V8 executable-stack JVM warnings from stderr text."""
    if not stderr_text:
        return ""
    filtered_lines: List[str] = []
    for line in stderr_text.splitlines():
        lowered = line.strip().lower()
        if (
            "openjdk 64-bit server vm warning" in lowered
            and "libj2v8" in lowered
        ):
            continue
        if "the vm will try to fix the stack guard now." in lowered:
            continue
        if "it's highly recommended that you fix the library with 'execstack -c <libfile>'" in lowered:
            continue
        if "or link it with '-z noexecstack'." in lowered:
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def _best_effort_prepare_j2v8_for_refdiff() -> None:
    """
    Best-effort Linux prep for ReffDiff's J2V8 native lib.

    Some container hosts reject loading ELF objects that request executable stack.
    We (re-)extract and clear the exec-stack flag right before ReffDiff runs.
    """
    if os.name == "nt":
        return
    lib_path = Path("/root/libj2v8_linux_x86_64.so")
    jar_path = Path("/curation/curation/tools/runtime/refdiff/lib/j2v8_linux_x86_64-4.6.0.jar")

    # Re-extract on each run to handle cases where the runtime loader overwrites the file.
    if jar_path.exists():
        subprocess.run(
            [
                "unzip",
                "-j",
                str(jar_path),
                "libj2v8_linux_x86_64.so",
                "-d",
                "/root",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    if not lib_path.exists():
        return

    # Prefer patchelf when available; fall back to execstack if present.
    if shutil.which("patchelf"):
        subprocess.run(
            ["patchelf", "--clear-execstack", str(lib_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    if shutil.which("execstack"):
        subprocess.run(
            ["execstack", "-c", str(lib_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )


def _is_js_parser_plugin_failure(stderr_text: Optional[str]) -> bool:
    """Return True for known RefDiff/Babel parser-plugin limitations on modern JS syntax."""
    text = str(stderr_text or "").lower()
    if "requires enabling the parser plugin" not in text:
        return False
    return any(
        token in text
        for token in (
            "optionalchaining",
            "nullishcoalescingoperator",
            "importmeta",
        )
    )


def _normalize_refactoring_type(raw_type: Optional[str]) -> Optional[str]:
    """Map tool-specific type labels into a shared operation type vocabulary."""
    return canonicalize_refactoring_type(raw_type)


def _normalize_location(location: Any) -> Optional[Dict[str, Any]]:
    """Normalize tool-specific location dictionaries into one shared schema."""
    if not isinstance(location, dict):
        return None
    return {
        "file_path": location.get("filePath")
        or location.get("file_path")
        or location.get("file"),
        "start_line": location.get("startLine")
        or location.get("start_line")
        or location.get("beginLine")
        or location.get("line"),
        "end_line": location.get("endLine")
        or location.get("end_line")
        or location.get("finishLine")
        or location.get("line"),
        "start_column": location.get("startColumn")
        or location.get("start_column")
        or location.get("beginColumn")
        or location.get("column"),
        "end_column": location.get("endColumn")
        or location.get("end_column")
        or location.get("finishColumn")
        or location.get("column"),
        "code_element_type": location.get("codeElementType")
        or location.get("code_element_type")
        or location.get("nodeType")
        or location.get("type"),
        "role": location.get("description") or location.get("role"),
        "code_element": location.get("codeElement")
        or location.get("code_element")
        or location.get("name")
        or location.get("signature"),
    }


def _normalized_locations(items: Any) -> List[Dict[str, Any]]:
    """Normalize and filter a list of location-like objects."""
    normalized: List[Dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        location = _normalize_location(item)
        if location:
            normalized.append(location)
    return normalized


def _compact_location_for_artifact(location: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_path": location.get("file_path"),
        "start_line": location.get("start_line"),
        "end_line": location.get("end_line"),
        "start_column": location.get("start_column"),
        "end_column": location.get("end_column"),
        "code_element_type": location.get("code_element_type"),
        "role": location.get("role"),
        "code_element": _compact_text(location.get("code_element"), max_chars=240),
    }


def _compact_operation_for_artifact(operation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "operation_id": operation.get("operation_id"),
        "tool": operation.get("tool"),
        "language": operation.get("language"),
        "snapshot_label": operation.get("snapshot_label"),
        "snapshot_commit": operation.get("snapshot_commit"),
        "commit_id": operation.get("commit_id"),
        "raw_type": operation.get("raw_type"),
        "standardized_type": operation.get("standardized_type"),
        "description": _compact_text(operation.get("description"), max_chars=500),
        "murphy_hill_level": operation.get("murphy_hill_level"),
        "taxonomy": operation.get("taxonomy"),
        "source_locations": [
            _compact_location_for_artifact(loc)
            for loc in (operation.get("source_locations") or [])
            if isinstance(loc, dict)
        ],
        "target_locations": [
            _compact_location_for_artifact(loc)
            for loc in (operation.get("target_locations") or [])
            if isinstance(loc, dict)
        ],
    }


def _standardized_operation(
    *,
    tool_name: str,
    snapshot_label: str,
    snapshot_commit: Optional[str],
    language: Optional[str],
    operation_index: int,
    raw_type: Optional[str],
    description: Optional[str],
    source_locations: List[Dict[str, Any]],
    target_locations: List[Dict[str, Any]],
    raw_operation: Any,
    commit_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create one normalized refactoring operation entry."""
    standardized_type = _normalize_refactoring_type(raw_type)
    ref_taxonomy = classify_refactoring_taxonomy(standardized_type)
    murphy_hill_level = ref_taxonomy.get("murphy_hill_level")
    return {
        "operation_id": f"{snapshot_label}:{operation_index}",
        "tool": tool_name,
        "language": language,
        "snapshot_label": snapshot_label,
        "snapshot_commit": snapshot_commit,
        "commit_id": commit_id,
        "raw_type": raw_type,
        "standardized_type": standardized_type,
        "description": description,
        "murphy_hill_level": murphy_hill_level,
        "taxonomy": ref_taxonomy,
        "source_locations": source_locations,
        "target_locations": target_locations,
        "raw_operation": raw_operation,
    }


def _extract_refactoringminer_operations(
    payload: Any,
    *,
    tool_name: str,
    snapshot_label: str,
    snapshot_commit: Optional[str],
    language: Optional[str],
) -> List[Dict[str, Any]]:
    """Parse RefactoringMiner-style JSON into standardized operations."""
    if payload is None:
        return []

    commit_entries: List[Dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("commits"), list):
        commit_entries = [entry for entry in payload["commits"] if isinstance(entry, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("refactorings"), list):
        commit_entries = [payload]
    elif isinstance(payload, list):
        commit_entries = [{"refactorings": payload}]

    operations: List[Dict[str, Any]] = []
    for commit_entry in commit_entries:
        commit_id = (
            commit_entry.get("sha1")
            or commit_entry.get("commitId")
            or commit_entry.get("commit")
            or snapshot_commit
        )
        for refactoring in commit_entry.get("refactorings") or []:
            if not isinstance(refactoring, dict):
                continue
            operation_index = len(operations)
            operations.append(
                _standardized_operation(
                    tool_name=tool_name,
                    snapshot_label=snapshot_label,
                    snapshot_commit=snapshot_commit,
                    language=language,
                    operation_index=operation_index,
                    raw_type=refactoring.get("type"),
                    description=refactoring.get("description"),
                    source_locations=_normalized_locations(
                        refactoring.get("leftSideLocations")
                        or refactoring.get("left")
                        or refactoring.get("beforeLocations")
                    ),
                    target_locations=_normalized_locations(
                        refactoring.get("rightSideLocations")
                        or refactoring.get("right")
                        or refactoring.get("afterLocations")
                    ),
                    raw_operation=refactoring,
                    commit_id=commit_id,
                )
            )
    return operations


def _extract_refdiff_operations_from_json(
    payload: Any,
    *,
    tool_name: str,
    snapshot_label: str,
    snapshot_commit: Optional[str],
    language: Optional[str],
) -> List[Dict[str, Any]]:
    """Parse a RefDiff-style JSON payload or a compatible wrapper output."""
    if payload is None:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("commits"), list):
        return _extract_refactoringminer_operations(
            payload,
            tool_name=tool_name,
            snapshot_label=snapshot_label,
            snapshot_commit=snapshot_commit,
            language=language,
        )

    if isinstance(payload, dict):
        candidates = payload.get("relationships")
        if not isinstance(candidates, list):
            candidates = payload.get("refactorings")
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = None

    operations: List[Dict[str, Any]] = []
    for rel in candidates if isinstance(candidates, list) else []:
        if not isinstance(rel, dict):
            continue
        raw_type = (
            rel.get("type")
            or rel.get("relationshipType")
            or rel.get("relationship_type")
        )
        if is_non_refactoring_relationship_type(raw_type):
            continue
        before_locations = _normalized_locations(
            rel.get("leftSideLocations")
            or rel.get("beforeLocations")
            or ([rel.get("nodeBefore")] if isinstance(rel.get("nodeBefore"), dict) else [])
        )
        after_locations = _normalized_locations(
            rel.get("rightSideLocations")
            or rel.get("afterLocations")
            or ([rel.get("nodeAfter")] if isinstance(rel.get("nodeAfter"), dict) else [])
        )
        operations.append(
            _standardized_operation(
                tool_name=tool_name,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                language=language,
                operation_index=len(operations),
                raw_type=raw_type,
                description=rel.get("description")
                or rel.get("standardDescription")
                or rel.get("standard_description"),
                source_locations=before_locations,
                target_locations=after_locations,
                raw_operation=rel,
                commit_id=rel.get("commitId") or rel.get("commit") or snapshot_commit,
            )
        )
    return operations


def _extract_refdiff_operations_from_stdout(
    stdout: str,
    *,
    tool_name: str,
    snapshot_label: str,
    snapshot_commit: Optional[str],
    language: Optional[str],
) -> List[Dict[str, Any]]:
    """Parse plain-text RefDiff output into standardized operations."""
    operations: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        description = line.strip()
        if not description or description.startswith("Refactorings found in "):
            continue
        raw_type = None
        for candidate in (
            "Change Signature of Method/Function",
            "Extract Method/Function",
            "Inline Method/Function",
            "Extract Supertype",
            "Move and Rename",
            "Pull Up Method",
            "Push Down Method",
            "Convert Type",
            "Rename",
            "Move",
        ):
            if description.startswith(candidate):
                raw_type = candidate
                break
        operations.append(
            _standardized_operation(
                tool_name=tool_name,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                language=language,
                operation_index=len(operations),
                raw_type=raw_type,
                description=description,
                source_locations=[],
                target_locations=[],
                raw_operation={"description": description},
                commit_id=snapshot_commit,
            )
        )
    return operations


def _count_by_field(
    operations: List[Dict[str, Any]],
    field_name: str,
) -> Dict[str, int]:
    """Count standardized operations by a given taxonomy field."""
    counts: Dict[str, int] = {}
    for operation in operations:
        value = operation.get(field_name)
        key = str(value) if value else "unclassified"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _line_span(location: Dict[str, Any]) -> range:
    """Return the inclusive line span for one normalized location."""
    start_line = location.get("start_line")
    end_line = location.get("end_line")
    if start_line is None and end_line is None:
        return range(0, 0)
    start = int(start_line if start_line is not None else end_line)
    end = int(end_line if end_line is not None else start_line)
    if end < start:
        start, end = end, start
    return range(start, end + 1)


def _covered_lines_by_file(
    operations: List[Dict[str, Any]],
    location_field: str,
) -> Dict[str, set[int]]:
    """Collect unique covered lines per file for one operation side."""
    covered: Dict[str, set[int]] = {}
    for operation in operations:
        for location in operation.get(location_field, []):
            file_path = location.get("file_path")
            if not file_path:
                continue
            covered.setdefault(str(file_path), set()).update(_line_span(location))
    return covered


def _distinct_file_count(operations: List[Dict[str, Any]]) -> int:
    """Count distinct files touched by standardized operations."""
    file_paths: set[str] = set()
    for operation in operations:
        for location_field in ("source_locations", "target_locations"):
            for location in operation.get(location_field, []):
                file_path = location.get("file_path")
                if file_path:
                    file_paths.add(str(file_path))
    return len(file_paths)


def _shannon_diversity(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy over operation-type counts."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        proportion = count / total
        entropy -= proportion * math.log2(proportion)
    return entropy


def _read_text_cached(path: Path, cache: Dict[str, Optional[str]]) -> Optional[str]:
    """Read a file once and cache its contents for repeated retention checks."""
    key = str(path)
    if key in cache:
        return cache[key]
    try:
        cache[key] = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        cache[key] = None
    return cache[key]


def _trackable_locations(operation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the best available locations for retention tracking."""
    locations = operation.get("target_locations") or operation.get("source_locations") or []
    return [location for location in locations if location.get("file_path")]


def _future_overlap_locations(operation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return only after-side locations that can be compared to after->future diffs."""
    locations = operation.get("target_locations") or []
    return [location for location in locations if location.get("file_path")]


def _merge_line_ranges(ranges: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    normalized: List[Dict[str, int]] = []
    for line_range in ranges:
        range_start = int(line_range.get("start", 0))
        range_end = int(line_range.get("end", range_start - 1))
        if range_end < range_start:
            continue
        normalized.append({"start": range_start, "end": range_end})
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    merged: List[Dict[str, int]] = []
    for line_range in normalized:
        if not merged or line_range["start"] > merged[-1]["end"] + 1:
            merged.append(dict(line_range))
            continue
        merged[-1]["end"] = max(merged[-1]["end"], line_range["end"])
    return merged


def _line_range_overlap_lines(
    location_start: int,
    location_end: int,
    ranges: List[Dict[str, Any]],
) -> set[int]:
    merged_ranges = _merge_line_ranges(ranges)
    overlap: set[int] = set()
    for line_range in merged_ranges:
        range_start = int(line_range.get("start", 0))
        range_end = int(line_range.get("end", range_start - 1))
        start = max(location_start, range_start)
        end = min(location_end, range_end)
        if end >= start:
            overlap.update(range(start, end + 1))
    return overlap


def _operation_future_diff_overlap_lines_by_file(
    operation: Dict[str, Any],
    future_diff: Dict[str, Any],
) -> Dict[str, set[int]]:
    old_ranges_by_file = future_diff.get("old_line_ranges_by_file") or {}
    touched_lines_by_file: Dict[str, set[int]] = {}
    for location in _future_overlap_locations(operation):
        file_path = location.get("file_path")
        if not file_path or file_path not in old_ranges_by_file:
            continue
        location_span = _line_span(location)
        if not location_span:
            continue
        start = location_span.start
        end = location_span.stop - 1
        if end < start:
            continue
        overlap = _line_range_overlap_lines(
            start,
            end,
            old_ranges_by_file.get(str(file_path), []),
        )
        if overlap:
            touched_lines_by_file.setdefault(str(file_path), set()).update(overlap)
    return touched_lines_by_file


def _operation_future_diff_overlap(
    operation: Dict[str, Any],
    future_diff: Dict[str, Any],
) -> Dict[str, Any]:
    """Return whether a PR operation overlaps the future diff on the after-side lines."""
    old_ranges_by_file = future_diff.get("old_line_ranges_by_file") or {}
    locations = _future_overlap_locations(operation)
    if not locations:
        return {
            "touched_files": [],
            "touched_file_count": 0,
            "overlapping_line_count": 0,
            "has_file_overlap": False,
            "has_line_overlap": False,
            "is_trackable": False,
        }
    touched_files: List[str] = []
    overlapping_lines = 0
    for location in locations:
        file_path = location.get("file_path")
        if not file_path or file_path not in old_ranges_by_file:
            continue
        touched_files.append(str(file_path))
    touched_lines_by_file = _operation_future_diff_overlap_lines_by_file(operation, future_diff)
    overlapping_lines = sum(len(lines) for lines in touched_lines_by_file.values())
    unique_touched_files = sorted(set(touched_files))
    return {
        "touched_files": unique_touched_files,
        "touched_file_count": len(unique_touched_files),
        "overlapping_line_count": overlapping_lines,
        "overlapping_line_counts_by_file": {
            path: len(lines) for path, lines in sorted(touched_lines_by_file.items())
        },
        "has_file_overlap": bool(unique_touched_files),
        "has_line_overlap": overlapping_lines > 0,
        "is_trackable": True,
    }


def _evaluate_operation_retention(
    operation: Dict[str, Any],
    snapshot_label: str,
    snapshot_meta: Dict[str, Any],
    file_cache: Dict[str, Optional[str]],
    future_diff: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate whether a PR refactoring operation is retained in a future snapshot.

    Heuristic:
    - retained: at least one location's code element is still present in the future file
    - weakly_retained: relevant file still exists, but code element string was not matched
    - not_retained: no tracked files remain
    - unknown: snapshot or trackable locations are missing
    """
    snapshot_path = snapshot_meta.get("path")
    snapshot_commit = snapshot_meta.get("commit")
    if not snapshot_path or not snapshot_commit:
        return {
            "status": "unknown",
            "snapshot_label": snapshot_label,
            "snapshot_commit": snapshot_commit,
            "reason": "missing_snapshot",
            "checked_locations": 0,
            "file_matches": 0,
            "exact_matches": 0,
            "matched_locations": [],
        }

    locations = _trackable_locations(operation)
    if not locations:
        return {
            "status": "unknown",
            "snapshot_label": snapshot_label,
            "snapshot_commit": snapshot_commit,
            "reason": "no_trackable_locations",
            "checked_locations": 0,
            "file_matches": 0,
            "exact_matches": 0,
            "matched_locations": [],
        }

    checked_locations = 0
    file_matches = 0
    exact_matches = 0
    matched_locations: List[Dict[str, Any]] = []

    for location in locations:
        file_path = str(location.get("file_path"))
        code_element = (location.get("code_element") or "").strip()
        snapshot_file = Path(snapshot_path) / Path(file_path)
        checked_locations += 1

        file_exists = snapshot_file.exists()
        exact_match = False
        if file_exists:
            file_matches += 1
            if code_element:
                text = _read_text_cached(snapshot_file, file_cache)
                if text and code_element in text:
                    exact_match = True
                    exact_matches += 1

        matched_locations.append(
            {
                "file_path": file_path,
                "code_element": code_element or None,
                "file_exists": file_exists,
                "exact_match": exact_match,
            }
        )

    if exact_matches > 0:
        status = "retained"
    elif file_matches > 0:
        status = "weakly_retained"
    else:
        status = "not_retained"

    diff_overlap = _operation_future_diff_overlap(operation, future_diff or {})

    return {
        "status": status,
        "snapshot_label": snapshot_label,
        "snapshot_commit": snapshot_commit,
        "reason": None,
        "checked_locations": checked_locations,
        "file_matches": file_matches,
        "exact_matches": exact_matches,
        "matched_locations": matched_locations,
        "future_diff_overlap": diff_overlap,
    }


def _future_impact_summary(hydration: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize future diff overlap with the PR's changed files and lines."""
    future_tracking = (hydration.get("diff_tracking") or {}).get("future") or {}
    future_commit_activity = (
        (hydration.get("longitudinal_commit_activity") or {}).get("future_snapshot_intervals") or {}
    )
    per_snapshot: Dict[str, Dict[str, Any]] = {}
    snapshots_with_file_overlap = 0
    snapshots_with_line_overlap = 0
    total_touched_pr_lines = 0
    total_touching_commits = 0
    coverage_values: List[float] = []
    for label in FUTURE_REFACTORING_SNAPSHOT_LABELS:
        tracking = future_tracking.get(label)
        if not isinstance(tracking, dict):
            continue
        overlap = tracking.get("overlap_with_pr") or {}
        commit_activity = future_commit_activity.get(label) or {}
        commit_activity_reliable = (
            commit_activity.get("commit_activity_status") == "ok"
            and commit_activity.get("ancestry_status") is True
            and commit_activity.get("touching_commits_count") is not None
        )
        touching_commits_count = (
            int(commit_activity.get("touching_commits_count"))
            if commit_activity_reliable
            else None
        )
        touch_coverage_pct = (
            float(commit_activity.get("touch_coverage_pct", 0.0) or 0.0)
            if commit_activity_reliable
            else None
        )
        summary = {
            "changed_files_count": tracking.get("changed_files_count", 0),
            "changed_paths": tracking.get("changed_paths") or [],
            "commit_activity_status": commit_activity.get("commit_activity_status"),
            "commit_activity_reliable": commit_activity_reliable,
            "touching_commits_count": touching_commits_count,
            "touched_pr_files_count": overlap.get("touched_pr_files_count", 0),
            "touched_pr_files": overlap.get("touched_pr_files") or [],
            "touch_coverage_pct": touch_coverage_pct,
            "touched_pr_lines_count": overlap.get("touched_pr_lines_count", 0),
            "touched_pr_line_files": overlap.get("touched_pr_line_files") or [],
            "has_file_overlap": bool(overlap.get("has_file_overlap")),
            "has_line_overlap": bool(overlap.get("has_line_overlap")),
        }
        per_snapshot[label] = summary
        if summary["has_file_overlap"]:
            snapshots_with_file_overlap += 1
        if summary["has_line_overlap"]:
            snapshots_with_line_overlap += 1
        total_touched_pr_lines += int(summary["touched_pr_lines_count"])
        if touching_commits_count is not None:
            total_touching_commits += touching_commits_count
        if touch_coverage_pct is not None:
            coverage_values.append(touch_coverage_pct)
    mean_touch_coverage_pct = (
        sum(coverage_values) / float(len(coverage_values)) if coverage_values else 0.0
    )
    return {
        "per_snapshot": per_snapshot,
        "snapshots_with_file_overlap": snapshots_with_file_overlap,
        "snapshots_with_line_overlap": snapshots_with_line_overlap,
        "total_touched_pr_lines": total_touched_pr_lines,
        "total_touching_commits": total_touching_commits,
        "mean_touch_coverage_pct": mean_touch_coverage_pct,
    }


def _refactoring_zone_future_impact_summary(
    retention_summary: Dict[str, Any],
    pr_future_impact: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize future diff overlap with original PR refactoring target locations."""
    retention_by_snapshot = retention_summary.get("per_snapshot") or {}
    pr_impact_by_snapshot = pr_future_impact.get("per_snapshot") or {}
    per_snapshot: Dict[str, Dict[str, Any]] = {}
    snapshots_with_file_overlap = 0
    snapshots_with_line_overlap = 0
    total_touched_refactoring_lines = 0
    total_operations_with_future_file_overlap = 0
    total_operations_with_future_line_overlap = 0

    for label in FUTURE_REFACTORING_SNAPSHOT_LABELS:
        retention = retention_by_snapshot.get(label)
        if not isinstance(retention, dict):
            continue
        pr_impact = (
            pr_impact_by_snapshot.get(label)
            if isinstance(pr_impact_by_snapshot.get(label), dict)
            else {}
        )
        zone_line_count = int(retention.get("refactoring_zone_lines", 0) or 0)
        touched_line_count = int(retention.get("touched_refactoring_zone_lines", 0) or 0)
        operations_with_file_overlap = int(
            retention.get(
                "operations_with_future_file_overlap",
                retention.get("operations_touched_by_future_diff", 0),
            )
            or 0
        )
        operations_with_line_overlap = int(
            retention.get("operations_with_future_line_overlap", 0) or 0
        )
        has_file_overlap = operations_with_file_overlap > 0
        has_line_overlap = touched_line_count > 0
        if has_file_overlap:
            snapshots_with_file_overlap += 1
        if has_line_overlap:
            snapshots_with_line_overlap += 1
        total_touched_refactoring_lines += touched_line_count
        total_operations_with_future_file_overlap += operations_with_file_overlap
        total_operations_with_future_line_overlap += operations_with_line_overlap
        per_snapshot[label] = {
            "snapshot_commit": retention.get("snapshot_commit"),
            "tracked_refactoring_operations": int(
                retention.get("trackable_refactoring_operations", 0) or 0
            ),
            "refactoring_zone_lines": zone_line_count,
            "operations_with_future_file_overlap": operations_with_file_overlap,
            "operations_with_future_line_overlap": operations_with_line_overlap,
            "touched_refactoring_zone_lines_count": touched_line_count,
            "touched_refactoring_zone_line_files": (
                retention.get("touched_refactoring_zone_line_files") or []
            ),
            "touched_refactoring_zone_line_counts_by_file": (
                retention.get("touched_refactoring_zone_line_counts_by_file") or {}
            ),
            "refactoring_zone_touch_coverage_pct": (
                (100.0 * touched_line_count / zone_line_count)
                if zone_line_count > 0
                else None
            ),
            "has_file_overlap": has_file_overlap,
            "has_line_overlap": has_line_overlap,
            "pr_changed_line_context": pr_impact,
        }

    return {
        "per_snapshot": per_snapshot,
        "snapshots_with_file_overlap": snapshots_with_file_overlap,
        "snapshots_with_line_overlap": snapshots_with_line_overlap,
        "total_touched_refactoring_zone_lines": total_touched_refactoring_lines,
        "total_operations_with_future_file_overlap": total_operations_with_future_file_overlap,
        "total_operations_with_future_line_overlap": total_operations_with_future_line_overlap,
    }


def _compute_retention_summary(
    operations: List[Dict[str, Any]],
    hydration: Dict[str, Any],
) -> Dict[str, Any]:
    """Annotate PR operations with future-snapshot retention and aggregate the results."""
    future_snapshots = (hydration.get("snapshots") or {}).get("future") or {}
    future_diff_tracking = (hydration.get("diff_tracking") or {}).get("future") or {}
    pr_operations = [operation for operation in operations if operation.get("snapshot_label") == "pr"]
    refactoring_zone_lines_by_file = _covered_lines_by_file(pr_operations, "target_locations")
    refactoring_zone_line_count = sum(
        len(lines) for lines in refactoring_zone_lines_by_file.values()
    )
    trackable_refactoring_operations = sum(
        1 for operation in pr_operations if _future_overlap_locations(operation)
    )
    file_cache: Dict[str, Optional[str]] = {}
    retention_by_snapshot: Dict[str, Dict[str, Any]] = {}
    latest_snapshot_label: Optional[str] = None

    for label in FUTURE_REFACTORING_SNAPSHOT_LABELS:
        snapshot_meta = future_snapshots.get(label)
        if not isinstance(snapshot_meta, dict) or not snapshot_meta.get("commit"):
            continue
        latest_snapshot_label = label
        counts = {
            "retained": 0,
            "weakly_retained": 0,
            "not_retained": 0,
            "unknown": 0,
        }
        touched_by_future_diff = 0
        operations_with_future_line_overlap = 0
        overlapping_future_lines = 0
        touched_refactoring_lines_by_file: Dict[str, set[int]] = {}
        future_diff = future_diff_tracking.get(label) if isinstance(future_diff_tracking.get(label), dict) else {}
        for operation in pr_operations:
            retention = _evaluate_operation_retention(
                operation,
                label,
                snapshot_meta,
                file_cache,
                future_diff,
            )
            operation.setdefault("retention", {})[label] = retention
            counts[retention["status"]] = counts.get(retention["status"], 0) + 1
            diff_overlap = retention.get("future_diff_overlap") or {}
            if diff_overlap.get("has_file_overlap"):
                touched_by_future_diff += 1
            if diff_overlap.get("has_line_overlap"):
                operations_with_future_line_overlap += 1
            for path, lines in _operation_future_diff_overlap_lines_by_file(
                operation,
                future_diff,
            ).items():
                touched_refactoring_lines_by_file.setdefault(path, set()).update(lines)
        touched_refactoring_line_counts_by_file = {
            path: len(lines) for path, lines in sorted(touched_refactoring_lines_by_file.items())
        }
        overlapping_future_lines = sum(touched_refactoring_line_counts_by_file.values())
        total = sum(counts.values())
        retained_total = counts.get("retained", 0) + counts.get("weakly_retained", 0)
        public_counts = {
            "retained": counts.get("retained", 0),
            "not_retained": counts.get("not_retained", 0),
            "unknown": counts.get("unknown", 0),
        }
        retention_by_snapshot[label] = {
            "snapshot_commit": snapshot_meta.get("commit"),
            "counts": public_counts,
            "trackable_refactoring_operations": trackable_refactoring_operations,
            "refactoring_zone_lines": refactoring_zone_line_count,
            "retained_total": retained_total,
            "retention_rate": (retained_total / total) if total > 0 else 0.0,
            "operations_touched_by_future_diff": touched_by_future_diff,
            "operations_with_future_file_overlap": touched_by_future_diff,
            "operations_with_future_line_overlap": operations_with_future_line_overlap,
            "overlapping_future_lines": overlapping_future_lines,
            "touched_refactoring_zone_lines": overlapping_future_lines,
            "touched_refactoring_zone_line_files": sorted(touched_refactoring_lines_by_file),
            "touched_refactoring_zone_line_counts_by_file": touched_refactoring_line_counts_by_file,
            "future_overlap_with_pr": (
                future_diff.get("overlap_with_pr") if isinstance(future_diff, dict) else {}
            ),
        }
        _log_info(
            "Retention {label}: retained={retained} weakly_retained={weak} not_retained={not_retained} unknown={unknown}".format(
                label=label,
                retained=counts["retained"],
                weak=counts["weakly_retained"],
                not_retained=counts["not_retained"],
                unknown=counts["unknown"],
            )
        )

    latest_summary = retention_by_snapshot.get(latest_snapshot_label) if latest_snapshot_label else None
    return {
        "per_snapshot": retention_by_snapshot,
        "latest_snapshot_label": latest_snapshot_label,
        "latest_retained_count": latest_summary.get("retained_total", 0) if latest_summary else 0,
        "latest_retention_rate": latest_summary.get("retention_rate", 0.0) if latest_summary else 0.0,
    }


def _compute_refactoring_metric_values(
    operations: List[Dict[str, Any]],
    hydration: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute derived metrics and summaries from standardized operations."""
    type_counts = _count_by_field(operations, "standardized_type")
    murphy_hill_counts = _count_by_field(operations, "murphy_hill_level")
    added_lines_by_file = _covered_lines_by_file(operations, "target_locations")
    removed_lines_by_file = _covered_lines_by_file(operations, "source_locations")
    refactor_count = len(operations)
    refactor_added_lines = sum(len(lines) for lines in added_lines_by_file.values())
    refactor_removed_lines = sum(len(lines) for lines in removed_lines_by_file.values())
    refactor_magnitude_lines = refactor_added_lines + refactor_removed_lines
    refactor_magnitude_files = _distinct_file_count(operations)
    refactor_diversity = _shannon_diversity(type_counts)
    refactor_density = (
        refactor_count / (refactor_magnitude_lines / 1000.0)
        if refactor_magnitude_lines > 0
        else 0.0
    )
    retention_summary = _compute_retention_summary(operations, hydration)
    pr_future_impact = _future_impact_summary(hydration)
    refactoring_zone_future_impact = _refactoring_zone_future_impact_summary(
        retention_summary,
        pr_future_impact,
    )
    future_availability = (
        hydration.get("future_snapshot_availability")
        if isinstance(hydration.get("future_snapshot_availability"), dict)
        else {}
    )
    future_snapshots = (hydration.get("snapshots") or {}).get("future") or {}
    if not isinstance(future_snapshots, dict):
        future_snapshots = {}
    after_commit = hydration.get("after_commit")
    future_snapshot_metrics: Dict[str, Dict[str, Any]] = {}
    for label in FUTURE_REFACTORING_SNAPSHOT_LABELS:
        future_snapshot = (
            future_snapshots.get(label)
            if isinstance(future_snapshots.get(label), dict)
            else {}
        )
        snapshot_commit = future_snapshot.get("commit") if isinstance(future_snapshot, dict) else None
        availability = (
            future_availability.get(label)
            if isinstance(future_availability.get(label), dict)
            else {}
        )
        future_snapshot_metrics[label] = {
            "status": "future_refactoring_tools_not_run",
            "snapshot_commit": snapshot_commit,
            "start_commit": after_commit,
            "end_commit": snapshot_commit,
            "refactoring_tool_collected": False,
            "not_collected_reason": "future_refactoring_tools_not_run",
            "snapshot_available": availability.get("available"),
            "missing_reason": availability.get("missing_reason"),
            "target_timestamp": availability.get("target_timestamp"),
            "repository_observation_cutoff": availability.get("repository_observation_cutoff"),
            "files_expected": availability.get("files_expected"),
            "files_copied": availability.get("files_copied"),
            "files_missing": availability.get("files_missing"),
            "missing_files": availability.get("missing_files") or [],
            "deleted_files": availability.get("deleted_files") or [],
            "renamed_files": availability.get("renamed_files") or [],
            "file_availability_status": availability.get("file_availability_status"),
            "refactor_count": 0,
            "refactor_type_count": {},
            "refactor_murphyhill_count": {},
            "refactor_diversity": 0.0,
            "refactor_added_lines": 0,
            "refactor_removed_lines": 0,
            "refactor_magnitude_lines": 0,
            "refactor_magnitude_files": 0,
            "refactor_density": 0.0,
            "future_impact": (
                refactoring_zone_future_impact.get("per_snapshot") or {}
            ).get(label, {}),
            "pr_changed_line_future_impact": (
                pr_future_impact.get("per_snapshot") or {}
            ).get(label, {}),
            "retention": (retention_summary.get("per_snapshot") or {}).get(label, {}),
        }
    return {
        "refactor_count": refactor_count,
        "refactor_type_count": type_counts,
        "refactor_murphyhill_count": murphy_hill_counts,
        "refactor_diversity": refactor_diversity,
        "refactor_added_lines": refactor_added_lines,
        "refactor_removed_lines": refactor_removed_lines,
        "refactor_magnitude_lines": refactor_magnitude_lines,
        "refactor_magnitude_files": refactor_magnitude_files,
        "refactor_density": refactor_density,
        "refactor_retention_summary": retention_summary["per_snapshot"],
        "refactor_latest_retention_snapshot": retention_summary["latest_snapshot_label"],
        "refactor_latest_retained_count": retention_summary["latest_retained_count"],
        "refactor_latest_retention_rate": retention_summary["latest_retention_rate"],
        "refactor_persistence": {
            label: float((summary or {}).get("retention_rate", 0.0))
            for label, summary in (retention_summary.get("per_snapshot") or {}).items()
        },
        "refactor_future_impact_summary": refactoring_zone_future_impact["per_snapshot"],
        "refactor_future_snapshots_with_file_overlap": refactoring_zone_future_impact[
            "snapshots_with_file_overlap"
        ],
        "refactor_future_snapshots_with_line_overlap": refactoring_zone_future_impact[
            "snapshots_with_line_overlap"
        ],
        "refactor_future_touched_refactoring_lines_total": refactoring_zone_future_impact[
            "total_touched_refactoring_zone_lines"
        ],
        "refactor_future_operations_with_file_overlap_total": refactoring_zone_future_impact[
            "total_operations_with_future_file_overlap"
        ],
        "refactor_future_operations_with_line_overlap_total": refactoring_zone_future_impact[
            "total_operations_with_future_line_overlap"
        ],
        "refactor_future_pr_impact_summary": pr_future_impact["per_snapshot"],
        "refactor_future_pr_touched_lines_total": pr_future_impact["total_touched_pr_lines"],
        "refactor_future_touched_pr_lines_total": pr_future_impact["total_touched_pr_lines"],
        "refactor_future_touching_commits_total": pr_future_impact["total_touching_commits"],
        "refactor_future_touch_coverage_mean_pct": pr_future_impact["mean_touch_coverage_pct"],
        "refactor_future_snapshot_availability": future_availability,
        "refactor_future_snapshot_metrics": future_snapshot_metrics,
        "taxonomy_summary": {
            "murphy_hill": murphy_hill_counts,
        },
        "retention_summary": retention_summary,
        "future_impact_summary": refactoring_zone_future_impact,
        "pr_changed_line_future_impact_summary": pr_future_impact,
    }


def _determine_stage_status(
    tool: Optional["RefactoringToolRunner"],
    mining_results: List[Dict[str, Any]],
) -> str:
    """Summarize the stage status from the selected tool and snapshot-level results."""
    if not mining_results:
        return "missing_snapshot"
    if not tool:
        return "unsupported_language"
    if any(result.get("status") == "invocation_failed" for result in mining_results):
        return "invocation_failed"
    if any(result.get("status") in {"tool_failed", "timed_out"} for result in mining_results):
        return "partial_failure"
    return "success"


def _write_json_artifact(path: Path, payload: Any) -> str:
    """Write one JSON artifact atomically and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)
    return str(path)


def _read_json_artifact(path: Optional[str]) -> Any:
    """Load JSON from an artifact path, returning None if unavailable."""
    if not path:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    try:
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _refactoring_artifact_paths(
    snapshot_path: Optional[str],
    tool_name: str,
) -> Dict[str, Optional[Path]]:
    """Return per-snapshot artifact paths for refactoring mining."""
    if not snapshot_path:
        return {
            "tool_output_path": None,
            "artifact_path": None,
        }
    root = Path(str(snapshot_path))
    stem = tool_name.lower()
    return {
        "tool_output_path": root / f"{stem}_tool_output.tmp.json",
        "artifact_path": root / f"{stem}_tool_results.json",
    }


def _is_checkpointed_refactoring_result(result: Any) -> bool:
    """Return True when a snapshot result is safe to reuse from a prior checkpoint."""
    if not isinstance(result, dict):
        return False
    return str(result.get("status")) in {
        "success",
        "unsupported_language",
        "missing_snapshot_path",
        "missing_commit_range",
    }


def _checkpointed_refactoring_results(existing_stage: Any) -> Dict[str, Dict[str, Any]]:
    """Return reusable snapshot results from a previously checkpointed stage payload."""
    if not isinstance(existing_stage, dict):
        return {}
    results = existing_stage.get("results")
    if not isinstance(results, list):
        mining = existing_stage.get("refactoring_operation_mining")
        if not isinstance(mining, dict):
            return {}
        results = mining.get("results")
    if not isinstance(results, list):
        return {}
    reusable: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if not _is_checkpointed_refactoring_result(result):
            continue
        label = result.get("snapshot_label")
        if label:
            reusable[str(label)] = result
    return reusable


@dataclass
class SnapshotMiningTarget:
    """Commit range and snapshot metadata for one mining checkpoint."""

    label: str
    snapshot: Dict[str, Any]
    start_commit: Optional[str]
    end_commit: Optional[str]
    baseline_snapshot_path: Optional[str] = None


@dataclass
class RefactoringToolRunner:
    """Base wrapper for a refactoring-mining command-line tool."""

    config: RefactoringToolConfig

    def _resolve_args_for_snapshot(
        self,
        repo_hydrator: RepositoryHydrator,
        snapshot_label: str,
        snapshot_meta: Dict[str, Any],
        start_commit: str,
        end_commit: str,
        output_path: Path,
        language: Optional[str],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Return command arguments and optional invocation metadata."""
        return (
            self._resolve_args(
                repo_hydrator,
                start_commit,
                end_commit,
                output_path,
                language,
            ),
            {},
        )

    @property
    def tool_name(self) -> str:
        """Compatibility accessor used by stage-level logging/payload code."""
        return self.config.tool_name

    def _resolve_command(self) -> str:
        """Return the configured command or a default executable name."""
        return self.config.resolve_command()

    def _resolve_args(
        self,
        repo_hydrator: RepositoryHydrator,
        start_commit: str,
        end_commit: str,
        output_path: Path,
        language: Optional[str],
    ) -> List[str]:
        """Return argument list for invoking the tool."""
        template = self.config.resolve_args_template()
        if template:
            formatted = template.format(
                repo=str(repo_hydrator.repo_dir.resolve()),
                start_commit=start_commit,
                end_commit=end_commit,
                output=str(output_path.resolve()),
                language=language or "",
            )
            return shlex.split(formatted)
        return self._default_args(
            repo_hydrator,
            start_commit,
            end_commit,
            output_path,
            language,
        )

    def _default_args(
        self,
        repo_hydrator: RepositoryHydrator,
        start_commit: str,
        end_commit: str,
        output_path: Path,
        language: Optional[str],
    ) -> List[str]:
        """Return tool-specific default arguments."""
        raise NotImplementedError

    def _extract_operations(
        self,
        parsed_output: Any,
        stdout: str,
        snapshot_label: str,
        snapshot_commit: Optional[str],
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Parse tool output into the shared operation schema."""
        return []

    def _build_result(
        self,
        *,
        pr: Any,
        repo_hydrator: RepositoryHydrator,
        snapshot_label: str,
        snapshot_commit: Optional[str],
        snapshot_path: Optional[str],
        start_commit: Optional[str],
        end_commit: Optional[str],
        status: str,
        notes: str,
        command: Optional[List[str]] = None,
        return_code: Optional[int] = None,
        artifact_path: Optional[str] = None,
        stdout_path: Optional[str] = None,
        stderr_path: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        timed_out: Optional[bool] = None,
        operations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build the standardized result envelope for one snapshot mining attempt."""
        operations = operations or []
        return {
            "status": status,
            "tool": self.config.tool_name,
            "repository_owner": repo_hydrator.owner,
            "repository_name": repo_hydrator.name,
            "pr_number": _get_pr_number(pr),
            "snapshot_label": snapshot_label,
            "snapshot_commit": snapshot_commit,
            "snapshot_path": snapshot_path,
            "start_commit": start_commit,
            "end_commit": end_commit,
            "command": command,
            "return_code": return_code,
            "artifact_path": artifact_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "elapsed_seconds": elapsed_seconds,
            "timed_out": timed_out,
            "operation_count": len(operations),
            "notes": notes,
        }

    def mine_operations(
        self,
        pr: Any,
        repo_hydrator: RepositoryHydrator,
        snapshot_label: str,
        snapshot_meta: Dict[str, Any],
        start_commit: Optional[str],
        end_commit: Optional[str],
        language: Optional[str],
    ) -> Dict[str, Any]:
        """Invoke the configured tool for one original-PR mining target."""
        snapshot_path = snapshot_meta.get("path")
        snapshot_commit = snapshot_meta.get("commit")
        if not snapshot_path:
            return self._build_result(
                pr=pr,
                repo_hydrator=repo_hydrator,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                snapshot_path=snapshot_path,
                start_commit=start_commit,
                end_commit=end_commit,
                status="missing_snapshot_path",
                notes="Cannot invoke tool without a snapshot path.",
            )

        artifact_paths = _refactoring_artifact_paths(
            snapshot_path,
            self.config.tool_name,
        )
        output_path = artifact_paths["tool_output_path"]
        artifact_path = artifact_paths["artifact_path"]
        if output_path:
            output_path = output_path.resolve()
        if artifact_path:
            artifact_path = artifact_path.resolve()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.unlink(missing_ok=True)

        if not start_commit or not end_commit:
            return self._build_result(
                pr=pr,
                repo_hydrator=repo_hydrator,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                snapshot_path=snapshot_path,
                start_commit=start_commit,
                end_commit=end_commit,
                status="missing_commit_range",
                notes="Cannot invoke tool without both start and end commits.",
            )

        args, invocation_metadata = self._resolve_args_for_snapshot(
            repo_hydrator,
            snapshot_label,
            snapshot_meta,
            start_commit,
            end_commit,
            output_path,
            language,
        )
        command = [self._resolve_command(), *args]
        if self.config.tool_name.lower() == "reffdiff":
            _best_effort_prepare_j2v8_for_refdiff()
        _log_info(
            "PR {pr_number}: invoking {tool} for {label}".format(
                pr_number=_get_pr_number(pr),
                tool=self.config.tool_name,
                label=snapshot_label,
            )
        )
        output_dir = (
            output_path.parent
            if output_path
            else Path(str(snapshot_path)).resolve() / "refactoring"
        )
        try:
            result = _run_external_tool_file_backed(
                command,
                cwd=repo_hydrator.repo_dir,
                output_dir=output_dir,
                label=f"{self.config.tool_name}-{snapshot_label}",
                timeout_seconds=self.config.timeout_seconds,
            )
        except Exception as exc:
            _log_error(
                "PR {pr_number}: {tool} {label} invocation failed: {error}".format(
                    pr_number=_get_pr_number(pr),
                    tool=self.config.tool_name,
                    label=snapshot_label,
                    error=exc,
                )
            )
            result_payload = self._build_result(
                pr=pr,
                repo_hydrator=repo_hydrator,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                snapshot_path=snapshot_path,
                start_commit=start_commit,
                end_commit=end_commit,
                status="invocation_failed",
                notes=str(exc),
                command=command,
            )
            result_payload.update(invocation_metadata)
            return result_payload

        if result.timed_out:
            stdout_text = ""
            stdout_parse_limited = False
            stderr_text = result.stderr_tail
        else:
            stdout_text = read_text_if_within_limit(
                result.stdout_path,
                max_bytes=DEFAULT_OUTPUT_PARSE_BYTES,
            )
            stdout_parse_limited = stdout_text is None
            if stdout_text is None:
                stdout_text = ""
            stderr_text = read_text_if_within_limit(
                result.stderr_path,
                max_bytes=DEFAULT_OUTPUT_PARSE_BYTES,
            )
            if stderr_text is None:
                stderr_text = result.stderr_tail
        sanitized_stderr = _strip_known_j2v8_warnings(stderr_text or "")
        result_status = (
            "timed_out"
            if result.timed_out
            else ("success" if result.returncode == 0 else "tool_failed")
        )
        if result_status != "success":
            stderr_lines = sanitized_stderr.strip().splitlines()
            summary = stderr_lines[0] if stderr_lines else "no stderr output"
            stderr_tail = " | ".join(stderr_lines[-20:]) if stderr_lines else ""
            command_text = " ".join(str(part) for part in command)
            artifact_path_text = str(artifact_path) if artifact_path else "none"
            parser_limit = (
                self.config.tool_name.lower() == "reffdiff"
                and str(language or "").strip().lower() == "javascript"
                and _is_js_parser_plugin_failure(sanitized_stderr)
            )
            message = (
                "PR {pr_number}: {tool} {label} {status} return_code={code} stderr={stderr} "
                "artifact_path={artifact_path} command={command} stderr_tail={stderr_tail}".format(
                    pr_number=_get_pr_number(pr),
                    tool=self.config.tool_name,
                    label=snapshot_label,
                    status=result_status,
                    code=result.returncode,
                    stderr=summary,
                    artifact_path=artifact_path_text,
                    command=command_text,
                    stderr_tail=stderr_tail if stderr_tail else "no stderr output",
                )
            )
            if parser_limit:
                _log_info(f"{message} [known_js_parser_limit=true]")
            else:
                _log_error(message)

        parsed_output = None
        if result.timed_out:
            operations: List[Dict[str, Any]] = []
        else:
            if output_path and output_path.exists():
                try:
                    parsed_output = json.loads(output_path.read_text(encoding="utf-8"))
                except Exception:
                    parsed_output = None
            if output_path and not output_path.exists() and stdout_text.strip():
                try:
                    parsed_output = json.loads(stdout_text)
                except Exception:
                    parsed_output = None
                _write_json_artifact(
                    output_path,
                    parsed_output if parsed_output is not None else {"stdout_preview": result.stdout_preview},
                )

            operations = self._extract_operations(
                parsed_output,
                stdout_text,
                snapshot_label,
                snapshot_commit,
                language,
            )
        if result.timed_out:
            timeout_seconds = self.config.timeout_seconds
            timeout_label = (
                f"{timeout_seconds:.0f} seconds"
                if timeout_seconds is not None
                else "the configured timeout"
            )
            notes = (
                f"Invoked {self.config.tool_name}; timed out after {timeout_label}. "
                "Partial output was not parsed."
            )
        else:
            notes = f"Invoked {self.config.tool_name} and normalized the mined operations."
        invocation_note = invocation_metadata.get("refactoring_invocation_note")
        if invocation_note:
            notes += f" {invocation_note}"
        if stdout_parse_limited:
            notes += (
                f" stdout exceeded {DEFAULT_OUTPUT_PARSE_BYTES} bytes; stdout fallback parsing was skipped."
            )
        if artifact_path:
            artifact_payload: Dict[str, Any] = {
                "tool": self.config.tool_name,
                "snapshot_label": snapshot_label,
                "snapshot_commit": snapshot_commit,
                "status": result_status,
                "command": _compact_command(command),
                "return_code": result.returncode,
                "elapsed_seconds": result.elapsed_seconds,
                "timed_out": result.timed_out,
                "stdout_path": result.stdout_path,
                "stderr_path": result.stderr_path,
                "notes": notes,
                "stdout_preview": _compact_text(result.stdout_preview, max_chars=1500),
                "stderr_preview": _compact_text(sanitized_stderr, max_chars=2500),
                "standardized_operations": [
                    _compact_operation_for_artifact(op)
                    for op in operations
                    if isinstance(op, dict)
                ],
            }
            artifact_payload.update(invocation_metadata)
            _write_json_artifact(artifact_path, artifact_payload)
        if output_path and output_path.exists():
            output_path.unlink(missing_ok=True)
        _log_info(
            "PR {pr_number}: {tool} {label} status={status} operations={count}".format(
                pr_number=_get_pr_number(pr),
                tool=self.config.tool_name,
                label=snapshot_label,
                status=result_status,
                count=len(operations),
            )
        )

        result_payload = self._build_result(
            pr=pr,
            repo_hydrator=repo_hydrator,
            snapshot_label=snapshot_label,
            snapshot_commit=snapshot_commit,
            snapshot_path=snapshot_path,
            start_commit=start_commit,
            end_commit=end_commit,
            status=result_status,
            notes=notes,
            command=command,
            return_code=result.returncode,
            artifact_path=str(artifact_path) if artifact_path else None,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            elapsed_seconds=result.elapsed_seconds,
            timed_out=result.timed_out,
            operations=operations,
        )
        result_payload.update(invocation_metadata)
        return result_payload


@dataclass
class RefactoringMinerRunner(RefactoringToolRunner):
    """Runner for RefactoringMiner."""

    config: RefactoringToolConfig = REFACTORING_MINER_CONFIG

    def _git_first_parent(self, repo_dir: Path, commit: str) -> Optional[str]:
        """Return the first parent for a commit, or None if it cannot be resolved."""
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), "show", "-s", "--format=%P", commit],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            return None
        parents = completed.stdout.strip().split()
        return parents[0] if parents else None

    def _create_synthetic_pr_commit(
        self,
        repo_dir: Path,
        base_commit: str,
        after_commit: str,
    ) -> str:
        """Create a single-parent commit that represents the PR net diff."""
        env = os.environ.copy()
        env.setdefault("GIT_AUTHOR_NAME", "MOSAIC Curation")
        env.setdefault("GIT_AUTHOR_EMAIL", "mosaic-curation@example.invalid")
        env.setdefault("GIT_COMMITTER_NAME", "MOSAIC Curation")
        env.setdefault("GIT_COMMITTER_EMAIL", "mosaic-curation@example.invalid")
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "commit-tree",
                f"{after_commit}^{{tree}}",
                "-p",
                base_commit,
                "-m",
                "synthetic PR net diff for RefactoringMiner",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if completed.returncode != 0:
            message = completed.stderr or completed.stdout or "git commit-tree failed"
            raise RuntimeError(message.strip())
        synthetic_commit = completed.stdout.strip()
        if not synthetic_commit:
            raise RuntimeError("git commit-tree did not return a synthetic commit SHA")
        return synthetic_commit

    def _resolve_args_for_snapshot(
        self,
        repo_hydrator: RepositoryHydrator,
        snapshot_label: str,
        snapshot_meta: Dict[str, Any],
        start_commit: str,
        end_commit: str,
        output_path: Path,
        language: Optional[str],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Prefer single-commit PR mining over RefactoringMiner range traversal."""
        if snapshot_label != "pr":
            return super()._resolve_args_for_snapshot(
                repo_hydrator,
                snapshot_label,
                snapshot_meta,
                start_commit,
                end_commit,
                output_path,
                language,
            )

        first_parent = self._git_first_parent(repo_hydrator.repo_dir, end_commit)
        if first_parent == start_commit:
            metadata = {
                "refactoring_invocation_mode": "single_commit",
                "refactoring_invocation_commit": end_commit,
                "refactoring_invocation_parent": first_parent,
                "refactoring_invocation_note": (
                    "Used RefactoringMiner -c because the PR after commit's first "
                    "parent matches the recorded base commit."
                ),
            }
            return (
                [
                    "-c",
                    str(repo_hydrator.repo_dir.resolve()),
                    end_commit,
                    "-json",
                    str(output_path.resolve()),
                ],
                metadata,
            )

        try:
            synthetic_commit = self._create_synthetic_pr_commit(
                repo_hydrator.repo_dir,
                start_commit,
                end_commit,
            )
        except Exception as exc:
            metadata = {
                "refactoring_invocation_mode": "range_fallback",
                "refactoring_invocation_parent": first_parent,
                "refactoring_invocation_note": (
                    "Fell back to RefactoringMiner -bc because synthetic "
                    f"single-commit creation failed: {exc}"
                ),
            }
            return (
                self._default_args(
                    repo_hydrator,
                    start_commit,
                    end_commit,
                    output_path,
                    language,
                ),
                metadata,
            )

        metadata = {
            "refactoring_invocation_mode": "synthetic_single_commit",
            "refactoring_invocation_commit": synthetic_commit,
            "refactoring_invocation_parent": start_commit,
            "refactoring_invocation_tree_commit": end_commit,
            "refactoring_invocation_original_parent": first_parent,
            "refactoring_invocation_note": (
                "Used RefactoringMiner -c on a synthetic single-parent commit "
                "whose tree is the PR after commit and whose parent is the "
                "recorded base commit."
            ),
        }
        return (
            [
                "-c",
                str(repo_hydrator.repo_dir.resolve()),
                synthetic_commit,
                "-json",
                str(output_path.resolve()),
            ],
            metadata,
        )

    def _default_args(
        self,
        repo_hydrator: RepositoryHydrator,
        start_commit: str,
        end_commit: str,
        output_path: Path,
        language: Optional[str],
    ) -> List[str]:
        return [
            "-bc",
            str(repo_hydrator.repo_dir.resolve()),
            start_commit,
            end_commit,
            "-json",
            str(output_path.resolve()),
        ]

    def _extract_operations(
        self,
        parsed_output: Any,
        stdout: str,
        snapshot_label: str,
        snapshot_commit: Optional[str],
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        return _extract_refactoringminer_operations(
            parsed_output,
            tool_name=self.config.tool_name,
            snapshot_label=snapshot_label,
            snapshot_commit=snapshot_commit,
            language=language,
        )


@dataclass
class ReffDiffRunner(RefactoringToolRunner):
    """Runner for ReffDiff."""

    config: RefactoringToolConfig = REFFDIFF_CONFIG

    def _default_args(
        self,
        repo_hydrator: RepositoryHydrator,
        start_commit: str,
        end_commit: str,
        output_path: Path,
        language: Optional[str],
    ) -> List[str]:
        args = [
            "--repo",
            str(repo_hydrator.repo_dir.resolve()),
            "--start-commit",
            start_commit,
            "--end-commit",
            end_commit,
            "--output",
            str(output_path.resolve()),
        ]
        if language:
            args.extend(["--language", language])
        return args

    def _extract_operations(
        self,
        parsed_output: Any,
        stdout: str,
        snapshot_label: str,
        snapshot_commit: Optional[str],
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        operations = _extract_refdiff_operations_from_json(
            parsed_output,
            tool_name=self.config.tool_name,
            snapshot_label=snapshot_label,
            snapshot_commit=snapshot_commit,
            language=language,
        )
        if operations:
            return operations
        return _extract_refdiff_operations_from_stdout(
            stdout,
            tool_name=self.config.tool_name,
            snapshot_label=snapshot_label,
            snapshot_commit=snapshot_commit,
            language=language,
        )


@dataclass
class RefactoringMinerPlusPlusRunner(RefactoringToolRunner):
    """Runner for RefactoringMiner++."""

    config: RefactoringToolConfig = REFACTORING_MINER_PP_CONFIG

    def mine_operations(
        self,
        pr: Any,
        repo_hydrator: RepositoryHydrator,
        snapshot_label: str,
        snapshot_meta: Dict[str, Any],
        start_commit: Optional[str],
        end_commit: Optional[str],
        language: Optional[str],
    ) -> Dict[str, Any]:
        if os.name == "nt":
            # Keep Windows compatibility fallback on commit-range mode.
            return super().mine_operations(
                pr=pr,
                repo_hydrator=repo_hydrator,
                snapshot_label=snapshot_label,
                snapshot_meta=snapshot_meta,
                start_commit=start_commit,
                end_commit=end_commit,
                language=language,
            )

        _ensure_refactoringminerpp_runtime()
        snapshot_path_raw = snapshot_meta.get("path")
        snapshot_commit = snapshot_meta.get("commit")
        baseline_path_raw = snapshot_meta.get("baseline_snapshot_path")
        if not snapshot_path_raw or not baseline_path_raw:
            return self._build_result(
                pr=pr,
                repo_hydrator=repo_hydrator,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                snapshot_path=snapshot_path_raw,
                start_commit=start_commit,
                end_commit=end_commit,
                status="missing_snapshot_path",
                notes="CAP-based RM++ needs both current snapshot path and baseline snapshot path.",
            )

        snapshot_path = Path(str(snapshot_path_raw)).resolve()
        baseline_path = Path(str(baseline_path_raw)).resolve()
        if not snapshot_path.exists() or not baseline_path.exists():
            return self._build_result(
                pr=pr,
                repo_hydrator=repo_hydrator,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                snapshot_path=snapshot_path_raw,
                start_commit=start_commit,
                end_commit=end_commit,
                status="missing_snapshot_path",
                notes=(
                    f"CAP-based RM++ snapshot paths unavailable: "
                    f"baseline_exists={baseline_path.exists()} current_exists={snapshot_path.exists()}"
                ),
            )

        cpp_pairs = _extract_cpp_file_pairs(
            pr,
            use_previous_filenames=(snapshot_label == "pr"),
        )
        if not cpp_pairs:
            return self._build_result(
                pr=pr,
                repo_hydrator=repo_hydrator,
                snapshot_label=snapshot_label,
                snapshot_commit=snapshot_commit,
                snapshot_path=snapshot_path_raw,
                start_commit=start_commit,
                end_commit=end_commit,
                status="success",
                notes="No comparable C++ file pairs were available for CAP/RM++ model analysis.",
                command=None,
                return_code=0,
                artifact_path=None,
                operations=[],
            )

        artifact_paths = _refactoring_artifact_paths(
            snapshot_path_raw,
            self.config.tool_name,
        )
        output_path = artifact_paths["tool_output_path"]
        artifact_path = artifact_paths["artifact_path"]
        if output_path:
            output_path = output_path.resolve()
        if artifact_path:
            artifact_path = artifact_path.resolve()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        all_operations: List[Dict[str, Any]] = []
        file_runs: List[Dict[str, Any]] = []
        command_history: List[List[str]] = []
        rmpp_success_count = 0
        cap_timed_out = False
        rmpp_timed_out = False

        with tempfile.TemporaryDirectory(prefix="rmpp_cpp_models_") as temp_dir:
            temp_root = Path(temp_dir)
            for idx, pair in enumerate(cpp_pairs):
                before_rel = pair["before_path"]
                after_rel = pair["after_path"]
                before_file = (baseline_path / Path(before_rel)).resolve()
                after_file = (snapshot_path / Path(after_rel)).resolve()
                run_entry: Dict[str, Any] = {
                    "before_path": before_rel,
                    "after_path": after_rel,
                    "before_exists": before_file.exists(),
                    "after_exists": after_file.exists(),
                    "status": "skipped",
                }
                if not before_file.exists() or not after_file.exists():
                    run_entry["notes"] = "Skipping pair because baseline/current file is missing."
                    file_runs.append(run_entry)
                    continue

                before_model = (temp_root / f"pair_{idx:03d}.before.json").resolve()
                after_model = (temp_root / f"pair_{idx:03d}.after.json").resolve()
                cap_before_cmd = [str(_RMPP_CAP_BIN), str(before_file), str(before_model)]
                cap_after_cmd = [str(_RMPP_CAP_BIN), str(after_file), str(after_model)]
                command_history.extend([cap_before_cmd, cap_after_cmd])
                log_dir = artifact_path.parent if artifact_path else temp_root

                with tool_slot("cpp_tools", max(0, int(CPP_TOOL_CONCURRENCY_LIMIT))):
                    cap_before = _run_external_tool_file_backed(
                        cap_before_cmd,
                        cwd=repo_hydrator.repo_dir,
                        output_dir=log_dir,
                        label=f"{self.config.tool_name}-cap-before-{idx:03d}",
                        timeout_seconds=REFACTORING_MINER_PP_CAP_TIMEOUT_SECONDS,
                    )
                if cap_before.returncode != 0 or not before_model.exists():
                    run_entry["status"] = (
                        "timed_out" if cap_before.timed_out else "cap_failed"
                    )
                    run_entry["return_code"] = cap_before.returncode
                    run_entry["elapsed_seconds"] = cap_before.elapsed_seconds
                    run_entry["timed_out"] = cap_before.timed_out
                    run_entry["termination_reason"] = cap_before.termination_reason
                    run_entry["stdout_path"] = cap_before.stdout_path
                    run_entry["stderr_path"] = cap_before.stderr_path
                    run_entry["stdout_preview"] = cap_before.stdout_preview
                    run_entry["stderr_preview"] = cap_before.stderr_preview
                    run_entry["notes"] = _compact_text(
                        f"CAP baseline model generation failed rc={cap_before.returncode}: {cap_before.stderr_tail}",
                        max_chars=1400,
                    )
                    file_runs.append(run_entry)
                    if cap_before.timed_out:
                        cap_timed_out = True
                        break
                    continue

                with tool_slot("cpp_tools", max(0, int(CPP_TOOL_CONCURRENCY_LIMIT))):
                    cap_after = _run_external_tool_file_backed(
                        cap_after_cmd,
                        cwd=repo_hydrator.repo_dir,
                        output_dir=log_dir,
                        label=f"{self.config.tool_name}-cap-after-{idx:03d}",
                        timeout_seconds=REFACTORING_MINER_PP_CAP_TIMEOUT_SECONDS,
                    )
                if cap_after.returncode != 0 or not after_model.exists():
                    run_entry["status"] = (
                        "timed_out" if cap_after.timed_out else "cap_failed"
                    )
                    run_entry["return_code"] = cap_after.returncode
                    run_entry["elapsed_seconds"] = cap_after.elapsed_seconds
                    run_entry["timed_out"] = cap_after.timed_out
                    run_entry["termination_reason"] = cap_after.termination_reason
                    run_entry["stdout_path"] = cap_after.stdout_path
                    run_entry["stderr_path"] = cap_after.stderr_path
                    run_entry["stdout_preview"] = cap_after.stdout_preview
                    run_entry["stderr_preview"] = cap_after.stderr_preview
                    run_entry["notes"] = _compact_text(
                        f"CAP current model generation failed rc={cap_after.returncode}: {cap_after.stderr_tail}",
                        max_chars=1400,
                    )
                    file_runs.append(run_entry)
                    if cap_after.timed_out:
                        cap_timed_out = True
                        break
                    continue

                rmpp_cmd = [
                    self._resolve_command(),
                    "-cpp",
                    str(before_model),
                    str(after_model),
                ]
                command_history.append(rmpp_cmd)
                with tool_slot("cpp_tools", max(0, int(CPP_TOOL_CONCURRENCY_LIMIT))):
                    rmpp_result = _run_external_tool_file_backed(
                        rmpp_cmd,
                        cwd=repo_hydrator.repo_dir,
                        output_dir=log_dir,
                        label=f"{self.config.tool_name}-rmpp-{idx:03d}",
                        timeout_seconds=self.config.timeout_seconds,
                    )
                if rmpp_result.returncode != 0:
                    run_entry["status"] = (
                        "timed_out" if rmpp_result.timed_out else "tool_failed"
                    )
                    run_entry["return_code"] = rmpp_result.returncode
                    run_entry["elapsed_seconds"] = rmpp_result.elapsed_seconds
                    run_entry["timed_out"] = rmpp_result.timed_out
                    run_entry["termination_reason"] = rmpp_result.termination_reason
                    run_entry["stdout_path"] = rmpp_result.stdout_path
                    run_entry["stderr_path"] = rmpp_result.stderr_path
                    run_entry["stdout_preview"] = rmpp_result.stdout_preview
                    run_entry["stderr_preview"] = rmpp_result.stderr_preview
                    run_entry["notes"] = _compact_text(
                        f"RM++ -cpp failed rc={rmpp_result.returncode}: {rmpp_result.stderr_tail}",
                        max_chars=1600,
                    )
                    file_runs.append(run_entry)
                    if rmpp_result.timed_out:
                        rmpp_timed_out = True
                        break
                    continue

                rmpp_success_count += 1
                rmpp_stdout = read_text_if_within_limit(
                    rmpp_result.stdout_path,
                    max_bytes=DEFAULT_OUTPUT_PARSE_BYTES,
                )
                if rmpp_stdout is None:
                    rmpp_stdout = ""
                    run_entry["notes"] = (
                        f"RM++ stdout exceeded {DEFAULT_OUTPUT_PARSE_BYTES} bytes; parsing skipped."
                    )
                operations = _extract_rmpp_text_operations(
                    rmpp_stdout,
                    tool_name=self.config.tool_name,
                    snapshot_label=snapshot_label,
                    snapshot_commit=snapshot_commit,
                    language=language,
                )
                for operation in operations:
                    operation.setdefault("raw_operation", {})
                    if isinstance(operation["raw_operation"], dict):
                        operation["raw_operation"].setdefault("before_path", before_rel)
                        operation["raw_operation"].setdefault("after_path", after_rel)
                all_operations.extend(operations)
                run_entry["status"] = "success"
                run_entry["return_code"] = rmpp_result.returncode
                run_entry["elapsed_seconds"] = rmpp_result.elapsed_seconds
                run_entry["stdout_path"] = rmpp_result.stdout_path
                run_entry["stderr_path"] = rmpp_result.stderr_path
                run_entry["stdout_preview"] = rmpp_result.stdout_preview
                run_entry["stderr_preview"] = rmpp_result.stderr_preview
                run_entry["operation_count"] = len(operations)
                run_entry["notes"] = run_entry.get("notes") or "CAP + RM++ model comparison completed."
                file_runs.append(run_entry)

        final_status = (
            "timed_out"
            if cap_timed_out or rmpp_timed_out
            else ("success" if rmpp_success_count > 0 else "tool_failed")
        )
        notes = (
            f"Invoked CAP+RM++ on {len(file_runs)} C++ file pairs. "
            f"successful_pairs={rmpp_success_count} operations={len(all_operations)} "
            f"cap_timeout_seconds={REFACTORING_MINER_PP_CAP_TIMEOUT_SECONDS:.0f}"
        )
        if cap_timed_out:
            notes += " Aborted C++ RM++ stage after CAP timed out."
        elif rmpp_timed_out:
            notes += " Aborted C++ RM++ stage after RM++ -cpp timed out."
        if final_status != "success":
            _log_error(
                "PR {pr_number}: {tool} {label} status={status} notes={notes}".format(
                    pr_number=_get_pr_number(pr),
                    tool=self.config.tool_name,
                    label=snapshot_label,
                    status=final_status,
                    notes=notes,
                )
            )

        if artifact_path:
            _write_json_artifact(
                artifact_path,
                {
                    "tool": self.config.tool_name,
                    "snapshot_label": snapshot_label,
                    "snapshot_commit": snapshot_commit,
                    "status": final_status,
                    "command": _compact_text(
                        " ; ".join(" ".join(map(str, cmd)) for cmd in command_history),
                        max_chars=5000,
                    ),
                    "return_code": (
                        0
                        if final_status == "success"
                        else (124 if final_status == "timed_out" else 1)
                    ),
                    "timed_out": final_status == "timed_out",
                    "notes": notes,
                    "file_runs": file_runs,
                    "standardized_operations": [
                        _compact_operation_for_artifact(op)
                        for op in all_operations
                        if isinstance(op, dict)
                    ],
                },
            )

        _log_info(
            "PR {pr_number}: {tool} {label} status={status} operations={count}".format(
                pr_number=_get_pr_number(pr),
                tool=self.config.tool_name,
                label=snapshot_label,
                status=final_status,
                count=len(all_operations),
            )
        )
        return self._build_result(
            pr=pr,
            repo_hydrator=repo_hydrator,
            snapshot_label=snapshot_label,
            snapshot_commit=snapshot_commit,
            snapshot_path=snapshot_path_raw,
            start_commit=start_commit,
            end_commit=end_commit,
            status=final_status,
            notes=notes,
            command=command_history[-1] if command_history else None,
            return_code=(
                0
                if final_status == "success"
                else (124 if final_status == "timed_out" else 1)
            ),
            artifact_path=str(artifact_path) if artifact_path else None,
            timed_out=final_status == "timed_out",
            operations=all_operations,
        )

    def _default_args(
        self,
        repo_hydrator: RepositoryHydrator,
        start_commit: str,
        end_commit: str,
        output_path: Path,
        language: Optional[str],
    ) -> List[str]:
        return [
            "-bc",
            str(repo_hydrator.repo_dir.resolve()),
            start_commit,
            end_commit,
            "-json",
            str(output_path.resolve()),
        ]

    def _extract_operations(
        self,
        parsed_output: Any,
        stdout: str,
        snapshot_label: str,
        snapshot_commit: Optional[str],
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        return _extract_refactoringminer_operations(
            parsed_output,
            tool_name=self.config.tool_name,
            snapshot_label=snapshot_label,
            snapshot_commit=snapshot_commit,
            language=language,
        )


@dataclass
class RefactoringMetricsStage:
    """Mine original-PR refactoring operations and summarize persistence metrics."""

    def _select_refactoring_tool(self, language: Optional[str]) -> Optional[RefactoringToolRunner]:
        """Choose the refactoring-mining tool based on PR language."""
        if language in {"java", "python"}:
            return RefactoringMinerRunner()
        if language == "javascript":
            # ReffDiff is the default JS refactoring tool. Linux/Docker support
            # relies on the container patching j2v8_linux_x86_64-4.6.0.jar.
            return ReffDiffRunner()
        if language == "c++":
            return RefactoringMinerPlusPlusRunner()
        return None

    def _snapshot_targets(
        self,
        hydration: Dict[str, Any],
        *,
        include_future: bool,
    ) -> List[SnapshotMiningTarget]:
        """Return only the original before/after PR mining target."""
        snapshots = hydration.get("snapshots") or {}
        targets: List[SnapshotMiningTarget] = []
        base_commit = hydration.get("base_commit")
        after_commit = hydration.get("after_commit")
        before_snapshot = snapshots.get("before")
        before_path = before_snapshot.get("path") if isinstance(before_snapshot, dict) else None

        after_snapshot = snapshots.get("after")
        if isinstance(after_snapshot, dict) and after_snapshot.get("commit"):
            targets.append(
                SnapshotMiningTarget(
                    label="pr",
                    snapshot=after_snapshot,
                    start_commit=base_commit,
                    end_commit=after_commit,
                    baseline_snapshot_path=before_path,
                )
            )

        if include_future:
            _log_info(
                "Future refactoring persistence uses hydrated diff tracking; "
                "refactoring tools run only for the original PR."
            )

        return targets

    def _result_operations(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load standardized operations from the snapshot artifact reference."""
        artifact = _read_json_artifact(result.get("artifact_path"))
        if isinstance(artifact, dict):
            operations = artifact.get("standardized_operations")
            if isinstance(operations, list):
                return operations
        return []

    def _completed_snapshot_labels(self, mining_results: List[Dict[str, Any]]) -> List[str]:
        """Return completed snapshot labels in processing order."""
        labels: List[str] = []
        for result in mining_results:
            label = result.get("snapshot_label")
            if label:
                labels.append(str(label))
        return labels

    def _build_stage_progress(
        self,
        *,
        language: Optional[str],
        tool: Optional[RefactoringToolRunner],
        snapshot_targets: List[SnapshotMiningTarget],
        mining_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a lightweight checkpoint payload for refactoring mining progress."""
        return {
            "status": _determine_stage_status(tool, mining_results),
            "stage": "refactoring_metrics",
            "current_phase": "refactoring_metrics",
            "selected_tool": tool.tool_name if tool else None,
            "pr_language": language,
            "snapshot_labels": [target.label for target in snapshot_targets],
            "completed_snapshot_labels": self._completed_snapshot_labels(mining_results),
            "operation_count": sum(int(result.get("operation_count", 0)) for result in mining_results),
            "results": mining_results,
        }

    def _build_stage_payload(
        self,
        *,
        pr: Any,
        hydration: Dict[str, Any],
        repo_hydrator: RepositoryHydrator,
        language: Optional[str],
        tool: Optional[RefactoringToolRunner],
        snapshot_targets: List[SnapshotMiningTarget],
        mining_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the refactoring-stage payload from the currently available snapshot results."""
        pr_number = _get_pr_number(pr)
        standardized_operations = [
            operation
            for result in mining_results
            for operation in self._result_operations(result)
        ]
        metric_values = _compute_refactoring_metric_values(
            standardized_operations,
            hydration,
        )
        stage_status = _determine_stage_status(tool, mining_results)
        return {
            "status": stage_status,
            "stage": "refactoring_metrics",
            "repository_owner": repo_hydrator.owner,
            "repository_name": repo_hydrator.name,
            "pr_number": pr_number,
            "pr_language": language,
            "has_hydration": bool(hydration),
            "has_future_snapshots": bool(hydration.get("has_future_snapshots")),
            "refactoring_operation_mining": {
                "selected_tool": tool.tool_name if tool else None,
                "snapshot_labels": [target.label for target in snapshot_targets],
                "total_operations": metric_values["refactor_count"],
                "standardized_operations": standardized_operations,
                "taxonomy_summary": metric_values["taxonomy_summary"],
                "retention_summary": metric_values["retention_summary"],
                "results": mining_results,
            },
            "refactoring_metrics": {
                "status": stage_status,
                "metrics": {
                    key: value
                    for key, value in metric_values.items()
                    if key not in {"taxonomy_summary", "retention_summary"}
                },
                "notes": (
                    "Taxonomy classification is populated from standardized operations."
                    " Diversity uses Shannon entropy over standardized operation types."
                    " Added/removed line magnitudes use unique covered line ranges from"
                    " normalized source/target locations. Retention is a best-effort"
                    " heuristic based on PR operation target locations and future"
                    " snapshot file/code-element matches. Future impact for"
                    " refactoring metrics is scoped to original PR refactoring"
                    " target locations; broad PR changed-line overlap is retained"
                    " separately as context. Future refactoring tools are not run."
                ),
            },
            "notes": (
                "Refactoring mining invocation is wired by language and snapshot."
                " Standardized operations now include Murphy-Hill and Fowler taxonomy labels."
            ),
        }

    def compute(
        self,
        pr: Any,
        hydration: Dict[str, Any],
        repo_hydrator: RepositoryHydrator,
        *,
        include_future: bool = True,
        existing_stage: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run original PR refactoring mining for one hydrated PR."""
        language = _select_pr_language(pr)
        tool = self._select_refactoring_tool(language)
        snapshot_targets = self._snapshot_targets(hydration, include_future=include_future)
        mined_refactorings: List[Dict[str, Any]] = []
        pr_number = _get_pr_number(pr)
        checkpointed_results = _checkpointed_refactoring_results(existing_stage)

        _log_info(
            "PR {pr_number}: language={language} tool={tool} snapshots={count}".format(
                pr_number=pr_number,
                language=language or "unknown",
                tool=tool.tool_name if tool else "unsupported",
                count=len(snapshot_targets),
            )
        )

        for target in snapshot_targets:
            label = target.label
            snapshot_meta = dict(target.snapshot)
            if target.baseline_snapshot_path:
                snapshot_meta["baseline_snapshot_path"] = target.baseline_snapshot_path
            start_commit = target.start_commit
            end_commit = target.end_commit
            checkpointed_result = checkpointed_results.get(label)
            if checkpointed_result:
                _log_info(f"PR {pr_number}: reusing checkpointed refactoring result for {label}")
                mined_refactorings.append(checkpointed_result)
                if progress_callback:
                    progress_callback(
                        self._build_stage_progress(
                            language=language,
                            tool=tool,
                            snapshot_targets=snapshot_targets,
                            mining_results=mined_refactorings,
                        )
                    )
                continue
            if tool is None:
                mined_refactorings.append(
                    {
                        "status": "unsupported_language",
                        "snapshot_label": label,
                        "snapshot_commit": snapshot_meta.get("commit"),
                        "snapshot_path": snapshot_meta.get("path"),
                        "start_commit": start_commit,
                        "end_commit": end_commit,
                        "operation_count": 0,
                        "operations": [],
                        "notes": "No refactoring mining tool mapping is defined for this language.",
                    }
                )
                if progress_callback:
                    progress_callback(
                        self._build_stage_progress(
                            language=language,
                            tool=tool,
                            snapshot_targets=snapshot_targets,
                            mining_results=mined_refactorings,
                        )
                    )
                continue
            _log_info(
                "PR {pr_number}: mining {label} with {tool_name}".format(
                    pr_number=pr_number,
                    label=label,
                    tool_name=tool.tool_name,
                )
            )
            mined_refactorings.append(
                tool.mine_operations(
                    pr,
                    repo_hydrator,
                    label,
                    snapshot_meta,
                    start_commit,
                    end_commit,
                    language,
                )
            )
            latest = mined_refactorings[-1]
            latest_status = str(latest.get("status") or "")
            if latest_status != "success":
                _log_error(
                    "PR {pr_number}: {tool} {label} status={status} notes={notes}".format(
                        pr_number=pr_number,
                        tool=tool.tool_name,
                        label=label,
                        status=latest_status,
                        notes=latest.get("notes"),
                    )
                )
            if progress_callback:
                progress_callback(
                    self._build_stage_progress(
                        language=language,
                        tool=tool,
                        snapshot_targets=snapshot_targets,
                        mining_results=mined_refactorings,
                    )
                )

        payload = self._build_stage_payload(
            pr=pr,
            hydration=hydration,
            repo_hydrator=repo_hydrator,
            language=language,
            tool=tool,
            snapshot_targets=snapshot_targets,
            mining_results=mined_refactorings,
        )
        metric_values = payload["refactoring_metrics"]["metrics"]

        _log_info(
            "PR {pr_number}: mined {count} operations, latest retention={rate:.3f}".format(
                pr_number=pr_number,
                count=metric_values["refactor_count"],
                rate=metric_values["refactor_latest_retention_rate"],
            )
        )
        return payload
