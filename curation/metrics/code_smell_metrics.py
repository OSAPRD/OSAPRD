"""Code-smell detection backed by external static-analysis tools.

This stage is an active companion to Multimetric. It normalizes findings from
DesigniteJava, DesignitePython/DPy, PMD, ESLint, Cppcheck, and clang-tidy into
the same finding schema used by aggregate curation outputs.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from curation.config.code_smell_standardization_config import canonicalize_code_smell_type
from curation.config.code_smell_taxonomy_config import (
    CODE_SMELL_TAXONOMY_VERSION,
    classify_code_smell_taxonomy,
)
from curation.config.maintainability_config import (
    get_code_smell_tool_command_templates,
    get_code_smell_tool_timeout_seconds,
)
from curation.config.run_config import CPP_TOOL_CONCURRENCY_LIMIT, EXTERNAL_TOOL_CONCURRENCY_LIMIT
from curation.metrics.multimetric_runner import SnapshotTask, discover_source_files, json_dumps
from curation.utility.subprocess_output import (
    DEFAULT_OUTPUT_PARSE_BYTES,
    read_text_if_within_limit,
    run_command_file_backed,
)
from curation.utility.tool_concurrency import tool_slot

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".c": "c++",
    ".cc": "c++",
    ".cpp": "c++",
    ".cxx": "c++",
    ".h": "c++",
    ".hh": "c++",
    ".hpp": "c++",
    ".hxx": "c++",
}
TOOL_KEYS_BY_LANGUAGE = {
    "java": ("designite_java", "pmd"),
    "python": ("designite_python",),
    "javascript": ("eslint",),
    "c++": ("cppcheck", "clang_tidy"),
}
TOOL_DISPLAY_NAMES = {
    "designite_java": "DesigniteJava",
    "designite_python": "DesignitePython",
    "pmd": "PMD",
    "eslint": "ESLint",
    "cppcheck": "Cppcheck",
    "clang_tidy": "clang-tidy",
}
CODE_SMELL_SELECTED_TOOLS = tuple(TOOL_DISPLAY_NAMES.values())


def _compact_text(value: Any, *, max_chars: int = 1200) -> str:
    """Return bounded text for persisted tool notes and previews."""
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated {len(text) - max_chars} chars]"


def _quote_arg(value: str | Path) -> str:
    """Quote a template argument that may contain spaces."""
    text = str(value)
    escaped = text.replace('"', '\\"')
    return f'"{escaped}"'


def _format_template(template: str, *, root: Path, out: Path, file_path: Path | None = None) -> str:
    """Fill supported command-template placeholders without touching tool placeholders."""
    result = str(template or "")
    result = result.replace("{root}", _quote_arg(root))
    result = result.replace("{out}", _quote_arg(out))
    if file_path is not None:
        result = result.replace("{file}", _quote_arg(file_path))
    return result


def _split_command(command_text: str) -> List[str]:
    """Split a command template into subprocess arguments."""
    parts = shlex.split(command_text, posix=os.name != "nt")
    if os.name == "nt":
        return [part.strip('"') for part in parts]
    return parts


def _command_unavailable(command: Sequence[str]) -> str | None:
    """Return a human-readable missing-tool reason, or None when runnable."""
    if not command:
        return "empty command template"
    executable = str(command[0])
    if executable.lower() == "java" and "-jar" in command:
        jar_index = list(command).index("-jar") + 1
        if jar_index >= len(command):
            return "java -jar command is missing a jar path"
        jar_path = Path(str(command[jar_index]).strip('"'))
        if not jar_path.exists():
            return f"jar not found: {jar_path}"
        return None if shutil.which(executable) else f"executable not found: {executable}"
    if executable.lower() in {"python", "python.exe"} and len(command) > 1:
        script = Path(str(command[1]).strip('"'))
        if script.suffix and not script.exists():
            return f"script not found: {script}"
    if Path(executable).exists():
        return None
    return None if shutil.which(executable) else f"executable not found: {executable}"


def _fresh_tool_output_dir(out_root: Path, tool_key: str) -> Path:
    """Recreate one generated tool-output directory before a fresh tool run."""
    resolved_root = out_root.resolve()
    out_dir = out_root / tool_key
    resolved_out_dir = out_dir.resolve()
    try:
        resolved_out_dir.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"tool output path escapes output root: {out_dir}") from exc
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _read_output(path: str | None) -> str:
    """Read a bounded subprocess output file."""
    text = read_text_if_within_limit(path, max_bytes=DEFAULT_OUTPUT_PARSE_BYTES)
    return text if isinstance(text, str) else ""


def _source_files_by_language(snapshot_path: Path) -> Dict[str, List[Path]]:
    """Return supported source files grouped by language."""
    grouped: Dict[str, List[Path]] = {}
    for path in discover_source_files(snapshot_path):
        language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
        if language:
            grouped.setdefault(language, []).append(path)
    return grouped


def _languages_for_snapshot(snapshot_path: Path) -> List[str]:
    """Infer languages present in one hydrated snapshot."""
    grouped = _source_files_by_language(snapshot_path)
    return [
        language
        for language in ("java", "python", "javascript", "c++")
        if grouped.get(language)
    ]


def _clean_key(value: Any) -> str:
    """Normalize a loose CSV/JSON key for parser matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _pick_value(record: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    """Pick a value from a loose external-tool record."""
    normalized = {_clean_key(key): value for key, value in record.items()}
    for key in keys:
        value = normalized.get(_clean_key(key))
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _safe_int(value: Any) -> int | None:
    """Coerce line/column values to integers when possible."""
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _relative_path(value: Any, root: Path) -> str | None:
    """Normalize an external path into a snapshot-relative path when possible."""
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip().strip('"').replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            return path.name or raw
    return raw.lstrip("./")


def _finding(
    *,
    tool: str,
    snapshot: SnapshotTask,
    language: str | None,
    rule_id: str | None,
    category: str | None,
    severity: str | None,
    file_path: str | None,
    line: int | None,
    column: int | None,
    message: str | None,
) -> Dict[str, Any]:
    """Build one standardized code-smell finding."""
    standardized = canonicalize_code_smell_type(
        rule_id=rule_id,
        category=category,
        message=message,
        tool=tool,
    )
    public_rule = standardized or rule_id or tool
    taxonomy = classify_code_smell_taxonomy(
        rule_id=public_rule,
        category=category,
        message=message,
        language=language,
        file_path=file_path,
    )
    return {
        "tool": tool,
        "snapshot_label": snapshot.snapshot_label,
        "snapshot_commit": snapshot.snapshot_commit,
        "language": language,
        "category": category or "maintainability",
        "raw_rule_id": rule_id,
        "rule_id": public_rule,
        "standardized_smell_type": standardized,
        "severity": severity or "info",
        "file_path": file_path,
        "line": line,
        "end_line": line,
        "column": column,
        "message": _compact_text(message or rule_id or tool, max_chars=500),
        "taxonomy": taxonomy,
    }


def _parse_designite_csv(path: Path, *, snapshot: SnapshotTask, root: Path, language: str) -> List[Dict[str, Any]]:
    """Parse a Designite CSV smell report."""
    findings: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return findings
    for row in rows:
        if not isinstance(row, dict):
            continue
        smell = _pick_value(row, ("smell", "smell name", "smellname", "name", "rule", "type"))
        if not smell:
            continue
        file_path = _relative_path(
            _pick_value(row, ("file", "file path", "filepath", "path", "package")),
            root,
        )
        findings.append(
            _finding(
                tool="DesigniteJava" if language == "java" else "DesignitePython",
                snapshot=snapshot,
                language=language,
                rule_id=smell,
                category=_pick_value(row, ("category", "smell category", "level")),
                severity=_pick_value(row, ("severity", "priority")),
                file_path=file_path,
                line=_safe_int(_pick_value(row, ("line", "start line", "beginline"))),
                column=_safe_int(_pick_value(row, ("column", "start column", "begincolumn"))),
                message=_pick_value(row, ("description", "message", "details")) or smell,
            )
        )
    return findings


def _iter_json_records(value: Any) -> Iterable[Dict[str, Any]]:
    """Yield dict records from common nested JSON report shapes."""
    if isinstance(value, dict):
        yielded_child = False
        for key in ("smells", "issues", "violations", "results", "data", "items"):
            child = value.get(key)
            if isinstance(child, (list, dict)):
                yielded_child = True
                yield from _iter_json_records(child)
        if not yielded_child:
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_records(item)


def _parse_designite_json(path: Path, *, snapshot: SnapshotTask, root: Path, language: str) -> List[Dict[str, Any]]:
    """Parse a Designite JSON smell report."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    findings: List[Dict[str, Any]] = []
    for row in _iter_json_records(payload):
        smell = _pick_value(row, ("smell", "name", "rule", "type"))
        if not smell:
            continue
        findings.append(
            _finding(
                tool="DesigniteJava" if language == "java" else "DesignitePython",
                snapshot=snapshot,
                language=language,
                rule_id=smell,
                category=_pick_value(row, ("category", "smellCategory", "level")),
                severity=_pick_value(row, ("severity", "priority")),
                file_path=_relative_path(_pick_value(row, ("file", "filePath", "path")), root),
                line=_safe_int(_pick_value(row, ("line", "startLine", "beginLine"))),
                column=_safe_int(_pick_value(row, ("column", "startColumn", "beginColumn"))),
                message=_pick_value(row, ("description", "message", "details")) or smell,
            )
        )
    return findings


def _parse_designite_outputs(out_dir: Path, *, snapshot: SnapshotTask, root: Path, language: str) -> List[Dict[str, Any]]:
    """Parse all likely Designite smell report files under an output directory."""
    findings: List[Dict[str, Any]] = []
    if not out_dir.exists():
        return findings
    for csv_path in out_dir.rglob("*.csv"):
        findings.extend(_parse_designite_csv(csv_path, snapshot=snapshot, root=root, language=language))
    for json_path in out_dir.rglob("*.json"):
        findings.extend(_parse_designite_json(json_path, snapshot=snapshot, root=root, language=language))
    return findings


def _parse_pmd(stdout: str, stderr: str, *, snapshot: SnapshotTask, root: Path) -> List[Dict[str, Any]]:
    """Parse PMD JSON output."""
    text = stdout.strip() or stderr.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    findings: List[Dict[str, Any]] = []
    for file_item in payload.get("files", []) if isinstance(payload, dict) else []:
        if not isinstance(file_item, dict):
            continue
        file_path = _relative_path(file_item.get("filename"), root)
        for violation in file_item.get("violations", []) or []:
            if not isinstance(violation, dict):
                continue
            rule = str(violation.get("rule") or violation.get("ruleName") or "PMD").strip()
            findings.append(
                _finding(
                    tool="PMD",
                    snapshot=snapshot,
                    language="java",
                    rule_id=rule,
                    category=str(violation.get("ruleset") or "maintainability"),
                    severity=str(violation.get("priority") or "info"),
                    file_path=file_path,
                    line=_safe_int(violation.get("beginline")),
                    column=_safe_int(violation.get("begincolumn")),
                    message=str(violation.get("description") or rule),
                )
            )
    return findings


def _eslint_severity(value: Any) -> str:
    """Map ESLint numeric severity to the shared severity vocabulary."""
    try:
        severity = int(value or 0)
    except (TypeError, ValueError):
        severity = 0
    if severity >= 2:
        return "high"
    if severity == 1:
        return "medium"
    return "info"


def _eslint_category(rule: str, message: Any) -> str:
    """Classify ESLint rules into a coarse smell category."""
    text = f"{rule} {message or ''}".lower()
    if any(keyword in text for keyword in ("complexity", "cognitive", "depth", "max-")):
        return "design"
    return "maintainability"


def _parse_eslint(stdout: str, stderr: str, *, snapshot: SnapshotTask, root: Path) -> List[Dict[str, Any]]:
    """Parse ESLint JSON output."""
    text = stdout.strip() or stderr.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []

    findings: List[Dict[str, Any]] = []
    for file_item in payload:
        if not isinstance(file_item, dict):
            continue
        file_path = _relative_path(file_item.get("filePath"), root)
        for message in file_item.get("messages", []) or []:
            if not isinstance(message, dict):
                continue
            rule = str(message.get("ruleId") or "eslint").strip()
            findings.append(
                _finding(
                    tool="ESLint",
                    snapshot=snapshot,
                    language="javascript",
                    rule_id=rule,
                    category=_eslint_category(rule, message.get("message")),
                    severity=_eslint_severity(message.get("severity")),
                    file_path=file_path,
                    line=_safe_int(message.get("line")),
                    column=_safe_int(message.get("column")),
                    message=str(message.get("message") or rule),
                )
            )
    return findings


_CPPCHECK_RE = re.compile(
    r"^(?P<file>.*?):(?P<line>\d+):(?P<rule>[^:]+):(?P<severity>[^:]+):(?P<message>.*)$"
)
_CLANG_TIDY_RE = re.compile(
    r"^(?P<file>.*?):(?P<line>\d+):(?P<column>\d+):\s*(?P<severity>warning|error|note):\s*(?P<message>.*?)(?:\s*\[(?P<rule>[^\]]+)\])?\s*$",
    re.IGNORECASE,
)


def _parse_cppcheck(text: str, *, snapshot: SnapshotTask, root: Path) -> List[Dict[str, Any]]:
    """Parse Cppcheck template output."""
    findings: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        match = _CPPCHECK_RE.match(raw_line.strip())
        if not match:
            continue
        findings.append(
            _finding(
                tool="Cppcheck",
                snapshot=snapshot,
                language="c++",
                rule_id=match.group("rule"),
                category="maintainability",
                severity=match.group("severity"),
                file_path=_relative_path(match.group("file"), root),
                line=_safe_int(match.group("line")),
                column=None,
                message=match.group("message"),
            )
        )
    return findings


def _parse_clang_tidy(text: str, *, snapshot: SnapshotTask, root: Path) -> List[Dict[str, Any]]:
    """Parse clang-tidy diagnostics."""
    findings: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        match = _CLANG_TIDY_RE.match(raw_line.strip())
        if not match:
            continue
        rule = match.group("rule") or "clang-tidy"
        findings.append(
            _finding(
                tool="clang-tidy",
                snapshot=snapshot,
                language="c++",
                rule_id=rule,
                category="maintainability",
                severity=match.group("severity"),
                file_path=_relative_path(match.group("file"), root),
                line=_safe_int(match.group("line")),
                column=_safe_int(match.group("column")),
                message=match.group("message"),
            )
        )
    return findings


def _count_by_field(findings: List[Dict[str, Any]], field_name: str) -> Dict[str, int]:
    """Count findings by a top-level field."""
    counts: Dict[str, int] = {}
    for finding in findings:
        value = finding.get(field_name)
        key = str(value) if value else "unclassified"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_by_taxonomy(findings: List[Dict[str, Any]], taxonomy_key: str) -> Dict[str, int]:
    """Count findings by one taxonomy field."""
    counts: Dict[str, int] = {}
    for finding in findings:
        taxonomy = finding.get("taxonomy") if isinstance(finding.get("taxonomy"), dict) else {}
        value = taxonomy.get(taxonomy_key)
        key = str(value) if value else "unclassified"
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_code_smells(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return aggregate smell metrics for a list of normalized findings."""
    smell_type_count = _count_by_field(findings, "rule_id")
    return {
        "smell_count": len(findings),
        "smell_count_by_tool": _count_by_field(findings, "tool"),
        "smell_count_by_snapshot": _count_by_field(findings, "snapshot_label"),
        "smell_count_by_rule": smell_type_count,
        "smell_type_count": smell_type_count,
        "smell_count_by_severity": _count_by_field(findings, "severity"),
        "smell_count_by_category": _count_by_field(findings, "category"),
        "smell_count_by_mantyla": _count_by_taxonomy(findings, "mantyla"),
        "taxonomy_version": CODE_SMELL_TAXONOMY_VERSION,
        "smell_file_count": len({finding["file_path"] for finding in findings if finding.get("file_path")}),
    }


class CodeSmellDetectionStage:
    """Run active code-smell tools for one maintainability snapshot."""

    def __init__(self) -> None:
        self.templates = get_code_smell_tool_command_templates()
        self.timeout_seconds = get_code_smell_tool_timeout_seconds()

    def _selected_tool_keys(self, root: Path) -> List[str]:
        languages = _languages_for_snapshot(root)
        selected: List[str] = []
        for language in languages:
            selected.extend(TOOL_KEYS_BY_LANGUAGE.get(language, ()))
        return list(dict.fromkeys(selected))

    def _run_command(
        self,
        *,
        tool_key: str,
        command: List[str],
        cwd: Path,
        log_dir: Path,
    ):
        limit_name = "cpp_tools" if tool_key in {"cppcheck", "clang_tidy"} else "external_tools"
        limit = CPP_TOOL_CONCURRENCY_LIMIT if limit_name == "cpp_tools" else EXTERNAL_TOOL_CONCURRENCY_LIMIT
        with tool_slot(limit_name, max(0, int(limit))):
            return run_command_file_backed(
                command,
                cwd=cwd,
                output_dir=log_dir,
                label=f"code-smell-{tool_key}",
                timeout_seconds=self.timeout_seconds,
            )

    def _tool_run_not_configured(self, *, tool_key: str, command: List[str], reason: str) -> Dict[str, Any]:
        return {
            "tool": tool_key,
            "display_tool": TOOL_DISPLAY_NAMES.get(tool_key, tool_key),
            "status": "tool_not_configured",
            "command": command,
            "return_code": None,
            "timed_out": False,
            "notes": reason,
        }

    def _run_one_template(
        self,
        *,
        tool_key: str,
        root: Path,
        out_dir: Path,
        log_dir: Path,
        file_path: Path | None = None,
    ) -> Dict[str, Any]:
        template = self.templates.get(tool_key, "")
        command_text = _format_template(template, root=root, out=out_dir, file_path=file_path)
        command = _split_command(command_text)
        missing = _command_unavailable(command)
        if missing:
            return self._tool_run_not_configured(tool_key=tool_key, command=command, reason=missing)
        try:
            result = self._run_command(tool_key=tool_key, command=command, cwd=root, log_dir=log_dir)
        except Exception as exc:
            return {
                "tool": tool_key,
                "display_tool": TOOL_DISPLAY_NAMES.get(tool_key, tool_key),
                "status": "invocation_failed",
                "command": command,
                "return_code": None,
                "timed_out": False,
                "notes": str(exc),
            }
        accepted_nonzero = result.returncode in {1, 4}
        status = (
            "timed_out"
            if result.timed_out
            else ("completed" if result.returncode == 0 or accepted_nonzero else "tool_failed")
        )
        return {
            "tool": tool_key,
            "display_tool": TOOL_DISPLAY_NAMES.get(tool_key, tool_key),
            "status": status,
            "command": result.command,
            "return_code": result.returncode,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "stdout_path": result.stdout_path,
            "stderr_path": result.stderr_path,
            "stdout_preview": _compact_text(result.stdout_preview, max_chars=1000),
            "stderr_preview": _compact_text(result.stderr_preview, max_chars=1500),
            "notes": result.termination_reason,
        }

    def _run_clang_tidy(self, *, root: Path, out_dir: Path, log_dir: Path) -> List[Dict[str, Any]]:
        template = self.templates.get("clang_tidy", "")
        if "{file}" not in template:
            return [
                self._run_one_template(
                    tool_key="clang_tidy",
                    root=root,
                    out_dir=out_dir,
                    log_dir=log_dir,
                )
            ]
        cpp_files = _source_files_by_language(root).get("c++", [])
        if not cpp_files:
            return []
        runs: List[Dict[str, Any]] = []
        for file_path in cpp_files:
            runs.append(
                self._run_one_template(
                    tool_key="clang_tidy",
                    root=root,
                    out_dir=out_dir,
                    log_dir=log_dir,
                    file_path=file_path,
                )
            )
        return runs

    def analyze_snapshot(self, task: SnapshotTask) -> Dict[str, Any]:
        """Run smell tools for one snapshot and return normalized findings."""
        root = task.snapshot_path
        if root is None or not root.exists() or not root.is_dir():
            return {
                "status": "missing_snapshot",
                "snapshot_label": task.snapshot_label,
                "issue_count": 0,
                "findings": [],
                "tool_runs": [],
                "summary": summarize_code_smells([]),
            }

        maintainability_root = root / "maintainability"
        out_root = maintainability_root / "code-smell-tool-output"
        log_dir = maintainability_root / "tool-logs"
        out_root.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        selected = self._selected_tool_keys(root)
        tool_runs: List[Dict[str, Any]] = []
        findings: List[Dict[str, Any]] = []
        for tool_key in selected:
            out_dir = _fresh_tool_output_dir(out_root, tool_key)
            runs = (
                self._run_clang_tidy(root=root, out_dir=out_dir, log_dir=log_dir)
                if tool_key == "clang_tidy"
                else [
                    self._run_one_template(
                        tool_key=tool_key,
                        root=root,
                        out_dir=out_dir,
                        log_dir=log_dir,
                    )
                ]
            )
            tool_runs.extend(runs)
            stdout = "\n".join(_read_output(run.get("stdout_path")) for run in runs)
            stderr = "\n".join(_read_output(run.get("stderr_path")) for run in runs)
            if tool_key == "designite_java":
                findings.extend(_parse_designite_outputs(out_dir, snapshot=task, root=root, language="java"))
            elif tool_key == "designite_python":
                findings.extend(_parse_designite_outputs(out_dir, snapshot=task, root=root, language="python"))
            elif tool_key == "pmd":
                findings.extend(_parse_pmd(stdout, stderr, snapshot=task, root=root))
            elif tool_key == "eslint":
                findings.extend(_parse_eslint(stdout, stderr, snapshot=task, root=root))
            elif tool_key == "cppcheck":
                findings.extend(_parse_cppcheck("\n".join((stdout, stderr)), snapshot=task, root=root))
            elif tool_key == "clang_tidy":
                findings.extend(_parse_clang_tidy("\n".join((stdout, stderr)), snapshot=task, root=root))

        statuses = {str(run.get("status") or "") for run in tool_runs}
        if not selected:
            status = "not_applicable"
        elif statuses and statuses <= {"tool_not_configured"}:
            status = "tool_not_configured"
        elif statuses.intersection({"completed"}) and statuses.intersection(
            {"tool_failed", "invocation_failed", "timed_out", "tool_not_configured"}
        ):
            status = "partial_success"
        elif statuses == {"completed"}:
            status = "success"
        elif statuses.intersection({"completed"}):
            status = "partial_success"
        else:
            status = "failed"

        artifact_path = maintainability_root / "code_smell_tool_results.json"
        summary = summarize_code_smells(findings)
        payload = {
            "schema_version": 1,
            "engine": "code_smell_tools",
            "snapshot_label": task.snapshot_label,
            "snapshot_commit": task.snapshot_commit,
            "status": status,
            "selected_tools": [TOOL_DISPLAY_NAMES.get(key, key) for key in selected],
            "tool_runs": tool_runs,
            "summary": summary,
            "findings": findings,
            "findings_json": json_dumps(findings),
        }
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {
            "status": status,
            "snapshot_label": task.snapshot_label,
            "snapshot_commit": task.snapshot_commit,
            "artifact_path": str(artifact_path),
            "issue_count": len(findings),
            "findings": findings,
            "summary": summary,
            "selected_tools": payload["selected_tools"],
            "tool_runs": tool_runs,
        }
