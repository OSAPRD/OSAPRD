"""Language-selection helpers shared by preprocessing, sampling, and metrics.

The curation pipeline keeps one effective benchmark language per PR. Selection
is based on changed lines first, changed file count second, and a deterministic
tie-break priority last.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from extraction.utility.language_labeller import infer_language

_LANGUAGE_ALIASES = {
    "javascript": "javascript",
    "js": "javascript",
    "python": "python",
    "java": "java",
    "c++": "c++",
    "cpp": "c++",
}


def normalize_language(language: Optional[str]) -> Optional[str]:
    """Normalize user-facing language labels to canonical lower-case tokens."""
    if not language:
        return None
    return _LANGUAGE_ALIASES.get(language.strip().lower(), language.strip().lower())


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dict row or DTO-like object."""
    return getattr(obj, name, default) if not isinstance(obj, dict) else obj.get(name, default)


def _safe_int(value: Any) -> int:
    """Convert numeric-ish values to int, returning 0 for missing/bad values."""
    try:
        if value is None:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def dominant_pr_language(
    pr: Any,
    *,
    supported_languages: Iterable[str],
    tie_break_priority: Iterable[str],
) -> Optional[str]:
    """Return the deterministic benchmark language for one PR, if available."""
    supported = {normalize_language(item) for item in supported_languages}
    supported.discard(None)

    explicit_effective = normalize_language(_get_attr(pr, "pr_primary_language_effective"))
    if explicit_effective in supported:
        return explicit_effective

    raw_files = _get_attr(pr, "files")
    if raw_files is None:
        files: list[Any] = []
    elif isinstance(raw_files, list):
        files = raw_files
    else:
        to_list = getattr(raw_files, "tolist", None)
        if callable(to_list):
            try:
                converted = to_list()
                files = converted if isinstance(converted, list) else [converted]
            except Exception:
                try:
                    files = list(raw_files)
                except Exception:
                    files = [raw_files]
        else:
            try:
                files = list(raw_files)
            except Exception:
                files = [raw_files]

    loc_by_language: Dict[str, int] = {}
    files_by_language: Dict[str, int] = {}
    for fc in files:
        path = _get_attr(fc, "path")
        language = normalize_language(_get_attr(fc, "language")) or normalize_language(
            infer_language(path)
        )
        if language not in supported:
            continue
        changed_loc = _safe_int(_get_attr(fc, "additions")) + _safe_int(
            _get_attr(fc, "deletions")
        )
        loc_by_language[language] = loc_by_language.get(language, 0) + changed_loc
        files_by_language[language] = files_by_language.get(language, 0) + 1

    if not loc_by_language:
        # Fallback to PR-level language labels.
        for lang in (_get_attr(pr, "file_languages") or []):
            normalized = normalize_language(str(lang))
            if normalized in supported:
                return normalized
        return None

    priority_rank = {
        normalize_language(lang): index
        for index, lang in enumerate(tie_break_priority)
        if normalize_language(lang)
    }
    ranked = sorted(
        loc_by_language.keys(),
        key=lambda lang: (
            -loc_by_language.get(lang, 0),
            -files_by_language.get(lang, 0),
            priority_rank.get(lang, 10_000),
            lang,
        ),
    )
    return ranked[0] if ranked else None
