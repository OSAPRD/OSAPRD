"""Preprocessing pipeline for language labeling and scientific filtering.

This stage reads local parquet rows, normalizes them into PR DTOs or lightweight
sampling candidates, infers changed-file languages, and filters to merged PRs
whose effective language is one of the configured benchmark languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from curation.config.run_config import ONLY_MERGED_PRS, TARGET_LANGUAGES
from curation.utility.language_selection import dominant_pr_language
from curation.utility.loader import iter_pr_rows, load_prs, load_prs_by_urls
from extraction.dtos.dtos import FileChange, PullRequest
from extraction.utility.language_labeller import infer_language


@dataclass
class SamplingCandidatePR:
    """Lightweight PR row used during streaming sampling."""

    id: str
    url: str
    number: int
    created_at: str
    merged_at: Optional[str]
    primary_language: Optional[str]
    pr_primary_language_effective: Optional[str]
    file_languages: List[str]
    additions: int
    deletions: int
    base_repository_full: Optional[Dict[str, Any]]
    base_repository: Optional[Dict[str, Any]]
    files: Optional[List[Dict[str, Any]]] = None


def _normalize_languages(languages: List[str]) -> set[str]:
    """Normalize a list of language labels to lowercase for comparison."""
    return {lang.strip().lower() for lang in languages if lang}


def _safe_int(value: Any) -> int:
    """Convert numeric-ish values to int, returning 0 for missing/bad values."""
    try:
        if value is None:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _primary_language_from_files(files: List[FileChange]) -> Optional[str]:
    """
    Compute primary language using:
    1) changed LOC per language, 2) changed file count, 3) lexical tie break.
    """
    loc_by_language: Dict[str, int] = {}
    files_by_language: Dict[str, int] = {}
    for fc in files:
        if not isinstance(fc, FileChange):
            continue
        language = fc.language or infer_language(fc.path)
        if not language:
            continue
        changed_loc = _safe_int(fc.additions) + _safe_int(fc.deletions)
        loc_by_language[language] = loc_by_language.get(language, 0) + changed_loc
        files_by_language[language] = files_by_language.get(language, 0) + 1
    if not loc_by_language:
        return None
    ranked = sorted(
        loc_by_language.keys(),
        key=lambda lang: (
            -loc_by_language.get(lang, 0),
            -files_by_language.get(lang, 0),
            str(lang).lower(),
        ),
    )
    return ranked[0] if ranked else None


def _primary_language_from_file_rows(files: List[Dict[str, Any]]) -> Optional[str]:
    """Dictionary variant of _primary_language_from_files for streaming pass-1."""
    loc_by_language: Dict[str, int] = {}
    files_by_language: Dict[str, int] = {}
    for fc in files:
        if not isinstance(fc, dict):
            continue
        language = fc.get("language") or infer_language(fc.get("path"))
        if not language:
            continue
        changed_loc = _safe_int(fc.get("additions")) + _safe_int(fc.get("deletions"))
        loc_by_language[language] = loc_by_language.get(language, 0) + changed_loc
        files_by_language[language] = files_by_language.get(language, 0) + 1
    if not loc_by_language:
        return None
    ranked = sorted(
        loc_by_language.keys(),
        key=lambda lang: (
            -loc_by_language.get(lang, 0),
            -files_by_language.get(lang, 0),
            str(lang).lower(),
        ),
    )
    return ranked[0] if ranked else None


def _effective_primary_language(pr: PullRequest) -> Optional[str]:
    """Resolve the benchmark language used for filtering and stratification."""
    pr_view = {
        "files": getattr(pr, "files", None),
        "file_languages": getattr(pr, "file_languages", None),
        "primary_language": getattr(pr, "primary_language", None),
        "pr_primary_language_effective": None,
    }
    return dominant_pr_language(
        pr_view,
        supported_languages=tuple(lang.lower() for lang in TARGET_LANGUAGES if lang),
        tie_break_priority=("c++", "java", "javascript", "python"),
    )


def _coerce_file_rows(raw_files: Any) -> List[Dict[str, Any]]:
    """Normalize a raw files column into a list of dictionaries."""
    if raw_files is None:
        return []
    if isinstance(raw_files, list):
        files = raw_files
    else:
        try:
            files = list(raw_files)
        except Exception:
            files = []
    return [fc for fc in files if isinstance(fc, dict)]


def _normalize_repo_dict(repo: Any) -> Optional[Dict[str, Any]]:
    """Normalize nested repository dictionaries for lightweight candidates."""
    if not isinstance(repo, dict):
        return None
    normalized = dict(repo)
    owner = normalized.get("owner")
    if isinstance(owner, dict):
        normalized["owner"] = owner.get("login") or owner.get("name") or owner.get("id")
    return normalized


def preprocess_prs(cohort: Optional[str] = None) -> Tuple[List[PullRequest], Dict[str, Any]]:
    """
    Load PRs for a cohort, label file/PR languages, and filter by target languages.

    Args:
        cohort: "human", "agentic", or a specific agent name (e.g., "claude").

    Returns:
        PullRequest DTOs matching target languages plus summary statistics.
    """
    prs = load_prs(cohort)
    print(f"[preprocess] Loaded PRs: {len(prs)} (cohort={cohort})")
    filtered: List[PullRequest] = []
    target_languages = _normalize_languages(TARGET_LANGUAGES)
    per_language_counts: dict[str, int] = {}
    filtered_out = 0
    filtered_merged = 0
    filtered_added_only = 0
    filtered_primary_notebook = 0
    missing_files = 0
    missing_languages = 0
    unknown_extensions: dict[str, int] = {}

    for pr in prs:
        if ONLY_MERGED_PRS and not pr.merged_at:
            filtered_out += 1
            filtered_merged += 1
            continue
        file_languages = set()
        files = pr.files if pr.files is not None else []
        if not isinstance(files, list):
            files = list(files)
        if not files:
            missing_files += 1
        if files:
            has_modified = False
            for fc in files:
                if not isinstance(fc, FileChange):
                    continue
                change_type = (fc.change_type or fc.status or "").lower()
                if change_type == "modified":
                    has_modified = True
                    break
            if not has_modified:
                filtered_out += 1
                filtered_added_only += 1
                continue
        for fc in files:
            if not isinstance(fc, FileChange):
                continue
            language = infer_language(fc.path)
            fc.language = language
            if language:
                file_languages.add(language)
            else:
                path = fc.path or ""
                if "." in path:
                    ext = path.rsplit(".", 1)[-1].lower()
                else:
                    ext = ""
                if ext:
                    unknown_extensions[ext] = unknown_extensions.get(ext, 0) + 1
        pr.file_languages = sorted(file_languages)
        if not getattr(pr, "primary_language", None):
            pr.primary_language = _primary_language_from_files(files)
        pr.pr_primary_language_effective = _effective_primary_language(pr)
        if str(pr.primary_language or "").strip().lower() == "jupyter notebook":
            filtered_out += 1
            filtered_primary_notebook += 1
            continue
        normalized = _normalize_languages(pr.file_languages)
        if not normalized:
            missing_languages += 1
        for lang in normalized:
            per_language_counts[lang] = per_language_counts.get(lang, 0) + 1
        if not pr.pr_primary_language_effective or (
            target_languages and pr.pr_primary_language_effective not in target_languages
        ):
            filtered_out += 1
            continue
        filtered.append(pr)

    print("[preprocess] PRs per language (post-labeling):")
    for lang in sorted(per_language_counts):
        print(f"  - {lang}: {per_language_counts[lang]}")
    if unknown_extensions:
        unknown_total = sum(unknown_extensions.values())
        print(f"  - unknown: {unknown_total}")
    print(f"[preprocess] PRs filtered out: {filtered_out}")
    if ONLY_MERGED_PRS:
        print(f"[preprocess] Filtered non-merged PRs: {filtered_merged}")
    print(f"[preprocess] Filtered added-only PRs: {filtered_added_only}")
    print(f"[preprocess] Filtered primary-language Jupyter Notebook PRs: {filtered_primary_notebook}")
    print(f"[preprocess] PRs with no files: {missing_files}")
    print(f"[preprocess] PRs with no inferred languages: {missing_languages}")
    print(f"[preprocess] PRs retained: {len(filtered)}")

    stats = {
        "loaded_prs": len(prs),
        "retained_prs": len(filtered),
        "filtered_out_prs": filtered_out,
        "filtered_non_merged": filtered_merged,
        "filtered_added_only": filtered_added_only,
        "filtered_primary_notebook": filtered_primary_notebook,
        "missing_files": missing_files,
        "missing_languages": missing_languages,
        "per_language_counts": dict(sorted(per_language_counts.items())),
        "unknown_language_total": sum(unknown_extensions.values()) if unknown_extensions else 0,
    }
    return filtered, stats


def preprocess_candidates_streaming(
    cohort: Optional[str] = None,
    *,
    batch_size: int = 256,
    exclude_urls: Optional[set[str]] = None,
    exclude_identifiers: Optional[set[str]] = None,
    deduplicate_urls: bool = True,
) -> Tuple[List[SamplingCandidatePR], Dict[str, Any]]:
    """Run streaming pass 1 and emit lightweight candidates for sampling.

    This applies the same language/merge/file-change policy as
    :func:`preprocess_prs` but avoids constructing full PR DTOs until after
    sampling. The optional exclusion sets support sample-history and top-up
    runs without scanning prior output directories implicitly.
    """
    print(f"[preprocess] Streaming pass-1 start (cohort={cohort}, batch_size={batch_size})")
    filtered: List[SamplingCandidatePR] = []
    target_languages = _normalize_languages(TARGET_LANGUAGES)
    per_language_counts: dict[str, int] = {}
    filtered_out = 0
    filtered_merged = 0
    filtered_added_only = 0
    filtered_primary_notebook = 0
    missing_files = 0
    missing_languages = 0
    unknown_extensions: dict[str, int] = {}
    loaded = 0
    seen_urls: set[str] = set()
    excluded_previously_seen = 0
    dedup_skipped = 0
    exclude_urls = exclude_urls or set()
    exclude_identifiers = exclude_identifiers or set()
    pass1_columns = [
        "id",
        "url",
        "number",
        "created_at",
        "merged_at",
        "files",
        "additions",
        "deletions",
        "base_repository_full",
        "base_repository",
    ]

    for row in iter_pr_rows(
        group=cohort,
        batch_size=batch_size,
        columns=pass1_columns,
    ):
        if not isinstance(row, dict):
            continue
        loaded += 1
        url = str(row.get("url") or "").strip()
        if not url:
            filtered_out += 1
            continue
        pr_id = str(row.get("id") or "").strip()
        if deduplicate_urls:
            if url in seen_urls:
                dedup_skipped += 1
                continue
            seen_urls.add(url)
        if (
            url in exclude_urls
            or url.rstrip("/") in exclude_urls
            or url in exclude_identifiers
            or url.rstrip("/") in exclude_identifiers
            or pr_id in exclude_identifiers
        ):
            excluded_previously_seen += 1
            continue
        if ONLY_MERGED_PRS and not row.get("merged_at"):
            filtered_out += 1
            filtered_merged += 1
            continue

        files = _coerce_file_rows(row.get("files"))
        if not files:
            missing_files += 1
        if files:
            has_modified = False
            for fc in files:
                change_type = str(fc.get("change_type") or fc.get("status") or "").lower()
                if change_type == "modified":
                    has_modified = True
                    break
            if not has_modified:
                filtered_out += 1
                filtered_added_only += 1
                continue

        file_languages = set()
        for fc in files:
            language = infer_language(fc.get("path"))
            fc["language"] = language
            if language:
                file_languages.add(language)
            else:
                path = str(fc.get("path") or "")
                ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                if ext:
                    unknown_extensions[ext] = unknown_extensions.get(ext, 0) + 1

        existing_primary_language = str(row.get("primary_language") or "").strip()
        primary_language = existing_primary_language or _primary_language_from_file_rows(files)
        if str(primary_language or "").strip().lower() == "jupyter notebook":
            filtered_out += 1
            filtered_primary_notebook += 1
            continue

        row_view = {
            "files": files,
            "file_languages": sorted(file_languages),
            "primary_language": primary_language,
            "pr_primary_language_effective": None,
        }
        effective = dominant_pr_language(
            row_view,
            supported_languages=tuple(lang.lower() for lang in TARGET_LANGUAGES if lang),
            tie_break_priority=("c++", "java", "javascript", "python"),
        )

        normalized = _normalize_languages(row_view["file_languages"])
        if not normalized:
            missing_languages += 1
        for lang in normalized:
            per_language_counts[lang] = per_language_counts.get(lang, 0) + 1

        if not effective or (target_languages and effective not in target_languages):
            filtered_out += 1
            continue

        candidate = SamplingCandidatePR(
            id=str(row.get("id") or ""),
            url=url,
            number=_safe_int(row.get("number")),
            created_at=str(row.get("created_at") or ""),
            merged_at=(str(row.get("merged_at")) if row.get("merged_at") else None),
            primary_language=primary_language,
            pr_primary_language_effective=effective,
            file_languages=sorted(file_languages),
            additions=_safe_int(row.get("additions")),
            deletions=_safe_int(row.get("deletions")),
            base_repository_full=_normalize_repo_dict(row.get("base_repository_full")),
            base_repository=(
                dict(row.get("base_repository")) if isinstance(row.get("base_repository"), dict) else None
            ),
            # Keep candidates lightweight for downstream sampling/top-up.
            files=None,
        )
        if candidate.url:
            filtered.append(candidate)
        else:
            filtered_out += 1

    print(f"[preprocess] Loaded PRs: {loaded} (cohort={cohort})")
    print("[preprocess] PRs per language (post-labeling):")
    for lang in sorted(per_language_counts):
        print(f"  - {lang}: {per_language_counts[lang]}")
    if unknown_extensions:
        unknown_total = sum(unknown_extensions.values())
        print(f"  - unknown: {unknown_total}")
    print(f"[preprocess] PRs filtered out: {filtered_out}")
    if ONLY_MERGED_PRS:
        print(f"[preprocess] Filtered non-merged PRs: {filtered_merged}")
    print(f"[preprocess] Filtered added-only PRs: {filtered_added_only}")
    print(f"[preprocess] Filtered primary-language Jupyter Notebook PRs: {filtered_primary_notebook}")
    print(f"[preprocess] PRs with no files: {missing_files}")
    print(f"[preprocess] PRs with no inferred languages: {missing_languages}")
    print(f"[preprocess] PRs retained: {len(filtered)}")

    stats = {
        "loaded_prs": loaded,
        "retained_prs": len(filtered),
        "filtered_out_prs": filtered_out,
        "filtered_non_merged": filtered_merged,
        "filtered_added_only": filtered_added_only,
        "filtered_primary_notebook": filtered_primary_notebook,
        "missing_files": missing_files,
        "missing_languages": missing_languages,
        "per_language_counts": dict(sorted(per_language_counts.items())),
        "unknown_language_total": sum(unknown_extensions.values()) if unknown_extensions else 0,
        "excluded_previously_seen": excluded_previously_seen,
        "excluded_sample_history": excluded_previously_seen,
        "dedup_skipped": dedup_skipped,
    }
    return filtered, stats


def materialize_candidates_to_full_prs(
    candidates: List[SamplingCandidatePR],
    cohort: Optional[str] = None,
    *,
    batch_size: int = 256,
) -> List[PullRequest]:
    """Run streaming pass 2 and load full PR DTOs for selected candidate URLs."""
    ordered_urls = [candidate.url for candidate in candidates if candidate.url]
    prs = load_prs_by_urls(ordered_urls, group=cohort, batch_size=batch_size)
    by_url = {pr.url: pr for pr in prs if getattr(pr, "url", None)}
    materialized = [by_url[url] for url in ordered_urls if url in by_url]
    missing = len(ordered_urls) - len(materialized)
    if missing:
        print(f"[preprocess] Warning: pass-2 could not materialize {missing} selected PR(s).")
    return materialized
