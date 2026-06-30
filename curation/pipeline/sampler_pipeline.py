"""Sampling pipeline stage for curation.

The sampler receives already-filtered candidates and creates two local sample
stores: the main curated cohort and the longitudinal subset. Both use the same
language/time/popularity bucket semantics so later comparisons stay aligned.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from curation.config.run_config import (
    POPULARITY_BUCKETS,
    TARGET_LANGUAGES,
    TARGET_NO_PRS,
    TIME_BUCKET_GRANULARITY,
)
from curation.config.storage_config import LOCAL_OUTPUT_DIR
from curation.sampler.sampler import (
    POPULARITY_BUCKET_POLICY,
    _popularity_value,
    _select_language,
    _time_bucket,
    assign_tie_aware_popularity_buckets,
    stratified_sample,
    tie_aware_quantile_cut_points,
)
from extraction.dtos.dtos import PullRequest

SOFT_BALANCE_ALPHA_TIME = 1.0
SOFT_BALANCE_ALPHA_POPULARITY = 1.0


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dict row or DTO-like object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _repo_full_name(pr: PullRequest) -> Optional[str]:
    """Return ``owner/name`` for the PR base repository when available."""
    repo = _get_attr(pr, "base_repository_full")
    if repo is None:
        return None
    if isinstance(repo, dict):
        owner = repo.get("owner")
        name = repo.get("name")
    else:
        owner = getattr(repo, "owner", None)
        name = getattr(repo, "name", None)
    if owner and name:
        return f"{owner}/{name}"
    return None


def _set_repo_popularity_label(pr: PullRequest, label: str) -> None:
    """Persist computed popularity bucket onto PR repository metadata."""
    if not label:
        return
    human_label = _humanize_popularity_label(label)

    def _set_fields(target: Any) -> None:
        if target is None:
            return
        if isinstance(target, dict):
            target["popularity_bucket"] = label
            target["popularity_label"] = human_label
            return
        try:
            setattr(target, "popularity_bucket", label)
        except Exception:
            pass
        try:
            setattr(target, "popularity_label", human_label)
        except Exception:
            pass

    base_full = _get_attr(pr, "base_repository_full")
    _set_fields(base_full)
    base_peek = _get_attr(pr, "base_repository")
    _set_fields(base_peek)
    _set_fields(pr)


def _humanize_popularity_pop_index(index: int) -> str:
    """Convert numeric popularity bucket indexes into display labels."""
    if POPULARITY_BUCKETS == 3:
        labels = ("low", "medium", "high")
        if 0 <= index < len(labels):
            return labels[index]
    if POPULARITY_BUCKETS == 2:
        labels = ("low", "high")
        if 0 <= index < len(labels):
            return labels[index]
    return f"bucket_{index}"


def _humanize_popularity_label(bucket: str) -> str:
    """Return a stable human-readable popularity label from pop-bucket keys."""
    if not bucket:
        return "unknown"
    if bucket.startswith("pop"):
        suffix = bucket[3:]
        if suffix.isdigit():
            return _humanize_popularity_pop_index(int(suffix))
    return bucket


def _pr_payload(pr: PullRequest) -> Dict[str, Any]:
    """Best-effort full payload extraction for sampled-PR metadata."""
    if hasattr(pr, "to_dict"):
        payload = pr.to_dict()
        if isinstance(payload, dict):
            return payload
    try:
        payload = asdict(pr)
        if isinstance(payload, dict):
            return payload
    except TypeError:
        pass
    if hasattr(pr, "__dict__"):
        payload = dict(getattr(pr, "__dict__", {}) or {})
        if isinstance(payload, dict):
            return payload
    return {}


def _write_sample_metadata(
    prs: List[PullRequest],
    cohort: Optional[str],
    *,
    basename: str = "sampled_prs",
    stratification_rows: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Persist sampled PR ids/urls and a structured metadata sidecar."""
    output_dir = Path(LOCAL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_label = (cohort or "all").strip().lower()
    output_path = output_dir / f"{basename}_{cohort_label}.txt"
    with output_path.open("w", encoding="utf-8") as f:
        for pr in prs:
            f.write(f"{pr.id}\t{pr.url}\n")

    rows = stratification_rows or []
    jsonl_path = output_dir / f"{basename}_{cohort_label}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"[sampler_pipeline] Sample list written to: {output_path}")
    print(f"[sampler_pipeline] Sample structured metadata written to: {jsonl_path}")
    return output_path


def _log_balance(name: str, counts: dict[str, int]) -> None:
    """Log basic balance stats and chi-square against uniform."""
    if not counts:
        print(f"[sampler_pipeline] {name} balance: no buckets")
        return
    values = list(counts.values())
    total = sum(values)
    buckets = len(values)
    mean = total / buckets
    if buckets <= 1:
        print(f"[sampler_pipeline] {name} balance: buckets=1 total={total}")
        return
    variance = sum((v - mean) ** 2 for v in values) / buckets
    std = sqrt(variance)
    cv = std / mean if mean else 0.0
    chi = sum(((v - mean) ** 2) / mean for v in values if mean)
    print(
        f"[sampler_pipeline] {name} balance: buckets={buckets} total={total} "
        f"min={min(values)} max={max(values)} mean={mean:.2f} std={std:.2f} cv={cv:.3f} chi2={chi:.2f}"
    )


def sample_prs(
    prs: List[PullRequest],
    cohort: Optional[str] = None,
    *,
    target_no_prs: Optional[int] = None,
    metadata_basename: str = "sampled_prs",
) -> Tuple[List[PullRequest], Dict[str, Any]]:
    """
    Sample PRs with stratification and persist sample metadata.

    Args:
        prs: Candidate PRs after preprocessing.
        cohort: "human", "agentic", or a specific agent name.

    Returns:
        Sampled PRs plus stratification summary statistics.
    """
    target_no_prs = TARGET_NO_PRS if target_no_prs is None else max(0, int(target_no_prs))
    print(f"[sampler_pipeline] Candidates: {len(prs)}")
    popularity_cut_points = tie_aware_quantile_cut_points(
        [_popularity_value(pr) for pr in prs],
        POPULARITY_BUCKETS,
    )
    sampled = stratified_sample(
        prs,
        target_no_prs,
        seed=42,
        target_languages=TARGET_LANGUAGES,
        time_granularity=TIME_BUCKET_GRANULARITY,
        popularity_buckets=POPULARITY_BUCKETS,
        soft_balance_alpha_time=SOFT_BALANCE_ALPHA_TIME,
        soft_balance_alpha_popularity=SOFT_BALANCE_ALPHA_POPULARITY,
    )
    stats: Dict[str, Any] = {
        "candidates": len(prs),
        "sampled": len(sampled),
        "time_counts": {},
        "language_counts": {},
        "popularity_counts": {},
        "popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
        "popularity_cut_points": list(popularity_cut_points),
    }
    stratification_rows: List[Dict[str, Any]] = []
    if sampled:
        language_labels = [lang.lower() for lang in TARGET_LANGUAGES if lang]
        popularity_bucket_by_pr = assign_tie_aware_popularity_buckets(
            prs,
            POPULARITY_BUCKETS,
            "pop",
        )

        time_counts: dict[str, int] = {}
        lang_counts: dict[str, int] = {}
        pop_counts: dict[str, int] = {}
        for pr in sampled:
            time_bucket = _time_bucket(pr, "day")
            time_counts[time_bucket] = time_counts.get(time_bucket, 0) + 1

            lang = _select_language(pr, language_labels) if language_labels else "unknown"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            popularity_value = _popularity_value(pr)
            pop_bucket = popularity_bucket_by_pr.get(id(pr), "pop0")
            pop_counts[pop_bucket] = pop_counts.get(pop_bucket, 0) + 1
            _set_repo_popularity_label(pr, pop_bucket)

            stratification_rows.append(
                {
                    "pr_id": _get_attr(pr, "id"),
                    "pr_number": _get_attr(pr, "number"),
                    "pr_url": _get_attr(pr, "url"),
                    "repo_full_name": _repo_full_name(pr),
                    "created_at": _get_attr(pr, "created_at"),
                    "merged_at": _get_attr(pr, "merged_at"),
                    "selected_language_bucket": lang,
                    "time_bucket_day": time_bucket,
                    "time_bucket_sampling": _time_bucket(pr, TIME_BUCKET_GRANULARITY),
                    "sampling_popularity_value": popularity_value,
                    "popularity_bucket": pop_bucket,
                    "popularity_value": popularity_value,
                    "sampling_popularity_bucket": pop_bucket,
                    "sampling_popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
                    "sampling_popularity_cut_points": list(popularity_cut_points),
                    "target_languages": list(TARGET_LANGUAGES),
                    "time_bucket_granularity": TIME_BUCKET_GRANULARITY,
                    "popularity_buckets": POPULARITY_BUCKETS,
                    "soft_balance_alpha_time": SOFT_BALANCE_ALPHA_TIME,
                    "soft_balance_alpha_popularity": SOFT_BALANCE_ALPHA_POPULARITY,
                    "original_pr_payload": _pr_payload(pr),
                }
            )

        time_line = ", ".join(f"{key}={time_counts[key]}" for key in sorted(time_counts))
        lang_line = ", ".join(f"{key}={lang_counts[key]}" for key in sorted(lang_counts))
        pop_line = ", ".join(f"{key}={pop_counts[key]}" for key in sorted(pop_counts))
        print(f"[sampler_pipeline] Sample distribution by time bucket: {time_line}")
        print(f"[sampler_pipeline] Sample distribution by language bucket: {lang_line}")
        print(f"[sampler_pipeline] Sample distribution by popularity bucket: {pop_line}")
        _log_balance("time", time_counts)
        _log_balance("language", lang_counts)
        _log_balance("popularity", pop_counts)
        stats["time_counts"] = dict(sorted(time_counts.items()))
        stats["language_counts"] = dict(sorted(lang_counts.items()))
        stats["popularity_counts"] = dict(sorted(pop_counts.items()))
    _write_sample_metadata(
        sampled,
        cohort,
        basename=metadata_basename,
        stratification_rows=stratification_rows,
    )
    return sampled, stats


