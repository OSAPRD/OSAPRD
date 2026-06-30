"""Deterministic stratified sampling for curation cohorts.

Sampling balances PRs across benchmark language, time, and repository
popularity. Popularity buckets are tie-aware so repositories with the same star
count are not split across bucket boundaries.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from extraction.dtos.dtos import PullRequest

POPULARITY_BUCKET_POLICY = "tie_aware_quantile"


@dataclass(frozen=True)
class StratificationKey:
    """Bucket key for stratified sampling."""

    language: str
    time_bucket: str
    popularity_bucket: str


def _parse_created_at(value: str | None) -> datetime | None:
    """Parse ISO timestamps into datetime when possible."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _popularity_value(pr: PullRequest) -> int:
    """Return stargazer count for base repo, falling back to 0."""
    repo = pr.base_repository_full
    if repo is None:
        return 0
    if isinstance(repo, dict):
        value = repo.get("stargazer_count")
        return int(value) if value is not None else 0
    if repo.stargazer_count is not None:
        return int(repo.stargazer_count)
    return 0


def _assign_bucket(value: int, cut_points: List[int], prefix: str) -> str:
    """Assign a numeric value into a bucket using cut points."""
    for idx, cut in enumerate(cut_points):
        if value <= cut:
            return f"{prefix}{idx}"
    return f"{prefix}{len(cut_points)}"


def tie_aware_quantile_cut_points_from_counts(
    star_counts: Dict[int, int],
    buckets: int,
) -> List[int]:
    """
    Return quantile cut points without splitting equal star counts.

    Large tied star-count groups may collapse multiple target quantiles into one
    cut point, producing fewer effective buckets but preserving interpretability.
    """
    b = max(1, int(buckets))
    if b <= 1:
        return []
    normalized: Dict[int, int] = {
        max(0, int(stars)): max(0, int(count))
        for stars, count in (star_counts or {}).items()
        if max(0, int(count)) > 0
    }
    total = sum(normalized.values())
    if total <= 0:
        return []
    unique_stars = sorted(normalized)
    if len(unique_stars) <= 1:
        return []

    targets = [(total * idx) / float(b) for idx in range(1, b)]
    cut_points: List[int] = []
    target_idx = 0
    cumulative = 0
    for stars in unique_stars[:-1]:
        cumulative += normalized[stars]
        while target_idx < len(targets) and cumulative >= targets[target_idx]:
            cut_points.append(stars)
            target_idx += 1
    return sorted(set(cut_points))


def tie_aware_quantile_cut_points(
    values: List[int],
    buckets: int,
) -> List[int]:
    """Return tie-aware popularity cut points from raw star-count values."""
    return tie_aware_quantile_cut_points_from_counts(dict(Counter(max(0, int(v)) for v in values)), buckets)


def assign_tie_aware_popularity_bucket(
    stars: int,
    cut_points: List[int],
    prefix: str = "pop",
) -> str:
    """Assign one star count to a tie-aware quantile popularity bucket."""
    return _assign_bucket(max(0, int(stars)), cut_points, prefix)


def assign_tie_aware_popularity_buckets(
    prs: List[PullRequest],
    bins: int,
    prefix: str = "pop",
) -> Dict[int, str]:
    """Assign popularity buckets without splitting PRs that share a star count."""
    values = [_popularity_value(pr) for pr in prs]
    cut_points = tie_aware_quantile_cut_points(values, bins)
    return {
        id(pr): assign_tie_aware_popularity_bucket(_popularity_value(pr), cut_points, prefix)
        for pr in prs
    }


def target_language_quotas(
    total_target: int,
    target_languages: Iterable[str],
) -> Dict[str, int]:
    """Allocate a target count evenly across configured language buckets."""
    langs = [str(v).strip().lower() for v in target_languages if str(v).strip()]
    if not langs or total_target <= 0:
        return {}
    base = total_target // len(langs)
    remainder = total_target % len(langs)
    return {
        lang: base + (1 if idx < remainder else 0)
        for idx, lang in enumerate(langs)
    }


def _allocate_counts_proportional_by_size(
    sizes: Dict[str, int],
    target: int,
) -> Dict[str, int]:
    """Allocate target counts proportionally across capacity-limited partitions."""
    if target <= 0:
        return {k: 0 for k in sizes}
    total = sum(max(0, int(v)) for v in sizes.values())
    if total <= target:
        return {k: max(0, int(v)) for k, v in sizes.items()}

    allocations: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    allocated = 0
    for key, size in sizes.items():
        s = max(0, int(size))
        if s <= 0:
            allocations[key] = 0
            continue
        exact = (s / total) * target
        base = min(s, int(exact))
        allocations[key] = base
        allocated += base
        remainders.append((exact - base, key))

    remaining = target - allocated
    if remaining > 0:
        remainders.sort(reverse=True)
        idx = 0
        while remaining > 0 and remainders:
            _, key = remainders[idx % len(remainders)]
            if allocations[key] < max(0, int(sizes.get(key, 0))):
                allocations[key] += 1
                remaining -= 1
            idx += 1
            if idx > len(remainders) * 4 and all(
                allocations[name] >= max(0, int(sizes.get(name, 0)))
                for _, name in remainders
            ):
                break
    return allocations


def stratified_sample_rows_from_existing_buckets(
    rows: List[Dict[str, Any]],
    *,
    target: int,
    target_languages: Iterable[str],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample already-materialized rows using their existing language/time/popularity buckets.

    This is used for longitudinal selection so bucket semantics stay aligned with
    the main sampled universe instead of being recalculated on the subset.
    """
    if target <= 0 or not rows:
        return []
    if len(rows) <= target:
        return list(rows)

    partitions: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        lang = str(row.get("sampling_language_bucket") or "unknown")
        time_bucket = str(row.get("sampling_time_bucket") or "unknown")
        pop_bucket = str(row.get("sampling_popularity_bucket") or "unknown")
        partitions.setdefault((lang, time_bucket, pop_bucket), []).append(row)

    def _key_id(key: Tuple[str, str, str]) -> str:
        return "\x1f".join(key)

    target_alloc: Dict[Tuple[str, str, str], int] = {key: 0 for key in partitions}
    quotas = target_language_quotas(target, target_languages)
    by_lang: Dict[str, List[Tuple[str, str, str]]] = {}
    for key in partitions:
        by_lang.setdefault(key[0], []).append(key)

    unfilled = 0
    for lang, desired in quotas.items():
        keys = by_lang.get(lang, [])
        sizes = {_key_id(key): len(partitions[key]) for key in keys}
        allocation = _allocate_counts_proportional_by_size(sizes, desired)
        filled = 0
        id_to_key = {_key_id(key): key for key in keys}
        for key_id, count in allocation.items():
            key = id_to_key.get(key_id)
            if key is None:
                continue
            target_alloc[key] += int(count)
            filled += int(count)
        if filled < desired:
            unfilled += desired - filled

    if unfilled > 0:
        remaining_sizes = {
            _key_id(key): max(0, len(items) - int(target_alloc.get(key, 0)))
            for key, items in partitions.items()
        }
        extra_allocation = _allocate_counts_proportional_by_size(remaining_sizes, unfilled)
        id_to_key = {_key_id(key): key for key in partitions}
        for key_id, count in extra_allocation.items():
            key = id_to_key.get(key_id)
            if key is not None:
                target_alloc[key] += int(count)

    selected: List[Dict[str, Any]] = []
    for key in sorted(partitions):
        take = min(int(target_alloc.get(key, 0)), len(partitions[key]))
        if take <= 0:
            continue
        items = list(partitions[key])
        part_seed = int(
            hashlib.sha1(f"{seed}:{_key_id(key)}".encode("utf-8")).hexdigest()[:8],
            16,
        )
        random.Random(part_seed).shuffle(items)
        selected.extend(items[:take])

    if len(selected) > target:
        random.Random(seed).shuffle(selected)
        selected = selected[:target]
    elif len(selected) < target:
        selected_urls = {str(row.get("url") or "").strip() for row in selected}
        remaining = [
            row
            for row in rows
            if str(row.get("url") or "").strip() not in selected_urls
        ]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: max(0, target - len(selected))])

    return selected


def _time_bucket(pr: PullRequest, granularity: str = "month") -> str:
    """Bucket by creation time at the requested granularity."""
    created = _parse_created_at(pr.created_at)
    if not created:
        return "unknown"
    granularity = (granularity or "month").strip().lower()
    if granularity == "hour":
        return created.strftime("%Y-%m-%d-%H")
    if granularity == "day":
        return created.strftime("%Y-%m-%d")
    return created.strftime("%Y-%m")


def _select_language(pr: PullRequest, target_languages: List[str]) -> str:
    """Choose a single target language for stratification."""
    effective = (getattr(pr, "pr_primary_language_effective", None) or "").strip().lower()
    if effective in {lang.lower() for lang in target_languages}:
        return effective
    primary = (getattr(pr, "primary_language", None) or "").strip().lower()
    if primary in {lang.lower() for lang in target_languages}:
        return primary
    available = {lang.lower() for lang in (pr.file_languages or []) if lang}
    for lang in target_languages:
        if lang.lower() in available:
            return lang.lower()
    return "unknown"


def _allocate_counts(
    keys: List[StratificationKey],
    sizes: Dict[StratificationKey, int],
    target: int,
    weights: Optional[Dict[StratificationKey, float]] = None,
) -> Dict[StratificationKey, int]:
    """Allocate target counts across strata using weighted shares with capacity caps."""
    if target <= 0:
        return {k: 0 for k in keys}
    total = sum(sizes.get(k, 0) for k in keys)
    if total <= target:
        return {k: sizes.get(k, 0) for k in keys}

    allocations: Dict[StratificationKey, int] = {}
    remainders: List[Tuple[float, StratificationKey]] = []
    allocated = 0

    effective_weights: Dict[StratificationKey, float] = {}
    for k in keys:
        size = sizes.get(k, 0)
        if size <= 0:
            effective_weights[k] = 0.0
            continue
        if weights is None:
            effective_weights[k] = float(size)
        else:
            effective_weights[k] = max(0.0, float(weights.get(k, 0.0)))
    weight_sum = sum(effective_weights.values())
    if weight_sum <= 0:
        # Fall back to proportional-by-size allocation when provided weights are empty/invalid.
        for k in keys:
            size = sizes.get(k, 0)
            effective_weights[k] = float(size if size > 0 else 0)
        weight_sum = sum(effective_weights.values())
    if weight_sum <= 0:
        return {k: 0 for k in keys}

    for k in keys:
        size = sizes.get(k, 0)
        if size == 0:
            allocations[k] = 0
            continue
        exact = (effective_weights[k] / weight_sum) * target
        base = int(exact)
        base = min(base, size)
        allocations[k] = base
        allocated += base
        remainders.append((exact - base, k))

    remaining = target - allocated
    if remaining > 0:
        remainders.sort(key=lambda x: x[0], reverse=True)
        while remaining > 0:
            progressed = False
            for _, k in remainders:
                if remaining <= 0:
                    break
                if allocations[k] < sizes.get(k, 0):
                    allocations[k] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                # All strata reached capacity.
                break

    return allocations


def _build_blended_strata_weights(
    keys: List[StratificationKey],
    sizes: Dict[StratificationKey, int],
    *,
    alpha_time: float,
    alpha_popularity: float,
) -> Dict[StratificationKey, float]:
    """
    Build soft-balancing weights per stratum by blending observed and uniform
    marginals for time and popularity buckets.
    """
    if not keys:
        return {}
    total = sum(max(0, sizes.get(k, 0)) for k in keys)
    if total <= 0:
        return {}

    time_totals: Dict[str, int] = {}
    pop_totals: Dict[str, int] = {}
    for k in keys:
        size = max(0, sizes.get(k, 0))
        if size <= 0:
            continue
        time_totals[k.time_bucket] = time_totals.get(k.time_bucket, 0) + size
        pop_totals[k.popularity_bucket] = pop_totals.get(k.popularity_bucket, 0) + size

    time_count = max(1, len(time_totals))
    pop_count = max(1, len(pop_totals))
    uniform_time = 1.0 / time_count
    uniform_pop = 1.0 / pop_count

    weights: Dict[StratificationKey, float] = {}
    for k in keys:
        size = max(0, sizes.get(k, 0))
        if size <= 0:
            weights[k] = 0.0
            continue
        obs_time = float(time_totals.get(k.time_bucket, 0)) / float(total)
        obs_pop = float(pop_totals.get(k.popularity_bucket, 0)) / float(total)
        blended_time = (1.0 - alpha_time) * obs_time + alpha_time * uniform_time
        blended_pop = (1.0 - alpha_popularity) * obs_pop + alpha_popularity * uniform_pop
        weights[k] = max(0.0, blended_time * blended_pop)
    return weights


def stratified_sample(
    prs: List[PullRequest],
    target: int,
    seed: int | None = None,
    target_languages: List[str] | None = None,
    time_granularity: str = "month",
    popularity_buckets: int = 4,
    soft_balance_alpha_time: float = 0.0,
    soft_balance_alpha_popularity: float = 0.0,
) -> List[PullRequest]:
    """
    Sample PRs using stratification on time and repo popularity.

    If target >= available, returns all PRs.
    """
    if target <= 0:
        return []
    if len(prs) <= target:
        print(f"[sampler] Target {target} exceeds availability {len(prs)}; returning all.")
        return list(prs)

    target_languages = target_languages or []
    language_labels = [lang.lower() for lang in target_languages if lang]
    if not language_labels:
        language_labels = ["unknown"]
    alpha_time = min(1.0, max(0.0, float(soft_balance_alpha_time)))
    alpha_popularity = min(1.0, max(0.0, float(soft_balance_alpha_popularity)))

    popularity_bucket_by_pr = assign_tie_aware_popularity_buckets(
        prs,
        popularity_buckets,
        "pop",
    )

    rng = random.Random(seed)
    strata: Dict[StratificationKey, List[PullRequest]] = {}
    for pr in prs:
        language = _select_language(pr, language_labels)
        time_bucket = _time_bucket(pr, time_granularity)
        popularity_bucket = popularity_bucket_by_pr.get(id(pr), "pop0")
        key = StratificationKey(
            language=language, time_bucket=time_bucket, popularity_bucket=popularity_bucket
        )
        strata.setdefault(key, []).append(pr)

    total = len(prs)
    sampled: List[PullRequest] = []

    print(f"[sampler] Sampling {target} of {total} using {len(strata)} strata.")
    per_lang_target = target // len(language_labels)
    remainder = target % len(language_labels)
    desired_by_lang: Dict[str, int] = {}
    for idx, lang in enumerate(language_labels):
        desired_by_lang[lang] = per_lang_target + (1 if idx < remainder else 0)
    sampled_by_lang: Dict[str, int] = {lang: 0 for lang in language_labels}

    for idx, lang in enumerate(language_labels):
        lang_target = desired_by_lang[lang]
        lang_keys = [key for key in strata if key.language == lang]
        sizes = {key: len(strata[key]) for key in lang_keys}
        lang_total = sum(sizes.values())
        if lang_total == 0:
            continue
        weights = None
        if alpha_time > 0.0 or alpha_popularity > 0.0:
            weights = _build_blended_strata_weights(
                lang_keys,
                sizes,
                alpha_time=alpha_time,
                alpha_popularity=alpha_popularity,
            )
        allocation = _allocate_counts(lang_keys, sizes, lang_target, weights=weights)
        for key in lang_keys:
            items = list(strata[key])
            rng.shuffle(items)
            take = allocation.get(key, 0)
            selected = items[:take]
            sampled.extend(selected)
            sampled_by_lang[lang] = sampled_by_lang.get(lang, 0) + len(selected)

    # Adjust if we overshot or undershot.
    if len(sampled) > target:
        sampled = sampled[:target]
    elif len(sampled) < target:
        remaining = [pr for pr in prs if pr not in sampled]
        needed = target - len(sampled)
        if remaining:
            # First pass: strict language-balance top-up to fill language deficits.
            for lang in language_labels:
                if needed <= 0:
                    break
                deficit = max(0, desired_by_lang.get(lang, 0) - sampled_by_lang.get(lang, 0))
                if deficit <= 0:
                    continue
                lang_remaining = [pr for pr in remaining if _select_language(pr, language_labels) == lang]
                if not lang_remaining:
                    continue
                rng.shuffle(lang_remaining)
                take = min(deficit, needed, len(lang_remaining))
                selected = lang_remaining[:take]
                sampled.extend(selected)
                sampled_by_lang[lang] = sampled_by_lang.get(lang, 0) + len(selected)
                needed -= len(selected)
                remaining = [pr for pr in remaining if pr not in selected]

            # Second pass: violate language balance if needed to still hit target size.
            if needed > 0:
                # Stratified top-up by time/popularity to avoid additional skew.
                remaining_strata: Dict[Tuple[str, str], List[PullRequest]] = {}
                for pr in remaining:
                    key = (
                        _time_bucket(pr, time_granularity),
                        popularity_bucket_by_pr.get(id(pr), "pop0"),
                    )
                    remaining_strata.setdefault(key, []).append(pr)
                remaining_keys = list(remaining_strata.keys())
                sizes = {k: len(remaining_strata[k]) for k in remaining_keys}
                allocation = _allocate_counts(
                    [
                        StratificationKey(language="any", time_bucket=k[0], popularity_bucket=k[1])
                        for k in remaining_keys
                    ],
                    {
                        StratificationKey(
                            language="any", time_bucket=k[0], popularity_bucket=k[1]
                        ): sizes[k]
                        for k in remaining_keys
                    },
                    needed,
                )
                filled = 0
                for k in remaining_keys:
                    items = list(remaining_strata[k])
                    rng.shuffle(items)
                    key_obj = StratificationKey(
                        language="any", time_bucket=k[0], popularity_bucket=k[1]
                    )
                    take = allocation.get(key_obj, 0)
                    selected = items[:take]
                    sampled.extend(selected)
                    filled += len(selected)
                if filled < needed:
                    print(
                        "[sampler] Could not fully hit target due to global candidate exhaustion "
                        f"(requested_extra={needed}, filled={filled})."
                    )

    print(f"[sampler] Sampled {len(sampled)} PRs.")
    return sampled
