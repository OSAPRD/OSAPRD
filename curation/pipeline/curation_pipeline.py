"""Standard curation pipeline orchestration.

This module owns the single supported curation workflow:

1. Read local extraction parquet rows.
2. Filter/sample PRs and select the longitudinal subset.
3. Hydrate before/after/future snapshots for selected PRs.
4. Run original PR refactoring mining plus Multimetric/custom duplication metrics.
5. Persist local aggregate outputs and run metadata.

The CLI in :mod:`curation.run` resolves a ``CurationSettings`` object and then
imports this module after applying those settings to the environment. The
existing lower-level pipeline modules still read config constants at import
time, so the delayed import keeps CLI, env, and module defaults aligned.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import sys
import hashlib
import tempfile
from dataclasses import asdict
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curation.config.settings import CurationSettings
from curation.config.run_config import (
    COHORT,
    CURATION_PARTITION_WRITE_BUFFER_ROWS,
    CURATION_PASS2_URL_CHUNK_SIZE,
    CURATION_RESUME_FROM_EXISTING_SAMPLE,
    CURATION_SKIP_INITIAL_SAMPLE_PROCESSING,
    DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING,
    LONGITUDINAL_TARGET_NO_PRS,
    ONLY_MERGED_PRS,
    POPULARITY_BUCKETS,
    STREAMING_PARQUET_BATCH_SIZE,
    USE_TWO_PASS_STREAMING_SAMPLING,
    RESUME_PROCESSING,
    TARGET_LANGUAGES,
    TARGET_NO_PRS,
    TIME_BUCKET_GRANULARITY,
)
from curation.config.storage_config import CURATION_LOCAL_DATA_FORMAT
from curation.config.storage_config import (
    LOCAL_DIRECTORIES,
    LOCAL_OUTPUT_DIR,
    SAMPLE_HISTORY_DIR,
)
from curation.pipeline.hydration_pipeline import (
    process_prs,
)
from curation.pipeline.preprocessing_pipeline import (
    SamplingCandidatePR,
    materialize_candidates_to_full_prs,
    preprocess_candidates_streaming,
    preprocess_prs,
)
from curation.pipeline.sampler_pipeline import sample_prs
from curation.sampler.sampler import (
    POPULARITY_BUCKET_POLICY,
    _time_bucket,
    assign_tie_aware_popularity_bucket,
    stratified_sample_rows_from_existing_buckets,
    tie_aware_quantile_cut_points_from_counts,
)
from curation.utility.loader import (
    coerce_pull_request_record,
    iter_pr_rows,
    preflight_local_group_input,
)
from curation.utility.sample_history import load_sample_history_pr_identifiers

TOPUP_MAX_ATTEMPTS = 3


def _pr_url(pr: Any) -> str | None:
    """Return the PR URL from a dict row or DTO-like object."""
    if isinstance(pr, dict):
        value = pr.get("url")
    else:
        value = getattr(pr, "url", None)
    return str(value) if value else None


def _pr_changed_lines(pr: Any) -> int:
    """Return total changed lines for lightweight candidate ranking."""
    files = getattr(pr, "files", None) or []
    total = 0
    for fc in files:
        try:
            if isinstance(fc, dict):
                total += int(fc.get("additions", 0) or 0) + int(fc.get("deletions", 0) or 0)
            else:
                total += int(getattr(fc, "additions", 0) or 0) + int(
                    getattr(fc, "deletions", 0) or 0
                )
        except Exception:
            continue
    if total == 0:
        try:
            total = int(getattr(pr, "additions", 0) or 0) + int(getattr(pr, "deletions", 0) or 0)
        except Exception:
            total = 0
    return total


def _normalize_langs(pr: Any) -> set[str]:
    """Return normalized file language labels from a PR object."""
    langs = getattr(pr, "file_languages", None) or []
    return {str(item).strip().lower() for item in langs if item}


def _materialize_for_processing(prs: list[Any], cohort: str) -> list[Any]:
    """Materialize lightweight streaming candidates into full PullRequest DTOs."""
    if not USE_TWO_PASS_STREAMING_SAMPLING:
        return prs
    urls = [_pr_url(pr) for pr in prs]
    urls = [url for url in urls if url]
    if not urls:
        return []
    materialized = materialize_candidates_to_full_prs(
        [pr for pr in prs if _pr_url(pr)],
        cohort,
        batch_size=max(1, int(STREAMING_PARQUET_BATCH_SIZE)),
    )
    sampled_lang_by_url: dict[str, str] = {}
    for candidate in prs:
        url = _pr_url(candidate)
        if not url:
            continue
        lang = (
            getattr(candidate, "pr_primary_language_effective", None)
            if not isinstance(candidate, dict)
            else candidate.get("pr_primary_language_effective")
        )
        if lang and str(lang).strip():
            sampled_lang_by_url[url] = str(lang).strip().lower()

    if sampled_lang_by_url:
        overrides_applied = 0
        for pr in materialized:
            url = _pr_url(pr)
            if not url:
                continue
            sampled_lang = sampled_lang_by_url.get(url)
            if not sampled_lang:
                continue
            if isinstance(pr, dict):
                pr["pr_primary_language_effective"] = sampled_lang
            else:
                setattr(pr, "pr_primary_language_effective", sampled_lang)
            overrides_applied += 1
        if overrides_applied:
            print(
                "[Curation] Applied sampled language override for "
                f"{overrides_applied}/{len(materialized)} rematerialized PRs."
            )
    return materialized


def _candidate_store_path(output_root: Path, cohort: str) -> Path:
    """Return the JSONL store used for lightweight sampling candidates."""
    return output_root / f"candidates_{_safe_cohort_component(cohort)}.jsonl"


def _topup_index_path(output_root: Path, cohort: str) -> Path:
    """Return the sidecar index used when selecting replacement PRs."""
    return output_root / f"candidate_topup_index_{_safe_cohort_component(cohort)}.jsonl"


def _candidate_to_row(pr: Any) -> Dict[str, Any]:
    """Serialize a candidate PR into the lightweight sampling JSONL schema."""
    if isinstance(pr, SamplingCandidatePR):
        return asdict(pr)
    if isinstance(pr, dict):
        return dict(pr)
    return {
        "id": str(getattr(pr, "id", "") or ""),
        "url": str(getattr(pr, "url", "") or ""),
        "number": int(getattr(pr, "number", 0) or 0),
        "created_at": str(getattr(pr, "created_at", "") or ""),
        "merged_at": getattr(pr, "merged_at", None),
        "primary_language": getattr(pr, "primary_language", None),
        "pr_primary_language_effective": getattr(pr, "pr_primary_language_effective", None),
        "file_languages": list(getattr(pr, "file_languages", []) or []),
        "additions": int(getattr(pr, "additions", 0) or 0),
        "deletions": int(getattr(pr, "deletions", 0) or 0),
        "base_repository_full": getattr(pr, "base_repository_full", None),
        "base_repository": getattr(pr, "base_repository", None),
        "files": None,
    }


def _write_candidates_jsonl(path: Path, prs: list[Any]) -> None:
    """Persist lightweight sampling candidates to a deterministic JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pr in prs:
            row = _candidate_to_row(pr)
            if not row.get("url"):
                continue
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _iter_candidates_jsonl(path: Path) -> Iterator[SamplingCandidatePR]:
    """Yield sampling candidates from a previously written JSONL store."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or not row.get("url"):
                continue
            try:
                yield SamplingCandidatePR(
                    id=str(row.get("id") or ""),
                    url=str(row.get("url") or ""),
                    number=int(row.get("number") or 0),
                    created_at=str(row.get("created_at") or ""),
                    merged_at=(str(row.get("merged_at")) if row.get("merged_at") else None),
                    primary_language=row.get("primary_language"),
                    pr_primary_language_effective=row.get("pr_primary_language_effective"),
                    file_languages=[
                        str(v) for v in (row.get("file_languages") or []) if str(v).strip()
                    ],
                    additions=int(row.get("additions") or 0),
                    deletions=int(row.get("deletions") or 0),
                    base_repository_full=(
                        dict(row.get("base_repository_full"))
                        if isinstance(row.get("base_repository_full"), dict)
                        else None
                    ),
                    base_repository=(
                        dict(row.get("base_repository"))
                        if isinstance(row.get("base_repository"), dict)
                        else None
                    ),
                    files=None,
                )
            except Exception:
                continue


def _repo_star_count(pr: Any) -> int:
    repo = getattr(pr, "base_repository_full", None)
    try:
        if isinstance(repo, dict):
            return max(0, int(repo.get("stargazer_count") or 0))
        return max(0, int(getattr(repo, "stargazer_count", 0) or 0))
    except Exception:
        return 0


def _popularity_cut_points_from_candidate_store(candidate_store: Path) -> list[int]:
    star_counts: Dict[int, int] = defaultdict(int)
    for pr in _iter_candidates_jsonl(candidate_store):
        star_counts[_repo_star_count(pr)] += 1
    return tie_aware_quantile_cut_points_from_counts(star_counts, POPULARITY_BUCKETS)


def _write_topup_index(candidate_store: Path, index_path: Path) -> int:
    count = 0
    index_path.parent.mkdir(parents=True, exist_ok=True)
    popularity_cut_points = _popularity_cut_points_from_candidate_store(candidate_store)
    with index_path.open("w", encoding="utf-8") as out:
        for pr in _iter_candidates_jsonl(candidate_store):
            url = _pr_url(pr)
            if not url:
                continue
            lang = _primary_target_language(pr) or "unknown"
            time_bucket = _time_bucket(pr, TIME_BUCKET_GRANULARITY)
            stars = _repo_star_count(pr)
            changed = _pr_changed_lines(pr)
            pop_bucket = assign_tie_aware_popularity_bucket(
                stars,
                popularity_cut_points,
                "pop",
            )
            out.write(
                json.dumps(
                    {
                        "url": url,
                        "primary_lang": lang,
                        "time_bucket": time_bucket,
                        "popularity_value": stars,
                        "popularity_bucket": pop_bucket,
                        "popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
                        "popularity_cut_points": list(popularity_cut_points),
                        "sampling_popularity_value": stars,
                        "sampling_popularity_bucket": pop_bucket,
                        "sampling_popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
                        "sampling_popularity_cut_points": list(popularity_cut_points),
                        "changed_lines": int(changed or 0),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def _sampling_partitions_root(output_root: Path, cohort: str) -> Path:
    return output_root / "sampling_partitions" / _safe_cohort_component(cohort)


def _partition_candidates_for_sampling(
    candidate_store: Path,
    partitions_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Partition candidate JSONL by (language, time_bucket, popularity_bucket) on disk.
    Returns partition metadata keyed by partition id.
    """
    partitions_root.mkdir(parents=True, exist_ok=True)
    flush_threshold = max(1, int(CURATION_PARTITION_WRITE_BUFFER_ROWS))
    print("[sampler_pipeline] Partition writer settings: " f"buffer_rows={flush_threshold}")
    handles: Dict[str, Any] = {}
    buffers: Dict[str, list[str]] = defaultdict(list)

    def _flush_buffer(pid: str) -> None:
        buf = buffers.get(pid) or []
        if not buf:
            return
        handle = handles.get(pid)
        if handle is None:
            path = Path(str(partition_meta[pid]["path"]))
            handle = path.open("a", encoding="utf-8")
            handles[pid] = handle
        handle.write("".join(buf))
        buffers[pid].clear()

    def _get_handle(pid: str):
        handle = handles.get(pid)
        if handle is not None:
            return handle
        path = Path(str(partition_meta[pid]["path"]))
        handle = path.open("a", encoding="utf-8")
        handles[pid] = handle
        return handle

    partition_meta: Dict[str, Dict[str, Any]] = {}
    popularity_cut_points = _popularity_cut_points_from_candidate_store(candidate_store)
    try:
        for pr in _iter_candidates_jsonl(candidate_store):
            url = _pr_url(pr)
            if not url:
                continue
            lang = _primary_target_language(pr) or "unknown"
            time_bucket = _time_bucket(pr, TIME_BUCKET_GRANULARITY)
            stars = _repo_star_count(pr)
            repo = getattr(pr, "base_repository_full", None)
            pop_bucket = assign_tie_aware_popularity_bucket(
                stars,
                popularity_cut_points,
                "pop",
            )
            changed = _pr_changed_lines(pr)
            key = f"{lang}|{time_bucket}|{pop_bucket}"
            pid = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            path = partitions_root / f"part_{pid}.jsonl"
            if pid not in partition_meta:
                partition_meta[pid] = {
                    "partition_id": pid,
                    "path": str(path),
                    "language": lang,
                    "time_bucket": time_bucket,
                    "popularity_bucket": pop_bucket,
                    "popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
                    "popularity_cut_points": list(popularity_cut_points),
                    "size": 0,
                }
            row = json.dumps(
                {
                    "id": str(getattr(pr, "id", "") or ""),
                    "url": url,
                    "number": int(getattr(pr, "number", 0) or 0),
                    "created_at": str(getattr(pr, "created_at", "") or ""),
                    "merged_at": getattr(pr, "merged_at", None),
                    "primary_language": getattr(pr, "primary_language", None),
                    "pr_primary_language_effective": lang,
                    "file_languages": list(getattr(pr, "file_languages", []) or []),
                    "additions": int(getattr(pr, "additions", 0) or 0),
                    "deletions": int(getattr(pr, "deletions", 0) or 0),
                    "base_repository_full": (dict(repo) if isinstance(repo, dict) else None),
                    "base_repository": (
                        dict(getattr(pr, "base_repository", None))
                        if isinstance(getattr(pr, "base_repository", None), dict)
                        else None
                    ),
                    "files": None,
                    "sampling_language_bucket": lang,
                    "sampling_time_bucket": time_bucket,
                    "sampling_popularity_value": stars,
                    "sampling_popularity_bucket": pop_bucket,
                    "sampling_popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
                    "sampling_popularity_cut_points": list(popularity_cut_points),
                    "sampling_changed_lines": int(changed or 0),
                },
                ensure_ascii=False,
                default=str,
            )
            _get_handle(pid)
            buffers[pid].append(row + "\n")
            if len(buffers[pid]) >= flush_threshold:
                _flush_buffer(pid)
            partition_meta[pid]["size"] = int(partition_meta[pid]["size"]) + 1
    finally:
        for pid in list(handles.keys()):
            try:
                _flush_buffer(pid)
            except Exception:
                pass
        for handle in list(handles.values()):
            try:
                handle.close()
            except Exception:
                pass
    return partition_meta


def _allocate_counts_proportional(
    sizes: Dict[str, int],
    target: int,
) -> Dict[str, int]:
    if target <= 0:
        return {k: 0 for k in sizes}
    total = sum(max(0, int(v)) for v in sizes.values())
    if total <= target:
        return {k: max(0, int(v)) for k, v in sizes.items()}
    alloc: Dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for k, size in sizes.items():
        s = max(0, int(size))
        if s <= 0:
            alloc[k] = 0
            continue
        exact = (s / total) * target
        base = min(s, int(exact))
        alloc[k] = base
        allocated += base
        remainders.append((exact - base, k))
    remaining = target - allocated
    if remaining > 0:
        remainders.sort(reverse=True)
        idx = 0
        while remaining > 0 and remainders:
            _, k = remainders[idx % len(remainders)]
            if alloc[k] < max(0, int(sizes.get(k, 0))):
                alloc[k] += 1
                remaining -= 1
            idx += 1
            if (
                idx > len(remainders) * 4
                and remaining > 0
                and all(alloc[name] >= max(0, int(sizes.get(name, 0))) for _, name in remainders)
            ):
                break
    return alloc


def _reservoir_sample_rows(path: Path, k: int, *, seed: int) -> list[Dict[str, Any]]:
    if k <= 0 or not path.exists():
        return []
    rng = random.Random(seed)
    sample: list[Dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            seen += 1
            if len(sample) < k:
                sample.append(row)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    sample[j - 1] = row
    return sample


def _row_to_candidate(row: Dict[str, Any]) -> SamplingCandidatePR | None:
    try:
        return SamplingCandidatePR(
            id=str(row.get("id") or ""),
            url=str(row.get("url") or ""),
            number=int(row.get("number") or 0),
            created_at=str(row.get("created_at") or ""),
            merged_at=(str(row.get("merged_at")) if row.get("merged_at") else None),
            primary_language=row.get("primary_language"),
            pr_primary_language_effective=row.get("pr_primary_language_effective"),
            file_languages=[str(v) for v in (row.get("file_languages") or []) if str(v).strip()],
            additions=int(row.get("additions") or 0),
            deletions=int(row.get("deletions") or 0),
            base_repository_full=(
                dict(row.get("base_repository_full"))
                if isinstance(row.get("base_repository_full"), dict)
                else None
            ),
            base_repository=(
                dict(row.get("base_repository"))
                if isinstance(row.get("base_repository"), dict)
                else None
            ),
            files=None,
        )
    except Exception:
        return None


def _write_sample_metadata_from_rows(
    rows: list[Dict[str, Any]],
    cohort: str,
    *,
    basename: str,
) -> None:
    output_dir = Path(LOCAL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_cohort_component(cohort).lower()
    txt_path = output_dir / f"{basename}_{safe}.txt"
    jsonl_path = output_dir / f"{basename}_{safe}.jsonl"
    with (
        txt_path.open("w", encoding="utf-8") as f_txt,
        jsonl_path.open("w", encoding="utf-8") as f_jsonl,
    ):
        for row in rows:
            pr_id = str(row.get("id") or "")
            pr_url = str(row.get("url") or "")
            if pr_url:
                f_txt.write(f"{pr_id}\t{pr_url}\n")
            f_jsonl.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"[sampler_pipeline] Sample list written to: {txt_path}")
    print(f"[sampler_pipeline] Sample structured metadata written to: {jsonl_path}")


def _sample_from_partitioned_candidates(
    candidate_store: Path,
    *,
    cohort: str,
    target: int,
    basename: str,
    seed: int = 42,
) -> tuple[Path, Dict[str, Any]]:
    partitions_root = _sampling_partitions_root(Path(LOCAL_OUTPUT_DIR) / "output", cohort)
    partition_meta = _partition_candidates_for_sampling(candidate_store, partitions_root)
    partition_items = list(partition_meta.values())
    popularity_cut_points = (
        list(partition_items[0].get("popularity_cut_points") or []) if partition_items else []
    )
    total_candidates = sum(int(item.get("size", 0)) for item in partition_items)
    print(f"[sampler_pipeline] Candidates: {total_candidates}")
    output_dir = Path(LOCAL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_store_path = (
        output_dir / f"{basename}_{_safe_cohort_component(cohort).lower()}_store.jsonl"
    )
    if total_candidates <= 0 or target <= 0:
        _write_sample_metadata_from_rows([], cohort, basename=basename)
        sampled_store_path.write_text("", encoding="utf-8")
        return sampled_store_path, {
            "candidates": total_candidates,
            "sampled": 0,
            "time_counts": {},
            "language_counts": {},
            "popularity_counts": {},
            "popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
            "popularity_cut_points": list(popularity_cut_points),
        }
    if target >= total_candidates:
        target = total_candidates

    # Language-first allocation to preserve quotas.
    quotas = _target_language_quotas(target)
    by_lang: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for item in partition_items:
        by_lang[str(item.get("language") or "unknown")].append(item)

    target_alloc: Dict[str, int] = {pid: 0 for pid in partition_meta}
    unfilled = 0
    for lang, desired in quotas.items():
        parts = by_lang.get(lang, [])
        sizes = {str(p["partition_id"]): int(p.get("size", 0)) for p in parts}
        alloc = _allocate_counts_proportional(sizes, desired)
        filled = 0
        for pid, cnt in alloc.items():
            target_alloc[pid] += cnt
            filled += cnt
        if filled < desired:
            unfilled += desired - filled

    if unfilled > 0:
        remaining_sizes = {
            str(item["partition_id"]): max(
                0, int(item.get("size", 0)) - int(target_alloc.get(str(item["partition_id"]), 0))
            )
            for item in partition_items
        }
        extra_alloc = _allocate_counts_proportional(remaining_sizes, unfilled)
        for pid, cnt in extra_alloc.items():
            target_alloc[pid] += cnt

    sampled_rows: list[Dict[str, Any]] = []
    for item in partition_items:
        pid = str(item["partition_id"])
        take = int(target_alloc.get(pid, 0))
        if take <= 0:
            continue
        path = Path(str(item["path"]))
        part_seed = int(hashlib.sha1(f"{seed}:{pid}".encode("utf-8")).hexdigest()[:8], 16)
        sampled_rows.extend(_reservoir_sample_rows(path, take, seed=part_seed))

    if len(sampled_rows) > target:
        rng = random.Random(seed)
        rng.shuffle(sampled_rows)
        sampled_rows = sampled_rows[:target]

    # Stats
    time_counts: Dict[str, int] = defaultdict(int)
    lang_counts: Dict[str, int] = defaultdict(int)
    pop_counts: Dict[str, int] = defaultdict(int)
    candidates: list[SamplingCandidatePR] = []
    for row in sampled_rows:
        t = str(row.get("sampling_time_bucket") or "unknown")
        l = str(row.get("sampling_language_bucket") or "unknown")
        p = str(row.get("sampling_popularity_bucket") or "unknown")
        time_counts[t] += 1
        lang_counts[l] += 1
        pop_counts[p] += 1
        pr = _row_to_candidate(row)
        if pr is not None:
            candidates.append(pr)

    time_line = ", ".join(f"{k}={time_counts[k]}" for k in sorted(time_counts))
    lang_line = ", ".join(f"{k}={lang_counts[k]}" for k in sorted(lang_counts))
    pop_line = ", ".join(f"{k}={pop_counts[k]}" for k in sorted(pop_counts))
    print(f"[sampler_pipeline] Sample distribution by time bucket: {time_line}")
    print(f"[sampler_pipeline] Sample distribution by language bucket: {lang_line}")
    print(f"[sampler_pipeline] Sample distribution by popularity bucket: {pop_line}")

    _write_sample_metadata_from_rows(sampled_rows, cohort, basename=basename)
    with sampled_store_path.open("w", encoding="utf-8") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return sampled_store_path, {
        "candidates": total_candidates,
        "sampled": len(candidates),
        "time_counts": dict(sorted(time_counts.items())),
        "language_counts": dict(sorted(lang_counts.items())),
        "popularity_counts": dict(sorted(pop_counts.items())),
        "popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
        "popularity_cut_points": list(popularity_cut_points),
    }


def _iter_candidates_from_rows_store(
    path: Path, *, chunk_size: int = 256
) -> Iterator[list[SamplingCandidatePR]]:
    if chunk_size <= 0:
        chunk_size = 256
    chunk: list[SamplingCandidatePR] = []
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            pr = _row_to_candidate(row)
            if pr is None:
                continue
            chunk.append(pr)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _pass2_language_override(row: Dict[str, Any]) -> str | None:
    """Return the sampled language label to keep pass-2 DTOs aligned with sampling."""
    for key in ("pr_primary_language_effective", "sampling_language_bucket"):
        value = row.get(key)
        if value and str(value).strip():
            return str(value).strip().lower()
    return None


def _populate_pass2_url_index(conn: sqlite3.Connection, sampled_store: Path) -> tuple[int, int]:
    """Populate a disk-backed URL membership index for one-scan pass-2 materialization."""
    conn.execute(
        """
        CREATE TABLE selected_urls (
            url TEXT PRIMARY KEY,
            language TEXT,
            seen INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    sampled_rows = 0
    insert_buffer: list[tuple[str, str | None]] = []

    def _flush() -> None:
        if not insert_buffer:
            return
        conn.executemany(
            "INSERT OR IGNORE INTO selected_urls(url, language) VALUES (?, ?)",
            insert_buffer,
        )
        insert_buffer.clear()

    if not sampled_store.exists():
        return 0, 0

    with sampled_store.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            sampled_rows += 1
            insert_buffer.append((url, _pass2_language_override(row)))
            if len(insert_buffer) >= 1000:
                _flush()
    _flush()
    conn.commit()
    unique_urls = int(conn.execute("SELECT COUNT(*) FROM selected_urls").fetchone()[0])
    duplicates = max(0, sampled_rows - unique_urls)
    return unique_urls, duplicates


def _pass2_pending_languages_for_urls(
    conn: sqlite3.Connection,
    urls: list[str],
) -> dict[str, str | None]:
    """Return pending selected URLs present in a bounded source-row batch."""
    if not urls:
        return {}
    pending: dict[str, str | None] = {}
    unique_urls = list(dict.fromkeys(urls))
    max_sqlite_params = 900
    for start in range(0, len(unique_urls), max_sqlite_params):
        chunk = unique_urls[start : start + max_sqlite_params]
        placeholders = ",".join("?" for _ in chunk)
        query = (
            "SELECT url, language FROM selected_urls " f"WHERE seen = 0 AND url IN ({placeholders})"
        )
        for url, language in conn.execute(query, chunk):
            pending[str(url)] = language
    return pending


def _stratified_sample_rows_from_existing_buckets(
    rows: list[Dict[str, Any]],
    *,
    target: int,
    seed: int = 42,
) -> list[Dict[str, Any]]:
    return stratified_sample_rows_from_existing_buckets(
        rows,
        target=target,
        target_languages=TARGET_LANGUAGES,
        seed=seed,
    )


def _iter_materialized_pr_batches_single_scan(
    sampled_store: Path,
    cohort: str,
    *,
    output_root: Path,
    parquet_batch_size: int,
    output_batch_size: int,
) -> Iterator[list[Any]]:
    """
    Materialize sampled PRs with one parquet scan and bounded heap usage.

    The sampled URL membership index lives in a temporary SQLite database instead of
    a Python set, so memory stays bounded by the parquet row batch plus the output
    processing batch.
    """
    output_batch_size = max(1, int(output_batch_size))
    parquet_batch_size = max(1, int(parquet_batch_size))
    if not sampled_store.exists():
        print(f"[Curation] Pass-2 skipped: sampled store not found at {sampled_store}")
        return

    temp_parent = output_root / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"pass2_url_index_{_safe_cohort_component(cohort)}_",
        dir=str(temp_parent),
    ) as temp_dir:
        db_path = Path(temp_dir) / "selected_urls.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            # The index is disposable and can be rebuilt on resume; keep SQLite's
            # cache small so this stage does not trade scan count for heap growth.
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute("PRAGMA cache_size=-4096")

            selected_count, duplicate_count = _populate_pass2_url_index(conn, sampled_store)
            print(
                "[Curation] Pass-2 URL index built: "
                f"selected_urls={selected_count}, duplicate_rows={duplicate_count}, "
                f"index={db_path}"
            )
            if selected_count <= 0:
                return

            materialized = 0
            parse_failures = 0
            batch: list[Any] = []
            source_rows: list[Dict[str, Any]] = []

            def _materialize_source_rows(rows: list[Dict[str, Any]]) -> Iterator[list[Any]]:
                nonlocal materialized, parse_failures, batch
                urls_by_row = [
                    (str(row.get("url") or "").strip(), row)
                    for row in rows
                    if str(row.get("url") or "").strip()
                ]
                pending_languages = _pass2_pending_languages_for_urls(
                    conn,
                    [url for url, _ in urls_by_row],
                )
                consumed_urls: set[str] = set()
                for url, row in urls_by_row:
                    if url in consumed_urls or url not in pending_languages:
                        continue
                    try:
                        pr = coerce_pull_request_record(row)
                    except Exception:
                        parse_failures += 1
                        continue
                    language = pending_languages.get(url)
                    if language:
                        setattr(pr, "pr_primary_language_effective", str(language))
                    conn.execute("UPDATE selected_urls SET seen = 1 WHERE url = ?", (url,))
                    consumed_urls.add(url)
                    materialized += 1
                    batch.append(pr)
                    if len(batch) >= output_batch_size:
                        conn.commit()
                        yield batch
                        batch = []

            for row in iter_pr_rows(
                group=cohort,
                batch_size=parquet_batch_size,
            ):
                if not isinstance(row, dict):
                    continue
                source_rows.append(row)
                if len(source_rows) >= parquet_batch_size:
                    yield from _materialize_source_rows(source_rows)
                    source_rows = []

            if source_rows:
                yield from _materialize_source_rows(source_rows)

            if batch:
                conn.commit()
                yield batch

            conn.commit()
            missing = int(
                conn.execute("SELECT COUNT(*) FROM selected_urls WHERE seen = 0").fetchone()[0]
            )
            print(
                "[Curation] Pass-2 one-scan materialization complete: "
                f"materialized={materialized}/{selected_count}, missing={missing}, "
                f"parse_failures={parse_failures}"
            )
            if missing:
                print(
                    "[preprocess] Warning: pass-2 could not materialize "
                    f"{missing} selected PR(s) after one source scan."
                )
            if parse_failures:
                print(
                    "[preprocess] Warning: pass-2 failed to parse "
                    f"{parse_failures} selected PR row(s)."
                )
        finally:
            conn.close()


def _sample_longitudinal_urls_from_store(
    sampled_store: Path,
    *,
    cohort: str,
    target: int,
    seed: int = 42,
) -> tuple[set[str], Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if target > 0 and sampled_store.exists():
        with sampled_store.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    sampled_rows = _stratified_sample_rows_from_existing_buckets(
        rows,
        target=target,
        seed=seed,
    )
    _write_sample_metadata_from_rows(sampled_rows, cohort, basename="longitudinal_prs")
    urls = {
        str(row.get("url") or "").strip()
        for row in sampled_rows
        if str(row.get("url") or "").strip()
    }
    time_counts: Dict[str, int] = defaultdict(int)
    lang_counts: Dict[str, int] = defaultdict(int)
    pop_counts: Dict[str, int] = defaultdict(int)
    for row in sampled_rows:
        time_counts[str(row.get("sampling_time_bucket") or "unknown")] += 1
        lang_counts[str(row.get("sampling_language_bucket") or "unknown")] += 1
        pop_counts[str(row.get("sampling_popularity_bucket") or "unknown")] += 1
    popularity_policy = POPULARITY_BUCKET_POLICY
    popularity_cut_points: list[Any] = []
    for row in sampled_rows:
        if row.get("sampling_popularity_bucket_policy"):
            popularity_policy = str(row.get("sampling_popularity_bucket_policy"))
        if isinstance(row.get("sampling_popularity_cut_points"), list):
            popularity_cut_points = list(row.get("sampling_popularity_cut_points") or [])
        if popularity_cut_points:
            break
    return urls, {
        "candidates": len(rows),
        "sampled": len(sampled_rows),
        "time_counts": dict(sorted(time_counts.items())),
        "language_counts": dict(sorted(lang_counts.items())),
        "popularity_counts": dict(sorted(pop_counts.items())),
        "popularity_bucket_policy": popularity_policy,
        "popularity_cut_points": popularity_cut_points,
    }


def _sample_store_path(cohort: str, *, basename: str = "sampled_prs") -> Path:
    safe = _safe_cohort_component(cohort).lower()
    return Path(LOCAL_OUTPUT_DIR) / f"{basename}_{safe}_store.jsonl"


def _sample_metadata_jsonl_path(cohort: str, *, basename: str) -> Path:
    safe = _safe_cohort_component(cohort).lower()
    return Path(LOCAL_OUTPUT_DIR) / f"{basename}_{safe}.jsonl"


def _load_sample_rows(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _sample_stats_from_rows(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    time_counts: Dict[str, int] = defaultdict(int)
    lang_counts: Dict[str, int] = defaultdict(int)
    pop_counts: Dict[str, int] = defaultdict(int)
    popularity_policy = POPULARITY_BUCKET_POLICY
    popularity_cut_points: list[Any] = []
    for row in rows:
        time_counts[str(row.get("sampling_time_bucket") or "unknown")] += 1
        lang_counts[str(row.get("sampling_language_bucket") or "unknown")] += 1
        pop_counts[str(row.get("sampling_popularity_bucket") or "unknown")] += 1
        if row.get("sampling_popularity_bucket_policy"):
            popularity_policy = str(row.get("sampling_popularity_bucket_policy"))
        if not popularity_cut_points and isinstance(
            row.get("sampling_popularity_cut_points"), list
        ):
            popularity_cut_points = list(row.get("sampling_popularity_cut_points") or [])
    return {
        "candidates": len(rows),
        "sampled": len(rows),
        "time_counts": dict(sorted(time_counts.items())),
        "language_counts": dict(sorted(lang_counts.items())),
        "popularity_counts": dict(sorted(pop_counts.items())),
        "popularity_bucket_policy": popularity_policy,
        "popularity_cut_points": popularity_cut_points,
    }


def _load_existing_longitudinal_urls(cohort: str) -> tuple[set[str], Dict[str, Any], Path | None]:
    candidates = [
        _sample_store_path(cohort, basename="longitudinal_prs"),
        _sample_metadata_jsonl_path(cohort, basename="longitudinal_prs"),
    ]
    for path in candidates:
        rows = _load_sample_rows(path)
        if not rows:
            continue
        urls = {
            str(row.get("url") or "").strip() for row in rows if str(row.get("url") or "").strip()
        }
        return urls, _sample_stats_from_rows(rows), path
    return set(), _sample_stats_from_rows([]), None


def _load_sampled_meta_from_store(path: Path) -> dict[str, Dict[str, Any]]:
    out: dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            lang = str(
                row.get("sampling_language_bucket")
                or row.get("pr_primary_language_effective")
                or "unknown"
            )
            time_bucket = str(row.get("sampling_time_bucket") or "unknown")
            pop_bucket = str(row.get("sampling_popularity_bucket") or "pop0")
            changed = int(row.get("sampling_changed_lines") or 0)
            candidate = _row_to_candidate(row)
            if candidate is None:
                continue
            out[url] = {
                "pr": candidate,
                "primary_lang": lang,
                "time_bucket": time_bucket,
                "popularity_bucket": pop_bucket,
                "changed_lines": changed,
            }
    return out


def _build_replacement_pool_from_index(
    index_path: Path,
    *,
    selected_urls: set[str],
    attempted_urls: set[str],
    per_bucket_cap: int = 4,
) -> dict[str, Dict[str, Any]]:
    grouped_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    out: dict[str, Dict[str, Any]] = {}
    if not index_path.exists():
        return out
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not url or url in selected_urls or url in attempted_urls:
                continue
            lang = str(row.get("primary_lang") or "unknown")
            time_bucket = str(row.get("time_bucket") or "unknown")
            pop_bucket = str(row.get("popularity_bucket") or "pop0")
            changed = int(row.get("changed_lines") or 0)
            key = (lang, time_bucket, pop_bucket)
            if grouped_counts[key] >= per_bucket_cap:
                continue
            grouped_counts[key] += 1
            candidate = SamplingCandidatePR(
                id="",
                url=url,
                number=0,
                created_at="",
                merged_at=None,
                primary_language=lang,
                pr_primary_language_effective=lang,
                file_languages=[lang] if lang and lang != "unknown" else [],
                additions=0,
                deletions=0,
                base_repository_full=None,
                base_repository=None,
                files=None,
            )
            out[url] = {
                "pr": candidate,
                "primary_lang": lang,
                "time_bucket": time_bucket,
                "popularity_bucket": pop_bucket,
                "changed_lines": changed,
            }
    return out


def _primary_target_language(pr: Any) -> str | None:
    explicit_effective = (
        getattr(pr, "pr_primary_language_effective", None)
        if not isinstance(pr, dict)
        else pr.get("pr_primary_language_effective")
    )
    if explicit_effective:
        normalized_effective = str(explicit_effective).strip().lower()
        for target in [str(v).strip().lower() for v in TARGET_LANGUAGES if v]:
            if normalized_effective == target:
                return target
    explicit_primary = (
        getattr(pr, "primary_language", None)
        if not isinstance(pr, dict)
        else pr.get("primary_language")
    )
    if explicit_primary:
        normalized_primary = str(explicit_primary).strip().lower()
        for target in [str(v).strip().lower() for v in TARGET_LANGUAGES if v]:
            if normalized_primary == target:
                return target
    langs = _normalize_langs(pr)
    for target in [str(v).strip().lower() for v in TARGET_LANGUAGES if v]:
        if target in langs:
            return target
    return next(iter(langs), None)


def _load_aggregates_by_url(output_root: Path, cohort: str) -> dict[str, dict[str, Any]]:
    root = output_root / "processed-data" / cohort / "metrics-json"
    by_url: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return by_url
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["__aggregate_path"] = str(path)
        pr = payload.get("pr")
        if not isinstance(pr, dict):
            continue
        url = str(pr.get("url") or "")
        if url:
            by_url[url] = payload
    return by_url


def _cleanup_failed_pr_outputs(output_root: Path, aggregate: dict[str, Any] | None) -> None:
    if not isinstance(aggregate, dict):
        return

    # Remove per-PR aggregate metrics JSON.
    aggregate_path_raw = aggregate.get("__aggregate_path")
    if aggregate_path_raw:
        try:
            aggregate_path = Path(str(aggregate_path_raw))
            if aggregate_path.exists():
                aggregate_path.unlink()
        except Exception:
            pass

    if not DELETE_SNAPSHOT_ARTIFACTS_AFTER_PROCESSING:
        return

    # Remove hydrated PR snapshot directory if present.
    hydration = aggregate.get("hydration")
    snapshots = hydration.get("snapshots") if isinstance(hydration, dict) else None
    candidate_paths: list[str] = []
    if isinstance(snapshots, dict):
        before = snapshots.get("before")
        after = snapshots.get("after")
        future = snapshots.get("future")
        if isinstance(before, dict):
            candidate_paths.append(str(before.get("path") or ""))
        if isinstance(after, dict):
            candidate_paths.append(str(after.get("path") or ""))
        if isinstance(future, dict):
            for value in future.values():
                if isinstance(value, dict):
                    candidate_paths.append(str(value.get("path") or ""))

    for raw in candidate_paths:
        if not raw:
            continue
        try:
            path = Path(raw)
            pr_root = path.parent if path.name in {"before", "after"} else path
            if path.name == "future":
                pr_root = path.parent
            if pr_root.exists():
                shutil.rmtree(pr_root, ignore_errors=True)
                break
        except Exception:
            continue


def _pr_tool_gate_status(aggregate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Return (ok, reasons) for replacement top-up eligibility.

    Top-up policy:
    - Replace when hard-gate checks fail (currently refactoring failures only).
    - Maintainability failures are reported in aggregate/errors but do not trigger replacement.
    - Missing aggregate/metrics is handled by the caller and still triggers replacement.
    """
    reasons: list[str] = []
    metrics = aggregate.get("metrics")
    if not isinstance(metrics, dict):
        return False, ["missing_metrics"]

    ref_success_statuses = {"success", "partial_success", "completed", "missing_snapshot", "skipped"}
    ref = metrics.get("refactoring")
    if isinstance(ref, dict):
        ref_stage_status = str(ref.get("status") or "").strip().lower()
        if ref_stage_status and ref_stage_status not in ref_success_statuses:
            reasons.append(f"refactoring_stage:{ref_stage_status}")
        for snapshot in ref.get("snapshot_results") or []:
            if not isinstance(snapshot, dict):
                continue
            status = str(snapshot.get("status") or "").strip().lower()
            if status and status not in {"success", "missing_snapshot"}:
                label = str(snapshot.get("snapshot_label") or "unknown")
                tool = str(snapshot.get("tool") or "unknown")
                reasons.append(f"refactoring_snapshot:{label}:{tool}:{status}")
    else:
        reasons.append("missing_refactoring")

    return (len(reasons) == 0), reasons


def _target_language_quotas(total_target: int) -> Dict[str, int]:
    langs = [str(v).strip().lower() for v in TARGET_LANGUAGES if str(v).strip()]
    if not langs or total_target <= 0:
        return {}
    base = total_target // len(langs)
    remainder = total_target % len(langs)
    quotas: Dict[str, int] = {}
    for idx, lang in enumerate(langs):
        quotas[lang] = base + (1 if idx < remainder else 0)
    return quotas


def _current_selected_language_counts(
    selected_urls: set[str],
    candidate_meta: dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for url in selected_urls:
        meta = candidate_meta.get(url)
        if not meta:
            continue
        lang = str(meta.get("primary_lang") or "")
        if lang:
            counts[lang] += 1
    return dict(counts)


def _pick_quota_topups(
    *,
    selected_urls: set[str],
    attempted_urls: set[str],
    candidate_meta: dict[str, Dict[str, Any]],
    language_quotas: Dict[str, int],
    target_count: int,
) -> list[Any]:
    if target_count <= 0 or not language_quotas:
        return []

    selected_counts = _current_selected_language_counts(selected_urls, candidate_meta)
    chosen: list[Any] = []

    # Fill per-language deficits only, in quota order.
    for lang in language_quotas:
        if len(chosen) >= target_count:
            break
        deficit = max(0, int(language_quotas.get(lang, 0)) - int(selected_counts.get(lang, 0)))
        if deficit <= 0:
            continue
        candidates: list[Dict[str, Any]] = []
        for url, meta in candidate_meta.items():
            if url in selected_urls or url in attempted_urls:
                continue
            if str(meta.get("primary_lang") or "") != lang:
                continue
            candidates.append(meta)
        if not candidates:
            continue
        candidates.sort(
            key=lambda m: (
                str(m.get("time_bucket") or ""),
                str(m.get("popularity_bucket") or ""),
                int(m.get("changed_lines") or 0),
                str(_pr_url(m.get("pr")) or ""),
            )
        )
        take = min(deficit, target_count - len(chosen), len(candidates))
        for meta in candidates[:take]:
            pr = meta.get("pr")
            url = _pr_url(pr)
            if not url:
                continue
            chosen.append(pr)
            selected_urls.add(url)
            attempted_urls.add(url)
            selected_counts[lang] = int(selected_counts.get(lang, 0)) + 1

    return chosen


def _pick_replacement(
    *,
    failed_url: str,
    selected_urls: set[str],
    attempted_urls: set[str],
    candidate_meta: dict[str, Dict[str, Any]],
) -> Any | None:
    failed = candidate_meta.get(failed_url)
    if not failed:
        return None
    lang = str(failed.get("primary_lang") or "")
    target_time = str(failed.get("time_bucket") or "")
    target_pop = str(failed.get("popularity_bucket") or "")
    target_changed = int(failed.get("changed_lines") or 0)
    candidates: list[Dict[str, Any]] = []
    for url, meta in candidate_meta.items():
        if url in selected_urls or url in attempted_urls:
            continue
        if str(meta.get("primary_lang") or "") != lang:
            continue
        candidates.append(meta)
    if not candidates:
        return None

    def _score(meta: Dict[str, Any]) -> tuple[int, int, int, str]:
        same_pop = str(meta.get("popularity_bucket") or "") == target_pop
        same_time = str(meta.get("time_bucket") or "") == target_time
        if same_pop and same_time:
            rank = 0
        elif same_pop:
            rank = 1
        elif same_time:
            rank = 2
        else:
            rank = 3
        changed = int(meta.get("changed_lines") or 0)
        return (rank, abs(changed - target_changed), changed, str(_pr_url(meta.get("pr")) or ""))

    candidates.sort(key=_score)
    return candidates[0].get("pr")


def _merge_processing_stats(total: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    timing_buckets = ("with_longitudinal", "without_longitudinal")
    total_seconds = dict(total.get("language_time_total_seconds", {}))
    total_counts = dict(total.get("language_time_counts", {}))
    for lang, bucket_map in (current.get("language_time_total_seconds", {}) or {}).items():
        lang_key = str(lang)
        if not isinstance(total_seconds.get(lang_key), dict):
            total_seconds[lang_key] = {}
        for bucket, seconds in (bucket_map or {}).items():
            bucket_key = str(bucket)
            prior = float(total_seconds[lang_key].get(bucket_key, 0.0))
            total_seconds[lang_key][bucket_key] = prior + float(seconds)
    for lang, bucket_map in (current.get("language_time_counts", {}) or {}).items():
        lang_key = str(lang)
        if not isinstance(total_counts.get(lang_key), dict):
            total_counts[lang_key] = {}
        for bucket, count in (bucket_map or {}).items():
            bucket_key = str(bucket)
            prior = int(total_counts[lang_key].get(bucket_key, 0))
            total_counts[lang_key][bucket_key] = prior + int(count)
    avg_seconds = {}
    for lang in ("C++", "Java", "JavaScript", "Python"):
        avg_seconds[lang] = {}
        for bucket in timing_buckets:
            count = int((total_counts.get(lang) or {}).get(bucket, 0))
            secs = float((total_seconds.get(lang) or {}).get(bucket, 0.0))
            avg_seconds[lang][bucket] = (secs / count) if count else 0.0
    merged: Dict[str, Any] = {
        "processed_prs": int(total.get("processed_prs", 0)) + int(current.get("processed_prs", 0)),
        "longitudinal_selected_prs": int(total.get("longitudinal_selected_prs", 0))
        + int(current.get("longitudinal_selected_prs", 0)),
        "metrics_computed": int(total.get("metrics_computed", 0))
        + int(current.get("metrics_computed", 0)),
        "metrics_failed": int(total.get("metrics_failed", 0))
        + int(current.get("metrics_failed", 0)),
        "with_before": int(total.get("with_before", 0)) + int(current.get("with_before", 0)),
        "with_after": int(total.get("with_after", 0)) + int(current.get("with_after", 0)),
        "with_future_any": int(total.get("with_future_any", 0))
        + int(current.get("with_future_any", 0)),
        "skipped_failed_on_resume": int(total.get("skipped_failed_on_resume", 0))
        + int(current.get("skipped_failed_on_resume", 0)),
        "prefiltered_completed_on_resume": int(total.get("prefiltered_completed_on_resume", 0))
        + int(current.get("prefiltered_completed_on_resume", 0)),
        "prefiltered_persisted_on_resume": int(total.get("prefiltered_persisted_on_resume", 0))
        + int(current.get("prefiltered_persisted_on_resume", 0)),
        "future_counts": dict(total.get("future_counts", {})),
        "language_time_total_seconds": total_seconds,
        "language_time_counts": total_counts,
        "language_time_avg_seconds": avg_seconds,
    }
    for label, count in (current.get("future_counts", {}) or {}).items():
        merged["future_counts"][label] = merged["future_counts"].get(label, 0) + int(count)
    return merged


def _safe_cohort_component(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in (value or "").strip()
    )


def _load_run_errors(output_root: Path, cohort: str) -> Dict[str, Any]:
    safe_cohort = _safe_cohort_component(cohort)
    path = output_root / f"run_errors_{safe_cohort}.json"
    if not path.exists():
        return {"summary": {}, "errors": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"summary": {}, "errors": []}
    if not isinstance(payload, dict):
        return {"summary": {}, "errors": []}
    summary = payload.get("summary")
    errors = payload.get("errors")
    return {
        "summary": summary if isinstance(summary, dict) else {},
        "errors": errors if isinstance(errors, list) else [],
    }


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _final_selected_distributions(
    output_root: Path,
    cohort: str,
) -> Dict[str, Any]:
    aggregates = _load_aggregates_by_url(output_root, cohort)
    language_all: Dict[str, int] = defaultdict(int)
    language_longitudinal: Dict[str, int] = defaultdict(int)
    language_non_longitudinal: Dict[str, int] = defaultdict(int)
    popularity_all: Dict[str, int] = defaultdict(int)
    popularity_longitudinal: Dict[str, int] = defaultdict(int)
    popularity_non_longitudinal: Dict[str, int] = defaultdict(int)
    longitudinal_count = 0
    non_longitudinal_count = 0

    for payload in aggregates.values():
        if not isinstance(payload, dict):
            continue
        pr = payload.get("pr") if isinstance(payload.get("pr"), dict) else {}
        repo_meta = (
            payload.get("repository_metadata")
            if isinstance(payload.get("repository_metadata"), dict)
            else {}
        )
        is_longitudinal = bool(pr.get("longitudinal_selected"))
        lang = str(
            pr.get("pr_primary_language_effective")
            or pr.get("primary_language")
            or repo_meta.get("pr_primary_language_effective")
            or repo_meta.get("primary_language")
            or "unknown"
        )
        pop = str(repo_meta.get("popularity_bucket") or "unknown")
        language_all[lang] += 1
        popularity_all[pop] += 1
        if is_longitudinal:
            longitudinal_count += 1
            language_longitudinal[lang] += 1
            popularity_longitudinal[pop] += 1
        else:
            non_longitudinal_count += 1
            language_non_longitudinal[lang] += 1
            popularity_non_longitudinal[pop] += 1

    return {
        "total_selected": len(aggregates),
        "longitudinal_selected": longitudinal_count,
        "non_longitudinal_selected": non_longitudinal_count,
        "language_all": dict(sorted(language_all.items())),
        "language_longitudinal": dict(sorted(language_longitudinal.items())),
        "language_non_longitudinal": dict(sorted(language_non_longitudinal.items())),
        "popularity_all": dict(sorted(popularity_all.items())),
        "popularity_longitudinal": dict(sorted(popularity_longitudinal.items())),
        "popularity_non_longitudinal": dict(sorted(popularity_non_longitudinal.items())),
    }


def _has_snapshot(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("available") is not False
        and bool(payload.get("commit") or payload.get("path"))
    )


def _recount_processing_stats_from_outputs(
    output_root: Path,
    cohort: str,
    existing_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Recount final PR counters from persisted aggregate outputs after top-up cleanup."""
    aggregates = _load_aggregates_by_url(output_root, cohort)
    processed = 0
    metrics_computed = 0
    with_before = 0
    with_after = 0
    with_future_any = 0
    longitudinal_selected = 0
    future_counts: Dict[str, int] = {}

    for aggregate in aggregates.values():
        if not isinstance(aggregate, dict):
            continue
        processed += 1
        if isinstance(aggregate.get("metrics"), dict):
            metrics_computed += 1
        pr = aggregate.get("pr") if isinstance(aggregate.get("pr"), dict) else {}
        if pr.get("longitudinal_selected"):
            longitudinal_selected += 1
        hydration = (
            aggregate.get("hydration") if isinstance(aggregate.get("hydration"), dict) else {}
        )
        snapshots = (
            hydration.get("snapshots") if isinstance(hydration.get("snapshots"), dict) else {}
        )
        if _has_snapshot(snapshots.get("before")):
            with_before += 1
        if _has_snapshot(snapshots.get("after")):
            with_after += 1
        future = snapshots.get("future")
        has_future = False
        if isinstance(future, dict):
            for label, snapshot in future.items():
                if _has_snapshot(snapshot):
                    label_key = str(label)
                    future_counts[label_key] = int(future_counts.get(label_key, 0)) + 1
                    has_future = True
        if has_future:
            with_future_any += 1

    recalculated = dict(existing_stats)
    recalculated.update(
        {
            "processed_prs": processed,
            "longitudinal_selected_prs": longitudinal_selected,
            "metrics_computed": metrics_computed,
            "with_before": with_before,
            "with_after": with_after,
            "with_future_any": with_future_any,
            "future_counts": dict(sorted(future_counts.items())),
        }
    )
    return recalculated


def run_curation(settings: CurationSettings | None = None) -> None:
    """Run the single supported curation pipeline.

    ``settings`` is already applied to environment variables by the CLI before
    this module is imported. It is still accepted here so run metadata can
    report the exact user-facing configuration that launched the run.
    """
    settings = settings or CurationSettings.from_env()
    start_time = datetime.now(timezone.utc)
    output_root = Path(LOCAL_OUTPUT_DIR) / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_timestamp = start_time.strftime("%Y%m%dT%H%M%SZ")
    metadata_inprogress_path = (
        output_root / f"run_metadata_{COHORT}_{metadata_timestamp}_inprogress.json"
    )
    metadata_json_path = output_root / f"run_metadata_{COHORT}_{metadata_timestamp}.json"

    current_preprocessing: Dict[str, Any] = {}
    current_sampling: Dict[str, Any] = {}
    current_longitudinal_sampling: Dict[str, Any] = {}
    current_processing: Dict[str, Any] = {}
    resume_from_existing_sample = bool(CURATION_RESUME_FROM_EXISTING_SAMPLE) or str(
        os.environ.get("CURATION_RESUME_FROM_EXISTING_SAMPLE", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    skip_initial_sample_processing = bool(CURATION_SKIP_INITIAL_SAMPLE_PROCESSING) or str(
        os.environ.get("CURATION_SKIP_INITIAL_SAMPLE_PROCESSING", "")
    ).strip().lower() in {"1", "true", "yes", "on"}

    def _metadata_config() -> Dict[str, Any]:
        return {
            "target_languages": list(TARGET_LANGUAGES),
            "only_merged_prs": ONLY_MERGED_PRS,
            "target_no_prs": TARGET_NO_PRS,
            "time_bucket_granularity": TIME_BUCKET_GRANULARITY,
            "popularity_buckets": POPULARITY_BUCKETS,
            "popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
            "fetch_repo_metadata": True,
            "longitudinal_target_no_prs": LONGITUDINAL_TARGET_NO_PRS,
            "metrics_backend": "multimetric_plus_custom_duplicated_lines_density",
            "curation_pipeline": "single_pass",
            "local_data_format": CURATION_LOCAL_DATA_FORMAT,
            "pass2_url_chunk_size": CURATION_PASS2_URL_CHUNK_SIZE,
            "partition_write_buffer_rows": CURATION_PARTITION_WRITE_BUFFER_ROWS,
            "input_dirs": [str(path) for path in settings.input_dirs],
            "output_dir": str(settings.output_dir),
            "sample_history_dir": str(SAMPLE_HISTORY_DIR) if SAMPLE_HISTORY_DIR else None,
            "resume_from_existing_sample": resume_from_existing_sample,
            "skip_initial_sample_processing": skip_initial_sample_processing,
            "github_token_count": len(settings.github_tokens),
        }

    def _write_inprogress_metadata(phase: str) -> None:
        payload = {
            "cohort": COHORT,
            "status": "in_progress",
            "phase": phase,
            "start_time_utc": start_time.isoformat(),
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            "config": _metadata_config(),
            "preprocessing": current_preprocessing,
            "sampling": current_sampling,
            "longitudinal_sampling": current_longitudinal_sampling,
            "processing": current_processing,
        }
        try:
            _write_json_atomic(metadata_inprogress_path, payload)
        except Exception as exc:
            print(f"[Curation] Warning: failed to persist in-progress run metadata: {exc}")

    _write_inprogress_metadata("starting")
    print(
        "[Curation] Input mode: source=local "
        f"local_data_format={CURATION_LOCAL_DATA_FORMAT} "
        f"roots={LOCAL_DIRECTORIES}"
    )
    preflight_manifest = preflight_local_group_input(
        COHORT,
        strict=True,
    )
    if not preflight_manifest.get("skipped", False):
        print(
            "[Curation] Input preflight complete: "
            f"format={preflight_manifest.get('local_data_format')} "
            f"shards={preflight_manifest.get('shard_count')} "
            f"rows={preflight_manifest.get('total_rows')}"
        )
    topup_index = _topup_index_path(output_root, COHORT)
    if resume_from_existing_sample:
        sampled_store = _sample_store_path(COHORT, basename="sampled_prs")
        sampled_rows = _load_sample_rows(sampled_store)
        if not sampled_store.exists() or not sampled_rows:
            raise RuntimeError(
                "CURATION_RESUME_FROM_EXISTING_SAMPLE=1 but existing sample store "
                f"was not found or is empty: {sampled_store}"
            )
        sampler_stats = _sample_stats_from_rows(sampled_rows)
        sampled_count = int(sampler_stats.get("sampled", 0))
        print(
            "[Curation] Resume-from-existing-sample enabled; skipping preprocessing "
            f"and sampling. Using sampled store: {sampled_store} (rows={sampled_count})"
        )
        current_preprocessing = {
            "skipped": True,
            "reason": "resume_from_existing_sample",
            "input_preflight": {
                "skipped": bool(preflight_manifest.get("skipped", False)),
                "local_data_format": preflight_manifest.get("local_data_format"),
                "shard_count": int(preflight_manifest.get("shard_count", 0) or 0),
                "total_rows": int(preflight_manifest.get("total_rows", 0) or 0),
                "invalid_shards": len(preflight_manifest.get("invalid_shards") or []),
                "missing_required_columns_shards": len(
                    preflight_manifest.get("missing_required_columns") or {}
                ),
            },
        }
        current_sampling = {
            "skipped": True,
            "sampled": sampled_count,
            "sampled_by_language": sampler_stats.get("language_counts", {}),
            "sampled_by_popularity": sampler_stats.get("popularity_counts", {}),
            "popularity_bucket_policy": sampler_stats.get("popularity_bucket_policy"),
            "popularity_cut_points": sampler_stats.get("popularity_cut_points", []),
            "sampled_store": str(sampled_store),
        }
        longitudinal_urls, longitudinal_stats, longitudinal_source = (
            _load_existing_longitudinal_urls(COHORT)
        )
        print(
            "[Curation] Loaded existing longitudinal sample: "
            f"{longitudinal_source or 'none'} (urls={len(longitudinal_urls)})"
        )
        current_longitudinal_sampling = {
            "skipped": True,
            "sampled": longitudinal_stats.get("sampled", 0),
            "sampled_by_language": longitudinal_stats.get("language_counts", {}),
            "sampled_by_popularity": longitudinal_stats.get("popularity_counts", {}),
            "popularity_bucket_policy": longitudinal_stats.get("popularity_bucket_policy"),
            "popularity_cut_points": longitudinal_stats.get("popularity_cut_points", []),
            "sampled_store": str(longitudinal_source) if longitudinal_source else None,
        }
        _write_inprogress_metadata("existing_sample_loaded")
    else:
        sample_history_identifiers = load_sample_history_pr_identifiers()
        print(f"[Curation] Starting preprocessing for cohort={COHORT}")
        if USE_TWO_PASS_STREAMING_SAMPLING:
            prs, preprocess_stats = preprocess_candidates_streaming(
                COHORT,
                batch_size=max(1, int(STREAMING_PARQUET_BATCH_SIZE)),
                exclude_identifiers=sample_history_identifiers,
                deduplicate_urls=True,
            )
        else:
            prs, preprocess_stats = preprocess_prs(COHORT)
            seen_urls: set[str] = set()
            filtered: list[Any] = []
            excluded_sample_history = 0
            dedup_skipped = 0
            for pr in prs:
                url = _pr_url(pr)
                if not url:
                    continue
                pr_id = str(getattr(pr, "id", "") or "").strip()
                if url in seen_urls:
                    dedup_skipped += 1
                    continue
                seen_urls.add(url)
                if (
                    url in sample_history_identifiers
                    or url.rstrip("/") in sample_history_identifiers
                    or pr_id in sample_history_identifiers
                ):
                    excluded_sample_history += 1
                    continue
                filtered.append(pr)
            prs = filtered
            preprocess_stats["retained_prs"] = len(prs)
            preprocess_stats["excluded_previously_seen"] = excluded_sample_history
            preprocess_stats["excluded_sample_history"] = excluded_sample_history
            preprocess_stats["dedup_skipped"] = dedup_skipped
        print(f"[Curation] Preprocessing complete. PRs retained: {len(prs)}")
        candidate_store_filtered = _candidate_store_path(output_root, COHORT)
        _write_candidates_jsonl(candidate_store_filtered, prs)
        filtered_count = len(prs)
        excluded_sample_history = int(
            preprocess_stats.get(
                "excluded_sample_history",
                preprocess_stats.get("excluded_previously_seen", 0),
            )
        )
        print(
            "[Curation] Excluded sample-history PRs: {excluded} (remaining candidates: {remaining})".format(
                excluded=excluded_sample_history,
                remaining=filtered_count,
            )
        )
        indexed_count = _write_topup_index(candidate_store_filtered, topup_index)
        print(f"[Curation] Top-up index written: {topup_index} (rows={indexed_count})")
        prs = []
        current_preprocessing = {
            "prs_loaded": preprocess_stats.get("loaded_prs", 0),
            "prs_retained": preprocess_stats.get("retained_prs", 0),
            "prs_excluded_sample_history": excluded_sample_history,
            "prs_excluded_previously_seen": excluded_sample_history,
            "prs_dedup_skipped": preprocess_stats.get("dedup_skipped", 0),
            "sample_history_identifier_count": len(sample_history_identifiers),
            "previously_seen_url_count": len(sample_history_identifiers),
            "prs_filtered": preprocess_stats.get("filtered_out_prs", 0),
            "prs_filtered_non_merged": preprocess_stats.get("filtered_non_merged", 0),
            "prs_filtered_added_only": preprocess_stats.get("filtered_added_only", 0),
            "prs_filtered_primary_notebook": preprocess_stats.get("filtered_primary_notebook", 0),
            "prs_missing_files": preprocess_stats.get("missing_files", 0),
            "prs_missing_languages": preprocess_stats.get("missing_languages", 0),
            "unknown_language_total": preprocess_stats.get("unknown_language_total", 0),
            "prs_per_language": preprocess_stats.get("per_language_counts", {}),
            "input_preflight": {
                "skipped": bool(preflight_manifest.get("skipped", False)),
                "local_data_format": preflight_manifest.get("local_data_format"),
                "shard_count": int(preflight_manifest.get("shard_count", 0) or 0),
                "total_rows": int(preflight_manifest.get("total_rows", 0) or 0),
                "invalid_shards": len(preflight_manifest.get("invalid_shards") or []),
                "missing_required_columns_shards": len(
                    preflight_manifest.get("missing_required_columns") or {}
                ),
            },
        }
        _write_inprogress_metadata("preprocessing_complete")
        sampled_store, sampler_stats = _sample_from_partitioned_candidates(
            candidate_store_filtered,
            cohort=COHORT,
            target=int(TARGET_NO_PRS),
            basename="sampled_prs",
            seed=42,
        )
        sampled_count = int(sampler_stats.get("sampled", 0))
        print(f"[Curation] Sampling complete. PRs retained: {sampled_count}")
        current_sampling = {
            "candidates": sampler_stats.get("candidates", 0),
            "sampled": sampler_stats.get("sampled", 0),
            "sampled_by_language": sampler_stats.get("language_counts", {}),
            "sampled_by_popularity": sampler_stats.get("popularity_counts", {}),
            "popularity_bucket_policy": sampler_stats.get("popularity_bucket_policy"),
            "popularity_cut_points": sampler_stats.get("popularity_cut_points", []),
        }
        # Candidate list is no longer needed after initial sampling.
        prs = []
        _write_inprogress_metadata("sampling_complete")
        longitudinal_urls, longitudinal_stats = _sample_longitudinal_urls_from_store(
            sampled_store,
            cohort=COHORT,
            target=int(LONGITUDINAL_TARGET_NO_PRS),
            seed=42,
        )
        print(
            "[Curation] Longitudinal sampling complete. PRs retained: " f"{len(longitudinal_urls)}"
        )
        current_longitudinal_sampling = {
            "candidates": longitudinal_stats.get("candidates", 0),
            "sampled": longitudinal_stats.get("sampled", 0),
            "sampled_by_language": longitudinal_stats.get("language_counts", {}),
            "sampled_by_popularity": longitudinal_stats.get("popularity_counts", {}),
            "popularity_bucket_policy": longitudinal_stats.get("popularity_bucket_policy"),
            "popularity_cut_points": longitudinal_stats.get("popularity_cut_points", []),
        }
        _write_inprogress_metadata("longitudinal_sampling_complete")

    print(f"[Curation] Starting processing for cohort={COHORT}")
    processing_stats: Dict[str, Any] = {
        "processed_prs": 0,
        "longitudinal_selected_prs": 0,
        "metrics_computed": 0,
        "metrics_failed": 0,
        "with_before": 0,
        "with_after": 0,
        "with_future_any": 0,
        "skipped_failed_on_resume": 0,
        "prefiltered_completed_on_resume": 0,
        "prefiltered_persisted_on_resume": 0,
        "future_counts": {},
        "language_time_total_seconds": {},
        "language_time_counts": {},
        "language_time_avg_seconds": {},
    }
    selected_total = sampled_count
    materialized_total = 0
    batch_count = 0
    pass2_chunk_size = max(1, int(CURATION_PASS2_URL_CHUNK_SIZE))
    pass2_parquet_batch_size = max(1, int(STREAMING_PARQUET_BATCH_SIZE))
    print(
        "[Curation] Pass-2 settings: "
        f"url_chunk_size={pass2_chunk_size}, parquet_batch_size={pass2_parquet_batch_size}"
    )
    if skip_initial_sample_processing:
        print(
            "[Curation] Initial sampled PR processing skipped; existing aggregate "
            "outputs will be used to drive top-up."
        )
        processing_stats = _recount_processing_stats_from_outputs(
            output_root,
            COHORT,
            processing_stats,
        )
        current_processing = {
            "processed_prs": processing_stats.get("processed_prs", 0),
            "longitudinal_selected_prs": processing_stats.get("longitudinal_selected_prs", 0),
            "metrics_computed": processing_stats.get("metrics_computed", 0),
            "metrics_failed": processing_stats.get("metrics_failed", 0),
            "prs_with_before": processing_stats.get("with_before", 0),
            "prs_with_after": processing_stats.get("with_after", 0),
            "prs_with_future_any": processing_stats.get("with_future_any", 0),
            "skipped_failed_on_resume": processing_stats.get("skipped_failed_on_resume", 0),
            "prefiltered_completed_on_resume": processing_stats.get(
                "prefiltered_completed_on_resume",
                0,
            ),
            "prefiltered_persisted_on_resume": processing_stats.get(
                "prefiltered_persisted_on_resume",
                0,
            ),
            "initial_sample_processing_skipped": True,
        }
        _write_inprogress_metadata("initial_sample_processing_skipped")
    elif USE_TWO_PASS_STREAMING_SAMPLING:
        for batch in _iter_materialized_pr_batches_single_scan(
            sampled_store,
            COHORT,
            output_root=output_root,
            parquet_batch_size=pass2_parquet_batch_size,
            output_batch_size=pass2_chunk_size,
        ):
            batch_count += 1
            materialized_total += len(batch)
            print(
                "[Curation] Streaming pass-2 processing batch "
                f"{batch_count}: materialized={len(batch)} "
                f"cumulative={materialized_total}/{selected_total}"
            )
            if not batch:
                continue
            batch_stats = process_prs(
                batch,
                COHORT,
                resume=RESUME_PROCESSING,
                longitudinal_urls=longitudinal_urls,
                skip_failed_prs_on_resume=resume_from_existing_sample,
            )
            processing_stats = _merge_processing_stats(processing_stats, batch_stats)
    else:
        for candidate_chunk in _iter_candidates_from_rows_store(
            sampled_store,
            chunk_size=pass2_chunk_size,
        ):
            batch_count += 1
            batch = candidate_chunk
            materialized_total += len(batch)
            if not batch:
                continue
            batch_stats = process_prs(
                batch,
                COHORT,
                resume=RESUME_PROCESSING,
                longitudinal_urls=longitudinal_urls,
                skip_failed_prs_on_resume=resume_from_existing_sample,
            )
            processing_stats = _merge_processing_stats(processing_stats, batch_stats)

    current_processing = {
        "processed_prs": processing_stats.get("processed_prs", 0),
        "longitudinal_selected_prs": processing_stats.get("longitudinal_selected_prs", 0),
        "metrics_computed": processing_stats.get("metrics_computed", 0),
        "metrics_failed": processing_stats.get("metrics_failed", 0),
        "prs_with_before": processing_stats.get("with_before", 0),
        "prs_with_after": processing_stats.get("with_after", 0),
        "prs_with_future_any": processing_stats.get("with_future_any", 0),
        "skipped_failed_on_resume": processing_stats.get("skipped_failed_on_resume", 0),
        "prefiltered_completed_on_resume": processing_stats.get(
            "prefiltered_completed_on_resume",
            0,
        ),
        "prefiltered_persisted_on_resume": processing_stats.get(
            "prefiltered_persisted_on_resume",
            0,
        ),
    }
    _write_inprogress_metadata("processing_initial_complete")
    language_quotas = _target_language_quotas(int(TARGET_NO_PRS))
    sampled_urls = set()
    if sampled_store.exists():
        with sampled_store.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url") or "").strip()
                if url:
                    sampled_urls.add(url)
    aggregated_processing_stats = dict(processing_stats)
    sampled_meta: dict[str, Dict[str, Any]] = _load_sampled_meta_from_store(sampled_store)
    attempted_urls: set[str] = set(sampled_urls)
    topup_rounds = 0
    for attempt in range(TOPUP_MAX_ATTEMPTS):
        output_root = Path(LOCAL_OUTPUT_DIR) / "output"
        aggregates_by_url = _load_aggregates_by_url(output_root, COHORT)
        failed_urls: list[str] = []
        for url in sorted(sampled_urls):
            aggregate = aggregates_by_url.get(url)
            if not isinstance(aggregate, dict):
                print(f"[Curation] Tool gate failed for {url}: missing aggregate metrics")
                failed_urls.append(url)
                continue
            ok, reasons = _pr_tool_gate_status(aggregate)
            if not ok:
                print(f"[Curation] Tool gate failed for {url}: {', '.join(reasons)}")
                failed_urls.append(url)
                _cleanup_failed_pr_outputs(output_root, aggregate)
        valid_processed_count = max(0, len(aggregates_by_url) - len(set(failed_urls)))
        processed_gap = max(0, int(TARGET_NO_PRS) - valid_processed_count)
        if processed_gap <= 0 and not failed_urls:
            break

        replacement_pool_meta = _build_replacement_pool_from_index(
            topup_index,
            selected_urls=sampled_urls,
            attempted_urls=attempted_urls,
        )
        candidate_meta: dict[str, Dict[str, Any]] = {}
        candidate_meta.update(sampled_meta)
        candidate_meta.update(replacement_pool_meta)
        if not candidate_meta:
            print("[Curation] Top-up stopped: candidate metadata unavailable.")
            break

        replacements: list[Any] = []
        replacement_longitudinal_urls: set[str] = set()
        for failed_url in failed_urls:
            replacement = _pick_replacement(
                failed_url=failed_url,
                selected_urls=sampled_urls,
                attempted_urls=attempted_urls,
                candidate_meta=candidate_meta,
            )
            if replacement is None:
                continue
            replacement_url = _pr_url(replacement)
            if not replacement_url:
                continue
            failed_meta = candidate_meta.get(failed_url) or {}
            replacement_meta = candidate_meta.get(replacement_url) or {}
            print(
                "[Curation] Replacing failed PR {failed} -> {replacement} "
                "(lang={lang}, time {ft}->{rt}, popularity {fp}->{rp})".format(
                    failed=failed_url,
                    replacement=replacement_url,
                    lang=failed_meta.get("primary_lang"),
                    ft=failed_meta.get("time_bucket"),
                    rt=replacement_meta.get("time_bucket"),
                    fp=failed_meta.get("popularity_bucket"),
                    rp=replacement_meta.get("popularity_bucket"),
                )
            )
            sampled_urls.discard(failed_url)
            sampled_urls.add(replacement_url)
            attempted_urls.add(replacement_url)
            replacements.append(replacement)
            if replacement_url in replacement_pool_meta:
                sampled_meta[replacement_url] = replacement_pool_meta[replacement_url]
            sampled_meta.pop(failed_url, None)
            if failed_url in longitudinal_urls:
                longitudinal_urls.discard(failed_url)
                longitudinal_urls.add(replacement_url)
                replacement_longitudinal_urls.add(replacement_url)

        additional_topups: list[Any] = []
        additional_longitudinal_urls: set[str] = set()
        if processed_gap > 0:
            remaining_pool = [
                meta.get("pr")
                for url, meta in candidate_meta.items()
                if (
                    url not in sampled_urls
                    and url not in attempted_urls
                    and meta.get("pr") is not None
                )
            ]
            if remaining_pool:
                topup_target = min(processed_gap, len(remaining_pool))
                quota_topups = _pick_quota_topups(
                    selected_urls=sampled_urls,
                    attempted_urls=attempted_urls,
                    candidate_meta=candidate_meta,
                    language_quotas=language_quotas,
                    target_count=topup_target,
                )
                if quota_topups:
                    additional_topups.extend(quota_topups)
                    print(
                        "[Curation] Quota top-up selected {count} PR(s).".format(
                            count=len(quota_topups)
                        )
                    )
                remaining_slots = max(0, topup_target - len(additional_topups))
                if remaining_slots > 0:
                    remaining_pool = [
                        meta.get("pr")
                        for url, meta in candidate_meta.items()
                        if (
                            url not in sampled_urls
                            and url not in attempted_urls
                            and meta.get("pr") is not None
                        )
                    ]
                    if remaining_pool:
                        print(
                            "[Curation] Top-up note: filling remaining {count} slot(s) "
                            "by relaxing language balance to preserve target count.".format(
                                count=remaining_slots
                            )
                        )
                        topup_sampled, _ = sample_prs(
                            remaining_pool,
                            COHORT,
                            target_no_prs=min(remaining_slots, len(remaining_pool)),
                            metadata_basename="sampled_prs_topup_relaxed_language",
                        )
                        additional_topups.extend(topup_sampled)
                        sampled_urls.update({_pr_url(pr) for pr in topup_sampled if _pr_url(pr)})
                        attempted_urls.update({_pr_url(pr) for pr in topup_sampled if _pr_url(pr)})
                        for pr in topup_sampled:
                            url = _pr_url(pr)
                            if not url:
                                continue
                            if url in replacement_pool_meta:
                                sampled_meta[url] = replacement_pool_meta[url]
                    else:
                        print(
                            "[Curation] Top-up note: no remaining unsampled candidates to fill "
                            f"remaining {remaining_slots} slot(s)."
                        )
                topup_longitudinal_target = min(
                    len(additional_topups),
                    max(
                        0,
                        round(
                            len(additional_topups)
                            * (LONGITUDINAL_TARGET_NO_PRS / max(1, TARGET_NO_PRS))
                        ),
                    ),
                )
                topup_longitudinal_sampled, _ = sample_prs(
                    additional_topups,
                    COHORT,
                    target_no_prs=topup_longitudinal_target,
                    metadata_basename="longitudinal_prs_topup",
                )
                additional_longitudinal_urls = {
                    _pr_url(pr) for pr in topup_longitudinal_sampled if _pr_url(pr)
                }
                longitudinal_urls.update(additional_longitudinal_urls)
            else:
                print(
                    "[Curation] Top-up note: processed_prs below target but no remaining unsampled candidates."
                )

        batch_by_url: dict[str, Any] = {}
        for pr in replacements + additional_topups:
            url = _pr_url(pr)
            if url:
                batch_by_url[url] = pr
        batch = list(batch_by_url.values())
        batch_longitudinal_urls = set(replacement_longitudinal_urls).union(
            additional_longitudinal_urls
        )

        for pr in batch:
            url = _pr_url(pr)
            if not url:
                continue
            if url in replacement_pool_meta:
                sampled_meta[url] = replacement_pool_meta[url]

        if not batch:
            print(
                "[Curation] Top-up stopped: no replacements or additional PRs available for this round."
            )
            break

        topup_rounds += 1
        print(
            f"[Curation] Top-up round {topup_rounds}/{TOPUP_MAX_ATTEMPTS}: "
            f"processing batch size={len(batch)} "
            f"(replacements={len(replacements)}, additional={len(additional_topups)}, gap={processed_gap})."
        )
        batch_for_processing = _materialize_for_processing(batch, COHORT)
        if USE_TWO_PASS_STREAMING_SAMPLING:
            print(
                "[Curation] Streaming pass-2 materialized top-up PRs: "
                f"{len(batch_for_processing)} / selected={len(batch)}"
            )
        topup_processing_stats = process_prs(
            batch_for_processing,
            COHORT,
            resume=RESUME_PROCESSING,
            longitudinal_urls=batch_longitudinal_urls,
            force_reprocess=True,
        )
        aggregated_processing_stats = _merge_processing_stats(
            aggregated_processing_stats, topup_processing_stats
        )
        aggregated_processing_stats = _recount_processing_stats_from_outputs(
            output_root,
            COHORT,
            aggregated_processing_stats,
        )
        current_processing = {
            "processed_prs": aggregated_processing_stats.get("processed_prs", 0),
            "longitudinal_selected_prs": aggregated_processing_stats.get(
                "longitudinal_selected_prs", 0
            ),
            "metrics_computed": aggregated_processing_stats.get("metrics_computed", 0),
            "metrics_failed": aggregated_processing_stats.get("metrics_failed", 0),
            "prs_with_before": aggregated_processing_stats.get("with_before", 0),
            "prs_with_after": aggregated_processing_stats.get("with_after", 0),
            "prs_with_future_any": aggregated_processing_stats.get("with_future_any", 0),
            "skipped_failed_on_resume": aggregated_processing_stats.get(
                "skipped_failed_on_resume",
                0,
            ),
            "prefiltered_completed_on_resume": aggregated_processing_stats.get(
                "prefiltered_completed_on_resume",
                0,
            ),
            "prefiltered_persisted_on_resume": aggregated_processing_stats.get(
                "prefiltered_persisted_on_resume",
                0,
            ),
            "topup_rounds_completed": topup_rounds,
        }
        _write_inprogress_metadata(f"topup_round_{topup_rounds}_complete")
    processing_stats = _recount_processing_stats_from_outputs(
        output_root,
        COHORT,
        aggregated_processing_stats,
    )
    print("[Curation] Processing complete.")
    run_errors = _load_run_errors(output_root, COHORT)
    final_selected = _final_selected_distributions(output_root, COHORT)
    errors = run_errors.get("errors", []) if isinstance(run_errors, dict) else []
    tool_gate_failures = sum(
        1
        for entry in errors
        if isinstance(entry, dict)
        and str(entry.get("source") or "") == "runtime"
        and str(entry.get("stage") or "") == "tool_gate"
    )
    repo_prepare_failures = sum(
        1
        for entry in errors
        if isinstance(entry, dict)
        and str(entry.get("source") or "") == "runtime"
        and str(entry.get("stage") or "") == "repo_prepare"
    )
    skipped_prs = tool_gate_failures
    skipped_total = tool_gate_failures + repo_prepare_failures
    end_time = datetime.now(timezone.utc)
    duration = end_time - start_time

    metadata_payload = {
        "cohort": COHORT,
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "topup_rounds": topup_rounds,
        "config": {
            "target_languages": list(TARGET_LANGUAGES),
            "only_merged_prs": ONLY_MERGED_PRS,
            "target_no_prs": TARGET_NO_PRS,
            "time_bucket_granularity": TIME_BUCKET_GRANULARITY,
            "popularity_buckets": POPULARITY_BUCKETS,
            "popularity_bucket_policy": POPULARITY_BUCKET_POLICY,
            "fetch_repo_metadata": True,
            "longitudinal_target_no_prs": LONGITUDINAL_TARGET_NO_PRS,
            "metrics_backend": "multimetric_plus_custom_duplicated_lines_density",
            "curation_pipeline": "single_pass",
            "local_data_format": CURATION_LOCAL_DATA_FORMAT,
            "input_dirs": [str(path) for path in settings.input_dirs],
            "output_dir": str(settings.output_dir),
            "sample_history_dir": str(SAMPLE_HISTORY_DIR) if SAMPLE_HISTORY_DIR else None,
            "resume_from_existing_sample": resume_from_existing_sample,
            "skip_initial_sample_processing": skip_initial_sample_processing,
            "github_token_count": len(settings.github_tokens),
        },
        "preprocessing": current_preprocessing,
        "sampling": current_sampling,
        "longitudinal_sampling": current_longitudinal_sampling,
        "processing": {
            "processed_prs": processing_stats.get("processed_prs", 0),
            "longitudinal_selected_prs": processing_stats.get("longitudinal_selected_prs", 0),
            "tool_gate_failures": tool_gate_failures,
            "repos_skipped_repo_prepare": repo_prepare_failures,
            "prs_skipped": skipped_prs,
            "total_skipped": skipped_total,
            "metrics_computed": processing_stats.get("metrics_computed", 0),
            "metrics_failed": processing_stats.get("metrics_failed", 0),
            "prs_with_before": processing_stats.get("with_before", 0),
            "prs_with_after": processing_stats.get("with_after", 0),
            "prs_with_future_any": processing_stats.get("with_future_any", 0),
            "skipped_failed_on_resume": processing_stats.get("skipped_failed_on_resume", 0),
            "prefiltered_completed_on_resume": processing_stats.get(
                "prefiltered_completed_on_resume",
                0,
            ),
            "prefiltered_persisted_on_resume": processing_stats.get(
                "prefiltered_persisted_on_resume",
                0,
            ),
            "final_selected": final_selected,
            "future_snapshot_counts": processing_stats.get("future_counts", {}),
            "language_time_avg_seconds": processing_stats.get("language_time_avg_seconds", {}),
            "language_time_counts": processing_stats.get("language_time_counts", {}),
            "language_time_total_seconds": processing_stats.get("language_time_total_seconds", {}),
        },
    }
    no_failures = (
        int((run_errors.get("summary") or {}).get("total_errors", 0)) == 0
        and int(processing_stats.get("metrics_failed", 0)) == 0
        and int(tool_gate_failures) == 0
        and int(repo_prepare_failures) == 0
    )
    cleanup_actions = {
        "enabled_condition": "no_failures",
        "no_failures": no_failures,
        "deleted_paths": [],
        "skipped_reason": None,
    }
    if no_failures:
        for directory in (output_root / "clones", output_root / "checkpoints"):
            if directory.exists():
                try:
                    shutil.rmtree(directory, ignore_errors=False)
                    cleanup_actions["deleted_paths"].append(str(directory))
                except Exception as exc:
                    cleanup_actions["skipped_reason"] = (
                        f"cleanup_failed:{directory.name}:{type(exc).__name__}:{exc}"
                    )
    else:
        cleanup_actions["skipped_reason"] = "failures_detected"
    metadata_payload["cleanup"] = cleanup_actions
    _write_json_atomic(metadata_json_path, metadata_payload)
    _write_json_atomic(
        metadata_inprogress_path,
        {
            **metadata_payload,
            "status": "completed",
            "phase": "completed",
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[Curation] Run metadata JSON written to: {metadata_json_path}")
    run_errors_path = output_root / f"run_errors_{_safe_cohort_component(COHORT)}.json"
    if run_errors_path.exists():
        print(f"[Curation] Run errors JSON written to: {run_errors_path}")


if __name__ == "__main__":
    run_curation()
