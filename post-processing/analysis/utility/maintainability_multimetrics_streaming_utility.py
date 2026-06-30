"""Streaming Multimetric detail analysis.

New curation runs embed Multimetric/custom-duplication snapshot rows directly in
processed PR parquet. Older runs may keep those rows in separate
``multimetric_snapshot_metrics.parquet`` files. This module supports both
sources and normalizes them into the same compact payload for JSON results and
plots.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from longitudinal_analysis_utility import build_longitudinal_results_from_payload
from maintainability_multimetrics_utility import (
    ALIAS_MULTIMETRIC_BASE_METRICS,
    DERIVED_MULTIMETRIC_METRICS,
    LONGITUDINAL_MULTIMETRIC_TIMEPOINTS,
    MULTIMETRIC_METRICS,
    ORIGINAL_MAINTAINABILITY_REPLACEMENT_METRICS,
    OUTPUT_FILENAME,
    RAW_MULTIMETRIC_METRICS,
    PER_KLOC_MULTIMETRIC_BASE_METRICS,
    discover_multimetric_snapshot_parquets,
)
from plotting_utility import order_humans_first
from streaming_parquet_utility import (
    coerce_float,
    coerce_int,
    coerce_str,
    iter_parquet_rows,
)


@dataclass(frozen=True)
class MultimetricPrIdentity:
    """Stable PR identity and grouping data used to match snapshot rows."""

    analysis_row_id: str
    cohort: str
    authorship_group: str
    characteristic_levels: dict[str, list[str]]
    maintainability_metrics: dict[str, float | None]


@dataclass
class MultimetricStreamingPayload:
    """Compact multimetric facts needed for result JSON and plots."""

    parquets: tuple[Path, ...]
    eligible_pr_count: int
    source: str = "external"
    embedded_snapshot_row_count: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    successful_snapshot_counts: Counter[str] = field(default_factory=Counter)
    matched_pr_ids: set[str] = field(default_factory=set)
    delta_pr_ids: set[str] = field(default_factory=set)
    longitudinal_pr_ids: set[str] = field(default_factory=set)
    delta_values_by_metric: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    delta_metadata_by_metric: dict[str, list[tuple[Any, Any, Any, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    characteristic_metric_values: dict[
        str, dict[str, dict[str, dict[str, list[float]]]]
    ] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )
    snapshots_by_pr: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    failed_parquets: list[dict[str, str]] = field(default_factory=list)

    def add_snapshot(
        self,
        *,
        identity: MultimetricPrIdentity,
        label: str,
        values: dict[str, float | None],
        metadata: tuple[Any, Any, Any, Any],
    ) -> None:
        """Record one successful snapshot and feed longitudinal accumulators."""
        if label in self.snapshots_by_pr[identity.analysis_row_id]:
            return
        self.matched_pr_ids.add(identity.analysis_row_id)
        self.successful_snapshot_counts[label] += 1
        self.snapshots_by_pr[identity.analysis_row_id][label] = {
            "identity": identity,
            "values": values,
            "metadata": metadata,
        }
        if label == "after":
            label_key = "0d"
        else:
            label_key = label
        if label_key == "after":
            label_key = "0d"
        if label_key == "0d" or label_key in LONGITUDINAL_MULTIMETRIC_TIMEPOINTS:
            for metric, value in values.items():
                if value is None:
                    continue
                self.longitudinal_values[identity.cohort][label_key][metric].append(
                    float(value)
                )
            self.longitudinal_pr_ids.add(identity.analysis_row_id)

    def finalize_deltas(self) -> None:
        """Compute before-to-after deltas once all snapshots have streamed."""
        for analysis_row_id, snapshots in self.snapshots_by_pr.items():
            before = snapshots.get("before")
            after = snapshots.get("after")
            if before is None or after is None:
                continue
            identity = after["identity"]
            before_values = before["values"]
            after_values = after["values"]
            metadata = after["metadata"]
            wrote_any = False
            for metric in MULTIMETRIC_METRICS:
                before_value = before_values.get(metric)
                after_value = after_values.get(metric)
                if before_value is None or after_value is None:
                    continue
                self.delta_values_by_metric[metric][identity.cohort].append(
                    float(after_value) - float(before_value)
                )
                self.delta_metadata_by_metric[metric].append(metadata)
                delta_value = float(after_value) - float(before_value)
                for dimension, levels in identity.characteristic_levels.items():
                    for level in levels:
                        scoped = self.characteristic_metric_values[dimension][level][metric]
                        scoped[f"authorship:{identity.authorship_group}"].append(
                            delta_value
                        )
                        scoped[f"cohort:{identity.cohort}"].append(delta_value)
                wrote_any = True
            if wrote_any:
                self.delta_pr_ids.add(analysis_row_id)

    def result_payload(self) -> dict[str, Any]:
        """Return coverage and schema metadata for the Multimetric analysis."""
        longitudinal_results = build_longitudinal_results_from_payload(
            self.longitudinal_values,
            metrics=MULTIMETRIC_METRICS,
        )
        coverage = self.coverage_summary()
        return {
            "eligible_pr_count": coverage["eligible_pr_count"],
            "matched_pr_count": coverage["matched_pr_count"],
            "unmatched_pr_count": coverage["unmatched_pr_count"],
            "delta_pr_count": coverage["delta_pr_count"],
            "delta_missing_pr_count": coverage["delta_missing_pr_count"],
            "longitudinal_pr_count": coverage["longitudinal_pr_count"],
            "longitudinal_missing_pr_count": coverage[
                "longitudinal_missing_pr_count"
            ],
            "multimetric_snapshot_parquet_count": len(self.parquets),
            "multimetric_snapshot_parquets": [str(path) for path in self.parquets],
            "multimetric_source": self.source,
            "embedded_snapshot_row_count": int(self.embedded_snapshot_row_count),
            "failed_multimetric_parquet_count": len(self.failed_parquets),
            "failed_multimetric_parquets": list(self.failed_parquets),
            "metrics": list(MULTIMETRIC_METRICS),
            "raw_metrics": list(RAW_MULTIMETRIC_METRICS),
            "per_kloc_metrics": list(DERIVED_MULTIMETRIC_METRICS),
            "alias_metrics": list(ALIAS_MULTIMETRIC_BASE_METRICS.values()),
            "cohorts": order_humans_first(
                {
                    cohort
                    for by_cohort in self.delta_values_by_metric.values()
                    for cohort in by_cohort
                }
            ),
            "status_counts": dict(self.status_counts),
            "successful_snapshot_counts": dict(self.successful_snapshot_counts),
            "delta_metric_missingness": {
                metric: {
                    "total_rows": sum(
                        len(values)
                        for values in self.delta_values_by_metric.get(metric, {}).values()
                    ),
                    "missing_rows": 0,
                    "present_rows": sum(
                        len(values)
                        for values in self.delta_values_by_metric.get(metric, {}).values()
                    ),
                }
                for metric in MULTIMETRIC_METRICS
            },
            "longitudinal": {
                "eligible_pr_count": longitudinal_results.get("eligible_pr_count", 0),
                "timepoints": longitudinal_results.get("timepoints", []),
                "metrics": list((longitudinal_results.get("metrics") or {}).keys()),
            },
        }

    def coverage_summary(self) -> dict[str, int]:
        """Return PR-level matching and longitudinal coverage counts."""
        eligible = int(self.eligible_pr_count)
        matched = len(self.matched_pr_ids)
        delta = len(self.delta_pr_ids)
        longitudinal = len(self.longitudinal_pr_ids)
        return {
            "eligible_pr_count": eligible,
            "matched_pr_count": matched,
            "unmatched_pr_count": max(0, eligible - matched),
            "delta_pr_count": delta,
            "delta_missing_pr_count": max(0, eligible - delta),
            "longitudinal_pr_count": longitudinal,
            "longitudinal_missing_pr_count": max(0, eligible - longitudinal),
        }

    def plot_payload(self) -> dict[str, Any]:
        """Return compact values consumed by Multimetric plotters."""
        return {
            "delta_values_by_metric": self.delta_values_by_metric,
            "delta_metadata_by_metric": {
                metric: _metadata_summary(rows)
                for metric, rows in self.delta_metadata_by_metric.items()
            },
            "longitudinal_values": self.longitudinal_values,
            "characteristic_metric_values": self.characteristic_metric_values,
        }


def stream_multimetric_analysis(
    *,
    multimetric_output_dir: Path | str | None,
    pr_index: list[dict[str, Any]],
    logger: Any | None = None,
) -> MultimetricStreamingPayload | None:
    """Stream legacy external Multimetric snapshot parquet files."""
    if multimetric_output_dir is None:
        return None
    parquets = discover_multimetric_snapshot_parquets(multimetric_output_dir)
    if not parquets:
        raise FileNotFoundError(
            "No multimetric snapshot parquet files found under "
            f"{Path(multimetric_output_dir)}"
        )
    by_url, by_repo_number = _identity_maps(pr_index)
    payload = MultimetricStreamingPayload(
        parquets=parquets,
        eligible_pr_count=len(pr_index),
        source="external",
    )
    for parquet_path in parquets:
        try:
            for row in iter_parquet_rows(parquet_path):
                _consume_snapshot_row(
                    row,
                    payload=payload,
                    by_url=by_url,
                    by_repo_number=by_repo_number,
                )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            payload.failed_parquets.append(
                {
                    "path": str(parquet_path),
                    "error": error_message,
                }
            )
            if logger is not None:
                logger.log(
                    "maintainability_multimetric_parquet_read_failed",
                    path=str(parquet_path),
                    error=error_message,
                )
            continue
    payload.finalize_deltas()
    return payload


def stream_embedded_multimetric_analysis(
    *,
    snapshot_rows: list[dict[str, Any]],
    pr_index: list[dict[str, Any]],
    logger: Any | None = None,
) -> MultimetricStreamingPayload | None:
    """Stream Multimetric snapshot rows embedded in curation parquet input."""
    if not snapshot_rows:
        return None
    by_url, by_repo_number = _identity_maps(pr_index)
    payload = MultimetricStreamingPayload(
        parquets=(),
        eligible_pr_count=len(pr_index),
        source="input",
        embedded_snapshot_row_count=len(snapshot_rows),
    )
    for row in snapshot_rows:
        try:
            _consume_snapshot_row(
                row,
                payload=payload,
                by_url=by_url,
                by_repo_number=by_repo_number,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            payload.failed_parquets.append(
                {
                    "path": "<embedded-curation-row>",
                    "error": error_message,
                }
            )
            if logger is not None:
                logger.log(
                    "maintainability_embedded_multimetric_row_failed",
                    error=error_message,
                )
    payload.finalize_deltas()
    return payload


def _consume_snapshot_row(
    row: Mapping[str, Any],
    *,
    payload: MultimetricStreamingPayload,
    by_url: dict[tuple[str, str], MultimetricPrIdentity],
    by_repo_number: dict[tuple[str, str, str], MultimetricPrIdentity],
) -> None:
    """Match and add one normalized snapshot row to the streaming payload."""
    status = str(coerce_str(row.get("status")) or "missing").strip().casefold()
    payload.status_counts[status] += 1
    if status != "success":
        return
    identity = _match_identity(row, by_url, by_repo_number)
    if identity is None:
        return
    label = _snapshot_label(row)
    if label is None:
        return
    payload.add_snapshot(
        identity=identity,
        label=label,
        values=_apply_original_maintainability_values(
            _metric_values(row),
            identity,
            label=label,
        ),
        metadata=(
            coerce_int(row.get("files_considered")),
            coerce_int(row.get("files_analyzed")),
            coerce_str(row.get("tool_version")),
            coerce_str(row.get("maintindex_mode")),
        ),
    )


def _identity_maps(
    pr_index: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], MultimetricPrIdentity],
    dict[tuple[str, str, str], MultimetricPrIdentity],
]:
    """Build URL and repository/PR-number lookup maps from curated PR rows."""
    by_url: dict[tuple[str, str], MultimetricPrIdentity] = {}
    by_repo_number: dict[tuple[str, str, str], MultimetricPrIdentity] = {}
    for record in pr_index:
        cohort = str(record.get("cohort") or "").strip().casefold()
        if not cohort:
            continue
        identity = MultimetricPrIdentity(
            analysis_row_id=str(record.get("analysis_row_id")),
            cohort=str(record.get("cohort")),
            authorship_group=str(record.get("authorship_group") or ""),
            characteristic_levels=_normalize_characteristic_levels(
                record.get("characteristic_levels")
            ),
            maintainability_metrics=_normalize_maintainability_metrics(
                record.get("metrics")
            ),
        )
        pr_url = coerce_str(record.get("pr_url"))
        if pr_url:
            by_url[(cohort, pr_url)] = identity
        repository_key = coerce_str(record.get("repository_key"))
        pr_number = coerce_str(record.get("pr_number"))
        if repository_key and pr_number:
            by_repo_number[(cohort, repository_key.casefold(), pr_number)] = identity
    return by_url, by_repo_number


def _match_identity(
    row: Mapping[str, Any],
    by_url: dict[tuple[str, str], MultimetricPrIdentity],
    by_repo_number: dict[tuple[str, str, str], MultimetricPrIdentity],
) -> MultimetricPrIdentity | None:
    """Match one snapshot row to a curated PR identity."""
    cohort = str(coerce_str(row.get("cohort")) or "").strip().casefold()
    if not cohort:
        return None
    pr_url = coerce_str(row.get("pr_url"))
    if pr_url is not None:
        identity = by_url.get((cohort, pr_url))
        if identity is not None:
            return identity
    repository_key = coerce_str(row.get("repository_key"))
    pr_number = coerce_str(row.get("pr_number"))
    if repository_key and pr_number:
        return by_repo_number.get((cohort, repository_key.casefold(), pr_number))
    return None


def _snapshot_label(row: Mapping[str, Any]) -> str | None:
    """Normalize snapshot identity to before, after, or configured timepoint."""
    label = str(coerce_str(row.get("snapshot_label")) or "").strip()
    kind = str(coerce_str(row.get("snapshot_kind")) or "").strip().casefold()
    days = coerce_int(row.get("days_after_merge"))
    if label.casefold() in {"before", "after"}:
        return label.casefold()
    if kind in {"before", "after"}:
        return kind
    if label in LONGITUDINAL_MULTIMETRIC_TIMEPOINTS:
        return label
    if days is not None:
        candidate = f"+{int(days)}d"
        if candidate in LONGITUDINAL_MULTIMETRIC_TIMEPOINTS:
            return candidate
    return None


def _metric_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract raw, alias, and per-KLOC Multimetric values from a row."""
    values = {metric: coerce_float(row.get(metric)) for metric in RAW_MULTIMETRIC_METRICS}
    loc = values.get("loc")
    for base_metric, derived_metric in PER_KLOC_MULTIMETRIC_BASE_METRICS.items():
        base_value = values.get(base_metric)
        values[derived_metric] = (
            None
            if loc is None or loc <= 0 or base_value is None
            else float(base_value) / (float(loc) / 1000.0)
        )
    for base_metric, alias_metric in ALIAS_MULTIMETRIC_BASE_METRICS.items():
        values[alias_metric] = values.get(base_metric)
    for metric in ORIGINAL_MAINTAINABILITY_REPLACEMENT_METRICS:
        values[metric] = None
    return values


def _apply_original_maintainability_values(
    values: dict[str, float | None],
    identity: MultimetricPrIdentity,
    *,
    label: str,
) -> dict[str, float | None]:
    """Overlay curation's custom duplication metric for before/after rows."""
    values = dict(values)
    if label == "after":
        values["original_duplication_density"] = identity.maintainability_metrics.get(
            "DuplicationDensity_Post"
        )
    elif label == "before":
        values["original_duplication_density"] = identity.maintainability_metrics.get(
            "DuplicationDensity_Before"
        )
    return values


def _normalize_characteristic_levels(value: Any) -> dict[str, list[str]]:
    """Normalize characteristic labels into dimension -> levels lists."""
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, list[str]] = {}
    for dimension, levels in value.items():
        if isinstance(levels, (list, tuple, set)):
            normalized[str(dimension)] = [
                str(level)
                for level in levels
                if str(level).strip()
            ]
    return normalized


def _normalize_maintainability_metrics(value: Any) -> dict[str, float | None]:
    """Coerce persisted maintainability metrics to float-or-null values."""
    if not isinstance(value, Mapping):
        return {}
    return {
        str(metric): coerce_float(metric_value)
        for metric, metric_value in value.items()
    }


def _numeric_metadata_summary(values: list[float]) -> dict[str, Any]:
    """Summarize numeric Multimetric metadata such as files analyzed."""
    if not values:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "n": len(values),
        "min": float(min(values)),
        "median": float(median(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


def _metadata_summary(rows: list[tuple[Any, Any, Any, Any]]) -> dict[str, Any]:
    """Summarize snapshot metadata retained for auditability."""
    files_considered = [float(row[0]) for row in rows if row[0] is not None]
    files_analyzed = [float(row[1]) for row in rows if row[1] is not None]
    return {
        "files_considered": _numeric_metadata_summary(files_considered),
        "files_analyzed": _numeric_metadata_summary(files_analyzed),
        "tool_versions": sorted(
            {
                str(row[2])
                for row in rows
                if row[2] is not None and str(row[2]).strip()
            }
        ),
        "maintindex_modes": sorted(
            {
                str(row[3])
                for row in rows
                if row[3] is not None and str(row[3]).strip()
            }
        ),
    }
