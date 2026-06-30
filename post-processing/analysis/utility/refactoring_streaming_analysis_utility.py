"""Streaming refactoring analysis accumulator."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from curation_parquet_utility import CohortParquetFiles
from refactoring_analysis_utility import (
    MURPHY_HILL_COUNT_SOURCE_STORED,
    MURPHY_HILL_COUNT_SOURCE_TAXONOMY,
    MURPHY_HILL_LEVELS,
    REFACTORING_LONGITUDINAL_TIMEPOINTS,
    REFACTORING_SUCCESS_STATUS,
    _load_refactoring_taxonomy_classifier,
    _named_refactoring_type_counts,
    _taxonomy_murphy_hill_level,
    normalize_refactoring_tool_status,
)
from characteristics_analysis_utility import CHARACTERISTIC_DOMAINS
from streaming_parquet_utility import (
    StreamingPrFacts,
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_str,
    extract_common_pr_facts,
    iter_cohort_parquet_rows,
    json_mapping,
    nested_get,
)
from topic_groups_utility import TopicGroupRecord, filter_topic_group_records_by_confidence


_COMMON_REFACTORING_COLUMNS = (
    "id",
    "url",
    "number",
    "authored_by_agent",
    "author_agent",
    "discovered_agent",
    "longitudinal_selected",
    "base_repository",
    "base_repository_full",
    "pr_primary_language_effective",
    "repository_metadata.id",
    "repository_metadata.name_with_owner",
    "repository_metadata.stargazer_count",
    "metrics.refactoring_metrics.status",
    "metrics.refactoring_metrics.refactoring_metrics.metrics.refactor_type_count",
    "metrics.refactoring_metrics.refactoring_metrics.metrics.refactor_murphyhill_count",
    "metrics.refactoring_metrics.refactoring_metrics.metrics.refactor_added_lines",
    "metrics.refactoring_metrics.refactoring_metrics.metrics.refactor_removed_lines",
    "metrics.maintainability_metrics.maintainability_indicators.summary."
    "snapshot_measures.before.ncloc",
)
REFACTORING_STREAM_COLUMNS = _COMMON_REFACTORING_COLUMNS + tuple(
    column
    for label in REFACTORING_LONGITUDINAL_TIMEPOINTS
    for column in (
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.status",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.available",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.snapshot_available",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.snapshot_commit",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.refactoring_tool_collected",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.refactor_count",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.refactor_density",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.refactor_diversity",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.refactor_magnitude_lines",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.retention.trackable_refactoring_operations",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.retention.retention_rate",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.retention.touched_refactoring_zone_lines",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.future_impact.touched_refactoring_zone_lines_count",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.future_impact.touched_refactoring_zone_lines",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.future_impact.touching_commits_count",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.future_impact.pr_changed_line_context.touching_commits_count",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.pr_changed_line_future_impact.touching_commits_count",
    )
)


@dataclass
class RefactoringPrMetricRow:
    """Compact successful-PR metric row retained for statistics and plots."""

    analysis_row_id: str
    cohort: str
    authorship_group: str
    language: str | None
    base_repository_id: str | None
    repository_key: str | None
    stargazer_count: int | None
    ref_count: float
    ref_density: float | None
    ref_diversity: float
    ref_added: float | None
    ref_removed: float | None
    ref_mag_lines: float | None


@dataclass
class RefactoringStreamingAccumulator:
    """Compact streaming state for refactoring analysis."""

    murphy_hill_count_source: str = MURPHY_HILL_COUNT_SOURCE_TAXONOMY
    topic_group_records: Iterable[TopicGroupRecord] = ()
    seen_stable_pr_keys: set[str] = field(default_factory=set)
    all_pr_count: int = 0
    failed_pr_count: int = 0
    successful_prs: list[RefactoringPrMetricRow] = field(default_factory=list)
    type_counts_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    type_counts_by_language_scope: dict[str, dict[str, Counter[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    murphy_counts_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    prs_with_refop_by_scope: Counter[str] = field(default_factory=Counter)
    prs_with_murphy_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    per_pr_murphy_counts_by_scope: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    metric_values_by_scope: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    refop_positive_metric_values_by_scope: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    characteristic_metric_values: dict[
        str, dict[str, dict[str, dict[str, list[float]]]]
    ] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )
    characteristic_counts_by_cohort: dict[str, dict[str, Counter[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    def __post_init__(self) -> None:
        self._analysis_row_counter = 0
        self._taxonomy_classifier = (
            _load_refactoring_taxonomy_classifier()
            if self.murphy_hill_count_source == MURPHY_HILL_COUNT_SOURCE_TAXONOMY
            else None
        )
        records = filter_topic_group_records_by_confidence(list(self.topic_group_records))
        self._topics_by_repository_id: dict[str, list[TopicGroupRecord]] = defaultdict(list)
        self._topics_by_repository_key: dict[str, list[TopicGroupRecord]] = defaultdict(list)
        for record in records:
            if record.repository_id:
                self._topics_by_repository_id[str(record.repository_id)].append(record)
            if record.repository_key:
                self._topics_by_repository_key[str(record.repository_key)].append(record)

    def add_row(
        self,
        row: Mapping[str, Any],
        *,
        cohort: str,
        source_file: Path | str,
        source_row_number: int,
        excluded_agents: tuple[str, ...] = (),
    ) -> bool:
        """Consume one raw curation row. Return whether it survived filtering/dedup."""
        facts = extract_common_pr_facts(
            row,
            cohort=cohort,
            source_file=source_file,
            source_row_number=source_row_number,
        )
        if _is_excluded_agent(facts, excluded_agents):
            return False
        if facts.stable_pr_key is not None:
            stable_key = f"{facts.cohort}:{facts.stable_pr_key}"
            if stable_key in self.seen_stable_pr_keys:
                return False
            self.seen_stable_pr_keys.add(stable_key)
        self.all_pr_count += 1
        status = normalize_refactoring_tool_status(
            nested_get(row, "metrics", "refactoring_metrics", "status")
        )
        if status == "failed":
            self.failed_pr_count += 1
        if status != REFACTORING_SUCCESS_STATUS:
            return True
        self._add_success_row(row, facts)
        return True

    def _add_success_row(self, row: Mapping[str, Any], facts: StreamingPrFacts) -> None:
        metrics = json_mapping(
            nested_get(
                row,
                "metrics",
                "refactoring_metrics",
                "refactoring_metrics",
                "metrics",
            )
        )
        type_counts = _named_refactoring_type_counts(
            _int_counter_dict(metrics.get("refactor_type_count"))
        )
        ref_count = sum(type_counts.values())
        ref_diversity = _shannon_diversity(type_counts)
        nloc_before = coerce_float(
            nested_get(
                row,
                "metrics",
                "maintainability_metrics",
                "maintainability_indicators",
                "summary",
                "snapshot_measures",
                "before",
                "ncloc",
            )
        )
        ref_density = (
            float(ref_count) / (float(nloc_before) / 1000.0)
            if nloc_before is not None and nloc_before > 0
            else None
        )
        added_raw = coerce_float(metrics.get("refactor_added_lines")) or 0.0
        removed_raw = coerce_float(metrics.get("refactor_removed_lines")) or 0.0
        magnitude_raw = added_raw + removed_raw
        ref_added = added_raw / ref_count if ref_count > 0 else None
        ref_removed = removed_raw / ref_count if ref_count > 0 else None
        ref_mag_lines = magnitude_raw / ref_count if ref_count > 0 else None
        self._analysis_row_counter += 1
        metric_row = RefactoringPrMetricRow(
            analysis_row_id=str(self._analysis_row_counter),
            cohort=facts.cohort,
            authorship_group=facts.authorship_group,
            language=facts.language,
            base_repository_id=facts.base_repository_id,
            repository_key=facts.repository_key,
            stargazer_count=facts.stargazer_count,
            ref_count=float(ref_count),
            ref_density=ref_density,
            ref_diversity=float(ref_diversity),
            ref_added=ref_added,
            ref_removed=ref_removed,
            ref_mag_lines=ref_mag_lines,
        )
        self.successful_prs.append(metric_row)
        scopes = _scopes_for_row(facts)
        if ref_count > 0:
            for scope in scopes:
                self.prs_with_refop_by_scope[scope] += 1
        for refactoring_type, count in type_counts.items():
            for scope in scopes:
                self.type_counts_by_scope[scope][refactoring_type] += int(count)
                if facts.language:
                    self.type_counts_by_language_scope[scope][facts.language][
                        refactoring_type
                    ] += int(count)
        murphy_counts = self._murphy_counts(metrics, type_counts)
        if ref_count > 0:
            for scope in scopes:
                for level in MURPHY_HILL_LEVELS:
                    self.per_pr_murphy_counts_by_scope[scope][level].append(
                        float(murphy_counts.get(level, 0))
                    )
        for level, count in murphy_counts.items():
            if count <= 0:
                continue
            for scope in scopes:
                self.murphy_counts_by_scope[scope][level] += int(count)
                self.prs_with_murphy_by_scope[scope][level] += 1
        self._add_metric_values(metric_row, scopes)
        self._add_characteristic_values(metric_row, int(ref_count))
        self._add_longitudinal_values(row, facts, ref_count)

    def _murphy_counts(
        self,
        metrics: Mapping[str, Any],
        type_counts: Mapping[str, int],
    ) -> Counter[str]:
        counts: Counter[str] = Counter()
        if self.murphy_hill_count_source == MURPHY_HILL_COUNT_SOURCE_STORED:
            for level, count in _int_counter_dict(
                metrics.get("refactor_murphyhill_count")
            ).items():
                if str(level) in MURPHY_HILL_LEVELS and count > 0:
                    counts[str(level)] += int(count)
            return counts
        classifier = self._taxonomy_classifier
        if classifier is None:
            return counts
        for refactoring_type, count in type_counts.items():
            level = _taxonomy_murphy_hill_level(refactoring_type, classifier)
            if level in MURPHY_HILL_LEVELS and count > 0:
                counts[level] += int(count)
        return counts

    def _add_metric_values(
        self,
        metric_row: RefactoringPrMetricRow,
        scopes: list[str],
    ) -> None:
        values = {
            "RefCount": metric_row.ref_count,
            "RefDensity": metric_row.ref_density,
            "RefDiversity": metric_row.ref_diversity,
            "RefAdded": metric_row.ref_added,
            "RefRemoved": metric_row.ref_removed,
            "RefMagLines": metric_row.ref_mag_lines,
        }
        for scope in scopes:
            for metric, value in values.items():
                if value is not None:
                    self.metric_values_by_scope[metric][scope].append(float(value))
                    if metric_row.ref_count > 0:
                        self.refop_positive_metric_values_by_scope[metric][
                            scope
                        ].append(float(value))

    def _add_characteristic_values(
        self,
        metric_row: RefactoringPrMetricRow,
        item_count: int,
    ) -> None:
        if item_count <= 0:
            return
        values = {
            "RefCount": metric_row.ref_count,
            "RefDensity": metric_row.ref_density,
            "RefDiversity": metric_row.ref_diversity,
            "RefAdded": metric_row.ref_added,
            "RefRemoved": metric_row.ref_removed,
            "RefMagLines": metric_row.ref_mag_lines,
        }
        levels_by_dimension = _characteristic_levels_for_row(
            metric_row.language,
            metric_row.base_repository_id,
            metric_row.repository_key,
            metric_row.stargazer_count,
            self._topics_by_repository_id,
            self._topics_by_repository_key,
        )
        for dimension, levels in levels_by_dimension.items():
            for level in levels:
                self.characteristic_counts_by_cohort[dimension][
                    metric_row.cohort
                ][level] += int(item_count) if item_count > 0 else 0
                for metric, value in values.items():
                    if value is None:
                        continue
                    value_float = float(value)
                    scoped = self.characteristic_metric_values[dimension][level][metric]
                    scoped[f"authorship:{metric_row.authorship_group}"].append(value_float)
                    scoped[f"cohort:{metric_row.cohort}"].append(value_float)

    def _add_longitudinal_values(
        self,
        row: Mapping[str, Any],
        facts: StreamingPrFacts,
        ref_count: int,
    ) -> None:
        if not facts.longitudinal_selected or ref_count <= 0:
            return
        future = json_mapping(
            nested_get(
                row,
                "metrics",
                "refactoring_metrics",
                "refactoring_metrics",
                "metrics",
                "refactor_future_snapshot_metrics",
            )
        )
        if not future:
            return
        future_by_label = {
            label: json_mapping(future.get(label))
            for label in REFACTORING_LONGITUDINAL_TIMEPOINTS
        }
        available_by_label = {
            label: payload
            for label, payload in future_by_label.items()
            if payload and _future_snapshot_available(payload)
        }
        if not available_by_label:
            return

        baseline_values = self.longitudinal_values[facts.cohort]["0d"]
        if any(
            (
                _first_float(
                    json_mapping(payload.get("retention")).get(
                        "trackable_refactoring_operations"
                    )
                )
                or 0.0
            )
            > 0.0
            for payload in available_by_label.values()
        ):
            self._append_longitudinal(baseline_values, "RefRetentionRate", 1.0)
        self._append_longitudinal(baseline_values, "RefZoneFutureTouchedLines", 0.0)
        self._append_longitudinal(baseline_values, "FutureTouchingCommits", 0.0)

        for label, payload in future_by_label.items():
            if not payload:
                continue
            if not _future_snapshot_available(payload):
                continue
            cohort_values = self.longitudinal_values[facts.cohort][label]
            retention = json_mapping(payload.get("retention"))
            future_impact = json_mapping(payload.get("future_impact"))
            pr_changed_impact = json_mapping(payload.get("pr_changed_line_future_impact"))
            status = str(payload.get("status") or "").strip().casefold()
            refactoring_tool_collected = coerce_bool(
                payload.get("refactoring_tool_collected")
            )
            if refactoring_tool_collected is None:
                refactoring_tool_collected = status == REFACTORING_SUCCESS_STATUS
            if status == REFACTORING_SUCCESS_STATUS and refactoring_tool_collected:
                self._append_longitudinal(
                    cohort_values,
                    "RefCount",
                    coerce_float(payload.get("refactor_count")),
                )
                self._append_longitudinal(
                    cohort_values,
                    "RefDensity",
                    coerce_float(payload.get("refactor_density")),
                )
                self._append_longitudinal(
                    cohort_values,
                    "RefDiversity",
                    coerce_float(payload.get("refactor_diversity")),
                )
                future_magnitude = coerce_float(payload.get("refactor_magnitude_lines"))
                if future_magnitude is not None:
                    future_magnitude = future_magnitude / float(ref_count)
                self._append_longitudinal(
                    cohort_values,
                    "RefMagLines",
                    future_magnitude,
                )
            trackable_refops = _first_float(
                retention.get("trackable_refactoring_operations")
            )
            if trackable_refops is not None and trackable_refops > 0.0:
                self._append_longitudinal(
                    cohort_values,
                    "RefRetentionRate",
                    coerce_float(retention.get("retention_rate")),
                )
            touched_lines = _first_float(
                future_impact.get("touched_refactoring_zone_lines_count"),
                future_impact.get("touched_refactoring_zone_lines"),
                retention.get("touched_refactoring_zone_lines"),
            )
            if touched_lines is None:
                touched_lines = 0.0
            self._append_longitudinal(
                cohort_values,
                "RefZoneFutureTouchedLines",
                touched_lines / float(ref_count),
            )
            touching_commits = _first_float(
                pr_changed_impact.get("touching_commits_count"),
                json_mapping(future_impact.get("pr_changed_line_context")).get(
                    "touching_commits_count"
                ),
                future_impact.get("touching_commits_count"),
            )
            self._append_longitudinal(
                cohort_values,
                "FutureTouchingCommits",
                touching_commits,
            )

    @staticmethod
    def _append_longitudinal(
        values: dict[str, list[float]],
        metric: str,
        value: float | None,
    ) -> None:
        if value is not None:
            values[metric].append(float(value))

    def compact_payload(self) -> dict[str, Any]:
        """Return compact facts for later result/plot payload builders."""
        return {
            "all_pr_count": self.all_pr_count,
            "failed_pr_count": self.failed_pr_count,
            "success_pr_count": len(self.successful_prs),
            "type_counts_by_scope": _counter_tree_to_dict(self.type_counts_by_scope),
            "type_counts_by_language_scope": {
                scope: {
                    language: dict(counter)
                    for language, counter in by_language.items()
                }
                for scope, by_language in self.type_counts_by_language_scope.items()
            },
            "murphy_counts_by_scope": _counter_tree_to_dict(self.murphy_counts_by_scope),
            "prs_with_refop_by_scope": dict(self.prs_with_refop_by_scope),
            "prs_with_murphy_by_scope": {
                scope: dict(counter)
                for scope, counter in self.prs_with_murphy_by_scope.items()
            },
            "per_pr_murphy_counts_by_scope": {
                scope: {
                    level: values
                    for level, values in by_level.items()
                }
                for scope, by_level in self.per_pr_murphy_counts_by_scope.items()
            },
            "metric_values_by_scope": {
                metric: {
                    scope: values
                    for scope, values in by_scope.items()
                }
                for metric, by_scope in self.metric_values_by_scope.items()
            },
            "refop_positive_metric_values_by_scope": {
                metric: {
                    scope: values
                    for scope, values in by_scope.items()
                }
                for metric, by_scope in self.refop_positive_metric_values_by_scope.items()
            },
            "characteristic_metric_values": {
                dimension: {
                    level: {
                        metric: {
                            scope: values
                            for scope, values in by_scope.items()
                        }
                        for metric, by_scope in by_metric.items()
                    }
                    for level, by_metric in by_level.items()
                }
                for dimension, by_level in self.characteristic_metric_values.items()
            },
            "characteristic_counts_by_cohort": {
                dimension: {
                    cohort: dict(counter)
                    for cohort, counter in by_cohort.items()
                }
                for dimension, by_cohort in self.characteristic_counts_by_cohort.items()
            },
            "longitudinal_values": {
                cohort: {
                    label: {
                        metric: values
                        for metric, values in by_metric.items()
                    }
                    for label, by_metric in by_label.items()
                }
                for cohort, by_label in self.longitudinal_values.items()
            },
        }

    def result_payload(self) -> dict[str, Any]:
        """Return the public refactoring result JSON payload."""
        cohorts = sorted(
            {
                scope[len("cohort:") :]
                for scope in self.metric_values_by_scope.get("RefCount", {})
                if scope.startswith("cohort:")
            },
            key=lambda value: (
                0 if value.casefold() in {"human", "humans"} else 1,
                value,
            ),
        )
        return {
            "eligible_pr_count": self._eligible_count("overall"),
            "eligible_pr_count_per_cohort": {
                cohort: self._eligible_count(f"cohort:{cohort}")
                for cohort in cohorts
            },
            "refactoring_tool_failed": {
                "pull_request_count": int(self.failed_pr_count),
                "total_pull_request_count": int(self.all_pr_count),
                "percentage": _safe_percentage(self.failed_pr_count, self.all_pr_count),
            },
            "overall": self._summary_for_scope("overall"),
            "per_cohort": {
                cohort: self._summary_for_scope(f"cohort:{cohort}")
                for cohort in cohorts
            },
            "agents_vs_humans": {
                group: self._summary_for_scope(f"authorship_group:{group}")
                for group in ("human", "agent")
            },
        }

    def _summary_for_scope(self, scope: str) -> dict[str, Any]:
        total_refops = sum(self.type_counts_by_scope.get(scope, Counter()).values())
        eligible_pr_count = self._eligible_count(scope)
        return {
            "total_standardized_refops": int(total_refops),
            "low_murphy_hill_refops": _refop_count_payload(
                self.murphy_counts_by_scope.get(scope, Counter()).get("low", 0),
                total_refops,
            ),
            "medium_murphy_hill_refops": _refop_count_payload(
                self.murphy_counts_by_scope.get(scope, Counter()).get("medium", 0),
                total_refops,
            ),
            "high_murphy_hill_refops": _refop_count_payload(
                self.murphy_counts_by_scope.get(scope, Counter()).get("high", 0),
                total_refops,
            ),
            "prs_with_standardized_refop": _count_payload(
                self.prs_with_refop_by_scope.get(scope, 0),
                eligible_pr_count,
            ),
            "prs_with_low_murphy_hill_refop": _count_payload(
                self.prs_with_murphy_by_scope.get(scope, Counter()).get("low", 0),
                eligible_pr_count,
            ),
            "prs_with_medium_murphy_hill_refop": _count_payload(
                self.prs_with_murphy_by_scope.get(scope, Counter()).get("medium", 0),
                eligible_pr_count,
            ),
            "prs_with_high_murphy_hill_refop": _count_payload(
                self.prs_with_murphy_by_scope.get(scope, Counter()).get("high", 0),
                eligible_pr_count,
            ),
            "top_standardized_refactoring_operations": _top_items(
                self.type_counts_by_scope.get(scope, Counter()),
            ),
            "top_standardized_refactoring_operations_per_language": {
                language: _top_items(counter)
                for language, counter in sorted(
                    self.type_counts_by_language_scope.get(scope, {}).items()
                )
            },
        }

    def _eligible_count(self, scope: str) -> int:
        return len(self.metric_values_by_scope.get("RefCount", {}).get(scope, []))


def stream_refactoring_analysis(
    cohort_inputs: list[CohortParquetFiles],
    *,
    excluded_agents: tuple[str, ...],
    murphy_hill_count_source: str,
    topic_group_records: list[TopicGroupRecord] | None = None,
    progress_logger: Callable[..., Any] | None = None,
    batch_size: int = 256,
) -> RefactoringStreamingAccumulator:
    """Stream raw parquet rows into a refactoring accumulator."""
    accumulator = RefactoringStreamingAccumulator(
        murphy_hill_count_source=murphy_hill_count_source,
        topic_group_records=topic_group_records or [],
    )
    for cohort, source_file, source_row_number, row in iter_cohort_parquet_rows(
        cohort_inputs,
        columns=REFACTORING_STREAM_COLUMNS,
        batch_size=batch_size,
        progress_logger=progress_logger,
    ):
        accumulator.add_row(
            row,
            cohort=cohort,
            source_file=source_file,
            source_row_number=source_row_number,
            excluded_agents=excluded_agents,
        )
    return accumulator


def _safe_percentage(numerator: float, denominator: float) -> float:
    """Return a zero-safe ratio for count payloads."""
    return float(numerator / denominator) if denominator else 0.0


def _count_payload(total: int, denominator: int) -> dict[str, int | float]:
    """Return PR count plus denominator and percentage metadata."""
    return {
        "pull_request_count": int(total),
        "eligible_pull_request_count": int(denominator),
        "percentage": _safe_percentage(total, denominator),
    }


def _refop_count_payload(refop_count: int, denominator: int) -> dict[str, int | float]:
    """Return refactoring-operation count plus percentage metadata."""
    return {
        "refop_count": int(refop_count),
        "percentage": _safe_percentage(refop_count, denominator),
    }


def _top_items(
    counter: Counter[str],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the most frequent standardized refactoring operation types."""
    total = sum(counter.values())
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))
    return [
        {
            "rank": index,
            "refactoring_type": str(label),
            "refop_count": int(count),
            "percentage_within_group": _safe_percentage(count, total),
        }
        for index, (label, count) in enumerate(ordered[:limit], start=1)
    ]


def _is_excluded_agent(
    facts: StreamingPrFacts,
    excluded_agents: tuple[str, ...],
) -> bool:
    """Return whether a row belongs to an excluded agent cohort."""
    if facts.authorship_group != "agent":
        return False
    excluded = {agent.strip().casefold() for agent in excluded_agents if agent.strip()}
    if not excluded:
        return False
    return (
        (facts.agent_label or "").strip().casefold() in excluded
        or facts.cohort.strip().casefold() in excluded
    )


def _scopes_for_row(facts: StreamingPrFacts) -> list[str]:
    """Return the aggregate scopes affected by one PR row."""
    scopes = [
        "overall",
        f"cohort:{facts.cohort}",
        f"authorship_group:{facts.authorship_group}",
    ]
    if facts.language:
        scopes.append(f"language:{facts.language}")
    return scopes


def _int_counter_dict(value: Any) -> dict[str, int]:
    """Parse a JSON-like mapping into positive integer counts."""
    raw = json_mapping(value)
    counts: dict[str, int] = {}
    for label, count in raw.items():
        parsed_count = coerce_int(count)
        if parsed_count is not None and parsed_count > 0:
            counts[str(label)] = int(parsed_count)
    return counts


def _first_float(*values: Any) -> float | None:
    """Return the first value that can be coerced to a float."""
    for value in values:
        parsed = coerce_float(value)
        if parsed is not None:
            return parsed
    return None


def _future_snapshot_available(payload: Mapping[str, Any]) -> bool:
    """Return whether a future snapshot metric payload is usable."""
    explicit_available = coerce_bool(payload.get("available"))
    if explicit_available is not None:
        return explicit_available
    snapshot_available = coerce_bool(payload.get("snapshot_available"))
    if snapshot_available is not None:
        return snapshot_available
    return coerce_str(payload.get("snapshot_commit")) is not None


def _shannon_diversity(counts: Mapping[str, int]) -> float:
    """Compute Shannon diversity over standardized refactoring counts."""
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    diversity = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        proportion = float(count) / float(total)
        diversity += -1.0 * proportion * (math.log(proportion) / math.log(2.0))
    return diversity


def _counter_tree_to_dict(tree: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    """Convert nested counters into JSON-safe dictionaries."""
    return {
        str(scope): {str(key): int(value) for key, value in counter.items()}
        for scope, counter in tree.items()
    }


def _characteristic_levels_for_row(
    language: str | None,
    base_repository_id: str | None,
    repository_key: str | None,
    stargazer_count: int | None,
    topics_by_repository_id: Mapping[str, list[TopicGroupRecord]],
    topics_by_repository_key: Mapping[str, list[TopicGroupRecord]],
) -> dict[str, list[str]]:
    """Return characteristic levels used by companion subgroup analyses."""
    levels: dict[str, list[str]] = {}
    language_key = _characteristic_language(language)
    if language_key is not None:
        levels["language"] = [language_key]
    popularity_key = _popularity_group(stargazer_count)
    if popularity_key is not None:
        levels["popularity"] = [popularity_key]
    domain_values: list[str] = []
    if base_repository_id:
        domain_values.extend(
            record.topic_group
            for record in topics_by_repository_id.get(str(base_repository_id), [])
        )
    if not domain_values and repository_key:
        domain_values.extend(
            record.topic_group
            for record in topics_by_repository_key.get(str(repository_key), [])
        )
    domain_values = [
        domain
        for domain in CHARACTERISTIC_DOMAINS
        if domain in set(domain_values)
    ]
    if domain_values:
        levels["domain"] = domain_values
    return levels


def _characteristic_language(language: str | None) -> str | None:
    """Normalize supported implementation languages to canonical labels."""
    value = str(language or "").strip().casefold()
    if value == "python":
        return "python"
    if value in {"javascript", "js"}:
        return "javascript"
    if value == "java":
        return "java"
    if value in {"c++", "cpp", "cxx"}:
        return "c++"
    return None


def _popularity_group(stargazer_count: int | None) -> str | None:
    """Bucket repository popularity using the configured three-level scheme."""
    if stargazer_count is None:
        return None
    if int(stargazer_count) <= 0:
        return "low"
    if int(stargazer_count) <= 18:
        return "medium"
    return "high"
