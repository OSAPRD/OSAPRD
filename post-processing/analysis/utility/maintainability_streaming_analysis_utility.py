"""Streaming maintainability analysis accumulator."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from curation_parquet_utility import CohortParquetFiles
from longitudinal_analysis_utility import NONNEGATIVE_LONGITUDINAL_METRICS
from maintainability_analysis_utility import (
    MAINTAINABILITY_LONGITUDINAL_TIMEPOINT_DAYS,
    MAINTAINABILITY_SUCCESS_STATUS,
    MANTYLA_CATEGORIES,
    MANTYLA_COUNT_SOURCE_STORED,
    MANTYLA_COUNT_SOURCE_TAXONOMY,
    MANTYLA_TAXONOMY_CATEGORIES,
    _load_code_smell_taxonomy_classifier,
    _taxonomy_mantyla_category,
    normalize_maintainability_tool_status,
)
from maintainability_multimetrics_utility import RAW_MULTIMETRIC_METRICS
from refactoring_analysis_utility import _named_refactoring_type_counts
from characteristics_analysis_utility import CHARACTERISTIC_DOMAINS
from streaming_parquet_utility import (
    StreamingPrFacts,
    coerce_float,
    coerce_int,
    extract_common_pr_facts,
    iter_cohort_parquet_rows,
    json_mapping,
    nested_get,
)
from topic_groups_utility import TopicGroupRecord, filter_topic_group_records_by_confidence


STANDARDIZED_SMELL_UNCLASSIFIED = "unclassified"
_COMMON_MAINTAINABILITY_COLUMNS = (
    "id",
    "url",
    "number",
    "authored_by_agent",
    "author_agent",
    "discovered_agent",
    "base_repository",
    "base_repository_full",
    "pr_primary_language_effective",
    "repository_metadata.id",
    "repository_metadata.name_with_owner",
    "repository_metadata.stargazer_count",
    "metrics.refactoring_metrics.refactoring_metrics.metrics.refactor_type_count",
    "metrics.maintainability_metrics.status",
    "metrics.maintainability_metrics.maintainability_indicators.status",
    "metrics.maintainability_metrics.maintainability_indicators.summary.smells_diversity_pre",
    "metrics.maintainability_metrics.maintainability_indicators.summary.smells_diversity_post",
    "metrics.maintainability_metrics.maintainability_indicators.summary.smells_by_mantyla_post",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures",
    "metrics.maintainability_metrics.maintainability_indicators.summary.multimetric_snapshot_rows",
    "metrics.maintainability_metrics.maintainability_indicators.summary.multimetric_snapshot_rows_json",
    "metrics.maintainability_metrics.maintainability_indicators.summary.maintainability_future_snapshot_metrics",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.before.ncloc",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.before.complexity",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.before.halstead_volume",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.before.maintainability_index",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.before.duplicated_lines_density",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.before.comment_lines_density",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.after.ncloc",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.after.complexity",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.after.halstead_volume",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.after.maintainability_index",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.after.duplicated_lines_density",
    "metrics.maintainability_metrics.maintainability_indicators.summary.snapshot_measures.after.comment_lines_density",
)
MAINTAINABILITY_STREAM_COLUMNS = _COMMON_MAINTAINABILITY_COLUMNS + tuple(
    column
    for label in MAINTAINABILITY_LONGITUDINAL_TIMEPOINT_DAYS
    if label != "0d"
    for column in (
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.smell_count",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.summary.smell_count",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.measures.code_smells",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.measures.ncloc",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.measures.complexity",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.measures.halstead_volume",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.measures.maintainability_index",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.measures.duplicated_lines_density",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.measures.comment_lines_density",
    )
)


@dataclass
class MaintainabilityPrMetricRow:
    """Compact successful maintainability PR metrics."""

    analysis_row_id: str
    cohort: str
    authorship_group: str
    pr_url: str | None
    pr_number: str | None
    language: str | None
    base_repository_id: str | None
    repository_key: str | None
    stargazer_count: int | None
    metrics: dict[str, float | None]
    embedded_multimetric_snapshot_rows: list[dict[str, Any]]


@dataclass
class MaintainabilityStreamingAccumulator:
    """Compact streaming state for maintainability analysis."""

    mantyla_count_source: str = MANTYLA_COUNT_SOURCE_TAXONOMY
    require_refops: bool = False
    topic_group_records: Iterable[TopicGroupRecord] = ()
    seen_stable_pr_keys: set[str] = field(default_factory=set)
    all_pr_count: int = 0
    failed_pr_count: int = 0
    successful_prs: list[MaintainabilityPrMetricRow] = field(default_factory=list)
    smell_type_counts_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    smell_type_counts_by_language_scope: dict[str, dict[str, Counter[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    mantyla_counts_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    prs_with_smell_by_scope: Counter[str] = field(default_factory=Counter)
    prs_with_mantyla_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    per_pr_mantyla_counts_by_scope: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    metric_values_by_scope: dict[str, dict[str, list[float]]] = field(
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
            _load_code_smell_taxonomy_classifier()
            if self.mantyla_count_source == MANTYLA_COUNT_SOURCE_TAXONOMY
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
        if self.require_refops and _refop_count_for_filter(row) <= 0:
            return False
        self.all_pr_count += 1
        maintainability_root = nested_get(row, "metrics", "maintainability_metrics")
        status = normalize_maintainability_tool_status(
            nested_get(maintainability_root, "status"),
            nested_get(maintainability_root, "maintainability_indicators", "status"),
        )
        if status == "failed":
            self.failed_pr_count += 1
        if status != MAINTAINABILITY_SUCCESS_STATUS:
            return True
        self._add_success_row(row, facts)
        return True

    def _add_success_row(self, row: Mapping[str, Any], facts: StreamingPrFacts) -> None:
        summary = json_mapping(
            nested_get(
                row,
                "metrics",
                "maintainability_metrics",
                "maintainability_indicators",
                "summary",
            )
        )
        snapshot_measures = json_mapping(summary.get("snapshot_measures"))
        before = json_mapping(snapshot_measures.get("before"))
        after = json_mapping(snapshot_measures.get("after"))
        smell_counts_pre = _named_smell_counts(summary.get("smells_diversity_pre"))
        smell_counts_post = _named_smell_counts(summary.get("smells_diversity_post"))
        smell_count_pre = sum(smell_counts_pre.values())
        smell_count_post = sum(smell_counts_post.values())
        nloc_before = coerce_float(before.get("ncloc"))
        nloc_after = coerce_float(after.get("ncloc"))
        smell_density_before = _per_kloc(smell_count_pre, nloc_before)
        smell_density_post = _per_kloc(smell_count_post, nloc_after)
        cc_before = coerce_float(before.get("complexity"))
        cc_after = coerce_float(after.get("complexity"))
        hv_before = coerce_float(before.get("halstead_volume"))
        hv_after = coerce_float(after.get("halstead_volume"))
        metrics = {
            "SmellCount": float(smell_count_post),
            "SmellDensity": smell_density_post,
            "SmellsDelta": (
                float(smell_count_post - smell_count_pre)
                if summary.get("smells_diversity_pre") is not None
                else None
            ),
            "MI": _delta(
                coerce_float(before.get("maintainability_index")),
                coerce_float(after.get("maintainability_index")),
            ),
            "CC": _delta(cc_before, cc_after),
            "HV": _delta(hv_before, hv_after),
            "CCDensity": _delta(_per_kloc(cc_before, nloc_before), _per_kloc(cc_after, nloc_after)),
            "HVDensity": _delta(_per_kloc(hv_before, nloc_before), _per_kloc(hv_after, nloc_after)),
            "DuplicationDensity": _delta(
                coerce_float(before.get("duplicated_lines_density")),
                coerce_float(after.get("duplicated_lines_density")),
            ),
            "DuplicationDensity_Before": coerce_float(
                before.get("duplicated_lines_density")
            ),
            "DuplicationDensity_Post": coerce_float(
                after.get("duplicated_lines_density")
            ),
            "CommentDensity": _delta(
                coerce_float(before.get("comment_lines_density")),
                coerce_float(after.get("comment_lines_density")),
            ),
            "NLOC": _delta(nloc_before, nloc_after),
            "CodeSmellDensityDelta": _delta(smell_density_before, smell_density_post),
            "CodeSmellDensity_Before": smell_density_before,
            "CodeSmellDensity_Post": smell_density_post,
            "NLOC_Before": nloc_before,
            "NLOC_Post": nloc_after,
            "KLOC_Before": _kloc(nloc_before),
            "KLOC_Post": _kloc(nloc_after),
        }
        self._analysis_row_counter += 1
        metric_row = MaintainabilityPrMetricRow(
            analysis_row_id=str(self._analysis_row_counter),
            cohort=facts.cohort,
            authorship_group=facts.authorship_group,
            pr_url=facts.pr_url,
            pr_number=facts.pr_number,
            language=facts.language,
            base_repository_id=facts.base_repository_id,
            repository_key=facts.repository_key,
            stargazer_count=facts.stargazer_count,
            metrics=metrics,
            embedded_multimetric_snapshot_rows=_embedded_multimetric_snapshot_rows(
                summary,
                facts,
            ),
        )
        self.successful_prs.append(metric_row)
        scopes = _scopes_for_row(facts)
        if smell_count_post > 0:
            for scope in scopes:
                self.prs_with_smell_by_scope[scope] += 1
        for smell_type, count in smell_counts_post.items():
            for scope in scopes:
                self.smell_type_counts_by_scope[scope][smell_type] += int(count)
                if facts.language:
                    self.smell_type_counts_by_language_scope[scope][facts.language][
                        smell_type
                    ] += int(count)
        mantyla_counts = self._mantyla_counts(summary, smell_counts_post)
        for scope in scopes:
            for category in MANTYLA_CATEGORIES:
                self.per_pr_mantyla_counts_by_scope[scope][category].append(
                    float(mantyla_counts.get(category, 0))
                )
        for category, count in mantyla_counts.items():
            if count <= 0:
                continue
            for scope in scopes:
                self.mantyla_counts_by_scope[scope][category] += int(count)
                self.prs_with_mantyla_by_scope[scope][category] += 1
        for scope in scopes:
            for metric, value in metrics.items():
                if value is not None:
                    self.metric_values_by_scope[metric][scope].append(float(value))
        self._add_characteristic_values(metric_row, int(smell_count_post))
        self._add_longitudinal_values(summary, facts, metric_row.metrics)

    def _mantyla_counts(
        self,
        summary: Mapping[str, Any],
        smell_counts_post: Mapping[str, int],
    ) -> Counter[str]:
        counts: Counter[str] = Counter()
        if self.mantyla_count_source == MANTYLA_COUNT_SOURCE_STORED:
            for category, count in _positive_int_counts(summary.get("smells_by_mantyla_post")).items():
                if category in MANTYLA_CATEGORIES:
                    counts[category] += int(count)
            return counts
        classifier = self._taxonomy_classifier
        if classifier is None:
            return counts
        for smell_type, count in smell_counts_post.items():
            category = _taxonomy_mantyla_category(smell_type, classifier)
            if category is not None and count > 0:
                counts[category] += int(count)
        return counts

    def _add_characteristic_values(
        self,
        metric_row: MaintainabilityPrMetricRow,
        item_count: int,
    ) -> None:
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
                for metric, value in metric_row.metrics.items():
                    if value is None:
                        continue
                    value_float = float(value)
                    scoped = self.characteristic_metric_values[dimension][level][metric]
                    scoped[f"authorship:{metric_row.authorship_group}"].append(value_float)
                    scoped[f"cohort:{metric_row.cohort}"].append(value_float)

    def _add_longitudinal_values(
        self,
        summary: Mapping[str, Any],
        facts: StreamingPrFacts,
        current_metrics: Mapping[str, Any],
    ) -> None:
        future = json_mapping(summary.get("maintainability_future_snapshot_metrics"))
        snapshot_measures = json_mapping(summary.get("snapshot_measures"))
        after = json_mapping(snapshot_measures.get("after"))
        nloc_after = coerce_float(after.get("ncloc"))
        cc_after = coerce_float(after.get("complexity"))
        hv_after = coerce_float(after.get("halstead_volume"))
        post_values = {
            "SmellCount": current_metrics.get("SmellCount"),
            "CodeSmellDensity": current_metrics.get("SmellDensity"),
            "CC": cc_after,
            "HV": hv_after,
            "CCDensity": _per_kloc(cc_after, nloc_after),
            "HVDensity": _per_kloc(hv_after, nloc_after),
            "MI": coerce_float(after.get("maintainability_index")),
            "DuplicationDensity": coerce_float(after.get("duplicated_lines_density")),
            "CommentDensity": coerce_float(after.get("comment_lines_density")),
            "NLOC": nloc_after,
            "KLOC": _kloc(nloc_after),
        }
        for metric, value in post_values.items():
            self._append_longitudinal(facts.cohort, "0d", metric, value)
        for label, days in MAINTAINABILITY_LONGITUDINAL_TIMEPOINT_DAYS.items():
            if label == "0d":
                continue
            payload = json_mapping(future.get(label))
            if not payload:
                continue
            measures = json_mapping(payload.get("measures"))
            nloc = _future_measure_value(payload, measures, "ncloc", field="loc")
            smell_count = _first_float(
                payload.get("smell_count"),
                nested_get(payload, "summary", "smell_count"),
                measures.get("code_smells"),
            )
            self._append_longitudinal(facts.cohort, label, "SmellCount", smell_count)
            self._append_longitudinal(
                facts.cohort,
                label,
                "CodeSmellDensity",
                _per_kloc(smell_count, nloc),
            )
            self._append_longitudinal(
                facts.cohort,
                label,
                "CC",
                _future_measure_value(
                    payload,
                    measures,
                    "complexity",
                    field="cyclomatic_complexity",
                ),
            )
            self._append_longitudinal(
                facts.cohort,
                label,
                "HV",
                _future_measure_value(payload, measures, "halstead_volume"),
            )
            self._append_longitudinal(
                facts.cohort,
                label,
                "CCDensity",
                _per_kloc(
                    _future_measure_value(
                        payload,
                        measures,
                        "complexity",
                        field="cyclomatic_complexity",
                    ),
                    nloc,
                ),
            )
            self._append_longitudinal(
                facts.cohort,
                label,
                "HVDensity",
                _per_kloc(
                    _future_measure_value(payload, measures, "halstead_volume"),
                    nloc,
                ),
            )
            self._append_longitudinal(
                facts.cohort,
                label,
                "MI",
                _future_measure_value(payload, measures, "maintainability_index"),
            )
            self._append_longitudinal(
                facts.cohort,
                label,
                "DuplicationDensity",
                _future_measure_value(payload, measures, "duplicated_lines_density"),
            )
            self._append_longitudinal(
                facts.cohort,
                label,
                "CommentDensity",
                _future_measure_value(
                    payload,
                    measures,
                    "comment_lines_density",
                    field="comment_ratio",
                ),
            )
            self._append_longitudinal(facts.cohort, label, "NLOC", nloc)
            self._append_longitudinal(facts.cohort, label, "KLOC", _kloc(nloc))

    def _append_longitudinal(
        self,
        cohort: str,
        label: str,
        metric: str,
        value: float | None,
    ) -> None:
        if value is not None:
            value_float = float(value)
            if metric in NONNEGATIVE_LONGITUDINAL_METRICS and value_float < 0.0:
                return
            self.longitudinal_values[cohort][label][metric].append(value_float)

    def compact_payload(self) -> dict[str, Any]:
        return {
            "all_pr_count": self.all_pr_count,
            "failed_pr_count": self.failed_pr_count,
            "success_pr_count": len(self.successful_prs),
            "multimetric_pr_index": [
                {
                    "analysis_row_id": row.analysis_row_id,
                    "cohort": row.cohort,
                    "authorship_group": row.authorship_group,
                    "pr_url": row.pr_url,
                    "repository_key": row.repository_key,
                    "pr_number": row.pr_number,
                    "language": row.language,
                    "base_repository_id": row.base_repository_id,
                    "stargazer_count": row.stargazer_count,
                    "metrics": {
                        "DuplicationDensity_Before": row.metrics.get(
                            "DuplicationDensity_Before"
                        ),
                        "DuplicationDensity_Post": row.metrics.get(
                            "DuplicationDensity_Post"
                        ),
                    },
                    "characteristic_levels": _characteristic_levels_for_row(
                        row.language,
                        row.base_repository_id,
                        row.repository_key,
                        row.stargazer_count,
                        self._topics_by_repository_id,
                        self._topics_by_repository_key,
                    ),
                }
                for row in self.successful_prs
            ],
            "embedded_multimetric_snapshot_rows": [
                snapshot_row
                for row in self.successful_prs
                for snapshot_row in row.embedded_multimetric_snapshot_rows
            ],
            "smell_type_counts_by_scope": _counter_tree_to_dict(
                self.smell_type_counts_by_scope
            ),
            "smell_type_counts_by_language_scope": {
                scope: {
                    language: dict(counter)
                    for language, counter in by_language.items()
                }
                for scope, by_language in self.smell_type_counts_by_language_scope.items()
            },
            "mantyla_counts_by_scope": _counter_tree_to_dict(
                self.mantyla_counts_by_scope
            ),
            "prs_with_smell_by_scope": dict(self.prs_with_smell_by_scope),
            "prs_with_mantyla_by_scope": {
                scope: dict(counter)
                for scope, counter in self.prs_with_mantyla_by_scope.items()
            },
            "per_pr_mantyla_counts_by_scope": {
                scope: {
                    category: values
                    for category, values in by_category.items()
                }
                for scope, by_category in self.per_pr_mantyla_counts_by_scope.items()
            },
            "metric_values_by_scope": {
                metric: {
                    scope: values
                    for scope, values in by_scope.items()
                }
                for metric, by_scope in self.metric_values_by_scope.items()
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
        """Return the public maintainability result JSON payload."""
        cohorts = sorted(
            {
                scope[len("cohort:") :]
                for scope in self.metric_values_by_scope.get("SmellCount", {})
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
        eligible_pr_count = self._eligible_count(scope)
        return {
            "prs_with_standardized_code_smell_after_merge": _count_payload(
                self.prs_with_smell_by_scope.get(scope, 0),
                eligible_pr_count,
            ),
            "prs_with_mantyla_category_code_smell": {
                category: _count_payload(
                    self.prs_with_mantyla_by_scope.get(scope, Counter()).get(
                        category,
                        0,
                    ),
                    eligible_pr_count,
                )
                for category in MANTYLA_TAXONOMY_CATEGORIES
            },
            "top_standardized_code_smells_per_language": {
                language: _top_smells(counter)
                for language, counter in sorted(
                    self.smell_type_counts_by_language_scope.get(scope, {}).items()
                )
            },
        }

    def _eligible_count(self, scope: str) -> int:
        return len(self.metric_values_by_scope.get("SmellCount", {}).get(scope, []))


def stream_maintainability_analysis(
    cohort_inputs: list[CohortParquetFiles],
    *,
    excluded_agents: tuple[str, ...],
    mantyla_count_source: str,
    require_refops: bool = False,
    topic_group_records: list[TopicGroupRecord] | None = None,
    progress_logger: Callable[..., Any] | None = None,
    batch_size: int = 256,
) -> MaintainabilityStreamingAccumulator:
    """Stream curated PR parquet rows into a maintainability accumulator."""
    accumulator = MaintainabilityStreamingAccumulator(
        mantyla_count_source=mantyla_count_source,
        require_refops=require_refops,
        topic_group_records=topic_group_records or [],
    )
    for cohort, source_file, source_row_number, row in iter_cohort_parquet_rows(
        cohort_inputs,
        columns=MAINTAINABILITY_STREAM_COLUMNS,
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


def _top_smells(
    counter: Counter[str],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the most frequent standardized smells for a scope."""
    total = sum(counter.values())
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))
    return [
        {
            "rank": index,
            "code_smell_type": str(label),
            "smell_count": int(count),
            "percentage_within_group": _safe_percentage(count, total),
        }
        for index, (label, count) in enumerate(ordered[:limit], start=1)
    ]


def _named_smell_counts(value: Any) -> dict[str, int]:
    """Drop unclassified smell counts from a positive-count mapping."""
    return {
        smell_type: count
        for smell_type, count in _positive_int_counts(value).items()
        if smell_type.strip().casefold() != STANDARDIZED_SMELL_UNCLASSIFIED
    }


def _positive_int_counts(value: Any) -> dict[str, int]:
    """Parse a JSON-like mapping into positive integer counts."""
    raw = json_mapping(value)
    counts: dict[str, int] = {}
    for label, count in raw.items():
        parsed_count = coerce_int(count)
        if parsed_count is not None and parsed_count > 0:
            counts[str(label)] = int(parsed_count)
    return counts


def _refop_count_for_filter(row: Mapping[str, Any]) -> int:
    """Return standardized refactoring-operation count used by refop filters."""
    type_counts = json_mapping(
        nested_get(
            row,
            "metrics",
            "refactoring_metrics",
            "refactoring_metrics",
            "metrics",
            "refactor_type_count",
        )
    )
    return sum(_named_refactoring_type_counts(_positive_int_counts(type_counts)).values())


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


def _delta(before: float | None, after: float | None) -> float | None:
    """Return after-before delta when both endpoints are present."""
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _first_float(*values: Any) -> float | None:
    """Return the first value that can be coerced to a float."""
    for value in values:
        parsed = coerce_float(value)
        if parsed is not None:
            return parsed
    return None


def _future_measure_value(
    payload: Mapping[str, Any],
    measures: Mapping[str, Any],
    measure_key: str,
    *,
    field: str | None = None,
) -> float | None:
    """Return a future snapshot value from old ``measures`` or new delta rows."""
    direct = coerce_float(measures.get(measure_key))
    if direct is not None:
        return direct
    metrics = json_mapping(payload.get("metrics"))
    metric_payload = json_mapping(metrics.get(field or measure_key))
    return coerce_float(metric_payload.get("future"))


def _embedded_multimetric_snapshot_rows(
    summary: Mapping[str, Any],
    facts: StreamingPrFacts,
) -> list[dict[str, Any]]:
    """Return Multimetric rows embedded in a curation parquet row.

    Current curation writes raw Multimetric/custom-duplication rows in
    ``multimetric_snapshot_rows``. Some older parquet exports retain only
    ``snapshot_measures`` and future delta summaries, so those are converted
    into minimal rows as a best-effort fallback.
    """
    rows = _json_list(summary.get("multimetric_snapshot_rows"))
    if not rows:
        rows = _json_list(summary.get("multimetric_snapshot_rows_json"))
    if rows:
        return [_normalize_embedded_multimetric_row(row, facts) for row in rows]
    return _synthetic_multimetric_snapshot_rows(summary, facts)


def _json_list(value: Any) -> list[dict[str, Any]]:
    """Return a list of dicts from list-like or JSON-string values."""
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [dict(item) for item in parsed if isinstance(item, Mapping)]


def _normalize_embedded_multimetric_row(
    row: Mapping[str, Any],
    facts: StreamingPrFacts,
) -> dict[str, Any]:
    """Attach PR identity fields needed by the Multimetric stream matcher."""
    normalized = dict(row)
    normalized.setdefault("cohort", facts.cohort)
    normalized.setdefault("pr_url", facts.pr_url)
    normalized.setdefault("repository_key", facts.repository_key)
    normalized.setdefault("pr_number", facts.pr_number)
    label = str(normalized.get("snapshot_label") or "").strip()
    if label and not normalized.get("snapshot_kind"):
        normalized["snapshot_kind"] = "future" if label.startswith("+") else label
    if not normalized.get("status") and any(
        coerce_float(normalized.get(metric)) is not None
        for metric in RAW_MULTIMETRIC_METRICS
    ):
        normalized["status"] = "success"
    return normalized


def _synthetic_multimetric_snapshot_rows(
    summary: Mapping[str, Any],
    facts: StreamingPrFacts,
) -> list[dict[str, Any]]:
    """Build minimal Multimetric-like rows from snapshot measures."""
    rows: list[dict[str, Any]] = []
    snapshot_measures = json_mapping(summary.get("snapshot_measures"))
    for label in ("before", "after"):
        measures = json_mapping(snapshot_measures.get(label))
        row = _synthetic_multimetric_row_from_measures(
            label=label,
            measures=measures,
            facts=facts,
        )
        if row is not None:
            rows.append(row)

    future = json_mapping(summary.get("maintainability_future_snapshot_metrics"))
    for label in MAINTAINABILITY_LONGITUDINAL_TIMEPOINT_DAYS:
        if label == "0d":
            continue
        payload = json_mapping(future.get(label))
        if not payload:
            continue
        measures = json_mapping(payload.get("measures"))
        if not measures:
            metrics = json_mapping(payload.get("metrics"))
            measures = {
                "loc": _future_metric_value(metrics, "loc"),
                "comment_ratio": _future_metric_value(metrics, "comment_ratio"),
                "cyclomatic_complexity": _future_metric_value(
                    metrics,
                    "cyclomatic_complexity",
                ),
                "halstead_volume": _future_metric_value(metrics, "halstead_volume"),
                "maintainability_index": _future_metric_value(
                    metrics,
                    "maintainability_index",
                ),
                "duplicated_lines_density": _future_metric_value(
                    metrics,
                    "duplicated_lines_density",
                ),
            }
        row = _synthetic_multimetric_row_from_measures(
            label=label,
            measures=measures,
            facts=facts,
        )
        if row is not None:
            rows.append(row)
    return rows


def _future_metric_value(metrics: Mapping[str, Any], field: str) -> float | None:
    """Return the future value from a curation future-metric delta item."""
    return coerce_float(json_mapping(metrics.get(field)).get("future"))


def _synthetic_multimetric_row_from_measures(
    *,
    label: str,
    measures: Mapping[str, Any],
    facts: StreamingPrFacts,
) -> dict[str, Any] | None:
    """Convert one snapshot-measures mapping into a Multimetric-like row."""
    values = {
        metric: coerce_float(measures.get(metric))
        for metric in RAW_MULTIMETRIC_METRICS
    }
    values["loc"] = _first_float(values.get("loc"), measures.get("ncloc"))
    values["comment_ratio"] = _first_float(
        values.get("comment_ratio"),
        measures.get("comment_lines_density"),
    )
    values["cyclomatic_complexity"] = _first_float(
        values.get("cyclomatic_complexity"),
        measures.get("complexity"),
    )
    if not any(value is not None for value in values.values()):
        return None
    row: dict[str, Any] = {
        "cohort": facts.cohort,
        "pr_url": facts.pr_url,
        "repository_key": facts.repository_key,
        "pr_number": facts.pr_number,
        "snapshot_label": label,
        "snapshot_kind": "future" if label.startswith("+") else label,
        "status": "success",
    }
    row.update(values)
    return row


def _kloc(nloc: float | None) -> float | None:
    """Convert non-comment lines of code to KLOC."""
    if nloc is None:
        return None
    return float(nloc) / 1000.0


def _per_kloc(value: float | int | None, nloc: float | None) -> float | None:
    """Normalize a count-like metric per thousand lines of code."""
    if value is None or nloc is None or nloc <= 0:
        return None
    return float(value) / (float(nloc) / 1000.0)


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
    domain_set = set(domain_values)
    domain_values = [domain for domain in CHARACTERISTIC_DOMAINS if domain in domain_set]
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
