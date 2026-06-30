"""Streaming dataset analysis accumulator.

The accumulator in this module is the dataset-side target shape for the
DuckDB-free migration: raw parquet rows are consumed once, compact counters and
numeric arrays are retained, and final result payloads are derived from those
small structures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from curation_parquet_utility import CohortParquetFiles
from popularity_buckets_utility import (
    build_popularity_bucket_scheme,
    popularity_bucket_for_stars,
)
from stable_deduplication_utility import stable_numeric_id
from streaming_parquet_utility import (
    StreamingPrFacts,
    coerce_bool,
    extract_common_pr_facts,
    iter_cohort_parquet_rows,
    json_mapping,
    nested_get,
)
from topic_groups_utility import (
    TOPIC_CONFIDENCE_THRESHOLD,
    TOPIC_GROUP_ORDER,
    TopicGroupRecord,
    filter_topic_group_records_by_confidence,
)


POPULARITY_BUCKET_RESULT_KEYS = {
    "pop0": "low",
    "pop1": "medium",
    "pop2": "high",
}
DATASET_TIMEPOINT_DAYS = {
    "0d": 0,
    "+3d": 3,
    "+7d": 7,
    "+31d": 31,
    "+61d": 61,
}
_COMMON_DATASET_COLUMNS = (
    "id",
    "url",
    "number",
    "created_at",
    "changed_files",
    "additions",
    "deletions",
    "authored_by_agent",
    "author_agent",
    "discovered_agent",
    "base_repository",
    "base_repository_full",
    "pr_primary_language_effective",
    "longitudinal_selected",
    "repository_metadata.id",
    "repository_metadata.name_with_owner",
    "repository_metadata.stargazer_count",
)
DATASET_STREAM_COLUMNS = _COMMON_DATASET_COLUMNS + tuple(
    column
    for label in DATASET_TIMEPOINT_DAYS
    if label != "0d"
    for column in (
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.available",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.snapshot_available",
        "metrics.refactoring_metrics.refactoring_metrics.metrics."
        f"refactor_future_snapshot_metrics.{label}.snapshot_commit",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.available",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.snapshot_available",
        "metrics.maintainability_metrics.maintainability_indicators.summary."
        f"maintainability_future_snapshot_metrics.{label}.snapshot_commit",
    )
)


@dataclass
class DatasetStreamingAccumulator:
    """Compact streaming state for dataset analysis."""

    topic_group_records: Iterable[TopicGroupRecord] = ()
    popularity_scheme: dict[str, Any] = field(
        default_factory=lambda: build_popularity_bucket_scheme([], bucket_count=3)
    )
    seen_stable_pr_keys: set[str] = field(default_factory=set)
    count_by_scope: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    longitudinal_count_by_scope: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    repository_identities_by_scope: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    language_counts_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    popularity_counts_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    domain_pr_counts_by_scope: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    domain_repo_identities_by_scope: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    creation_week_counts_by_cohort: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    creation_hour_counts: Counter[str] = field(default_factory=Counter)
    language_counts_by_cohort: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    popularity_counts_by_cohort: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    domain_pr_counts_by_cohort: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    domain_repo_identities_by_cohort: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    domain_confidence_values: dict[str, list[float]] = field(
        default_factory=lambda: {domain: [] for domain in TOPIC_GROUP_ORDER}
    )
    numeric_values_by_cohort: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    longitudinal_availability: dict[str, dict[str, Counter[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )

    def __post_init__(self) -> None:
        records = filter_topic_group_records_by_confidence(list(self.topic_group_records))
        self._topics_by_repository_id: dict[str, list[TopicGroupRecord]] = defaultdict(list)
        self._topics_by_repository_key: dict[str, list[TopicGroupRecord]] = defaultdict(list)
        self._domain_confidence_seen: set[tuple[str, str, float]] = set()
        for record in records:
            if record.repository_id:
                self._topics_by_repository_id[str(record.repository_id)].append(record)
            if record.repository_key:
                self._topics_by_repository_key[str(record.repository_key)].append(record)
            if record.confidence is not None:
                identity = record.repository_id or record.repository_key or ""
                key = (identity, record.topic_group, float(record.confidence))
                if key not in self._domain_confidence_seen:
                    self._domain_confidence_seen.add(key)
                    self.domain_confidence_values.setdefault(
                        record.topic_group,
                        [],
                    ).append(float(record.confidence))

    def add_row(
        self,
        row: Mapping[str, Any],
        *,
        cohort: str,
        source_file: Path | str,
        source_row_number: int,
        excluded_agents: tuple[str, ...] = (),
    ) -> bool:
        """Consume one raw curation row. Return whether it was retained."""
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
        self._add_retained_row(row, facts)
        return True

    def _add_retained_row(self, row: Mapping[str, Any], facts: StreamingPrFacts) -> None:
        scopes = ["overall", f"cohort:{facts.cohort}", f"group:{facts.authorship_group}"]
        repository_identity = _repository_identity(facts)
        domains = self._domains_for_pr(facts)
        bucket = popularity_bucket_for_stars(facts.stargazer_count, self.popularity_scheme)

        for scope in scopes:
            self.count_by_scope[scope] += 1
            if facts.longitudinal_selected:
                self.longitudinal_count_by_scope[scope] += 1
            if repository_identity:
                self.repository_identities_by_scope[scope].add(repository_identity)
            if facts.language:
                self.language_counts_by_scope[scope][facts.language] += 1
            self.popularity_counts_by_scope[scope][bucket] += 1
            for domain in domains:
                self.domain_pr_counts_by_scope[scope][domain] += 1
                if repository_identity:
                    self.domain_repo_identities_by_scope[scope][domain].add(
                        repository_identity
                    )

        if facts.language:
            self.language_counts_by_cohort[facts.cohort][facts.language] += 1
        self.popularity_counts_by_cohort[facts.cohort][bucket] += 1
        for domain in domains:
            self.domain_pr_counts_by_cohort[facts.cohort][domain] += 1
            if repository_identity:
                self.domain_repo_identities_by_cohort[facts.cohort][domain].add(
                    repository_identity
                )
        self._add_creation_time(facts)
        self._add_numeric_value(facts.cohort, "stargazer_count", facts.stargazer_count)
        self._add_numeric_value(
            facts.cohort,
            "changed_files_count",
            facts.changed_files_count,
        )
        self._add_numeric_value(
            facts.cohort,
            "changed_line_count",
            facts.changed_line_count,
        )
        if facts.longitudinal_selected:
            self._add_longitudinal_availability(row, facts)

    def _domains_for_pr(self, facts: StreamingPrFacts) -> list[str]:
        records: list[TopicGroupRecord] = []
        if facts.base_repository_id:
            records.extend(self._topics_by_repository_id.get(facts.base_repository_id, []))
        if facts.repository_key:
            records.extend(self._topics_by_repository_key.get(facts.repository_key, []))
        seen: set[str] = set()
        domains: list[str] = []
        for record in records:
            if record.topic_group in seen:
                continue
            seen.add(record.topic_group)
            domains.append(record.topic_group)
        return domains

    def _add_creation_time(self, facts: StreamingPrFacts) -> None:
        parsed = _parse_datetime(facts.created_at)
        if parsed is None:
            return
        week_start = _week_start(parsed).date().isoformat()
        self.creation_week_counts_by_cohort[facts.cohort][week_start] += 1
        self.creation_hour_counts[str(parsed.hour)] += 1

    def _add_numeric_value(self, cohort: str, metric: str, value: Any) -> None:
        if value is None:
            return
        self.numeric_values_by_cohort[metric][cohort].append(float(value))

    def _add_longitudinal_availability(
        self,
        row: Mapping[str, Any],
        facts: StreamingPrFacts,
    ) -> None:
        refactor_future = _future_snapshot_mapping(
            row,
            (
                "metrics",
                "refactoring_metrics",
                "refactoring_metrics",
                "metrics",
                "refactor_future_snapshot_metrics",
            ),
        )
        maintainability_future = _future_snapshot_mapping(
            row,
            (
                "metrics",
                "maintainability_metrics",
                "maintainability_indicators",
                "summary",
                "maintainability_future_snapshot_metrics",
            ),
        )
        for label in DATASET_TIMEPOINT_DAYS:
            total_key = f"{label}:total"
            available_key = f"{label}:available"
            self.longitudinal_availability[facts.cohort][label][total_key] += 1
            if label == "0d" or _snapshot_available(
                refactor_future.get(label),
            ) or _snapshot_available(maintainability_future.get(label)):
                self.longitudinal_availability[facts.cohort][label][available_key] += 1

    def result_payload(self) -> dict[str, Any]:
        """Return the public dataset result JSON payload."""
        per_cohort = {
            cohort_scope[len("cohort:") :]: self._summary_for_scope(cohort_scope)
            for cohort_scope in sorted(
                scope for scope in self.count_by_scope if scope.startswith("cohort:")
            )
        }
        return {
            "overall": self._summary_for_scope("overall"),
            "all_agents_vs_humans": {
                "humans": self._summary_for_scope("group:human"),
                "all_agents": self._summary_for_scope("group:agent"),
            },
            "per_cohort": dict(
                sorted(
                    per_cohort.items(),
                    key=lambda item: (
                        0 if item[0].casefold() in {"human", "humans"} else 1,
                        item[0],
                    ),
                )
            ),
        }

    def plot_payload(self) -> dict[str, Any]:
        """Return compact plot inputs for a future payload-based plotter."""
        cohorts = [
            scope[len("cohort:") :]
            for scope in sorted(
                scope for scope in self.count_by_scope if scope.startswith("cohort:")
            )
        ]
        return {
            "cohorts": cohorts,
            "creation_week_counts_by_cohort": _counter_tree_to_dict(
                self.creation_week_counts_by_cohort
            ),
            "creation_hour_counts": dict(self.creation_hour_counts),
            "language_counts_by_cohort": _counter_tree_to_dict(
                self.language_counts_by_cohort
            ),
            "popularity_counts_by_cohort": _counter_tree_to_dict(
                self.popularity_counts_by_cohort
            ),
            "domain_pr_counts_by_cohort": _counter_tree_to_dict(
                self.domain_pr_counts_by_cohort
            ),
            "domain_repo_counts_by_cohort": {
                cohort: {
                    domain: len(repository_identities)
                    for domain, repository_identities in domain_map.items()
                }
                for cohort, domain_map in self.domain_repo_identities_by_cohort.items()
            },
            "domain_confidence_values": {
                domain: values
                for domain, values in self.domain_confidence_values.items()
            },
            "numeric_values_by_cohort": {
                metric: {
                    cohort: values
                    for cohort, values in by_cohort.items()
                }
                for metric, by_cohort in self.numeric_values_by_cohort.items()
            },
            "longitudinal_availability": {
                cohort: {
                    label: dict(counter)
                    for label, counter in labels.items()
                }
                for cohort, labels in self.longitudinal_availability.items()
            },
        }

    def _summary_for_scope(self, scope: str) -> dict[str, Any]:
        return {
            "pull_request_count": int(self.count_by_scope.get(scope, 0)),
            "longitudinal_pull_request_count": int(
                self.longitudinal_count_by_scope.get(scope, 0)
            ),
            "unique_repository_count": len(
                self.repository_identities_by_scope.get(scope, set())
            ),
            "pull_requests_per_language": dict(
                self.language_counts_by_scope.get(scope, Counter())
            ),
            "pull_requests_per_repository_popularity": {
                POPULARITY_BUCKET_RESULT_KEYS.get(label, label): int(
                    self.popularity_counts_by_scope.get(scope, Counter()).get(label, 0)
                )
                for label in self.popularity_scheme.get("bucket_labels", [])
            },
            "pull_requests_per_domain": _domain_counts_payload(
                self.domain_pr_counts_by_scope.get(scope, Counter()),
                count_key="pull_request_count",
            ),
            "repositories_per_domain": _domain_counts_payload(
                {
                    domain: len(repository_identities)
                    for domain, repository_identities in self.domain_repo_identities_by_scope
                    .get(scope, {})
                    .items()
                },
                count_key="repository_count",
            ),
        }


def stream_dataset_analysis(
    cohort_inputs: list[CohortParquetFiles],
    *,
    topic_group_records: list[TopicGroupRecord],
    excluded_agents: tuple[str, ...],
    progress_logger: Callable[..., Any] | None = None,
    batch_size: int = 512,
) -> DatasetStreamingAccumulator:
    """Stream raw parquet rows into a dataset accumulator."""
    accumulator = DatasetStreamingAccumulator(topic_group_records=topic_group_records)
    for cohort, source_file, source_row_number, row in iter_cohort_parquet_rows(
        cohort_inputs,
        columns=DATASET_STREAM_COLUMNS,
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


def _domain_counts_payload(
    counts: Mapping[str, int],
    *,
    count_key: str,
) -> dict[str, dict[str, float | int]]:
    """Return ordered domain counts plus within-domain percentages."""
    total = sum(int(counts.get(domain, 0)) for domain in TOPIC_GROUP_ORDER)
    return {
        domain: {
            count_key: int(counts.get(domain, 0)),
            "percentage": (
                100.0 * int(counts.get(domain, 0)) / total if total else 0.0
            ),
        }
        for domain in TOPIC_GROUP_ORDER
    }


def _is_excluded_agent(
    facts: StreamingPrFacts,
    excluded_agents: tuple[str, ...],
) -> bool:
    """Return whether a retained row belongs to an omitted agent cohort."""
    if facts.authorship_group != "agent":
        return False
    excluded = {agent.strip().casefold() for agent in excluded_agents if agent.strip()}
    if not excluded:
        return False
    return (
        (facts.agent_label or "").strip().casefold() in excluded
        or facts.cohort.strip().casefold() in excluded
    )


def _repository_identity(facts: StreamingPrFacts) -> str | None:
    """Return a stable repository key for repository-level de-duplication."""
    repository_id = stable_numeric_id(facts.base_repository_id)
    if repository_id:
        return f"repository-id:{repository_id}"
    if facts.repository_key:
        return f"repository-key:{facts.repository_key}"
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse GitHub timestamp text into a naive UTC datetime."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _week_start(value: datetime) -> datetime:
    """Return Monday midnight for weekly creation-time aggregation."""
    midnight = datetime(value.year, value.month, value.day)
    return midnight - timedelta(days=midnight.weekday())


def _future_snapshot_mapping(
    row: Mapping[str, Any],
    path: tuple[str, ...],
) -> dict[str, Any]:
    """Read future snapshot metrics from nested or flattened parquet fields."""
    direct = nested_get(row, *path)
    parsed = json_mapping(direct)
    if parsed:
        return parsed
    # Some cache/resume paths flatten future metrics as JSON text columns.
    return json_mapping(row.get(path[-1]))


def _snapshot_available(payload: Any) -> bool:
    """Return whether a future snapshot payload contains usable evidence."""
    parsed = json_mapping(payload)
    if not parsed and isinstance(payload, Mapping):
        parsed = dict(payload)
    return bool(
        coerce_bool(parsed.get("available"), default=False)
        or coerce_bool(parsed.get("snapshot_available"), default=False)
        or str(parsed.get("snapshot_commit") or "").strip()
    )


def _counter_tree_to_dict(
    tree: Mapping[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    """Convert nested counters into JSON-serializable dictionaries."""
    return {
        str(key): {str(inner_key): int(value) for inner_key, value in counter.items()}
        for key, counter in tree.items()
    }
