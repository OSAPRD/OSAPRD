"""Shared helpers for language, popularity, and domain characteristics analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from balance_statistics_utility import (
    add_cliffs_delta_ci,
    apply_fdr_correction,
    mann_whitney_u_test,
    numeric_distribution_summary,
)
from plotting_utility import order_humans_first
from topic_groups_utility import (
    TopicGroupRecord,
    load_topic_group_records,
    resolve_topic_output_dir,
)


CHARACTERISTIC_LANGUAGES = ("python", "javascript", "java", "c++")
CHARACTERISTIC_LANGUAGE_LABELS = {
    "python": "Python",
    "javascript": "JavaScript",
    "java": "Java",
    "c++": "C++",
}
POPULARITY_BUCKET_LABELS = {
    "pop0": "low",
    "pop1": "medium",
    "pop2": "high",
}
POPULARITY_BUCKET_ORDER = ("low", "medium", "high")
CHARACTERISTIC_DOMAINS = (
    "AI, Data, and Science",
    "Backend, APIs, and Security",
    "Distributed and Embedded Systems",
    "Graphics",
    "Web and Mobile",
)
AUTHORSHIP_GROUPS = ("human", "agent")


def characteristics_output_dir(
    analysis_output_dir: Path | str,
    analysis_kind: str,
) -> Path:
    """Return the output directory for one characteristics analysis family."""
    return Path(analysis_output_dir) / "characteristics" / analysis_kind


def characteristics_output_path(
    analysis_output_dir: Path | str,
    analysis_kind: str,
    filename: str,
) -> Path:
    """Return a concrete JSON output path for characteristics results."""
    return characteristics_output_dir(analysis_output_dir, analysis_kind) / filename


def write_characteristics_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a stable, sorted characteristics JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_characteristics_topic_groups(
    topic_classification_output_dir: Path | str | None,
) -> tuple[list[TopicGroupRecord], bool]:
    """Load topic groups only when a domain folder is explicitly provided."""
    if topic_classification_output_dir is None:
        return [], False
    raw_text = str(topic_classification_output_dir).strip()
    if not raw_text:
        return [], False
    resolved = resolve_topic_output_dir(Path(raw_text))
    if resolved is None:
        return [], False
    allowed = {domain.casefold(): domain for domain in CHARACTERISTIC_DOMAINS}
    filtered: list[TopicGroupRecord] = []
    for record in load_topic_group_records(resolved):
        domain = allowed.get(str(record.topic_group).strip().casefold())
        if domain is None:
            continue
        filtered.append(
            TopicGroupRecord(
                repository_id=record.repository_id,
                repository_key=record.repository_key,
                topic_group=domain,
                topic=record.topic,
                confidence=record.confidence,
            )
        )
    return filtered, True


def _summary(values: list[float]) -> dict[str, Any]:
    """Return numeric summary statistics plus the raw count."""
    payload = numeric_distribution_summary(values)
    payload["n"] = len(values)
    return payload


def _comparison_payload(
    *,
    first_group: str,
    second_group: str,
    first_values: list[float],
    second_values: list[float],
) -> dict[str, Any]:
    """Build one pairwise comparison payload for characteristics JSON."""
    test = mann_whitney_u_test(first_values, second_values)
    add_cliffs_delta_ci(test, first_values, second_values)
    test["first_group"] = first_group
    test["second_group"] = second_group
    return {
        "statistics": {
            first_group: _summary(first_values),
            second_group: _summary(second_values),
        },
        "mann_whitney_u": test,
    }

def build_characteristics_results_from_payload(
    characteristic_metric_values: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    *,
    metrics: tuple[str, ...],
    include_domain: bool,
) -> dict[str, Any]:
    """Build JSON statistics for characteristics from streaming accumulator values."""
    dimensions = [
        ("language", CHARACTERISTIC_LANGUAGES),
        ("popularity", POPULARITY_BUCKET_ORDER),
    ]
    if include_domain:
        dimensions.append(("domain", CHARACTERISTIC_DOMAINS))
    payload: dict[str, Any] = {
        "metrics": list(metrics),
        "language_scope": list(CHARACTERISTIC_LANGUAGES),
        "popularity_buckets": {
            "low": "0 stars",
            "medium": "1-18 stars",
            "high": "19+ stars",
        },
        "domain_enabled": bool(include_domain),
        "dimensions": {},
    }
    if include_domain:
        payload["domain_scope"] = list(CHARACTERISTIC_DOMAINS)
    for dimension, levels in dimensions:
        level_payload: dict[str, Any] = {}
        for level in levels:
            metric_payload = {}
            for metric in metrics:
                metric_payload[metric] = _level_metric_payload_from_payload(
                    characteristic_metric_values,
                    dimension=dimension,
                    level=level,
                    metric=metric,
                )
            level_payload[level] = metric_payload
        payload["dimensions"][dimension] = level_payload
    return apply_fdr_correction(payload)


def _level_metric_payload_from_payload(
    characteristic_metric_values: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    *,
    dimension: str,
    level: str,
    metric: str,
) -> dict[str, Any]:
    """Build statistics for one characteristic level and metric."""
    scoped = (
        characteristic_metric_values
        .get(dimension, {})
        .get(level, {})
        .get(metric, {})
    )
    human_values = [float(value) for value in scoped.get("authorship:human", [])]
    agent_values = [float(value) for value in scoped.get("authorship:agent", [])]
    cohort_payload = {}
    cohorts = order_humans_first(
        scope[len("cohort:") :]
        for scope, values in scoped.items()
        if scope.startswith("cohort:")
        and scope[len("cohort:") :].strip().casefold() not in {"human", "humans"}
        and values
    )
    for cohort in cohorts:
        cohort_values = [
            float(value)
            for value in scoped.get(f"cohort:{cohort}", [])
        ]
        cohort_payload[cohort] = _comparison_payload(
            first_group=cohort,
            second_group="human",
            first_values=cohort_values,
            second_values=human_values,
        )
    return {
        "agents_vs_humans": _comparison_payload(
            first_group="agent",
            second_group="human",
            first_values=agent_values,
            second_values=human_values,
        ),
        "cohort_vs_humans": cohort_payload,
    }


def effect_size_matrix(
    results: dict[str, Any],
    *,
    dimension: str,
    metrics: tuple[str, ...],
) -> list[list[float | None]]:
    """Extract Cliff's delta values as a matrix for heatmap plotting."""
    dimension_payload = results.get("dimensions", {}).get(dimension, {})
    matrix: list[list[float | None]] = []
    for level_payload in dimension_payload.values():
        row: list[float | None] = []
        for metric in metrics:
            metric_payload = level_payload.get(metric, {})
            test = metric_payload.get("agents_vs_humans", {}).get(
                "mann_whitney_u",
                {},
            )
            value = test.get("cliffs_delta")
            row.append(None if value is None else float(value))
        matrix.append(row)
    return matrix
