"""Shared helpers for payload-based longitudinal analyses.

The refactoring and maintainability pipelines both produce compact
``cohort -> timepoint -> metric -> values`` payloads. This module owns the
common statistical summaries, attrition counts, and output path conventions so
the two longitudinal analyses stay aligned.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import stdev
from typing import Any, Iterable

from balance_statistics_utility import (
    add_cliffs_delta_ci,
    apply_fdr_correction,
    mann_whitney_u_test,
    numeric_distribution_summary,
)
from plotting_utility import order_humans_first


LONGITUDINAL_TIMEPOINTS = ("0d", "+3d", "+7d", "+31d", "+61d")
LONGITUDINAL_TIMEPOINT_DAYS = {
    "0d": 0,
    "+3d": 3,
    "+7d": 7,
    "+31d": 31,
    "+61d": 61,
}
NONNEGATIVE_LONGITUDINAL_METRICS = {
    "RefCount",
    "RefDensity",
    "RefDiversity",
    "RefMagLines",
    "RefAdded",
    "RefRemoved",
    "RefRetentionRate",
    "RefZoneFutureTouchedLines",
    "FutureTouchingCommits",
    "SmellCount",
    "CodeSmellDensity",
    "CC",
    "HV",
    "CCDensity",
    "HVDensity",
    "DuplicationDensity",
    "CommentDensity",
    "NLOC",
    "KLOC",
    "loc",
    "cyclomatic_complexity",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugprop",
    "halstead_timerequired",
    "fanout_internal",
    "fanout_external",
    "fanout_external_per_kloc",
    "operands_sum",
    "operands_uniq",
    "operators_sum",
    "operators_uniq",
    "comment_ratio",
    "pylint",
    "tiobe",
    "tiobe_complexity",
    "tiobe_duplication",
    "tiobe_functional",
    "multimetric_duplication_score",
    "original_smell_count",
    "original_code_smell_density",
    "original_duplication_density",
    "halstead_volume_per_kloc",
    "halstead_difficulty_per_kloc",
    "halstead_effort_per_kloc",
    "cyclomatic_complexity_per_kloc",
    "halstead_bugprop_per_kloc",
    "halstead_timerequired_per_kloc",
}


def parse_longitudinal_timepoint(label: str) -> int | None:
    """Return days-after-merge for a supported longitudinal label."""
    return LONGITUDINAL_TIMEPOINT_DAYS.get(str(label))


def longitudinal_output_dir(analysis_output_dir: Path | str, analysis_kind: str) -> Path:
    """Return the output directory for one longitudinal analysis kind."""
    return Path(analysis_output_dir) / "longitudinal" / analysis_kind


def longitudinal_output_path(
    analysis_output_dir: Path | str,
    analysis_kind: str,
    filename: str,
) -> Path:
    """Return a concrete output path under the longitudinal directory."""
    return longitudinal_output_dir(analysis_output_dir, analysis_kind) / filename


def write_longitudinal_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON used by downstream plotters and audits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _numeric_values(values: Iterable[Any]) -> list[float]:
    """Coerce iterable values to finite-compatible floats where possible."""
    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return numeric


def longitudinal_summary(
    values: Iterable[Any],
    *,
    metric: str | None = None,
) -> dict[str, Any]:
    """Return distribution statistics plus normal-approximation mean CI."""
    numeric = _numeric_values(values)
    payload = numeric_distribution_summary(numeric)
    n = len(numeric)
    payload["n"] = n
    if n >= 2 and payload["mean"] is not None:
        standard_error = stdev(numeric) / sqrt(n)
        margin = 1.96 * standard_error
        mean_ci95_low = float(payload["mean"] - margin)
        if metric in NONNEGATIVE_LONGITUDINAL_METRICS and mean_ci95_low < 0.0:
            mean_ci95_low = 0.0
        payload["mean_ci95_low"] = mean_ci95_low
        payload["mean_ci95_high"] = float(payload["mean"] + margin)
    else:
        payload["mean_ci95_low"] = None
        payload["mean_ci95_high"] = None
    return payload


def build_longitudinal_results_from_payload(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    """Build longitudinal results from compact streaming accumulator values."""
    metric_payload: dict[str, Any] = {}
    for metric in metrics:
        cohorts = order_humans_first(
            cohort
            for cohort, by_label in longitudinal_values.items()
            if any(by_metric.get(metric) for by_metric in by_label.values())
        )
        per_cohort = {
            cohort: _payload_timepoint_summaries(
                longitudinal_values,
                metric=metric,
                cohort=cohort,
            )
            for cohort in cohorts
        }
        metric_payload[metric] = {
            "overall": _payload_timepoint_summaries(
                longitudinal_values,
                metric=metric,
            ),
            "per_cohort": per_cohort,
            "agents_vs_humans": _payload_agents_vs_humans_by_timepoint(
                longitudinal_values,
                metric=metric,
            ),
            "plot_series": _payload_plot_series(
                longitudinal_values,
                metric=metric,
                cohorts=cohorts,
            ),
        }
    return apply_fdr_correction(
        {
            "eligible_pr_count": _payload_eligible_count(
                longitudinal_values,
                metrics=metrics,
            ),
            "timepoints": [
                {
                    "timepoint_label": label,
                    "days_after_merge": LONGITUDINAL_TIMEPOINT_DAYS[label],
                }
                for label in LONGITUDINAL_TIMEPOINTS
            ],
            "metrics": metric_payload,
            "attrition": _payload_attrition(
                longitudinal_values,
                metrics=metrics,
            ),
        }
    )


def _payload_values(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    metric: str,
    label: str,
    cohort: str | None = None,
    authorship_group: str | None = None,
) -> list[float]:
    """Select metric values for one timepoint, cohort, or authorship group."""
    values: list[float] = []
    for cohort_key, by_label in longitudinal_values.items():
        if cohort is not None and cohort_key != cohort:
            continue
        if authorship_group is not None:
            inferred_group = (
                "human"
                if cohort_key.strip().casefold() in {"human", "humans"}
                else "agent"
            )
            if inferred_group != authorship_group:
                continue
        values.extend(by_label.get(label, {}).get(metric, []))
    return _numeric_values(values)


def _payload_timepoint_summaries(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    metric: str,
    cohort: str | None = None,
) -> dict[str, Any]:
    """Summarize one metric across all configured longitudinal timepoints."""
    payload: dict[str, Any] = {}
    for label in LONGITUDINAL_TIMEPOINTS:
        summary = longitudinal_summary(
            _payload_values(
                longitudinal_values,
                metric=metric,
                label=label,
                cohort=cohort,
            ),
            metric=metric,
        )
        summary["days_after_merge"] = LONGITUDINAL_TIMEPOINT_DAYS[label]
        payload[label] = summary
    return payload


def _payload_agents_vs_humans_by_timepoint(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    metric: str,
) -> dict[str, Any]:
    """Compare agent and human cohorts at every longitudinal timepoint."""
    payload: dict[str, Any] = {}
    for label in LONGITUDINAL_TIMEPOINTS:
        agent_values = _payload_values(
            longitudinal_values,
            metric=metric,
            label=label,
            authorship_group="agent",
        )
        human_values = _payload_values(
            longitudinal_values,
            metric=metric,
            label=label,
            authorship_group="human",
        )
        test = mann_whitney_u_test(agent_values, human_values)
        add_cliffs_delta_ci(test, agent_values, human_values)
        test["first_group"] = "agent"
        test["second_group"] = "human"
        payload[label] = {
            "days_after_merge": LONGITUDINAL_TIMEPOINT_DAYS[label],
            "statistics": {
                "agent": longitudinal_summary(agent_values, metric=metric),
                "human": longitudinal_summary(human_values, metric=metric),
            },
            "mann_whitney_u": test,
        }
    return payload


def _payload_plot_series(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    metric: str,
    cohorts: Iterable[str],
) -> dict[str, list[dict[str, Any]]]:
    """Build mean/CI line-plot series for each cohort."""
    series: dict[str, list[dict[str, Any]]] = {}
    for cohort in cohorts:
        points: list[dict[str, Any]] = []
        summaries = _payload_timepoint_summaries(
            longitudinal_values,
            metric=metric,
            cohort=cohort,
        )
        for label, summary in summaries.items():
            points.append(
                {
                    "timepoint_label": label,
                    "days_after_merge": summary["days_after_merge"],
                    "mean": summary["mean"],
                    "mean_ci95_low": summary["mean_ci95_low"],
                    "mean_ci95_high": summary["mean_ci95_high"],
                    "n": summary["n"],
                }
            )
        series[cohort] = points
    return series


def _payload_eligible_count(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    metrics: tuple[str, ...],
) -> int:
    """Estimate the largest observed eligible count across metrics/timepoints."""
    best_count = 0
    for by_label in longitudinal_values.values():
        for by_metric in by_label.values():
            for metric in metrics:
                best_count = max(best_count, len(by_metric.get(metric, [])))
    return best_count


def _payload_count_map(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    metric: str,
    cohort: str | None = None,
) -> dict[str, dict[str, int | float]]:
    """Return timepoint counts and retention percentages for one metric."""
    counts = {
        label: {
            "days_after_merge": LONGITUDINAL_TIMEPOINT_DAYS[label],
            "pull_request_count": len(
                _payload_values(
                    longitudinal_values,
                    metric=metric,
                    label=label,
                    cohort=cohort,
                )
            ),
        }
        for label in LONGITUDINAL_TIMEPOINTS
    }
    baseline = int(counts["0d"]["pull_request_count"])
    for label in LONGITUDINAL_TIMEPOINTS:
        count = int(counts[label]["pull_request_count"])
        retention_proportion = float(count / baseline) if baseline else 0.0
        if label == "0d" and baseline:
            retention_proportion = 1.0
        counts[label]["baseline_pull_request_count"] = int(baseline)
        counts[label]["retention_proportion"] = retention_proportion
        counts[label]["retention_percentage"] = retention_proportion * 100.0
    return counts


def _payload_attrition(
    longitudinal_values: dict[str, dict[str, dict[str, list[float]]]],
    *,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    """Return attrition summaries overall, by cohort, and by authorship group."""
    primary_metric = next(iter(metrics), "")
    cohorts = order_humans_first(
        cohort
        for cohort, by_label in longitudinal_values.items()
        if any(by_metric.get(primary_metric) for by_metric in by_label.values())
    )
    by_authorship: dict[str, dict[str, dict[str, int | float]]] = {}
    for group in ("human", "agent"):
        group_values = {
            cohort: by_label
            for cohort, by_label in longitudinal_values.items()
            if (
                "human"
                if cohort.strip().casefold() in {"human", "humans"}
                else "agent"
            )
            == group
        }
        if group_values:
            by_authorship[group] = _payload_count_map(
                group_values,
                metric=primary_metric,
            )
    payload: dict[str, Any] = {
        "overall": _payload_count_map(longitudinal_values, metric=primary_metric),
        "by_cohort": {
            cohort: _payload_count_map(
                longitudinal_values,
                metric=primary_metric,
                cohort=cohort,
            )
            for cohort in cohorts
        },
        "by_authorship": by_authorship,
        "per_metric": {},
    }
    for metric in metrics:
        payload["per_metric"][metric] = {
            "overall": _payload_count_map(
                longitudinal_values,
                metric=metric,
            ),
            "by_cohort": {
                cohort: _payload_count_map(
                    longitudinal_values,
                    metric=metric,
                    cohort=cohort,
                )
                for cohort in order_humans_first(
                    cohort
                    for cohort, by_label in longitudinal_values.items()
                    if any(by_metric.get(metric) for by_metric in by_label.values())
                )
            },
            "by_authorship": by_authorship,
        }
    return payload
