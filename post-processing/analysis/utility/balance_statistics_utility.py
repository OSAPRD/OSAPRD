"""Statistical balance helpers shared by analysis pipelines.

The analysis package writes statistics into JSON alongside plots. These helpers
keep the calculations deterministic, normalize invalid numeric values, and
centralize SciPy imports so local dependency issues fail with one clear message.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from math import isfinite, sqrt
from random import Random
from statistics import median
from typing import Any, Iterable, Sequence


def _numeric_values(values: Iterable[Any]) -> list[float]:
    """Coerce an iterable to finite-ish floats, skipping invalid entries."""
    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return numeric


def _percentile(sorted_values: Sequence[float], percentile: float) -> float | None:
    """Return a linear-interpolated percentile from sorted numeric values."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def require_scipy_stats():
    """Import scipy.stats with a clear analysis-specific failure message."""
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "SciPy is required for post-processing analysis statistics. "
            "Use the analysis Docker image or install a compatible scipy/numpy "
            "runtime before running analysis."
        ) from exc
    return stats


def numeric_distribution_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    """Return standard numeric distribution summaries."""
    numeric = sorted(_numeric_values(values))
    nonzero = [value for value in numeric if value != 0.0]
    zero_count = len(numeric) - len(nonzero)
    if not numeric:
        return {
            "mean": None,
            "median": None,
            "median_nonzero": None,
            "zero_count": 0,
            "zero_percentage": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "iqr": None,
        }

    q1 = _percentile(numeric, 0.25)
    q3 = _percentile(numeric, 0.75)
    return {
        "mean": float(sum(numeric) / len(numeric)),
        "median": float(median(numeric)),
        "median_nonzero": float(median(nonzero)) if nonzero else None,
        "zero_count": int(zero_count),
        "zero_percentage": float(100.0 * zero_count / len(numeric)),
        "min": float(numeric[0]),
        "max": float(numeric[-1]),
        "q1": q1,
        "q3": q3,
        "iqr": None if q1 is None or q3 is None else float(q3 - q1),
    }


def median_confidence_interval(
    values: Iterable[Any],
    *,
    confidence: float = 0.95,
    resamples: int = 500,
    max_sample_size: int = 1000,
    seed: int = 20260608,
) -> dict[str, float | int | None]:
    """Return median and a deterministic bootstrap confidence interval."""
    numeric = _numeric_values(values)
    count = len(numeric)
    suffix = str(int(confidence * 100))
    low_key = f"ci{suffix}_low"
    high_key = f"ci{suffix}_high"
    if count == 0:
        return {
            "n": 0,
            "median": None,
            low_key: None,
            high_key: None,
        }

    median_value = float(median(numeric))
    if count == 1 or len(set(numeric)) == 1:
        return {
            "n": count,
            "median": median_value,
            low_key: median_value,
            high_key: median_value,
        }

    rng = Random(seed)
    if len(numeric) > max_sample_size:
        numeric = rng.sample(numeric, max_sample_size)

    stats = require_scipy_stats()
    try:
        result = stats.bootstrap(
            (numeric,),
            lambda sample: float(median(sample)),
            confidence_level=confidence,
            n_resamples=resamples,
            method="percentile",
            random_state=seed,
            vectorized=False,
        )
    except ValueError:
        return {
            "n": count,
            "median": median_value,
            low_key: None,
            high_key: None,
        }

    lower = float(result.confidence_interval.low)
    upper = float(result.confidence_interval.high)
    if not isfinite(lower) or not isfinite(upper):
        lower = None
        upper = None
    return {
        "n": count,
        "median": median_value,
        low_key: lower,
        high_key: upper,
    }


def benjamini_hochberg_adjust(
    p_values: Sequence[float | None],
) -> list[float | None]:
    """Return Benjamini-Hochberg FDR-adjusted p-values, preserving nulls."""
    indexed: list[tuple[int, float]] = []
    for index, value in enumerate(p_values):
        if value is None:
            continue
        try:
            p_value = float(value)
        except (TypeError, ValueError):
            continue
        if not isfinite(p_value):
            continue
        indexed.append((index, max(0.0, min(1.0, p_value))))
    adjusted: list[float | None] = [None for _value in p_values]
    if not indexed:
        return adjusted

    stats = require_scipy_stats()
    ordered_indexes = [index for index, _p_value in indexed]
    ordered_p_values = [p_value for _index, p_value in indexed]
    if hasattr(stats, "false_discovery_control"):
        adjusted_values = stats.false_discovery_control(
            ordered_p_values,
            method="bh",
        )
        for index, adjusted_p_value in zip(ordered_indexes, adjusted_values):
            adjusted[index] = float(max(0.0, min(1.0, adjusted_p_value)))
        return adjusted

    # SciPy < 1.11 does not expose false_discovery_control. Keep the small
    # traversal fallback here so JSON FDR output still works with older SciPy.
    ranked = sorted(indexed, key=lambda item: item[1])
    total = len(ranked)
    running_min = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(ranked), start=1):
        rank = total - rank_from_end + 1
        running_min = min(running_min, p_value * total / rank)
        adjusted[index] = max(0.0, min(1.0, running_min))
    return adjusted


def apply_fdr_correction(
    payload: Any,
    *,
    alpha: float = 0.05,
    method: str = "benjamini_hochberg",
) -> Any:
    """Add FDR adjusted p-values to every nested dict with a p_value field."""
    tests: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "p_value" in value:
                tests.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(payload)
    adjusted = benjamini_hochberg_adjust(
        [
            None if test.get("p_value") is None else float(test.get("p_value"))
            for test in tests
        ]
    )
    for test, adjusted_p_value in zip(tests, adjusted):
        test["adjusted_p_value"] = adjusted_p_value
        test["fdr_method"] = method
        test["significant_after_fdr"] = (
            bool(adjusted_p_value <= alpha)
            if adjusted_p_value is not None
            else False
        )
    return payload


def cliffs_delta_confidence_interval(
    first_values: Iterable[Any],
    second_values: Iterable[Any],
    *,
    confidence: float = 0.95,
    resamples: int = 300,
    max_sample_size: int = 500,
    seed: int = 20260608,
) -> tuple[float | None, float | None]:
    """Bootstrap a deterministic confidence interval for Cliff's delta."""
    first = _numeric_values(first_values)
    second = _numeric_values(second_values)
    if not first or not second or resamples <= 0:
        return None, None

    rng = Random(seed)
    if len(first) > max_sample_size:
        first = rng.sample(first, max_sample_size)
    if len(second) > max_sample_size:
        second = rng.sample(second, max_sample_size)

    def statistic(sample_first, sample_second) -> float:
        estimate = cliffs_delta(sample_first, sample_second)
        return 0.0 if estimate is None else float(estimate)

    stats = require_scipy_stats()
    try:
        result = stats.bootstrap(
            (first, second),
            statistic,
            confidence_level=confidence,
            n_resamples=resamples,
            method="percentile",
            paired=False,
            random_state=seed,
            vectorized=False,
        )
    except ValueError:
        return None, None

    lower = float(result.confidence_interval.low)
    upper = float(result.confidence_interval.high)
    if not isfinite(lower) or not isfinite(upper):
        return None, None
    return lower, upper


def add_cliffs_delta_ci(
    test: dict[str, Any],
    first_values: Iterable[Any],
    second_values: Iterable[Any],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Add a Cliff's delta confidence interval to a Mann-Whitney payload."""
    low, high = cliffs_delta_confidence_interval(
        first_values,
        second_values,
        confidence=confidence,
    )
    suffix = str(int(confidence * 100))
    test[f"cliffs_delta_ci{suffix}_low"] = low
    test[f"cliffs_delta_ci{suffix}_high"] = high
    return test


def build_contingency_matrix(
    rows: Iterable[tuple[Any, Any, Any]],
    *,
    row_labels: Sequence[Any] | None = None,
    column_labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build a dense matrix from row label, column label, count tuples."""
    counts: dict[tuple[str, str], int] = {}
    observed_rows: list[str] = []
    observed_columns: list[str] = []
    for row_label, column_label, count in rows:
        if row_label is None or column_label is None:
            continue
        row_key = str(row_label)
        column_key = str(column_label)
        if row_key not in observed_rows:
            observed_rows.append(row_key)
        if column_key not in observed_columns:
            observed_columns.append(column_key)
        counts[(row_key, column_key)] = counts.get((row_key, column_key), 0) + int(
            count or 0
        )

    resolved_rows = (
        [str(label) for label in row_labels]
        if row_labels is not None
        else sorted(observed_rows)
    )
    resolved_columns = (
        [str(label) for label in column_labels]
        if column_labels is not None
        else sorted(observed_columns)
    )
    matrix = [
        [counts.get((row_label, column_label), 0) for column_label in resolved_columns]
        for row_label in resolved_rows
    ]
    return {
        "row_labels": resolved_rows,
        "column_labels": resolved_columns,
        "matrix": matrix,
    }


def _compact_nonzero_matrix(
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    matrix: Sequence[Sequence[int]],
) -> tuple[list[str], list[str], list[list[int]]]:
    """Drop all-zero rows and columns before contingency testing."""
    matrix_lists = [[int(value or 0) for value in row] for row in matrix]
    row_indexes = [
        index for index, row in enumerate(matrix_lists)
        if sum(row) > 0
    ]
    column_indexes = [
        index for index in range(len(column_labels))
        if sum(matrix_lists[row_index][index] for row_index in row_indexes) > 0
    ]
    compact_rows = [str(row_labels[index]) for index in row_indexes]
    compact_columns = [str(column_labels[index]) for index in column_indexes]
    compact_matrix = [
        [matrix_lists[row_index][column_index] for column_index in column_indexes]
        for row_index in row_indexes
    ]
    return compact_rows, compact_columns, compact_matrix


def cramers_v(
    chi_square_statistic: float,
    n: int,
    row_count: int,
    column_count: int,
) -> float | None:
    """Compute Cramer's V from a chi-square statistic."""
    denominator_dimension = min(row_count - 1, column_count - 1)
    if n <= 0 or denominator_dimension <= 0:
        return None
    return float(sqrt(chi_square_statistic / (n * denominator_dimension)))


def _run_chi2_contingency(
    matrix: Sequence[Sequence[int]],
) -> tuple[float, float, int]:
    """Run SciPy's chi-square test and normalize scalar return values."""
    stats = require_scipy_stats()
    chi_square_statistic, p_value, degrees_of_freedom, _expected = (
        stats.chi2_contingency(matrix, correction=False)
    )
    return (
        float(chi_square_statistic),
        float(p_value),
        int(degrees_of_freedom),
    )


def chi_square_independence(
    matrix: Sequence[Sequence[int]],
    *,
    row_labels: Sequence[Any] | None = None,
    column_labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run Pearson's chi-square test and Cramer's V for a contingency table."""
    resolved_row_labels = [
        str(label) for label in (
            row_labels if row_labels is not None else range(len(matrix))
        )
    ]
    resolved_column_labels = [
        str(label)
        for label in (
            column_labels
            if column_labels is not None
            else range(len(matrix[0]) if matrix else 0)
        )
    ]
    compact_rows, compact_columns, compact_matrix = _compact_nonzero_matrix(
        resolved_row_labels,
        resolved_column_labels,
        matrix,
    )
    n = sum(sum(row) for row in compact_matrix)
    if len(compact_rows) < 2 or len(compact_columns) < 2 or n <= 0:
        return {
            "chi_square_statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "cramers_v": None,
            "n": int(n),
            "row_count": len(compact_rows),
            "column_count": len(compact_columns),
        }

    try:
        chi_square_statistic, p_value, degrees_of_freedom = _run_chi2_contingency(
            compact_matrix
        )
    except ValueError:
        return {
            "chi_square_statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "cramers_v": None,
            "n": int(n),
            "row_count": len(compact_rows),
            "column_count": len(compact_columns),
        }

    statistic = float(chi_square_statistic)
    normalized_p_value = float(p_value)
    if not isfinite(normalized_p_value):
        normalized_p_value = None
    return {
        "chi_square_statistic": statistic,
        "p_value": normalized_p_value,
        "degrees_of_freedom": int(degrees_of_freedom),
        "cramers_v": cramers_v(
            statistic,
            int(n),
            len(compact_rows),
            len(compact_columns),
        ),
        "n": int(n),
        "row_count": len(compact_rows),
        "column_count": len(compact_columns),
    }


def chi_square_goodness_of_fit_uniform(
    counts: Iterable[Any],
) -> dict[str, float | int | None]:
    """Run a chi-square goodness-of-fit test against a uniform distribution."""
    observed = []
    for count in counts:
        try:
            normalized = int(count or 0)
        except (TypeError, ValueError):
            normalized = 0
        observed.append(max(0, normalized))

    bucket_count = len(observed)
    n = sum(observed)
    if bucket_count < 2 or n <= 0:
        return {
            "chi_square_statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "n": int(n),
            "bucket_count": bucket_count,
        }

    expected = n / bucket_count
    stats = require_scipy_stats()
    result = stats.chisquare(
        f_obs=observed,
        f_exp=[expected for _count in observed],
    )
    statistic = float(result.statistic)
    degrees_of_freedom = bucket_count - 1
    p_value = float(result.pvalue)
    if not isfinite(p_value):
        p_value = None
    return {
        "chi_square_statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
        "n": int(n),
        "bucket_count": bucket_count,
    }


def cliffs_delta(first_values: Iterable[Any], second_values: Iterable[Any]) -> float | None:
    """Return Cliff's delta comparing first_values against second_values."""
    first = sorted(_numeric_values(first_values))
    second = sorted(_numeric_values(second_values))
    return _cliffs_delta_from_sorted(first, second)


def _cliffs_delta_from_sorted(
    first: Sequence[float],
    second: Sequence[float],
) -> float | None:
    """Return Cliff's delta for pre-normalized sorted numeric values."""
    if not first or not second:
        return None
    dominance = 0
    for value in first:
        dominance += bisect_left(second, value)
        dominance -= len(second) - bisect_right(second, value)
    return float(dominance / (len(first) * len(second)))


def mann_whitney_u_test(
    first_values: Iterable[Any],
    second_values: Iterable[Any],
) -> dict[str, float | int | None]:
    """Return Mann-Whitney U, two-sided SciPy p-value, and Cliff's delta."""
    first = _numeric_values(first_values)
    second = _numeric_values(second_values)
    n_first = len(first)
    n_second = len(second)
    if n_first == 0 or n_second == 0:
        return {
            "u_statistic": None,
            "p_value": None,
            "cliffs_delta": None,
            "n_first": n_first,
            "n_second": n_second,
        }

    stats = require_scipy_stats()
    result = stats.mannwhitneyu(
        first,
        second,
        alternative="two-sided",
        method="auto",
    )
    p_value = float(result.pvalue)
    if not isfinite(p_value):
        p_value = None
    return {
        "u_statistic": float(result.statistic),
        "p_value": p_value,
        "cliffs_delta": _cliffs_delta_from_sorted(sorted(first), sorted(second)),
        "n_first": n_first,
        "n_second": n_second,
    }


def kruskal_wallis_test(
    groups: dict[str, Iterable[Any]],
) -> dict[str, float | int | None]:
    """Return SciPy Kruskal-Wallis H, p-value, and epsilon-squared."""
    numeric_groups = {
        str(label): _numeric_values(values)
        for label, values in groups.items()
    }
    numeric_groups = {
        label: values for label, values in numeric_groups.items() if values
    }
    group_count = len(numeric_groups)
    n_total = sum(len(values) for values in numeric_groups.values())
    if group_count < 2 or n_total <= group_count:
        return {
            "h_statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "epsilon_squared": None,
            "n": n_total,
            "group_count": group_count,
        }

    all_values = [
        value
        for values in numeric_groups.values()
        for value in values
    ]
    if len(set(all_values)) < 2:
        return {
            "h_statistic": 0.0,
            "p_value": 1.0,
            "degrees_of_freedom": group_count - 1,
            "epsilon_squared": 0.0,
            "n": n_total,
            "group_count": group_count,
        }

    stats = require_scipy_stats()
    result = stats.kruskal(*numeric_groups.values())
    h_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not isfinite(p_value):
        p_value = None
    degrees_of_freedom = group_count - 1
    return {
        "h_statistic": h_statistic,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
        "epsilon_squared": max(
            0.0,
            (h_statistic - group_count + 1.0) / (n_total - group_count),
        ),
        "n": n_total,
        "group_count": group_count,
    }
