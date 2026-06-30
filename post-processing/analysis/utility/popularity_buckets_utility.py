"""Popularity bucket helpers shared by analysis pipelines."""

from __future__ import annotations

from typing import Any, Iterable


LOW_POPULARITY_MAX_STARS = 0
MEDIUM_POPULARITY_MAX_STARS = 18


def build_popularity_bucket_scheme(
    repository_star_counts: Iterable[Any],
    *,
    bucket_count: int = 3,
) -> dict[str, Any]:
    """Build fixed 3-bucket popularity scheme for repository star counts."""
    if int(bucket_count) != 3:
        raise ValueError("analysis popularity bucketing currently requires 3 buckets")
    # Consume the iterable so callers can pass generators without changing behavior.
    list(repository_star_counts)
    upper_bounds = [LOW_POPULARITY_MAX_STARS, MEDIUM_POPULARITY_MAX_STARS]

    labels = ["pop0", "pop1", "pop2"]
    cutoff_points = [
        {
            "bucket": "pop0",
            "label": "0 stars",
            "lower_star_bound": 0,
            "upper_star_bound": 0,
        },
        {
            "bucket": "pop1",
            "label": "1-18 stars",
            "lower_star_bound": 1,
            "upper_star_bound": MEDIUM_POPULARITY_MAX_STARS,
        },
        {
            "bucket": "pop2",
            "label": "19+ stars",
            "lower_star_bound": MEDIUM_POPULARITY_MAX_STARS + 1,
            "upper_star_bound": None,
        },
    ]
    return {
        "bucket_labels": labels,
        "upper_star_bounds": upper_bounds,
        "cutoff_points": cutoff_points,
    }


def popularity_bucket_for_stars(stars: Any, scheme: dict[str, Any]) -> str:
    """Return the popularity bucket label for a stargazer count."""
    try:
        normalized_stars = max(0, int(stars))
    except (TypeError, ValueError):
        normalized_stars = 0
    upper_bounds = [int(value) for value in scheme.get("upper_star_bounds") or []]
    for index, upper_bound in enumerate(upper_bounds):
        if normalized_stars <= upper_bound:
            return f"pop{index}"
    return f"pop{len(upper_bounds)}"

