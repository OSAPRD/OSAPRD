"""Constants and discovery helpers for optional multimetric analysis."""

from __future__ import annotations

from pathlib import Path


OUTPUT_DIR_NAME = "maintainability"
OUTPUT_FILENAME = "maintainability_multimetrics_results.json"
MULTIMETRIC_SNAPSHOT_FILENAME = "multimetric_snapshot_metrics.parquet"

RAW_MULTIMETRIC_METRICS = (
    "loc",
    "comment_ratio",
    "cyclomatic_complexity",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugprop",
    "halstead_timerequired",
    "maintainability_index",
    "fanout_internal",
    "fanout_external",
    "operands_sum",
    "operands_uniq",
    "operators_sum",
    "operators_uniq",
    "pylint",
    "tiobe",
    "tiobe_complexity",
    "tiobe_duplication",
    "tiobe_functional",
)
PER_KLOC_MULTIMETRIC_BASE_METRICS = {
    "halstead_volume": "halstead_volume_per_kloc",
    "halstead_difficulty": "halstead_difficulty_per_kloc",
    "halstead_effort": "halstead_effort_per_kloc",
    "cyclomatic_complexity": "cyclomatic_complexity_per_kloc",
    "halstead_bugprop": "halstead_bugprop_per_kloc",
    "fanout_external": "fanout_external_per_kloc",
    "halstead_timerequired": "halstead_timerequired_per_kloc",
}
DERIVED_MULTIMETRIC_METRICS = tuple(PER_KLOC_MULTIMETRIC_BASE_METRICS.values())
ALIAS_MULTIMETRIC_BASE_METRICS = {
    "tiobe_duplication": "multimetric_duplication_score",
}
ORIGINAL_MAINTAINABILITY_REPLACEMENT_METRICS = (
    "original_code_smell_density_delta",
    "original_duplication_density",
)
ALIAS_MULTIMETRIC_METRICS = tuple(ALIAS_MULTIMETRIC_BASE_METRICS.values())
MULTIMETRIC_METRICS = (
    RAW_MULTIMETRIC_METRICS
    + DERIVED_MULTIMETRIC_METRICS
    + ALIAS_MULTIMETRIC_METRICS
    + ORIGINAL_MAINTAINABILITY_REPLACEMENT_METRICS
)
MULTIMETRIC_REPLACEMENT_METRICS = (
    "cyclomatic_complexity_per_kloc",
    "halstead_volume_per_kloc",
    "comment_ratio",
    "maintainability_index",
    "original_duplication_density",
)
MULTIMETRIC_QUALITY_GRID_METRICS = (
    "original_code_smell_density_delta",
    "original_duplication_density",
    "comment_ratio",
    "cyclomatic_complexity_per_kloc",
    "halstead_volume_per_kloc",
    "maintainability_index",
)
MULTIMETRIC_ADDITIONAL_PLOT_METRICS = (
    "fanout_external_per_kloc",
    "tiobe",
    "halstead_bugprop_per_kloc",
    "halstead_difficulty_per_kloc",
    "halstead_effort_per_kloc",
    "halstead_timerequired_per_kloc",
    "tiobe_complexity",
    "tiobe_duplication",
)
MULTIMETRIC_PLOT_METRICS = tuple(
    dict.fromkeys(MULTIMETRIC_REPLACEMENT_METRICS + MULTIMETRIC_ADDITIONAL_PLOT_METRICS)
)
LONGITUDINAL_MULTIMETRIC_TIMEPOINTS = ("+3d", "+7d", "+31d", "+61d")
MULTIMETRIC_LABELS = {
    "loc": "Lines of Code",
    "comment_ratio": "Comment Ratio (%)",
    "cyclomatic_complexity": "Cyclomatic Complexity",
    "halstead_volume": "Halstead Volume",
    "halstead_difficulty": "Halstead Difficulty",
    "halstead_effort": "Halstead Effort",
    "halstead_bugprop": "Halstead Delivered Bugs",
    "halstead_timerequired": "Halstead Time Required",
    "maintainability_index": "Maintainability Index",
    "fanout_internal": "Fanout Internal",
    "fanout_external": "Fanout External",
    "operands_sum": "Sum of Operands",
    "operands_uniq": "Unique Operands",
    "operators_sum": "Sum of Operators",
    "operators_uniq": "Unique Operators",
    "pylint": "Pylint",
    "tiobe": "TIOBE Quality Score",
    "tiobe_complexity": "TIOBE Complexity",
    "tiobe_duplication": "TIOBE Duplication",
    "tiobe_functional": "Functional Defect Score",
    "halstead_volume_per_kloc": "Halstead Volume per KLOC",
    "halstead_difficulty_per_kloc": "Halstead Difficulty per KLOC",
    "halstead_effort_per_kloc": "Halstead Effort per KLOC",
    "cyclomatic_complexity_per_kloc": "Cyclomatic Complexity per KLOC",
    "halstead_bugprop_per_kloc": "Halstead Delivered Bugs per KLOC",
    "fanout_external_per_kloc": "Fan Out per KLOC",
    "halstead_timerequired_per_kloc": "Halstead Time Required per KLOC",
    "multimetric_duplication_score": "Duplication Score",
    "original_code_smell_density_delta": "Smells per KLOC",
    "original_duplication_density": "Duplicated Lines Density (%)",
}


def discover_multimetric_snapshot_parquets(root: Path | str | None) -> tuple[Path, ...]:
    """Discover multimetric snapshot parquet files under an optional output root."""
    if root is None:
        return ()
    root_path = Path(root)
    if not root_path.exists():
        return ()
    return tuple(sorted(root_path.rglob(MULTIMETRIC_SNAPSHOT_FILENAME)))
