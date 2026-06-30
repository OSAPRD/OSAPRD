"""Shared CSV helpers for post-processing outputs and plots."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries, returning an empty list when absent."""
    resolved_path = Path(path)
    if not resolved_path.is_file():
        return []
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    fieldnames: list[str],
) -> None:
    """Write dictionaries to a CSV file with a header."""
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_or_default(value: Any, default: float = 0.0) -> float:
    """Parse a float, returning a default for empty or malformed values."""
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def parse_optional_float(value: Any) -> float | None:
    """Parse a float, returning None for empty or malformed values."""
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
