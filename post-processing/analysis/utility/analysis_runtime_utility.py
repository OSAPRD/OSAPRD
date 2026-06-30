"""Runtime helpers for streaming analysis execution."""

from __future__ import annotations

import ctypes
import gc
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from curation_parquet_utility import CohortParquetFiles


class AnalysisLogger:
    """Small structured logger for long-running analysis stages."""

    def __init__(
        self,
        pipeline: str,
        *,
        start_time: float | None = None,
        prefix: str = "[post-processing/analysis]",
    ) -> None:
        self.pipeline = pipeline
        self.start_time = time.perf_counter() if start_time is None else start_time
        self.prefix = prefix

    def log(self, stage: str, **details: Any) -> None:
        """Print a timestamped progress record."""
        elapsed = time.perf_counter() - self.start_time
        timestamp = datetime.now().isoformat(timespec="seconds")
        detail_text = " ".join(
            f"{key}={_format_log_value(value)}"
            for key, value in details.items()
            if value is not None
        )
        suffix = f" {detail_text}" if detail_text else ""
        print(
            f"{self.prefix} {timestamp} {elapsed:8.1f}s "
            f"pipeline={self.pipeline} stage={stage}{suffix}",
            flush=True,
        )


def _format_log_value(value: Any) -> str:
    """Format values compactly for single-line progress logs."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def log_cohort_input_counts(
    logger: AnalysisLogger,
    cohort_inputs: list[CohortParquetFiles],
) -> None:
    """Log input parquet counts overall and by cohort."""
    total = sum(len(cohort_input.paths) for cohort_input in cohort_inputs)
    logger.log("input_parquet_count", total_files=total, cohort_count=len(cohort_inputs))
    for cohort_input in cohort_inputs:
        logger.log(
            "input_parquet_count_by_cohort",
            cohort=cohort_input.cohort,
            files=len(cohort_input.paths),
        )


def make_progress_logger(
    logger: AnalysisLogger,
    *,
    prefix: str,
) -> Callable[..., None]:
    """Return a simple callback suitable for per-file streaming utilities."""

    def _progress(stage: str, **details: Any) -> None:
        logger.log(f"{prefix}_{stage}", **details)

    return _progress


def release_process_memory(
    logger: AnalysisLogger | Any | None = None,
    *,
    stage: str = "process_memory_released",
) -> None:
    """Request Python and native allocator cleanup after memory-heavy phases."""
    collected = gc.collect()
    malloc_trim_result: int | None = None
    if os.name == "posix":
        try:
            malloc_trim_result = int(ctypes.CDLL("libc.so.6").malloc_trim(0))
        except Exception:
            malloc_trim_result = None
    _log(
        logger,
        stage,
        python_objects_collected=collected,
        malloc_trim_result=malloc_trim_result,
    )


def _log(logger: AnalysisLogger | Any | None, stage: str, **details: Any) -> None:
    """Call a logger-like object without requiring a concrete logger type."""
    if logger is None:
        return
    log_method = getattr(logger, "log", None)
    if callable(log_method):
        log_method(stage, **details)
