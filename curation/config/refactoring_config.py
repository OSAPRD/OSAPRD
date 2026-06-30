"""Configuration for original-PR refactoring mining metrics.

Refactoring tools run only on the original before/after PR comparison.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

FUTURE_REFACTORING_SNAPSHOT_LABELS = ("+3d", "+7d", "+31d", "+61d")

_CURATION_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_BIN = _CURATION_ROOT / "tools" / "bin"
_TOOLS_RUNTIME = _CURATION_ROOT / "tools" / "runtime"
_IS_WINDOWS = os.name == "nt"


def _float_env(name: str, default: float) -> float:
    """Read a float environment variable with a forgiving default fallback."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


REFACTORING_MINER_TIMEOUT_SECONDS = _float_env("REFACTORING_MINER_TIMEOUT_SECONDS", 300.0)
REFFDIFF_TIMEOUT_SECONDS = _float_env("REFFDIFF_TIMEOUT_SECONDS", 300.0)
REFACTORING_MINER_PP_CAP_TIMEOUT_SECONDS = _float_env(
    "REFACTORING_MINER_PP_CAP_TIMEOUT_SECONDS",
    240.0,
)
REFACTORING_MINER_PP_TIMEOUT_SECONDS = _float_env(
    "REFACTORING_MINER_PP_TIMEOUT_SECONDS",
    240.0,
)


@dataclass(frozen=True)
class RefactoringToolConfig:
    """Resolved command and timeout for one refactoring tool."""

    tool_name: str
    default_command: str
    timeout_seconds: float | None = None

    def resolve_command(self) -> str:
        """Return the command used to launch the tool."""
        return self.default_command

    def resolve_args_template(self) -> str | None:
        """Return an optional argument template; current tools build args in code."""
        return None


REFACTORING_MINER_CONFIG = RefactoringToolConfig(
    tool_name="RefactoringMiner",
    default_command=(
        str(_TOOLS_BIN / "RefactoringMiner.bat")
        if _IS_WINDOWS
        else str(_TOOLS_RUNTIME / "refactoringminer" / "bin" / "RefactoringMiner")
    ),
    timeout_seconds=REFACTORING_MINER_TIMEOUT_SECONDS,
)

REFFDIFF_CONFIG = RefactoringToolConfig(
    tool_name="ReffDiff",
    default_command=(
        str(_TOOLS_BIN / "ReffDiff.bat")
        if _IS_WINDOWS
        else str(_TOOLS_RUNTIME / "refdiff" / "bin" / "refdiff-example")
    ),
    timeout_seconds=REFFDIFF_TIMEOUT_SECONDS,
)

REFACTORING_MINER_PP_CONFIG = RefactoringToolConfig(
    tool_name="RefactoringMiner++",
    default_command=(
        str(_TOOLS_BIN / "RefactoringMinerPP.bat")
        if _IS_WINDOWS
        else os.getenv(
            "REFACTORING_MINER_PP_COMMAND",
            str(_TOOLS_RUNTIME / "refactoringminerpp" / "bin" / "RefactoringMiner"),
        )
    ),
    timeout_seconds=REFACTORING_MINER_PP_TIMEOUT_SECONDS,
)
