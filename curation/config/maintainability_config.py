"""Configuration for maintainability metrics and code-smell tools."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_FUTURE_SNAPSHOT_LABELS = ("+3d", "+7d", "+31d", "+61d")
DEFAULT_MULTIMETRIC_COMMAND = "multimetric"
DEFAULT_MULTIMETRIC_MAINTINDEX_MODE = "sei"
DEFAULT_CODE_SMELL_TOOL_TIMEOUT_SECONDS = 180.0

_IS_WINDOWS = os.name == "nt"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CURATION_ROOT = _REPO_ROOT / "curation"


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read comma-separated snapshot labels from an environment variable."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return tuple(default)
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or tuple(default)


def _float_env(name: str, default: float) -> float:
    """Read a float environment variable with a forgiving default fallback."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _quote_path(path: str | Path) -> str:
    """Quote a command-template path so spaces remain safe after formatting."""
    return f'"{path}"'


def _local_tool_path(*parts: str) -> Path:
    """Return a path under curation/tools for local tool defaults."""
    return _CURATION_ROOT / "tools" / Path(*parts)


FUTURE_MAINTAINABILITY_SNAPSHOT_LABELS = _parse_csv_env(
    "CURATION_FUTURE_SNAPSHOT_LABELS",
    DEFAULT_FUTURE_SNAPSHOT_LABELS,
)
MULTIMETRIC_COMMAND = os.environ.get(
    "CURATION_MULTIMETRIC_COMMAND",
    DEFAULT_MULTIMETRIC_COMMAND,
).strip() or DEFAULT_MULTIMETRIC_COMMAND
MULTIMETRIC_MAINTINDEX_MODE = os.environ.get(
    "CURATION_MULTIMETRIC_MAINTINDEX_MODE",
    DEFAULT_MULTIMETRIC_MAINTINDEX_MODE,
).strip() or DEFAULT_MULTIMETRIC_MAINTINDEX_MODE

# Code-smell detection is an active companion to Multimetric. Tool templates are
# environment-overridable because Designite/DPy are licensed/local tools and
# because Windows installations often use vendored binaries.
CODE_SMELL_TOOL_TIMEOUT_SECONDS = _float_env(
    "CURATION_CODE_SMELL_TOOL_TIMEOUT_SECONDS",
    DEFAULT_CODE_SMELL_TOOL_TIMEOUT_SECONDS,
)

_DESIGNITE_JAVA_JAR = os.environ.get(
    "CURATION_DESIGNITE_JAVA_JAR",
    str(_local_tool_path("vendors", "designitejava", "DesigniteJava.jar")),
).strip()
DESIGNITE_JAVA_COMMAND_TEMPLATE = os.environ.get(
    "CURATION_DESIGNITE_JAVA_COMMAND_TEMPLATE",
    f"java -jar {_quote_path(_DESIGNITE_JAVA_JAR)} -i {{root}} -o {{out}}",
).strip()

DESIGNITE_PYTHON_COMMAND_TEMPLATE = os.environ.get(
    "CURATION_DESIGNITE_PYTHON_COMMAND_TEMPLATE",
    "dpy analyze -i {root} -o {out}",
).strip()

_PMD_COMMAND = (
    _local_tool_path("vendors", "pmd-bin-7.23.0", "bin", "pmd.bat")
    if _IS_WINDOWS
    else Path("pmd")
)
PMD_COMMAND_TEMPLATE = os.environ.get(
    "CURATION_PMD_COMMAND_TEMPLATE",
    f"{_quote_path(_PMD_COMMAND)} check -d {{root}} -R rulesets/java/quickstart.xml -f json",
).strip()

_ESLINT_COMMAND = (
    Path("eslint.cmd")
    if _IS_WINDOWS
    else Path("eslint")
)
ESLINT_COMMAND_TEMPLATE = os.environ.get(
    "CURATION_ESLINT_COMMAND_TEMPLATE",
    (
        f"{_quote_path(_ESLINT_COMMAND)} -f json --no-eslintrc --no-error-on-unmatched-pattern "
        "--ext .js,.jsx,.mjs,.cjs --parser-options ecmaVersion:2022,sourceType:module "
        '--rule "complexity:[2,10]" --rule "max-depth:[2,4]" '
        '--rule "max-lines-per-function:[1,120]" --rule "max-params:[1,5]" '
        '--rule "no-nested-ternary:2" --rule "no-else-return:1" {root}'
    ),
).strip()

_CPPCHECK_COMMAND = (
    _local_tool_path("vendors", "cppcheck", "PFiles", "Cppcheck", "cppcheck.exe")
    if _IS_WINDOWS
    else Path("cppcheck")
)
CPPCHECK_COMMAND_TEMPLATE = os.environ.get(
    "CURATION_CPPCHECK_COMMAND_TEMPLATE",
    (
        f"{_quote_path(_CPPCHECK_COMMAND)} --enable=warning,style,performance,portability "
        "--inline-suppr --template={{file}}:{{line}}:{{id}}:{{severity}}:{{message}} {root}"
    ),
).strip()

_CLANG_TIDY_COMMAND = (
    _local_tool_path("vendors", "llvm-20.1.8", "bin", "clang-tidy.exe")
    if _IS_WINDOWS
    else Path("clang-tidy")
)
CLANG_TIDY_COMMAND_TEMPLATE = os.environ.get(
    "CURATION_CLANG_TIDY_COMMAND_TEMPLATE",
    f"{_quote_path(_CLANG_TIDY_COMMAND)} {{file}} -- -std=c++17",
).strip()


def get_code_smell_tool_timeout_seconds() -> float:
    """Return the timeout used for one code-smell tool invocation."""
    return float(CODE_SMELL_TOOL_TIMEOUT_SECONDS)


def get_code_smell_tool_command_templates() -> dict[str, str]:
    """Return active code-smell command templates keyed by tool id."""
    return {
        "designite_java": DESIGNITE_JAVA_COMMAND_TEMPLATE,
        "designite_python": DESIGNITE_PYTHON_COMMAND_TEMPLATE,
        "pmd": PMD_COMMAND_TEMPLATE,
        "eslint": ESLINT_COMMAND_TEMPLATE,
        "cppcheck": CPPCHECK_COMMAND_TEMPLATE,
        "clang_tidy": CLANG_TIDY_COMMAND_TEMPLATE,
    }
