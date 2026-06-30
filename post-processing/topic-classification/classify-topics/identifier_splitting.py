"""Identifier splitting helpers for topic-classification preprocessing.

Repository names and file paths contain compound identifiers. This module
prefers Spiral/Ronin for source-compatible splitting and provides a deterministic
regex fallback when that dependency is unavailable.
"""

from __future__ import annotations

import re
import collections
import collections.abc
from dataclasses import dataclass
from typing import Callable, Iterable


_IDENTIFIER_CANDIDATE_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_HARD_DELIMITER_RE = re.compile(r"[_./\\:\-]+")


@dataclass(frozen=True)
class IdentifierSplitter:
    """Callable identifier splitter plus metadata for manifests."""

    name: str
    paper_equivalent: bool
    implementation: str
    split_identifier: Callable[[str], list[str]]

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "paper_equivalent": self.paper_equivalent,
            "implementation": self.implementation,
        }

    def split_text(self, text: str) -> str:
        """Split identifier-looking spans while preserving non-identifier separators."""

        def replace(match: re.Match[str]) -> str:
            parts = self.split_identifier(match.group(0))
            return " ".join(part for part in parts if part)

        return _IDENTIFIER_CANDIDATE_RE.sub(replace, text)


def make_identifier_splitter(mode: str = "auto") -> IdentifierSplitter:
    """Create a splitter.

    Modes:
    - ``spiral``: require Spiral/Ronin and fail if unavailable.
    - ``regex``: use deterministic local splitting.
    - ``auto``: use Spiral/Ronin when installed, otherwise regex fallback.
    """

    normalized = str(mode or "auto").strip().lower()
    if normalized not in {"auto", "spiral", "regex"}:
        raise ValueError(
            "Identifier splitter mode must be one of: auto, spiral, regex; "
            f"got {mode!r}"
        )
    if normalized in {"auto", "spiral"}:
        try:
            return _make_spiral_splitter()
        except Exception as exc:
            if normalized == "spiral":
                raise RuntimeError(
                    "Paper-equivalent preprocessing requires the Spiral/Ronin "
                    "identifier splitter. Install post-processing requirements, "
                    "including git+https://github.com/casics/spiral.git, or set "
                    "POST_PROCESSING_TOPIC_CLASSIFICATION_IDENTIFIER_SPLITTER=regex "
                    "to use the documented non-paper-equivalent fallback."
                ) from exc
    return _make_regex_splitter()


def _make_spiral_splitter() -> IdentifierSplitter:
    # Spiral 1.1.0 still references collections.Iterable internally. Python 3.10+
    # moved it to collections.abc, so provide the old alias at runtime.
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]
    from spiral import ronin  # type: ignore[import-not-found]

    def split_identifier(identifier: str) -> list[str]:
        parts = ronin.split(str(identifier or ""))
        return [str(part) for part in parts if str(part)]

    return IdentifierSplitter(
        name="spiral_ronin",
        paper_equivalent=True,
        implementation="spiral.ronin.split",
        split_identifier=split_identifier,
    )


def _make_regex_splitter() -> IdentifierSplitter:
    return IdentifierSplitter(
        name="regex_fallback",
        paper_equivalent=False,
        implementation="local camel/acronym/digit/delimiter regex splitter",
        split_identifier=_regex_split_identifier,
    )


def _regex_split_identifier(identifier: str) -> list[str]:
    text = str(identifier or "")
    if text.isalpha() and (text.islower() or text.isupper()):
        return [text]
    text = _HARD_DELIMITER_RE.sub(" ", text)
    text = _CAMEL_ACRONYM_BOUNDARY.sub(" ", text)
    text = _CAMEL_BOUNDARY.sub(" ", text)
    text = _LETTER_DIGIT_BOUNDARY.sub(" ", text)
    return [part for part in text.split() if part]


def split_identifiers(values: Iterable[str], *, mode: str = "auto") -> list[str]:
    """Split a sequence of identifier-like strings into word tokens."""
    splitter = make_identifier_splitter(mode)
    tokens: list[str] = []
    for value in values:
        tokens.extend(splitter.split_identifier(value))
    return tokens
