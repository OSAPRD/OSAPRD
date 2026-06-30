"""Repository text and topic preprocessing for topic classification.

The preprocessor applies the vendored/generated rule profile to topic labels,
repository text, README/wiki text, and repository file paths. Model bundles can
also provide runtime vocabularies so classification uses the same token filters
that were recorded during training.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import re
import unicodedata
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from identifier_splitting import IdentifierSplitter, make_identifier_splitter


CLASSIFY_TOPICS_DIR = Path(__file__).resolve().parent
TOPIC_CLASSIFICATION_DIR = CLASSIFY_TOPICS_DIR.parent
DATA_PREPARATION_DIR = TOPIC_CLASSIFICATION_DIR / "data-preparation"
DEFAULT_RULES_DIR = DATA_PREPARATION_DIR / "rules"
DEFAULT_LISTS_DIR = DATA_PREPARATION_DIR / "lists"
DEFAULT_GENERATED_DATA_PREPARATION_DIR = DATA_PREPARATION_DIR / "generated" / "github-topics"
SKIPPED_STATIC_LOW_FREQ_RULES = ("low_freq_topics.csv",)
UPSTREAM_RECOMMENDER_REPOSITORY_URL = (
    "https://github.com/MalihehIzadi/SoftwareTagRecommender"
)

NAME_TOKEN_LIMIT = 10
DESCRIPTION_TOKEN_LIMIT = 50
README_TOKEN_LIMIT = 400
WIKI_TOKEN_LIMIT = 100
FILE_TOKEN_LIMIT = 100
TOKEN_LIMIT_TEXT_WINDOW_MULTIPLIER = 200
TOKEN_LIMIT_MIN_TEXT_WINDOW_CHARS = 4096
TEXT_PHRASE_REPLACEMENT_MAX_CHARS = 2048
DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS = frozenset(
    {
        "api",
        "app",
        "bot",
        "client",
        "config",
        "core",
        "demo",
        "extension",
        "framework",
        "git",
        "github",
        "json",
        "lib",
        "library",
        "manager",
        "module",
        "plugin",
        "server",
        "service",
        "tool",
        "ui",
        "web",
    }
)

TOPIC_RULE_APPLICATION_ORDER = (
    "topics_contains_version.csv",
    "topics_contains_number.csv",
    "split_dash_topics.csv",
    "contains_top_topics.csv",
    "remove_plural_topics.csv",
    "contains_selected_topics.csv",
    "contractions.csv",
    "remove_stopwords_topic.csv",
    "remove_lemmatize_topic.csv",
    "delete.csv",
    "low_freq_topics.csv",
)

# Legacy fallback copy of nltk.corpus.stopwords.words("english"). Runtime
# preprocessing calls NLTK directly, matching the vendored data-preparation
# notebook; this set is retained only as a local safety check.
NLTK_ENGLISH_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "ain",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "couldn't",
        "d",
        "did",
        "didn",
        "didn't",
        "do",
        "does",
        "doesn",
        "doesn't",
        "doing",
        "don",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "hadn't",
        "has",
        "hasn",
        "hasn't",
        "have",
        "haven",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "isn't",
        "it",
        "it'd",
        "it'll",
        "it's",
        "its",
        "itself",
        "just",
        "ll",
        "m",
        "ma",
        "me",
        "mightn",
        "mightn't",
        "more",
        "most",
        "mustn",
        "mustn't",
        "my",
        "myself",
        "needn",
        "needn't",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shan",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "should've",
        "shouldn",
        "shouldn't",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "that'll",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "wasn",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "y",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_LETTER_TOKEN_RE = re.compile(r"[a-z]+")
_FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_MARKDOWN_LINK_RE = re.compile(r"!\[[^\]]*]\([^)]*\)|\[([^\]]+)]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_USERNAME_RE = re.compile(r"(?<!\w)@[\w-]+")
_DATE_TIME_RE = re.compile(
    r"\b\d{1,4}[-/:.]\d{1,2}(?:[-/:.]\d{1,4})?(?:[tT ]\d{1,2}:\d{2}(?::\d{2})?)?\b"
)
_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_DICT_LINE_RE = re.compile(
    r'''"([^"]+)"\s*:\s*"([^"]*)"|\'([^\']+)\'\s*:\s*\'([^\']*)\''''
)
@dataclass(frozen=True)
class PreparedRepositoryText:
    """Prepared inference text plus source-level token counts and diagnostics."""

    text: str
    token_counts: dict[str, int]
    data_preparation: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 4), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_rule_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            values = [cell.strip().lower() for cell in row]
            while values and values[-1] == "":
                values.pop()
            if len(values) >= 2:
                rows.append(values)
    return rows


def _load_apply_rule(path: Path) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    for row in _read_csv_rule_rows(path):
        if len(row) < 3:
            continue
        source = row[1]
        replacements = tuple(item for item in row[2:] if item)
        if source and replacements:
            mapping[source] = replacements
    return mapping


def _load_replace_rule(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in _read_csv_rule_rows(path):
        if len(row) < 3:
            continue
        canonical = row[1]
        alias = row[2]
        if canonical and alias:
            mapping[alias] = canonical
    return mapping


def _load_aggregate_rules(path: Path) -> list[tuple[tuple[str, ...], str]]:
    rules: list[tuple[tuple[str, ...], str]] = []
    for row in _read_csv_rule_rows(path):
        if len(row) < 3:
            continue
        canonical = row[1]
        required = tuple(item for item in row[2:] if item)
        if canonical and required:
            rules.append((required, canonical))
    return rules


def _literal_assignments(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        module = ast.parse(text)
    except SyntaxError:
        return _fallback_dict_assignments(text)
    assignments: dict[str, Any] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value
    return assignments


def _fallback_dict_assignments(text: str) -> dict[str, dict[str, str]]:
    current_name: str | None = None
    result: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        assignment = _ASSIGNMENT_RE.match(line)
        if assignment:
            current_name = assignment.group(1)
            result.setdefault(current_name, {})
        if current_name is None:
            continue
        match = _DICT_LINE_RE.search(line)
        if match:
            key = match.group(1) or match.group(3)
            value = match.group(2) or match.group(4)
            result.setdefault(current_name, {})[key] = value
    return result


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key).strip().lower(): str(mapped).strip().lower()
        for key, mapped in value.items()
        if str(key).strip() and str(mapped).strip()
    }


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item).strip().lower() for item in value if str(item).strip()}


def _window_text_for_token_limit(text: str, limit: int | None) -> str:
    if limit is None or limit <= 0:
        return text
    max_chars = max(
        TOKEN_LIMIT_MIN_TEXT_WINDOW_CHARS,
        int(limit) * TOKEN_LIMIT_TEXT_WINDOW_MULTIPLIER,
    )
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _compile_phrase_replacement_pattern(
    replacements: Mapping[str, str],
) -> re.Pattern[str] | None:
    sources = [
        source
        for source in replacements
        if len(str(source).strip()) > 1
    ]
    if not sources:
        return None
    alternatives = "|".join(
        re.escape(source)
        for source in sorted(sources, key=lambda value: (-len(value), value))
    )
    return re.compile(
        rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


@lru_cache(maxsize=1)
def _nltk_english_stop_words() -> frozenset[str]:
    """Return the same stopword list API used by the source notebook."""

    try:
        from nltk.corpus import stopwords
    except Exception as exc:  # pragma: no cover - dependency/environment failure.
        raise RuntimeError(
            "Runtime topic preprocessing requires nltk.corpus.stopwords to match "
            "topic-classification/data-preparation. Install NLTK."
        ) from exc

    try:
        words = stopwords.words("english")
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords", quiet=True)
            words = stopwords.words("english")
        except Exception as exc:  # pragma: no cover - dependency/environment failure.
            raise RuntimeError(
                "Runtime topic preprocessing requires the NLTK stopwords corpus "
                "to match topic-classification/data-preparation. Install it with "
                "`python -m nltk.downloader stopwords`."
            ) from exc
    return frozenset(str(word).lower() for word in words)


class SoftwareTagDataPreprocessor:
    """Apply the vendored topic data-preparation rules."""

    def __init__(
        self,
        *,
        rules_dir: Path = DEFAULT_RULES_DIR,
        lists_dir: Path = DEFAULT_LISTS_DIR,
        identifier_splitter_mode: str = "auto",
        identifier_splitter: IdentifierSplitter | None = None,
        preprocessing_artifacts: Mapping[str, Any] | None = None,
        allow_heuristic_file_splits: bool = False,
        skipped_topic_rule_names: Iterable[str] = (),
    ) -> None:
        self.rules_dir = Path(rules_dir)
        self.lists_dir = Path(lists_dir)
        self.allow_heuristic_file_splits = bool(allow_heuristic_file_splits)
        self.skipped_topic_rule_names = {
            str(name).strip()
            for name in skipped_topic_rule_names
            if str(name).strip()
        }
        self.identifier_splitter = identifier_splitter or make_identifier_splitter(
            identifier_splitter_mode
        )
        self.topic_rules = [
            (
                name,
                {}
                if name in self.skipped_topic_rule_names
                else _load_apply_rule(self.rules_dir / name),
            )
            for name in TOPIC_RULE_APPLICATION_ORDER
        ]
        self.topic_replace_aliases = _load_replace_rule(self.rules_dir / "replace.csv")
        self.topic_aggregate_rules = _load_aggregate_rules(self.rules_dir / "abbr.csv")
        self.file_name_token_split_rules = self._load_file_name_token_split_rules()
        self.text_replacements = self._load_text_replacements()
        self.text_phrase_replacements = {
            source: target
            for source, target in self.text_replacements.items()
            if re.search(r"[^a-z0-9]", source)
        }
        self.text_phrase_replacement_pattern = _compile_phrase_replacement_pattern(
            self.text_phrase_replacements,
        )
        self.file_name_stop_tokens = self._load_file_name_stop_tokens()
        self.preprocessing_artifacts = dict(preprocessing_artifacts or {})
        (
            self.allowed_text_tokens,
            self.allowed_file_name_tokens,
            self.runtime_vocabulary_source,
        ) = _runtime_vocabularies_from_artifacts(self.preprocessing_artifacts)
        self.file_name_informative_split_tokens = (
            self._load_file_name_informative_split_tokens()
        )
        self.file_name_informative_split_sequence = tuple(
            sorted(
                set(self.file_name_informative_split_tokens),
                key=lambda value: (-len(value), value),
            )
        )
        self._file_name_token_expansion_cache: dict[str, tuple[str, ...]] = {}
        self._text_token_cache: dict[tuple[str, bool], tuple[str, ...]] = {}

    @property
    def source_token_limits(self) -> dict[str, int]:
        return {
            "name": NAME_TOKEN_LIMIT,
            "description": DESCRIPTION_TOKEN_LIMIT,
            "readme": README_TOKEN_LIMIT,
            "wiki": WIKI_TOKEN_LIMIT,
            "file_names": FILE_TOKEN_LIMIT,
        }

    @property
    def compact_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "topic_data_preparation_v3",
            "upstream_repository_url": UPSTREAM_RECOMMENDER_REPOSITORY_URL,
            "skipped_topic_rule_names": sorted(self.skipped_topic_rule_names),
            "source_token_limits": self.source_token_limits,
            "token_limit_text_window": {
                "enabled": True,
                "minimum_characters": TOKEN_LIMIT_MIN_TEXT_WINDOW_CHARS,
                "characters_per_requested_token": TOKEN_LIMIT_TEXT_WINDOW_MULTIPLIER,
            },
            "phrase_replacement_policy": {
                "enabled_for_text_up_to_characters": TEXT_PHRASE_REPLACEMENT_MAX_CHARS,
                "token_replacements_remain_enabled_for_all_text": True,
            },
            "identifier_splitter": self.identifier_splitter.manifest,
            "runtime_token_vocabulary": self._runtime_vocabulary_manifest(),
            "file_name_token_split_rule_count": len(self.file_name_token_split_rules),
            "file_name_informative_split_token_count": len(
                self.file_name_informative_split_tokens
            ),
            "allow_heuristic_file_splits": self.allow_heuristic_file_splits,
            "runtime_text_steps": self._runtime_text_steps(),
            "paper_equivalence_notes": self._paper_equivalence_notes(),
        }

    @property
    def manifest(self) -> dict[str, Any]:
        rule_files = [
            *(self.rules_dir / name for name in TOPIC_RULE_APPLICATION_ORDER),
            self.rules_dir / "replace.csv",
            self.rules_dir / "abbr.csv",
        ]
        list_files = [
            self.lists_dir / "Contarctions.txt",
            self.lists_dir / "SE_abbr.txt",
            self.lists_dir / "SE_topics.txt",
            self.lists_dir / "Slangs.txt",
            self.lists_dir / "Time_date.txt",
            self.lists_dir / "File names_confusing_tokens.txt",
        ]
        return {
            "schema_version": "topic_data_preparation_v3",
            "upstream_repository_url": UPSTREAM_RECOMMENDER_REPOSITORY_URL,
            "rules_dir": str(self.rules_dir),
            "lists_dir": str(self.lists_dir),
            "topic_rule_order": list(TOPIC_RULE_APPLICATION_ORDER),
            "skipped_topic_rule_names": sorted(self.skipped_topic_rule_names),
            "source_token_limits": self.source_token_limits,
            "token_limit_text_window": {
                "enabled": True,
                "minimum_characters": TOKEN_LIMIT_MIN_TEXT_WINDOW_CHARS,
                "characters_per_requested_token": TOKEN_LIMIT_TEXT_WINDOW_MULTIPLIER,
            },
            "phrase_replacement_policy": {
                "enabled_for_text_up_to_characters": TEXT_PHRASE_REPLACEMENT_MAX_CHARS,
                "token_replacements_remain_enabled_for_all_text": True,
            },
            "identifier_splitter": self.identifier_splitter.manifest,
            "runtime_token_vocabulary": self._runtime_vocabulary_manifest(),
            "file_name_token_split_rule_count": len(self.file_name_token_split_rules),
            "file_name_informative_split_token_count": len(
                self.file_name_informative_split_tokens
            ),
            "file_name_informative_split_tokens_source": (
                self._file_name_informative_split_tokens_source()
            ),
            "allow_heuristic_file_splits": self.allow_heuristic_file_splits,
            "runtime_text_steps": self._runtime_text_steps(),
            "paper_equivalence_notes": self._paper_equivalence_notes(),
            "rule_file_checksums": {
                path.name: _sha256_file(path) for path in rule_files if path.exists()
            },
            "list_file_checksums": {
                path.name: _sha256_file(path) for path in list_files if path.exists()
            },
        }

    def prepare_topics(self, topics: str | Iterable[str]) -> tuple[str, ...]:
        current = _split_topics(topics)
        for _, mapping in self.topic_rules[:-2]:
            current = self._apply_topic_rule(current, mapping)
        current = self._apply_replace_aliases(current)
        current = self._apply_aggregate_rules(current)
        for _, mapping in self.topic_rules[-2:]:
            current = self._apply_topic_rule(current, mapping)
        return tuple(sorted(topic for topic in current if topic))

    def prepare_training_text(self, text: Any) -> str:
        return " ".join(self.prepare_text_tokens(text))

    def prepare_repository_text(
        self,
        *,
        name_parts: Iterable[Any],
        description: Any = "",
        readme: Any = "",
        wiki: Any = "",
        file_paths: Iterable[Any] = (),
    ) -> PreparedRepositoryText:
        name_tokens = self.prepare_project_name_tokens(name_parts, NAME_TOKEN_LIMIT)
        description_tokens = self.prepare_text_tokens(description, DESCRIPTION_TOKEN_LIMIT)
        readme_tokens = self.prepare_text_tokens(readme, README_TOKEN_LIMIT)
        wiki_tokens = self.prepare_text_tokens(wiki, WIKI_TOKEN_LIMIT)
        file_tokens = self.prepare_file_name_tokens(file_paths, FILE_TOKEN_LIMIT)
        token_parts = [
            *name_tokens,
            *description_tokens,
            *readme_tokens,
            *wiki_tokens,
            *file_tokens,
        ]
        return PreparedRepositoryText(
            text=" ".join(token_parts),
            token_counts={
                "name": len(name_tokens),
                "description": len(description_tokens),
                "readme": len(readme_tokens),
                "wiki": len(wiki_tokens),
                "file_names": len(file_tokens),
                "total": len(token_parts),
            },
            data_preparation=self.compact_manifest,
        )

    def prepare_project_name_tokens(
        self,
        name_parts: Iterable[Any],
        limit: int | None = NAME_TOKEN_LIMIT,
    ) -> list[str]:
        tokens: list[str] = []
        for part in name_parts:
            for token in self.prepare_text_tokens(part, file_name=True):
                tokens.append(token)
                if limit is not None and len(tokens) >= limit:
                    return tokens
        return tokens

    def prepare_file_name_tokens(
        self,
        paths: Iterable[Any],
        limit: int | None = FILE_TOKEN_LIMIT,
    ) -> list[str]:
        tokens: list[str] = []
        for normalized in _sort_paths_root_first(paths):
            path_tokens_seen: set[str] = set()
            pieces = [part for part in normalized.split("/") if part]
            for piece in pieces:
                for token in self.prepare_text_tokens(piece, file_name=True):
                    if token in path_tokens_seen:
                        continue
                    path_tokens_seen.add(token)
                    tokens.append(token)
                    if limit is not None and len(tokens) >= limit:
                        return tokens
        return tokens

    def prepare_text_tokens(
        self,
        text: Any,
        limit: int | None = None,
        *,
        file_name: bool = False,
    ) -> list[str]:
        raw_text = _window_text_for_token_limit(str(text or ""), limit)
        cache_key: tuple[str, bool] | None = None
        if limit is None and len(raw_text) <= 256:
            cache_key = (raw_text, file_name)
            cached = self._text_token_cache.get(cache_key)
            if cached is not None:
                return list(cached)
        normalized = self._normalize_text(raw_text, file_name=file_name)
        tokens: list[str] = []
        for match in _LETTER_TOKEN_RE.finditer(normalized):
            raw_token = match.group(0)
            if len(raw_token) < 2:
                continue
            replacements = self.text_replacements.get(raw_token)
            replacement_tokens = replacements.split() if replacements else [raw_token]
            for replacement in replacement_tokens:
                for token in _iter_replacement_tokens(replacement):
                    for candidate in self._expand_file_name_token(
                        token,
                        file_name=file_name,
                    ):
                        if len(candidate) < 2:
                            continue
                        if candidate in _nltk_english_stop_words():
                            continue
                        candidate = _lemmatize_token(candidate)
                        if len(candidate) < 2:
                            continue
                        if file_name and candidate in self.file_name_stop_tokens:
                            continue
                        if not self._is_allowed_runtime_token(candidate, file_name=file_name):
                            continue
                        tokens.append(candidate)
                        if limit is not None and len(tokens) >= limit:
                            return tokens
        if cache_key is not None and len(self._text_token_cache) < 100000:
            self._text_token_cache[cache_key] = tuple(tokens)
        return tokens

    def _normalize_text(self, text: Any, *, file_name: bool = False) -> str:
        normalized = str(text or "")
        normalized = unicodedata.normalize("NFKD", normalized)
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
        if not file_name and len(normalized) <= TEXT_PHRASE_REPLACEMENT_MAX_CHARS:
            normalized = self._apply_text_phrase_replacements(normalized)
        if not file_name:
            normalized = self._remove_abstract_text_concepts(normalized)
        normalized = self.identifier_splitter.split_text(normalized)
        normalized = normalized.lower()
        return self._remove_non_letter_token_content(normalized)

    def _apply_text_phrase_replacements(self, text: str) -> str:
        pattern = self.text_phrase_replacement_pattern
        if pattern is None:
            return text

        def replace(match: re.Match[str]) -> str:
            matched = match.group(0).lower()
            return f" {self.text_phrase_replacements.get(matched, matched)} "

        return pattern.sub(replace, text)

    @staticmethod
    def _remove_abstract_text_concepts(text: str) -> str:
        normalized = _FENCED_CODE_RE.sub(" ", text)
        normalized = _INLINE_CODE_RE.sub(" ", normalized)
        normalized = _MARKDOWN_LINK_RE.sub(
            lambda match: f" {match.group(1) or ' '} ",
            normalized,
        )
        normalized = _URL_RE.sub(" ", normalized)
        normalized = _EMAIL_RE.sub(" ", normalized)
        normalized = _USERNAME_RE.sub(" ", normalized)
        normalized = _DATE_TIME_RE.sub(" ", normalized)
        return normalized

    @staticmethod
    def _remove_non_letter_token_content(text: str) -> str:
        normalized = re.sub(r"[_./\\-]+", " ", text)
        normalized = re.sub(r"\d+", " ", normalized)
        return re.sub(r"[^a-z\s]+", " ", normalized)

    def _is_allowed_runtime_token(self, token: str, *, file_name: bool) -> bool:
        allowed = self.allowed_file_name_tokens if file_name else self.allowed_text_tokens
        return allowed is None or token in allowed

    def _expand_file_name_token(self, token: str, *, file_name: bool) -> list[str]:
        if not file_name:
            return [token]
        cached = self._file_name_token_expansion_cache.get(token)
        if cached is not None:
            return list(cached)
        if token in self.file_name_token_split_rules:
            expanded = list(self.file_name_token_split_rules[token])
        elif token in self.file_name_stop_tokens:
            expanded = [token]
        else:
            expanded = _split_informative_file_token(
                token,
                self.file_name_informative_split_sequence,
                allowed_tokens=self.allowed_file_name_tokens,
            )
        if len(self._file_name_token_expansion_cache) < 100000:
            self._file_name_token_expansion_cache[token] = tuple(expanded)
        return expanded

    def _runtime_vocabulary_manifest(self) -> dict[str, Any]:
        return {
            "filter_enabled": (
                self.allowed_text_tokens is not None
                or self.allowed_file_name_tokens is not None
            ),
            "source": self.runtime_vocabulary_source,
            "allowed_text_token_count": (
                len(self.allowed_text_tokens)
                if self.allowed_text_tokens is not None
                else None
            ),
            "allowed_file_name_token_count": (
                len(self.allowed_file_name_tokens)
                if self.allowed_file_name_tokens is not None
                else None
            ),
            "artifact_schema_version": self.preprocessing_artifacts.get("schema_version"),
            "frequency_policy": self.preprocessing_artifacts.get("frequency_policy"),
            "source_token_policy": self.preprocessing_artifacts.get("source_token_policy"),
        }

    def _runtime_text_steps(self) -> list[str]:
        steps = [
            "normalize to ASCII before text replacements",
            "replace SE/CS abbreviations and aliases from vendored data-preparation lists",
            "remove URLs, email addresses, usernames, markdown/code snippets, dates, and times",
            (
                "split identifiers with "
                f"{self.identifier_splitter.name} "
                f"({self.identifier_splitter.implementation})"
            ),
            "lowercase after replacement, abstract-text removal, and identifier splitting",
            "remove punctuation, digits, and non-letter token content before tokenization",
            "remove English stop words with nltk.corpus.stopwords.words('english') before lemmatization",
            "lemmatize with nltk.stem.WordNetLemmatizer",
            "do not apply stemming; preserve WordNet lemmatized token forms",
            "process repository owner/name with file-name token rules and the file-name vocabulary",
            "split compact file-name tokens with vendored source rules and model-bundle informative split tokens",
            "remove confusing file-name tokens for repository file-list inputs",
        ]
        if self.allow_heuristic_file_splits:
            steps.append("allow heuristic informative file-name splits when no model artifact is available")
        if self.allowed_text_tokens is not None or self.allowed_file_name_tokens is not None:
            steps.append(
                "filter runtime tokens using preprocessing artifacts saved in the model bundle"
            )
        steps.append("sort repository file paths root-first before applying the file-name token cap")
        return steps

    def _paper_equivalence_notes(self) -> list[str]:
        notes = [
            "Runtime preprocessing mirrors the paper's published steps using vendored rule/list files where available.",
        ]
        if self.identifier_splitter.paper_equivalent:
            notes.append("Identifier splitting uses Spiral/Ronin, matching the paper's splitter family.")
        else:
            notes.append(
                "Identifier splitting is using the documented regex fallback; set "
                "POST_PROCESSING_TOPIC_CLASSIFICATION_IDENTIFIER_SPLITTER=spiral and "
                "install Spiral for paper-equivalent splitting."
            )
        if self.allowed_text_tokens is not None or self.allowed_file_name_tokens is not None:
            notes.append(
                "Runtime corpus-level token filtering applies the paper's frequency thresholds from model-bundle preprocessing artifacts."
            )
        else:
            notes.append(
                "Runtime corpus-level token filtering is disabled because no model-bundle preprocessing artifacts were provided."
            )
        if self.preprocessing_artifacts.get("raw_corpus_frequency_artifacts_available") is True:
            notes.append(
                "Model-bundle preprocessing artifacts include separate raw text and file-name token-frequency filters."
            )
        elif self.preprocessing_artifacts.get("separate_source_frequency_filters_available") is False:
            notes.append(
                "The released train/test CSVs do not expose separate raw text/file-name frequency lists; "
                "the saved runtime vocabulary is derived from prepared CSV text using the paper's "
                "text threshold of 50 and file-name threshold of 20."
            )
        notes.append(
            "Compact repository file-name tokens are split with vendored generated rules first; "
            "rules marked -2 preserve the original token, matching the source notebook behavior."
        )
        if self.file_name_informative_split_tokens:
            notes.append(
                "Additional informative file-name split tokens come from model-bundle preprocessing artifacts."
            )
        elif self.allow_heuristic_file_splits:
            notes.append(
                "Heuristic informative file-name split tokens are enabled; this is not exact paper equivalence."
            )
        else:
            notes.append(
                "Heuristic informative file-name split tokens are disabled; exact all-name split artifacts were not released."
            )
        return notes

    def _load_text_replacements(self) -> dict[str, str]:
        replacements: dict[str, str] = {}
        for file_name in (
            "Contarctions.txt",
            "Slangs.txt",
            "SE_abbr.txt",
            "SE_topics.txt",
            "Time_date.txt",
        ):
            assignments = _literal_assignments(self.lists_dir / file_name)
            for value in assignments.values():
                replacements.update(_string_mapping(value))
        return replacements

    def _load_file_name_stop_tokens(self) -> set[str]:
        assignments = _literal_assignments(self.lists_dir / "File names_confusing_tokens.txt")
        tokens: set[str] = set()
        for value in assignments.values():
            tokens.update(_string_set(value))
        tokens.difference_update(DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS)
        return tokens

    def _load_file_name_token_split_rules(self) -> dict[str, tuple[str, ...]]:
        split_rules: dict[str, tuple[str, ...]] = {}
        for file_name in (
            "contains_top_topics.csv",
            "contains_selected_topics.csv",
            "split_dash_topics.csv",
        ):
            for source, replacements in _load_apply_rule(self.rules_dir / file_name).items():
                if replacements and replacements[0] == "-2":
                    split_rules[source] = (source,)
                    continue
                split_rules[source] = tuple(
                    token
                    for token in replacements
                    if token and token not in {"-1", "-2"}
                )
        return split_rules

    def _load_file_name_informative_split_tokens(self) -> set[str]:
        artifact_tokens = _token_set_from_artifact(
            self.preprocessing_artifacts.get("file_name_informative_split_tokens")
        )
        if artifact_tokens is not None:
            tokens = artifact_tokens
        elif self.allow_heuristic_file_splits:
            tokens = set(DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS)
            if self.allowed_file_name_tokens is not None:
                tokens.update(
                    token
                    for token in self.allowed_file_name_tokens
                    if token in DEFAULT_FILE_INFORMATIVE_SPLIT_TOKENS
                )
        else:
            tokens = set()
        return {token for token in tokens if len(token) >= 2}

    def _file_name_informative_split_tokens_source(self) -> str:
        if self.preprocessing_artifacts.get("file_name_informative_split_tokens") is not None:
            return str(
                self.preprocessing_artifacts.get(
                    "file_name_informative_split_tokens_source",
                    "model_bundle",
                )
            )
        if self.allow_heuristic_file_splits:
            return "heuristic_default_tokens"
        return "none"

    @staticmethod
    def _apply_topic_rule(
        topics: Sequence[str],
        mapping: Mapping[str, Sequence[str]],
    ) -> list[str]:
        result: list[str] = []
        for topic in topics:
            replacements = mapping.get(topic)
            if not replacements:
                result.append(topic)
                continue
            first = replacements[0]
            if first == "-1":
                continue
            if first == "-2":
                result.append(topic)
                continue
            result.extend(item for item in replacements if item)
        return sorted(set(result))

    def _apply_replace_aliases(self, topics: Sequence[str]) -> list[str]:
        return sorted(
            set(self.topic_replace_aliases.get(topic, topic) for topic in topics)
        )

    def _apply_aggregate_rules(self, topics: Sequence[str]) -> list[str]:
        remaining = list(topics)
        additions: list[str] = []
        for required, canonical in self.topic_aggregate_rules:
            if set(required) & set(remaining) == set(required):
                additions.append(canonical)
                for topic in set(required):
                    while topic in remaining:
                        remaining.remove(topic)
        return sorted(set([*remaining, *additions]))


def _split_topics(topics: str | Iterable[str]) -> list[str]:
    if isinstance(topics, str):
        values = re.split(r"[,;\s]+", topics)
    else:
        values = [str(topic) for topic in topics]
    return sorted(
        {
            value.strip().lower()
            for value in values
            if value.strip() and value.strip().lower() not in {"none", "nan", "null"}
        }
    )


def _iter_replacement_tokens(replacement: str) -> Iterable[str]:
    raw = str(replacement or "").lower()
    if raw.isalpha():
        yield raw
        return
    normalized = re.sub(r"[_./\\-]+", " ", raw)
    normalized = re.sub(r"\d+", " ", normalized)
    normalized = re.sub(r"[^a-z\s]+", " ", normalized)
    for match in _LETTER_TOKEN_RE.finditer(normalized):
        yield match.group(0)


def _split_informative_file_token(
    token: str,
    informative_tokens: Iterable[str],
    *,
    allowed_tokens: set[str] | None,
) -> list[str]:
    normalized = str(token or "").strip().lower()
    if len(normalized) < 4:
        return [normalized] if normalized else []

    if isinstance(informative_tokens, tuple):
        informative_sequence = informative_tokens
    else:
        informative_sequence = tuple(
            sorted(set(informative_tokens), key=lambda value: (-len(value), value))
        )
    if allowed_tokens is not None:
        recursive_split = _split_allowed_informative_file_token(
            normalized,
            informative_sequence,
            allowed_tokens=allowed_tokens,
        )
        if recursive_split is not None:
            return recursive_split

    split_tokens = _split_informative_file_token_once(
        normalized,
        informative_sequence,
        allowed_tokens=allowed_tokens,
    )
    if split_tokens == [normalized]:
        return split_tokens

    expanded: list[str] = []
    for split_token in split_tokens:
        if split_token == normalized:
            expanded.append(split_token)
            continue
        nested = _split_informative_file_token_once(
            split_token,
            informative_sequence,
            allowed_tokens=allowed_tokens,
        )
        expanded.extend(nested)
    return expanded


def _split_allowed_informative_file_token(
    token: str,
    informative_tokens: Sequence[str],
    *,
    allowed_tokens: set[str],
) -> list[str] | None:
    if token in allowed_tokens:
        return [token]
    for informative in informative_tokens:
        if token == informative or len(informative) < 2 or informative not in allowed_tokens:
            continue
        if token.startswith(informative):
            remainder = token[len(informative):]
            nested = _split_allowed_informative_file_token(
                remainder,
                informative_tokens,
                allowed_tokens=allowed_tokens,
            )
            if nested is not None:
                return [informative, *nested]
        if token.endswith(informative):
            remainder = token[: -len(informative)]
            nested = _split_allowed_informative_file_token(
                remainder,
                informative_tokens,
                allowed_tokens=allowed_tokens,
            )
            if nested is not None:
                return [*nested, informative]
    return None


def _split_informative_file_token_once(
    token: str,
    informative_tokens: Sequence[str],
    *,
    allowed_tokens: set[str] | None,
) -> list[str]:
    for informative in informative_tokens:
        if token == informative or len(informative) < 2:
            continue
        if token.startswith(informative):
            remainder = token[len(informative):]
            if _valid_informative_split((informative, remainder), allowed_tokens):
                return [informative, remainder]
        if token.endswith(informative):
            remainder = token[: -len(informative)]
            if _valid_informative_split((remainder, informative), allowed_tokens):
                return [remainder, informative]
    return [token]


def _valid_informative_split(
    parts: Sequence[str],
    allowed_tokens: set[str] | None,
) -> bool:
    if any(len(part) < 2 for part in parts):
        return False
    if allowed_tokens is None:
        return True
    return all(part in allowed_tokens for part in parts)


@lru_cache(maxsize=200000)
def _lemmatize_token(token: str) -> str:
    """Lemmatize with the same WordNet lemmatizer used by the source notebook."""
    return str(_wordnet_lemmatizer().lemmatize(str(token or "")))


@lru_cache(maxsize=1)
def _wordnet_lemmatizer() -> Any:
    try:
        from nltk.stem import WordNetLemmatizer
    except Exception as exc:  # pragma: no cover - dependency/environment failure.
        raise RuntimeError(
            "Runtime topic preprocessing requires nltk.stem.WordNetLemmatizer "
            "to match topic-classification/data-preparation. Install NLTK."
        ) from exc

    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("watches")
    except LookupError:
        try:
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            lemmatizer.lemmatize("watches")
        except Exception as exc:  # pragma: no cover - dependency/environment failure.
            raise RuntimeError(
                "Runtime topic preprocessing requires the NLTK WordNet corpus "
                "to match topic-classification/data-preparation. Install it with "
                "`python -m nltk.downloader wordnet omw-1.4`."
            ) from exc
    return lemmatizer


def _sort_paths_root_first(paths: Iterable[Any]) -> list[str]:
    normalized_paths = [
        str(path or "").replace("\\", "/").lstrip("/")
        for path in paths
        if str(path or "").strip()
    ]

    def sort_key(path: str) -> tuple[int, str]:
        parts = [part for part in path.split("/") if part]
        return (len(parts), path.lower())

    return sorted(normalized_paths, key=sort_key)


def _runtime_vocabularies_from_artifacts(
    artifacts: Mapping[str, Any],
) -> tuple[set[str] | None, set[str] | None, str]:
    if not artifacts:
        return None, None, "none"
    allowed_text = _token_set_from_artifact(
        artifacts.get("allowed_text_tokens")
        or artifacts.get("text_token_vocabulary")
        or artifacts.get("allowed_runtime_tokens")
        or artifacts.get("token_vocabulary")
    )
    allowed_files = _token_set_from_artifact(
        artifacts.get("allowed_file_name_tokens")
        or artifacts.get("file_name_token_vocabulary")
        or artifacts.get("allowed_runtime_tokens")
        or artifacts.get("token_vocabulary")
    )
    source = str(
        artifacts.get("source")
        or artifacts.get("token_source")
        or artifacts.get("vocabulary_source")
        or "model_bundle"
    )
    return allowed_text, allowed_files, source


def _token_set_from_artifact(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tokens = value.split()
    elif isinstance(value, Mapping):
        tokens = value.keys()
    elif isinstance(value, IterableABC):
        tokens = value
    else:
        return None
    result = {
        str(token).strip().lower()
        for token in tokens
        if str(token).strip()
    }
    return result or None


def get_preprocessor_for_model(
    preprocessing_artifacts: Mapping[str, Any] | None,
    *,
    identifier_splitter_mode: str = "auto",
    allow_heuristic_file_splits: bool = False,
    data_preparation_root: Path | None = None,
    skipped_topic_rule_names: Iterable[str] = SKIPPED_STATIC_LOW_FREQ_RULES,
) -> SoftwareTagDataPreprocessor:
    """Create the runtime preprocessor expected by a trained model bundle."""
    profile_root = Path(data_preparation_root or DEFAULT_GENERATED_DATA_PREPARATION_DIR)
    return SoftwareTagDataPreprocessor(
        rules_dir=profile_root / "rules",
        lists_dir=profile_root / "lists",
        identifier_splitter_mode=identifier_splitter_mode,
        preprocessing_artifacts=preprocessing_artifacts,
        allow_heuristic_file_splits=allow_heuristic_file_splits,
        skipped_topic_rule_names=skipped_topic_rule_names,
    )


@lru_cache(maxsize=1)
def get_default_preprocessor() -> SoftwareTagDataPreprocessor:
    """Return the cached default preprocessor for callers without a model bundle."""
    return SoftwareTagDataPreprocessor(identifier_splitter_mode="spiral")
