"""Generate a GitHub-topics-derived data-preparation rule profile.

The generated profile preserves the shape expected by Izadi's original
preprocessing code while deriving topic mappings from the local
``github-topics`` catalog. The original vendored files remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


DATA_PREPARATION_DIR = Path(__file__).resolve().parent
TOPIC_CLASSIFICATION_DIR = DATA_PREPARATION_DIR.parent
DEFAULT_TOPICS_ROOT = TOPIC_CLASSIFICATION_DIR / "github-topics"
DEFAULT_SOURCE_DATA_PREPARATION_ROOT = DATA_PREPARATION_DIR
DEFAULT_OUTPUT_ROOT = DATA_PREPARATION_DIR / "generated" / "github-topics"

RULE_FILES = (
    "abbr.csv",
    "abstract.csv",
    "contains_selected_topics.csv",
    "contains_top_topics.csv",
    "contractions.csv",
    "delete.csv",
    "integer_topics.csv",
    "remove_lemmatize_topic.csv",
    "remove_plural_topics.csv",
    "remove_stemmed_topic.csv",
    "remove_stopwords_topic.csv",
    "replace.csv",
    "replace_alias.csv",
    "split_dash_topics.csv",
    "topics_contains_number.csv",
    "topics_contains_version.csv",
)
SKIPPED_RULE_FILES = ("low_freq_topics.csv",)
LIST_FILES = (
    "Contarctions.txt",
    "File names_confusing_tokens.txt",
    "SE_abbr.txt",
    "SE_topics.txt",
    "Slangs.txt",
    "Time_date.txt",
)

STOPWORD_RULE_FILES = {"remove_stopwords_topic.csv"}
CATALOG_SPLIT_RE = re.compile(r"[^a-z0-9]+")
VERSION_RE = re.compile(r"(?:^|[-_])v?\d+(?:[-_.]\d+)*$|v\d+")
CURATED_SE_ABBR_ADDITIONS = {
    "ai": "artificial intelligence",
    "api": "application programming interface",
    "ar": "augmented reality",
    "cd": "continuous delivery",
    "ci": "continuous integration",
    "cli": "command line interface",
    "cpu": "central processing unit",
    "css": "cascading style sheets",
    "gpu": "graphics processing unit",
    "gui": "graphical user interface",
    "html": "hypertext markup language",
    "http": "hypertext transfer protocol",
    "https": "hypertext transfer protocol secure",
    "ide": "integrated development environment",
    "io": "input output",
    "iot": "internet of things",
    "json": "javascript object notation",
    "jwt": "json web token",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "nosql": "not only sql",
    "oauth": "open authorization",
    "os": "operating system",
    "orm": "object relational mapping",
    "rest": "representational state transfer",
    "rpc": "remote procedure call",
    "sdk": "software development kit",
    "sql": "structured query language",
    "ui": "user interface",
    "uri": "uniform resource identifier",
    "url": "uniform resource locator",
    "ux": "user experience",
    "vr": "virtual reality",
    "xml": "extensible markup language",
    "yaml": "yaml markup language",
}


@dataclass(frozen=True)
class TopicCatalogEntry:
    """One topic entry parsed from the local GitHub topics catalog."""

    topic: str
    display_name: str
    aliases: tuple[str, ...]
    related: tuple[str, ...]
    source_path: Path
    description: str


@dataclass(frozen=True)
class AliasChoice:
    """Canonical alias mapping decision with conflict diagnostics."""

    term: str
    canonical: str
    conflict: bool
    sources: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class RuleCandidate:
    """Potential rule row before legacy/generated conflict resolution."""

    rule_file: str
    body: tuple[str, ...]
    source_kind: str
    source_row: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedLegacyRule:
    """Normalized legacy rule row plus audit action and reason."""

    candidate: RuleCandidate | None
    action: str
    reason: str


def main() -> None:
    """Run the rule-profile generator as a standalone script."""
    args = parse_args()
    manifest = generate_github_topic_rules(
        topics_root=args.topics_root,
        source_data_preparation_root=args.source_data_preparation_root,
        output_root=args.output_root,
    )
    print(
        "[post-processing/topic-rules] Generated "
        f"{manifest['catalog_topic_count']} topics and "
        f"{manifest['alias_map_row_count']} alias rows at {args.output_root}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the rule-profile generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics-root", type=Path, default=DEFAULT_TOPICS_ROOT)
    parser.add_argument(
        "--source-data-preparation-root",
        type=Path,
        default=DEFAULT_SOURCE_DATA_PREPARATION_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def generate_github_topic_rules(
    *,
    topics_root: Path = DEFAULT_TOPICS_ROOT,
    source_data_preparation_root: Path = DEFAULT_SOURCE_DATA_PREPARATION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Generate a complete rule/list profile from a GitHub topic catalog."""
    topics_root = Path(topics_root)
    source_root = Path(source_data_preparation_root)
    output_root = Path(output_root)
    entries = read_github_topic_catalog(topics_root)
    if not entries:
        raise ValueError(f"No GitHub topic entries found under {topics_root}")

    alias_choices, conflicts = build_alias_choices(entries)
    rules_root = output_root / "rules"
    lists_root = output_root / "lists"
    rules_root.mkdir(parents=True, exist_ok=True)
    lists_root.mkdir(parents=True, exist_ok=True)
    for skipped_file in SKIPPED_RULE_FILES:
        stale = rules_root / skipped_file
        if stale.exists():
            stale.unlink()

    alias_map = {choice.term: choice.canonical for choice in alias_choices}
    catalog_topics = {entry.topic for entry in entries}
    generated_rule_rows = build_rule_rows(entries, alias_choices, conflicts)
    merged_rule_rows, rule_merge_stats, legacy_audit_rows = merge_legacy_and_generated_rules(
        source_root=source_root,
        generated_rule_rows=generated_rule_rows,
        alias_map=alias_map,
        catalog_topics=catalog_topics,
    )
    rule_file_counts: dict[str, int] = {}
    rule_file_modes: dict[str, str] = {}
    for rule_file in RULE_FILES:
        target = rules_root / rule_file
        rows = merged_rule_rows.get(rule_file, [])
        write_csv_rows(target, rows)
        rule_file_counts[rule_file] = len(rows)
        if rule_file in STOPWORD_RULE_FILES:
            rule_file_modes[rule_file] = "copied_legacy_generic_rule"
        else:
            rule_file_modes[rule_file] = "merged_legacy_izadi_and_github_topics"

    list_file_modes: dict[str, str] = {}
    list_file_counts: dict[str, int] = {}
    list_file_additions: dict[str, int] = {}
    for list_file in LIST_FILES:
        target = lists_root / list_file
        additions = copy_or_generate_list_file(
            source_root / "lists" / list_file,
            target,
            list_file=list_file,
            alias_choices=alias_choices,
        )
        list_file_additions[list_file] = additions
        list_file_modes[list_file] = (
            "copied_plus_catalog_synonyms"
            if additions
            else "copied_unchanged_generic_list"
        )
        list_file_counts[list_file] = count_text_lines(target)

    alias_map_path = output_root / "github_topic_alias_map.csv"
    write_alias_map(alias_map_path, alias_choices)
    merge_audit_path = output_root / "github_topic_legacy_rule_merge_audit.csv"
    write_rule_merge_audit(merge_audit_path, legacy_audit_rows)
    catalog_path = output_root / "github_topic_catalog.json"
    write_json(
        catalog_path,
        [
            {
                "topic": entry.topic,
                "display_name": entry.display_name,
                "aliases": list(entry.aliases),
                "related": list(entry.related),
                "source_path": portable_path(entry.source_path),
                "description": entry.description,
            }
            for entry in entries
        ],
    )
    manifest = {
        "schema_version": "github_topic_rule_profile_v1",
        "created_at_utc": utc_now_z(),
        "topics_root": portable_path(topics_root),
        "source_data_preparation_root": portable_path(source_root),
        "output_root": portable_path(output_root),
        "rules_root": portable_path(rules_root),
        "lists_root": portable_path(lists_root),
        "catalog_path": portable_path(catalog_path),
        "alias_map_path": portable_path(alias_map_path),
        "legacy_rule_merge_audit_path": portable_path(merge_audit_path),
        "catalog_topic_count": len(entries),
        "catalog_alias_count": sum(len(entry.aliases) for entry in entries),
        "alias_map_row_count": len(alias_choices),
        "alias_conflict_count": len(conflicts),
        "alias_conflicts": conflicts,
        "rule_files": {
            rule_file: {
                "mode": rule_file_modes[rule_file],
                "row_count": rule_file_counts[rule_file],
                **rule_merge_stats[rule_file],
            }
            for rule_file in RULE_FILES
        },
        "rule_merge_summary": summarize_rule_merge(rule_merge_stats),
        "list_files": {
            list_file: {
                "mode": list_file_modes[list_file],
                "line_count": list_file_counts[list_file],
                "added_catalog_synonym_count": list_file_additions[list_file],
            }
            for list_file in LIST_FILES
        },
        "list_additions_by_file": list_file_additions,
        "skipped_rule_files": list(SKIPPED_RULE_FILES),
        "low_freq_topics_policy": (
            "not generated; create-training-data writes dataset-local low_freq_topics.csv"
        ),
        "checksums": checksums_for_profile(output_root),
    }
    write_json(output_root / "github_topic_rules_manifest.json", manifest)
    return manifest


def read_github_topic_catalog(topics_root: Path) -> list[TopicCatalogEntry]:
    """Read all topic `index.md` files under a GitHub topics checkout."""
    entries: list[TopicCatalogEntry] = []
    for index_path in sorted(Path(topics_root).glob("*/index.md")):
        metadata, body = parse_topic_markdown(index_path)
        topic = normalize_topic_slug(metadata.get("topic") or index_path.parent.name)
        if not topic:
            continue
        display_name = clean_scalar(metadata.get("display_name") or topic)
        aliases = tuple(
            alias
            for alias in split_front_matter_list(metadata.get("aliases", ""))
            if normalize_topic_slug(alias) and normalize_topic_slug(alias) != topic
        )
        related = tuple(split_front_matter_list(metadata.get("related", "")))
        entries.append(
            TopicCatalogEntry(
                topic=topic,
                display_name=display_name,
                aliases=tuple(dict.fromkeys(aliases)),
                related=related,
                source_path=index_path,
                description=body.strip(),
            )
        )
    return entries


def parse_topic_markdown(path: Path) -> tuple[dict[str, str], str]:
    """Parse front matter and body text from one topic markdown file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    front_matter_lines: list[str] = []
    body_start = len(lines)
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = index + 1
            break
        front_matter_lines.append(line)
    return parse_front_matter(front_matter_lines), "\n".join(lines[body_start:])


def parse_front_matter(lines: Iterable[str]) -> dict[str, str]:
    """Parse the simple YAML-like front matter used by GitHub topics."""
    result: dict[str, str] = {}
    current_key: str | None = None
    for line in lines:
        if not line.strip():
            continue
        if line[:1].isspace() and current_key:
            result[current_key] = f"{result[current_key]} {line.strip()}".strip()
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        current_key = key.strip()
        result[current_key] = clean_scalar(value)
    return result


def split_front_matter_list(value: str) -> list[str]:
    """Split comma-separated front-matter values into clean strings."""
    return [
        clean_scalar(item)
        for item in str(value or "").split(",")
        if clean_scalar(item)
    ]


def build_alias_choices(
    entries: list[TopicCatalogEntry],
) -> tuple[list[AliasChoice], list[dict[str, Any]]]:
    """Resolve canonical alias choices and record ambiguous catalog terms."""
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        for term, source_kind in iter_catalog_terms(entry):
            normalized = normalize_topic_slug(term)
            if not normalized:
                continue
            candidates[normalized].append(
                {
                    "term": normalized,
                    "canonical": entry.topic,
                    "source_kind": source_kind,
                    "source_path": str(entry.source_path),
                }
            )

    choices: list[AliasChoice] = []
    conflicts: list[dict[str, Any]] = []
    for term in sorted(candidates):
        sources = candidates[term]
        canonical_values = sorted({source["canonical"] for source in sources})
        conflict = len(canonical_values) > 1
        chosen = choose_canonical(term, sources)
        if conflict:
            conflicts.append(
                {
                    "term": term,
                    "chosen_canonical": chosen,
                    "candidate_canonicals": canonical_values,
                    "sources": [
                        {
                            **source,
                            "source_path": portable_path(Path(source["source_path"])),
                        }
                        for source in sources
                    ],
                }
            )
        choices.append(
            AliasChoice(
                term=term,
                canonical=chosen,
                conflict=conflict,
                sources=tuple(sources),
            )
        )
    return choices, conflicts


def iter_catalog_terms(entry: TopicCatalogEntry) -> Iterable[tuple[str, str]]:
    """Yield canonical and alias-like terms from one catalog entry."""
    yield entry.topic, "topic"
    compact = compact_topic(entry.topic)
    if compact and compact != entry.topic:
        yield compact, "topic_compact_variant"
    for alias in entry.aliases:
        normalized = normalize_topic_slug(alias)
        if normalized:
            yield normalized, "alias"
        alias_compact = compact_topic(normalized)
        if alias_compact and alias_compact != normalized:
            yield alias_compact, "alias_compact_variant"


def choose_canonical(term: str, sources: list[dict[str, str]]) -> str:
    """Choose the canonical topic for an alias conflict deterministically."""
    exact_self_matches = sorted(
        source["canonical"]
        for source in sources
        if source["canonical"] == term and source["source_kind"] == "topic"
    )
    if exact_self_matches:
        return exact_self_matches[0]
    topic_matches = sorted(
        source["canonical"]
        for source in sources
        if source["source_kind"] == "topic"
    )
    if topic_matches:
        return topic_matches[0]
    return sorted(source["canonical"] for source in sources)[0]


def build_rule_rows(
    entries: list[TopicCatalogEntry],
    alias_choices: list[AliasChoice],
    conflicts: list[dict[str, Any]],
) -> dict[str, list[list[str]]]:
    """Build generated rule rows before merging with the base rule profile."""
    alias_map = {choice.term: choice.canonical for choice in alias_choices}
    noncanonical_aliases = {
        choice.term: choice.canonical
        for choice in alias_choices
        if choice.term != choice.canonical
    }
    rows: dict[str, list[list[str]]] = {rule_file: [] for rule_file in RULE_FILES}
    rows["replace.csv"] = indexed_rows(
        [canonical, term]
        for term, canonical in sorted(noncanonical_aliases.items())
    )
    rows["replace_alias.csv"] = list(rows["replace.csv"])

    short_alias_rows = [
        [term, canonical]
        for term, canonical in sorted(noncanonical_aliases.items())
        if len(term) <= 6 and "-" not in term
    ]
    rows["contractions.csv"] = indexed_rows(short_alias_rows)
    rows["abstract.csv"] = indexed_rows(
        [term, *split_topic_parts(canonical)]
        for term, canonical in sorted(noncanonical_aliases.items())
        if len(term) <= 8 and len(split_topic_parts(canonical)) >= 1
    )

    aggregate_rules = set()
    for entry in entries:
        add_aggregate_rule(aggregate_rules, entry.topic, split_topic_parts(entry.topic))
        for alias in entry.aliases:
            add_aggregate_rule(
                aggregate_rules,
                entry.topic,
                split_topic_parts(normalize_topic_slug(alias)),
            )
    rows["abbr.csv"] = indexed_rows(
        [canonical, *parts]
        for canonical, parts in sorted(aggregate_rules)
    )

    split_dash = []
    contains_rows = []
    number_rows = []
    version_rows = []
    plural_rows = []
    lemmatize_rows = []
    integer_rows = []

    for choice in alias_choices:
        term = choice.term
        canonical = choice.canonical
        if "-" in term:
            split_dash.append([term, "-2" if term == canonical else canonical])
        elif term != canonical:
            contains_rows.append([term, canonical])
        if any(character.isdigit() for character in term):
            target = "-2" if term == canonical else canonical
            number_rows.append([term, target])
            if VERSION_RE.search(term):
                version_rows.append([term, target])
            if re.fullmatch(r"[0-9][a-z0-9-]*", term):
                integer_rows.append([term, target])

    for term, canonical in sorted(alias_map.items()):
        for source in plural_variants(term):
            if source != canonical:
                plural_rows.append([source, canonical])
                lemmatize_rows.append([source, canonical])

    conflict_delete_rows = [
        [conflict["term"], "-1"]
        for conflict in conflicts
        if conflict.get("term")
    ]

    rows["split_dash_topics.csv"] = indexed_rows(unique_rows(split_dash))
    rows["contains_top_topics.csv"] = indexed_rows(unique_rows(contains_rows))
    rows["contains_selected_topics.csv"] = indexed_rows(unique_rows(contains_rows))
    rows["topics_contains_number.csv"] = indexed_rows(unique_rows(number_rows))
    rows["topics_contains_version.csv"] = indexed_rows(unique_rows(version_rows))
    rows["remove_plural_topics.csv"] = indexed_rows(unique_rows(plural_rows))
    rows["remove_lemmatize_topic.csv"] = indexed_rows(unique_rows(lemmatize_rows))
    rows["remove_stemmed_topic.csv"] = indexed_rows(unique_rows(lemmatize_rows))
    rows["delete.csv"] = indexed_rows(unique_rows(conflict_delete_rows))
    rows["integer_topics.csv"] = indexed_rows(unique_rows(integer_rows))
    return rows


def merge_legacy_and_generated_rules(
    *,
    source_root: Path,
    generated_rule_rows: dict[str, list[list[str]]],
    alias_map: dict[str, str],
    catalog_topics: set[str],
) -> tuple[dict[str, list[list[str]]], dict[str, dict[str, int]], list[dict[str, Any]]]:
    """Merge base and generated rule rows, returning rows, stats, and audit data."""
    merged_rows: dict[str, list[list[str]]] = {}
    stats: dict[str, dict[str, int]] = {}
    audit_rows: list[dict[str, Any]] = []
    for rule_file in RULE_FILES:
        generated_candidates = [
            RuleCandidate(
                rule_file=rule_file,
                body=tuple(row[1:]),
                source_kind="github_topics",
                source_row=tuple(row),
            )
            for row in generated_rule_rows.get(rule_file, [])
            if len(row) > 1
        ]
        legacy_rows = read_csv_rows(source_root / "rules" / rule_file)
        legacy_candidates: list[RuleCandidate] = []
        dropped_non_catalog = 0
        malformed = 0
        for legacy_row in legacy_rows:
            normalized = normalize_legacy_rule_row(
                rule_file=rule_file,
                row=legacy_row,
                alias_map=alias_map,
                catalog_topics=catalog_topics,
            )
            audit_row = {
                "rule_file": rule_file,
                "source_kind": "legacy_izadi",
                "source_row": json.dumps(list(legacy_row), ensure_ascii=True),
                "normalized_row": (
                    json.dumps(list(normalized.candidate.body), ensure_ascii=True)
                    if normalized.candidate
                    else "[]"
                ),
                "action": normalized.action,
                "reason": normalized.reason,
            }
            audit_rows.append(audit_row)
            if normalized.candidate is None:
                if normalized.action == "dropped_non_catalog_output":
                    dropped_non_catalog += 1
                else:
                    malformed += 1
                continue
            legacy_candidates.append(normalized.candidate)

        active_candidates, merge_audit, merge_counts = resolve_rule_candidates(
            rule_file=rule_file,
            candidates=[*generated_candidates, *legacy_candidates],
        )
        audit_rows.extend(merge_audit)
        merged_rows[rule_file] = indexed_rows([list(candidate.body) for candidate in active_candidates])
        stats[rule_file] = {
            "legacy_row_count": len(legacy_rows),
            "generated_row_count": len(generated_candidates),
            "merged_active_row_count": len(merged_rows[rule_file]),
            "deduped_row_count": merge_counts["deduped"],
            "conflict_count": merge_counts["conflicts"],
            "dropped_non_catalog_legacy_row_count": dropped_non_catalog,
            "malformed_legacy_row_count": malformed,
        }
    return merged_rows, stats, audit_rows


def normalize_legacy_rule_row(
    *,
    rule_file: str,
    row: tuple[str, ...],
    alias_map: dict[str, str],
    catalog_topics: set[str],
) -> NormalizedLegacyRule:
    """Normalize one base rule row into the generated profile vocabulary."""
    if len(row) < 3:
        return NormalizedLegacyRule(None, "dropped_malformed", "row_has_fewer_than_three_columns")
    body = tuple(value.strip() for value in row[1:] if value.strip())
    if len(body) < 2:
        return NormalizedLegacyRule(None, "dropped_malformed", "row_has_no_active_mapping")

    if rule_file in {"replace.csv", "replace_alias.csv", "abbr.csv"}:
        canonical = resolve_catalog_output(body[0], alias_map, catalog_topics)
        if canonical is None or canonical in {"-1", "-2"}:
            return NormalizedLegacyRule(
                None,
                "dropped_non_catalog_output",
                f"canonical_output_not_in_github_topics:{body[0]}",
            )
        source_values = tuple(normalize_rule_source(value) for value in body[1:] if normalize_rule_source(value))
        if not source_values:
            return NormalizedLegacyRule(None, "dropped_malformed", "row_has_no_source_terms")
        normalized_body = (canonical, *source_values)
        return NormalizedLegacyRule(
            RuleCandidate(rule_file, normalized_body, "legacy_izadi", row),
            "candidate",
            "legacy_output_mapped_to_github_topic",
        )

    source = normalize_rule_source(body[0])
    if not source:
        return NormalizedLegacyRule(None, "dropped_malformed", "row_has_empty_source")
    normalized_outputs: list[str] = []
    for output in body[1:]:
        resolved = resolve_catalog_output(output, alias_map, catalog_topics)
        if resolved is None:
            return NormalizedLegacyRule(
                None,
                "dropped_non_catalog_output",
                f"output_not_in_github_topics:{output}",
            )
        normalized_outputs.append(resolved)
    if not normalized_outputs:
        return NormalizedLegacyRule(None, "dropped_malformed", "row_has_no_outputs")
    normalized_body = (source, *normalized_outputs)
    return NormalizedLegacyRule(
        RuleCandidate(rule_file, normalized_body, "legacy_izadi", row),
        "candidate",
        "legacy_outputs_mapped_to_github_topics_or_special_directives",
    )


def resolve_catalog_output(
    value: str,
    alias_map: dict[str, str],
    catalog_topics: set[str],
) -> str | None:
    """Resolve a rule output token to a canonical catalog topic when possible."""
    output = normalize_rule_source(value)
    if output in {"-1", "-2"}:
        return output
    normalized = normalize_topic_slug(output)
    compact = compact_topic(normalized)
    if output in catalog_topics:
        return output
    if normalized in catalog_topics:
        return normalized
    if output in alias_map:
        return alias_map[output]
    if normalized in alias_map:
        return alias_map[normalized]
    if compact in alias_map:
        return alias_map[compact]
    return None


def resolve_rule_candidates(
    *,
    rule_file: str,
    candidates: list[RuleCandidate],
) -> tuple[list[RuleCandidate], list[dict[str, Any]], dict[str, int]]:
    """Deduplicate rule candidates and audit conflict resolution decisions."""
    unique: dict[tuple[str, ...], RuleCandidate] = {}
    deduped = 0
    audit_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.body in unique:
            deduped += 1
            if candidate.source_kind == "legacy_izadi":
                audit_rows.append(rule_audit_row(candidate, "deduped", "exact_duplicate_row"))
            continue
        unique[candidate.body] = candidate

    grouped: dict[tuple[str, ...], list[RuleCandidate]] = defaultdict(list)
    for candidate in unique.values():
        grouped[rule_conflict_key(rule_file, candidate.body)].append(candidate)

    active: list[RuleCandidate] = []
    conflicts = 0
    for group in grouped.values():
        if len(group) == 1:
            active.append(group[0])
            if group[0].source_kind == "legacy_izadi":
                audit_rows.append(rule_audit_row(group[0], "included", "no_conflict"))
            continue
        conflicts += len(group) - 1
        winner = choose_rule_candidate(rule_file, group)
        active.append(winner)
        for candidate in group:
            if candidate.body == winner.body:
                if candidate.source_kind == "legacy_izadi":
                    audit_rows.append(rule_audit_row(candidate, "included", "conflict_winner"))
                continue
            audit_rows.append(
                rule_audit_row(
                    candidate,
                    "conflict_lost",
                    "preferred_catalog_backed_generated_or_lexicographic_mapping",
                )
            )
    return sorted(active, key=lambda candidate: candidate.body), audit_rows, {
        "deduped": deduped,
        "conflicts": conflicts,
    }


def choose_rule_candidate(rule_file: str, candidates: list[RuleCandidate]) -> RuleCandidate:
    """Select the preferred row for a rule-conflict group."""
    return sorted(
        candidates,
        key=lambda candidate: rule_candidate_priority(rule_file, candidate),
    )[0]


def rule_candidate_priority(rule_file: str, candidate: RuleCandidate) -> tuple[int, int, str]:
    """Return a deterministic priority tuple for resolving duplicate rules."""
    body = candidate.body
    exact_self = rule_candidate_exact_self_match(rule_file, body)
    source_priority = 0 if candidate.source_kind == "github_topics" else 1
    exact_priority = 0 if exact_self else 1
    return (exact_priority, source_priority, "\x1f".join(body))


def rule_candidate_exact_self_match(rule_file: str, body: tuple[str, ...]) -> bool:
    """Return whether a candidate row preserves an exact self-mapping."""
    if rule_file in {"replace.csv", "replace_alias.csv"} and len(body) >= 2:
        return body[0] == body[1]
    if rule_file == "abbr.csv":
        return False
    return len(body) >= 2 and body[1] == body[0]


def rule_conflict_key(rule_file: str, body: tuple[str, ...]) -> tuple[str, ...]:
    """Build the key used to group conflicting rule candidates."""
    if rule_file in {"replace.csv", "replace_alias.csv"} and len(body) >= 2:
        return (rule_file, body[1])
    if rule_file == "abbr.csv" and len(body) >= 2:
        return (rule_file, *body[1:])
    if body:
        return (rule_file, body[0])
    return (rule_file,)


def rule_audit_row(candidate: RuleCandidate, action: str, reason: str) -> dict[str, Any]:
    """Convert a rule merge decision into a CSV/JSON audit row."""
    return {
        "rule_file": candidate.rule_file,
        "source_kind": candidate.source_kind,
        "source_row": json.dumps(list(candidate.source_row), ensure_ascii=True),
        "normalized_row": json.dumps(list(candidate.body), ensure_ascii=True),
        "action": action,
        "reason": reason,
    }


def summarize_rule_merge(stats: dict[str, dict[str, int]]) -> dict[str, int]:
    """Aggregate per-file rule merge counters."""
    keys = (
        "legacy_row_count",
        "generated_row_count",
        "merged_active_row_count",
        "deduped_row_count",
        "conflict_count",
        "dropped_non_catalog_legacy_row_count",
        "malformed_legacy_row_count",
    )
    return {key: sum(rule_stats[key] for rule_stats in stats.values()) for key in keys}


def add_aggregate_rule(
    rules: set[tuple[str, tuple[str, ...]]],
    canonical: str,
    parts: list[str],
) -> None:
    """Append an aggregate-topic rule while avoiding weak one-token rules."""
    useful_parts = [part for part in parts if len(part) >= 2]
    if len(useful_parts) >= 2:
        rules.add((canonical, tuple(useful_parts)))


def indexed_rows(rows: Iterable[list[str]]) -> list[list[str]]:
    """Add one-based row ids to generated CSV rule bodies."""
    return [
        [str(index), *row]
        for index, row in enumerate(unique_rows(rows), start=1)
    ]


def unique_rows(rows: Iterable[list[str]]) -> list[list[str]]:
    """Deduplicate rows while preserving deterministic sorted output."""
    seen: set[tuple[str, ...]] = set()
    result: list[list[str]] = []
    for row in rows:
        normalized = tuple(str(value).strip() for value in row if str(value).strip())
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        result.append(list(normalized))
    return sorted(result)


def plural_variants(term: str) -> list[str]:
    """Return simple plural variants used by generated normalization rules."""
    variants: list[str] = []
    if term.endswith("y") and len(term) > 2:
        variants.append(f"{term[:-1]}ies")
    if not term.endswith("s"):
        variants.append(f"{term}s")
    if term.endswith("s") and len(term) > 3:
        variants.append(term[:-1])
    return variants


def split_topic_parts(value: str) -> list[str]:
    """Split a topic slug into meaningful alphanumeric parts."""
    return [
        part
        for part in CATALOG_SPLIT_RE.split(normalize_topic_slug(value))
        if part
    ]


def normalize_topic_slug(value: Any) -> str:
    """Normalize arbitrary text into the GitHub topic slug style."""
    text = clean_scalar(value).lower()
    text = text.replace("+", "plus").replace("#", "sharp")
    text = CATALOG_SPLIT_RE.sub("-", text)
    return re.sub(r"-+", "-", text).strip("-")


def compact_topic(value: str) -> str:
    """Remove separators from a normalized topic slug."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def clean_scalar(value: Any) -> str:
    """Trim a scalar value and unwrap matching quote characters."""
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def normalize_rule_source(value: Any) -> str:
    """Normalize a base-rule source token before catalog resolution."""
    return clean_scalar(value).lower()


def read_csv_rows(path: Path) -> list[tuple[str, ...]]:
    """Read non-empty CSV rows from a required rule file."""
    if not path.exists():
        raise FileNotFoundError(f"Required source rule file not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [
            tuple(str(value).strip() for value in row)
            for row in csv.reader(handle)
            if any(str(value).strip() for value in row)
        ]


def write_csv_rows(path: Path, rows: list[list[str]]) -> None:
    """Write generated rule rows using stable UTF-8 CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def write_alias_map(path: Path, choices: list[AliasChoice]) -> None:
    """Write the generated alias-to-canonical topic map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "raw_term",
                "cleaned_term",
                "canonical_topic",
                "source_kind",
                "source_path",
                "conflict",
            ],
        )
        writer.writeheader()
        for choice in choices:
            for source in choice.sources:
                writer.writerow(
                    {
                        "raw_term": source["term"],
                        "cleaned_term": choice.term,
                        "canonical_topic": choice.canonical,
                        "source_kind": source["source_kind"],
                        "source_path": portable_path(Path(source["source_path"])),
                        "conflict": choice.conflict,
                    }
                )


def write_rule_merge_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the base/generated rule merge audit CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rule_file",
                "source_kind",
                "source_row",
                "normalized_row",
                "action",
                "reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_required_file(source: Path, target: Path) -> dict[str, Any]:
    """Copy a required support file and return manifest metadata."""
    if not source.exists():
        raise FileNotFoundError(f"Required source file not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"source": portable_path(source), "target": portable_path(target)}


def copy_or_generate_list_file(
    source: Path,
    target: Path,
    *,
    list_file: str,
    alias_choices: list[AliasChoice],
) -> int:
    """Copy a list file and add catalog-derived synonyms when appropriate."""
    if list_file == "SE_abbr.txt":
        assignments = _literal_assignments_for_generator(source)
        existing: dict[str, str] = {}
        for value in assignments.values():
            if isinstance(value, dict):
                existing.update(
                    {
                        normalize_rule_source(key): normalize_rule_source(mapped)
                        for key, mapped in value.items()
                        if normalize_rule_source(key) and normalize_rule_source(mapped)
                    }
                )
        additions = {
            key: value
            for key, value in CURATED_SE_ABBR_ADDITIONS.items()
            if key not in existing
        }
        merged = {**existing, **additions}
        write_python_mapping(target, "abrv_CS_common", merged)
        return len(additions)
    if list_file != "SE_topics.txt":
        copy_required_file(source, target)
        return 0
    assignments = _literal_assignments_for_generator(source)
    existing: dict[str, str] = {}
    for value in assignments.values():
        if isinstance(value, dict):
            existing.update(
                {
                    normalize_rule_source(key): normalize_topic_slug(mapped)
                    for key, mapped in value.items()
                    if normalize_rule_source(key) and normalize_topic_slug(mapped)
                }
            )
    additions = safe_catalog_text_synonym_additions(alias_choices, existing)
    merged = {**existing, **additions}
    write_python_mapping(target, "topics_list", merged)
    return len(additions)


def write_python_mapping(path: Path, variable_name: str, mapping: dict[str, str]) -> None:
    """Write a Python dictionary assignment used by legacy preprocessing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{variable_name} = {{"]
    for index, key in enumerate(sorted(mapping)):
        suffix = "," if index < len(mapping) - 1 else ""
        lines.append(f"    {key!r}: {mapping[key]!r}{suffix}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_catalog_text_synonym_additions(
    alias_choices: list[AliasChoice],
    existing: dict[str, str],
) -> dict[str, str]:
    """Return conservative text synonym additions derived from catalog aliases."""
    unsafe = {
        "a",
        "an",
        "as",
        "at",
        "be",
        "by",
        "do",
        "go",
        "he",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "no",
        "of",
        "on",
        "or",
        "r",
        "so",
        "to",
        "up",
        "us",
        "we",
    }
    additions: dict[str, str] = {}
    for choice in alias_choices:
        if choice.term == choice.canonical:
            continue
        source = normalize_rule_source(choice.term)
        if not source or source in existing or source in unsafe:
            continue
        if len(source) < 2 or len(source) > 40:
            continue
        if re.fullmatch(r"\d+", source):
            continue
        compact_canonical = compact_topic(choice.canonical)
        safe_phrase_alias = bool(re.search(r"[^a-z0-9]", source))
        safe_compact_alias = source == compact_canonical and source != choice.canonical
        safe_js_alias = (
            source.endswith("js")
            and len(source) > 2
            and source[:-2] == compact_canonical
        )
        safe_lang_alias = (
            source.endswith("lang")
            and len(source) > 4
            and source[:-4] == compact_canonical
        )
        safe_short_multiword_abbreviation = (
            2 <= len(source) <= 6
            and "-" in choice.canonical
            and source.isalnum()
        )
        if not (
            safe_phrase_alias
            or safe_compact_alias
            or safe_js_alias
            or safe_lang_alias
            or safe_short_multiword_abbreviation
        ):
            continue
        additions[source] = choice.canonical
    return additions


def _literal_assignments_for_generator(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required source list file not found: {path}")
    import ast

    text = path.read_text(encoding="utf-8", errors="replace")
    module = ast.parse(text)
    assignments: dict[str, Any] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        value = ast.literal_eval(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value
    return assignments


def count_csv_rows(path: Path) -> int:
    """Count CSV rows in a generated rule file."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return sum(1 for row in csv.reader(handle) if row)


def count_text_lines(path: Path) -> int:
    """Count lines in a generated list file."""
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def checksums_for_profile(output_root: Path) -> dict[str, str]:
    """Return SHA-256 checksums for generated rules, lists, and manifests."""
    checksums: dict[str, str] = {}
    for path in sorted(output_root.rglob("*")):
        if path.name == "github_topic_rules_manifest.json":
            continue
        if path.is_file():
            checksums[str(path.relative_to(output_root)).replace("\\", "/")] = sha256_file(path)
    return checksums


def sha256_file(path: Path) -> str:
    """Hash a generated artifact in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable indentation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def portable_path(path: Path) -> str:
    """Return a path string stable enough for generated manifests."""
    resolved = Path(path).resolve()
    for base in (TOPIC_CLASSIFICATION_DIR, DATA_PREPARATION_DIR, Path.cwd()):
        try:
            return str(resolved.relative_to(base.resolve())).replace("\\", "/")
        except ValueError:
            continue
    return str(path).replace("\\", "/")


def utc_now_z() -> str:
    """Return the current UTC time in manifest-friendly ISO format."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    main()
