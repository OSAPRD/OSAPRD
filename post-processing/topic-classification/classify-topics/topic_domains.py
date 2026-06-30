"""Domain mapping validation for runtime topic classification.

Topic domains group model labels into reader-facing areas. Validation is strict
so classification runs fail early if the mapping drifts from the generated
GitHub topic catalog or the filtered-topic list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOPIC_DIR = Path(__file__).resolve().parent
TOPIC_CLASSIFICATION_DIR = TOPIC_DIR.parent
DEFAULT_TOPIC_DOMAINS_PATH = TOPIC_DIR / "topic_domains.json"
DEFAULT_FILTERED_TOPICS_PATH = TOPIC_CLASSIFICATION_DIR / "sample-training-data" / "filtered_topics.json"
DEFAULT_GITHUB_TOPICS_ROOT = TOPIC_CLASSIFICATION_DIR / "github-topics"
CANONICAL_TOPIC_DOMAINS = (
    "AI, Data, and Science",
    "Web and Mobile",
    "Backend, APIs, and Security",
    "Graphics",
    "Distributed and Embedded Systems",
)


@dataclass(frozen=True)
class TopicDomainMapping:
    """Validated topic-to-domain mapping plus manifest-ready diagnostics."""

    mapping_path: Path
    domains: tuple[str, ...]
    topic_domains: dict[str, str]
    filtered_topics: tuple[str, ...]
    filtered_topics_present: tuple[str, ...]
    retained_topics: tuple[str, ...]
    catalog_topic_count: int
    filtered_topic_count: int
    retained_topic_count: int
    manifest: dict[str, Any]


def normalize_topic_slug(value: Any) -> str:
    """Normalize a topic string to the slug form used across mappings."""
    return str(value or "").strip().lower()


def load_topic_domain_mapping(
    path: Path,
    *,
    topic_labels: Iterable[str],
) -> TopicDomainMapping:
    """Load and validate the runtime topic-to-domain mapping."""
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(
            "Topic domain mapping file not found: "
            f"{mapping_path}. Generate or commit classify-topics/topic_domains.json."
        )

    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Topic domain mapping must be a JSON object: {mapping_path}")

    domains = tuple(str(domain).strip() for domain in payload.get("domains") or ())
    if set(domains) != set(CANONICAL_TOPIC_DOMAINS):
        raise ValueError(
            "Topic domain mapping domains must match the five canonical domains. "
            f"Expected={CANONICAL_TOPIC_DOMAINS}; got={domains}"
        )

    github_topics_root = _resolve_mapping_path(
        payload.get("github_topics_root"),
        mapping_path=mapping_path,
        default=DEFAULT_GITHUB_TOPICS_ROOT,
    )
    filtered_topics_path = _resolve_mapping_path(
        payload.get("filtered_topics_path"),
        mapping_path=mapping_path,
        default=DEFAULT_FILTERED_TOPICS_PATH,
    )
    catalog_topics = _load_github_topic_slugs(github_topics_root)
    filtered_topics = _load_filtered_topics_json(filtered_topics_path)
    filtered_topics_present = tuple(sorted(catalog_topics & set(filtered_topics)))
    retained_topics = tuple(sorted(catalog_topics - set(filtered_topics_present)))

    raw_topic_domains = payload.get("topic_domains")
    if not isinstance(raw_topic_domains, dict):
        raise ValueError(f"topic_domains must be an object in {mapping_path}")
    topic_domains = {
        normalize_topic_slug(topic): str(domain).strip()
        for topic, domain in raw_topic_domains.items()
        if normalize_topic_slug(topic) and str(domain).strip()
    }
    unknown_domains = sorted(set(topic_domains.values()) - set(domains))
    if unknown_domains:
        raise ValueError(
            f"Topic domain mapping uses unknown domains: {unknown_domains}"
        )

    expected_topics = set(retained_topics)
    mapped_topics = set(topic_domains)
    missing_topics = sorted(expected_topics - mapped_topics)
    filtered_mapped_topics = sorted(set(filtered_topics_present) & mapped_topics)
    unknown_topics = sorted(mapped_topics - expected_topics)
    if missing_topics or filtered_mapped_topics or unknown_topics:
        raise ValueError(
            "Topic domain mapping must cover exactly github-topics minus filtered topics. "
            f"Missing={missing_topics[:20]} count={len(missing_topics)}; "
            f"filtered_mapped={filtered_mapped_topics[:20]} count={len(filtered_mapped_topics)}; "
            f"unknown={unknown_topics[:20]} count={len(unknown_topics)}"
        )

    label_topics = {normalize_topic_slug(label) for label in topic_labels if normalize_topic_slug(label)}
    mapped_model_labels = sorted(label_topics & mapped_topics)
    missing_model_labels = sorted(label_topics - mapped_topics)

    domain_counts: dict[str, int] = {domain: 0 for domain in domains}
    for domain in topic_domains.values():
        domain_counts[domain] += 1

    manifest = {
        "enabled": True,
        "path": str(mapping_path),
        "schema_version": payload.get("schema_version"),
        "domains": list(domains),
        "domain_counts": domain_counts,
        "filtered_topics_path": str(filtered_topics_path),
        "github_topics_root": str(github_topics_root),
        "catalog_topic_count": len(catalog_topics),
        "filtered_topic_count": len(filtered_topics),
        "filtered_topics_present_count": len(filtered_topics_present),
        "retained_topic_count": len(retained_topics),
        "topic_domain_count": len(topic_domains),
        "model_label_count": len(label_topics),
        "model_label_domain_mapped_count": len(mapped_model_labels),
        "model_label_unmapped_count": len(missing_model_labels),
        "model_label_unmapped_sample": missing_model_labels[:50],
        "unmapped_model_label_policy": "ignored_before_threshold_and_domain_output",
        "application_order": "score_all_topics_then_filter_out_topics_then_filter_to_domain_mapped_topics_then_apply_threshold_then_map_domains",
    }
    return TopicDomainMapping(
        mapping_path=mapping_path,
        domains=domains,
        topic_domains=topic_domains,
        filtered_topics=tuple(sorted(filtered_topics)),
        filtered_topics_present=filtered_topics_present,
        retained_topics=retained_topics,
        catalog_topic_count=len(catalog_topics),
        filtered_topic_count=len(filtered_topics),
        retained_topic_count=len(retained_topics),
        manifest=manifest,
    )


def _resolve_mapping_path(value: Any, *, mapping_path: Path, default: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path(default)
    path = Path(text)
    if not path.is_absolute():
        path = mapping_path.parent / path
    return path.resolve()


def _load_github_topic_slugs(root: Path) -> set[str]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"GitHub topics root not found: {root}")
    topics: set[str] = set()
    for path in sorted(root.glob("*/index.md")):
        topic = _front_matter_value(path, "topic") or path.parent.name
        normalized = normalize_topic_slug(topic)
        if normalized:
            topics.add(normalized)
    if not topics:
        raise ValueError(f"GitHub topics root did not contain any topic index files: {root}")
    return topics


def _front_matter_value(path: Path, field_name: str) -> str | None:
    in_front_matter = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "---":
            if not in_front_matter:
                in_front_matter = True
                continue
            break
        if not in_front_matter or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip() == field_name:
            return value.strip().strip('"').strip("'")
    return None


def _load_filtered_topics_json(path: Path) -> tuple[str, ...]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Filtered topics JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    topics: set[str] = set()

    def add_values(value: Any) -> None:
        if isinstance(value, str):
            normalized = normalize_topic_slug(value)
            if normalized:
                topics.add(normalized)
        elif isinstance(value, list):
            for item in value:
                add_values(item)
        elif isinstance(value, dict):
            for item in value.values():
                add_values(item)

    if isinstance(payload, dict):
        for key in ("filtered_topics", "topics", "programming_language_topics"):
            add_values(payload.get(key))
        add_values(payload.get("categories"))
    elif isinstance(payload, list):
        add_values(payload)
    else:
        raise ValueError(f"Unsupported filtered topics JSON shape: {path}")

    if not topics:
        raise ValueError(f"Filtered topics JSON did not define any topics: {path}")
    return tuple(sorted(topics))
