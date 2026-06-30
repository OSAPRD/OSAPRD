"""Topic-classification topic group loading for analysis domain counts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_deduplication_utility import stable_numeric_id


TOPIC_CONFIDENCE_THRESHOLD = 0.7
TOPIC_GROUP_ORDER = (
    "AI, Data, and Science",
    "Backend, APIs, and Security",
    "Distributed and Embedded Systems",
    "Graphics",
    "Web and Mobile",
)


@dataclass(frozen=True)
class TopicGroupRecord:
    """One repository-to-topic-group association."""

    repository_id: str | None
    repository_key: str | None
    topic_group: str
    topic: str | None = None
    confidence: float | None = None


def normalize_repository_key(value: Any) -> str | None:
    """Return normalized ``owner/name`` repository keys when available."""
    text = str(value or "").strip()
    if not text or "/" not in text:
        return None
    owner, name = text.split("/", 1)
    if not owner.strip() or not name.strip():
        return None
    return f"{owner.strip().lower()}/{name.strip().lower()}"


def resolve_topic_output_dir(path: Path | None) -> Path | None:
    """Resolve either a topic run root or its concrete ``output`` folder."""
    if path is None or not str(path).strip():
        return None
    root = Path(path)
    if (root / "repository_topics.jsonl").exists():
        return root
    output_dir = root / "output"
    if (output_dir / "repository_topics.jsonl").exists():
        return output_dir
    return None


def _repository_id_from_payload(payload: dict[str, Any]) -> str | None:
    """Extract a stable repository id from current or legacy topic payloads."""
    direct = stable_numeric_id(payload.get("repository_id"))
    if direct:
        return direct
    identity = str(payload.get("repository_identity_key") or "").strip()
    prefix = "repository-id:"
    if identity.startswith(prefix):
        return stable_numeric_id(identity[len(prefix) :])
    return None


def _topic_confidence(item: dict[str, Any]) -> float | None:
    """Return the best available confidence score for a topic item."""
    for field_name in ("score_max", "top_topic_score", "score_sum"):
        value = item.get(field_name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _topic_label(item: dict[str, Any]) -> str | None:
    """Return the most specific topic label exposed by a topic item."""
    for field_name in ("topic", "topic_name", "top_topic", "label"):
        value = str(item.get(field_name) or "").strip()
        if value:
            return value
    return None


def _topic_groups_from_payload(
    payload: dict[str, Any],
) -> list[tuple[str, str | None, float | None]]:
    """Extract ordered, de-duplicated topic groups from one JSONL payload."""
    groups: list[tuple[str, str | None, float | None]] = []
    seen: set[str] = set()
    domain_items = payload.get("predicted_topic_domains")
    if domain_items is None:
        domain_items = payload.get("predicted_topic_groups")
    for item in domain_items or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("topic_domain") or item.get("topic_group") or "").strip()
        if not label or label.lower() in seen:
            continue
        seen.add(label.lower())
        groups.append((label, _topic_label(item), _topic_confidence(item)))
    return groups


def canonical_topic_group(
    value: Any,
    *,
    allowed_topic_groups: tuple[str, ...] = TOPIC_GROUP_ORDER,
) -> str | None:
    """Return the canonical topic-group label when it is in the allowed set."""
    label = str(value or "").strip()
    if not label:
        return None
    allowed = {topic_group.casefold(): topic_group for topic_group in allowed_topic_groups}
    return allowed.get(label.casefold())


def filter_topic_group_records_by_confidence(
    records: list[TopicGroupRecord],
    *,
    minimum_confidence: float = TOPIC_CONFIDENCE_THRESHOLD,
    allowed_topic_groups: tuple[str, ...] | None = TOPIC_GROUP_ORDER,
) -> list[TopicGroupRecord]:
    """Keep topic records with recognized groups and confidence above threshold."""
    filtered: list[TopicGroupRecord] = []
    for record in records:
        if record.confidence is None or float(record.confidence) < minimum_confidence:
            continue
        topic_group = str(record.topic_group).strip()
        if allowed_topic_groups is not None:
            canonical = canonical_topic_group(
                topic_group,
                allowed_topic_groups=allowed_topic_groups,
            )
            if canonical is None:
                continue
            topic_group = canonical
        filtered.append(
            TopicGroupRecord(
                repository_id=record.repository_id,
                repository_key=record.repository_key,
                topic_group=topic_group,
                topic=record.topic,
                confidence=float(record.confidence),
            )
        )
    return filtered


def load_topic_group_records(path: Path | None) -> list[TopicGroupRecord]:
    """Load repository topic groups from topic-classification JSONL output."""
    output_dir = resolve_topic_output_dir(path)
    if output_dir is None:
        return []

    records: list[TopicGroupRecord] = []
    seen: set[tuple[str, str]] = set()
    topics_path = output_dir / "repository_topics.jsonl"
    with topics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            repository_id = _repository_id_from_payload(payload)
            repository_key = (
                normalize_repository_key(payload.get("repository_key"))
                or normalize_repository_key(payload.get("repository_full_name"))
            )
            if repository_id:
                repository_identity = f"id:{repository_id}"
            elif repository_key:
                repository_identity = f"key:{repository_key}"
            else:
                continue
            for topic_group, topic, confidence in _topic_groups_from_payload(payload):
                key = (repository_identity, topic_group.lower())
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    TopicGroupRecord(
                        repository_id=repository_id,
                        repository_key=repository_key,
                        topic_group=topic_group,
                        topic=topic,
                        confidence=confidence,
                    )
                )
    return records
