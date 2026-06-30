"""Model-bundle loading and top-k topic prediction helpers.

The training stage exports the vectorizer, classifier, ordered labels, and
preprocessing metadata as one bundle. This module loads that bundle and exposes
the small prediction DTOs used by the streaming classifier and output writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TopicPrediction:
    """One retained topic prediction with raw and post-filter ranks."""

    rank: int
    topic: str
    score: float
    raw_rank: int
    topic_domain: str | None = None
    topic_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "rank": self.rank,
            "topic": self.topic,
            "score": self.score,
            "raw_rank": self.raw_rank,
        }
        if self.topic_domain:
            payload["topic_domain"] = self.topic_domain
        if self.topic_group:
            payload["topic_group"] = self.topic_group
        return payload


@dataclass(frozen=True)
class TopicPredictionResult:
    """Retained predictions plus optional raw predictions for diagnostics."""

    predicted_topics: tuple[TopicPrediction, ...]
    raw_predictions: tuple[TopicPrediction, ...] = ()

    def predicted_topic_dicts(self) -> list[dict[str, Any]]:
        return [prediction.to_dict() for prediction in self.predicted_topics]

    def raw_prediction_dicts(self) -> list[dict[str, Any]]:
        return [prediction.to_dict() for prediction in self.raw_predictions]

    def with_topic_groups(
        self,
        topic_groups: Mapping[str, str],
        *,
        default_group: str = "Other",
    ) -> "TopicPredictionResult":
        normalized_groups = {
            str(topic).strip().lower(): str(group).strip()
            for topic, group in topic_groups.items()
            if str(topic).strip() and str(group).strip()
        }
        fallback = str(default_group or "Other").strip() or "Other"

        return TopicPredictionResult(
            predicted_topics=tuple(
                TopicPrediction(
                    rank=prediction.rank,
                    topic=prediction.topic,
                    score=prediction.score,
                    raw_rank=prediction.raw_rank,
                    topic_domain=prediction.topic_domain,
                    topic_group=normalized_groups.get(
                        prediction.topic.strip().lower(),
                        fallback,
                    ),
                )
                for prediction in self.predicted_topics
            ),
            raw_predictions=self.raw_predictions,
        )

    def with_topic_domains(
        self,
        topic_domains: Mapping[str, str],
    ) -> "TopicPredictionResult":
        normalized_domains = {
            str(topic).strip().lower(): str(domain).strip()
            for topic, domain in topic_domains.items()
            if str(topic).strip() and str(domain).strip()
        }

        return TopicPredictionResult(
            predicted_topics=tuple(
                TopicPrediction(
                    rank=prediction.rank,
                    topic=prediction.topic,
                    score=prediction.score,
                    raw_rank=prediction.raw_rank,
                    topic_domain=normalized_domains.get(
                        prediction.topic.strip().lower(),
                    ),
                    topic_group=normalized_domains.get(
                        prediction.topic.strip().lower(),
                    ),
                )
                for prediction in self.predicted_topics
            ),
            raw_predictions=self.raw_predictions,
        )

    def predicted_topic_group_dicts(self) -> list[dict[str, Any]]:
        return self._predicted_domain_aggregate_dicts(
            source_field="topic_group",
            output_field="topic_group",
        )

    def predicted_topic_domain_dicts(self) -> list[dict[str, Any]]:
        return self._predicted_domain_aggregate_dicts(
            source_field="topic_domain",
            output_field="topic_domain",
        )

    def _predicted_domain_aggregate_dicts(
        self,
        *,
        source_field: str,
        output_field: str,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for prediction in self.predicted_topics:
            domain_value = getattr(prediction, source_field)
            if not domain_value:
                continue
            group = str(domain_value)
            entry = grouped.setdefault(
                group,
                {
                    output_field: group,
                    "topic_count": 0,
                    "topics": [],
                    "top_topic": prediction.topic,
                    "top_topic_score": prediction.score,
                    "score_sum": 0.0,
                    "score_max": prediction.score,
                    "_first_topic_rank": prediction.rank,
                },
            )
            entry["topic_count"] += 1
            entry["topics"].append(prediction.topic)
            entry["score_sum"] = float(entry["score_sum"]) + prediction.score
            if prediction.score > float(entry["score_max"]):
                entry["score_max"] = prediction.score
            if prediction.rank < int(entry["_first_topic_rank"]):
                entry["_first_topic_rank"] = prediction.rank
                entry["top_topic"] = prediction.topic
                entry["top_topic_score"] = prediction.score

        records: list[dict[str, Any]] = []
        for rank, entry in enumerate(
            sorted(grouped.values(), key=lambda item: int(item["_first_topic_rank"])),
            start=1,
        ):
            record = dict(entry)
            record["rank"] = rank
            record.pop("_first_topic_rank", None)
            records.append(record)
        return records


class TopicModelClassifier:
    """Thin inference wrapper around a Part 1 topic model bundle."""

    def __init__(
        self,
        *,
        vectorizer: Any,
        classifier: Any,
        topic_labels: Iterable[str],
        preprocessing_artifacts: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        labels = tuple(str(label).strip() for label in topic_labels if str(label).strip())
        if not labels:
            raise ValueError("Topic model bundle does not contain topic labels.")
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.topic_labels = labels
        self.preprocessing_artifacts = dict(preprocessing_artifacts or {})
        self.metadata = dict(metadata or {})

    @property
    def model_info(self) -> dict[str, Any]:
        training_config = self.metadata.get("training_config")
        if not isinstance(training_config, dict):
            training_config = {}
        return {
            "schema_version": self.metadata.get("schema_version"),
            "created_at_utc": self.metadata.get("created_at_utc"),
            "model_name": training_config.get("model_name"),
            "algorithm": training_config.get("algorithm"),
            "label_count": len(self.topic_labels),
            "preprocessing_artifact_schema": self.preprocessing_artifacts.get("schema_version"),
            "runtime_token_vocabulary_size": self.preprocessing_artifacts.get(
                "allowed_runtime_token_count"
            ),
        }

    def predict(
        self,
        text: str,
        *,
        top_k: int = 5,
        excluded_topics: Iterable[str] = (),
        retained_topics: Iterable[str] = (),
        include_raw_predictions: bool = False,
        score_threshold: float | None = None,
    ) -> TopicPredictionResult:
        scores = self._score_text(text)
        excluded = {str(topic).strip().lower() for topic in excluded_topics if str(topic).strip()}
        retained = {str(topic).strip().lower() for topic in retained_topics if str(topic).strip()}
        ranked = [
            TopicPrediction(
                rank=raw_rank,
                topic=topic,
                score=score,
                raw_rank=raw_rank,
            )
            for raw_rank, (topic, score) in enumerate(
                sorted(
                    zip(self.topic_labels, scores, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                ),
                start=1,
            )
        ]
        filtered_ranked: list[TopicPrediction] = []
        for prediction in ranked:
            if prediction.topic.lower() in excluded:
                continue
            if retained and prediction.topic.lower() not in retained:
                continue
            if score_threshold is not None and prediction.score <= float(score_threshold):
                continue
            filtered_ranked.append(
                TopicPrediction(
                    rank=len(filtered_ranked) + 1,
                    topic=prediction.topic,
                    score=prediction.score,
                    raw_rank=prediction.raw_rank,
                )
            )
            if score_threshold is None and len(filtered_ranked) >= max(1, int(top_k)):
                break
        raw_predictions = tuple(ranked) if include_raw_predictions else ()
        return TopicPredictionResult(
            predicted_topics=tuple(filtered_ranked),
            raw_predictions=raw_predictions,
        )

    def _score_text(self, text: str) -> tuple[float, ...]:
        transformed = self.vectorizer.transform([str(text or "")])
        probabilities = self.classifier.predict_proba(transformed)
        row = _first_probability_row(probabilities, expected_count=len(self.topic_labels))
        if len(row) != len(self.topic_labels):
            raise ValueError(
                "Model returned " f"{len(row)} scores for {len(self.topic_labels)} topic labels."
            )
        return tuple(float(value) for value in row)


def _first_probability_row(probabilities: Any, *, expected_count: int) -> list[float]:
    """Return the first prediction row from sklearn- or test-double-style outputs."""
    if hasattr(probabilities, "tolist"):
        converted = probabilities.tolist()
    else:
        converted = probabilities
    if not isinstance(converted, list):
        raise ValueError("Classifier predict_proba returned an unsupported result.")
    if not converted:
        raise ValueError("Classifier predict_proba returned no scores.")
    if all(isinstance(value, (int, float)) for value in converted):
        return [float(value) for value in converted]
    first = converted[0]
    if hasattr(first, "tolist"):
        first = first.tolist()
    if not isinstance(first, list):
        raise ValueError("Classifier predict_proba returned an unsupported row.")
    if (
        len(first) == 2
        and expected_count != 2
        and all(isinstance(item, list) and len(item) == 2 for item in converted)
    ):
        return [float(item[1]) for item in converted]
    return [float(value) for value in first]


def load_topic_model_bundle(path: Path) -> TopicModelClassifier:
    """Load a Part 1 topic model bundle from disk."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Topic model bundle not found: {bundle_path}")
    try:
        import joblib
    except Exception as exc:  # pragma: no cover - environment-dependent.
        raise RuntimeError(
            "Unable to import joblib for topic-model loading. "
            "Install post-processing requirements first."
        ) from exc
    try:
        bundle = joblib.load(bundle_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load topic model bundle {bundle_path}: {exc}") from exc
    if not isinstance(bundle, dict):
        raise ValueError(f"Topic model bundle must be a dict: {bundle_path}")
    try:
        vectorizer = bundle["vectorizer"]
        classifier = bundle["classifier"]
        topic_labels = bundle["topic_labels"]
    except KeyError as exc:
        raise ValueError(f"Topic model bundle is missing key: {exc}") from exc
    preprocessing_artifacts = bundle.get("preprocessing_artifacts")
    if preprocessing_artifacts is not None and not isinstance(preprocessing_artifacts, dict):
        raise ValueError("Topic model bundle preprocessing_artifacts must be a dict.")
    metadata = bundle.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}
    return TopicModelClassifier(
        vectorizer=vectorizer,
        classifier=classifier,
        topic_labels=topic_labels,
        preprocessing_artifacts=preprocessing_artifacts,
        metadata=metadata,
    )
