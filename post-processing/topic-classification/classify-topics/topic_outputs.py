"""Streaming output writers for repository topic classification.

The classifier can process large curation outputs, so predictions, repository
to PR joins, and filtered repositories are written incrementally as JSONL. The
manifest is written once at the end with aggregate counts and config metadata.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from topic_classifier import TopicPredictionResult
from topic_features import RepositoryFeatures
from topic_loader import RepositoryInputRef


class TopicClassificationOutputWriter:
    """Write repository predictions and repository/PR join maps incrementally."""

    def __init__(self, output_dir: Path, *, run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.repository_topics_path = self.output_dir / "repository_topics.jsonl"
        self.repository_pr_map_path = self.output_dir / "repository_pr_map.jsonl"
        self.filtered_repositories_path = self.output_dir / "filtered_repositories.jsonl"
        self.manifest_path = self.output_dir / "topic_classification_manifest.json"
        self._topics_handle: Any | None = None
        self._pr_map_handle: Any | None = None
        self._filtered_handle: Any | None = None

    def __enter__(self) -> "TopicClassificationOutputWriter":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._topics_handle = self.repository_topics_path.open("w", encoding="utf-8", newline="\n")
        self._pr_map_handle = self.repository_pr_map_path.open("w", encoding="utf-8", newline="\n")
        self._filtered_handle = self.filtered_repositories_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    @property
    def output_paths(self) -> dict[str, str]:
        return {
            "repository_topics": str(self.repository_topics_path),
            "repository_pr_map": str(self.repository_pr_map_path),
            "filtered_repositories": str(self.filtered_repositories_path),
            "manifest": str(self.manifest_path),
        }

    def write_repository_topics(
        self,
        *,
        features: RepositoryFeatures,
        prediction_result: TopicPredictionResult,
        classifier_info: dict[str, Any],
        top_k: int | None,
        prediction_score_threshold: float,
        prediction_retention_policy: str,
        include_raw_predictions: bool,
    ) -> None:
        record: dict[str, Any] = {
            "schema_version": "repository_topic_predictions_v3",
            "run_id": self.run_id,
            "cohort": features.cohort,
            "cohorts": list(features.cohorts),
            "repository_owner": features.repository_owner,
            "repository_name": features.repository_name,
            "repository_full_name": features.repository_full_name,
            "repository_key": features.repository_key,
            "repository_id": features.repository_id,
            "repository_identity_key": features.repository_identity_key,
            "source_ref": features.source_ref,
            "source_commit": features.source_commit,
            "top_k": top_k,
            "prediction_score_threshold": float(prediction_score_threshold),
            "prediction_retention_policy": prediction_retention_policy,
            "predicted_topics": prediction_result.predicted_topic_dicts(),
            "predicted_topic_groups": prediction_result.predicted_topic_group_dicts(),
            "predicted_topic_domains": prediction_result.predicted_topic_domain_dicts(),
            "observed_topics": list(features.observed_topics),
            "classifier": classifier_info,
            "input_stats": features.input_stats,
        }
        self._write_jsonl(self._topics_handle, record)

    def write_repository_pr_map(self, ref: RepositoryInputRef) -> int:
        count = 0
        for pr_ref in ref.pr_record_refs:
            record = {
                "schema_version": "repository_pr_topic_map_v1",
                "run_id": self.run_id,
                "cohort": pr_ref.cohort,
                "cohorts": list(ref.cohorts or (ref.cohort,)),
                "repository_owner": ref.repository_owner,
                "repository_name": ref.repository_name,
                "repository_key": ref.repository_key,
                "repository_id": ref.repository_id,
                "repository_identity_key": ref.repository_identity_key,
                "pr_repository_key": pr_ref.repository_key,
                "pr_repository_id": pr_ref.repository_id,
                "record_format": pr_ref.record_format,
                "record_path": str(pr_ref.path),
                "record_line_number": pr_ref.line_number,
                "pr_source": pr_ref.source,
                "pr_number": pr_ref.pr_number,
                "pr_url": pr_ref.pr_url,
                "jsonl_path": str(pr_ref.path),
                "jsonl_line_number": pr_ref.line_number,
            }
            if pr_ref.record_format == "parquet":
                record["parquet_path"] = str(pr_ref.path)
                record["parquet_row_number"] = pr_ref.line_number
            self._write_jsonl(self._pr_map_handle, record)
            count += 1
        return count

    def write_filtered_repository(
        self,
        *,
        ref: RepositoryInputRef,
        reason: str,
        message: str,
        repository_file_count: int,
        metrics_payload_count: int,
        pr_payload_count: int,
        source_stats: dict[str, Any],
    ) -> None:
        record = {
            "schema_version": "repository_topic_classification_filtered_v1",
            "run_id": self.run_id,
            "cohort": ref.cohort,
            "cohorts": list(ref.cohorts or (ref.cohort,)),
            "repository_owner": ref.repository_owner,
            "repository_name": ref.repository_name,
            "repository_key": ref.repository_key,
            "repository_id": ref.repository_id,
            "repository_identity_key": ref.repository_identity_key,
            "safe_repository_key": ref.safe_repository_key,
            "excluded_stage": "runtime_input_filter",
            "excluded_reason": reason,
            "message": message,
            "repository_file_count": int(repository_file_count),
            "metrics_payload_count": int(metrics_payload_count),
            "pr_payload_count": int(pr_payload_count),
            "pr_record_ref_count": len(ref.pr_record_refs),
            "file_list_path": str(ref.file_list_path) if ref.file_list_path else None,
            "source_stats": source_stats,
        }
        self._write_jsonl(self._filtered_handle, record)

    def write_manifest(self, payload: dict[str, Any]) -> None:
        manifest = {
            "schema_version": "topic_classification_manifest_v1",
            "run_id": self.run_id,
            "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            **payload,
        }
        with self.manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def close(self) -> None:
        if self._topics_handle is not None:
            self._topics_handle.close()
            self._topics_handle = None
        if self._pr_map_handle is not None:
            self._pr_map_handle.close()
            self._pr_map_handle = None
        if self._filtered_handle is not None:
            self._filtered_handle.close()
            self._filtered_handle = None

    def _write_jsonl(self, handle: Any | None, record: dict[str, Any]) -> None:
        if handle is None:
            raise RuntimeError("Output writer is not open.")
        handle.write(json.dumps(record, sort_keys=True) + "\n")
