"""Command line entrypoint for curation-data publishing.

The stage reads local curation output, optional topic-classification output, and
optional longitudinal-refactoring summaries. It writes a local public parquet
package and can upload that prepared package to a Hugging Face dataset repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


UPLOAD_DIR = Path(__file__).resolve().parent
CONFIG_DIR = UPLOAD_DIR / "config"
REPO_ROOT = UPLOAD_DIR.parents[1]
UTILITY_DIR = UPLOAD_DIR.parents[0] / "utility"

# The directory name contains a hyphen, so this stage is run by script path. Add
# the local stage, config, repository, and shared post-processing utility roots
# explicitly before importing sibling modules.
for candidate in (REPO_ROOT, UPLOAD_DIR, CONFIG_DIR, UTILITY_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from hf_curation_uploader import HFCurationUploadPipeline  # noqa: E402
from settings import (  # noqa: E402
    COMMAND_ALL,
    COMMAND_PREPARE,
    COMMAND_UPLOAD,
    COMMANDS,
    UploadCurationSettings,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI for local preparation and optional upload."""
    parser = argparse.ArgumentParser(
        description="Prepare and optionally upload curation parquet data.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=COMMANDS,
        default=None,
        help="Run 'prepare', 'upload', or 'all'. Defaults to environment or 'all'.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Curation output root to package.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=None,
        help="Source-run directory name to skip during discovery. Repeat as needed.",
    )
    parser.add_argument(
        "--topic-classification-dir",
        type=Path,
        default=None,
        help="Optional topic-classification output root.",
    )
    parser.add_argument(
        "--topic-classification-top-k-topics",
        type=int,
        default=None,
        help="Maximum public topic labels per repository classification.",
    )
    parser.add_argument(
        "--longitudinal-refactoring-dir",
        type=Path,
        default=None,
        help="Optional longitudinal-refactoring output root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local staging directory for public parquet batches and manifests.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id, for example 'org/name'.",
    )
    parser.add_argument(
        "--repo-type",
        default=None,
        help="Hugging Face repo type. Keep the default 'dataset' for this stage.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. Prefer HF_TOKEN or HUGGINGFACE_HUB_TOKEN instead.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write an upload plan without contacting Hugging Face.",
    )

    # Output-shaping arguments affect the public parquet layout and local resume
    # state. They are exposed because changing them changes the staged package.
    parser.add_argument("--data-subdir", default=None, help="Staged data subdirectory.")
    parser.add_argument(
        "--output-batch-size",
        type=int,
        default=None,
        help="Rows per output parquet batch.",
    )
    parser.add_argument(
        "--max-files-per-directory",
        type=int,
        default=None,
        help="Shard parquet outputs before this file count is exceeded.",
    )
    parser.add_argument(
        "--parquet-compression",
        default=None,
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--state-db-filename",
        default=None,
        help="SQLite state filename stored under the output directory.",
    )
    parser.add_argument(
        "--schema-version",
        default=None,
        help="Public curation schema version recorded in manifests.",
    )
    parser.add_argument(
        "--blob-batch-bytes",
        type=int,
        default=None,
        help="Approximate byte threshold for snapshot-file blob parquet batches.",
    )

    # Hugging Face can throttle request volume and repository commit frequency.
    # These settings expose the conservative retry policy used by Docker runs.
    parser.add_argument("--upload-max-retries", type=int, default=None)
    parser.add_argument("--upload-retry-base-seconds", type=float, default=None)
    parser.add_argument(
        "--upload-short-term-rate-limit-window-seconds",
        type=float,
        default=None,
    )
    parser.add_argument("--upload-hourly-rate-limit-delay-seconds", type=float, default=None)
    parser.add_argument("--upload-consecutive-failure-threshold", type=int, default=None)
    parser.add_argument("--upload-consecutive-failure-delay-seconds", type=float, default=None)
    parser.add_argument("--upload-large-folder-num-workers", type=int, default=None)
    parser.add_argument(
        "--upload-large-folder-directory-cooldown-seconds",
        type=float,
        default=None,
    )
    return parser


def _build_pipeline(settings: UploadCurationSettings) -> HFCurationUploadPipeline:
    """Construct the curation to Hugging Face dataset pipeline."""
    return HFCurationUploadPipeline(
        curation_outputs_dir=settings.curation_outputs_dir,
        curation_exclude_dirs=settings.curation_exclude_dirs,
        topic_classification_outputs_dir=settings.topic_classification_outputs_dir,
        topic_classification_top_k_topics=settings.topic_classification_top_k_topics,
        longitudinal_refactoring_outputs_dir=settings.longitudinal_refactoring_outputs_dir,
        target_huggingface_repo_id=settings.repo_id,
        huggingface_token=settings.hf_token,
        local_output_dir=settings.output_dir,
        standardized_data_subdir=settings.data_subdir,
        output_batch_size=settings.output_batch_size,
        max_files_per_directory=settings.max_files_per_directory,
        parquet_compression=settings.parquet_compression,
        upload_max_retries=settings.upload_max_retries,
        upload_retry_base_seconds=settings.upload_retry_base_seconds,
        upload_short_term_rate_limit_window_seconds=(
            settings.upload_short_term_rate_limit_window_seconds
        ),
        upload_hourly_rate_limit_delay_seconds=(
            settings.upload_hourly_rate_limit_delay_seconds
        ),
        upload_consecutive_failure_threshold=settings.upload_consecutive_failure_threshold,
        upload_consecutive_failure_delay_seconds=(
            settings.upload_consecutive_failure_delay_seconds
        ),
        upload_large_folder_num_workers=settings.upload_large_folder_num_workers,
        upload_large_folder_directory_cooldown_seconds=(
            settings.upload_large_folder_directory_cooldown_seconds
        ),
        state_db_filename=settings.state_db_filename,
        curation_schema_version=settings.schema_version,
        blob_batch_bytes=settings.blob_batch_bytes,
    )


def _manifest_summary(manifest_path: Path) -> dict[str, Any] | None:
    """Return the prepared summary from an existing manifest when available.

    Upload-only runs do not scan source curation outputs, so their in-memory
    summary is intentionally sparse. The manifest preserves the previous
    prepare counts for user-facing reporting.
    """
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


def run(settings: UploadCurationSettings) -> dict[str, Any]:
    """Run the requested preparation/upload command."""
    pipeline = _build_pipeline(settings)
    try:
        if settings.command in {COMMAND_PREPARE, COMMAND_ALL}:
            pipeline.export()
        manifest_path = settings.output_dir / "upload_curation_manifest.json"
        # Prepare/all runs write a fresh manifest. Upload-only runs reuse the
        # existing manifest unless the staging directory does not have one yet.
        if settings.command in {COMMAND_PREPARE, COMMAND_ALL} or not manifest_path.exists():
            manifest_path = pipeline.write_run_manifest(
                settings=settings.redacted_manifest_settings()
            )
        if settings.command in {COMMAND_UPLOAD, COMMAND_ALL}:
            pipeline.upload_outputs(dry_run=settings.dry_run)
        reported_summary = pipeline.summary
        if settings.command == COMMAND_UPLOAD:
            reported_summary = _manifest_summary(manifest_path) or pipeline.summary
        return {
            "command": settings.command,
            "output_dir": str(settings.output_dir),
            "manifest_path": str(manifest_path),
            "summary": reported_summary,
        }
    finally:
        pipeline.state_store.close()


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the stage, and print a JSON summary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = UploadCurationSettings.from_cli(vars(args))
    summary = run(settings)
    print("[post-processing/upload-curation-data] Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
