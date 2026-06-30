"""Command line entrypoint for extraction-data publishing.

The stage reads local extraction parquet output, prepares public parquet entity
batches, and can upload the staged tree to a Hugging Face dataset repository.
It does not scrape GitHub and does not run curation.
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

# The directory name contains a hyphen, so this stage is executed as a script
# path rather than imported as a normal Python package.
for candidate in (REPO_ROOT, UPLOAD_DIR, CONFIG_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from hf_dataset_uploader import HFDatasetUploadPipeline  # noqa: E402
from settings import (  # noqa: E402
    COMMAND_ALL,
    COMMAND_PREPARE,
    COMMAND_UPLOAD,
    COMMANDS,
    UploadExtractionSettings,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI for local preparation and optional upload."""
    parser = argparse.ArgumentParser(
        description="Prepare and optionally upload extraction parquet data.",
    )

    # Command and path arguments define the stage boundary: read extraction
    # output, write a local public package, and optionally publish that package.
    parser.add_argument(
        "command",
        nargs="?",
        choices=COMMANDS,
        default=None,
        help="Run 'prepare', 'upload', or 'all'. Defaults to environment or 'all'.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        default=None,
        help="Extraction output root. Repeat for multiple roots.",
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

    # Output-shaping arguments affect parquet file layout and resume state. Keep
    # them visible in the CLI because they determine the staged dataset shape.
    parser.add_argument(
        "--data-subdir",
        default=None,
        help="Subdirectory under the staging root that contains parquet data.",
    )
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
        help="Public extraction schema version recorded in rows and manifests.",
    )

    # Hugging Face can throttle both request volume and repository commits. These
    # settings expose the same conservative retry policy used by the Docker run.
    parser.add_argument(
        "--upload-max-retries",
        type=int,
        default=None,
        help="Maximum attempts for each Hugging Face file or folder upload.",
    )
    parser.add_argument(
        "--upload-retry-base-seconds",
        type=float,
        default=None,
        help="Base retry delay used for non-quota upload failures.",
    )
    parser.add_argument(
        "--upload-short-term-rate-limit-window-seconds",
        type=float,
        default=None,
        help="Delay cap/window for short-term Hugging Face rate-limit retries.",
    )
    parser.add_argument(
        "--upload-hourly-rate-limit-delay-seconds",
        type=float,
        default=None,
        help="Delay used when Hugging Face reports an hourly repository quota.",
    )
    parser.add_argument(
        "--upload-consecutive-failure-threshold",
        type=int,
        default=None,
        help="Consecutive upload failures before using the failure cooldown.",
    )
    parser.add_argument(
        "--upload-consecutive-failure-delay-seconds",
        type=float,
        default=None,
        help="Cooldown after repeated upload failures.",
    )
    parser.add_argument(
        "--upload-large-folder-num-workers",
        type=int,
        default=None,
        help="Worker count passed to Hugging Face upload_large_folder.",
    )
    parser.add_argument(
        "--upload-large-folder-directory-cooldown-seconds",
        type=float,
        default=None,
        help="Pause between directory-scoped large-folder upload calls.",
    )
    return parser


def _build_pipeline(settings: UploadExtractionSettings) -> HFDatasetUploadPipeline:
    """Construct the raw extraction to Hugging Face dataset pipeline."""
    return HFDatasetUploadPipeline(
        local_directories=[str(path) for path in settings.source_dirs],
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
        upload_consecutive_failure_threshold=(
            settings.upload_consecutive_failure_threshold
        ),
        upload_consecutive_failure_delay_seconds=(
            settings.upload_consecutive_failure_delay_seconds
        ),
        upload_large_folder_num_workers=settings.upload_large_folder_num_workers,
        upload_large_folder_directory_cooldown_seconds=(
            settings.upload_large_folder_directory_cooldown_seconds
        ),
        state_db_filename=settings.state_db_filename,
        standardized_schema_version=settings.schema_version,
    )


def _manifest_summary(manifest_path: Path) -> dict[str, Any] | None:
    """Return the prepared summary from an existing run manifest when available."""
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


def run(settings: UploadExtractionSettings) -> dict[str, Any]:
    """Run the requested preparation/upload command."""
    pipeline = _build_pipeline(settings)
    try:
        if settings.command in {COMMAND_PREPARE, COMMAND_ALL}:
            pipeline.prepare_outputs()
        manifest_path = settings.output_dir / "upload_extraction_manifest.json"
        # Upload-only runs should not erase the preparation manifest. When the
        # staged directory was prepared earlier, keep its original counts.
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
        pipeline.close()


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the stage, and print a compact JSON summary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = UploadExtractionSettings.from_cli(vars(args))
    summary = run(settings)
    print("[post-processing/upload-extraction-data] Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
