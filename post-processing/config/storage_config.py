"""
Storage configuration for the post-processing upload-extraction-data component.

Controls where upstream parquet data is loaded from before upload.
"""

import os
from pathlib import Path


# Hugging Face target dataset repo id. Upload stages read this as a fallback;
# prefer stage-specific CLI flags or environment variables for new runs.
TARGET_HUGGINGFACE_REPO_ID = (
    os.environ.get("POST_PROCESSING_HF_REPO_ID")
    or os.environ.get("HF_DATASET_REPO_ID")
    or ""
)

# Hugging Face token for upload. Tokens must come from the environment.
HUGGINGFACE_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or ""
)

# Local roots to scan for upstream parquet input.
_LOCAL_DIRECTORIES_ENV = os.environ.get("POST_PROCESSING_LOCAL_DIRECTORIES")
if _LOCAL_DIRECTORIES_ENV:
    LOCAL_DIRECTORIES = [
        part.strip() for part in _LOCAL_DIRECTORIES_ENV.split(os.pathsep) if part.strip()
    ]
else:
    LOCAL_DIRECTORIES = [str(Path("/data/input"))]

# Curation outputs root for standalone post-processing analysis.
# This points at a directory containing one subdirectory per cohort. Each cohort
# subdirectory should contain the same structure as a curation run's
# outputs/<cohort> directory, including output/processed-data/<cohort>/metrics-json.
CURATION_OUTPUTS_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_CURATION_OUTPUTS_DIR",
        r"C:\path\to\curation-outputs",
    )
)

# Optional comma-separated subdirectory names or glob patterns to exclude under
# the configured curation outputs root, e.g. "validation_*,legacy_cohort".
_CURATION_EXCLUDE_DIRS_ENV = os.environ.get("POST_PROCESSING_CURATION_EXCLUDE_DIRS", "")
CURATION_EXCLUDE_DIRS = [
    part.strip() for part in _CURATION_EXCLUDE_DIRS_ENV.split(",") if part.strip()
]

_POST_PROCESSING_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _POST_PROCESSING_DIR.parent

_LONGITUDINAL_REFACTORING_OUTPUTS_DIR_ENV = os.environ.get(
    "POST_PROCESSING_LONGITUDINAL_REFACTORING_OUTPUTS_DIR"
)
LONGITUDINAL_REFACTORING_OUTPUTS_DIR = Path(
    _LONGITUDINAL_REFACTORING_OUTPUTS_DIR_ENV
    or os.environ.get("POST_PROCESSING_RUNS_ROOT")
    or os.environ.get("MOSAIC_RUNS_ROOT")
    or str(_PROJECT_ROOT / "runs")
)

ANALYSIS_OUTPUT_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_OUTPUT_DIR",
        str(Path(__file__).resolve().parents[1] / "analysis" / "output"),
    )
)

ANALYSIS_RANKINGS_DIR = ANALYSIS_OUTPUT_DIR / "rankings"
ANALYSIS_SIGNIFICANCE_TESTS_DIR = ANALYSIS_OUTPUT_DIR / "significance-tests"
ANALYSIS_DISTRIBUTION_TESTS_DIR = ANALYSIS_OUTPUT_DIR / "distribution-tests"

ANALYSIS_PLOTS_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_PLOTS_DIR",
        str(ANALYSIS_OUTPUT_DIR / "plots"),
    )
)

ANALYSIS_TEMP_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_TEMP_DIR",
        str(ANALYSIS_OUTPUT_DIR / ".analysis-tmp"),
    )
)

ANALYSIS_WORKSPACE_DB_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_WORKSPACE_DB_PATH",
        str(ANALYSIS_TEMP_DIR / "analysis_workspace.sqlite"),
    )
)

ANALYSIS_RETAIN_TEMP_WORKSPACE = str(
    os.environ.get("POST_PROCESSING_ANALYSIS_RETAIN_TEMP_WORKSPACE", "")
).strip().lower() in {"1", "true", "yes", "on"}

# Output file for numeric PR-metrics aggregation results.
ANALYSIS_NUMERIC_METRICS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_NUMERIC_METRICS_OUTPUT_PATH",
        str(ANALYSIS_OUTPUT_DIR / "numeric_metrics_aggregation.csv"),
    )
)

# Output file for standardized refactoring-type ranking aggregations.
ANALYSIS_REFACTORING_TYPE_RANKINGS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_REFACTORING_TYPE_RANKINGS_OUTPUT_PATH",
        str(ANALYSIS_RANKINGS_DIR / "refactoring_type_rankings.csv"),
    )
)

# Output file for Murphy-Hill level ranking aggregations.
ANALYSIS_REFACTORING_MURPHY_HILL_RANKINGS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_REFACTORING_MURPHY_HILL_RANKINGS_OUTPUT_PATH",
        str(ANALYSIS_RANKINGS_DIR / "refactoring_murphy_hill_rankings.csv"),
    )
)

# Output file for standardized refactoring-type density rankings.
ANALYSIS_REFACTORING_TYPE_DENSITY_RANKINGS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_REFACTORING_TYPE_DENSITY_RANKINGS_OUTPUT_PATH",
        str(ANALYSIS_RANKINGS_DIR / "refactoring_type_density_rankings.csv"),
    )
)

# Output file for Murphy-Hill density rankings.
ANALYSIS_REFACTORING_MURPHY_HILL_DENSITY_RANKINGS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_REFACTORING_MURPHY_HILL_DENSITY_RANKINGS_OUTPUT_PATH",
        str(ANALYSIS_RANKINGS_DIR / "refactoring_murphy_hill_density_rankings.csv"),
    )
)

# Output file for code smell frequency rankings across snapshots.
ANALYSIS_CODE_SMELL_RANKINGS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_CODE_SMELL_RANKINGS_OUTPUT_PATH",
        str(ANALYSIS_RANKINGS_DIR / "code_smell_rankings.csv"),
    )
)

# Output file for code smell density rankings across snapshots.
ANALYSIS_CODE_SMELL_DENSITY_RANKINGS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_CODE_SMELL_DENSITY_RANKINGS_OUTPUT_PATH",
        str(ANALYSIS_RANKINGS_DIR / "code_smell_density_rankings.csv"),
    )
)

# Output file for Mäntylä-only code smell density rankings.
ANALYSIS_MANTYLA_DENSITY_RANKINGS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_MANTYLA_DENSITY_RANKINGS_OUTPUT_PATH",
        str(ANALYSIS_RANKINGS_DIR / "mantyla_density_rankings.csv"),
    )
)

# Output manifest for standalone post-processing analysis runs.
ANALYSIS_OUTPUT_MANIFEST_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_OUTPUT_MANIFEST_PATH",
        str(ANALYSIS_OUTPUT_DIR / "aggregation_manifest.json"),
    )
)

# Output file for thesis-ready summaries derived from statistical CSV outputs.
ANALYSIS_RESULTS_SECTION_SUMMARY_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_RESULTS_SECTION_SUMMARY_OUTPUT_PATH",
        str(ANALYSIS_OUTPUT_DIR / "results_section_summary.csv"),
    )
)

# Maximum rows to retain per thesis-ready summary section.
ANALYSIS_RESULTS_SECTION_SUMMARY_TOP_N = int(
    os.environ.get("POST_PROCESSING_ANALYSIS_RESULTS_SECTION_SUMMARY_TOP_N", "10")
)

# Output file for pairwise numeric-metric significance tests.
ANALYSIS_NUMERIC_METRIC_PAIRWISE_TESTS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_NUMERIC_METRIC_PAIRWISE_TESTS_OUTPUT_PATH",
        str(ANALYSIS_SIGNIFICANCE_TESTS_DIR / "numeric_metric_pairwise_tests.csv"),
    )
)

# Minimum valid observations per group to execute a pairwise significance test.
ANALYSIS_MIN_GROUP_SIZE_FOR_SIGNIFICANCE_TEST = int(
    os.environ.get("POST_PROCESSING_ANALYSIS_MIN_GROUP_SIZE_FOR_SIGNIFICANCE_TEST", "2")
)

# Output file for multi-group numeric-metric significance tests.
ANALYSIS_NUMERIC_METRIC_MULTI_GROUP_TESTS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_NUMERIC_METRIC_MULTI_GROUP_TESTS_OUTPUT_PATH",
        str(ANALYSIS_SIGNIFICANCE_TESTS_DIR / "numeric_metric_multi_group_tests.csv"),
    )
)

# Output file for cohort x dominant language sampling-bias distributions.
ANALYSIS_COHORT_LANGUAGE_DISTRIBUTION_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_COHORT_LANGUAGE_DISTRIBUTION_OUTPUT_PATH",
        str(ANALYSIS_DISTRIBUTION_TESTS_DIR / "cohort_language_distribution.csv"),
    )
)

# Output file for cohort x popularity sampling-bias distributions.
ANALYSIS_COHORT_POPULARITY_DISTRIBUTION_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_COHORT_POPULARITY_DISTRIBUTION_OUTPUT_PATH",
        str(ANALYSIS_DISTRIBUTION_TESTS_DIR / "cohort_popularity_distribution.csv"),
    )
)

# Output file for cohort x created-month sampling-bias distributions.
ANALYSIS_COHORT_TIME_DISTRIBUTION_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_COHORT_TIME_DISTRIBUTION_OUTPUT_PATH",
        str(ANALYSIS_DISTRIBUTION_TESTS_DIR / "cohort_time_distribution.csv"),
    )
)

# Output file for sampling-bias chi-square test results.
ANALYSIS_DISTRIBUTION_BIAS_TESTS_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_DISTRIBUTION_BIAS_TESTS_OUTPUT_PATH",
        str(ANALYSIS_DISTRIBUTION_TESTS_DIR / "distribution_tests.csv"),
    )
)

# Output file for grouping-level PR/repository/author coverage summary.
ANALYSIS_GROUPING_COVERAGE_SUMMARY_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_GROUPING_COVERAGE_SUMMARY_OUTPUT_PATH",
        str(ANALYSIS_OUTPUT_DIR / "grouping_coverage_summary.csv"),
    )
)

# Output file for original-PR and future touching-commit count missing/zero reasons.
ANALYSIS_COMMIT_COUNT_MISSING_REASON_SUMMARY_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_COMMIT_COUNT_MISSING_REASON_SUMMARY_OUTPUT_PATH",
        str(ANALYSIS_OUTPUT_DIR / "commit_count_missing_reason_summary.csv"),
    )
)

# Output file for PR-level prevalence summaries across requested populations/groups.
ANALYSIS_PR_PRESENCE_SUMMARY_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_PR_PRESENCE_SUMMARY_OUTPUT_PATH",
        str(ANALYSIS_OUTPUT_DIR / "pr_presence_summary.csv"),
    )
)

# Output file for future snapshot availability and missing-reason summaries.
ANALYSIS_FUTURE_SNAPSHOT_AVAILABILITY_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_FUTURE_SNAPSHOT_AVAILABILITY_OUTPUT_PATH",
        str(ANALYSIS_OUTPUT_DIR / "future_snapshot_availability_summary.csv"),
    )
)

# Output file documenting grouping-level semantics used in analysis outputs.
ANALYSIS_GROUPING_LEVEL_REFERENCE_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_GROUPING_LEVEL_REFERENCE_OUTPUT_PATH",
        str(ANALYSIS_OUTPUT_DIR / "grouping_level_reference.csv"),
    )
)

# Output file documenting global popularity bucket thresholds used in analysis.
ANALYSIS_POPULARITY_BUCKET_REFERENCE_OUTPUT_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_POPULARITY_BUCKET_REFERENCE_OUTPUT_PATH",
        str(ANALYSIS_OUTPUT_DIR / "popularity_bucket_reference.json"),
    )
)

# Output directory for per-PR analysis update artifacts grouped by cohort.
ANALYSIS_PR_CANONICAL_ARTIFACTS_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_ANALYSIS_PR_CANONICAL_ARTIFACTS_DIR",
        str(ANALYSIS_OUTPUT_DIR / "per-cohort-update"),
    )
)

# Local output directory for standardized parquet batches and upload state.
LOCAL_OUTPUT_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_LOCAL_OUTPUT_DIR",
        str(Path(__file__).resolve().parents[1] / "upload-extraction-data" / "output"),
    )
)

# Local output directory for curation parquet batches, metadata, and upload state.
UPLOAD_CURATION_LOCAL_OUTPUT_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_UPLOAD_CURATION_LOCAL_OUTPUT_DIR",
        str(Path(__file__).resolve().parents[1] / "upload-curation-data" / "output"),
    )
)

# Target folder inside the local output dir and Hugging Face dataset repo.
# Parquet uploads will be stored as data/<cohort>/*.parquet in the HF dataset.
STANDARDIZED_DATA_SUBDIR = "data"

# Batch size for standardized parquet files written before upload.
OUTPUT_BATCH_SIZE = 10000

# Maximum parquet files allowed in one dataset directory before sharding.
MAX_FILES_PER_DIRECTORY = int(
    os.environ.get("POST_PROCESSING_MAX_FILES_PER_DIRECTORY", "9500")
)

# Compression for standardized parquet output.
PARQUET_COMPRESSION = "zstd"

# Hugging Face upload retry settings.
UPLOAD_MAX_RETRIES = 12
UPLOAD_RETRY_BASE_SECONDS = 5 * 60.0
UPLOAD_SHORT_TERM_RATE_LIMIT_WINDOW_SECONDS = 5 * 60.0
UPLOAD_HOURLY_RATE_LIMIT_DELAY_SECONDS = 60 * 60.0
UPLOAD_CONSECUTIVE_FAILURE_THRESHOLD = 1
UPLOAD_CONSECUTIVE_FAILURE_DELAY_SECONDS = 5 * 60.0
UPLOAD_LARGE_FOLDER_NUM_WORKERS = 1
UPLOAD_LARGE_FOLDER_DIRECTORY_COOLDOWN_SECONDS = float(
    os.environ.get(
        "POST_PROCESSING_UPLOAD_LARGE_FOLDER_DIRECTORY_COOLDOWN_SECONDS",
        "5.0",
    )
)

# SQLite-backed state for large-scale deduplication/upload resumability.
STATE_DB_FILENAME = "upload_state.sqlite3"
UPLOAD_CURATION_STATE_DB_FILENAME = "upload_curation_state.sqlite3"

# Dataset format version written into exported rows.
STANDARDIZED_SCHEMA_VERSION = "pull_request_standardized_v12"
CURATION_UPLOAD_SCHEMA_VERSION = "curation_upload_v1"

# Approximate raw-byte target for snapshot blob parquet batches.
CURATION_UPLOAD_BLOB_BATCH_BYTES = int(
    os.environ.get("POST_PROCESSING_CURATION_UPLOAD_BLOB_BATCH_BYTES", str(128 * 1024 * 1024))
)
