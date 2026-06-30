"""Configuration for topic-classification post-processing runs."""

import os
from pathlib import Path


def _strict_env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be either 'true' or 'false', got {value!r}.")


# Curation outputs root for topic classification. This should point at a
# directory containing one subdirectory per cohort.
CURATION_OUTPUTS_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_CURATION_OUTPUTS_DIR",
        r"C:\path\to\curation-outputs",
    )
)

# Optional comma-separated subdirectory names or glob patterns to exclude under
# the configured curation outputs root, e.g. "validation_*,legacy_cohort".
DEFAULT_CURATION_EXCLUDE_DIRS = ("openhands", "junie", "codegen", "cosine")
_CURATION_EXCLUDE_DIRS_ENV = os.environ.get("POST_PROCESSING_CURATION_EXCLUDE_DIRS", "")
CURATION_EXCLUDE_DIRS = [
    *DEFAULT_CURATION_EXCLUDE_DIRS,
    *[part.strip() for part in _CURATION_EXCLUDE_DIRS_ENV.split(",") if part.strip()],
]

TOPIC_CLASSIFICATION_BATCH_SIZE = int(
    os.environ.get("POST_PROCESSING_TOPIC_CLASSIFICATION_BATCH_SIZE", "10")
)

TOPIC_CLASSIFICATION_WORKERS = int(
    os.environ.get("POST_PROCESSING_TOPIC_CLASSIFICATION_WORKERS", "1")
)

TOPIC_CLASSIFICATION_INPUT_FORMAT = os.environ.get(
    "POST_PROCESSING_TOPIC_CLASSIFICATION_INPUT_FORMAT",
    "parquet",
).strip().lower()

TOPIC_CLASSIFICATION_LONGITUDINAL_ONLY = str(
    os.environ.get("POST_PROCESSING_TOPIC_CLASSIFICATION_LONGITUDINAL_ONLY", "")
).strip().lower() in {"1", "true", "yes", "on"}

_POST_PROCESSING_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _POST_PROCESSING_DIR.parent
_TOPIC_CLASSIFICATION_DIR = _POST_PROCESSING_DIR / "topic-classification"

TOPIC_CLASSIFICATION_MODEL_BUNDLE_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_MODEL_BUNDLE_PATH",
        str(_TOPIC_CLASSIFICATION_DIR / "model-output" / "topic_model_bundle.joblib"),
    )
)

_TOPIC_CLASSIFICATION_RUNS_DIR_ENV = os.environ.get(
    "POST_PROCESSING_TOPIC_CLASSIFICATION_RUNS_DIR"
) or os.environ.get("POST_PROCESSING_TOPIC_CLASSIFICATION_OUTPUT_DIR")
TOPIC_CLASSIFICATION_RUNS_DIR = Path(
    _TOPIC_CLASSIFICATION_RUNS_DIR_ENV or str(_PROJECT_ROOT / "runs")
)

TOPIC_CLASSIFICATION_RUN_ID = os.environ.get(
    "POST_PROCESSING_TOPIC_CLASSIFICATION_RUN_ID",
    "",
).strip()

TOPIC_CLASSIFICATION_OUTPUTS_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_OUTPUTS_DIR",
        str(TOPIC_CLASSIFICATION_RUNS_DIR),
    )
)

# Deprecated compatibility alias retained for callers that read the setting
# directly. run_classify_topics writes to TOPIC_CLASSIFICATION_RUNS_DIR/<run>/output.
TOPIC_CLASSIFICATION_OUTPUT_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_OUTPUT_DIR",
        str(TOPIC_CLASSIFICATION_RUNS_DIR),
    )
)

TOPIC_CLASSIFICATION_TOP_K_TOPICS = int(
    os.environ.get("POST_PROCESSING_TOPIC_CLASSIFICATION_TOP_K_TOPICS", "1")
)

TOPIC_CLASSIFICATION_PREDICTION_SCORE_THRESHOLD = float(
    os.environ.get("POST_PROCESSING_TOPIC_CLASSIFICATION_PREDICTION_SCORE_THRESHOLD", "0.7")
)

TOPIC_CLASSIFICATION_TOPIC_DOMAIN_MAPPING_PATH = Path(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_TOPIC_DOMAIN_MAPPING_PATH",
        str(_TOPIC_CLASSIFICATION_DIR / "classify-topics" / "topic_domains.json"),
    )
)

_TOPIC_CLASSIFICATION_EXCLUDED_TOPICS_ENV = os.environ.get(
    "POST_PROCESSING_TOPIC_CLASSIFICATION_EXCLUDED_TOPICS", ""
)
TOPIC_CLASSIFICATION_EXCLUDED_TOPICS = [
    part.strip()
    for part in _TOPIC_CLASSIFICATION_EXCLUDED_TOPICS_ENV.split(",")
    if part.strip()
]

TOPIC_CLASSIFICATION_OUTPUT_RAW_PREDICTIONS = str(
    os.environ.get("POST_PROCESSING_TOPIC_CLASSIFICATION_OUTPUT_RAW_PREDICTIONS", "")
).strip().lower() in {"1", "true", "yes", "on"}

TOPIC_CLASSIFICATION_IDENTIFIER_SPLITTER = os.environ.get(
    "POST_PROCESSING_TOPIC_CLASSIFICATION_IDENTIFIER_SPLITTER",
    "spiral",
).strip().lower()

TOPIC_CLASSIFICATION_REQUIRE_MODEL_PREPROCESSING_ARTIFACTS = str(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_REQUIRE_MODEL_PREPROCESSING_ARTIFACTS",
        "true",
    )
).strip().lower() in {"1", "true", "yes", "on"}

TOPIC_CLASSIFICATION_REQUIRE_RAW_FREQUENCY_ARTIFACTS = str(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_REQUIRE_RAW_FREQUENCY_ARTIFACTS",
        "false",
    )
).strip().lower() in {"1", "true", "yes", "on"}

TOPIC_CLASSIFICATION_ALLOW_HEURISTIC_FILE_SPLITS = str(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_ALLOW_HEURISTIC_FILE_SPLITS",
        "",
    )
).strip().lower() in {"1", "true", "yes", "on"}

TOPIC_CLASSIFICATION_ENABLE_LIVE_WIKI_FETCH = _strict_env_bool(
    "POST_PROCESSING_TOPIC_CLASSIFICATION_ENABLE_LIVE_WIKI_FETCH",
    default=True,
)

TOPIC_CLASSIFICATION_DATA_PREPARATION_ROOT = Path(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_DATA_PREPARATION_ROOT",
        str(_TOPIC_CLASSIFICATION_DIR / "data-preparation" / "generated" / "github-topics"),
    )
)

TOPIC_CLASSIFICATION_WIKI_CACHE_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_WIKI_CACHE_DIR",
        str(_TOPIC_CLASSIFICATION_DIR / "wiki-cache"),
    )
)

TOPIC_CLASSIFICATION_README_CACHE_DIR = Path(
    os.environ.get(
        "POST_PROCESSING_TOPIC_CLASSIFICATION_README_CACHE_DIR",
        str(TOPIC_CLASSIFICATION_WIKI_CACHE_DIR / "readme-cache"),
    )
)
