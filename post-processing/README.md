# Post-Processing Pipeline

This directory contains the downstream stages that operate after local
extraction and curation data exist. The stages can be run independently, but
they are designed to share a common layout, Docker-first workflow, and
configuration style.

Post-processing has four active packages:

- [Topic classification](topic-classification/README.md): collect labeled
  GitHub repositories, filter and sample training data, train a topic model,
  and classify repositories from curation outputs.
- [Analysis](analysis/README.md): read curation parquet plus optional topic
  classifications, then write local summaries, plot-data JSON, and figures for
  the dataset, refactoring, and maintainability pipelines.
- [Upload extraction data](upload-extraction-data/README.md): convert local
  extraction parquet into a public Hugging Face dataset layout and optionally
  upload it.
- [Upload curation data](upload-curation-data/README.md): convert local
  curation outputs, optional topic classifications, and optional longitudinal
  refactoring summaries into a public Hugging Face dataset layout and
  optionally upload it.

## Pipeline Flow

The post-processing packages are intentionally file-system oriented. They read
local outputs from earlier stages, write local artifacts first, and only contact
external services when a command explicitly needs GitHub or Hugging Face.

```text
local extraction output
  -> upload-extraction-data prepare/upload

local curation output
  -> topic-classification classify
  -> analysis dataset/refactoring/maintainability
  -> upload-curation-data prepare/upload

topic-classification training
  -> extract labeled repositories from GitHub
  -> filter, preprocess, and sample training data
  -> train topic model
  -> classify repositories from curation output
```

The analysis package keeps three separate pipelines:

- `dataset`: cohort counts, source coverage, topic/domain coverage, and
  snapshot availability.
- `refactoring`: original PR refactoring prevalence, type summaries,
  characteristics, and longitudinal refactoring persistence.
- `maintainability`: code-smell summaries, Mantyla categories, Multimetric
  maintainability values, custom duplicated-lines density, and longitudinal
  maintainability trends.

## Stage Responsibilities

Post-processing does not scrape pull requests for extraction and does not
hydrate curation snapshots. Those responsibilities stay in the upstream
packages.

Topic classification is the only post-processing package that can call GitHub.
It uses GitHub during topic-training extraction and, when enabled, live README
or wiki enrichment during repository classification.

Analysis is local-only. It never runs refactoring tools, code-smell tools,
Multimetric, or custom duplicated-lines computation. Those values must already
be present in the curation input.

Upload packages are local-first. `prepare` writes a staged dataset layout,
schema manifest, run manifest, and optional dry-run upload plan. `upload`
publishes an already prepared staging directory to Hugging Face.

## Inputs

Typical inputs:

- Local extraction output with `data/<cohort>/*.parquet`, `manifest.json`, and
  `extraction_run_manifest.json`.
- Local curation output with cohort directories, aggregate metric JSON,
  snapshot manifests, repository file lists, and processed parquet outputs.
- Optional topic-classification output with repository topic labels and
  repository-to-PR mappings.
- Optional longitudinal refactoring output with snapshot-level JSONL summaries.

The package READMEs document exact input shapes:

- [Topic classification input details](topic-classification/README.md)
- [Analysis input details](analysis/README.md)
- [Upload extraction input details](upload-extraction-data/README.md)
- [Upload curation input details](upload-curation-data/README.md)

## Configuration and Tokens

Each package resolves settings with the same precedence:

```text
CLI flags > environment variables > defaults
```

Common token variables:

- `GITHUB_TOKENS`: comma-separated GitHub tokens for topic-training extraction
  and optional live topic-classification enrichment.
- `GITHUB_TOKEN`: fallback for a single GitHub token.
- `HF_TOKEN`: preferred Hugging Face token for upload commands.
- `HUGGINGFACE_HUB_TOKEN`: Hugging Face token fallback.
- `HUGGING_FACE_HUB_TOKEN`: Hugging Face token fallback.

Stage-specific environment prefixes:

- `POST_PROCESSING_TOPIC_*` for topic classification.
- `POST_PROCESSING_ANALYSIS_*` for analysis.
- `POST_PROCESSING_UPLOAD_EXTRACTION_*` for extraction-data publishing.
- `POST_PROCESSING_UPLOAD_CURATION_*` for curation-data publishing.

Tokens are read from environment variables. Manifests write only redacted token
metadata and never persist raw token values.

## Requirements

Docker is the recommended runtime for all post-processing packages because it
isolates Python, parquet, plotting, Hugging Face, and NLP dependencies from the
host environment.

For Docker runs, you need:

- Docker.
- Local extraction and/or curation outputs mounted read-only.
- Writable local output directories mounted into the container.
- GitHub tokens only for topic-classification commands that contact GitHub.
- Hugging Face tokens only for real upload commands.

For local runs without Docker, use Python 3.12 and install the package-specific
requirements file rather than the broad root file when possible:

- `post-processing/topic-classification/requirements.txt`
- `post-processing/analysis/requirements.txt`
- `post-processing/upload-extraction-data/requirements.txt`
- `post-processing/upload-curation-data/requirements.txt`

The root `post-processing/requirements.txt` is a broad convenience dependency
set for shared local work.

## Step-by-Step Run: Docker

Use the package READMEs for full command arguments. This section shows the
recommended order and the smoke checks for each container.

1. Build the topic-classification image from the `post-processing` directory.

   ```bash
   cd post-processing
   docker build -f topic-classification/Dockerfile -t mosaic-topic-classification .
   docker run --rm mosaic-topic-classification \
     python -u post-processing/topic-classification/run.py --help
   cd ..
   ```

2. Run the topic-classification stages when repository topics are needed.

   ```bash
   docker run --rm mosaic-topic-classification \
     python -u post-processing/topic-classification/run.py extract --help

   docker run --rm mosaic-topic-classification \
     python -u post-processing/topic-classification/run.py prepare --help

   docker run --rm mosaic-topic-classification \
     python -u post-processing/topic-classification/run.py train --help

   docker run --rm mosaic-topic-classification \
     python -u post-processing/topic-classification/run.py classify --help
   ```

3. Build and check the analysis image from the repository root.

   ```bash
   docker build -f post-processing/analysis/Dockerfile -t mosaic-analysis:local .
   docker run --rm mosaic-analysis:local \
     python -u post-processing/analysis/run.py --help
   ```

4. Run all analysis pipelines.

   ```bash
   docker run --rm \
     -v "$PWD/outputs/curation:/data/curation-data:ro" \
     -v "$PWD/post-processing/topic-classification/classification-runs:/data/topic-classification:ro" \
     -v "$PWD/outputs/analysis:/output" \
     mosaic-analysis:local \
     python -u post-processing/analysis/run.py all
   ```

5. Build and check the extraction-data upload image.

   ```bash
   docker build \
     -f post-processing/upload-extraction-data/Dockerfile \
     -t mosaic-upload-extraction:local \
     .

   docker run --rm mosaic-upload-extraction:local \
     python post-processing/upload-extraction-data/run.py --help
   ```

6. Prepare extraction data locally, then dry-run or run upload.

   ```bash
   docker run --rm \
     -v /absolute/path/to/extraction-output:/data/input:ro \
     -v "$PWD/outputs/upload-extraction:/data/output" \
     mosaic-upload-extraction:local \
     python post-processing/upload-extraction-data/run.py prepare

   docker run --rm \
     -e POST_PROCESSING_UPLOAD_EXTRACTION_REPO_ID="your-org/your-dataset" \
     -v "$PWD/outputs/upload-extraction:/data/output" \
     mosaic-upload-extraction:local \
     python post-processing/upload-extraction-data/run.py upload --dry-run
   ```

7. Build and check the curation-data upload image.

   ```bash
   docker build \
     -f post-processing/upload-curation-data/Dockerfile \
     -t mosaic-upload-curation:local \
     .

   docker run --rm mosaic-upload-curation:local \
     python post-processing/upload-curation-data/run.py --help
   ```

8. Prepare curation data locally, then dry-run or run upload.

   ```bash
   mkdir -p outputs/empty-topic-classification
   mkdir -p outputs/empty-longitudinal-refactoring

   docker run --rm \
     -v /absolute/path/to/curation-output:/data/input:ro \
     -v "$PWD/outputs/empty-topic-classification:/data/topic-classification:ro" \
     -v "$PWD/outputs/empty-longitudinal-refactoring:/data/longitudinal-refactoring:ro" \
     -v "$PWD/outputs/upload-curation:/data/output" \
     mosaic-upload-curation:local \
     python post-processing/upload-curation-data/run.py prepare

   docker run --rm \
     -e POST_PROCESSING_UPLOAD_CURATION_REPO_ID="your-org/your-dataset" \
     -v "$PWD/outputs/upload-curation:/data/output" \
     mosaic-upload-curation:local \
     python post-processing/upload-curation-data/run.py upload --dry-run
   ```

## Step-by-Step Run: Local

Use local runs when Docker is unavailable or when debugging Python code.

1. Create a virtual environment from the repository root.

   ```bash
   python -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   ```

   On PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. Install the requirements for the package you are running.

   ```bash
   python -m pip install -r post-processing/topic-classification/requirements.txt
   python -m pip install -r post-processing/analysis/requirements.txt
   python -m pip install -r post-processing/upload-extraction-data/requirements.txt
   python -m pip install -r post-processing/upload-curation-data/requirements.txt
   ```

3. Check the local CLIs.

   ```bash
   python post-processing/topic-classification/run.py --help
   python post-processing/analysis/run.py --help
   python post-processing/upload-extraction-data/run.py --help
   python post-processing/upload-curation-data/run.py --help
   ```

4. Run package commands with local paths.

   ```bash
   python post-processing/analysis/run.py all \
     --curation-data-dir outputs/curation \
     --topic-classification-dir post-processing/topic-classification/classification-runs \
     --analysis-output-dir outputs/analysis

   python post-processing/upload-extraction-data/run.py prepare \
     --input-dir outputs/extraction \
     --output-dir outputs/upload-extraction

   python post-processing/upload-curation-data/run.py prepare \
     --input-dir outputs/curation \
     --topic-classification-dir post-processing/topic-classification/classification-runs \
     --output-dir outputs/upload-curation
   ```

## Output Layout

Common local output roots:

```text
outputs/
  analysis/
    dataset/
    refactoring/
    maintainability/
  upload-extraction/
    data/
    schema_manifest.json
    upload_extraction_manifest.json
  upload-curation/
    data/
    schema_manifest.json
    upload_curation_manifest.json

post-processing/
  topic-classification/
    runs/
    model-output/
    classification-runs/
```

Each package writes a manifest beside its output. Use those manifests for row
counts, settings, schema version, input references, and redacted credential
state.

## Troubleshooting

Docker cannot find files:

- Build from the directory documented by the package README.
- Mount host paths with absolute paths.
- Confirm the mounted container path matches the command arguments.

GitHub calls are rate-limited:

- Use `GITHUB_TOKENS` with multiple comma-separated tokens when available.
- Reduce workers or page counts for topic-training extraction.
- Resume with the same output directory so completed work is reused.

Upload commands do nothing:

- `prepare` never uploads.
- `upload --dry-run` writes a plan without contacting Hugging Face.
- Real upload requires `HF_TOKEN` and a Hugging Face dataset repository id.

Analysis outputs are missing topic/domain plots:

- Confirm topic classification has completed.
- Confirm the analysis command points at the classification output root.
- Core dataset, refactoring, and maintainability summaries can run without
  topic classification; topic/domain plots are skipped when topic data is
  unavailable.

## Files of Interest

- [topic-classification/README.md](topic-classification/README.md): four-stage
  topic pipeline runbook.
- [analysis/README.md](analysis/README.md): dataset, refactoring, and
  maintainability analysis runbook.
- [upload-extraction-data/README.md](upload-extraction-data/README.md):
  extraction-data packaging and upload runbook.
- [upload-curation-data/README.md](upload-curation-data/README.md):
  curation-data packaging and upload runbook.
- [config/](config/): shared post-processing configuration defaults.
- [utility/](utility/): shared discovery, JSON/CSV IO, repository-key, and
  output-layout helpers.
- [requirements.txt](requirements.txt): broad local dependency set for shared
  post-processing work.
