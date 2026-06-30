# Upload Extraction Data

This package turns local extraction parquet output into a public Hugging Face
dataset layout and optionally uploads it. It does not scrape GitHub and does not
run curation.

The stage has one input source and two commands:

- `prepare`: read local extraction parquet and write public entity parquet plus
  metadata locally.
- `upload`: upload an already prepared local staging directory to a Hugging Face
  dataset repository.

Use `all` to run `prepare` and `upload` in sequence.

## Pipeline Flow

Upload-extraction has three phases:

```text
CLI/env settings
  -> prepare: stream local extraction PR parquet into public entity tables
  -> manifest: write schema, dataset card, run manifest, and optional upload plan
  -> upload: publish the staged tree to a Hugging Face dataset repository
```

Prepare reads extraction output shaped like `data/<cohort>/*.parquet`. Each
source PR row is normalized through the extraction DTO schema, deduplicated with
stable PR keys when possible, and split into public entity tables.

Manifest writing records the local staging layout, schema version, safe settings
with token redaction, entity relationships, row counts, and upload-plan summary.
The generated dataset card and schema manifest are written beside the staged
`data/` tree.

Upload uses the staged directory as the resume boundary. It can run as a dry run
to write `upload_extraction_plan.json` without contacting Hugging Face.

## Hugging Face APIs Used

This stage uses the [Hugging Face Hub Python API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)
through `huggingface_hub.HfApi`.

Uploads use [`upload_large_folder`](https://huggingface.co/docs/huggingface_hub/en/guides/upload#upload-a-large-folder)
because prepared extraction datasets can contain many parquet shards. That API
is designed for large-folder upload and stores local upload cache files under
the folder being uploaded.

Authentication uses Hugging Face environment variables. The preferred variable
is `HF_TOKEN`; `HUGGINGFACE_HUB_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are also
accepted. Hugging Face documents `HF_TOKEN` and `HF_HOME` in its
[environment variable reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

## Rate Limits and Upload Resumability

Large dataset uploads can hit Hugging Face request or repository commit quotas.
This stage handles those limits by uploading one parquet directory group at a
time, retrying transient failures, and pausing between directory-scoped upload
calls when configured.

Resumability is local:

- `upload_extraction_state.sqlite3` records completed source batches, retained
  PR keys, stable global PR deduplication, and uploaded repo paths.
- Re-running `prepare` with the same output directory skips source parquet
  batches already marked complete.
- Re-running `upload` with the same output directory reuses the Hugging Face
  upload cache and records uploaded repo paths.
- Use a fresh output directory for an independent staged package.

## Token Handling

Set Hugging Face tokens only through environment variables:

- `HF_TOKEN`: preferred.
- `HUGGINGFACE_HUB_TOKEN`: fallback.
- `HUGGING_FACE_HUB_TOKEN`: fallback.

The token is used only for real uploads. `prepare` and `upload --dry-run` do not
require a token. Manifests record only whether a token was configured and the
environment-variable lookup order; they never write the token value.

PowerShell:

```powershell
$env:HF_TOKEN = "hf_..."
```

Bash:

```bash
export HF_TOKEN=hf_...
```

## Requirements

Docker is the recommended way to run upload-extraction because it isolates the
Python parquet stack and Hugging Face client from the host environment. The
image installs `post-processing/upload-extraction-data/requirements.txt`.

For Docker runs, you need:

- Docker.
- Local extraction output mounted read-only to `/data/input`.
- A local staging/output directory mounted writable to `/data/output`.
- A Hugging Face dataset repository id when uploading.
- `HF_TOKEN` with write access when uploading.

For local runs without Docker, use Python 3.12. The dependency file is
stage-scoped:

- `huggingface_hub`: repository creation and upload.
- `pandas`, `pyarrow`, and `numpy`: parquet reading/writing and schema handling.

## Step-by-Step Run: Docker

Use this path first unless you are actively debugging the Python process on the
host.

1. Prepare local paths.

   Use an extraction output root with this shape:

   ```text
   /absolute/path/to/extraction-output/
     data/
       humans/
       codex/
       claude/
       ...
     manifest.json
     extraction_run_manifest.json
   ```

   Create a writable staging directory:

   ```bash
   mkdir -p outputs/upload-extraction
   ```

2. Build the image from the repository root.

   ```bash
   docker build \
     -f post-processing/upload-extraction-data/Dockerfile \
     -t mosaic-upload-extraction:local \
     .
   ```

3. Check the container CLI.

   ```bash
   docker run --rm mosaic-upload-extraction:local \
     python post-processing/upload-extraction-data/run.py --help
   ```

4. Prepare a local public parquet package without uploading.

   ```bash
   docker run --rm \
     -v /absolute/path/to/extraction-output:/data/input:ro \
     -v "$(pwd)/outputs/upload-extraction:/data/output" \
     mosaic-upload-extraction:local \
     python post-processing/upload-extraction-data/run.py prepare
   ```

5. Inspect the prepared output.

   Check `outputs/upload-extraction/data/<cohort>/` for entity parquet batches,
   `outputs/upload-extraction/schema_manifest.json` for table definitions, and
   `outputs/upload-extraction/upload_extraction_manifest.json` for run metadata.

6. Create a dry-run upload plan.

   ```bash
   docker run --rm \
     -v "$(pwd)/outputs/upload-extraction:/data/output" \
     -e POST_PROCESSING_UPLOAD_EXTRACTION_REPO_ID="your-org/your-dataset" \
     mosaic-upload-extraction:local \
     python post-processing/upload-extraction-data/run.py upload --dry-run
   ```

   Inspect `outputs/upload-extraction/upload_extraction_plan.json` before a real
   upload.

7. Upload the prepared tree.

   ```bash
   export HF_TOKEN=hf_...

   docker run --rm \
     -e HF_TOKEN="$HF_TOKEN" \
     -e POST_PROCESSING_UPLOAD_EXTRACTION_REPO_ID="your-org/your-dataset" \
     -v "$(pwd)/outputs/upload-extraction:/data/output" \
     mosaic-upload-extraction:local \
     python post-processing/upload-extraction-data/run.py upload
   ```

8. Run prepare and upload in one command when the source and destination are
   already confirmed.

   ```bash
   docker run --rm \
     -e HF_TOKEN="$HF_TOKEN" \
     -e POST_PROCESSING_UPLOAD_EXTRACTION_REPO_ID="your-org/your-dataset" \
     -v /absolute/path/to/extraction-output:/data/input:ro \
     -v "$(pwd)/outputs/upload-extraction:/data/output" \
     mosaic-upload-extraction:local \
     python post-processing/upload-extraction-data/run.py all
   ```

## Step-by-Step Run: Local

Use the local path when Docker is unavailable.

1. Install dependencies from the repository root.

   ```bash
   python -m pip install -r post-processing/upload-extraction-data/requirements.txt
   ```

2. Check the CLI.

   ```bash
   python post-processing/upload-extraction-data/run.py --help
   ```

3. Prepare local public parquet output.

   ```bash
   python post-processing/upload-extraction-data/run.py prepare \
     --source-dir /absolute/path/to/extraction-output \
     --output-dir outputs/upload-extraction
   ```

4. Dry-run the upload.

   ```bash
   python post-processing/upload-extraction-data/run.py upload \
     --output-dir outputs/upload-extraction \
     --repo-id your-org/your-dataset \
     --dry-run
   ```

5. Upload.

   ```bash
   export HF_TOKEN=hf_...

   python post-processing/upload-extraction-data/run.py upload \
     --output-dir outputs/upload-extraction \
     --repo-id your-org/your-dataset
   ```

## CLI Reference

Check the Docker CLI first:

```bash
docker run --rm mosaic-upload-extraction:local \
  python post-processing/upload-extraction-data/run.py --help
```

For local runs:

```bash
python post-processing/upload-extraction-data/run.py --help
```

Common commands:

```bash
python post-processing/upload-extraction-data/run.py prepare \
  --source-dir outputs/extraction \
  --output-dir outputs/upload-extraction

python post-processing/upload-extraction-data/run.py upload \
  --output-dir outputs/upload-extraction \
  --repo-id your-org/your-dataset \
  --dry-run

python post-processing/upload-extraction-data/run.py all \
  --source-dir outputs/extraction \
  --output-dir outputs/upload-extraction \
  --repo-id your-org/your-dataset
```

Important CLI options:

- `prepare|upload|all`: command to run.
- `--source-dir`: extraction output root. Repeat to scan multiple roots.
- `--output-dir`: local staging directory.
- `--repo-id`: Hugging Face dataset repository id.
- `--dry-run`: write an upload plan without contacting Hugging Face.
- `--data-subdir`: staged data subdirectory, default `data`.
- `--output-batch-size`: rows per parquet shard.
- `--max-files-per-directory`: shard directories before this file count.
- `--parquet-compression`: parquet compression codec.
- `--state-db-filename`: SQLite state filename under the output directory.
- `--schema-version`: public extraction schema version recorded in rows and
  manifests.

Environment equivalents:

- `POST_PROCESSING_UPLOAD_EXTRACTION_SOURCE_DIRS`
- `POST_PROCESSING_UPLOAD_EXTRACTION_OUTPUT_DIR`
- `POST_PROCESSING_UPLOAD_EXTRACTION_REPO_ID`
- `POST_PROCESSING_UPLOAD_EXTRACTION_DRY_RUN`
- `POST_PROCESSING_UPLOAD_EXTRACTION_OUTPUT_BATCH_SIZE`
- `POST_PROCESSING_UPLOAD_EXTRACTION_MAX_FILES_PER_DIRECTORY`
- `POST_PROCESSING_UPLOAD_EXTRACTION_PARQUET_COMPRESSION`
- `HF_TOKEN`

## Output Layout

Prepared output:

```text
upload-extraction-output/
  README.md
  schema_manifest.json
  upload_extraction_manifest.json
  upload_extraction_plan.json
  upload_extraction_state.sqlite3
  data/
    codex/
      PullRequestRecords/
        shard-0000/
          pull_request_records_batch-000001.parquet
      AggregatedPullRequests/
        shard-0000/
          aggregated_pull_requests_batch-000001.parquet
      FileChangeRecords/
        shard-0000/
          file_change_records_batch-000001.parquet
      RepositoryRecords/
        shard-0000/
          repository_records_batch-000001.parquet
    humans/
      ...
```

Entity tables:

- `PullRequestRecords`: normalized PR-level rows.
- `AggregatedPullRequests`: one richer PR row with selected nested public fields
  for compatibility with downstream readers.
- `FileChangeRecords`: one row per retained changed file.
- `RepositoryRecords`: one row per retained base/head repository snapshot.

Metadata files:

- `README.md`: generated dataset card for the staged package.
- `schema_manifest.json`: entity schemas, primary keys, and foreign keys.
- `upload_extraction_manifest.json`: safe settings, counts, schema version, and
  upload-plan summary.
- `upload_extraction_plan.json`: dry-run upload file list and target repo paths.
- `upload_extraction_state.sqlite3`: local resume/dedup/upload state.

## Resume

Use the same `--output-dir` to resume a staged package.

Prepare resume behavior:

- Source parquet batches marked complete in `upload_extraction_state.sqlite3`
  are skipped.
- Existing parquet output batch numbers are detected so new batches continue at
  the next index.
- Stable PR deduplication prevents the same repository id and PR number from
  being retained twice across cohorts.

Upload resume behavior:

- `upload_large_folder` reuses its local cache under the staged output tree.
- Uploaded repo paths are recorded in SQLite.
- If an upload is interrupted, rerun `upload` with the same `--output-dir`.

Use a fresh output directory for a completely independent package.

## Troubleshooting

No source parquet files are discovered:

- Confirm `--source-dir` points at the extraction output root, not directly at a
  single cohort folder unless that is intentional.
- Expected extraction layout is `data/<cohort>/*.parquet`.
- In Docker, confirm the host path is mounted to `/data/input`.

Upload is skipped:

- Confirm `--repo-id` or `POST_PROCESSING_UPLOAD_EXTRACTION_REPO_ID` is set.
- Confirm `HF_TOKEN` is set for a real upload.
- `upload --dry-run` never contacts Hugging Face.

Upload hits rate limits:

- Rerun the same command after the cooldown window.
- Keep the same output directory so upload cache and SQLite state are reused.
- Reduce `--upload-large-folder-num-workers` if request pressure is too high.
- Increase `--upload-large-folder-directory-cooldown-seconds` for large exports.

Unexpected duplicate counts:

- The stable dedup key is repository id plus PR number when those fields are
  available.
- Use a fresh output directory when you need a fully independent package.

## Files of Interest

- `post-processing/upload-extraction-data/run.py`: CLI entrypoint and command
  dispatcher.
- `post-processing/upload-extraction-data/config/settings.py`: typed settings
  with CLI > environment > defaults precedence.
- `post-processing/upload-extraction-data/config/storage_config.py`: source,
  output, batching, compression, and retry defaults.
- `post-processing/upload-extraction-data/config/tokens_config.py`: Hugging Face
  token loading and redacted token metadata.
- `post-processing/upload-extraction-data/hf_dataset_uploader.py`: streaming
  conversion, local manifests, SQLite state, and upload orchestration.
- `post-processing/upload-extraction-data/Dockerfile`: Docker runtime.
- `post-processing/upload-extraction-data/Dockerfile.dockerignore`: Docker
  context exclusions for local data and tool caches.
- `post-processing/upload-extraction-data/requirements.txt`: stage-scoped Python
  dependencies.

The Python files use module docstrings for stage contracts and short comments
for non-obvious decisions such as schema stability, token redaction, retry
policy, and resumable state. Routine line-by-line behavior is intentionally left
to the code.
