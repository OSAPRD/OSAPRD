# Topic Classification

Repository topic classification has one stable entrypoint:

```bash
python post-processing/topic-classification/run.py <stage> [options]
```

The pipeline has four stages:

1. `extract`: collect topic-labeled repositories from GitHub.
2. `prepare`: filter, preprocess, and sample the training data.
3. `train`: train and export the topic model bundle.
4. `classify`: classify repositories from curation outputs.

## Prerequisites

Download the GitHub topic catalog from
[github/explore/topics](https://github.com/github/explore/tree/main/topics) and
place it in the local topic-classification package before running preparation,
training, classification, or building the Docker image.

Expected local layout:

```text
post-processing/topic-classification/github-topics/
  python/
    index.md
  javascript/
    index.md
  ...
```

Bash:

```bash
git clone --depth 1 https://github.com/github/explore.git /tmp/github-explore
mkdir -p post-processing/topic-classification/github-topics
cp -R /tmp/github-explore/topics/. post-processing/topic-classification/github-topics/
```

PowerShell:

```powershell
git clone --depth 1 https://github.com/github/explore.git $env:TEMP\github-explore
New-Item -ItemType Directory -Force post-processing/topic-classification/github-topics
Copy-Item -Recurse -Force $env:TEMP\github-explore\topics\* post-processing/topic-classification/github-topics\
```

The default topic-rule generator reads that catalog and writes the generated
preprocessing profile used by the active pipeline:

```bash
python post-processing/topic-classification/data-preparation/generate_github_topic_rules.py \
  --topics-root post-processing/topic-classification/github-topics \
  --output-root post-processing/topic-classification/data-preparation/generated/github-topics
```

For Docker, download the catalog before `docker build` because the Dockerfile
copies the local `post-processing` directory into the image.

## Run With Docker

Build from the `post-processing` directory so the Docker context contains the
shared config and utility modules:

```bash
cd post-processing
docker build -f topic-classification/Dockerfile -t mosaic-topic-classification .
```

Use `GITHUB_TOKENS` for one or more comma-separated GitHub tokens. `GITHUB_TOKEN`
also works for a single token.

### 1. Extract Data From GitHub

```bash
docker run --rm \
  -e GITHUB_TOKENS="$GITHUB_TOKENS" \
  -v "$PWD/topic-classification/runs:/data/topic-training-runs" \
  mosaic-topic-classification \
  python -u post-processing/topic-classification/run.py extract \
    --start-date 2024-01-01 \
    --end-date 2025-12-31 \
    --target-repos 200000 \
    --max-pages 10 \
    --output-dir /data/topic-training-runs
```

The command writes one timestamped extraction run containing repository search
candidates, enriched repository records, file lists, wiki/README cache metadata,
and `topic_training_extraction_manifest.json`.

### 2. Filter, Preprocess, And Sample

Replace `<extract-run>` with the folder created by stage 1.

```bash
docker run --rm \
  -v "$PWD/topic-classification/runs:/data/topic-training-runs" \
  mosaic-topic-classification \
  python -u post-processing/topic-classification/run.py prepare \
    --input-run /data/topic-training-runs/<extract-run> \
    --target-repos 200000 \
    --train-fraction 0.8 \
    --seed 42
```

This stage first creates filtered/preprocessed records, then runs deterministic
sampling and train/test splitting. The sampled output includes `topics_train.csv`,
`topics_test.csv`, `sample_train_test_manifest.json`, and sampled repository
metadata.

### 3. Train Model

Point `--train-csv`, `--test-csv`, and `--sample-manifest` at the sampled output
from stage 2.

```bash
docker run --rm \
  -v "$PWD/topic-classification/runs:/data/topic-training-runs" \
  -v "$PWD/topic-classification/model-output:/model-output" \
  mosaic-topic-classification \
  python -u post-processing/topic-classification/run.py train \
    --train-csv /data/topic-training-runs/<extract-run>/training-data/create-training-data/sampled-training-data/<sample>/topics_train.csv \
    --test-csv /data/topic-training-runs/<extract-run>/training-data/create-training-data/sampled-training-data/<sample>/topics_test.csv \
    --sample-manifest /data/topic-training-runs/<extract-run>/training-data/create-training-data/sampled-training-data/<sample>/sample_train_test_manifest.json \
    --output-dir /model-output
```

The trained bundle is written as `topic_model_bundle.joblib` with labels,
metrics, preprocessing artifacts, and a training manifest.

### 4. Run Classification

```bash
docker run --rm \
  -e GITHUB_TOKENS="$GITHUB_TOKENS" \
  -v "$PWD/../outputs/curation:/data/curation-outputs:ro" \
  -v "$PWD/topic-classification/model-output:/model-output:ro" \
  -v "$PWD/topic-classification/classification-runs:/data/classification-runs" \
  -v "$PWD/topic-classification/wiki-cache:/wiki-cache" \
  mosaic-topic-classification \
  python -u post-processing/topic-classification/run.py classify \
    --input-dir /data/curation-outputs \
    --model-bundle /model-output/topic_model_bundle.joblib \
    --runs-dir /data/classification-runs \
    --wiki-cache-dir /wiki-cache
```

Classification writes a new run folder containing repository-level predictions,
repository-to-PR mapping rows, filtered repository records, failure records, and
an output manifest.

## Run Locally

Install dependencies:

```bash
python -m pip install -r post-processing/topic-classification/requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

Then run the same four commands locally:

```bash
python post-processing/topic-classification/run.py extract --output-dir post-processing/topic-classification/runs
python post-processing/topic-classification/run.py prepare --input-run post-processing/topic-classification/runs/<extract-run>
python post-processing/topic-classification/run.py train --train-csv <topics_train.csv> --test-csv <topics_test.csv> --output-dir post-processing/topic-classification/model-output
python post-processing/topic-classification/run.py classify --input-dir outputs/curation --model-bundle post-processing/topic-classification/model-output/topic_model_bundle.joblib
```

## Stage Design

The stage folders each include a local README with file-level details:

- `extract-training-data/README.md`
- `create-training-data/README.md`
- `sample-training-data/README.md`
- `train-model/README.md`
- `classify-topics/README.md`
- `data-preparation/README.md`
- `data-preparation/generated/github-topics/README.md`

Python files carry module/class/function docstrings. CSV, JSON, TXT, notebook,
and license artifacts are documented from these README files instead of being
modified in place.

### Extraction

The extraction stage searches GitHub repositories with topics over a configured
creation-date window. It uses date buckets, page limits, checkpointing, and token
rotation to keep long runs restartable under GitHub API rate limits. Repository
metadata, topics, file lists, README text, and optional wiki text are persisted
as JSONL and manifest files under one run directory.

### Filtering, Preprocessing, And Sampling

The preparation stage reads an extraction run and applies repository-level
filters before building model-ready text and label vectors. It removes unusable
records, normalizes topic labels with the generated GitHub topic rules, applies
frequency thresholds, writes encoded records, and then performs deterministic
sampling across repository popularity and creation-time strata.

### Training

The training stage reads sampled `topics_train.csv` and `topics_test.csv`,
validates the topic label universe, trains the TF-IDF plus one-vs-rest logistic
regression model, evaluates top-k metrics, and exports a complete inference
bundle.

### Classification

The classification stage reads curation outputs, builds repository feature text,
fetches missing README/wiki text when enabled, applies the trained model, filters
predicted topics by score, and writes repository-topic plus repository-PR outputs.

## Configuration

Command line flags override environment variables. The most common environment
variables are:

- `GITHUB_TOKENS`: comma-separated GitHub tokens.
- `GITHUB_TOKEN`: single GitHub token fallback.
- `POST_PROCESSING_TOPIC_TRAINING_OUTPUT_DIR`: default extraction output root.
- `POST_PROCESSING_TOPIC_PREPARE_INPUT_RUN`: default prepare input run.
- `POST_PROCESSING_TOPIC_CLASSIFICATION_MODEL_BUNDLE_PATH`: model bundle path.
- `POST_PROCESSING_CURATION_OUTPUTS_DIR`: classification input root.
- `POST_PROCESSING_TOPIC_CLASSIFICATION_RUNS_DIR`: classification output root.
- `POST_PROCESSING_TOPIC_CLASSIFICATION_WORKERS`: classification worker count.
  The default is `1` so Docker and Windows local runs use the same stable path.
- `POST_PROCESSING_TOPIC_CLASSIFICATION_REQUIRE_RAW_FREQUENCY_ARTIFACTS`: set
  to `true` only when the model was trained with raw text and file-name
  frequency artifacts.

`post-processing/config/tokens_config.py` remains supported for local runs, but
environment variables are preferred for Docker and automation.
