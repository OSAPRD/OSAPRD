# Analysis Pipeline

This package reads local curation parquet outputs and optional topic
classification outputs, then writes local JSON summaries, plot-data JSON, and
publication-ready plots. It is file-system only: it does not scrape GitHub,
rerun curation tools, upload data, or mutate source parquet inputs.

Use one stable entrypoint:

```bash
python post-processing/analysis/run.py <pipeline> [options]
```

The available pipelines are:

- `dataset`: dataset counts, cohort balance, topic/domain coverage, and
  longitudinal snapshot availability.
- `refactoring`: refactoring prevalence, refactoring taxonomy summaries,
  characteristics analysis, and longitudinal refactoring persistence.
- `maintainability`: code-smell summaries, Mantyla category summaries,
  maintainability metric deltas, characteristics analysis, longitudinal
  maintainability trends, and optional Multimetric detail plots.
- `all`: run the three pipelines in order while keeping their output folders
  separate.

## Pipeline Flow

Analysis has four data-construction phases, followed by local persistence:

```text
CLI/env settings
  -> input loading: discover curation parquet and optional topic outputs
  -> streaming aggregation: read parquet rows into compact payloads
  -> analysis: compute summaries, statistical comparisons, and plot payloads
  -> persistence: write JSON summaries plus PDF/PNG/JSON plot artifacts
```

Settings are resolved in `post-processing/analysis/run.py` through
`AnalysisSettings`. CLI arguments take precedence over environment variables,
and environment variables take precedence over config defaults. The settings are
applied before pipeline modules are imported so existing env-backed config
modules resolve deterministically.

Input loading scans curation output roots recursively and discovers processed
parquet files by cohort. Topic-classification output is optional. When present,
it is used to add topic/domain dimensions to dataset, refactoring, and
maintainability characteristic analyses.

Streaming aggregation reads parquet batches with `pyarrow` and keeps compact
counts, numeric arrays, and per-cohort payloads in memory. This avoids loading
full curation exports into a dataframe or database.

Analysis then runs one or more independent pipelines:

- The dataset pipeline writes overall and per-cohort counts, sampling balance,
  language/popularity summaries, topic/domain coverage, and future-snapshot
  availability summaries.
- The refactoring pipeline reads original PR refactoring metrics from curation
  rows and writes prevalence, standardized refactoring type counts,
  Murphy-Hill summaries, characteristics heatmaps, and longitudinal
  retention/future-line-impact summaries.
- The maintainability pipeline reads code-smell, custom duplicated-lines
  density, and Multimetric-derived values from curation rows. It writes
  code-smell/Mantyla summaries, maintainability deltas, characteristics
  heatmaps, longitudinal metric trends, and optional Multimetric detail outputs.

Persistence writes local JSON result files, plot-data JSON, and paired PDF/PNG
figures under the configured analysis output directory.

## Data and Metrics Used

Analysis consumes outputs from the curation package. It does not rerun
refactoring tools, Multimetric, code-smell tools, or duplicated-lines-density
computation. Those metrics must already be present in the curation parquet
inputs.

Refactoring analysis uses:

- Standardized refactoring operation counts and type distributions.
- Murphy-Hill abstraction categories from stored curation fields or taxonomy
  reclassification.
- Original PR refactoring-zone persistence fields for longitudinal retention
  and future-line-touch summaries.

Maintainability analysis uses:

- Code-smell counts and standardized smell types from curation.
- Mantyla smell categories from stored curation fields or taxonomy
  reclassification.
- Curation's custom duplicated-lines density values.
- Multimetric-derived maintainability fields embedded in current curation
  parquet or stored in an external legacy Multimetric snapshot folder.

Topic/domain analysis uses topic-classification output when available:

- `repository_topics.jsonl`
- repository-to-PR mapping output from the topic-classification stage

If topic output is missing or not resolvable, core dataset/refactoring/
maintainability summaries still run, but domain-specific characteristic plots
are skipped.

## Multimetric Source

Current curation outputs embed Multimetric/custom-duplication snapshot rows
inside processed parquet. Older runs may store those rows in a separate folder
containing `multimetric_snapshot_metrics.parquet`.

Choose the source with `--multimetric-source`:

- `auto`: use embedded rows when present; otherwise use the external folder if
  configured.
- `input`: require embedded rows in curation parquet.
- `external`: require `--multimetric-output-dir`.
- `off`: skip Multimetric detail plots and results.

Use `external` only for older curated outputs where Multimetric detail rows were
not embedded in the processed parquet.

## Requirements

Docker is the recommended way to run analysis because it isolates the Python
parquet, SciPy, NumPy, and Matplotlib stack from the host environment. The image
installs `post-processing/analysis/requirements.txt` and copies only the
analysis package plus small taxonomy config files from `curation/config`.

For Docker runs, you need:

- Docker.
- Local curation parquet outputs mounted read-only to `/data/curation-data`.
- Optional topic-classification outputs mounted read-only to
  `/data/topic-classification`.
- A local output directory mounted to `/output`.

For local runs without Docker, use Python 3.12. The dependency file is
intentionally analysis-scoped:

- `pyarrow` for parquet streaming.
- `numpy` and `scipy` for statistical summaries and tests.
- `matplotlib` for plot rendering.

The analysis requirements pin `numpy==1.26.4` and `scipy==1.13.1`. If local
statistical calculations or plots fail while importing SciPy, use Docker or
install the pinned requirements into a clean environment.

## Step-by-Step Run: Docker

Use this path first unless you are actively developing analysis code on the
host.

1. Choose the input data source.

   Point analysis at the output directory produced by curation:

   ```text
   /path/to/curation-output/
     output/
       processed-data/
       aggregates/
       run_metadata_<cohort>_<timestamp>.json
   ```

2. Optionally choose topic-classification output.

   Domain/topic characteristic plots need a topic-classification run root or
   its `output/` folder:

   ```text
   /path/to/topic-classification-run/
     output/
       repository_topics.jsonl
       ...
   ```

   If you do not have topic output, mount an empty directory or leave the
   default path empty. Core summaries still run.

3. Prepare an output directory.

   ```bash
   mkdir -p outputs/analysis
   ```

4. Build the analysis image from the repository root.

   ```bash
   docker build -f post-processing/analysis/Dockerfile -t mosaic-analysis:local .
   ```

5. Check the container CLI.

   ```bash
   docker run --rm mosaic-analysis:local python -u post-processing/analysis/run.py --help
   ```

6. Run all analysis pipelines.

   ```bash
   docker run --rm \
     -v "$PWD/outputs/curation:/data/curation-data:ro" \
     -v "$PWD/post-processing/topic-classification/classification-runs:/data/topic-classification:ro" \
     -v "$PWD/outputs/analysis:/output" \
     mosaic-analysis:local \
     python -u post-processing/analysis/run.py all
   ```

7. Run one pipeline by changing the positional pipeline name.

   ```bash
   docker run --rm \
     -v "$PWD/outputs/curation:/data/curation-data:ro" \
     -v "$PWD/post-processing/topic-classification/classification-runs:/data/topic-classification:ro" \
     -v "$PWD/outputs/analysis:/output" \
     mosaic-analysis:local \
     python -u post-processing/analysis/run.py refactoring

   docker run --rm \
     -v "$PWD/outputs/curation:/data/curation-data:ro" \
     -v "$PWD/post-processing/topic-classification/classification-runs:/data/topic-classification:ro" \
     -v "$PWD/outputs/analysis:/output" \
     mosaic-analysis:local \
     python -u post-processing/analysis/run.py maintainability
   ```

8. Run old external Multimetric detail analysis when needed.

   ```bash
   docker run --rm \
     -v "$PWD/outputs/curation:/data/curation-data:ro" \
     -v "$PWD/outputs/legacy-multimetric:/data/multimetric:ro" \
     -v "$PWD/outputs/analysis:/output" \
     mosaic-analysis:local \
     python -u post-processing/analysis/run.py maintainability \
       --multimetric-source external \
       --multimetric-output-dir /data/multimetric
   ```

9. Inspect outputs.

   Check `outputs/analysis/` for pipeline result JSON files, companion
   `characteristics/` and `longitudinal/` outputs, and `plots/` folders
   containing PDF, PNG, and plot-data JSON files.

## Step-by-Step Run: Local

Use the local path when Docker is unavailable or when you need to debug the
Python process directly on the host.

1. Install dependencies from the repository root.

   ```bash
   python -m pip install -r post-processing/analysis/requirements.txt
   ```

2. Check the local CLI.

   ```bash
   python post-processing/analysis/run.py --help
   ```

3. Run all analysis pipelines.

   ```bash
   python post-processing/analysis/run.py all \
     --curation-data-dir outputs/curation \
     --topic-classification-output-dir post-processing/topic-classification/classification-runs \
     --analysis-output-dir outputs/analysis
   ```

4. Run one pipeline.

   ```bash
   python post-processing/analysis/run.py dataset \
     --curation-data-dir outputs/curation \
     --topic-classification-output-dir post-processing/topic-classification/classification-runs \
     --analysis-output-dir outputs/analysis

   python post-processing/analysis/run.py maintainability \
     --curation-data-dir outputs/curation \
     --topic-classification-output-dir post-processing/topic-classification/classification-runs \
     --analysis-output-dir outputs/analysis
   ```

5. Inspect outputs using the same output rules as the Docker run.

## CLI Reference

Check the Docker CLI first:

```bash
docker run --rm mosaic-analysis:local python -u post-processing/analysis/run.py --help
```

For local runs:

```bash
python post-processing/analysis/run.py --help
```

Common commands:

```bash
python post-processing/analysis/run.py all --curation-data-dir outputs/curation --analysis-output-dir outputs/analysis
python post-processing/analysis/run.py dataset --curation-data-dir outputs/curation --analysis-output-dir outputs/analysis
python post-processing/analysis/run.py refactoring --curation-data-dir outputs/curation --analysis-output-dir outputs/analysis
python post-processing/analysis/run.py maintainability --curation-data-dir outputs/curation --analysis-output-dir outputs/analysis
```

Useful options:

- `pipeline`: `dataset`, `refactoring`, `maintainability`, or `all`.
- `--curation-data-dir`: root containing curation parquet outputs.
- `--topic-classification-output-dir`: topic-classification output root.
- `--analysis-output-dir`: local analysis output root.
- `--excluded-agents`: comma-separated agent cohorts excluded from analysis.
- `--murphy-hill-count-source`: `taxonomy` or `stored`.
- `--mantyla-count-source`: `taxonomy` or `stored`.
- `--maintainability-require-refops` / `--no-maintainability-require-refops`:
  restrict maintainability analysis to PRs with mined refactoring operations.
- `--plot-mode` / `--no-plot-mode`: rewrite plots from compact payloads
  without extra plot-data exports.
- `--multimetric-source`: `auto`, `input`, `external`, or `off`.
- `--multimetric-output-dir`: legacy external Multimetric snapshot parquet
  root.

Run settings are assembled in `post-processing/analysis/config/settings.py`.
Storage defaults come from `post-processing/analysis/config/storage_config.py`.
Runtime behavior defaults come from
`post-processing/analysis/config/analysis_config.py`.

Supported environment variables:

- `POST_PROCESSING_ANALYSIS_PIPELINE`
- `POST_PROCESSING_ANALYSIS_CURATION_DATA_DIR`
- `POST_PROCESSING_ANALYSIS_TOPIC_CLASSIFICATION_OUTPUT_DIR`
- `POST_PROCESSING_ANALYSIS_OUTPUT_DIR`
- `POST_PROCESSING_ANALYSIS_EXCLUDED_AGENTS`
- `POST_PROCESSING_ANALYSIS_MURPHY_HILL_COUNT_SOURCE`
- `POST_PROCESSING_ANALYSIS_MANTYLA_COUNT_SOURCE`
- `POST_PROCESSING_ANALYSIS_MAINTAINABILITY_REQUIRE_REFOPS`
- `POST_PROCESSING_ANALYSIS_PLOT_MODE`
- `POST_PROCESSING_ANALYSIS_MULTIMETRIC_SOURCE`
- `POST_PROCESSING_ANALYSIS_MULTIMETRIC_OUTPUT_DIR`

## Output Layout

Outputs are grouped by analysis component:

```text
ANALYSIS_OUTPUT_DIR/
  dataset/
    data_analysis_results.json
    plots/
      *.pdf
      *.png
      *.json
  refactoring/
    refactoring_analysis_results.json
    plots/
  maintainability/
    maintainability_analysis_results.json
    maintainability_multimetrics_results.json
    plots/
  characteristics/
    refactoring/
      characteristics_refactoring_results.json
      plots/
    maintainability/
      characteristics_maintainability_metrics_results.json
      plots/
    maintainability-multimetric/
      characteristics_maintainability_multimetric_results.json
      plots/
  longitudinal/
    refactoring/
      longitudinal_refactoring_results.json
      plots/
    maintainability/
      longitudinal_maintainability_metrics_results.json
      plots/
    maintainability-multimetric/
      longitudinal_maintainability_multimetric_results.json
      plots/
```

The exact plot set depends on the selected pipeline and whether topic/domain and
Multimetric detail inputs are available. Plot writers emit both `*.pdf` and
`*.png`; when plot-data writing is enabled, they also emit sibling `*.json`
payloads.

## Resume and Re-Runs

Analysis does not keep checkpoints because it does not call external services or
run long source-code tools. Re-running the same command recomputes the selected
pipeline and overwrites the corresponding JSON and plot outputs.

Use a fresh `--analysis-output-dir` for a fully independent analysis run. Use
`--plot-mode` when you want to regenerate plots from compact payloads without
writing extra plot-data exports.

## Troubleshooting

**No curation parquet files found**

Check `--curation-data-dir`. It should point at the curation output root that
contains `output/processed-data/` folders, or a parent directory containing
cohort curation outputs.

**Domain/topic plots are missing**

Check `--topic-classification-output-dir`. It must point at a
topic-classification run root or its `output/` folder containing
`repository_topics.jsonl`. Core analysis can still run without this input.

**SciPy or NumPy import errors locally**

Use Docker, or install `post-processing/analysis/requirements.txt` into a clean
Python 3.12 environment. The pinned local runtime uses `numpy==1.26.4` and
`scipy==1.13.1`.

**External Multimetric files not found**

Use `--multimetric-source input` for current curation outputs with embedded
snapshot rows. Use `--multimetric-source external --multimetric-output-dir ...`
only when the directory contains `multimetric_snapshot_metrics.parquet`.

**Plot rendering fails because fonts or GUI backends are unavailable**

Use Docker. The image installs font packages and Matplotlib runs headlessly.

## Files of Interest

- `post-processing/analysis/run.py`: CLI entry point and pipeline selector.
- `post-processing/analysis/config/settings.py`: run-level settings resolution.
- `post-processing/analysis/config/storage_config.py`: local input/output
  defaults.
- `post-processing/analysis/config/analysis_config.py`: env-backed analysis
  behavior constants.
- `post-processing/analysis/pipelines/data_analysis_pipeline.py`: dataset
  composition, balance, and snapshot-availability analysis.
- `post-processing/analysis/pipelines/refactoring_analysis_pipeline.py`:
  refactoring prevalence and companion analysis orchestration.
- `post-processing/analysis/pipelines/maintainability_analysis_pipeline.py`:
  code-smell, maintainability, Multimetric, and companion analysis
  orchestration.
- `post-processing/analysis/pipelines/characteristics_refactoring_pipeline.py`:
  refactoring results by language, popularity, and topic/domain.
- `post-processing/analysis/pipelines/characteristics_maintainability_metrics_pipeline.py`:
  maintainability results by language, popularity, and topic/domain.
- `post-processing/analysis/pipelines/maintainability_multimetrics_pipeline.py`:
  embedded or external Multimetric detail summaries and plots.
- `post-processing/analysis/pipelines/longitudinal_refactoring_pipeline.py`:
  refactoring retention and future-line-touch analysis.
- `post-processing/analysis/pipelines/longitudinal_maintainability_metrics_pipeline.py`:
  maintainability evolution across future snapshots.
- `post-processing/analysis/utility/streaming_parquet_utility.py`: low-memory
  parquet streaming and common PR field extraction.
- `post-processing/analysis/utility/topic_groups_utility.py`: topic output
  loading and domain mapping.
- `post-processing/analysis/utility/balance_statistics_utility.py`: shared
  statistical tests, effect sizes, and corrections.
- `post-processing/analysis/plotters/`: deterministic PDF/PNG/JSON plot
  writers.
