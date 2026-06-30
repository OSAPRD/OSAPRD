# Curation Pipeline

This package turns local pull-request parquet data into curated local outputs for
refactoring and maintainability analysis. It reads either local extraction
output or a downloaded Hugging Face parquet export, samples and filters PRs,
hydrates source snapshots from GitHub, mines original PR refactorings, computes
[Multimetric](https://github.com/priv-kweihmann/multimetric) maintainability
metrics plus custom duplicated-lines density, and writes local parquet and JSON
artifacts.

The curation stage has one execution path and one metrics mode:

- Refactoring tools run only on the original PR `before -> after` comparison.
- Future snapshots are hydrated only for PRs selected for longitudinal analysis.
- Future refactoring persistence is measured by tracking original PR
  refactoring-zone lines through later commits.
- Future snapshots do not rerun refactoring tools.
- Maintainability metrics use
  [Multimetric](https://github.com/priv-kweihmann/multimetric) plus the custom
  duplicated-lines density implementation in
  `curation/metrics/duplication_metrics.py`.
- Code-smell detection runs for each available maintainability snapshot, using
  [DesigniteJava](https://www.designite-tools.com/products-dj),
  [DesignitePython/DPy](https://www.designite-tools.com/products-dpy),
  [PMD](https://pmd.github.io/), [ESLint](https://eslint.org/),
  [Cppcheck](https://cppcheck.sourceforge.io/), and
  [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) when the matching
  language and tool configuration are present.
- SonarQube, Sonar Scanner, SonarJS, and alternate maintainability
  backends are not part of the active curation package.
- Uploading or publishing curated data is outside this package.

## Pipeline Flow

Curation has eight data-construction phases, followed by local persistence:

```text
CLI/env settings
  -> input loading: read local extraction, Hugging Face, or old curated parquet
  -> preprocessing: keep merged source-code PRs in supported languages
  -> sampling: build the main cohort sample
  -> longitudinal selection: choose PRs for future snapshots
  -> hydration: clone/fetch repositories and export source snapshots
  -> metrics: run original PR refactoring mining and maintainability metrics
  -> persistence: write processed parquet, aggregate JSON, and run metadata
```

Settings are resolved in `curation/run.py` through `CurationSettings`. CLI
arguments take precedence over environment variables, and environment variables
take precedence over config defaults. The settings are applied before heavy
pipeline modules are imported so lower-level config constants resolve
deterministically.

Input loading scans local parquet roots recursively with
`curation/utility/loader.py` and filters rows to the selected cohort. Supported
input formats are `extractionpullrequests`, `fullpullrequests_sharded`, and
`curation_processed`.

Preprocessing keeps merged PRs that modify source files in the supported
languages: C++, Java, JavaScript, and Python. Multi-language PRs are assigned
one effective dominant language for sampling and tool selection.

Sampling builds a stratified sample by language, creation-time bucket, and
repository popularity. Popularity buckets use tie-aware quantiles so repositories
with the same star count are not split across buckets. Optional sample-history
input excludes PRs already used in previous samples.

Hydration clones or fetches each base repository, resolves the base, head,
merge, and future commits, and exports PR-scoped source snapshots: `before`,
`after`, and available future snapshots. Hydration also records PR-file diff
metadata used to track original changed lines through later commits.

Metric computation always runs two stages:

- `RefactoringMetricsStage`: mines original PR refactorings on the
  `before -> after` transition with
  [RefactoringMiner](https://github.com/tsantalis/RefactoringMiner),
  [RefDiff](https://github.com/aserg-ufmg/RefDiff), and
  [RefactoringMiner++](https://github.com/benzoinoo/RefactoringMinerPP) where
  applicable.
- `MaintainabilityMetricsStage`: runs
  [Multimetric](https://github.com/priv-kweihmann/multimetric) on hydrated
  snapshots, merges in custom duplicated-lines density, and attaches code-smell
  findings from the configured smell tools.

Persistence writes per-PR aggregate JSON, processed parquet batches,
checkpoint/progress records, run errors, and run metadata. These outputs are
local-only.

## GitHub and Tools Used

Curation uses GitHub through local `git` operations rather than broad API search.
Even when PR metadata already exists locally, the pipeline still needs GitHub
network access to clone or fetch repositories and resolve future commits.

The active refactoring tools are:

- [RefactoringMiner](https://github.com/tsantalis/RefactoringMiner):
  original PR refactoring mining for Java and Python paths.
- [RefDiff](https://github.com/aserg-ufmg/RefDiff): original PR
  refactoring mining for JavaScript paths.
- [RefactoringMiner++](https://github.com/benzoinoo/RefactoringMinerPP):
  original PR refactoring mining for C++ paths.

The active maintainability backend is:

- [Multimetric](https://github.com/priv-kweihmann/multimetric):
  maintainability index, cyclomatic complexity, Halstead metrics, fan out,
  comment ratio, and related source metrics.
- Custom duplicated-lines density: deterministic in-repository implementation
  used instead of Multimetric's duplication value.

The active code-smell tools are language-gated:

| Snapshot language | Tools                | Notes                                                                                                |
| ----------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| Java              | [DesigniteJava](https://www.designite-tools.com/products-dj), [PMD](https://pmd.github.io/) | DesigniteJava requires a local or mounted `DesigniteJava.jar`. PMD is installed in the Docker image. |
| Python            | [DesignitePython/DPy](https://www.designite-tools.com/products-dpy) | DPy must be installed locally, mounted into Docker, or provided by a custom image.                   |
| C++               | [Cppcheck](https://cppcheck.sourceforge.io/), [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) | Both tools are installed in the Docker image. `clang-tidy` runs per source file by default.          |
| JavaScript        | [ESLint](https://eslint.org/) | ESLint 8 is installed in the Docker image and runs with a config-independent rule set by default.    |

Each tool writes raw stdout/stderr and parsed findings under the snapshot's
`maintainability/` folder. Missing tools are recorded as `tool_not_configured`;
they do not stop Multimetric or custom duplication metrics for the snapshot.

## Token Handling

Set GitHub tokens only through environment variables:

- `GITHUB_TOKENS`: comma-separated or OS-path-separator-separated tokens.
- `GITHUB_TOKEN`: single-token fallback.

`GITHUB_TOKENS` takes precedence. During hydration, the token manager rotates
tokens across clone and fetch attempts. Invalid credentials are marked unusable;
on other Git failures, the next token is tried before the PR is marked failed.
This reduces failed hydration caused by credential-specific limits, repository
access differences, or transient network failures.

## Cohorts and Input Formats

Use one explicit cohort per run:

- `human`: human-authored PR cohort.
- `agentic`: all configured agent groups.
- `<agent>`: one configured agent key such as `codex`, `claude`, `copilot`,
  `cursor`, `devin`, `jules`, or `junie`.

Use one local PR data source:

- Extraction output from this repository, using
  `--input-format extractionpullrequests`.
- A downloaded Hugging Face parquet export from
  <https://huggingface.co/datasets/OSAPRD/OSAPRD/> in the legacy sharded
  layout, using `--input-format fullpullrequests_sharded`.
- Previously curated parquet, using `--input-format curation_processed`, only
  when read-only compatibility with old processed rows is needed.

## Requirements

Docker is the recommended way to run curation because it isolates Python,
Java, Git, and native C++ tool dependencies from the host environment. The image
installs `curation/requirements.txt`, validates
[Multimetric](https://github.com/priv-kweihmann/multimetric), and builds
[RefactoringMiner++](https://github.com/benzoinoo/RefactoringMinerPP) during
the image build. The image also installs and validates
[PMD](https://pmd.github.io/), [ESLint](https://eslint.org/),
[Cppcheck](https://cppcheck.sourceforge.io/), and
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) for open-source smell
detection.

For Docker runs, you need:

- Docker.
- Network access to GitHub.
- One local PR data source mounted read-only to `/data/input`.
- One or more GitHub tokens in `GITHUB_TOKENS` or `GITHUB_TOKEN`.
- A local output directory mounted to `/data/output`.
- Enough disk for repository clones, source snapshots, metrics JSON, and parquet
  batches.
- Optional mounted Designite tools when Designite findings are required. The
  Docker image includes PMD, ESLint, Cppcheck, and clang-tidy, but it does not
  download `DesigniteJava.jar` or DPy because those tools have separate
  distribution terms.

For local runs without Docker, use Python 3.12 and provide the same runtime
tools that the Docker image installs:

- `git` and `git-lfs`.
- Java runtime compatible with the bundled refactoring tools.
- [RefactoringMiner](https://github.com/tsantalis/RefactoringMiner) under
  `curation/tools/runtime/refactoringminer`.
- [RefDiff](https://github.com/aserg-ufmg/RefDiff) under
  `curation/tools/runtime/refdiff`.
- [RefactoringMiner++](https://github.com/benzoinoo/RefactoringMinerPP) under
  `curation/tools/runtime/refactoringminerpp`.
- [`multimetric==2.4.4`](https://github.com/priv-kweihmann/multimetric) from
  `curation/requirements.txt`.
- [PMD](https://pmd.github.io/), [ESLint 8](https://eslint.org/),
  [Cppcheck](https://cppcheck.sourceforge.io/), and
  [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) on `PATH`, or command
  templates configured through `CURATION_PMD_COMMAND_TEMPLATE`,
  `CURATION_ESLINT_COMMAND_TEMPLATE`, `CURATION_CPPCHECK_COMMAND_TEMPLATE`, and
  `CURATION_CLANG_TIDY_COMMAND_TEMPLATE`.
- [DesigniteJava](https://www.designite-tools.com/products-dj) and
  [DesignitePython/DPy](https://www.designite-tools.com/products-dpy) if those
  findings are required. Configure them with `CURATION_DESIGNITE_JAVA_JAR`,
  `CURATION_DESIGNITE_JAVA_COMMAND_TEMPLATE`, and
  `CURATION_DESIGNITE_PYTHON_COMMAND_TEMPLATE`.

Set one or more GitHub tokens:

```bash
export GITHUB_TOKENS=ghp_...,...   # preferred for rotation
export GITHUB_TOKEN=ghp_...        # fallback
```

PowerShell:

```powershell
$env:GITHUB_TOKENS = "ghp_...,..."
```

For local runs, confirm the active tools before starting:

```bash
python -m pip install -r curation/requirements.txt
python -m curation.run --help
multimetric --help
pmd --version
eslint --version
cppcheck --version
clang-tidy --version
```

If DesigniteJava or DPy are installed outside `PATH`, set the command-template
environment variables before running. Command templates may use `{root}` for the
snapshot source root and `{out}` for the tool output directory. The
`CURATION_CLANG_TIDY_COMMAND_TEMPLATE` may also use `{file}` because clang-tidy
is invoked once per C/C++ source file. Other braces are left for the underlying
tool, which is why Cppcheck can keep its own `{file}:{line}:...` output
template.

## Reconstructing `curation/tools`

`curation/tools` is a generated/local tool cache. It is safe to delete the
large local contents when you need a clean workspace, provided you recreate the
runtime layout below before building Docker or running curation locally. A clean
checkout may keep small wrapper files under `curation/tools/bin`; if you delete
the whole directory in a working tree, restore tracked wrapper files with
`git restore curation/tools/bin` or start from a clean checkout.

The Docker image builds RefactoringMiner++ itself, and installs PMD, ESLint,
Cppcheck, clang-tidy, Java, and Python dependencies. Before `docker build`, the
host only needs these generated runtime directories:

```text
curation/tools/runtime/refactoringminer/
curation/tools/runtime/refdiff/
```

Use these commands from the repository root on Linux/macOS or Git Bash:
they require `curl`, `unzip`, `git`, and a JDK capable of running Gradle.

```bash
set -euo pipefail

TOOLS_TMP="${TMPDIR:-/tmp}/mosaic-curation-tools"
rm -rf "$TOOLS_TMP"
mkdir -p "$TOOLS_TMP" curation/tools/runtime

curl -L \
  -o "$TOOLS_TMP/RefactoringMiner-3.1.3.zip" \
  "https://github.com/tsantalis/RefactoringMiner/releases/download/3.1.3/RefactoringMiner-3.1.3.zip"
unzip -q "$TOOLS_TMP/RefactoringMiner-3.1.3.zip" -d "$TOOLS_TMP/refactoringminer"
rm -rf curation/tools/runtime/refactoringminer
mkdir -p curation/tools/runtime/refactoringminer
cp -R "$TOOLS_TMP/refactoringminer/RefactoringMiner-3.1.3/." \
  curation/tools/runtime/refactoringminer/

git clone https://github.com/aserg-ufmg/RefDiff.git "$TOOLS_TMP/RefDiff"
git -C "$TOOLS_TMP/RefDiff" checkout 889b0bfbf2c18726d44f077371966606232cca0b
(cd "$TOOLS_TMP/RefDiff" && ./gradlew :refdiff-example:installDist)
rm -rf curation/tools/runtime/refdiff
mkdir -p curation/tools/runtime/refdiff
cp -R "$TOOLS_TMP/RefDiff/refdiff-example/build/install/refdiff-example/." \
  curation/tools/runtime/refdiff/
chmod +x curation/tools/runtime/refactoringminer/bin/RefactoringMiner
chmod +x curation/tools/runtime/refdiff/bin/refdiff-example
```

PowerShell equivalent:
it requires `git` and a JDK on `PATH` so `java` and `jar` are available.

```powershell
$tmp = Join-Path $env:TEMP "mosaic-curation-tools"
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp, "curation/tools/runtime" | Out-Null

$rmZip = Join-Path $tmp "RefactoringMiner-3.1.3.zip"
Invoke-WebRequest `
  -Uri "https://github.com/tsantalis/RefactoringMiner/releases/download/3.1.3/RefactoringMiner-3.1.3.zip" `
  -OutFile $rmZip
Expand-Archive -Force -Path $rmZip -DestinationPath (Join-Path $tmp "refactoringminer")
Remove-Item -Recurse -Force "curation/tools/runtime/refactoringminer" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "curation/tools/runtime/refactoringminer" | Out-Null
Copy-Item -Recurse `
  (Join-Path $tmp "refactoringminer/RefactoringMiner-3.1.3/*") `
  "curation/tools/runtime/refactoringminer"

git clone https://github.com/aserg-ufmg/RefDiff.git (Join-Path $tmp "RefDiff")
git -C (Join-Path $tmp "RefDiff") checkout 889b0bfbf2c18726d44f077371966606232cca0b
& (Join-Path $tmp "RefDiff/gradlew.bat") :refdiff-example:installDist
Remove-Item -Recurse -Force "curation/tools/runtime/refdiff" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "curation/tools/runtime/refdiff" | Out-Null
Copy-Item -Recurse `
  (Join-Path $tmp "RefDiff/refdiff-example/build/install/refdiff-example/*") `
  "curation/tools/runtime/refdiff"

# Local Windows ReffDiff runs need the J2V8 native DLL next to the runtime.
New-Item -ItemType Directory -Force -Path "curation/tools/runtime/refdiff/native" | Out-Null
Push-Location "curation/tools/runtime/refdiff/native"
jar xf "../lib/j2v8_win32_x86_64-4.6.0.jar" "libj2v8_win32_x86_64.dll"
Rename-Item -Force "libj2v8_win32_x86_64.dll" "j2v8_win32_x86_64.dll"
Pop-Location
```

For local C++ runs without Docker, also provide RefactoringMiner++ under
`curation/tools/runtime/refactoringminerpp` or set `REFACTORING_MINER_PP_COMMAND`
to a compatible local executable. Docker users do not need to do this on the
host because the image builds RefactoringMiner++ during `docker build`.

## Step-by-Step Run: Docker

Use this path first unless you are actively developing curation code on the
host.

1. Choose the input data source.

   For extraction output, point curation at the extraction output root:

   ```text
   /path/to/extraction-output/
     data/
       humans/
       codex/
       claude/
       ...
     manifest.json
     extraction_run_manifest.json
   ```

   For a Hugging Face download, use the OSAPRD dataset at
   <https://huggingface.co/datasets/OSAPRD/OSAPRD/>, keep the downloaded parquet
   folder structure intact, and use `--input-format fullpullrequests_sharded`.

2. Choose the cohort and sample sizes.

   Use `agentic` for all configured agent groups, `human` for the human cohort,
   or one configured agent such as `codex`, `claude`, `devin`, `jules`, or
   `junie`. Use a small `--target-prs` and `--longitudinal-prs` for a smoke run.

3. Prepare an output directory.

   ```bash
   mkdir -p /absolute/path/to/curation-output
   ```

4. Set GitHub tokens.

   ```bash
   export GITHUB_TOKENS=ghp_...,...
   ```

   PowerShell:

   ```powershell
   $env:GITHUB_TOKENS = "ghp_...,..."
   ```

   Optional: configure Designite tools if you want DesigniteJava or DPy smell
   findings in Docker. PMD, ESLint, Cppcheck, and clang-tidy are already
   installed in the image.

   ```bash
   # Example: mount a directory containing DesigniteJava.jar and a DPy executable.
   DESIGNITE_TOOLS=/absolute/path/to/designite-tools
   ```

   Add these flags to the `docker run` command when using that mount:

   ```bash
   -v "$DESIGNITE_TOOLS:/opt/designite:ro" \
   -e CURATION_DESIGNITE_JAVA_JAR=/opt/designite/DesigniteJava.jar \
   -e CURATION_DESIGNITE_PYTHON_COMMAND_TEMPLATE="/opt/designite/dpy analyze -i {root} -o {out}"
   ```

5. Recreate the generated refactoring tool cache if `curation/tools` was
   removed.

   Follow `Reconstructing curation/tools` above. This must be done before
   `docker build` because the image copies the RefactoringMiner and RefDiff
   runtimes from the local checkout.

6. Build the curation image from the repository root.

   ```bash
   docker build -f curation/Dockerfile -t mosaic-curation:local .
   ```

7. Check the container CLI.

   ```bash
   docker run --rm mosaic-curation:local --help
   ```

   Optional: check the bundled open-source smell tools.

   ```bash
   docker run --rm --entrypoint bash mosaic-curation:local -lc "pmd --version && eslint --version && cppcheck --version && clang-tidy --version"
   ```

8. Run curation for extracted data.

   ```bash
   docker run --rm \
     -e GITHUB_TOKENS="$GITHUB_TOKENS" \
     -v /absolute/path/to/extraction-output:/data/input:ro \
     -v /absolute/path/to/curation-output:/data/output \
     mosaic-curation:local \
       --cohort agentic \
       --input-dir /data/input \
       --input-format extractionpullrequests \
       --output-dir /data/output \
       --target-prs 50000 \
       --longitudinal-prs 5000
   ```

9. Run curation for a downloaded Hugging Face sharded export.

   ```bash
   docker run --rm \
     -e GITHUB_TOKENS="$GITHUB_TOKENS" \
     -v /absolute/path/to/hf-download:/data/input:ro \
     -v /absolute/path/to/curation-output:/data/output \
     mosaic-curation:local \
       --cohort agentic \
       --input-dir /data/input \
       --input-format fullpullrequests_sharded \
       --output-dir /data/output \
       --target-prs 50000 \
       --longitudinal-prs 5000
   ```

9. Run one agent by changing `--cohort`.

   ```bash
   docker run --rm \
     -e GITHUB_TOKENS="$GITHUB_TOKENS" \
     -v /absolute/path/to/extraction-output:/data/input:ro \
     -v /absolute/path/to/curation-output:/data/output \
     mosaic-curation:local \
       --cohort codex \
       --input-dir /data/input \
       --input-format extractionpullrequests \
       --output-dir /data/output \
       --target-prs 1000 \
       --longitudinal-prs 100
   ```

10. Inspect outputs.

    Check `/absolute/path/to/curation-output/output/processed-data/` for
    parquet batches, `/absolute/path/to/curation-output/output/aggregates/` for
    per-PR aggregate JSON, and
    `/absolute/path/to/curation-output/output/run_metadata_<cohort>_<timestamp>.json`
    for run metadata.

## Step-by-Step Run: Local

Use the local path when Docker is unavailable or when you need to debug the
Python process directly on the host.

1. Install dependencies from the repository root.

   ```bash
   python -m pip install -r curation/requirements.txt
   ```

2. Recreate `curation/tools` if it was removed.

   Follow `Reconstructing curation/tools` above. For local C++ refactoring
   metrics, also provide `curation/tools/runtime/refactoringminerpp` or set
   `REFACTORING_MINER_PP_COMMAND`.

3. Confirm required local system tools are available.

   - `git`, `git-lfs`
   - Java runtime compatible with the bundled refactoring tools
   - [RefactoringMiner](https://github.com/tsantalis/RefactoringMiner) under
     `curation/tools/runtime/refactoringminer`
   - [RefDiff](https://github.com/aserg-ufmg/RefDiff) under
     `curation/tools/runtime/refdiff`
   - [RefactoringMiner++](https://github.com/benzoinoo/RefactoringMinerPP)
     under `curation/tools/runtime/refactoringminerpp`
   - [Multimetric](https://github.com/priv-kweihmann/multimetric) from
     `curation/requirements.txt`
   - [PMD](https://pmd.github.io/) for Java smell detection
   - [ESLint 8](https://eslint.org/) for JavaScript smell detection
   - [Cppcheck](https://cppcheck.sourceforge.io/) and
     [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) for C++ smell
     detection
   - [DesigniteJava](https://www.designite-tools.com/products-dj) and
     [DesignitePython/DPy](https://www.designite-tools.com/products-dpy) if
     those licensed/local smell findings are required

4. Set GitHub tokens.

   ```bash
   export GITHUB_TOKENS=ghp_...,...
   ```

   PowerShell:

   ```powershell
   $env:GITHUB_TOKENS = "ghp_...,..."
   ```

5. Check the local CLI.

   ```bash
   python -m curation.run --help
   ```

6. Run curation from the repository root.

   ```bash
   python -m curation.run \
     --cohort agentic \
     --input-dir /absolute/path/to/extraction-output \
     --input-format extractionpullrequests \
     --output-dir /absolute/path/to/curation-output \
     --target-prs 50000 \
     --longitudinal-prs 5000
   ```

7. Inspect, resume, or restart using the same output rules as the Docker run.

## CLI Reference

Check the Docker CLI first:

```bash
docker run --rm mosaic-curation:local --help
```

For local runs:

```bash
python -m curation.run --help
```

Common commands:

```bash
python -m curation.run --cohort agentic --input-dir outputs/extraction --input-format extractionpullrequests --output-dir outputs/curation --target-prs 50000 --longitudinal-prs 5000
python -m curation.run --cohort codex --input-dir outputs/extraction --input-format extractionpullrequests --output-dir outputs/curation --target-prs 1000 --longitudinal-prs 100
python -m curation.run --cohort human --input-dir outputs/extraction --input-format extractionpullrequests --output-dir outputs/curation --target-prs 50000 --longitudinal-prs 5000
```

Useful options:

- `--cohort`: `human`, `agentic`, or one configured agent.
- `--input-dir`: local parquet root; can be passed more than once.
- `--input-format`: `extractionpullrequests`, `fullpullrequests_sharded`, or
  `curation_processed`.
- `--output-dir`: local output root.
- `--target-prs`: main sample size.
- `--longitudinal-prs`: future-snapshot subset size.
- `--resume` / `--no-resume`: skip or process PRs already in progress records.
- `--sample-history-dir`: directory of previously sampled PR IDs/URLs to
  exclude.
- `--delete-snapshots-after-processing`: opt-in snapshot cleanup after each PR.

Run settings are assembled in `curation/config/settings.py`. Storage defaults
come from `curation/config/storage_config.py`, GitHub token loading comes from
`curation/config/tokens_config.py`, maintainability defaults come from
`curation/config/maintainability_config.py`, and refactoring defaults come from
`curation/config/refactoring_config.py`.

Supported environment variables:

- `CURATION_COHORT` or `COHORT`
- `CURATION_INPUT_DIRS`, `CURATION_INPUT_DIR`, or
  `CURATION_LOCAL_DIRECTORIES`
- `CURATION_INPUT_FORMAT` or `CURATION_LOCAL_DATA_FORMAT`
- `CURATION_OUTPUT_DIR` or `LOCAL_OUTPUT_DIR`
- `CURATION_TARGET_PRS` or `TARGET_NO_PRS`
- `CURATION_LONGITUDINAL_PRS` or `LONGITUDINAL_TARGET_NO_PRS`
- `CURATION_SAMPLE_HISTORY_DIR` or `SAMPLE_HISTORY_DIR`
- `CURATION_DELETE_SNAPSHOTS_AFTER_PROCESSING`
- `CURATION_CODE_SMELL_TOOL_TIMEOUT_SECONDS`
- `CURATION_DESIGNITE_JAVA_JAR`
- `CURATION_DESIGNITE_JAVA_COMMAND_TEMPLATE`
- `CURATION_DESIGNITE_PYTHON_COMMAND_TEMPLATE`
- `CURATION_PMD_COMMAND_TEMPLATE`
- `CURATION_ESLINT_COMMAND_TEMPLATE`
- `CURATION_CPPCHECK_COMMAND_TEMPLATE`
- `CURATION_CLANG_TIDY_COMMAND_TEMPLATE`
- `GITHUB_TOKENS`, then `GITHUB_TOKEN`

The default code-smell command templates are:

```text
DesigniteJava: java -jar <DesigniteJava.jar> -i {root} -o {out}
DPy:           dpy analyze -i {root} -o {out}
PMD:           pmd check -d {root} -R rulesets/java/quickstart.xml -f json
ESLint 8:      eslint -f json --no-eslintrc --no-error-on-unmatched-pattern --ext .js,.jsx,.mjs,.cjs --parser-options ecmaVersion:2022,sourceType:module --rule "complexity:[2,10]" --rule "max-depth:[2,4]" --rule "max-lines-per-function:[1,120]" --rule "max-params:[1,5]" --rule "no-nested-ternary:2" --rule "no-else-return:1" {root}
Cppcheck:      cppcheck --enable=warning,style,performance,portability --inline-suppr --template={file}:{line}:{id}:{severity}:{message} {root}
clang-tidy:    clang-tidy {file} -- -std=c++17
```

Use environment overrides when local binary paths, rulesets, or compile flags
differ from these defaults.

## Output Layout

By default Docker writes under `/data/output/output`. Local runs write under the
directory passed with `--output-dir`.

```text
OUTPUT_DIR/
  sampled_prs_<cohort>_store.jsonl
  longitudinal_prs_<cohort>_store.jsonl
  output/
    snapshots/
    processed-data/
    aggregates/
    checkpoints/
    clones/
    run_metadata_<cohort>_<timestamp>.json
    run_errors_<cohort>.json
```

Key output folders and files:

- `sampled_prs_<cohort>_store.jsonl`: sampled PR metadata used for processing.
- `longitudinal_prs_<cohort>_store.jsonl`: sampled PRs selected for future
  snapshots.
- `output/snapshots/`: hydrated before/after/future source trees.
- `output/processed-data/`: processed PR parquet batches.
- `output/aggregates/`: per-PR aggregate JSON files.
- `output/run_metadata_<cohort>_<timestamp>.json`: run configuration, counts,
  input format, target sizes, and metrics backend.
- `output/run_errors_<cohort>.json`: runtime errors and tool-gate failures.

Within each snapshot, code-smell artifacts are written under
`maintainability/code-smell-tool-output/`, `maintainability/tool-logs/`, and
`maintainability/code_smell_tool_results.json`.

## Resume

Re-running with the same cohort and output directory can resume from processing
progress records. Use `--resume` to skip PRs that are already recorded in
progress files. Use a fresh output directory for a fully independent run.

Checkpoint and progress records are local-only. They are written so a long
hydration or metrics run can continue after interruption without rerunning every
completed PR.

## Metrics

Refactoring metrics are computed from standardized refactoring operations on the
original PR transition. They include frequency, normalized density, operation
type distributions, Murphy-Hill abstraction levels, diversity, and line scope.

Maintainability metrics are computed for each available snapshot:

- `before`
- `after`
- future labels for longitudinal PRs, default `+3d`, `+7d`, `+31d`, `+61d`

[Multimetric](https://github.com/priv-kweihmann/multimetric) supplies the
maintainability backend for metrics such as maintainability index, cyclomatic
complexity, Halstead volume, fan out, and comment ratio. Duplicated-lines
density is computed by curation, not read from Multimetric. The custom
implementation scans supported source files, ignores comments and blank lines,
normalizes whitespace, and counts repeated normalized code lines:

```text
duplicated_lines_density = duplicated_lines / ncloc * 100
```

Code-smell findings are collected for each maintainability snapshot when the
language-specific tools are available:

- Java snapshots run
  [DesigniteJava](https://www.designite-tools.com/products-dj) and
  [PMD](https://pmd.github.io/).
- Python snapshots run
  [DesignitePython/DPy](https://www.designite-tools.com/products-dpy).
- C++ snapshots run [Cppcheck](https://cppcheck.sourceforge.io/) and
  [clang-tidy](https://clang.llvm.org/extra/clang-tidy/).
- JavaScript snapshots run [ESLint](https://eslint.org/).

Findings are normalized with
`curation/config/code_smell_standardization_config.py` and taxonomy-classified
with `curation/config/code_smell_taxonomy_config.py`. Tool-specific failures are
recorded in snapshot artifacts; they do not change the Multimetric numeric
metric computation. The parsed findings are summarized by snapshot, tool, rule,
severity, category, and taxonomy label.

The custom duplication value and code-smell summaries are merged into each
snapshot row and into pre/post and future delta summaries.

## Analysis Components

Curation writes the data consumed by the downstream analysis package under
`post-processing/analysis/`.

- `data_analysis_pipeline.py`: dataset size, cohort counts, and sample balance.
- `refactoring_analysis_pipeline.py`: merge-time refactoring metrics and
  refactoring plots.
- `maintainability_analysis_pipeline.py`: merge-time and longitudinal
  maintainability analysis; calls the Multimetric-specific analysis and the
  longitudinal maintainability sub-pipeline.
- `maintainability_multimetrics_pipeline.py`: Multimetric metric summaries and
  plots for maintainability index, cyclomatic complexity, Halstead volume,
  fan out, comment ratio, and duplicated-lines density.
- `characteristics_refactoring_pipeline.py`: refactoring results stratified by
  language, repository popularity, and topic/domain groups.
- `characteristics_maintainability_metrics_pipeline.py`: maintainability
  results stratified by language, repository popularity, and topic/domain
  groups.
- `longitudinal_refactoring_pipeline.py`: retention and later-touch analysis
  for original PR refactoring zones.
- `longitudinal_maintainability_metrics_pipeline.py`: maintainability evolution
  across future snapshots.

Domain/topic-based analysis requires topic enrichment output from the
post-processing topic-classification workflow. Curation itself preserves
repository metadata and identifiers, but topic/domain labels are attached
downstream.

## Troubleshooting

**No GitHub tokens configured**

Set `GITHUB_TOKENS` or `GITHUB_TOKEN`.

**Hydration fails for many PRs**

Check token access, repository visibility, network access to GitHub, and local
disk space. Use multiple tokens in `GITHUB_TOKENS` when possible.

**Refactoring tool missing locally**

Use Docker, or run the commands in `Reconstructing curation/tools`.
[RefactoringMiner](https://github.com/tsantalis/RefactoringMiner) and
[RefDiff](https://github.com/aserg-ufmg/RefDiff) must exist under
`curation/tools/runtime/`; local C++ runs also need
[RefactoringMiner++](https://github.com/benzoinoo/RefactoringMinerPP) or
`REFACTORING_MINER_PP_COMMAND`.

**Multimetric missing locally**

Install `curation/requirements.txt` and confirm
[`multimetric`](https://github.com/priv-kweihmann/multimetric) works with
`multimetric --help`.

**Code-smell tools missing or marked `tool_not_configured`**

Install [PMD](https://pmd.github.io/), [ESLint 8](https://eslint.org/),
[Cppcheck](https://cppcheck.sourceforge.io/), and
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) locally, or use Docker
for those tools. For [DesigniteJava](https://www.designite-tools.com/products-dj)
and [DesignitePython/DPy](https://www.designite-tools.com/products-dpy),
provide the licensed/local artifacts and set the Designite command-template
environment variables if the defaults do not match your installation.

## Files of Interest

- `curation/run.py`: CLI entry point.
- `curation/config/settings.py`: run-level settings resolution.
- `curation/config/storage_config.py`: local input/output defaults.
- `curation/config/tokens_config.py`: GitHub token environment loading.
- `curation/config/maintainability_config.py`: Multimetric, code-smell tool
  commands, and future snapshot labels.
- `curation/config/refactoring_config.py`: refactoring tool commands and
  timeouts.
- `curation/utility/loader.py`: local parquet discovery and format loading.
- `curation/pipeline/curation_pipeline.py`: preprocessing, sampling, and
  hydration orchestration.
- `curation/pipeline/hydration_pipeline.py`: per-PR hydration, metrics, and
  local persistence.
- `curation/sampler/sampler.py`: stratified sample construction.
- `curation/hydration/repository_hydrator.py`: repository clone/fetch and
  snapshot export.
- `curation/metrics/pr_metrics.py`: active per-PR metric stage orchestration.
- `curation/metrics/refactoring_metrics.py`: original PR refactoring mining and
  persistence tracking.
- `curation/metrics/maintainability_multimetric_metrics.py`: Multimetric,
  custom duplicated-lines-density, and code-smell payload assembly.
- `curation/metrics/code_smell_metrics.py`: DesigniteJava, DPy, PMD, ESLint,
  Cppcheck, and clang-tidy smell detection.
- `curation/metrics/duplication_metrics.py`: custom duplicated-lines-density
  implementation.
