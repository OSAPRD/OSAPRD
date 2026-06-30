# Extraction Pipeline

This package extracts GitHub pull-request data from live GitHub scraping. It discovers pull requests, enriches them with GitHub metadata, and writes local parquet outputs plus local manifests.

Publishing or uploading extraction outputs is intentionally not part of this stage. Treat upload to Hugging Face or another archive as post-processing.

## Pipeline Flow

Extraction has two data-construction phases, followed by local persistence:

```text
CLI/env settings
  -> discovery: find lightweight PR candidates
  -> enrichment: hydrate each candidate into the full PR row schema
  -> storage: write grouped local parquet batches and manifests
```

Discovery intentionally fetches only minimal PR fields: URL, title, body text, author peek, creation/merge/close timestamps, draft flag, head branch, GitHub node id, and database id. This keeps the broad search phase cheaper and lets the scraper checkpoint progress frequently.

Enrichment takes each discovered candidate and fetches the heavier fields used by downstream curation: PR core fields, changed files, file language labels, repository metadata, topics, merge/head/base metadata, and first-commit author signals when needed. The enriched `PullRequest` DTO preserves the existing parquet row schema.

Storage buffers enriched PRs by output group and writes numbered parquet batches. `manifest.json` stores stable PR identifiers for duplicate detection. `extraction_run_manifest.json` records run metadata such as target, resolved agent list, date window, page cap, batch size, GraphQL flag, output groups, counts, schema version, run label, and redacted token count.

## GitHub APIs Used

This stage uses both official GitHub APIs:

- [GitHub GraphQL API](https://docs.github.com/en/graphql): used for broad PR search and default PR enrichment.
- [GraphQL `search` query](https://docs.github.com/en/graphql/reference/queries#search): used during discovery with issue search strings such as `is:pr head:codex/ created:... sort:created-asc`.
- [GraphQL rate and query limits](https://docs.github.com/en/graphql/overview/rate-limits-and-query-limits-for-the-graphql-api): used to guide token rotation, backoff, and resource-limit window slicing.
- [GitHub REST API](https://docs.github.com/en/rest): used for commit-search backfill, repository metadata, first-commit lookup, and REST fallback enrichment.
- [REST Search API](https://docs.github.com/en/rest/search/search): used for commit search backfill for agents that are better identified by first-commit author.
- [REST Pull Requests API](https://docs.github.com/en/rest/pulls/pulls): used for PR files, PR commit lists, and REST fallback PR details.
- [REST rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api): used through `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers to update token state.

GraphQL is the default for discovery because it can search PRs and return structured PR nodes with cursor pagination. GraphQL is also the default enrichment path because it fetches core PR fields and paginated changed-file nodes in fewer requests than REST.

REST is still required for three things:

- Commit-search backfill: `/search/commits` finds commits by known agent accounts, then `/repos/{owner}/{repo}/commits/{sha}/pulls` maps commits back to PRs.
- First-commit verification: `/repos/{owner}/{repo}/pulls/{number}/commits?per_page=1&page=1` checks whether the first commit author identifies an agent.
- Repository and fallback metadata: `/repos/{owner}/{repo}`, REST PR details, and REST PR files provide data not present in the GraphQL query or provide a deterministic fallback when GraphQL enrichment is disabled.

## Rate Limits, Token Rotation, and Window Slicing

GitHub imposes rate limits and query-cost/resource limits. Extraction handles those constraints explicitly because large discovery runs can span many search clauses, many date windows, and many enrichment calls.

Token rotation:

- Configure multiple tokens with `GITHUB_TOKENS=token1,token2,...`.
- `GITHUB_TOKENS` takes precedence over the single-token fallback `GITHUB_TOKEN`.
- Each scraper uses `TokenManager` to choose the current token.
- REST responses update token state from `X-RateLimit-Remaining` and `X-RateLimit-Reset`.
- GraphQL responses query `rateLimit { limit cost remaining resetAt }` and update the same token state.
- If a token is invalid, it is marked invalid and skipped.
- If a token is exhausted, the scraper rotates to another usable token.
- If all tokens are exhausted, discovery waits until the earliest reset instead of hammering the API.

Window slicing:

- GitHub search is not a full database export interface. Large search queries can hit result caps or GraphQL resource limits.
- Discovery searches in ascending creation-time order with `created:<start>..<end> sort:created-asc`.
- Each GraphQL page requests up to 100 PRs.
- When a search window approaches GitHub's search cap, the scraper advances the lower bound to the last observed `createdAt` and restarts pagination.
- When GitHub returns GraphQL resource-limit errors, the scraper halves the active time window and retries.
- The minimum resource-limit slice is one hour. If the same one-hour slice keeps failing after retries, discovery advances by one second to keep the run moving.
- Checkpoints record target, date window, clause index, cursor, active start/end, count in window, and commit backfill page so interrupted runs can resume.

This is why the code has both token rotation and window slicing: token rotation handles quota exhaustion, while window slicing handles expensive or overly broad search queries.

## Targets

Use one explicit scrape target per run:

- `human`: time-stratified human PR sampling with agent and bot exclusions.
- `agentic`: all configured agents in `extraction/config/agent_config.py`.
- `<agent>`: one configured agent key from `AGENT_RULES`.

Single-agent targets use only that agent's GraphQL discovery clauses. Commit-search backfill also runs only for that agent when it has entries in `FIRST_COMMIT_AUTHORS`.

`agentic` runs use every configured agent rule and write discovered PRs under each resolved agent group. Single-agent runs write only that agent's PRs under that agent group. Human runs write under `humans`.

## Requirements

Docker is the recommended way to run extraction because it isolates the Python
parquet stack from the host environment. The image installs the same
`extraction/requirements.txt` used by local runs.

For Docker runs, you need:

- Docker.
- Network access to GitHub.
- One or more GitHub tokens in `GITHUB_TOKENS` or `GITHUB_TOKEN`.
- A local output directory mounted to `/data` in the container.

For local runs without Docker, use Python 3.12. The dependency file is
intentionally small and extraction-scoped:

- Runtime: `requests` for GitHub API calls, `pandas` + `pyarrow` for parquet
  output, and their pinned transitive dependencies.
- Not included: Hugging Face, async HTTP clients, dataset publishing tools, test
  frameworks, or post-processing dependencies.

The parquet stack is pinned to `numpy==2.3.5`, `pandas==2.3.3`, and
`pyarrow==22.0.0`. `numpy==2.4.0` is avoided because pip reports that release
as yanked.

Set one or more GitHub tokens:

```bash
export GITHUB_TOKENS=ghp_...,...   # preferred for rotation
export GITHUB_TOKEN=ghp_...        # fallback
```

PowerShell:

```powershell
$env:GITHUB_TOKENS = "ghp_...,..."
```

## Step-by-Step Run: Docker

Use this path first unless you are actively developing extraction code on the
host.

1. Choose the target.

   Use `agentic` for all configured agents, `human` for human sampling, or one
   configured agent such as `codex`, `junie`, `devin`, `claude`, `jules`, or
   `openhands`.

2. Choose the date window.

   Dates are inclusive UTC dates in `YYYY-MM-DD` format. Start with a small
   range and a low `--max-pages` value for a smoke run.

3. Set GitHub tokens.

   ```bash
   export GITHUB_TOKENS=ghp_...,...
   ```

   PowerShell:

   ```powershell
   $env:GITHUB_TOKENS = "ghp_...,..."
   ```

4. Build the extraction image from the repository root.

   ```bash
   docker build -f extraction/Dockerfile -t mosaic-extraction:local extraction
   ```

5. Check the container CLI.

   ```bash
   docker run --rm mosaic-extraction:local python -m extraction.run --help
   ```

6. Run a small extraction smoke run.

   ```bash
   mkdir -p outputs/extraction

   docker run --rm \
     -e GITHUB_TOKENS="$GITHUB_TOKENS" \
     -v "$(pwd)/outputs/extraction:/data" \
     mosaic-extraction:local \
     python -m extraction.run \
       --target codex \
       --start 2025-12-01 \
       --end 2025-12-03 \
       --max-pages 50 \
       --output-dir /data
   ```

7. Run another target by changing `--target`.

   ```bash
   docker run --rm \
     -e GITHUB_TOKENS="$GITHUB_TOKENS" \
     -v "$(pwd)/outputs/extraction:/data" \
     mosaic-extraction:local \
     python -m extraction.run --target agentic --start 2025-12-01 --end 2025-12-03 --max-pages 50 --output-dir /data

   docker run --rm \
     -e GITHUB_TOKENS="$GITHUB_TOKENS" \
     -v "$(pwd)/outputs/extraction:/data" \
     mosaic-extraction:local \
     python -m extraction.run --target human --start 2025-12-01 --end 2025-12-03 --max-pages 50 --output-dir /data
   ```

8. Inspect outputs.

   Check `outputs/extraction/data/<group>/` for parquet batches,
   `outputs/extraction/manifest.json` for duplicate-detection IDs, and
   `outputs/extraction/extraction_run_manifest.json` for run metadata.

9. Resume or restart.

   Re-running the same target, date window, and output directory resumes from
   checkpoints and skips PRs already listed in `manifest.json`. Use a new output
   directory for a completely independent run.

## Step-by-Step Run: Local

Use the local path when Docker is unavailable or when you need to debug the
Python process directly on the host.

1. Install dependencies from the repository root.

   ```bash
   python -m pip install -r extraction/requirements.txt
   ```

2. Set GitHub tokens.

   ```bash
   export GITHUB_TOKENS=ghp_...,...
   ```

   PowerShell:

   ```powershell
   $env:GITHUB_TOKENS = "ghp_...,..."
   ```

3. Run extraction from the repository root.

   ```bash
   python -m extraction.run \
     --target codex \
     --start 2025-12-01 \
     --end 2025-12-03 \
     --max-pages 50 \
     --output-dir outputs/extraction
   ```

4. Inspect, resume, or restart using the same output rules as the Docker run.

## CLI Reference

Check the Docker CLI first:

```bash
docker run --rm mosaic-extraction:local python -m extraction.run --help
```

For local runs:

```bash
python -m extraction.run --help
```

Common commands:

```bash
python -m extraction.run --target agentic --start 2025-12-01 --end 2025-12-03 --max-pages 50
python -m extraction.run --target codex --start 2025-12-01 --end 2025-12-03 --max-pages 50
python -m extraction.run --target human --start 2025-12-01 --end 2025-12-03 --max-pages 50
```

Useful options:

- `--target`: `human`, `agentic`, or one key from `AGENT_RULES`.
- `--start`, `--end`: inclusive UTC dates in `YYYY-MM-DD`.
- `--max-pages`: discovery pagination cap per query/window.
- `--output-dir`: local output root.
- `--batch-size`: enriched PRs per parquet batch.
- `--use-graphql-enrichment` / `--no-use-graphql-enrichment`: GraphQL enrichment toggle.

Run settings are assembled in `extraction/config/settings.py`. Storage defaults come from `extraction/config/storage_config.py`, GitHub token loading comes from `extraction/config/tokens_config.py`, and CLI arguments take precedence over environment variables and config defaults.

Supported environment variables:

- `EXTRACTION_TARGET` or `TARGET`
- `EXTRACTION_START_DATE` or `START_DATE`
- `EXTRACTION_END_DATE` or `END_DATE`
- `EXTRACTION_MAX_PAGES` or `MAX_PAGES`
- `EXTRACTION_LOCAL_OUTPUT_DIR` or `LOCAL_OUTPUT_DIR`
- `EXTRACTION_BATCH_SIZE` or `BATCH_SIZE`
- `EXTRACTION_ENRICHMENT_USE_GRAPHQL` or `ENRICHMENT_USE_GRAPHQL`
- `GITHUB_TOKENS`, then `GITHUB_TOKEN`

## Output Layout

Local files are grouped by agent or `humans`:

```text
LOCAL_OUTPUT_DIR/
  data/
    codex/
      codex_pr_<run_label>_batch-0001.parquet
    devin/
      agentic_pr_<run_label>_batch-0001.parquet
    humans/
      human_pr_<run_label>_batch-0001.parquet
  manifest.json
  extraction_run_manifest.json
  checkpoints/
    discovery_checkpoint.json
    enrichment_checkpoint.json
```

## Resume

Discovery checkpoints are scoped to target and date window. Enrichment checkpoints are also scoped to target and date window. Re-running the same target/window resumes from local checkpoints and skips PRs already present in `manifest.json`.

Use a fresh output directory for a fully independent run.

## Troubleshooting

**No GitHub tokens configured**

Set `GITHUB_TOKENS` or `GITHUB_TOKEN`.

**Discovery takes long**

Reduce the date range, lower `--max-pages`, provide more GitHub tokens, or run one agent target at a time.

**Duplicate PRs skipped**

This is expected when `manifest.json` already contains the PR URL or database ID. Use a fresh output directory for a fully independent run.

**GraphQL resource limits**

This is expected for broad or expensive windows. The scraper will slice the active time window, retry, checkpoint, and continue.

## Files of Interest

- `extraction/run.py`: CLI entry point.
- `extraction/config/settings.py`: run-level settings resolution.
- `extraction/config/agent_config.py`: agent discovery clauses and first-commit author signatures.
- `extraction/config/human_config.py`: human sampling language and exclusion policy.
- `extraction/config/storage_config.py`: local output directory and parquet batch-size settings.
- `extraction/config/tokens_config.py`: GitHub token environment loading.
- `extraction/managers/scraper_manager.py`: discovery -> enrichment -> storage orchestration and run manifest.
- `extraction/samplers/human_sampler.py`: time-stratified human candidate sampling.
- `extraction/scrapers/discovery_scraper.py`: GraphQL/REST discovery, token rotation, window slicing, and checkpointing.
- `extraction/scrapers/enrichment_scraper.py`: GraphQL/REST enrichment and DTO normalization.
- `extraction/utility/storage_handler.py`: local parquet batching, duplicate manifest, and durable journal support.
- `extraction/utility/token_manager.py`: token rotation and rate-limit state tracking.
