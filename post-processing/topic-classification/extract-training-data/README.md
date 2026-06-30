# Extraction Stage Files

This folder implements stage 1 of topic classification: collecting
topic-labeled repositories from GitHub and writing a restartable local run
directory.

## Source Files

- `common.py`: shared path constants, JSON/JSONL helpers, repository key helpers,
  environment parsing, and GitHub token resolution for extraction.
- `github_training_client.py`: small GitHub REST client for repository search,
  repository metadata, README text, file lists, and wiki-related text fetches.
- `pipeline.py`: orchestration for date-window search, capped-window splitting,
  enrichment, checkpointing, artifact writing, and manifest generation.

## Outputs

The stage writes one timestamped output folder containing:

- `raw/repository_search_candidates.jsonl`
- `checkpoints/repository_search_pages.jsonl`
- `repositories/repository_training_records.jsonl`
- README/wiki cache files
- repository file-list snapshots
- `topic_training_extraction_manifest.json`

These artifacts are local inputs to the preparation stage.
