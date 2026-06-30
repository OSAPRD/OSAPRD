# Create-Training-Data Stage Files

This folder implements the first half of stage 2: filtering and preprocessing
extracted repository records into encoded, model-ready repository records.

## Source Files

- `pipeline.py`: reads one extraction run, filters repositories with insufficient
  source text, applies topic/text/file-name preprocessing, builds corpus-level
  frequency counts, writes label vectors, and emits a manifest.

## Outputs

The stage writes under `<extract-run>/training-data/create-training-data`:

- `filtered_repository_records.jsonl`
- `excluded_repository_records.jsonl`
- `preprocessed_repository_records.jsonl`
- `encoded_repository_records.jsonl`
- `topic_label_universe.json`
- corpus frequency artifacts
- `create_training_data_manifest.json`

The sampling stage reads these outputs together with extraction metadata.
