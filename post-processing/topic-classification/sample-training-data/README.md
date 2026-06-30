# Sampling Stage Files

This folder implements the second half of stage 2: deterministic sampling and
train/test splitting for topic-model training.

## Source Files

- `pipeline.py`: joins encoded repository records with extraction metadata,
  filters the retained topic universe, samples by popularity and creation-time
  strata, and writes train/test CSVs.

## Data Files

- `filtered_topics.json`: topics excluded from model training and runtime
  classification, including language/generic labels that should not be model
  targets.

## Outputs

The stage writes a named sample folder containing:

- `topics_train.csv`
- `topics_test.csv`
- `sampled_repository_metadata.csv`
- `sampled_encoded_repository_records.jsonl`
- `stratum_counts.csv`
- `sample_train_test_manifest.json`
