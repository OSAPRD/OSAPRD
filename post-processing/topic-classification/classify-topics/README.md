# Classification Stage Files

This folder implements stage 4: classifying repositories from curation outputs
with a trained topic model.

## Source Files

- `topic_classification_pipeline.py`: streaming stage orchestrator for input
  discovery, README/wiki enrichment, feature construction, prediction, output
  writing, and manifest generation.
- `topic_loader.py`: repository-level index builder over parquet and legacy JSON
  curation outputs.
- `topic_features.py`: converts repository metadata, README/wiki text, and file
  paths into model inference text.
- `topic_classifier.py`: loads `topic_model_bundle.joblib` and applies top-k,
  score-threshold, exclusion, and topic-domain filtering.
- `topic_outputs.py`: writes repository predictions, repository-to-PR maps,
  filtered repository records, and the output manifest.
- `topic_preprocessing.py`: applies generated topic/text/file-name rules and
  runtime vocabulary filters from the model bundle.
- `identifier_splitting.py`: wraps Spiral/Ronin identifier splitting with a
  deterministic regex fallback.
- `readme_enrichment.py`: cache-first GitHub README fetcher.
- `wiki_enrichment.py`: cache-first optional GitHub wiki fetcher.
- `topic_domains.py`: validates the topic-to-domain mapping.

## Data Files

- `topic_domains.json`: canonical runtime mapping from retained topics to topic
  domains.
