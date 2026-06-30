# Data Preparation

This directory contains the rule and list files used by the topic
preprocessing stage. The active pipeline reads these files through
`classify-topics/topic_preprocessing.py` when it prepares repository names,
descriptions, README text, wiki text, file names, and topic labels.

The base rules come from
[MalihehIzadi/SoftwareTagRecommender](https://github.com/MalihehIzadi/SoftwareTagRecommender).
Only the data-preparation subset is kept here; model training and inference are
implemented in this package.

## Contents

- `rules/`: CSV transformation rules for normalizing and mapping GitHub topics.
- `lists/`: token dictionaries for contractions, abbreviations, slang, dates,
  confusing file-name tokens, and software-engineering terms.
- `generated/github-topics/`: generated rule profile built from the local
  `github-topics` catalog plus the vendored base rules.
- `raw-frequency/`: optional token-count artifacts used by model training to
  export source-specific runtime vocabulary filters.
- `topic_preprocess.ipynb`: upstream exploratory notebook retained as reference
  material, not as the active execution path.

## Active Use

The default training-data preparation stage points at
`generated/github-topics`. That generated profile keeps the same file layout as
the vendored rule set while deriving topic aliases and mappings from the local
GitHub topic catalog.

Regenerate the profile with:

```bash
python post-processing/topic-classification/data-preparation/generate_github_topic_rules.py
```

The command writes a manifest, generated rule files, generated list files, a
topic catalog JSON file, an alias map, and a merge audit.
