# Raw Frequency Artifacts

This optional directory is for source-specific token-count artifacts used for
corpus-frequency filtering during runtime repository classification.

Expected files:

- `text_token_counts.csv`: token counts from raw repository text sources such as description, README, and wiki.
- `file_name_token_counts.csv`: token counts from raw repository file-name sources.

Each file may be CSV, TSV, JSON, or repeated one-token-per-line text where repeated lines encode counts. CSV/TSV files should preferably use `token,count` headers.

When both files are present, `python post-processing/topic-classification/run.py train`
applies the source-specific thresholds separately:

- text tokens: keep tokens with count >= 50
- file-name tokens: keep tokens with count >= 20

When these files are absent, default training falls back to the prepared
train/test CSV text vocabulary and records that the resulting runtime filter is
threshold-aligned but not raw-corpus equivalent. Use
`--require-raw-frequency-artifacts` to fail unless the source-specific artifacts
are available.
