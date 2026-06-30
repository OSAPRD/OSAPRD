# Training Stage Files

This folder implements stage 3: training and exporting the topic model bundle.

## Source Files

- `train_model.py`: validates train/test CSV labels, loads training
  dependencies, trains the TF-IDF plus one-vs-rest logistic-regression model,
  evaluates top-k metrics, exports runtime preprocessing artifacts, and writes
  the model bundle plus manifest.

## Outputs

Training writes artifacts such as:

- `topic_model_bundle.joblib`
- ordered label files
- preprocessing artifact JSON
- metrics JSON
- training manifest JSON

The classification stage consumes `topic_model_bundle.joblib`.
