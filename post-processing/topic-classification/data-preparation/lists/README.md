# Token Lists

These text files provide dictionary-style preprocessing inputs for repository
text and file names. They are loaded by `classify-topics/topic_preprocessing.py`
and copied into generated rule profiles when
`data-preparation/generate_github_topic_rules.py` runs.

Examples:

- `Contarctions.txt` expands contractions such as `ain't`.
- `SE_abbr.txt` expands software-engineering abbreviations such as `async` and
  `bg`.
- `SE_topics.txt` protects terms such as `c++` and `c#` before punctuation is
  removed.
- `File names_confusing_tokens.txt` removes generic file-name tokens that do not
  help topic classification.
