# Topic Rules

This directory contains the base CSV rule files used to normalize repository
topics. The runtime preprocessor loads these files in a fixed order, lowercases
tokens, applies alias and aggregate mappings, removes stopword-like topics, and
keeps labels aligned with the model vocabulary.

The active pipeline normally uses the generated profile under
`../generated/github-topics/rules`, which preserves this file layout while
deriving aliases and canonical topics from the local GitHub topics catalog.
These base files remain useful as generic normalization rules and as input to
the generator.
