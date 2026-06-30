"""Single-pass PR curation package.

The package consumes local extraction parquet, samples and hydrates PRs,
computes refactoring and maintainability metrics, and writes local curated
artifacts. Runtime behavior is configured through :mod:`curation.config`.
"""
