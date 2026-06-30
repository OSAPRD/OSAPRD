"""GitHub API scrapers for extraction.

Scrapers own network-facing behavior: GitHub search, REST/GraphQL retries,
pagination, and payload normalization into DTOs. They should not decide output
locations or write parquet; managers and storage utilities own those concerns.
"""
