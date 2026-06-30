"""Managers that coordinate extraction components.

Managers own orchestration only: they connect settings, filters, scrapers,
checkpoints, and storage. They should not contain GitHub request code or parquet
serialization internals; those belong to scraper and utility modules.
"""
