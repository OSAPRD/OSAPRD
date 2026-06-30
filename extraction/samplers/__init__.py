"""Sampling utilities for extraction discovery.

Samplers decide which discovered candidates should proceed to enrichment. They
should not call enrichment or storage directly; the manager owns those stages.
"""
