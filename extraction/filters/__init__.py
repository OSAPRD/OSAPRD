"""Filter package for extraction-stage PR classification.

Filters translate configured GitHub search fragments into local checks used
after discovery/enrichment. They should stay deterministic and side-effect free:
no API calls, no storage writes, and no metric derivation.
"""
