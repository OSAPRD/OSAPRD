"""
Shared utility helpers for the extraction stage.

This package keeps cross-cutting concerns out of the scrapers and managers:
checkpoint files make long GitHub runs resumable, token rotation handles rate
limits consistently, language labels are inferred from changed file paths, and
storage writes local reproducibility artifacts.
"""
