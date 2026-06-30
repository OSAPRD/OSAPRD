"""Configuration package for the live GitHub extraction stage.

`settings.py` resolves one run-level `ExtractionSettings` object from CLI,
environment, and config defaults. The neighboring modules own domain-specific
configuration: `agent_config.py` for agent signals, `human_config.py` for human
sampling policy, `storage_config.py` for local output settings, and
`tokens_config.py` for GitHub token loading.

The extraction stage is intentionally local-only. Publishing locations such as
Hugging Face belong to post-processing, not this package.
"""
