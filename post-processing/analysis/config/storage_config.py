"""Local input/output roots for post-processing analysis."""

from __future__ import annotations

from settings import AnalysisSettings


_SETTINGS = AnalysisSettings.from_env()

CURATION_DATA_DIR = _SETTINGS.curation_data_dir
TOPIC_CLASSIFICATION_OUTPUT_DIR = _SETTINGS.topic_classification_output_dir
ANALYSIS_OUTPUT_DIR = _SETTINGS.analysis_output_dir
MULTIMETRIC_OUTPUT_DIR = _SETTINGS.multimetric_output_dir
