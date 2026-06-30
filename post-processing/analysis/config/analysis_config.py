"""Runtime behavior constants for analysis pipeline modules."""

from __future__ import annotations

from settings import (
    MANTYLA_COUNT_SOURCE_STORED,
    MANTYLA_COUNT_SOURCE_TAXONOMY,
    MANTYLA_COUNT_SOURCES,
    MURPHY_HILL_COUNT_SOURCE_STORED,
    MURPHY_HILL_COUNT_SOURCE_TAXONOMY,
    MURPHY_HILL_COUNT_SOURCES,
    MULTIMETRIC_SOURCE_AUTO,
    MULTIMETRIC_SOURCE_EXTERNAL,
    MULTIMETRIC_SOURCE_INPUT,
    MULTIMETRIC_SOURCE_OFF,
    MULTIMETRIC_SOURCES,
    AnalysisSettings,
)


_SETTINGS = AnalysisSettings.from_env()

EXCLUDED_AGENTS = _SETTINGS.excluded_agents
MURPHY_HILL_COUNT_SOURCE = _SETTINGS.murphy_hill_count_source
MANTYLA_COUNT_SOURCE = _SETTINGS.mantyla_count_source
MAINTAINABILITY_REQUIRE_REFOPS = _SETTINGS.maintainability_require_refops
PLOT_MODE = _SETTINGS.plot_mode
MULTIMETRIC_SOURCE = _SETTINGS.multimetric_source
