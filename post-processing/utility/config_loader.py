"""Shared storage configuration loading helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


DEFAULT_STORAGE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "storage_config.py"
)
DEFAULT_TOPIC_CLASSIFICATION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "topic_classification_config.py"
)


def _load_config_module(
    *,
    module_name: str,
    config_path: Path,
    description: str,
) -> ModuleType:
    resolved_config_path = Path(config_path)
    spec = importlib.util.spec_from_file_location(module_name, resolved_config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {description} config from {resolved_config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_storage_config(
    *,
    module_name: str = "post_processing_storage_config",
    config_path: Path | None = None,
) -> ModuleType:
    """Load the shared post-processing storage config module from disk."""
    return _load_config_module(
        module_name=module_name,
        config_path=Path(config_path or DEFAULT_STORAGE_CONFIG_PATH),
        description="storage",
    )


def load_topic_classification_config(
    *,
    module_name: str = "post_processing_topic_classification_config",
    config_path: Path | None = None,
) -> ModuleType:
    """Load the topic-classification post-processing config module from disk."""
    return _load_config_module(
        module_name=module_name,
        config_path=Path(config_path or DEFAULT_TOPIC_CLASSIFICATION_CONFIG_PATH),
        description="topic-classification",
    )
