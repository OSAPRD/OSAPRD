"""Load explicitly configured sample-history identifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from curation.config.storage_config import SAMPLE_HISTORY_DIR


def sample_history_identifier_variants(value: Any) -> set[str]:
    """Return comparable identifier variants for one PR id/url token."""
    raw = str(value or "").strip()
    if not raw or raw.startswith("#"):
        return set()
    if raw.lower() in {"id", "url", "pr_id", "pr_url"}:
        return set()
    variants = {raw}
    if raw.startswith("http"):
        variants.add(raw.rstrip("/"))
    return {item for item in variants if item}


def sample_history_identifiers_from_line(line: str) -> set[str]:
    """Parse one tab-delimited history line into PR id/url identifiers."""
    line = str(line or "").strip()
    if not line or line.startswith("#"):
        return set()
    identifiers: set[str] = set()
    for token in line.split("\t"):
        token = token.strip()
        if token:
            identifiers.update(sample_history_identifier_variants(token))
    if "\t" not in line:
        identifiers.update(sample_history_identifier_variants(line))
    return identifiers


def load_sample_history_pr_identifiers(
    history_dir: Path | None = SAMPLE_HISTORY_DIR,
) -> set[str]:
    """
    Load PR ids/URLs from text files in the configured sample-history directory.

    The directory is intentionally a single explicit source of truth. No run
    output folders, input roots, run_errors files, or recursive history roots are
    scanned when this directory is unset.
    """
    if history_dir is None:
        print("[Curation] Sample history exclusion disabled: no directory configured.")
        return set()
    root = Path(history_dir)
    if not root.exists() or not root.is_dir():
        print(f"[Curation] Sample history exclusion disabled: directory not found: {root}")
        return set()

    identifiers: set[str] = set()
    files_scanned = 0
    for path in sorted(root.glob("*.txt")):
        if not path.is_file():
            continue
        files_scanned += 1
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    identifiers.update(sample_history_identifiers_from_line(line))
        except Exception as exc:
            print(f"[Curation] Warning: failed reading sample history file {path}: {exc}")
    print(
        "[Curation] Sample history exclusion loaded: "
        f"dir={root}, files={files_scanned}, identifiers={len(identifiers)}"
    )
    return identifiers
