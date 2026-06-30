"""
Checkpoint persistence for resumable extraction runs.

The extraction stage can run for a long time because GitHub search windows may
need to be sliced and enrichment performs one request per pull request. This
module stores small JSON state files so discovery and enrichment can continue
from the last completed unit of work instead of starting over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class CheckpointHandler:
    """
    Small JSON-backed checkpoint helper.
    """

    def __init__(self, path: Path) -> None:
        """Create a checkpoint handler bound to one JSON file path."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """
        Load checkpoint state from disk if present.
        """
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    print(f"[checkpoint] Loaded checkpoint from {self.path}")
                    return data
            except Exception:
                # Checkpoints are a resume optimization, not the source data.
                # Corrupt state should not prevent a fresh extraction run.
                pass
        return {}

    def save(self, state: Dict[str, Any]) -> None:
        """Persist checkpoint state to disk as readable JSON."""
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[checkpoint] Failed to save checkpoint: {e}")

    def clear(self) -> None:
        """Delete the checkpoint file after the owning stage completes."""
        try:
            if self.path.exists():
                self.path.unlink()
                print(f"[checkpoint] Cleared checkpoint at {self.path}")
        except Exception as e:
            print(f"[checkpoint] Failed to clear checkpoint: {e}")
