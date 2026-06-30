"""
Local persistence for enriched extraction records.

Extraction is local-only: this module writes grouped Parquet
batches plus duplicate-detection manifests under the configured output root.
Publishing to Hugging Face or another repository is handled by later
post-processing, not by this storage layer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from extraction.config.settings import ExtractionSettings
from extraction.config.storage_config import BATCH_SIZE, LOCAL_OUTPUT_DIR


class StorageHandler:
    """
    Persist scraper outputs as grouped local Parquet batches.

    The handler owns three pieces of state:
    - in-memory buffers keyed by output group and base file name;
    - ``manifest.json`` / recorded-id journal entries for duplicate detection;
    - optional durable JSONL journals for crash recovery before Parquet flushes.
    """

    def __init__(
        self,
        local_dir: Optional[Path] = None,
        run_label: Optional[str] = None,
        batch_size: Optional[int] = None,
        data_subdir: str = "data",
        durable_journal: bool = False,
        durable_journal_fsync_interval: Optional[int] = None,
        settings: Optional[ExtractionSettings] = None,
    ) -> None:
        """Initialize storage paths, manifests, and optional durable journaling."""
        if settings is not None:
            # Explicit constructor arguments win over settings so small callers
            # can override one storage concern without rebuilding settings.
            if local_dir is None:
                local_dir = settings.local_output_dir
            if batch_size is None:
                batch_size = settings.batch_size
        self.local_dir = Path(local_dir or LOCAL_OUTPUT_DIR)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.data_subdir = self._sanitize_label(data_subdir) or "data"
        self.data_root = self.local_dir / self.data_subdir
        self.data_root.mkdir(parents=True, exist_ok=True)
        # The manifest is intentionally simple: it records every stable PR
        # identifier that has reached durable local output.
        self.manifest_path = self.local_dir / "manifest.json"
        self.durable_journal = durable_journal
        self.journal_root = self.local_dir / "journal"
        # The recorded-id journal lets durable mode reject duplicates even when
        # a record has not yet been compacted into Parquet.
        self.recorded_ids_path = self.journal_root / "recorded_ids.jsonl"
        self.compaction_state_path = self.journal_root / "compaction_state.json"
        self.run_label = self._sanitize_label(run_label) if run_label else None
        resolved_batch = BATCH_SIZE if batch_size is None else batch_size
        self.batch_size = max(1, int(resolved_batch))
        # Buffer values include sequence numbers in durable mode so replayed
        # records can be compacted exactly once after an interrupted run.
        self._buffers: dict[tuple[str, str], list[tuple[int, set[str], dict]]] = {}
        self._batch_counts: dict[tuple[str, str], int] = {}
        self._pending_ids: set[str] = set()
        self._seen_ids: set[str] = set()
        self._compacted_seq_by_key: dict[tuple[str, str], int] = {}
        self._pending_journal_sync_paths: set[Path] = set()
        self._journal_appends_since_sync = 0
        self._next_seq = 1
        if durable_journal_fsync_interval is None:
            self.durable_journal_fsync_interval = 1
        else:
            self.durable_journal_fsync_interval = max(0, int(durable_journal_fsync_interval))
        if self.durable_journal:
            self.journal_root.mkdir(parents=True, exist_ok=True)
            self._load_compaction_state()
            self._seen_ids = self._load_recorded_ids()
            self._load_journal_backlog()
        else:
            self._seen_ids = self._load_manifest()

    def _to_dicts(self, records: Iterable) -> list[dict]:
        """Convert DTO/dataclass-like records into serializable dictionaries."""
        data = []
        for rec in records:
            if hasattr(rec, "to_dict"):
                data.append(rec.to_dict())
            elif hasattr(rec, "__dict__"):
                data.append(asdict(rec))
            else:
                data.append(rec)
        return data

    def _record_ids(self, rec: Any, rec_dict: Optional[dict] = None) -> set[str]:
        """Extract stable identifiers used to deduplicate pull requests."""
        ids: set[str] = set()
        url = getattr(rec, "url", None) or (rec_dict.get("url") if rec_dict else None)
        if url:
            ids.add(str(url))
        database_id = getattr(rec, "database_id", None) or (
            rec_dict.get("database_id") if rec_dict else None
        )
        if database_id:
            ids.add(str(database_id))
        raw_id = getattr(rec, "id", None) or (rec_dict.get("id") if rec_dict else None)
        if raw_id is not None:
            raw_id_str = str(raw_id)
            if raw_id_str.isdigit():
                ids.add(raw_id_str)
        return ids

    def _sanitize_label(self, label: str) -> str:
        """Sanitize group, run, and file labels for portable filesystem use."""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", label)

    def _ensure_group_dirs(self, group: str) -> Path:
        """Create (if needed) and return the group data directory."""
        safe_group = self._sanitize_label(group)
        data_dir = self.data_root / safe_group
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def _name_with_run_label(self, base_name: str) -> str:
        """Append the run label to a base filename when configured."""
        if self.run_label:
            return f"{base_name}_{self.run_label}"
        return base_name

    def _load_manifest(self) -> set[str]:
        """Load IDs for records already written to Parquet."""
        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return set(str(x) for x in data)
            except Exception:
                # The output Parquet files remain the authoritative data. A
                # broken manifest only disables duplicate skipping for this run.
                pass
        return set()

    def _load_recorded_ids(self) -> set[str]:
        """Load duplicate-detection IDs from manifest and durable journal."""
        ids = self._load_manifest()
        if not self.recorded_ids_path.exists():
            return ids
        try:
            with self.recorded_ids_path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    value = payload.get("id")
                    if value:
                        ids.add(str(value))
        except Exception:
            pass
        return ids

    def _save_manifest(self) -> None:
        """Persist duplicate-detection IDs for non-durable storage mode."""
        try:
            with self.manifest_path.open("w", encoding="utf-8") as f:
                json.dump(sorted(self._seen_ids), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[storage] Failed to save manifest: {e}")

    def _journal_file(self, group: str, base_name: str) -> Path:
        """Return the append-only journal path for a group/base_name pair."""
        safe_group = self._sanitize_label(group)
        journal_dir = self.journal_root / safe_group
        journal_dir.mkdir(parents=True, exist_ok=True)
        return journal_dir / f"{base_name}.jsonl"

    def _state_key(self, group: str, base_name: str) -> str:
        """Return a stable serialized key for compaction state."""
        return f"{self._sanitize_label(group)}::{base_name}"

    def _load_compaction_state(self) -> None:
        """Load persisted batch counters and journal compaction cursors."""
        if not self.compaction_state_path.exists():
            return
        try:
            payload = json.loads(self.compaction_state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        compacted = payload.get("last_compacted_seq_by_key") or {}
        for raw_key, seq in compacted.items():
            if not isinstance(raw_key, str):
                continue
            group, _, base_name = raw_key.partition("::")
            if not group or not base_name:
                continue
            try:
                self._compacted_seq_by_key[(group, base_name)] = max(0, int(seq))
            except (TypeError, ValueError):
                continue
        batches = payload.get("batch_counts") or {}
        for raw_key, count in batches.items():
            if not isinstance(raw_key, str):
                continue
            group, _, base_name = raw_key.partition("::")
            if not group or not base_name:
                continue
            try:
                self._batch_counts[(group, base_name)] = max(0, int(count))
            except (TypeError, ValueError):
                continue

    def _save_compaction_state(self) -> None:
        """Persist compaction cursors and batch counters via temp-file replace."""
        if not self.durable_journal:
            return
        payload = {
            "last_compacted_seq_by_key": {
                self._state_key(group, base_name): seq
                for (group, base_name), seq in self._compacted_seq_by_key.items()
            },
            "batch_counts": {
                self._state_key(group, base_name): count
                for (group, base_name), count in self._batch_counts.items()
            },
        }
        temp_path = self.compaction_state_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.compaction_state_path)

    def _sync_journal_paths(self, paths: Optional[Iterable[Path]] = None) -> None:
        """fsync pending journal files, normally before a Parquet batch is written."""
        target_paths = set(paths or self._pending_journal_sync_paths)
        if not target_paths:
            return
        for path in sorted(target_paths, key=str):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.flush()
                    os.fsync(f.fileno())
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"[storage] Failed to sync journal file {path}: {exc}")
        self._pending_journal_sync_paths.difference_update(target_paths)
        if not self._pending_journal_sync_paths:
            self._journal_appends_since_sync = 0

    def _append_jsonl_many(
        self,
        path: Path,
        payloads: Iterable[dict],
        *,
        sync: Optional[bool] = None,
    ) -> None:
        """Append JSON objects to a JSONL file with configurable fsync cadence."""
        items = list(payloads)
        if not items:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for payload in items:
                f.write(json.dumps(payload, ensure_ascii=False, default=str))
                f.write("\n")
            f.flush()
            self._pending_journal_sync_paths.add(path)
            self._journal_appends_since_sync += len(items)
            should_sync = (
                bool(sync)
                if sync is not None
                else (
                    self.durable_journal_fsync_interval > 0
                    and self._journal_appends_since_sync >= self.durable_journal_fsync_interval
                )
            )
            if should_sync:
                os.fsync(f.fileno())
                self._pending_journal_sync_paths.discard(path)
                if not self._pending_journal_sync_paths:
                    self._journal_appends_since_sync = 0

    def _append_jsonl(self, path: Path, payload: dict, *, sync: Optional[bool] = None) -> None:
        """Append one JSON object to a JSONL file."""
        self._append_jsonl_many(path, [payload], sync=sync)

    def _append_recorded_ids(
        self,
        rec_ids: set[str],
        group: str,
        base_name: str,
        seq: int,
        *,
        sync: Optional[bool] = None,
    ) -> None:
        """Append durable duplicate-detection IDs for one accepted record."""
        self._append_jsonl_many(
            self.recorded_ids_path,
            (
                {
                    "seq": seq,
                    "group": self._sanitize_label(group),
                    "base_name": base_name,
                    "id": str(rec_id),
                }
                for rec_id in sorted(rec_ids)
            ),
            sync=sync,
        )

    def _load_journal_backlog(self) -> None:
        """Replay uncompacted journaled records back into memory buffers."""
        if not self.journal_root.exists():
            return
        max_seq = 0
        for journal_path in sorted(self.journal_root.glob("*/*.jsonl")):
            group = journal_path.parent.name
            base_name = journal_path.stem
            key = (group, base_name)
            last_compacted = self._compacted_seq_by_key.get(key, 0)
            try:
                with journal_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            continue
                        seq = int(payload.get("seq") or 0)
                        if seq > max_seq:
                            max_seq = seq
                        if seq <= last_compacted:
                            continue
                        rec_ids = {str(value) for value in (payload.get("ids") or []) if value}
                        record = payload.get("record")
                        if not isinstance(record, dict) or not rec_ids:
                            continue
                        self._buffers.setdefault(key, []).append((seq, rec_ids, record))
            except Exception as exc:
                print(f"[storage] Failed to load journal backlog from {journal_path}: {exc}")
        self._next_seq = max_seq + 1

    def _save_parquet(self, data: Sequence[dict], name: str, group: str) -> Optional[Path]:
        """Write one Parquet batch to the requested output group."""
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            print("pandas not available; skipping Parquet export.")
            return None

        def _normalize(value: Any) -> Any:
            """Normalize nested values into Parquet-friendly structures."""
            if isinstance(value, dict):
                if not value:
                    return None
                return {str(k): _normalize(v) for k, v in value.items()}
            if isinstance(value, list):
                if not value:
                    return None
                return [_normalize(item) for item in value]
            if isinstance(value, tuple):
                if not value:
                    return None
                return [_normalize(item) for item in value]
            return value

        data_dir = self._ensure_group_dirs(group)
        path = data_dir / f"{name}.parquet"
        normalized = [_normalize(row) for row in data]
        df = pd.DataFrame(normalized)
        df.to_parquet(path, index=False)
        return path

    def _flush_group(self, group: str, base_name: str) -> None:
        """Flush one buffered group/base-name pair to a numbered Parquet batch."""
        key = (group, base_name)
        buffered = self._buffers.get(key, [])
        if not buffered:
            return
        batch_idx = self._batch_counts.get(key, 0) + 1
        self._batch_counts[key] = batch_idx
        fname = f"{base_name}_batch-{batch_idx:04d}"
        records = [rec for _, _, rec in buffered]
        if self.durable_journal:
            # In durable mode, force journal data to disk before declaring that
            # the batch has a Parquet representation.
            self._sync_journal_paths(
                {
                    self._journal_file(group, base_name),
                    self.recorded_ids_path,
                }
            )
        path = self._save_parquet(records, fname, group)
        if path is None:
            return
        if self.durable_journal:
            highest_seq = max(seq for seq, _, _ in buffered)
            self._compacted_seq_by_key[key] = highest_seq
            self._save_compaction_state()
        else:
            # Non-durable mode only records IDs after Parquet succeeds; pending
            # IDs still prevent duplicates inside the active in-memory batch.
            for _, rec_ids, _ in buffered:
                for rec_id in rec_ids:
                    self._seen_ids.add(rec_id)
                    self._pending_ids.discard(rec_id)
            self._save_manifest()
        self._buffers[key] = []

    def persist_one(self, record: Any, base_name: str = "pr", group: str = "unknown") -> bool:
        """Persist a single record into the requested group folder."""
        rec_dicts = self._to_dicts([record])
        rec = rec_dicts[0] if rec_dicts else None
        rec_ids = self._record_ids(record, rec)
        if not rec_ids:
            print("[storage] Skipping record with no identifiable id/url.")
            return False
        if any(
            rec_id in self._seen_ids or (not self.durable_journal and rec_id in self._pending_ids)
            for rec_id in rec_ids
        ):
            print("[storage] Skipping duplicate record (id/url already recorded).")
            return False
        rec_id = next(iter(rec_ids))

        safe_group = self._sanitize_label(group)
        base_name = self._name_with_run_label(base_name)
        print(f"[storage] Buffering record {rec_id} (group={safe_group}, batch={base_name})")
        key = (safe_group, base_name)
        if self.durable_journal:
            # Durable mode writes the record and its deduplication IDs before
            # buffering so restart replay can recover any unflushed batch.
            seq = self._next_seq
            self._next_seq += 1
            sync_journal_write = self.durable_journal_fsync_interval == 1
            self._append_jsonl(
                self._journal_file(safe_group, base_name),
                {
                    "seq": seq,
                    "group": safe_group,
                    "base_name": base_name,
                    "ids": sorted(rec_ids),
                    "record": rec,
                },
                sync=sync_journal_write,
            )
            self._append_recorded_ids(
                rec_ids,
                safe_group,
                base_name,
                seq,
                sync=sync_journal_write,
            )
            self._buffers.setdefault(key, []).append((seq, rec_ids, rec))
            self._seen_ids.update(rec_ids)
        else:
            # Non-durable mode is faster and adequate for small smoke runs, but
            # pending records only live in memory until the next Parquet flush.
            self._buffers.setdefault(key, []).append((0, rec_ids, rec))
            self._pending_ids.update(rec_ids)
        if len(self._buffers[key]) >= self.batch_size:
            self._flush_group(safe_group, base_name)

        return True

    def is_recorded_any(self, url: Optional[str] = None, database_id: Optional[str] = None) -> bool:
        """Return True when any provided identifier is already in the manifest."""
        if url and url in self._seen_ids:
            return True
        if database_id and database_id in self._seen_ids:
            return True
        return False

    def is_recorded(self, url: Optional[str]) -> bool:
        """Return True when a PR URL has already reached durable local output."""
        return self.is_recorded_any(url=url)

    def seen_ids(self) -> set[str]:
        """Return a copy of all recorded ids."""
        return set(self._seen_ids)

    def flush_local(self) -> None:
        """Flush all buffered records to local Parquet files."""
        for group, base_name in list(self._buffers.keys()):
            self._flush_group(group, base_name)
        if self.durable_journal:
            self._sync_journal_paths()
