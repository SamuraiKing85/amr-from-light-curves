"""
Checkpoint & Resume Manager
===============================
Provides consistent checkpoint/resume behaviour across all pipeline scripts.
Uses atomic writes (write to .tmp, then rename) to prevent corruption on crash.

Usage:
    from common.progress import CheckpointManager
    from config.paths import CHECKPOINT_DIR

    cp = CheckpointManager("mmt9_download", CHECKPOINT_DIR)

    # Check if an item was already processed
    if cp.is_done("file_001.txt"):
        continue

    # Process the item...

    # Mark as done
    cp.mark_done("file_001.txt")

    # Store arbitrary metadata alongside completed items
    cp.save_metadata({"last_page": 42, "total_objects": 89054})

    # Clean up after successful completion
    cp.clear()
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """Manages checkpoint state for resumable operations.

    State is stored as a JSON file:
        {
            "name": "mmt9_download",
            "created_at": "2025-02-24T10:30:00",
            "updated_at": "2025-02-24T11:45:30",
            "completed": ["file_001.txt", "file_002.txt", ...],
            "failed": ["file_099.txt", ...],
            "metadata": { ... }
        }

    All writes are atomic: data is written to a .tmp file first,
    then renamed over the target. This prevents corruption if the
    process is killed mid-write.

    Args:
        name:           Unique name for this checkpoint (becomes the filename).
        checkpoint_dir: Directory where checkpoint files are stored.
        auto_save:      If > 0, automatically flush to disk every N mark_done() calls.
    """

    def __init__(self, name: str, checkpoint_dir: Optional[Path] = None,
                 auto_save: int = 50):
        if checkpoint_dir is None:
            # Import here to avoid circular imports at module level
            from config.paths import CHECKPOINT_DIR
            checkpoint_dir = CHECKPOINT_DIR

        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._path = self._dir / f"{name}.json"
        self._tmp_path = self._dir / f"{name}.json.tmp"
        self._name = name
        self._auto_save = auto_save
        self._dirty_count = 0

        # Internal state
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._metadata: dict[str, Any] = {}
        self._created_at: Optional[str] = None
        self._updated_at: Optional[str] = None

        # Load existing checkpoint if present
        self._load()

    # ── Properties ────

    @property
    def completed_count(self) -> int:
        """Number of items marked as done."""
        return len(self._completed)

    @property
    def failed_count(self) -> int:
        """Number of items marked as failed."""
        return len(self._failed)

    @property
    def completed_items(self) -> set[str]:
        """Set of all completed item IDs (read-only copy)."""
        return set(self._completed)

    @property
    def failed_items(self) -> set[str]:
        """Set of all failed item IDs (read-only copy)."""
        return set(self._failed)

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata dict."""
        return self._metadata

    @property
    def exists(self) -> bool:
        """Whether a checkpoint file exists on disk."""
        return self._path.exists()

    # ── Core Operations ───

    def is_done(self, item_id: str) -> bool:
        """Check if an item has been completed."""
        return str(item_id) in self._completed

    def is_failed(self, item_id: str) -> bool:
        """Check if an item previously failed."""
        return str(item_id) in self._failed

    def mark_done(self, item_id: str) -> None:
        """Mark an item as successfully completed.

        If the item was previously in the failed set, it is removed
        from there (retry succeeded).
        """
        key = str(item_id)
        self._completed.add(key)
        self._failed.discard(key)
        self._dirty_count += 1

        if self._auto_save > 0 and self._dirty_count >= self._auto_save:
            self.save()

    def mark_failed(self, item_id: str) -> None:
        """Mark an item as failed (can be retried later)."""
        key = str(item_id)
        self._failed.add(key)
        self._dirty_count += 1

        if self._auto_save > 0 and self._dirty_count >= self._auto_save:
            self.save()

    def save_metadata(self, data: dict[str, Any]) -> None:
        """Update the metadata dict and flush to disk."""
        self._metadata.update(data)
        self.save()

    # ── Persistence ───

    def save(self) -> None:
        """Flush current state to disk atomically.

        Writes to a .tmp file first, then renames. This ensures the
        checkpoint file is never in a half-written state.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self._created_at is None:
            self._created_at = now

        state = {
            "name": self._name,
            "created_at": self._created_at,
            "updated_at": now,
            "completed": sorted(self._completed),
            "failed": sorted(self._failed),
            "metadata": self._metadata,
        }

        # Atomic write: tmp file → rename
        try:
            with open(self._tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            # os.replace is atomic on POSIX. On Windows it's atomic if
            # the destination doesn't exist or is on the same filesystem.
            if os.name == "nt" and self._path.exists():
                self._path.unlink()
            os.replace(str(self._tmp_path), str(self._path))

        except Exception:
            # If rename fails, try to clean up the tmp file
            if self._tmp_path.exists():
                try:
                    self._tmp_path.unlink()
                except OSError:
                    pass
            raise

        self._dirty_count = 0
        self._updated_at = now

    def _load(self) -> None:
        """Load state from disk if a checkpoint exists."""
        if not self._path.exists():
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self._completed = set(state.get("completed", []))
            self._failed = set(state.get("failed", []))
            self._metadata = state.get("metadata", {})
            self._created_at = state.get("created_at")
            self._updated_at = state.get("updated_at")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupted checkpoint - start fresh but keep the broken file
            backup = self._path.with_suffix(".json.corrupt")
            self._path.rename(backup)
            print(f"  [WARN] Corrupted checkpoint moved to {backup.name}")
            print(f"         Error: {e}")
            print(f"         Starting fresh.")

    def clear(self) -> None:
        """Remove the checkpoint file (call after successful completion)."""
        if self._path.exists():
            self._path.unlink()
        if self._tmp_path.exists():
            self._tmp_path.unlink()
        self._completed.clear()
        self._failed.clear()
        self._metadata.clear()
        self._created_at = None
        self._updated_at = None
        self._dirty_count = 0

    # ── Display ───────

    def print_status(self) -> None:
        """Print a summary of the checkpoint state."""
        if not self.exists and not self._completed:
            print(f"  No checkpoint found for '{self._name}'.")
            return

        print(f"  Checkpoint: {self._name}")
        print(f"    Completed:  {self.completed_count:,}")
        if self._failed:
            print(f"    Failed:     {self.failed_count:,}")
        if self._updated_at:
            print(f"    Last saved: {self._updated_at}")
        if self._metadata:
            for key, val in self._metadata.items():
                print(f"    {key}: {val}")

    def __repr__(self) -> str:
        return (f"CheckpointManager('{self._name}', "
                f"done={self.completed_count}, failed={self.failed_count})")
