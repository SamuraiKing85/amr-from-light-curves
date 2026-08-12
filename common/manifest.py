"""
Data Manifest - Track Downloaded and Processed Files
========================================================
Keeps a persistent record of which files have been downloaded, processed,
or both. Allows raw data to be safely deleted after processing because
the manifest remembers what was handled.

The manifest is a JSON file stored alongside checkpoints. Each entry
records the filename, when it was first seen, when it was processed,
its size, and optionally a hash.

Usage:
    from common.manifest import DataManifest

    manifest = DataManifest("mmt9_raw")

    # Record a download
    manifest.record_downloaded("20140101.txt", size_bytes=2_400_000)

    # Later, after processing
    manifest.record_processed("20140101.txt")

    # Check if we've already handled a file (even if deleted)
    if manifest.was_processed("20140101.txt"):
        print("Already processed, skip")

    # Find new files that exist on disk but aren't in the manifest
    new_files = manifest.find_new_files(directory, "*.txt")

    # Summary
    manifest.print_status()
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

__all__ = ["DataManifest"]


class DataManifest:
    """Persistent manifest tracking downloaded and processed files.

    Stored as JSON:
        {
            "name": "mmt9_raw",
            "files": {
                "20140101.txt": {
                    "downloaded_at": "2025-02-24T10:30:00",
                    "processed_at": "2025-02-24T11:00:00",
                    "size_bytes": 2400000,
                    "source": "https://..."
                },
                ...
            },
            "metadata": {}
        }

    Args:
        name:          Manifest identifier (becomes the filename).
        manifest_dir:  Directory to store manifest files.
                       Defaults to config.paths.CHECKPOINT_DIR.
    """

    def __init__(self, name: str, manifest_dir: Optional[Path] = None):
        if manifest_dir is None:
            from config.paths import CHECKPOINT_DIR
            manifest_dir = CHECKPOINT_DIR

        self._dir = Path(manifest_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"manifest_{name}.json"
        self._name = name

        self._files: dict = {}
        self._metadata: dict = {}
        self._load()

    # ── Properties ─

    @property
    def total_files(self) -> int:
        return len(self._files)

    @property
    def downloaded_count(self) -> int:
        return sum(1 for f in self._files.values() if f.get("downloaded_at"))

    @property
    def processed_count(self) -> int:
        return sum(1 for f in self._files.values() if f.get("processed_at"))

    @property
    def all_filenames(self) -> set:
        return set(self._files.keys())

    @property
    def metadata(self) -> dict:
        return self._metadata

    # ── Recording ──

    def record_downloaded(self, filename: str, size_bytes: int = 0,
                          source: str = "") -> None:
        """Record that a file has been downloaded."""
        if filename not in self._files:
            self._files[filename] = {}

        entry = self._files[filename]
        entry["downloaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if size_bytes:
            entry["size_bytes"] = size_bytes
        if source:
            entry["source"] = source

        self._save()

    def record_processed(self, filename: str) -> None:
        """Record that a file has been processed."""
        if filename not in self._files:
            self._files[filename] = {}

        self._files[filename]["processed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save()

    def record_batch_downloaded(self, filenames: list, source: str = "") -> None:
        """Record multiple files as downloaded at once."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for fname in filenames:
            if fname not in self._files:
                self._files[fname] = {}
            self._files[fname]["downloaded_at"] = now
            if source:
                self._files[fname]["source"] = source
        self._save()

    def record_batch_processed(self, filenames: list) -> None:
        """Record multiple files as processed at once."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for fname in filenames:
            if fname not in self._files:
                self._files[fname] = {}
            self._files[fname]["processed_at"] = now
        self._save()

    def save_metadata(self, data: dict) -> None:
        """Update manifest metadata."""
        self._metadata.update(data)
        self._save()

    # ── Queries ────

    def was_downloaded(self, filename: str) -> bool:
        """Check if a file was ever downloaded (even if since deleted)."""
        return filename in self._files and bool(self._files[filename].get("downloaded_at"))

    def was_processed(self, filename: str) -> bool:
        """Check if a file was ever processed (even if since deleted)."""
        return filename in self._files and bool(self._files[filename].get("processed_at"))

    def get_entry(self, filename: str) -> Optional[dict]:
        """Get full entry for a file, or None."""
        return self._files.get(filename)

    def find_new_files(self, directory: Path, pattern: str = "*") -> list[str]:
        """Find files on disk that aren't in the manifest yet.

        Useful for detecting newly released files.
        """
        if not directory.exists():
            return []

        on_disk = {f.name for f in directory.glob(pattern) if f.is_file()}
        known = self.all_filenames
        return sorted(on_disk - known)

    def find_unprocessed(self, directory: Path = None,
                         pattern: str = "*") -> list[str]:
        """Find files that were downloaded but not yet processed.

        If directory is given, also checks that the file still exists on disk.
        """
        unprocessed = []
        for fname, entry in self._files.items():
            if entry.get("downloaded_at") and not entry.get("processed_at"):
                if directory:
                    if (directory / fname).exists():
                        unprocessed.append(fname)
                else:
                    unprocessed.append(fname)
        return sorted(unprocessed)

    def find_deletable(self, directory: Path, pattern: str = "*") -> list[Path]:
        """Find files on disk that have been fully processed and can be deleted.

        Only returns files that exist AND have been marked as processed.
        """
        deletable = []
        for f in directory.glob(pattern):
            if f.is_file() and self.was_processed(f.name):
                deletable.append(f)
        return sorted(deletable)

    # ── Display ────

    def print_status(self) -> None:
        """Print a summary of the manifest."""
        print(f"  Manifest: {self._name}")
        print(f"    Total tracked:   {self.total_files:,}")
        print(f"    Downloaded:      {self.downloaded_count:,}")
        print(f"    Processed:       {self.processed_count:,}")

        # Count what's still on disk vs what's been cleaned up
        if self._metadata.get("directory"):
            d = Path(self._metadata["directory"])
            if d.exists():
                on_disk = sum(1 for f in d.iterdir() if f.is_file())
                print(f"    On disk:         {on_disk:,}")
                deletable = len(self.find_deletable(d))
                if deletable:
                    print(f"    Safe to delete:  {deletable:,}")

    # ── Persistence ────

    def _save(self) -> None:
        """Save manifest to disk atomically."""
        state = {
            "name": self._name,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "files": self._files,
            "metadata": self._metadata,
        }
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        if os.name == "nt" and self._path.exists():
            self._path.unlink()
        os.replace(str(tmp), str(self._path))

    def _load(self) -> None:
        """Load manifest from disk."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._files = state.get("files", {})
            self._metadata = state.get("metadata", {})
        except (json.JSONDecodeError, KeyError):
            backup = self._path.with_suffix(".json.corrupt")
            self._path.rename(backup)
            print(f"  [WARN] Corrupted manifest moved to {backup.name}")

    def clear(self) -> None:
        """Remove the manifest file entirely."""
        if self._path.exists():
            self._path.unlink()
        self._files.clear()
        self._metadata.clear()

    def __repr__(self) -> str:
        return (f"DataManifest('{self._name}', "
                f"files={self.total_files}, processed={self.processed_count})")
