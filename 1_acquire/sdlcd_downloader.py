"""
SDLCD Light Curve Downloader
================================
Downloads light curve data files from the SDLCD (Slovak Database of
Light Curves of Debris) using the database.json metadata file.

The database.json must be placed in data/raw/sdlcd/ before running.
Download it from: https://www.sdlcd.space-debris.sk

Interactive menu (no arguments):
    python sdlcd_downloader.py

CLI mode:
    python sdlcd_downloader.py --download
    python sdlcd_downloader.py --download --type debris
    python sdlcd_downloader.py --status
    python sdlcd_downloader.py --verify
    python sdlcd_downloader.py --download --fit-files
"""

__version__ = "2.0.0"

import sys
import os
import json
import hashlib
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import RAW_SDLCD, ensure_dirs
from common.menu import (
    print_header, prompt_choice, prompt_confirm, prompt_path,
    print_section, print_status_bar, print_summary_table,
    print_warning, print_error,
)
from common.progress import CheckpointManager
from common.network import create_session, RateLimiter
from common.manifest import DataManifest
from common.logging_setup import setup_logging


#  REMOTE ENDPOINTS - edit these if URLs change


STATIC_DATA_URL = "https://www.sdlcd.space-debris.sk/static/data/"
DATABASE_URL    = "https://www.sdlcd.space-debris.sk/static/data/database.json"


#  DOWNLOAD SETTINGS


DOWNLOAD_TIMEOUT = 30    # Seconds per file
REQUEST_DELAY    = 0.3   # Seconds between downloads
MAX_RETRIES      = 3     # Retries per file


#  SETUP


logger = setup_logging("sdlcd_downloader")



#  DATABASE LOADING


def find_database(raw_dir: Path) -> Path:
    """Locate database.json in the raw SDLCD directory.

    If not found locally, offers to download it from the SDLCD website.
    """
    candidates = [
        raw_dir / "database.json",
        raw_dir / "Database.json",
        Path("database.json"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # Not found, try to download it
    print_warning(f"database.json not found in {raw_dir}")
    print(f"          Downloading from SDLCD website...")

    target_path = raw_dir / "database.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        session = create_session("SDLCD-Downloader/2.0")
        resp = session.get(DATABASE_URL, timeout=30, verify=False)
        resp.raise_for_status()

        # Validate it's actual JSON
        data = resp.json()
        if not data or not isinstance(data, dict):
            print_error("Downloaded file doesn't look like valid SDLCD data.")
            session.close()
            return None

        target_path.write_text(resp.text, encoding="utf-8")
        session.close()

        print(f"          Saved to: {target_path}")
        print(f"          ({len(data)} entries)\n")
        return target_path

    except Exception as e:
        print_error(f"Failed to download database.json: {e}")
        print(f"          Download manually from:")
        print(f"          {DATABASE_URL}")
        print(f"          and place it in: {raw_dir}/")
        return None


def load_database(json_path: Path) -> list[dict]:
    """Parse SDLCD database.json into a list of entry dicts."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for idx, entry in data.items():
        entries.append({
            "sdlcd_id": int(idx),
            "cospar": entry.get("object-id", ""),
            "object_name": entry.get("object-name", ""),
            "object_type": entry.get("object-type", ""),
            "norad": entry.get("norad", ""),
            "data_file": entry.get("data-file", ""),
            "data_fit_file": entry.get("data-fit-file", ""),
        })

    logger.info(f"Loaded {len(entries)} entries from {json_path.name}")
    return entries


def build_file_list(entries: list[dict], object_type: str = None,
                    include_fit: bool = False) -> list[dict]:
    """Build the list of files to download from database entries.

    Args:
        entries:      Parsed database entries.
        object_type:  Filter by type (e.g., "debris", "spacecraft", "rocket body").
        include_fit:  Also include _DATA_fit.txt files.

    Returns:
        List of {"filename": str, "url": str, "entry": dict} dicts.
    """
    files = []

    for entry in entries:
        # Type filter
        if object_type:
            if entry["object_type"].lower() != object_type.lower():
                continue

        if entry["data_file"]:
            files.append({
                "filename": entry["data_file"],
                "url": STATIC_DATA_URL + entry["data_file"],
                "entry": entry,
                "is_fit": False,
            })

        if include_fit and entry["data_fit_file"]:
            files.append({
                "filename": entry["data_fit_file"],
                "url": STATIC_DATA_URL + entry["data_fit_file"],
                "entry": entry,
                "is_fit": True,
            })

    return files


def print_database_summary(entries: list[dict]):
    """Print a summary of the database contents."""
    types = {}
    for e in entries:
        t = e["object_type"] or "Unknown"
        types[t] = types.get(t, 0) + 1

    with_data = sum(1 for e in entries if e["data_file"])
    with_fit = sum(1 for e in entries if e["data_fit_file"])

    print_section("DATABASE SUMMARY")
    print_summary_table([
        ("Total entries:", f"{len(entries):,}"),
        ("With data files:", f"{with_data:,}"),
        ("With fit files:", f"{with_fit:,}"),
    ])
    print()
    print(f"  By object type:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t:<20s}  {count:,}")
    print()



#  DOWNLOAD ENGINE


def download_files(
    session,
    file_list: list[dict],
    output_dir: Path,
    checkpoint: CheckpointManager,
    limiter: RateLimiter,
    manifest: DataManifest = None,
    dry_run: bool = False,
    redownload_failed: bool = False,
) -> dict:
    """Download light curve files with checkpoint/resume.

    Returns:
        {"downloaded": int, "skipped": int, "failed": int, "bytes": int}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0, "not_found": 0}
    total = len(file_list)

    for i, item in enumerate(file_list, 1):
        filename = item["filename"]

        # Skip completed
        if checkpoint.is_done(filename):
            stats["skipped"] += 1
            continue

        # Skip failed unless retrying
        if checkpoint.is_failed(filename) and not redownload_failed:
            stats["skipped"] += 1
            continue

        # Skip if file exists on disk
        outpath = output_dir / filename
        if outpath.exists() and outpath.stat().st_size > 10:
            checkpoint.mark_done(filename)
            if manifest:
                manifest.record_downloaded(filename, size_bytes=outpath.stat().st_size,
                                           source=STATIC_DATA_URL)
            stats["skipped"] += 1
            continue

        if dry_run:
            print(f"  [{i}/{total}] Would download: {filename}")
            stats["downloaded"] += 1
            continue

        print(f"  [{i}/{total}] {filename}...", end="", flush=True)

        try:
            resp = session.get(item["url"], timeout=DOWNLOAD_TIMEOUT, verify=False)

            if resp.status_code == 404:
                checkpoint.mark_failed(filename)
                stats["not_found"] += 1
                print(f" 404 not found")
                logger.debug(f"404: {item['url']}")
                limiter.wait()
                continue

            resp.raise_for_status()

            # Validate response has actual data
            if len(resp.text.strip()) <= 10:
                checkpoint.mark_failed(filename)
                stats["failed"] += 1
                print(f" empty response")
                limiter.wait()
                continue

            outpath.write_text(resp.text, encoding="utf-8")
            size_kb = len(resp.content) / 1024

            checkpoint.mark_done(filename)
            if manifest:
                manifest.record_downloaded(filename, size_bytes=len(resp.content),
                                           source=STATIC_DATA_URL)
            stats["downloaded"] += 1
            stats["bytes"] += len(resp.content)
            print(f" {size_kb:.1f} KB")

        except Exception as e:
            checkpoint.mark_failed(filename)
            stats["failed"] += 1
            print(f" FAILED: {e}")
            logger.warning(f"Failed to download {filename}: {e}")

        limiter.wait()

    checkpoint.save()
    return stats



#  VERIFICATION


def verify_downloads(output_dir: Path, file_list: list[dict]) -> dict:
    """Check downloaded files for completeness."""
    result = {"ok": 0, "missing": 0, "empty": 0, "total": len(file_list)}

    for item in file_list:
        path = output_dir / item["filename"]
        if not path.exists():
            result["missing"] += 1
        elif path.stat().st_size <= 10:
            result["empty"] += 1
        else:
            result["ok"] += 1

    return result


def write_manifest(output_dir: Path) -> Path:
    """Generate SHA256 manifest of all downloaded files."""
    manifest_path = output_dir / "MANIFEST.sha256"
    files = sorted(f for f in output_dir.iterdir()
                   if f.suffix == ".txt" and f.name != "MANIFEST.sha256")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for fp in files:
            sha = hashlib.sha256(fp.read_bytes()).hexdigest()
            f.write(f"{sha}  {fp.name}\n")

    return manifest_path



#  STATUS


def show_status(output_dir: Path, checkpoint: CheckpointManager, total_files: int):
    """Display download status."""
    print_section("DOWNLOAD STATUS")

    local = [f for f in output_dir.iterdir() if f.suffix == ".txt"] if output_dir.exists() else []
    total_size = sum(f.stat().st_size for f in local) / (1024 * 1024)

    print_summary_table([
        ("Files on disk:",        f"{len(local):,}"),
        ("Total size:",           f"{total_size:.1f} MB"),
        ("Checkpoint completed:", f"{checkpoint.completed_count:,}"),
        ("Checkpoint failed:",    f"{checkpoint.failed_count:,}"),
    ])
    print()
    print_status_bar("Progress", checkpoint.completed_count, total_files)
    print()



#  PRINT SUMMARY


def print_download_summary(stats: dict, output_dir: Path):
    """Print summary after download run."""
    print_section("DOWNLOAD SUMMARY")
    rows = [
        ("Downloaded:",  f"{stats['downloaded']:,}"),
        ("Skipped:",     f"{stats['skipped']:,}"),
        ("Failed:",      f"{stats['failed']:,}"),
    ]
    if stats.get("not_found", 0) > 0:
        rows.append(("Not found (404):", f"{stats['not_found']:,}"))
    rows.extend([
        ("Data fetched:", f"{stats['bytes'] / (1024*1024):.1f} MB"),
        ("Output:",      str(output_dir)),
    ])
    print_summary_table(rows)
    print()



#  INTERACTIVE MENU


def run_interactive(output_dir: Path):
    """Run the interactive menu loop."""
    print_header("SDLCD Downloader", __version__,
                 "Download light curve files from SDLCD database")

    checkpoint = CheckpointManager("sdlcd_download")
    manifest = DataManifest("sdlcd_raw")
    limiter = RateLimiter(requests_per_minute=120)
    session = create_session("SDLCD-Downloader/2.0")

    # Find and load database
    db_path = find_database(output_dir)
    if db_path is None:
        db_path = prompt_path("Path to database.json", must_exist=True, is_dir=False)
        if db_path is None:
            return

    entries = load_database(db_path)
    print_database_summary(entries)

    lc_dir = output_dir / "lightcurves"
    manifest.save_metadata({"directory": str(lc_dir)})

    while True:
        options = [
            ("download",    "Download all light curve files"),
            ("type",        "Download by object type"),
        ]

        has_data = checkpoint.completed_count > 0 or (lc_dir.exists() and list(lc_dir.glob("*.txt")))
        if has_data:
            options.append(("status",   "Check download status"))
            options.append(("verify",   "Verify downloaded files"))
            options.append(("sha256",   "Generate SHA256 manifest"))
            options.append(("cleanup",  "Show files safe to delete (already processed)"))

        if checkpoint.failed_count > 0:
            options.append(("retry", f"Re-download {checkpoint.failed_count} failed files"))

        options.append(("quit", "Quit"))

        choice = prompt_choice("What would you like to do?", options)

        if choice is None or choice == "quit":
            print("\n  Goodbye!\n")
            break

        elif choice == "download":
            file_list = build_file_list(entries)
            print(f"\n  {len(file_list)} files to process\n")
            stats = download_files(
                session, file_list, lc_dir, checkpoint, limiter,
                manifest=manifest,
            )
            print_download_summary(stats, lc_dir)

        elif choice == "type":
            types = sorted(set(e["object_type"] for e in entries if e["object_type"]))
            type_options = [(t.lower(), t) for t in types]
            type_options.append(("back", "Back to main menu"))
            selected = prompt_choice("Select object type:", type_options)

            if selected and selected != "back":
                file_list = build_file_list(entries, object_type=selected)
                print(f"\n  {len(file_list)} {selected} files to process\n")
                stats = download_files(
                    session, file_list, lc_dir, checkpoint, limiter,
                    manifest=manifest,
                )
                print_download_summary(stats, lc_dir)

        elif choice == "retry":
            file_list = build_file_list(entries)
            failed = [f for f in file_list if checkpoint.is_failed(f["filename"])]
            print(f"\n  Retrying {len(failed)} failed file(s)...\n")
            stats = download_files(
                session, failed, lc_dir, checkpoint, limiter,
                manifest=manifest, redownload_failed=True,
            )
            print_download_summary(stats, lc_dir)

        elif choice == "status":
            file_list = build_file_list(entries)
            show_status(lc_dir, checkpoint, total_files=len(file_list))
            if manifest.total_files > 0:
                manifest.print_status()
                print()

        elif choice == "verify":
            file_list = build_file_list(entries)
            result = verify_downloads(lc_dir, file_list)
            print_section("VERIFICATION")
            print_summary_table([
                ("Expected files:", f"{result['total']:,}"),
                ("OK on disk:",     f"{result['ok']:,}"),
                ("Missing:",        f"{result['missing']:,}"),
                ("Empty:",          f"{result['empty']:,}"),
            ])
            print()

        elif choice == "sha256":
            print(f"\n  Generating SHA256 manifest...", end="", flush=True)
            path = write_manifest(lc_dir)
            n = sum(1 for _ in lc_dir.glob("*.txt"))
            print(f" done ({n} files)")
            print(f"  Saved to: {path}\n")

        elif choice == "cleanup":
            deletable = manifest.find_deletable(lc_dir, "*.txt")
            if deletable:
                total_mb = sum(f.stat().st_size for f in deletable) / (1024 * 1024)
                print(f"\n  {len(deletable)} file(s) ({total_mb:.1f} MB) can be safely deleted.")
                print(f"  These files have already been converted to per-satellite CSVs")
                print(f"  by the SDLCD processor. The processed data in")
                print(f"  data/processed/sdlcd/ contains everything needed for training.")
                if prompt_confirm(f"\n  Delete {len(deletable)} processed raw file(s)?"):
                    for f in deletable:
                        f.unlink()
                    print(f"  Deleted {len(deletable)} files, freed {total_mb:.1f} MB.\n")
                else:
                    print(f"  No files deleted.\n")
            else:
                print(f"\n  No files are safe to delete yet.")
                print(f"  Run 2_process/sdlcd_processor.py first to process raw files.\n")

    session.close()



#  CLI MODE


def run_cli(args, output_dir: Path):
    """Run in CLI mode."""
    print_header("SDLCD Downloader", __version__)

    checkpoint = CheckpointManager("sdlcd_download")
    limiter = RateLimiter(requests_per_minute=120)
    session = create_session("SDLCD-Downloader/2.0")

    db_path = Path(args.database) if args.database else find_database(output_dir)
    if db_path is None:
        sys.exit(1)

    entries = load_database(db_path)
    lc_dir = output_dir / "lightcurves"

    try:
        if args.status:
            file_list = build_file_list(entries, object_type=args.type)
            show_status(lc_dir, checkpoint, total_files=len(file_list))

        elif args.verify:
            file_list = build_file_list(entries, object_type=args.type)
            result = verify_downloads(lc_dir, file_list)
            print_section("VERIFICATION")
            print_summary_table([
                ("Expected files:", f"{result['total']:,}"),
                ("OK on disk:",     f"{result['ok']:,}"),
                ("Missing:",        f"{result['missing']:,}"),
                ("Empty:",          f"{result['empty']:,}"),
            ])
            print()
            sys.exit(0 if result["missing"] == 0 else 1)

        elif args.download:
            file_list = build_file_list(
                entries, object_type=args.type, include_fit=args.fit_files,
            )
            print(f"\n  {len(file_list)} files to process\n")
            stats = download_files(
                session, file_list, lc_dir, checkpoint, limiter,
                dry_run=args.dry_run,
            )
            print_download_summary(stats, lc_dir)
            sys.exit(0 if stats["failed"] == 0 else 1)

    finally:
        session.close()



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="SDLCD Light Curve Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sdlcd_downloader.py                           # Interactive menu
  python sdlcd_downloader.py --download                # Download all
  python sdlcd_downloader.py --download --type debris  # Only debris
  python sdlcd_downloader.py --status                  # Check progress
  python sdlcd_downloader.py --verify                  # Verify files
  python sdlcd_downloader.py --download --fit-files    # Include fit files
        """,
    )
    parser.add_argument("--download", action="store_true", help="Download files")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--verify", action="store_true", help="Verify downloads")
    parser.add_argument("--type", type=str, default=None,
                        help="Filter by object type (debris, spacecraft, rocket body)")
    parser.add_argument("--database", type=str, default=None,
                        help="Path to database.json")
    parser.add_argument("--fit-files", action="store_true",
                        help="Also download _DATA_fit.txt files")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    ensure_dirs()
    output_dir = Path(args.output) if args.output else RAW_SDLCD

    has_action = args.download or args.status or args.verify

    try:
        if has_action:
            run_cli(args, output_dir)
        else:
            run_interactive(output_dir)
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Progress saved.")
        sys.exit(2)


if __name__ == "__main__":
    main()
