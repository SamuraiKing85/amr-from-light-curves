"""
MMT-9 Light Curve Downloader
================================
Downloads raw .txt observation files from the MMT-9 satellite data archive.

Interactive menu (no arguments):
    python mmt9_downloader.py

CLI mode:
    python mmt9_downloader.py --download
    python mmt9_downloader.py --status
    python mmt9_downloader.py --verify
    python mmt9_downloader.py --download --dry-run
"""

__version__ = "2.0.0"

import sys
import os
import hashlib
import argparse
import urllib3
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import RAW_MMT9, ensure_dirs
from common.menu import (
    print_header, prompt_choice, prompt_confirm,
    print_section, print_status_bar, print_summary_table,
    print_warning, print_error,
)
from common.progress import CheckpointManager
from common.network import create_session, fetch_with_retry, RateLimiter
from common.manifest import DataManifest
from common.logging_setup import setup_logging


#  REMOTE ENDPOINTS - edit these if URLs change


BASE_URL   = "https://relay.sao.ru/lynx/karpov/satellites/"
VERIFY_SSL = False   # MMT-9 server has a broken/expired SSL certificate


#  DOWNLOAD SETTINGS


DOWNLOAD_TIMEOUT  = 120   # Seconds per file (some are large)
REQUEST_DELAY     = 0.3   # Seconds between downloads (be polite)
MAX_RETRIES       = 3     # Retries per file on failure


#  SETUP


logger = setup_logging("mmt9_downloader")

# Suppress SSL warnings only for the MMT-9 domain
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



#  REMOTE FILE LISTING


def get_remote_file_list(session) -> list[str]:
    """Fetch the MMT-9 directory page and extract all .txt filenames."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print_error("beautifulsoup4 is required. Run: python setup.py --install")
        sys.exit(1)

    logger.info(f"Fetching file list from {BASE_URL}")
    resp = fetch_with_retry(session, BASE_URL, timeout=30, verify=VERIFY_SSL)

    soup = BeautifulSoup(resp.text, "html.parser")
    files = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".txt"):
            files.append(href)

    logger.info(f"Found {len(files)} .txt files on remote server")
    return sorted(files)



#  DOWNLOAD ENGINE


def download_files(
    session,
    remote_files: list[str],
    output_dir: Path,
    checkpoint: CheckpointManager,
    limiter: RateLimiter,
    manifest: DataManifest = None,
    dry_run: bool = False,
    redownload_failed: bool = False,
) -> dict:
    """Download files with checkpoint/resume support.

    Returns:
        {"downloaded": int, "skipped": int, "failed": int, "bytes": int}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0}
    total = len(remote_files)

    for i, filename in enumerate(remote_files, 1):
        # Skip if already completed (and not retrying failures)
        if checkpoint.is_done(filename):
            stats["skipped"] += 1
            continue

        # Skip previously failed files unless explicitly retrying
        if checkpoint.is_failed(filename) and not redownload_failed:
            stats["skipped"] += 1
            continue

        # Skip if file already exists on disk (quick resume without checkpoint)
        outpath = output_dir / filename
        if outpath.exists() and not redownload_failed:
            checkpoint.mark_done(filename)
            if manifest:
                manifest.record_downloaded(filename, size_bytes=outpath.stat().st_size,
                                           source=BASE_URL)
            stats["skipped"] += 1
            continue

        if dry_run:
            print(f"  [{i}/{total}] Would download: {filename}")
            stats["downloaded"] += 1
            continue

        url = BASE_URL + filename
        print(f"  [{i}/{total}] {filename}...", end="", flush=True)

        try:
            resp = fetch_with_retry(
                session, url,
                max_retries=MAX_RETRIES,
                timeout=DOWNLOAD_TIMEOUT,
                verify=VERIFY_SSL,
            )
            outpath.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024

            checkpoint.mark_done(filename)
            if manifest:
                manifest.record_downloaded(filename, size_bytes=len(resp.content),
                                           source=BASE_URL)
            stats["downloaded"] += 1
            stats["bytes"] += len(resp.content)
            print(f" {size_kb:.1f} KB")

        except Exception as e:
            checkpoint.mark_failed(filename)
            stats["failed"] += 1
            print(f" FAILED: {e}")
            logger.warning(f"Failed to download {filename}: {e}")

        limiter.wait()

    # Final save
    checkpoint.save()
    return stats



#  VERIFICATION


def verify_downloads(output_dir: Path, remote_files: list[str]) -> dict:
    """Check downloaded files for completeness and basic integrity.

    Returns:
        {"ok": int, "missing": int, "empty": int, "total_remote": int}
    """
    result = {"ok": 0, "missing": 0, "empty": 0, "total_remote": len(remote_files)}

    for filename in remote_files:
        path = output_dir / filename
        if not path.exists():
            result["missing"] += 1
        elif path.stat().st_size == 0:
            result["empty"] += 1
        else:
            result["ok"] += 1

    return result


def write_manifest(output_dir: Path) -> Path:
    """Generate SHA256 manifest of all downloaded files."""
    manifest_path = output_dir / "MANIFEST.sha256"
    files = sorted(output_dir.glob("*.txt"))

    with open(manifest_path, "w", encoding="utf-8") as f:
        for fp in files:
            sha = hashlib.sha256(fp.read_bytes()).hexdigest()
            f.write(f"{sha}  {fp.name}\n")

    return manifest_path



#  STATUS DISPLAY


def show_status(output_dir: Path, checkpoint: CheckpointManager, remote_count: int = None):
    """Display current download status."""
    print_section("DOWNLOAD STATUS")

    local_files = sorted(output_dir.glob("*.txt"))
    n_local = len(local_files)
    total_size = sum(f.stat().st_size for f in local_files) / (1024 * 1024)

    print_summary_table([
        ("Files on disk:",       f"{n_local:,}"),
        ("Total size:",          f"{total_size:.1f} MB"),
        ("Checkpoint completed:", f"{checkpoint.completed_count:,}"),
        ("Checkpoint failed:",   f"{checkpoint.failed_count:,}"),
    ])

    if remote_count:
        remaining = remote_count - checkpoint.completed_count
        print()
        print_status_bar("Progress", checkpoint.completed_count, remote_count)
        if remaining > 0:
            print(f"  Remaining: {remaining:,} files")
    print()



#  PRINT SUMMARY


def print_download_summary(stats: dict, output_dir: Path):
    """Print summary after a download run."""
    print_section("DOWNLOAD SUMMARY")
    print_summary_table([
        ("Downloaded:",  f"{stats['downloaded']:,}"),
        ("Skipped:",     f"{stats['skipped']:,}"),
        ("Failed:",      f"{stats['failed']:,}"),
        ("Data fetched:", f"{stats['bytes'] / (1024*1024):.1f} MB"),
        ("Output:",      str(output_dir)),
    ])
    print()



#  INTERACTIVE MENU


def run_interactive(output_dir: Path):
    """Run the interactive menu loop."""
    print_header("MMT-9 Downloader", __version__,
                 "Download raw light curve files from MMT-9 archive")

    checkpoint = CheckpointManager("mmt9_download")
    manifest = DataManifest("mmt9_raw")
    manifest.save_metadata({"directory": str(output_dir)})
    limiter = RateLimiter(requests_per_minute=120)
    session = create_session("MMT9-Downloader/2.0")
    session.verify = VERIFY_SSL

    remote_files = None

    def _fetch_remote():
        """Fetch remote file list with error handling."""
        nonlocal remote_files
        if remote_files is not None:
            return True
        try:
            remote_files = get_remote_file_list(session)
            return True
        except Exception as e:
            print_error(f"Could not connect to MMT-9 server: {e}")
            print(f"          Check your internet connection. The MMT-9 server")
            print(f"          ({BASE_URL}) may also be temporarily down.")
            return False

    while True:
        options = [("download", "Download all files (resume if interrupted)")]
        options.append(("new", "Check for new files on server"))

        has_data = checkpoint.completed_count > 0 or list(output_dir.glob("*.txt"))
        if has_data:
            options.append(("status", "Check download status"))
            options.append(("verify", "Verify downloaded files"))
            options.append(("sha256", "Generate SHA256 manifest"))
            options.append(("cleanup", "Show files safe to delete (already processed)"))

        if checkpoint.failed_count > 0:
            options.append(("retry", f"Re-download {checkpoint.failed_count} failed files"))

        options.append(("quit", "Quit"))

        choice = prompt_choice("What would you like to do?", options)

        if choice is None or choice == "quit":
            print("\n  Goodbye!\n")
            break

        elif choice == "download":
            if not _fetch_remote():
                continue
            print()
            stats = download_files(
                session, remote_files, output_dir, checkpoint, limiter,
                manifest=manifest,
            )
            print_download_summary(stats, output_dir)

        elif choice == "new":
            if not _fetch_remote():
                continue
            known = manifest.all_filenames
            new_files = [f for f in remote_files if f not in known]
            if new_files:
                print(f"\n  {len(new_files)} new file(s) found on server:")
                for f in new_files[:20]:
                    print(f"    + {f}")
                if len(new_files) > 20:
                    print(f"    ... and {len(new_files) - 20} more")
                if prompt_confirm(f"\n  Download {len(new_files)} new file(s)?"):
                    stats = download_files(
                        session, new_files, output_dir, checkpoint, limiter,
                        manifest=manifest,
                    )
                    print_download_summary(stats, output_dir)
            else:
                print(f"\n  No new files - server has {len(remote_files)} files, "
                      f"all {manifest.total_files} are tracked.\n")

        elif choice == "retry":
            if not _fetch_remote():
                continue
            failed = [f for f in remote_files if checkpoint.is_failed(f)]
            print(f"\n  Retrying {len(failed)} failed file(s)...\n")
            stats = download_files(
                session, failed, output_dir, checkpoint, limiter,
                manifest=manifest, redownload_failed=True,
            )
            print_download_summary(stats, output_dir)

        elif choice == "status":
            if not _fetch_remote():
                show_status(output_dir, checkpoint, remote_count=None)
            else:
                show_status(output_dir, checkpoint, remote_count=len(remote_files))
            # Also show manifest info
            if manifest.total_files > 0:
                manifest.print_status()
                print()

        elif choice == "verify":
            if not _fetch_remote():
                continue
            result = verify_downloads(output_dir, remote_files)
            print_section("VERIFICATION")
            print_summary_table([
                ("Remote files:",  f"{result['total_remote']:,}"),
                ("OK on disk:",    f"{result['ok']:,}"),
                ("Missing:",       f"{result['missing']:,}"),
                ("Empty (0 bytes):", f"{result['empty']:,}"),
            ])
            print()

        elif choice == "sha256":
            print(f"\n  Generating SHA256 manifest...", end="", flush=True)
            path = write_manifest(output_dir)
            n = len(list(output_dir.glob("*.txt")))
            print(f" done ({n} files)")
            print(f"  Saved to: {path}\n")

        elif choice == "cleanup":
            deletable = manifest.find_deletable(output_dir, "*.txt")
            if deletable:
                total_mb = sum(f.stat().st_size for f in deletable) / (1024 * 1024)
                print(f"\n  {len(deletable)} file(s) ({total_mb:.1f} MB) can be safely deleted.")
                print()
                print(f"  WHY THESE ARE SAFE TO DELETE:")
                print(f"  These raw .txt dump files have already been processed by")
                print(f"  mmt9_processor.py into individual per-satellite CSV files")
                print(f"  in data/processed/mmt9/lightcurves/. All observation data")
                print(f"  has been extracted and catalogued. The raw files are only")
                print(f"  needed if you want to reprocess from scratch.")
                print()
                print(f"  The manifest in data/checkpoints/ will remember these files")
                print(f"  existed, so the 'check for new files' feature still works.")
                if prompt_confirm(f"\n  Delete {len(deletable)} processed raw file(s)?"):
                    for f in deletable:
                        f.unlink()
                    print(f"  Deleted {len(deletable)} files, freed {total_mb:.1f} MB.\n")
                else:
                    print(f"  No files deleted.\n")
            else:
                print(f"\n  No files are safe to delete yet.")
                print(f"  Run 2_process/mmt9_processor.py first to process raw files.\n")

    session.close()



#  CLI MODE


def run_cli(args, output_dir: Path):
    """Run in CLI mode based on parsed arguments."""
    print_header("MMT-9 Downloader", __version__)

    checkpoint = CheckpointManager("mmt9_download")
    limiter = RateLimiter(requests_per_minute=120)
    session = create_session("MMT9-Downloader/2.0")
    session.verify = VERIFY_SSL

    try:
        if args.status:
            remote_count = None
            try:
                resp = session.get(BASE_URL, timeout=10, verify=VERIFY_SSL)
                if resp.ok:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")
                    remote_count = sum(1 for a in soup.find_all("a", href=True)
                                       if a["href"].endswith(".txt"))
            except Exception:
                print_warning("Could not reach MMT-9 server - showing local status only")
            show_status(output_dir, checkpoint, remote_count=remote_count)

        elif args.verify:
            try:
                remote_files = get_remote_file_list(session)
            except Exception as e:
                print_error(f"Could not connect to MMT-9 server: {e}")
                sys.exit(1)
            result = verify_downloads(output_dir, remote_files)
            print_section("VERIFICATION")
            print_summary_table([
                ("Remote files:",    f"{result['total_remote']:,}"),
                ("OK on disk:",      f"{result['ok']:,}"),
                ("Missing:",         f"{result['missing']:,}"),
                ("Empty (0 bytes):", f"{result['empty']:,}"),
            ])
            print()
            sys.exit(0 if result["missing"] == 0 else 1)

        elif args.download:
            try:
                remote_files = get_remote_file_list(session)
            except Exception as e:
                print_error(f"Could not connect to MMT-9 server: {e}")
                sys.exit(1)
            stats = download_files(
                session, remote_files, output_dir, checkpoint, limiter,
                dry_run=args.dry_run,
            )
            print_download_summary(stats, output_dir)
            sys.exit(0 if stats["failed"] == 0 else 1)

    finally:
        session.close()



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="MMT-9 Light Curve Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mmt9_downloader.py                # Interactive menu
  python mmt9_downloader.py --download     # Download all (resume-aware)
  python mmt9_downloader.py --status       # Check progress
  python mmt9_downloader.py --verify       # Verify file integrity
  python mmt9_downloader.py --download --dry-run  # Preview only
        """,
    )
    parser.add_argument("--download", action="store_true", help="Download all files")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded files")
    parser.add_argument("--dry-run", action="store_true", help="Preview downloads without fetching")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    ensure_dirs()
    output_dir = Path(args.output) if args.output else RAW_MMT9

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
