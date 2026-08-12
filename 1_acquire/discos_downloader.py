"""
ESA DISCOS Metadata Downloader
================================
Downloads the complete ESA DISCOS satellite catalogue via the v2 API,
then optionally cross-matches with light curve catalogues.

Credentials are loaded from config/credentials.json (DISCOS token).

Interactive menu (no arguments):
    python discos_downloader.py

CLI mode:
    python discos_downloader.py --download
    python discos_downloader.py --download --dry-run
    python discos_downloader.py --status
    python discos_downloader.py --crossmatch ./data/processed/mmt9/catalogue.csv
"""

__version__ = "2.0.0"

import sys
import csv
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import RAW_DISCOS, PROC_DISCOS, PROC_MMT9, PROC_SDLCD, ensure_dirs
from common.menu import (
    print_header, prompt_choice, prompt_confirm, prompt_path,
    print_section, print_status_bar, print_summary_table,
    print_warning, print_error,
)
from common.progress import CheckpointManager
from common.network import create_session, load_credentials, RateLimiter
from common.logging_setup import setup_logging


#  REMOTE ENDPOINTS - edit these if URLs change


BASE_URL = "https://discosweb.esoc.esa.int/api"


#  API SETTINGS


API_VERSION     = "2"
PAGE_SIZE       = 100     # Max objects per request (API cap)
RATE_LIMIT_RPM  = 17      # Requests per minute (API limit is 20, stay under)
MAX_RETRIES     = 3


#  FIELD DEFINITIONS


DISCOS_FIELDS = [
    "discos_id", "norad_id", "cospar_id", "name", "object_class",
    "mass_kg", "shape", "length_m", "height_m", "depth_m",
    "xsect_min_m2", "xsect_max_m2", "xsect_avg_m2",
    "launch_date", "reentry_epoch", "country",
]


#  SETUP


logger = setup_logging("discos_downloader")



#  API CLIENT


def create_discos_session() -> tuple:
    """Create an authenticated session for the DISCOS API.

    Returns:
        (session, token) tuple.
    """
    creds = load_credentials("discos")
    token = creds["token"]

    session = create_session("DISCOS-Downloader/2.0", extra_headers={
        "DiscosWeb-Api-Version": API_VERSION,
        "Accept": "application/vnd.api+json",
        "Authorization": f"Bearer {token}",
    })

    return session, token


def test_connection(session) -> int:
    """Test API connection and return total object count."""
    from common.network import fetch_with_retry

    url = f"{BASE_URL}/objects"
    params = {"page[number]": 1, "page[size]": 1}

    resp = fetch_with_retry(session, url, params=params, max_retries=2, timeout=30)
    data = resp.json()
    total = data.get("meta", {}).get("totalObjects", 0)
    return total


def fetch_page(session, page_num: int) -> dict:
    """Fetch a single page of objects from the DISCOS API."""
    from common.network import fetch_with_retry

    url = f"{BASE_URL}/objects"
    params = {
        "page[number]": page_num,
        "page[size]": PAGE_SIZE,
    }

    resp = fetch_with_retry(
        session, url,
        params=params,
        max_retries=MAX_RETRIES,
        timeout=60,
        rate_limit_delay=60.0,
    )
    return resp.json()


def parse_object(item: dict) -> dict:
    """Extract relevant fields from a JSON:API response object."""
    attrs = item.get("attributes", {})
    return {
        "discos_id": item.get("id"),
        "norad_id": attrs.get("satno"),
        "cospar_id": attrs.get("cosparId"),
        "name": attrs.get("name"),
        "object_class": attrs.get("objectClass"),
        "mass_kg": attrs.get("mass"),
        "shape": attrs.get("shape"),
        "length_m": attrs.get("length"),
        "height_m": attrs.get("height"),
        "depth_m": attrs.get("depth"),
        "xsect_min_m2": attrs.get("xSectMin"),
        "xsect_max_m2": attrs.get("xSectMax"),
        "xsect_avg_m2": attrs.get("xSectAvg"),
        "launch_date": attrs.get("launchDate"),
        "reentry_epoch": attrs.get("reentryEpoch"),
        "country": attrs.get("country"),
    }



#  INCREMENTAL CSV WRITER

#
#  Instead of holding all objects in memory and writing once,
#  we append each page to CSV as we go. The checkpoint stores
#  only the page number, not the data.
#

def init_csv(output_path: Path) -> None:
    """Write CSV header if the file doesn't exist yet."""
    if not output_path.exists():
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DISCOS_FIELDS)
            writer.writeheader()


def append_to_csv(output_path: Path, objects: list[dict]) -> None:
    """Append a batch of parsed objects to the CSV."""
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DISCOS_FIELDS)
        writer.writerows(objects)



#  DOWNLOAD ENGINE


def download_catalogue(
    session,
    output_path: Path,
    checkpoint: CheckpointManager,
    limiter: RateLimiter,
    dry_run: bool = False,
) -> dict:
    """Page through the entire DISCOS catalogue with incremental saves.

    Checkpoint stores the last completed page number, so resume
    only re-fetches from the next page (no data duplication).

    Returns:
        {"pages": int, "objects": int}
    """
    # Determine resume point from checkpoint
    start_page = int(checkpoint.metadata.get("last_page", 0)) + 1
    total_objects = int(checkpoint.metadata.get("total_objects_so_far", 0))

    if start_page > 1:
        logger.info(f"Resuming from page {start_page} ({total_objects:,} objects so far)")
        print(f"  Resuming from page {start_page} ({total_objects:,} objects already saved)")
    else:
        # Fresh start - initialise CSV with header
        init_csv(output_path)

    stats = {"pages": 0, "objects": total_objects}
    page_num = start_page

    while True:
        if dry_run:
            print(f"  Page {page_num}: would fetch {PAGE_SIZE} objects")
            stats["pages"] += 1
            if stats["pages"] >= 5:
                print(f"  ... (dry-run limited to 5 pages)")
                break
            page_num += 1
            continue

        print(f"  Page {page_num}...", end="", flush=True)

        try:
            data = fetch_page(session, page_num)
        except Exception as e:
            logger.error(f"Failed on page {page_num}: {e}")
            print(f" FAILED: {e}")
            print(f"  Progress saved. Re-run to resume from page {page_num}.")
            checkpoint.save_metadata({
                "last_page": page_num - 1,
                "total_objects_so_far": stats["objects"],
            })
            break

        items = data.get("data", [])
        if not items:
            print(" no data returned, stopping.")
            break

        # Parse and append to CSV
        parsed = [parse_object(item) for item in items]
        append_to_csv(output_path, parsed)

        stats["objects"] += len(parsed)
        stats["pages"] += 1

        catalogue_total = data.get("meta", {}).get("totalObjects", "?")
        print(f" {len(parsed)} objects (total: {stats['objects']:,} / {catalogue_total})")

        # Update checkpoint (just page number + count, not data)
        checkpoint.save_metadata({
            "last_page": page_num,
            "total_objects_so_far": stats["objects"],
        })

        # Check for next page
        next_link = data.get("links", {}).get("next")
        if not next_link:
            print("\n  Reached last page.")
            break

        page_num += 1
        limiter.wait()

    return stats



#  POST-PROCESSING


def compute_am_ratio(csv_path: Path) -> None:
    """Add area-to-mass ratio column to the downloaded catalogue."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    df["am_ratio_avg"] = None
    mask = (df["mass_kg"].notna()) & (df["mass_kg"] > 0) & (df["xsect_avg_m2"].notna())
    df.loc[mask, "am_ratio_avg"] = df.loc[mask, "xsect_avg_m2"] / df.loc[mask, "mass_kg"]

    df.to_csv(csv_path, index=False)

    print_summary_table([
        ("Total objects:",       f"{len(df):,}"),
        ("Payloads:",            f"{(df['object_class'] == 'Payload').sum():,}"),
        ("Rocket Bodies:",       f"{(df['object_class'] == 'Rocket Body').sum():,}"),
        ("Debris:",              f"{(df['object_class'] == 'Debris').sum():,}"),
        ("With mass:",           f"{df['mass_kg'].notna().sum():,}"),
        ("With cross-section:",  f"{df['xsect_avg_m2'].notna().sum():,}"),
        ("With computable A/m:", f"{mask.sum():,}"),
    ])



#  CROSS-MATCH


def cross_match(discos_path: Path, catalogue_path: Path, output_dir: Path) -> None:
    """Cross-match DISCOS with a light curve catalogue on NORAD ID."""
    import pandas as pd

    print(f"\n  Cross-matching with: {catalogue_path.name}")

    discos = pd.read_csv(discos_path)
    cat = pd.read_csv(catalogue_path)

    if "norad_id" not in cat.columns:
        print_error("Catalogue must have a 'norad_id' column.")
        return

    discos["norad_id"] = pd.to_numeric(discos["norad_id"], errors="coerce").astype("Int64")
    cat["norad_id"] = pd.to_numeric(cat["norad_id"], errors="coerce").astype("Int64")

    merged = cat.merge(discos, on="norad_id", how="left", suffixes=("", "_discos"))

    matched = merged["discos_id"].notna().sum()
    total = len(merged)

    print_summary_table([
        ("Catalogue satellites:", f"{total:,}"),
        ("Matched in DISCOS:",   f"{matched:,} ({100*matched/total:.1f}%)"),
        ("With mass data:",      f"{merged['mass_kg'].notna().sum():,}"),
        ("With cross-section:",  f"{merged['xsect_avg_m2'].notna().sum():,}"),
        ("With A/m ratio:",      f"{merged['am_ratio_avg'].notna().sum():,}"),
    ])

    cat_name = catalogue_path.stem.replace("catalogue", "").strip("_")
    if not cat_name:
        cat_name = catalogue_path.parent.name
    out_name = f"{cat_name}_discos_merged.csv"
    out_path = output_dir / out_name

    merged.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")

    # Save unmatched IDs
    unmatched = merged[merged["discos_id"].isna()]["norad_id"]
    if len(unmatched) > 0:
        un_path = output_dir / f"{cat_name}_unmatched_ids.txt"
        unmatched.to_csv(un_path, index=False, header=False)
        print(f"  {len(unmatched):,} unmatched NORAD IDs saved to {un_path.name}")



#  STATUS


def show_status(output_path: Path, checkpoint: CheckpointManager):
    """Show download status."""
    print_section("DISCOS STATUS")

    if output_path.exists():
        # Count lines (minus header)
        with open(output_path) as f:
            n_lines = sum(1 for _ in f) - 1
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print_summary_table([
            ("CSV file:",     str(output_path)),
            ("Objects:",      f"{n_lines:,}"),
            ("File size:",    f"{size_mb:.1f} MB"),
        ])
    else:
        print(f"  No catalogue downloaded yet.")

    last_page = checkpoint.metadata.get("last_page", 0)
    if last_page:
        print(f"  Last completed page: {last_page}")
        print(f"  Objects so far: {checkpoint.metadata.get('total_objects_so_far', '?'):,}")
    print()



#  INTERACTIVE MENU


def run_interactive(raw_dir: Path, proc_dir: Path):
    """Run the interactive menu."""
    print_header("DISCOS Downloader", __version__,
                 "Download ESA DISCOS satellite catalogue")

    checkpoint = CheckpointManager("discos_download")
    limiter = RateLimiter(requests_per_minute=RATE_LIMIT_RPM)
    output_csv = raw_dir / "discos_catalogue.csv"

    session = None  # Lazy (only connect when needed)

    while True:
        options = [("download", "Download full DISCOS catalogue")]

        has_data = output_csv.exists()
        if has_data:
            options.append(("status", "View download status"))
            options.append(("crossmatch", "Cross-match with a light curve catalogue"))

        if checkpoint.metadata.get("last_page", 0) > 0 and not has_data:
            options[0] = ("download", "Resume catalogue download")

        options.append(("quit", "Quit"))

        choice = prompt_choice("What would you like to do?", options)

        if choice is None or choice == "quit":
            print("\n  Goodbye!\n")
            break

        elif choice == "download":
            if session is None:
                try:
                    session, _ = create_discos_session()
                    print(f"  Testing API connection...", end="", flush=True)
                    total = test_connection(session)
                    print(f" OK ({total:,} objects in catalogue)\n")
                except Exception as e:
                    print(f" FAILED: {e}")
                    print_error("Check your DISCOS token in config/credentials.json")
                    continue

            print_section("DOWNLOADING CATALOGUE")
            stats = download_catalogue(session, output_csv, checkpoint, limiter)

            if stats["objects"] > 0 and not False:  # not dry_run
                print(f"\n  Computing A/m ratios...\n")
                compute_am_ratio(output_csv)

                # Copy to processed dir
                import shutil
                proc_csv = proc_dir / "discos_catalogue.csv"
                shutil.copy2(output_csv, proc_csv)
                print(f"\n  Also copied to: {proc_csv}")

                # Clear checkpoint on success
                checkpoint.clear()

            print()

        elif choice == "status":
            show_status(output_csv, checkpoint)

        elif choice == "crossmatch":
            # Auto-detect available catalogues
            available = []
            mmt9_cat = PROC_MMT9 / "catalogue.csv"
            sdlcd_cat = PROC_SDLCD / "catalogue.csv"
            if mmt9_cat.exists():
                available.append(("mmt9", f"MMT-9 catalogue ({mmt9_cat})"))
            if sdlcd_cat.exists():
                available.append(("sdlcd", f"SDLCD catalogue ({sdlcd_cat})"))
            available.append(("custom", "Enter custom catalogue path"))
            available.append(("back", "Back"))

            cm_choice = prompt_choice("Select catalogue to cross-match:", available)

            if cm_choice == "mmt9":
                cross_match(output_csv, mmt9_cat, proc_dir)
            elif cm_choice == "sdlcd":
                cross_match(output_csv, sdlcd_cat, proc_dir)
            elif cm_choice == "custom":
                custom = prompt_path("Path to catalogue CSV", must_exist=True, is_dir=False)
                if custom:
                    cross_match(output_csv, custom, proc_dir)
            print()

    if session:
        session.close()



#  CLI MODE


def run_cli(args, raw_dir: Path, proc_dir: Path):
    """Run in CLI mode."""
    print_header("DISCOS Downloader", __version__)

    checkpoint = CheckpointManager("discos_download")
    limiter = RateLimiter(requests_per_minute=RATE_LIMIT_RPM)
    output_csv = raw_dir / "discos_catalogue.csv"

    if args.status:
        show_status(output_csv, checkpoint)
        return

    if args.crossmatch:
        cat_path = Path(args.crossmatch)
        if not cat_path.exists():
            print_error(f"Catalogue not found: {cat_path}")
            sys.exit(1)
        if not output_csv.exists():
            print_error("DISCOS catalogue not downloaded yet. Run --download first.")
            sys.exit(1)
        cross_match(output_csv, cat_path, proc_dir)
        return

    if args.download:
        session, _ = create_discos_session()

        try:
            print(f"  Testing API connection...", end="", flush=True)
            total = test_connection(session)
            print(f" OK ({total:,} objects)\n")

            stats = download_catalogue(
                session, output_csv, checkpoint, limiter, dry_run=args.dry_run,
            )

            if stats["objects"] > 0 and not args.dry_run:
                print(f"\n  Computing A/m ratios...\n")
                compute_am_ratio(output_csv)

                import shutil
                proc_csv = proc_dir / "discos_catalogue.csv"
                shutil.copy2(output_csv, proc_csv)
                print(f"\n  Also copied to: {proc_csv}")

                checkpoint.clear()
            print()

        except Exception as e:
            print_error(str(e))
            sys.exit(1)
        finally:
            session.close()



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="ESA DISCOS Metadata Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python discos_downloader.py                     # Interactive menu
  python discos_downloader.py --download          # Download full catalogue
  python discos_downloader.py --status            # Check progress
  python discos_downloader.py --crossmatch catalogue.csv  # Cross-match
  python discos_downloader.py --download --dry-run  # Preview only
        """,
    )
    parser.add_argument("--download", action="store_true", help="Download full catalogue")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--crossmatch", type=str, default=None,
                        help="Cross-match with a catalogue CSV")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--output", type=str, default=None,
                        help="Override raw output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    ensure_dirs()
    raw_dir = Path(args.output) if args.output else RAW_DISCOS
    proc_dir = PROC_DISCOS

    has_action = args.download or args.status or args.crossmatch

    try:
        if has_action:
            run_cli(args, raw_dir, proc_dir)
        else:
            run_interactive(raw_dir, proc_dir)
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Progress saved.")
        sys.exit(2)


if __name__ == "__main__":
    main()
