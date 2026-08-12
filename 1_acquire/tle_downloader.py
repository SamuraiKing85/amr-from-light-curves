"""
Space-Track TLE API Downloader
=================================
Downloads GP (General Perturbations) history from the Space-Track API
for satellites listed in a catalogue CSV. Used as a gap-filler when
bulk TLE files don't cover all satellites.

For the full TLE pipeline (scan + bulk + API), use 2_process/tle_manager.py.
This script is a standalone API-only tool.

Credentials are loaded from config/credentials.json (Space-Track).

Interactive menu (no arguments):
    python tle_downloader.py

CLI mode:
    python tle_downloader.py --download --catalogue catalogue.csv
    python tle_downloader.py --download --norad-ids 25544 43013 48274
    python tle_downloader.py --status
"""

__version__ = "2.0.0"

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import PROC_TLE, PROC_MMT9, PROC_SDLCD, ensure_dirs, find_active_catalogues
from common.menu import (
    print_header, prompt_choice, prompt_confirm, prompt_path, prompt_input,
    print_section, print_status_bar, print_summary_table,
    print_warning, print_error,
)
from common.progress import CheckpointManager
from common.network import create_session, load_credentials, RateLimiter
from common.logging_setup import setup_logging


#  REMOTE ENDPOINTS - edit these if URLs change


LOGIN_URL      = "https://www.space-track.org/ajaxauth/login"
GP_HISTORY_URL = "https://www.space-track.org/basicspacedata/query/class/gp_history"


#  API SETTINGS


RATE_LIMIT_RPM  = 25      # Requests per minute (Space-Track limit ~30)
TIME_CHUNK_DAYS = 730     # 2-year chunks per request
PADDING_DAYS    = 365     # 1 year padding around observation window
MAX_RETRIES     = 3
REQUEST_TIMEOUT = 180     # Some queries take a while


#  GP RECORD FIELDS


GP_FIELDS = [
    "NORAD_CAT_ID", "OBJECT_NAME", "EPOCH", "MEAN_MOTION",
    "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER",
    "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS",
    "REV_AT_EPOCH", "ELEMENT_SET_NO", "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE", "OBJECT_ID", "OBJECT_TYPE",
    "RCS_SIZE", "COUNTRY_CODE", "LAUNCH_DATE", "SITE", "DECAY_DATE",
]


#  SETUP


logger = setup_logging("tle_downloader")



#  SPACE-TRACK CLIENT


class SpaceTrackClient:
    """Authenticated Space-Track API client with session management."""

    def __init__(self):
        self.session = create_session("TLE-Downloader/2.0")
        self.logged_in = False
        creds = load_credentials("spacetrack")
        self._username = creds["username"]
        self._password = creds["password"]

    def login(self) -> None:
        """Authenticate with Space-Track."""
        logger.info("Logging in to Space-Track...")
        resp = self.session.post(LOGIN_URL, data={
            "identity": self._username,
            "password": self._password,
        }, timeout=30)

        if resp.status_code != 200 or "Failed" in resp.text:
            print_error(f"Login failed (HTTP {resp.status_code})")
            raise ConnectionError("Space-Track authentication failed")

        self.logged_in = True
        logger.info("Login successful")

    def _relogin(self, session=None):
        """Callback for fetch_with_retry on 401."""
        self.login()

    def fetch_gp_history(self, norad_id: int, date_start: str,
                         date_end: str) -> list[dict]:
        """Fetch GP history for a single satellite and date range."""
        if not self.logged_in:
            self.login()

        url = (
            f"{GP_HISTORY_URL}"
            f"/NORAD_CAT_ID/{norad_id}"
            f"/EPOCH/{date_start}--{date_end}"
            f"/orderby/EPOCH asc"
            f"/format/json"
        )

        from common.network import fetch_with_retry

        try:
            resp = fetch_with_retry(
                self.session, url,
                max_retries=MAX_RETRIES,
                timeout=REQUEST_TIMEOUT,
                on_401=self._relogin,
            )
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Failed to fetch NORAD {norad_id} ({date_start}–{date_end}): {e}")
            return []

    def close(self):
        self.session.close()



#  CATALOGUE LOADING


def load_catalogue(cat_path: Path) -> list[dict]:
    """Load a catalogue CSV and compute TLE query date ranges.

    Returns:
        List of {"norad_id": int, "tle_start": str, "tle_end": str}
    """
    import pandas as pd

    df = pd.read_csv(cat_path)
    df["norad_id"] = pd.to_numeric(df["norad_id"], errors="coerce")
    df = df.dropna(subset=["norad_id"])

    df["first_obs"] = pd.to_datetime(df["first_obs"], format="mixed", errors="coerce")
    df["last_obs"] = pd.to_datetime(df["last_obs"], format="mixed", errors="coerce")

    df["tle_start"] = df["first_obs"] - timedelta(days=PADDING_DAYS)
    df["tle_end"] = df["last_obs"] + timedelta(days=PADDING_DAYS)

    earliest = pd.Timestamp("2000-01-01")
    latest = pd.Timestamp(datetime.utcnow())
    df["tle_start"] = df["tle_start"].clip(lower=earliest)
    df["tle_end"] = df["tle_end"].clip(upper=latest)

    sats = []
    for _, row in df.iterrows():
        if pd.notna(row["tle_start"]) and pd.notna(row["tle_end"]):
            sats.append({
                "norad_id": int(row["norad_id"]),
                "tle_start": row["tle_start"].strftime("%Y-%m-%d"),
                "tle_end": row["tle_end"].strftime("%Y-%m-%d"),
            })

    logger.info(f"Loaded {len(sats)} satellites from {cat_path.name}")
    return sats


def build_norad_list(norad_ids: list[int]) -> list[dict]:
    """Build satellite list from explicit NORAD IDs (full date range)."""
    start = "2000-01-01"
    end = datetime.utcnow().strftime("%Y-%m-%d")
    return [{"norad_id": nid, "tle_start": start, "tle_end": end}
            for nid in norad_ids]



#  DOWNLOAD ENGINE


def download_tles(
    client: SpaceTrackClient,
    satellites: list[dict],
    output_dir: Path,
    checkpoint: CheckpointManager,
    limiter: RateLimiter,
    dry_run: bool = False,
) -> dict:
    """Download GP history for all satellites with time chunking.

    Returns:
        {"satellites": int, "records": int, "skipped": int, "failed": int}
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"satellites": 0, "records": 0, "skipped": 0, "failed": 0}
    total = len(satellites)

    for i, sat in enumerate(satellites, 1):
        norad_id = sat["norad_id"]
        nid_str = str(norad_id)

        if checkpoint.is_done(nid_str):
            stats["skipped"] += 1
            continue

        if dry_run:
            print(f"  [{i}/{total}] NORAD {norad_id} - would download "
                  f"{sat['tle_start']} to {sat['tle_end']}")
            stats["satellites"] += 1
            continue

        # Time chunking: split into TIME_CHUNK_DAYS segments
        start = datetime.strptime(sat["tle_start"], "%Y-%m-%d")
        end = datetime.strptime(sat["tle_end"], "%Y-%m-%d")
        sat_records = 0

        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=TIME_CHUNK_DAYS), end)
            start_str = chunk_start.strftime("%Y-%m-%d")
            end_str = chunk_end.strftime("%Y-%m-%d")

            print(f"  [{i}/{total}] NORAD {norad_id:>6d} "
                  f"{start_str} to {end_str}...", end="", flush=True)

            records = client.fetch_gp_history(norad_id, start_str, end_str)

            if not records:
                print(f" 0 records")
            else:
                # Extract relevant fields
                recs = [{k: r.get(k) for k in GP_FIELDS} for r in records]
                sat_path = output_dir / f"{norad_id}.csv"

                df = pd.DataFrame(recs)
                if sat_path.exists():
                    existing = pd.read_csv(sat_path, dtype=str)
                    df = pd.concat([existing, df], ignore_index=True)
                    df.drop_duplicates(subset=["EPOCH"], keep="last", inplace=True)

                df.to_csv(sat_path, index=False)
                sat_records += len(recs)
                stats["records"] += len(recs)
                print(f" {len(recs):,} records [sat total: {sat_records:,}]")

            limiter.wait()
            chunk_start = chunk_end

        checkpoint.mark_done(nid_str)
        stats["satellites"] += 1

    checkpoint.save()
    return stats



#  STATUS


def show_status(output_dir: Path, checkpoint: CheckpointManager):
    """Show download status."""
    print_section("TLE DOWNLOAD STATUS")

    tle_dir = output_dir / "tle_histories"
    n_files = len(list(tle_dir.glob("*.csv"))) if tle_dir.exists() else 0

    print_summary_table([
        ("TLE files on disk:",    f"{n_files:,}"),
        ("Satellites completed:", f"{checkpoint.completed_count:,}"),
        ("Satellites failed:",   f"{checkpoint.failed_count:,}"),
    ])
    print()



#  INTERACTIVE MENU


def run_interactive(output_dir: Path):
    """Run the interactive menu."""
    print_header("TLE API Downloader", __version__,
                 "Download GP history from Space-Track API")

    checkpoint = CheckpointManager("tle_api_download")
    limiter = RateLimiter(requests_per_minute=RATE_LIMIT_RPM)
    tle_dir = output_dir / "tle_histories"
    client = None

    satellites = None

    while True:
        options = [
            ("catalogue", "Load catalogue and download TLEs"),
            ("norad",     "Download TLEs for specific NORAD IDs"),
            ("status",    "View download status"),
        ]

        if checkpoint.completed_count > 0:
            options.append(("reset", "Clear checkpoint (start fresh)"))

        options.append(("quit", "Quit"))

        choice = prompt_choice("What would you like to do?", options)

        if choice is None or choice == "quit":
            print("\n  Goodbye!\n")
            break

        elif choice == "catalogue":
            # Auto-detect catalogues using shared helper
            cat_options = []
            active_cats = find_active_catalogues()
            
            for i, cat_path in enumerate(active_cats):
                # Dynamically get the source name from the parent folder (e.g., 'mmt9', 'sdlcd')
                source_name = cat_path.parent.name.upper()
                cat_options.append((str(i), f"{source_name} ({cat_path})"))
            
            cat_options.append(("custom", "Enter custom path"))
            cat_options.append(("back", "Back"))

            cat_choice = prompt_choice("Select catalogue:", cat_options)
            
            if cat_choice == "custom":
                path = prompt_path("Catalogue CSV path", must_exist=True, is_dir=False)
                if path:
                    satellites = load_catalogue(path)
            elif cat_choice == "back":
                continue
            elif cat_choice is not None:
                # The choice matches the index of the active_cats list
                selected_path = active_cats[int(cat_choice)]
                satellites = load_catalogue(selected_path)

        elif choice == "norad":
            raw = prompt_input("Enter NORAD IDs (comma or space separated)")
            if raw:
                try:
                    ids = [int(x.strip()) for x in raw.replace(",", " ").split()
                           if x.strip().isdigit()]
                except ValueError:
                    print_error("Invalid NORAD IDs")
                    continue

                if ids:
                    satellites = build_norad_list(ids)
                    print(f"\n  Downloading TLEs for {len(ids)} satellite(s)")

                    if client is None:
                        client = SpaceTrackClient()
                        client.login()
                        print()

                    stats = download_tles(
                        client, satellites, tle_dir, checkpoint, limiter,
                    )
                    print_section("DOWNLOAD SUMMARY")
                    print_summary_table([
                        ("Satellites processed:", f"{stats['satellites']:,}"),
                        ("Records downloaded:",   f"{stats['records']:,}"),
                    ])
                    print()

        elif choice == "status":
            show_status(output_dir, checkpoint)

        elif choice == "reset":
            if prompt_confirm("Clear download checkpoint? (files on disk are kept)"):
                checkpoint.clear()
                print("  Checkpoint cleared.\n")

    if client:
        client.close()



#  CLI MODE


def run_cli(args, output_dir: Path):
    """Run in CLI mode."""
    print_header("TLE API Downloader", __version__)

    checkpoint = CheckpointManager("tle_api_download")
    limiter = RateLimiter(requests_per_minute=RATE_LIMIT_RPM)
    tle_dir = output_dir / "tle_histories"

    if args.status:
        show_status(output_dir, checkpoint)
        return

    if args.download:
        # Build satellite list
        if args.norad_ids:
            satellites = build_norad_list(args.norad_ids)
        elif args.catalogue:
            cat_path = Path(args.catalogue)
            if not cat_path.exists():
                print_error(f"Catalogue not found: {cat_path}")
                sys.exit(1)
            satellites = load_catalogue(cat_path)
        else:
            print_error("Need --catalogue or --norad-ids with --download")
            sys.exit(1)

        remaining = [s for s in satellites
                     if not checkpoint.is_done(str(s["norad_id"]))]
        print(f"  Satellites: {len(satellites)}, remaining: {len(remaining)}\n")

        client = SpaceTrackClient()
        client.login()
        print()

        try:
            stats = download_tles(
                client, satellites, tle_dir, checkpoint, limiter,
                dry_run=args.dry_run,
            )
            print_section("SUMMARY")
            print_summary_table([
                ("Satellites processed:", f"{stats['satellites']:,}"),
                ("Records downloaded:",   f"{stats['records']:,}"),
                ("Skipped (done):",       f"{stats['skipped']:,}"),
            ])
            print()
            sys.exit(0 if stats["failed"] == 0 else 1)
        finally:
            client.close()



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="Space-Track TLE API Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tle_downloader.py                                    # Interactive
  python tle_downloader.py --download --catalogue cat.csv     # From catalogue
  python tle_downloader.py --download --norad-ids 25544 43013 # Specific IDs
  python tle_downloader.py --status                           # Check progress
  python tle_downloader.py --download --catalogue cat.csv --dry-run
        """,
    )
    parser.add_argument("--download", action="store_true", help="Start downloading")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--catalogue", type=str, default=None,
                        help="Catalogue CSV with norad_id, first_obs, last_obs")
    parser.add_argument("--norad-ids", type=int, nargs="+", default=None,
                        help="Specific NORAD IDs to download")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    ensure_dirs()
    output_dir = Path(args.output) if args.output else PROC_TLE

    has_action = args.download or args.status

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
