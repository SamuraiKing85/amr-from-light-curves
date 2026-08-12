"""
Unified TLE Manager
====================
A single script to manage TLE data for ALL light curve datasets.
Automatically detects active catalogues and processes missing TLE data
using bulk historical files and the Space-Track API.

Three-stage pipeline:
  Stage 1 - SCAN:    Check which NORAD IDs already have adequate TLE data
  Stage 2 - BULK:    Process yearly bulk TLE files for any missing satellites
  Stage 3 - API:     Download remaining gaps from Space-Track API

Usage:
    # Interactive menu (Auto-detects everything)
    python tle_manager.py

    # Full pipeline with CLI overrides
    python tle_manager.py --api-only
    python tle_manager.py --no-api
"""

__version__ = "2.1.0"

import argparse
import json
import math
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import RAW_TLE_BULK, PROC_TLE, ensure_dirs, find_active_catalogues
from common.menu import (
    print_header as _print_header,
    prompt_choice as _prompt_choice,
    prompt_confirm,
    print_section, print_summary_table, print_status_bar,
    print_warning, print_error,
)
from common.logging_setup import setup_logging

_logger = setup_logging("tle_manager")


#  CONSTANTS & CONFIGURATION


MU = 398600.4418
RE = 6378.135
TWO_PI = 2.0 * math.pi
PADDING_DAYS = 365
TIME_CHUNK_DAYS = 730
RATE_LIMIT_DELAY = 12.5

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
GP_HISTORY_URL = "https://www.space-track.org/basicspacedata/query/class/gp_history"

TLE_BULK_HELP = """
  ┌────────────────────────────────────────────────────────┐
  │  HOW TO GET BULK TLE FILES                             │
  ├────────────────────────────────────────────────────────┤
  │  1. Log in to https://www.space-track.org              │
  │  2. Go to: Signed-In > Data Cloud                      │
  │  3. Download yearly TLE .txt files (e.g., tle2022.txt) │
  │  4. Place them in: {tle_dir}                           │
  └────────────────────────────────────────────────────────┘
"""

GP_FIELDS = [
    "NORAD_CAT_ID", "OBJECT_NAME", "EPOCH", "MEAN_MOTION",
    "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER",
    "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS",
    "REV_AT_EPOCH", "ELEMENT_SET_NO", "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE", "OBJECT_ID", "OBJECT_TYPE",
    "RCS_SIZE", "COUNTRY_CODE", "LAUNCH_DATE", "SITE", "DECAY_DATE",
]


#  CATALOGUE LOADING


def load_catalogue(cat_path):
    df = pd.read_csv(cat_path)
    df["norad_id"] = pd.to_numeric(df["norad_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["norad_id"])
    df["first_obs"] = pd.to_datetime(df["first_obs"], format="mixed", errors="coerce")
    df["last_obs"] = pd.to_datetime(df["last_obs"], format="mixed", errors="coerce")

    df["tle_start"] = df["first_obs"] - timedelta(days=PADDING_DAYS)
    df["tle_end"] = df["last_obs"] + timedelta(days=PADDING_DAYS)

    earliest = pd.Timestamp("2000-01-01")
    latest = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    df["tle_start"] = df["tle_start"].clip(lower=earliest)
    df["tle_end"] = df["tle_end"].clip(upper=latest)

    return df[["norad_id", "first_obs", "last_obs", "tle_start", "tle_end"]].copy()

def load_multiple_catalogues(cat_paths):
    all_dfs = []
    for path in cat_paths:
        df = load_catalogue(path)
        print(f"  Loaded {path.parent.name}/{path.name}: {len(df)} satellites")
        all_dfs.append(df)

    if not all_dfs:
        print_error("No valid catalogues loaded.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    merged = combined.groupby("norad_id").agg(
        first_obs=("first_obs", "min"),
        last_obs=("last_obs", "max"),
        tle_start=("tle_start", "min"),
        tle_end=("tle_end", "max"),
    ).reset_index()
    return merged


#  STAGE 1: SCAN EXISTING TLE DATA


def scan_existing_tles(tle_dir, catalogue):
    covered, partial, missing = set(), {}, set()
    all_ids = set(catalogue["norad_id"].dropna().astype(int).tolist())
    total_ids = len(all_ids)

    for i, norad_id in enumerate(all_ids, 1):
        if i % 1000 == 0 or i == total_ids:
            print(f"\r  Scanning: {i:,}/{total_ids:,} satellites...", end="", flush=True)

        sat_path = tle_dir / f"{int(norad_id)}.parquet"
        if not sat_path.exists():
            missing.add(norad_id)
            continue

        try:
            df = pd.read_parquet(sat_path)
            if df.empty:
                missing.add(norad_id)
                continue

            df["EPOCH"] = pd.to_datetime(df["EPOCH"], format="mixed", errors="coerce")
            first_epoch, last_epoch = df["EPOCH"].min(), df["EPOCH"].max()
            
            cat_row = catalogue[catalogue["norad_id"] == norad_id].iloc[0]
            start_ok = pd.notna(first_epoch) and first_epoch <= cat_row["tle_start"] + timedelta(days=30)
            end_ok = pd.notna(last_epoch) and last_epoch >= cat_row["tle_end"] - timedelta(days=30)

            if start_ok and end_ok:
                covered.add(norad_id)
            else:
                partial[norad_id] = {
                    "first_epoch": first_epoch, "last_epoch": last_epoch,
                    "missing_early": not start_ok, "missing_late": not end_ok,
                }
        except Exception:
            missing.add(norad_id)

    print() # Add a newline when the progress bar is finished
    return covered, partial, missing


#  STAGE 2: BULK TLE FILE PROCESSING


def parse_tle_decimal(s):
    s = s.strip()
    if not s or s in ('00000-0', '00000+0'): return 0.0
    try:
        sign = -1.0 if s[0] == '-' else 1.0
        s = s[1:] if s[0] in '+-' else s
        if '+' in s[1:] or '-' in s[1:]:
            for i in range(len(s) - 1, 0, -1):
                if s[i] in '+-':
                    return sign * float('0.' + s[:i].strip()) * (10 ** int(s[i:]))
        return sign * float('0.' + s)
    except (ValueError, IndexError):
        return 0.0

def epoch_to_datetime(epoch_year, epoch_day):
    try:
        year = (2000 + int(epoch_year)) if epoch_year < 57 else (1900 + int(epoch_year))
        return datetime(year, 1, 1) + timedelta(days=float(epoch_day) - 1)
    except (ValueError, OverflowError):
        return None

def derive_orbital_params(mean_motion_revday, eccentricity):
    try:
        if mean_motion_revday <= 0: return None, None, None, None
        n_rad_s = mean_motion_revday * TWO_PI / 86400.0
        a = (MU / (n_rad_s ** 2)) ** (1.0 / 3.0)
        return round(a, 3), round(TWO_PI / n_rad_s / 60.0, 4), round(a * (1 + eccentricity) - RE, 3), round(a * (1 - eccentricity) - RE, 3)
    except (ValueError, ZeroDivisionError):
        return None, None, None, None

def parse_tle_pair(line1, line2):
    try:
        epoch_dt = epoch_to_datetime(int(line1[18:20].strip()), float(line1[20:32].strip()))
        if not epoch_dt: return None
        mm, ecc = float(line2[52:63].strip()), float('0.' + line2[26:33].strip())
        sma, period, apogee, perigee = derive_orbital_params(mm, ecc)

        return {
            "NORAD_CAT_ID": int(line1[2:7].strip()), "OBJECT_ID": line1[9:17].strip(),
            "EPOCH": epoch_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "MEAN_MOTION": mm, "ECCENTRICITY": ecc, "INCLINATION": float(line2[8:16].strip()),
            "RA_OF_ASC_NODE": float(line2[17:25].strip()), "ARG_OF_PERICENTER": float(line2[34:42].strip()),
            "MEAN_ANOMALY": float(line2[43:51].strip()), "BSTAR": parse_tle_decimal(line1[53:61]),
            "MEAN_MOTION_DOT": float(line1[33:43].strip()), "MEAN_MOTION_DDOT": parse_tle_decimal(line1[44:52]),
            "SEMIMAJOR_AXIS": sma, "PERIOD": period, "APOAPSIS": apogee, "PERIAPSIS": perigee,
            "REV_AT_EPOCH": line2[63:68].strip() if len(line2) > 63 else '',
            "CLASSIFICATION_TYPE": line1[7], "EPHEMERIS_TYPE": line1[62].strip() if len(line1) > 62 else '0',
            "ELEMENT_SET_NO": line1[64:68].strip() if len(line1) > 64 else '',
        }
    except (ValueError, IndexError):
        return None

def parse_tle_file(filepath, norad_ids_set, log_fn=print):
    path = Path(filepath)
    records, matched, total_pairs = [], 0, 0
    lines_buffer = []
    
    file_size = path.stat().st_size
    bytes_read = 0

    log_fn(f"  Parsing {path.name} ({file_size / (1024*1024):.0f} MB)...")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            bytes_read += len(raw_line)
            line = raw_line.rstrip()
            if not line: continue
            if line[0] == '1' and len(line) >= 69:
                lines_buffer = [line]
            elif line[0] == '2' and len(line) >= 69 and len(lines_buffer) == 1:
                total_pairs += 1
                try:
                    if int(lines_buffer[0][2:7].strip()) not in norad_ids_set:
                        lines_buffer = []
                        continue
                except ValueError:
                    continue
                
                record = parse_tle_pair(lines_buffer[0], line)
                if record:
                    records.append(record)
                    matched += 1
                lines_buffer = []

                if matched > 0 and matched % 50000 == 0:
                    pct = (bytes_read / file_size) * 100
                    print(f"\r    Parsing: {matched:,} matched / {total_pairs:,} total ({pct:.0f}%)", end="", flush=True)

    print(f"\r    Result: {matched:,} matched out of {total_pairs:,} total TLE pairs          ")
    return records

def write_per_satellite(records, tle_dir, log_fn=print):
    if not records: return 0
    
    # 1. Group using native Python (extremely fast, low RAM)
    by_sat = {}
    for rec in records:
        nid = rec["NORAD_CAT_ID"]
        if nid not in by_sat:
            by_sat[nid] = []
        by_sat[nid].append(rec)
        
    written = 0
    total_sats = len(by_sat)
    
    # 2. Write to disk and update progress bar
    for i, (norad_id, sat_records) in enumerate(by_sat.items(), 1):
        sat_path = tle_dir / f"{int(norad_id)}.parquet"
        new_df = pd.DataFrame(sat_records)
        
        if sat_path.exists():
            existing = pd.read_parquet(sat_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["EPOCH"], keep="last", inplace=True)
            combined.to_parquet(sat_path, index=False)
        else:
            new_df.to_parquet(sat_path, index=False)
            
        written += len(sat_records)

        # Progress Feedback
        if i % 100 == 0 or i == total_sats:
            print(f"\r    Writing: {i:,}/{total_sats:,} satellites...", end="", flush=True)
            
    print() # Add a newline when finished
    return written

def run_bulk_processing(tle_input_dir, tle_output_dir, norad_ids_set, log_fn=print):
    tle_files = sorted(Path(tle_input_dir).glob("*.txt"))
    if not tle_files:
        log_fn(f"  No .txt files found in {tle_input_dir}")
        log_fn(TLE_BULK_HELP.format(tle_dir=tle_input_dir))
        return 0

    progress_path = tle_output_dir / ".bulk_progress.json"
    completed = set(json.loads(progress_path.read_text())) if progress_path.exists() else set()
    files_to_process = [f for f in tle_files if f.name not in completed]
    
    if len(tle_files) - len(files_to_process):
        log_fn(f"  Skipping {len(tle_files) - len(files_to_process)} already-processed file(s)")

    total_records = 0
    for idx, tle_file in enumerate(files_to_process, 1):
        log_fn(f"\n  [{idx}/{len(files_to_process)}] {tle_file.name}")
        records = parse_tle_file(tle_file, norad_ids_set, log_fn)
        if records:
            n = write_per_satellite(records, tle_output_dir, log_fn)
            total_records += n
            log_fn(f"    Wrote {n:,} records")
        completed.add(tle_file.name)
        progress_path.write_text(json.dumps(sorted(completed)))

    return total_records


#  STAGE 3: SPACE-TRACK API DOWNLOAD


def load_credentials(cred_file=None):
    try:
        from common.credentials import get_credentials
        creds = get_credentials("spacetrack")
        return creds["username"], creds["password"]
    except Exception:
        print_warning("Credentials not found. Set up config/credentials.json.")
        return None, None

class SpaceTrackClient:
    def __init__(self, username, password):
        self.session = requests.Session()
        self.username, self.password, self.logged_in = username, password, False

    def login(self):
        print("  Logging in to Space-Track...")
        if self.session.post(LOGIN_URL, data={"identity": self.username, "password": self.password}).status_code == 200:
            self.logged_in = True
            print("  Login successful.\n")
            return True
        print_error("Login failed.")
        return False

    def fetch_gp_history(self, norad_id, date_start, date_end):
        if not self.logged_in and not self.login(): return []
        url = f"{GP_HISTORY_URL}/NORAD_CAT_ID/{norad_id}/EPOCH/{date_start}--{date_end}/orderby/EPOCH asc/format/json"
        
        for attempt in range(1, 4):
            try:
                resp = self.session.get(url, timeout=180)
                if resp.status_code == 429: time.sleep(60); continue
                if resp.status_code == 401: self.login(); continue
                if resp.status_code == 200: return resp.json() if isinstance(resp.json(), list) else []
            except requests.exceptions.RequestException:
                time.sleep(20 * attempt)
        return []

def run_api_download(gaps_df, tle_output_dir, cred_file=None, log_fn=print):
    if gaps_df.empty: return 0
    username, password = load_credentials(cred_file)
    if not username: return 0

    client = SpaceTrackClient(username, password)
    if not client.login(): return 0

    progress_path = tle_output_dir / ".api_progress.json"
    completed = set(json.loads(progress_path.read_text())) if progress_path.exists() else set()
    remaining = gaps_df[~gaps_df["norad_id"].isin(completed)]
    
    if remaining.empty:
        log_fn("  All API downloads already completed.")
        return 0

    total_records = 0
    for idx, (_, row) in enumerate(remaining.iterrows(), 1):
        nid, d_start, d_end = int(row["norad_id"]), pd.Timestamp(row["query_start"]), pd.Timestamp(row["query_end"])
        chunk_start = d_start
        
        while chunk_start < d_end:
            chunk_end = min(chunk_start + timedelta(days=TIME_CHUNK_DAYS), d_end)
            s_str, e_str = chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
            print(f"  [{idx}/{len(remaining)}] NORAD {nid:>6d} {s_str} to {e_str}...", end="", flush=True)
            
            records = client.fetch_gp_history(nid, s_str, e_str)
            if records:
                recs = [{k: r.get(k) for k in GP_FIELDS} for r in records]
                sat_path = tle_output_dir / f"{nid}.parquet"
                df = pd.DataFrame(recs)
                
                df["EPOCH"] = pd.to_datetime(df["EPOCH"], format="mixed", errors="coerce")
              
                numeric_cols = [
                    "NORAD_CAT_ID", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", 
                    "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", 
                    "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT", "SEMIMAJOR_AXIS", 
                    "PERIOD", "APOAPSIS", "PERIAPSIS", "REV_AT_EPOCH", "ELEMENT_SET_NO",
                    "EPHEMERIS_TYPE" 
                ]
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                
                if sat_path.exists():
                    existing_df = pd.read_parquet(sat_path)
                    existing_df["EPOCH"] = pd.to_datetime(existing_df["EPOCH"], format="mixed", errors="coerce")
                    df = pd.concat([existing_df, df], ignore_index=True)
                    df.drop_duplicates(subset=["EPOCH"], keep="last", inplace=True)
                    
                df.to_parquet(sat_path, index=False)
                total_records += len(recs)
                print(f" {len(recs):,} records")
            else:
                print(" 0 records")
            time.sleep(RATE_LIMIT_DELAY)
            chunk_start = chunk_end
            
        completed.add(nid)
        progress_path.write_text(json.dumps(sorted(int(n) for n in completed)))
    return total_records


#  PIPELINE ORCHESTRATION


def build_tle_index(tle_dir):
    rows = []
    parquet_files = sorted(Path(tle_dir).glob("*.parquet"))
    total_files = len(parquet_files)

    for i, file_path in enumerate(parquet_files, 1):
        if i % 500 == 0 or i == total_files:
            print(f"\r  Building Index: {i:,}/{total_files:,} files...", end="", flush=True)
            
        try:
            df = pd.read_parquet(file_path)
            if df.empty: continue
            rows.append({
                "norad_id": int(file_path.stem),
                "total_gp_records": len(df),
                "first_epoch": df["EPOCH"].min(),
                "last_epoch": df["EPOCH"].max()
            })
        except Exception: continue
        
    print() # Add a newline when finished
    return pd.DataFrame(rows)

def run_pipeline(catalogue, tle_output_dir, run_bulk=True, run_api=False):
    all_ids = set(catalogue["norad_id"].dropna().astype(int).tolist())

    print(f"\n{'─'*60}\n  STAGE 1: Scanning existing TLE data\n{'─'*60}")
    covered, partial, missing = scan_existing_tles(tle_output_dir, catalogue)
    print(f"  Fully covered:    {len(covered):,}\n  Partial coverage: {len(partial):,}\n  No data at all:   {len(missing):,}")

    need_processing = missing | set(partial.keys())
    if not need_processing:
        print(f"\n  All {len(all_ids)} satellites have adequate TLE coverage!")
        return 0, 0

    bulk_records, need_api = 0, need_processing
    if run_bulk:
        print(f"\n{'─'*60}\n  STAGE 2: Processing bulk TLE files\n{'─'*60}")
        bulk_records = run_bulk_processing(RAW_TLE_BULK, tle_output_dir, need_processing)
        covered2, partial2, missing2 = scan_existing_tles(tle_output_dir, catalogue)
        need_api = missing2 | set(partial2.keys())
        print(f"\n  Now covered: {len(covered2):,} | Still need data: {len(need_api):,}")

    api_records = 0
    if need_api:
        gap_rows = []
        p_ref = partial2 if run_bulk else partial
        for nid in sorted(need_api):
            row = catalogue[catalogue["norad_id"] == nid].iloc[0]
            p_info = p_ref.get(nid, {})
            
            # If the satellite has absolutely no data at all
            if not p_info:
                gap_rows.append({
                    "norad_id": nid, 
                    "query_start": str(row["tle_start"])[:10], 
                    "query_end": str(row["tle_end"])[:10]
                })
            else:
                # If it has data, but is missing the early years
                if p_info.get("missing_early"):
                    gap_rows.append({
                        "norad_id": nid, 
                        "query_start": str(row["tle_start"])[:10], 
                        "query_end": str(p_info["first_epoch"])[:10]
                    })
                # If it has data, but is missing the latest years
                if p_info.get("missing_late"):
                    gap_rows.append({
                        "norad_id": nid, 
                        "query_start": str(p_info["last_epoch"])[:10], 
                        "query_end": str(row["tle_end"])[:10]
                    })
            
        gaps_df = pd.DataFrame(gap_rows)
        gaps_df.to_csv(PROC_TLE / "tle_gaps.csv", index=False)

        if run_api:
            print(f"\n{'─'*60}\n  STAGE 3: Downloading via Space-Track API\n{'─'*60}")
            api_records = run_api_download(gaps_df, tle_output_dir)

    build_tle_index(tle_output_dir).to_csv(PROC_TLE / "tle_index.csv", index=False)
    return bulk_records, api_records

def print_status(tle_output_dir, catalogue):
    print(f"\n  TLE Directory: {tle_output_dir}")
    print(f"  Files on disk: {len(list(tle_output_dir.glob('*.parquet'))):,}")
    if not catalogue.empty:
        covered, partial, missing = scan_existing_tles(tle_output_dir, catalogue)
        pct = len(covered) / len(catalogue) * 100
        print(f"\n  ✓ Fully covered:    {len(covered):,}")
        print(f"  ~ Partial coverage: {len(partial):,}")
        print(f"  ✗ No data:          {len(missing):,}")
        print(f"  Coverage:           {pct:.1f}%")
    print()
    
def estimate_api_time():
    """Reads the generated gaps file and calculates total download time."""
    gaps_path = PROC_TLE / "tle_gaps.csv"
    
    if not gaps_path.exists():
        print_warning("\n  No gaps file found. Please run the scan (Option 3) first!")
        return

    try:
        df = pd.read_csv(gaps_path)
        if df.empty:
            print("\n  No gaps found! You have 100% of the required data.")
            return

        df['query_start'] = pd.to_datetime(df['query_start'], format='mixed', dayfirst=True)
        df['query_end'] = pd.to_datetime(df['query_end'], format='mixed', dayfirst=True)
        
        # Calculate days and required 2-year chunks
        df['days'] = (df['query_end'] - df['query_start']).dt.days
        df['requests'] = df['days'].apply(lambda x: math.ceil(x / TIME_CHUNK_DAYS) if x > 0 else 1)
        
        total_requests = df['requests'].sum()
        total_seconds = total_requests * RATE_LIMIT_DELAY
        total_hours = total_seconds / 3600
        total_days = total_hours / 24
        
        print(f"\n  ┌────────────────────────────────────────────────────────┐")
        print(f"  │  API DOWNLOAD ESTIMATOR                                │")
        print(f"  ├────────────────────────────────────────────────────────┤")
        print(f"  │  Total gap windows : {len(df):,} windows")
        print(f"  │  Total API requests: {total_requests:,} requests (max 2-years each)")
        print(f"  │  Rate limit pacing : {RATE_LIMIT_DELAY} seconds per request")
        print(f"  │  Estimated time    : {total_hours:.2f} Hours ({total_days:.2f} Days)")
        print(f"  └────────────────────────────────────────────────────────┘\n")

    except Exception as e:
        print_error(f"Could not calculate estimate: {e}")


#  MAIN ENTRY POINTS


def main():
    parser = argparse.ArgumentParser(description="Unified TLE Manager (Auto-Detecting)")
    parser.add_argument("--no-api", action="store_true", help="Skip API download")
    parser.add_argument("--api-only", action="store_true", help="Skip bulk processing")
    args = parser.parse_args()

    ensure_dirs()
    tle_output_dir = PROC_TLE / "tle_histories"
    tle_output_dir.mkdir(parents=True, exist_ok=True)

    cat_paths = find_active_catalogues()
    if not cat_paths:
        print_error("No catalogues detected! Run mmt9_processor or sdlcd_processor first.")
        sys.exit(1)

    _print_header("Unified TLE Manager", __version__, "Manage TLE data for all tracked datasets")
    print("  Auto-detected catalogues:")
    for cp in cat_paths: print(f"    ✓ {cp.parent.name}/{cp.name}")
    
    catalogue = load_multiple_catalogues(cat_paths)

    # CLI override behavior
    if args.no_api or args.api_only:
        run_pipeline(catalogue, tle_output_dir, run_bulk=not args.api_only, run_api=not args.no_api)
        return

    # Interactive Loop
    while True:
        choice = _prompt_choice("\n  What would you like to do?", [
            ("status", "Status - View current TLE coverage"),
            ("full", "Run full pipeline (scan → bulk → prompt for API)"),
            ("bulk", "Run scan + bulk processing only (no API)"),
            ("estimate", "Estimate time to download remaining API gaps"),
            ("api", "Download missing TLEs from Space-Track API"),
            ("quit", "Quit")
        ])

        if choice in (None, "quit"): break
        elif choice == "status": print_status(tle_output_dir, catalogue)
        elif choice == "full":
            run_pipeline(catalogue, tle_output_dir, run_bulk=True, run_api=False)
            if prompt_confirm("\n  Download remaining gaps from Space-Track API now?"):
                run_pipeline(catalogue, tle_output_dir, run_bulk=False, run_api=True)
        elif choice == "bulk": run_pipeline(catalogue, tle_output_dir, run_bulk=True, run_api=False)
        elif choice == "estimate": estimate_api_time()
        elif choice == "api": run_pipeline(catalogue, tle_output_dir, run_bulk=False, run_api=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Progress saved.")
        sys.exit(2)