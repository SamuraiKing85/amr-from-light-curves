"""
SDLCD Light Curve Processor
================================
Converts SDLCD light curve data (downloaded by 1_acquire/sdlcd_downloader.py)
into MMT-9 compatible format for the ML pipeline.

Input:  data/raw/sdlcd/database.json + data/raw/sdlcd/lightcurves/*.txt
Output: data/processed/sdlcd/lightcurves/*.parquet + data/processed/sdlcd/catalogue.csv

Processing uses CSV as a streaming working format (supports efficient append),
then converts to parquet as a final step for faster downstream loading and
smaller file sizes.

The per-satellite files have the first 9 columns identical to MMT-9 format,
with additional SDLCD-unique columns appended (mag errors, ADU, rotation period, etc.).

Interactive menu (no arguments):
    python sdlcd_processor.py

CLI mode:
    python sdlcd_processor.py --process
    python sdlcd_processor.py --process --database /path/to/database.json
    python sdlcd_processor.py --rebuild-catalogue
    python sdlcd_processor.py --status
"""

__version__ = "3.0.0"

import sys
import csv
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import RAW_SDLCD, PROC_SDLCD, ensure_dirs
from common.menu import (
    print_header, prompt_choice, prompt_path, prompt_confirm,
    print_section, print_status_bar, print_summary_table,
    print_warning, print_error,
)
from common.progress import CheckpointManager
from common.manifest import DataManifest
from common.logging_setup import setup_logging

# ═══════════════════════════════════════════════════════════════
#  REMOTE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

DATABASE_URL = "https://www.sdlcd.space-debris.sk/static/data/database.json"

# ═══════════════════════════════════════════════════════════════
#  COLUMN DEFINITIONS
# ═══════════════════════════════════════════════════════════════

# First 9 columns match MMT-9 exactly; rest are SDLCD-unique
OUTPUT_COLUMNS = [
    # ── MMT-9 compatible (columns 0-8) ──
    "Datetime", "Track", "Channel", "StdMag", "Mag",
    "Filter", "Penumbra", "Distance_km", "Phase_deg",
    # ── SDLCD-unique per-observation ──
    "Mag_err", "ADU", "ADU_err", "Exposure_s",
    # ── SDLCD-unique per-session (constant within a track) ──
    "COSPAR", "Object_name", "Object_type",
    "Rotation_period_s", "Period_error_s",
    "Amplitude_mag", "Amplitude_math_mag", "Amp_error_mag",
    "Session_duration_s", "Mean_sampling_s",
    "Orbital_period_min", "Inclination_deg",
    "Perigee_km", "Apogee_km", "Decayed", "Source",
]

FILTER_MAP = {
    'C': 'Clear', 'R': 'R', 'V': 'V', 'B': 'B', 'I': 'I', '': 'Clear',
}

# Typed columns for parquet conversion
PARQUET_DTYPES = {
    "Track": "int64",
    "Channel": "int16",
    "StdMag": "float32",
    "Mag": "float32",
    "Penumbra": "int8",
    "Distance_km": "float32",
    "Phase_deg": "float32",
    "Mag_err": "float32",
    "ADU": "float32",
    "ADU_err": "float32",
    "Exposure_s": "float32",
    "Rotation_period_s": "float32",
    "Period_error_s": "float32",
    "Amplitude_mag": "float32",
    "Amplitude_math_mag": "float32",
    "Amp_error_mag": "float32",
    "Session_duration_s": "float32",
    "Mean_sampling_s": "float32",
    "Orbital_period_min": "float32",
    "Inclination_deg": "float32",
    "Perigee_km": "float32",
    "Apogee_km": "float32",
}

# ═══════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════

logger = setup_logging("sdlcd_processor")


# ═══════════════════════════════════════════════════════════════
#  DATABASE LOADING
# ═══════════════════════════════════════════════════════════════

def find_database(raw_dir: Path) -> Path:
    """Locate database.json. Downloads it from the SDLCD website if missing."""
    for name in ["database.json", "Database.json"]:
        p = raw_dir / name
        if p.exists():
            return p

    # Not found - try to download
    print_warning(f"database.json not found in {raw_dir}")
    print(f"          Downloading from SDLCD website...")

    target_path = raw_dir / "database.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(DATABASE_URL, timeout=30, verify=False,
                            headers={"User-Agent": "SDLCD-Processor/3.0 (University Research)"})
        resp.raise_for_status()

        data = resp.json()
        if not data or not isinstance(data, dict):
            print_error("Downloaded file doesn't look like valid SDLCD data.")
            return None

        target_path.write_text(resp.text, encoding="utf-8")
        print(f"          Saved to: {target_path}")
        print(f"          ({len(data)} entries)\n")
        return target_path

    except Exception as e:
        print_error(f"Failed to download database.json: {e}")
        print(f"          Download manually from:")
        print(f"          {DATABASE_URL}")
        print(f"          and place it in: {raw_dir}/")
        return None


def load_metadata(database_path: Path) -> dict:
    """Load database.json and build metadata lookup by SDLCD ID."""
    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    metadata = {}
    for idx, entry in database.items():
        norad = entry.get("norad", "")
        try:
            norad_int = int(norad)
        except (ValueError, TypeError):
            continue

        metadata[int(idx)] = {
            "sdlcd_id": int(idx),
            "norad_id": norad_int,
            "cospar": entry.get("object-id", ""),
            "object_name": entry.get("object-name", ""),
            "object_type": entry.get("object-type", ""),
            "data_file": entry.get("data-file", ""),
            "data_fit_file": entry.get("data-fit-file", ""),
            "filter": entry.get("filter", ""),
            "period_s": entry.get("period", ""),
            "period_error_s": entry.get("period-error", ""),
            "amplitude_mag": entry.get("ampdat", ""),
            "amplitude_math_mag": entry.get("ampmath", ""),
            "amp_error_mag": entry.get("amperr", ""),
            "exposure_s": entry.get("exposure", ""),
            "num_points": entry.get("points", ""),
            "orbital_period_min": entry.get("orbital-period", ""),
            "inclination_deg": entry.get("inclination", ""),
            "perigee_km": entry.get("perigee", ""),
            "apogee_km": entry.get("apogee", ""),
            "decayed": entry.get("decayed", False),
            "observation_date": entry.get("date", ""),
            "mean_sampling_s": entry.get("apd", ""),
            "median_pd_s": entry.get("medpd", ""),
            "mode_pd_s": entry.get("modepd", ""),
            "std_pd_s": entry.get("stdpd", ""),
            "mag_error_avg": entry.get("mag-error", ""),
            "adu_error_avg": entry.get("adu-error", ""),
            "nmax": entry.get("nmax", ""),
            "nmin": entry.get("nmin", ""),
        }

    return metadata


# ═══════════════════════════════════════════════════════════════
#  LIGHT CURVE PARSER
# ═══════════════════════════════════════════════════════════════

def parse_sdlcd_lightcurve(filepath: Path) -> dict:
    """Parse a single SDLCD _DATA.txt file.

    Returns dict with 'header', 'period_info', 'observations'.
    """
    header = {}
    period_info = {}
    observations = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            if line.startswith("#"):
                clean = line.lstrip("#").strip()
                if not clean or clean.startswith("="):
                    continue

                if "Obs. time" in clean:
                    _, val = clean.split(":", 1)
                    header["obs_time"] = val.strip()
                elif "Duration" in clean:
                    _, val = clean.split(":", 1)
                    try:
                        header["duration_s"] = float(val.strip())
                    except ValueError:
                        pass
                elif clean[0].isdigit() or (clean[0] in "+-" and len(clean) > 1):
                    parts = clean.split()
                    if len(parts) >= 6:
                        try:
                            period_info = {
                                "period_s": float(parts[0]),
                                "period_err": float(parts[1]),
                                "adu_err_avg": float(parts[2]),
                                "mag_err_avg": float(parts[3]),
                                "num_points": int(parts[4]),
                                "exposure_s": float(parts[5]),
                            }
                        except (ValueError, IndexError):
                            pass
                elif not any(kw in clean for kw in ["Per[s]", "Time", "ADU"]):
                    if "_" in clean and ":" not in clean:
                        header["file_id"] = clean.strip()
            else:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        obs = {
                            "time_s": float(parts[0]),
                            "exposure": float(parts[1]),
                            "adu": float(parts[2]),
                            "adu_err": float(parts[3]),
                            "mag": float(parts[4]),
                        }
                        if len(parts) > 5:
                            obs["mag_err"] = float(parts[5])
                        observations.append(obs)
                    except ValueError:
                        continue

    return {"header": header, "period_info": period_info, "observations": observations}


# ═══════════════════════════════════════════════════════════════
#  FORMAT CONVERSION
# ═══════════════════════════════════════════════════════════════

def convert_to_mmt9_format(parsed_lc: dict, entry: dict) -> list:
    """Convert parsed SDLCD light curve into rows with MMT-9 base columns
    plus SDLCD-unique extra columns."""
    rows = []

    obs_time_str = parsed_lc["header"].get("obs_time", "")
    try:
        base_dt = datetime.strptime(obs_time_str, "%d-%b-%Y %H:%M:%S")
    except ValueError:
        try:
            base_dt = pd.to_datetime(obs_time_str)
        except Exception:
            return rows

    sdlcd_id = entry.get("sdlcd_id", 0)
    filter_code = entry.get("filter", "C")
    filter_name = FILTER_MAP.get(filter_code, filter_code)

    # Estimate distance from orbital parameters
    try:
        perigee = float(entry.get("perigee_km", 0) or 0)
        apogee = float(entry.get("apogee_km", 0) or 0)
        if perigee > 0 and apogee > 0:
            distance_km = round((perigee + apogee) / 2.0, 3)
        elif perigee > 0:
            distance_km = round(perigee, 3)
        elif apogee > 0:
            distance_km = round(apogee, 3)
        else:
            distance_km = 0.0
    except (ValueError, TypeError):
        distance_km = 0.0
        perigee = 0.0
        apogee = 0.0

    track_id = 900000000 + sdlcd_id

    for obs in parsed_lc["observations"]:
        dt = base_dt + timedelta(seconds=obs["time_s"])
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        rows.append([
            # ── MMT-9 compatible (cols 0-8) ──
            dt_str, track_id, 1, round(obs["mag"], 4), round(obs["mag"], 4),
            filter_name, 0, distance_km, 0.0,
            # ── SDLCD-unique per-point ──
            obs.get("mag_err", ""), round(obs.get("adu", 0), 2),
            round(obs.get("adu_err", 0), 2), obs.get("exposure", ""),
            # ── SDLCD-unique per-session ──
            entry.get("cospar", ""), entry.get("object_name", ""),
            entry.get("object_type", ""), entry.get("period_s", ""),
            entry.get("period_error_s", ""), entry.get("amplitude_mag", ""),
            entry.get("amplitude_math_mag", ""), entry.get("amp_error_mag", ""),
            parsed_lc["header"].get("duration_s", ""), entry.get("mean_sampling_s", ""),
            entry.get("orbital_period_min", ""), entry.get("inclination_deg", ""),
            perigee, apogee, entry.get("decayed", ""), "SDLCD",
        ])

    return rows


# ═══════════════════════════════════════════════════════════════
#  CSV TO PARQUET CONVERSION
# ═══════════════════════════════════════════════════════════════

def convert_csvs_to_parquet(lc_dir: Path, log_fn=print,
                            delete_csvs: bool = True) -> int:
    """Convert all per-satellite CSV files to parquet.

    Reads each CSV, coerces numeric columns to proper dtypes for
    compression, and writes a snappy-compressed parquet file.
    Optionally deletes the CSV afterwards.

    Returns:
        Number of files converted.
    """
    csv_files = sorted(lc_dir.glob("*.csv"))
    if not csv_files:
        log_fn("  No CSV files to convert.")
        return 0

    log_fn(f"  Converting {len(csv_files)} CSV files to parquet...")

    converted = 0
    total_csv_bytes = 0
    total_pq_bytes = 0

    for i, csv_path in enumerate(csv_files, 1):
        try:
            sat_id = int(csv_path.stem)
        except ValueError:
            continue

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                if delete_csvs:
                    csv_path.unlink()
                continue

            # Parse datetime
            df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")

            # Coerce numeric columns - use errors='coerce' since SDLCD
            # has many columns that may contain empty strings
            for col, dtype in PARQUET_DTYPES.items():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

            # Write parquet
            pq_path = lc_dir / f"{sat_id}.parquet"
            df.to_parquet(pq_path, index=False, engine="pyarrow",
                          compression="snappy")

            total_csv_bytes += csv_path.stat().st_size
            total_pq_bytes += pq_path.stat().st_size
            converted += 1

            if delete_csvs:
                csv_path.unlink()

        except Exception as e:
            log_fn(f"  [WARN] Failed to convert {csv_path.name}: {e}")
            continue

        if i % 200 == 0:
            log_fn(f"    ... converted {i}/{len(csv_files)}")

    if converted > 0:
        csv_mb = total_csv_bytes / (1024 * 1024)
        pq_mb = total_pq_bytes / (1024 * 1024)
        ratio = pq_mb / csv_mb * 100 if csv_mb > 0 else 0
        log_fn(f"  Converted {converted} files: "
               f"{csv_mb:.1f} MB CSV -> {pq_mb:.1f} MB parquet ({ratio:.0f}%)")
        if delete_csvs:
            log_fn(f"  Working CSV files removed.")

    return converted


# ═══════════════════════════════════════════════════════════════
#  CATALOGUE BUILDER
# ═══════════════════════════════════════════════════════════════

def build_catalogue(lc_dir: Path, log_fn=print) -> pd.DataFrame:
    """Build catalogue by scanning per-satellite files.
    Prefers parquet, falls back to CSV. First 14 columns match MMT-9
    catalogue; additional columns for SDLCD-unique stats.
    """
    pq_files = sorted(Path(lc_dir).glob("*.parquet"))
    csv_files = sorted(Path(lc_dir).glob("*.csv"))

    if pq_files:
        data_files = pq_files
        reader = pd.read_parquet
        fmt = "parquet"
    elif csv_files:
        data_files = csv_files
        reader = pd.read_csv
        fmt = "CSV"
    else:
        log_fn("  No satellite files found.")
        return pd.DataFrame()

    log_fn(f"  Scanning {len(data_files)} {fmt} satellite file(s)...")
    rows = []

    for fpath in data_files:
        try:
            sat_id = int(fpath.stem)
        except ValueError:
            continue

        try:
            df = reader(fpath)
            if df.empty:
                continue

            if "Datetime" in df.columns and df["Datetime"].dtype == object:
                df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")

            n = len(df)

            row = {
                "norad_id": sat_id,
                "total_observations": n,
                "num_tracks": df["Track"].nunique(),
                "num_channels": df["Channel"].nunique(),
                "first_obs": df["Datetime"].min(),
                "last_obs": df["Datetime"].max(),
                "mean_stdmag": round(df["StdMag"].mean(), 4),
                "std_stdmag": round(df["StdMag"].std(), 4) if n > 1 else 0.0,
                "min_stdmag": df["StdMag"].min(),
                "max_stdmag": df["StdMag"].max(),
                "mean_distance_km": round(df["Distance_km"].mean(), 2),
                "mean_phase_deg": round(df["Phase_deg"].mean(), 2),
                "filters_used": ",".join(sorted(df["Filter"].dropna().unique())),
            }

            # SDLCD-unique catalogue columns
            for col, key in [("COSPAR", "cospar"), ("Object_name", "object_name"),
                             ("Object_type", "object_type")]:
                if col in df.columns:
                    row[key] = df[col].iloc[0]

            if "ADU" in df.columns:
                adu = pd.to_numeric(df["ADU"], errors="coerce")
                row["mean_adu"] = round(adu.mean(), 2) if adu.notna().any() else None

            if "Mag_err" in df.columns:
                merr = pd.to_numeric(df["Mag_err"], errors="coerce")
                row["mean_mag_err"] = round(merr.mean(), 6) if merr.notna().any() else None

            if "Rotation_period_s" in df.columns:
                periods = pd.to_numeric(df["Rotation_period_s"], errors="coerce").dropna()
                real = periods[~periods.isin([9999, 5000, 6000, 7000])]
                row["rotation_period_s"] = real.iloc[0] if len(real) > 0 else None

            for col, key in [("Amplitude_mag", "amplitude_mag"),
                             ("Inclination_deg", "inclination_deg")]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    row[key] = vals.iloc[0] if len(vals) > 0 else None

            for col, key in [("Perigee_km", "perigee_km"), ("Apogee_km", "apogee_km"),
                             ("Orbital_period_min", "orbital_period_min")]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    row[key] = vals.iloc[0] if len(vals) > 0 else None

            if "Decayed" in df.columns:
                row["decayed"] = df["Decayed"].iloc[0]

            row["source"] = "SDLCD"
            rows.append(row)

        except Exception as e:
            logger.debug(f"Failed to process {fpath.name}: {e}")
            continue

    catalogue = pd.DataFrame(rows)
    if not catalogue.empty:
        catalogue["obs_span_days"] = (
            (catalogue["last_obs"] - catalogue["first_obs"]).dt.total_seconds() / 86400
        )
        catalogue.sort_values("total_observations", ascending=False, inplace=True)
        catalogue.reset_index(drop=True, inplace=True)

    return catalogue


def generate_tle_catalogue(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Generate catalogue formatted for tle_downloader/tle_manager."""
    if catalogue.empty:
        return pd.DataFrame(columns=["norad_id", "first_obs", "last_obs"])
    tle_cat = catalogue[["norad_id", "first_obs", "last_obs"]].copy()
    tle_cat["norad_id"] = tle_cat["norad_id"].astype("Int64")
    return tle_cat


# ═══════════════════════════════════════════════════════════════
#  MAIN PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_processing(database_path: Path, raw_lc_dir: Path, output_dir: Path,
                   lc_out_dir: Path, checkpoint: CheckpointManager,
                   manifest: DataManifest = None, log_fn=print) -> dict:
    """Main pipeline: parse raw SDLCD files and convert to MMT-9 format.

    Three phases:
      1. Parse raw .txt files into per-satellite CSVs (streaming append)
      2. Convert CSVs to parquet (final format)
      3. Build catalogue + metadata from parquet files
    """
    lc_out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load database.json
    log_fn(f"  Loading {database_path.name}...")
    metadata = load_metadata(database_path)
    log_fn(f"  {len(metadata)} entries with valid NORAD IDs\n")

    # Step 2: Group entries by NORAD ID
    norad_entries = defaultdict(list)
    for sdlcd_id, meta in metadata.items():
        norad_entries[meta["norad_id"]].append(meta)

    # ── Phase 1: Parse and convert to CSV ──
    log_fn(f"  Processing light curves into MMT-9 format...")

    known_sats = set()
    if checkpoint.completed_count > 0:
        for pattern in ["*.csv", "*.parquet"]:
            for f in lc_out_dir.glob(pattern):
                try:
                    known_sats.add(int(f.stem))
                except ValueError:
                    continue
        log_fn(f"  Resuming: {checkpoint.completed_count} entries done, "
               f"{len(known_sats)} satellites on disk")

    total_rows = 0
    total_files_parsed = 0
    sdlcd_meta_rows = []

    for norad_id, entries in norad_entries.items():
        all_rows = []

        for entry in entries:
            data_file = entry.get("data_file", "")
            if not data_file:
                continue

            if checkpoint.is_done(data_file):
                continue

            filepath = raw_lc_dir / data_file
            if not filepath.exists():
                continue

            parsed = parse_sdlcd_lightcurve(filepath)
            if not parsed["observations"]:
                checkpoint.mark_done(data_file)
                if manifest:
                    manifest.record_processed(data_file)
                continue

            rows = convert_to_mmt9_format(parsed, entry)
            all_rows.extend(rows)
            total_files_parsed += 1

            # Collect SDLCD-specific metadata
            sdlcd_meta_rows.append({
                "norad_id": norad_id,
                "sdlcd_id": entry["sdlcd_id"],
                "cospar": entry["cospar"],
                "object_name": entry["object_name"],
                "object_type": entry["object_type"],
                "data_file": data_file,
                "filter": entry["filter"],
                "observation_date": entry["observation_date"],
                "period_s": entry["period_s"],
                "period_error_s": entry["period_error_s"],
                "amplitude_mag": entry["amplitude_mag"],
                "exposure_s": entry["exposure_s"],
                "num_points": entry["num_points"],
                "num_points_parsed": len(parsed["observations"]),
                "duration_s": parsed["header"].get("duration_s", ""),
                "source": "SDLCD",
            })

            checkpoint.mark_done(data_file)
            if manifest:
                manifest.record_processed(data_file)

        if not all_rows:
            continue

        # Write per-NORAD CSV (working format)
        sat_path = lc_out_dir / f"{norad_id}.csv"
        is_new = norad_id not in known_sats

        mode = "w" if is_new else "a"
        with open(sat_path, mode, newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if is_new:
                w.writerow(OUTPUT_COLUMNS)
                known_sats.add(norad_id)
            w.writerows(all_rows)

        total_rows += len(all_rows)

    checkpoint.save()

    log_fn(f"  Parsed {total_files_parsed} LC files")
    log_fn(f"  Wrote {total_rows:,} observations for {len(known_sats)} satellites\n")

    # ── Phase 2: Convert CSVs to parquet ──
    converted = convert_csvs_to_parquet(lc_out_dir, log_fn, delete_csvs=True)

    # ── Phase 3: Build catalogue + metadata ──
    log_fn("\n  Building catalogue...")
    catalogue = build_catalogue(lc_out_dir, log_fn)
    cat_path = output_dir / "catalogue.csv"
    catalogue.to_csv(cat_path, index=False)
    log_fn(f"  Catalogue: {cat_path} ({len(catalogue)} satellites)\n")

    # Save SDLCD metadata
    if sdlcd_meta_rows:
        meta_df = pd.DataFrame(sdlcd_meta_rows)
        meta_path = output_dir / "sdlcd_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        log_fn(f"  SDLCD metadata: {meta_path}\n")

    # Generate TLE catalogue
    tle_cat = generate_tle_catalogue(catalogue)
    tle_path = output_dir / "sdlcd_tle_catalogue.csv"
    tle_cat.to_csv(tle_path, index=False)
    log_fn(f"  TLE catalogue: {tle_path} ({len(tle_cat)} satellites)")

    # Summary
    if not catalogue.empty and sdlcd_meta_rows:
        meta_df = pd.DataFrame(sdlcd_meta_rows)
        log_fn(f"\n  Object types:")
        type_counts = meta_df.groupby("object_type")["norad_id"].nunique()
        for t, c in type_counts.items():
            log_fn(f"    {t:<20s}  {c} satellites")

    return {
        "rows": total_rows,
        "files_parsed": total_files_parsed,
        "satellites": len(known_sats),
        "converted": converted,
    }


# ═══════════════════════════════════════════════════════════════
#  STATUS
# ═══════════════════════════════════════════════════════════════

def show_status(raw_lc_dir: Path, output_dir: Path, lc_out_dir: Path,
                checkpoint: CheckpointManager):
    """Display processing status."""
    print_section("PROCESSING STATUS")

    raw_files = list(raw_lc_dir.glob("*_DATA*.txt")) if raw_lc_dir.exists() else []
    pq_files = list(lc_out_dir.glob("*.parquet")) if lc_out_dir.exists() else []
    csv_files = list(lc_out_dir.glob("*.csv")) if lc_out_dir.exists() else []
    cat_path = output_dir / "catalogue.csv"

    rows = [
        ("Raw LC files:",            f"{len(raw_files):,}"),
        ("Files processed:",         f"{checkpoint.completed_count:,}"),
        ("Satellite parquet files:", f"{len(pq_files):,}"),
    ]

    if csv_files:
        rows.append(("Working CSV files:", f"{len(csv_files):,} (not yet converted)"))

    rows.append(("Catalogue exists:", "Yes" if cat_path.exists() else "No"))

    print_summary_table(rows)

    if raw_files:
        print()
        print_status_bar("Progress", checkpoint.completed_count, len(raw_files))
    print()


# ═══════════════════════════════════════════════════════════════
#  INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════

def run_interactive(raw_dir: Path, output_dir: Path, lc_out_dir: Path):
    """Run the interactive menu."""
    print_header("SDLCD Processor", __version__,
                 "Convert SDLCD light curves to per-satellite parquet files")

    checkpoint = CheckpointManager("sdlcd_process")
    manifest = DataManifest("sdlcd_raw")

    # Locate database and raw LC directory
    db_path = find_database(raw_dir)
    raw_lc_dir = raw_dir / "lightcurves"

    if db_path:
        print(f"  Database: {db_path}")
    else:
        print(f"          Run 1_acquire/sdlcd_downloader.py to download data first.\n")

    if raw_lc_dir.exists():
        n_raw = len(list(raw_lc_dir.glob("*_DATA*.txt")))
        print(f"  Raw LC files: {n_raw}")
    else:
        print_warning(f"No lightcurves/ folder in {raw_dir}")

    while True:
        options = []

        if db_path and raw_lc_dir.exists():
            options.append(("process", "Process raw SDLCD files (resume if interrupted)"))

        has_parquet = lc_out_dir.exists() and list(lc_out_dir.glob("*.parquet"))
        has_csv = lc_out_dir.exists() and list(lc_out_dir.glob("*.csv"))

        if has_parquet or has_csv:
            options.append(("rebuild", "Rebuild catalogue from existing lightcurve files"))

        if has_csv:
            n_csv = len(list(lc_out_dir.glob("*.csv")))
            options.append(("convert", f"Convert {n_csv} working CSV files to parquet"))

        options.append(("status", "View processing status"))

        if checkpoint.completed_count > 0:
            options.append(("reset", "Clear checkpoint (reprocess everything)"))

        options.append(("quit", "Quit"))

        choice = prompt_choice("What would you like to do?", options)

        if choice is None or choice == "quit":
            print("\n  Goodbye!\n")
            break

        elif choice == "process":
            print_section("PROCESSING")
            result = run_processing(
                db_path, raw_lc_dir, output_dir, lc_out_dir, checkpoint,
                manifest=manifest,
            )
            print_section("SUMMARY")
            print_summary_table([
                ("Observations:",         f"{result['rows']:,}"),
                ("Files parsed:",         f"{result['files_parsed']:,}"),
                ("Satellites:",           f"{result['satellites']:,}"),
                ("Parquet files created:", f"{result['converted']:,}"),
            ])
            print()

        elif choice == "convert":
            print_section("CONVERTING CSV TO PARQUET")
            converted = convert_csvs_to_parquet(lc_out_dir, delete_csvs=True)
            print(f"  Converted {converted} files.\n")

        elif choice == "rebuild":
            print(f"\n  Rebuilding catalogue...")
            catalogue = build_catalogue(lc_out_dir)
            cat_path = output_dir / "catalogue.csv"
            catalogue.to_csv(cat_path, index=False)
            print(f"  Saved: {cat_path} ({len(catalogue)} satellites)\n")

        elif choice == "status":
            show_status(raw_lc_dir, output_dir, lc_out_dir, checkpoint)

        elif choice == "reset":
            if prompt_confirm("Clear checkpoint? Files on disk are kept."):
                checkpoint.clear()
                print("  Checkpoint cleared.\n")


# ═══════════════════════════════════════════════════════════════
#  CLI MODE
# ═══════════════════════════════════════════════════════════════

def run_cli(args, raw_dir: Path, output_dir: Path, lc_out_dir: Path):
    """Run in CLI mode."""
    print_header("SDLCD Processor", __version__)
    checkpoint = CheckpointManager("sdlcd_process")
    manifest = DataManifest("sdlcd_raw")

    raw_lc_dir = raw_dir / "lightcurves"

    if args.status:
        show_status(raw_lc_dir, output_dir, lc_out_dir, checkpoint)

    elif args.rebuild_catalogue:
        print(f"  Rebuilding catalogue...")
        catalogue = build_catalogue(lc_out_dir)
        cat_path = output_dir / "catalogue.csv"
        catalogue.to_csv(cat_path, index=False)
        print(f"  Saved: {cat_path} ({len(catalogue)} satellites)\n")

    elif args.convert:
        print(f"  Converting CSV files to parquet...")
        converted = convert_csvs_to_parquet(lc_out_dir, delete_csvs=not args.keep_csv)
        print(f"  Converted {converted} files.\n")

    elif args.process:
        db_path = Path(args.database) if args.database else find_database(raw_dir)
        if db_path is None:
            print_error(f"database.json not found. Use --database or place it in {raw_dir}")
            sys.exit(1)
        if not raw_lc_dir.exists():
            print_error(f"No lightcurves/ folder in {raw_dir}. Run sdlcd_downloader.py first.")
            sys.exit(1)

        result = run_processing(db_path, raw_lc_dir, output_dir, lc_out_dir,
                                checkpoint, manifest=manifest)
        print_section("SUMMARY")
        print_summary_table([
            ("Observations:", f"{result['rows']:,}"),
            ("Files parsed:", f"{result['files_parsed']:,}"),
            ("Satellites:",   f"{result['satellites']:,}"),
            ("Converted:",    f"{result['converted']:,}"),
        ])
        print()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SDLCD Light Curve Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sdlcd_processor.py                        # Interactive menu
  python sdlcd_processor.py --process              # Process all
  python sdlcd_processor.py --process --database db.json
  python sdlcd_processor.py --convert              # Convert working CSVs to parquet
  python sdlcd_processor.py --rebuild-catalogue    # Rebuild from existing files
  python sdlcd_processor.py --status               # Show progress
        """,
    )
    parser.add_argument("--process", action="store_true", help="Process raw files")
    parser.add_argument("--convert", action="store_true",
                        help="Convert working CSV files to parquet")
    parser.add_argument("--keep-csv", action="store_true",
                        help="Keep CSV files after parquet conversion")
    parser.add_argument("--rebuild-catalogue", action="store_true",
                        help="Rebuild catalogue from existing lightcurve files")
    parser.add_argument("--status", action="store_true", help="Show processing status")
    parser.add_argument("--database", type=str, default=None,
                        help="Path to database.json")
    parser.add_argument("--input", type=str, default=None,
                        help="Override raw input directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Override processed output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    ensure_dirs()
    raw_dir = Path(args.input) if args.input else RAW_SDLCD
    output_dir = Path(args.output) if args.output else PROC_SDLCD
    lc_out_dir = output_dir / "lightcurves"

    has_action = args.process or args.rebuild_catalogue or args.status or args.convert

    try:
        if has_action:
            run_cli(args, raw_dir, output_dir, lc_out_dir)
        else:
            run_interactive(raw_dir, output_dir, lc_out_dir)
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Progress saved.")
        sys.exit(2)


if __name__ == "__main__":
    main()
