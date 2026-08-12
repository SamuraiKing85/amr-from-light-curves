"""
MMT-9 Light Curve Processor
================================
Ingests raw MMT-9 text dumps (downloaded by 1_acquire/mmt9_downloader.py),
splits them by NORAD ID, and produces:
  - One parquet file per satellite in processed/mmt9/lightcurves/{norad_id}.parquet
  - A master catalogue at processed/mmt9/catalogue.csv

Processing uses CSV as a streaming working format (supports efficient append),
then converts to parquet as a final step for faster downstream loading and
smaller file sizes.

Supports resume: if interrupted, re-run and it picks up where it left off.

Interactive menu (no arguments):
    python mmt9_processor.py

CLI mode:
    python mmt9_processor.py --process
    python mmt9_processor.py --rebuild-catalogue
    python mmt9_processor.py --status
"""

__version__ = "3.0.0"

import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import RAW_MMT9, PROC_MMT9, ensure_dirs
from common.menu import (
    print_header, prompt_choice, prompt_path, prompt_confirm,
    print_section, print_status_bar, print_summary_table,
    print_warning, print_error,
)
from common.progress import CheckpointManager
from common.manifest import DataManifest
from common.logging_setup import setup_logging


#  COLUMN DEFINITIONS


INPUT_COLUMNS = [
    "Date", "Time", "StdMag", "Mag", "Filter",
    "Penumbra", "Distance_km", "Phase_deg", "Channel", "Track", "Satellite",
]

OUTPUT_COLUMNS = [
    "Datetime", "Track", "Channel", "StdMag", "Mag",
    "Filter", "Penumbra", "Distance_km", "Phase_deg",
]

# Data types for parquet conversion (keeps files small and typed)
PARQUET_DTYPES = {
    "Track": "int32",
    "Channel": "int16",
    "StdMag": "float32",
    "Mag": "float32",
    "Penumbra": "int8",
    "Distance_km": "float32",
    "Phase_deg": "float32",
}


#  SETUP


logger = setup_logging("mmt9_processor")



#  CORE PROCESSING ENGINE (STREAMING CSV)


def parse_and_route(filepath: Path, lc_dir: Path, known_sats: set,
                    stats: dict, log_fn=print) -> tuple[int, int]:
    """Parse a single raw file and append rows to per-satellite CSV files.

    Uses CSV as the working format because it supports efficient streaming
    append. The CSV files are converted to parquet in a later step.

    Buffers rows per satellite in memory, then writes them in bulk.
    This avoids hitting OS open file limits.

    Returns:
        (rows_written, num_satellites_in_file)
    """
    rows_written = 0
    skipped = 0
    sats_in_file = set()
    buffer = {}  # sat_id -> list of rows

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("date"):
                continue

            parts = line.split()
            if len(parts) != len(INPUT_COLUMNS):
                skipped += 1
                continue

            try:
                sat_id = int(parts[10])
            except (ValueError, IndexError):
                skipped += 1
                continue

            dt_str = parts[0] + " " + parts[1]

            try:
                stdmag = float(parts[2])
                mag = float(parts[3])
                penumbra = int(parts[5])
                distance = float(parts[6])
                phase = float(parts[7])
                channel = int(parts[8])
                track = int(parts[9])
            except (ValueError, IndexError):
                skipped += 1
                continue

            row = [dt_str, track, channel, stdmag, mag,
                   parts[4], penumbra, distance, phase]

            if sat_id not in buffer:
                buffer[sat_id] = []
            buffer[sat_id].append(row)
            sats_in_file.add(sat_id)

            # Update running statistics
            if sat_id not in stats:
                stats[sat_id] = {
                    "total_obs": 0, "tracks": set(), "channels": set(),
                    "first_obs": dt_str, "last_obs": dt_str,
                    "stdmag_sum": 0.0, "stdmag_sq_sum": 0.0,
                    "stdmag_min": float("inf"), "stdmag_max": float("-inf"),
                    "distance_sum": 0.0, "phase_sum": 0.0,
                    "filters": set(),
                }

            s = stats[sat_id]
            s["total_obs"] += 1
            s["tracks"].add(track)
            s["channels"].add(channel)
            if dt_str < s["first_obs"]:
                s["first_obs"] = dt_str
            if dt_str > s["last_obs"]:
                s["last_obs"] = dt_str
            s["stdmag_sum"] += stdmag
            s["stdmag_sq_sum"] += stdmag * stdmag
            if stdmag < s["stdmag_min"]:
                s["stdmag_min"] = stdmag
            if stdmag > s["stdmag_max"]:
                s["stdmag_max"] = stdmag
            s["distance_sum"] += distance
            s["phase_sum"] += phase
            s["filters"].add(parts[4])

    # Write buffered rows to per-satellite CSVs (working format)
    for sat_id, sat_rows in buffer.items():
        sat_path = lc_dir / f"{sat_id}.csv"
        is_new = sat_id not in known_sats

        mode = "w" if is_new else "a"
        with open(sat_path, mode, newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if is_new:
                w.writerow(OUTPUT_COLUMNS)
                known_sats.add(sat_id)
            w.writerows(sat_rows)

        rows_written += len(sat_rows)

    if skipped > 0:
        log_fn(f"  [WARN] {filepath.name}: skipped {skipped} malformed row(s)")

    return rows_written, len(sats_in_file)



#  CSV TO PARQUET CONVERSION


def convert_csvs_to_parquet(lc_dir: Path, log_fn=print,
                            delete_csvs: bool = True) -> int:
    """Convert all per-satellite CSV files to parquet.

    Reads each CSV, applies proper dtypes for compression, and writes
    a snappy-compressed parquet file. Optionally deletes the CSV
    afterwards since it is only a working intermediate.

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

            # Apply typed columns for smaller parquet files
            df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")
            for col, dtype in PARQUET_DTYPES.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)

            # Write parquet with compression
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

        if i % 1000 == 0:
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



#  CATALOGUE BUILDER


def build_catalogue_from_files(lc_dir: Path, log_fn=print) -> pd.DataFrame:
    """Build catalogue by scanning per-satellite parquet files on disk.

    Falls back to CSV files if no parquet files exist (pre-conversion).
    This is accurate even after resume/append operations since it reads
    what is actually on disk rather than relying on in-memory stats.
    """
    # Prefer parquet, fall back to CSV
    pq_files = sorted(lc_dir.glob("*.parquet"))
    csv_files = sorted(lc_dir.glob("*.csv"))

    if pq_files:
        data_files = pq_files
        reader = pd.read_parquet
        fmt_name = "parquet"
    elif csv_files:
        data_files = csv_files
        reader = pd.read_csv
        fmt_name = "CSV"
    else:
        log_fn("  No satellite files found.")
        return pd.DataFrame()

    log_fn(f"  Scanning {len(data_files)} {fmt_name} satellite file(s)...")

    rows = []
    for i, fpath in enumerate(data_files, 1):
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
            mean_std = df["StdMag"].mean()
            std_std = df["StdMag"].std() if n > 1 else 0.0

            rows.append({
                "norad_id": sat_id,
                "total_observations": n,
                "num_tracks": df["Track"].nunique(),
                "num_channels": df["Channel"].nunique(),
                "first_obs": df["Datetime"].min(),
                "last_obs": df["Datetime"].max(),
                "mean_stdmag": round(mean_std, 4),
                "std_stdmag": round(std_std, 4) if pd.notna(std_std) else 0.0,
                "min_stdmag": df["StdMag"].min(),
                "max_stdmag": df["StdMag"].max(),
                "mean_distance_km": round(df["Distance_km"].mean(), 2),
                "mean_phase_deg": round(df["Phase_deg"].mean(), 2),
                "filters_used": ",".join(sorted(df["Filter"].dropna().unique())),
            })
        except Exception as e:
            log_fn(f"  [WARN] Failed to read {fpath.name}: {e}")
            continue

        if i % 500 == 0:
            log_fn(f"    ... scanned {i}/{len(data_files)}")

    catalogue = pd.DataFrame(rows)
    if not catalogue.empty:
        catalogue["obs_span_days"] = (
            (catalogue["last_obs"] - catalogue["first_obs"]).dt.total_seconds() / 86400
        )
        catalogue.sort_values("total_observations", ascending=False, inplace=True)
        catalogue.reset_index(drop=True, inplace=True)

    return catalogue



#  MAIN PROCESSING PIPELINE


def run_processing(input_dir: Path, output_dir: Path, lc_dir: Path,
                   checkpoint: CheckpointManager, manifest: DataManifest = None,
                   log_fn=print) -> dict:
    """Full streaming processing pipeline with checkpoint/resume.

    Three phases:
      1. Parse raw .txt files into per-satellite CSVs (streaming append)
      2. Convert CSVs to parquet (final format)
      3. Build catalogue from parquet files

    Returns:
        {"rows": int, "files_processed": int, "satellites": int, "converted": int}
    """
    lc_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        log_fn(f"  [ERROR] No .txt files found in {input_dir}")
        return {"rows": 0, "files_processed": 0, "satellites": 0, "converted": 0}

    log_fn(f"  Found {len(txt_files)} raw file(s) in {input_dir}\n")

    # Track existing satellite files for append mode
    # Check both CSV (working) and parquet (finished) to know which sats exist
    known_sats = set()
    for pattern in ["*.csv", "*.parquet"]:
        for f in lc_dir.glob(pattern):
            try:
                known_sats.add(int(f.stem))
            except ValueError:
                continue

    if checkpoint.completed_count > 0:
        log_fn(f"  Resuming: {checkpoint.completed_count} file(s) already processed")
        log_fn(f"  Found {len(known_sats)} existing satellite file(s) for appending\n")

    # ── Phase 1: Parse raw files into per-satellite CSVs ──
    stats = {}
    total_rows = 0
    files_processed = 0

    for i, fp in enumerate(txt_files, 1):
        if checkpoint.is_done(fp.name):
            continue

        log_fn(f"  [{i}/{len(txt_files)}] Parsing {fp.name}...")
        rows, n_sats = parse_and_route(fp, lc_dir, known_sats, stats, log_fn)
        total_rows += rows
        files_processed += 1

        if rows > 0:
            log_fn(f"    -> {rows:,} rows, {n_sats} satellite(s)  [total: {total_rows:,}]")
        else:
            log_fn(f"    -> empty / no valid rows")

        checkpoint.mark_done(fp.name)
        if manifest:
            manifest.record_processed(fp.name)

    log_fn(f"\n  This run: {total_rows:,} observations, {files_processed} file(s)")
    log_fn(f"  Total satellites with data: {len(known_sats):,}")

    # ── Phase 2: Convert CSVs to parquet ──
    log_fn("")
    converted = convert_csvs_to_parquet(lc_dir, log_fn, delete_csvs=True)

    # ── Phase 3: Build catalogue from parquet files ──
    log_fn("\n  Building catalogue from satellite files...")
    catalogue = build_catalogue_from_files(lc_dir, log_fn)
    cat_path = output_dir / "catalogue.csv"
    catalogue.to_csv(cat_path, index=False)
    log_fn(f"  Catalogue saved: {cat_path} ({len(catalogue)} satellites)")

    return {
        "rows": total_rows,
        "files_processed": files_processed,
        "satellites": len(known_sats),
        "converted": converted,
    }



#  STATUS


def show_status(input_dir: Path, output_dir: Path, lc_dir: Path,
                checkpoint: CheckpointManager):
    """Display processing status."""
    print_section("PROCESSING STATUS")

    # Raw files
    txt_files = sorted(input_dir.glob("*.txt"))
    remaining = sum(1 for f in txt_files if not checkpoint.is_done(f.name))

    # Processed files - count both formats
    pq_files = list(lc_dir.glob("*.parquet")) if lc_dir.exists() else []
    csv_files = list(lc_dir.glob("*.csv")) if lc_dir.exists() else []

    cat_path = output_dir / "catalogue.csv"
    cat_exists = cat_path.exists()

    rows = [
        ("Raw .txt files:",       f"{len(txt_files):,}"),
        ("Files processed:",      f"{checkpoint.completed_count:,}"),
        ("Files remaining:",      f"{remaining:,}"),
        ("Satellite parquet files:", f"{len(pq_files):,}"),
    ]

    if csv_files:
        rows.append(("Working CSV files:", f"{len(csv_files):,} (not yet converted)"))

    rows.append(("Catalogue exists:", "Yes" if cat_exists else "No"))

    print_summary_table(rows)

    if txt_files:
        print()
        print_status_bar("Progress", checkpoint.completed_count, len(txt_files))
    print()



#  INTERACTIVE MENU


def run_interactive(input_dir: Path, output_dir: Path, lc_dir: Path):
    """Run the interactive menu."""
    print_header("MMT-9 Processor", __version__,
                 "Process raw light curve dumps into per-satellite parquet files")

    checkpoint = CheckpointManager("mmt9_process")
    manifest = DataManifest("mmt9_raw")
    manifest.save_metadata({"directory": str(input_dir)})

    while True:
        options = [("process", "Process raw files (resume if interrupted)")]

        has_parquet = lc_dir.exists() and list(lc_dir.glob("*.parquet"))
        has_csv = lc_dir.exists() and list(lc_dir.glob("*.csv"))

        if has_parquet or has_csv:
            options.append(("rebuild", "Rebuild catalogue from existing lightcurve files"))

        # Offer conversion if there are unconverted CSVs
        if has_csv:
            n_csv = len(list(lc_dir.glob("*.csv")))
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
            result = run_processing(input_dir, output_dir, lc_dir, checkpoint,
                                    manifest=manifest)
            print_section("SUMMARY")
            print_summary_table([
                ("Observations written:", f"{result['rows']:,}"),
                ("Files processed:",      f"{result['files_processed']:,}"),
                ("Satellites:",           f"{result['satellites']:,}"),
                ("Parquet files created:", f"{result['converted']:,}"),
                ("Lightcurves:",          str(lc_dir)),
                ("Catalogue:",            str(output_dir / "catalogue.csv")),
            ])
            print()

        elif choice == "convert":
            print_section("CONVERTING CSV TO PARQUET")
            converted = convert_csvs_to_parquet(lc_dir, delete_csvs=True)
            print(f"  Converted {converted} files.\n")

        elif choice == "rebuild":
            print(f"\n  Rebuilding catalogue from lightcurve files...")
            catalogue = build_catalogue_from_files(lc_dir)
            cat_path = output_dir / "catalogue.csv"
            catalogue.to_csv(cat_path, index=False)
            print(f"  Saved: {cat_path} ({len(catalogue)} satellites)\n")

        elif choice == "status":
            show_status(input_dir, output_dir, lc_dir, checkpoint)

        elif choice == "reset":
            if prompt_confirm("Clear checkpoint? Files on disk are kept."):
                checkpoint.clear()
                print("  Checkpoint cleared.\n")



#  CLI MODE


def run_cli(args, input_dir: Path, output_dir: Path, lc_dir: Path):
    """Run in CLI mode."""
    print_header("MMT-9 Processor", __version__)
    checkpoint = CheckpointManager("mmt9_process")
    manifest = DataManifest("mmt9_raw")

    if args.status:
        show_status(input_dir, output_dir, lc_dir, checkpoint)

    elif args.rebuild_catalogue:
        print(f"  Rebuilding catalogue from lightcurve files...")
        catalogue = build_catalogue_from_files(lc_dir)
        cat_path = output_dir / "catalogue.csv"
        catalogue.to_csv(cat_path, index=False)
        print(f"  Saved: {cat_path} ({len(catalogue)} satellites)\n")

    elif args.convert:
        print(f"  Converting CSV files to parquet...")
        converted = convert_csvs_to_parquet(lc_dir, delete_csvs=not args.keep_csv)
        print(f"  Converted {converted} files.\n")

    elif args.process:
        result = run_processing(input_dir, output_dir, lc_dir, checkpoint,
                                manifest=manifest)
        print_section("SUMMARY")
        print_summary_table([
            ("Observations:", f"{result['rows']:,}"),
            ("Files:",        f"{result['files_processed']:,}"),
            ("Satellites:",   f"{result['satellites']:,}"),
            ("Converted:",    f"{result['converted']:,}"),
        ])
        print()
        sys.exit(0 if result["files_processed"] >= 0 else 1)



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="MMT-9 Light Curve Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mmt9_processor.py                      # Interactive menu
  python mmt9_processor.py --process            # Process all raw files
  python mmt9_processor.py --convert            # Convert working CSVs to parquet
  python mmt9_processor.py --rebuild-catalogue  # Rebuild from existing files
  python mmt9_processor.py --status             # Show progress
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
    parser.add_argument("--input", type=str, default=None,
                        help="Override raw input directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Override processed output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    ensure_dirs()
    input_dir = Path(args.input) if args.input else RAW_MMT9
    output_dir = Path(args.output) if args.output else PROC_MMT9
    lc_dir = output_dir / "lightcurves"

    has_action = args.process or args.rebuild_catalogue or args.status or args.convert

    try:
        if has_action:
            run_cli(args, input_dir, output_dir, lc_dir)
        else:
            run_interactive(input_dir, output_dir, lc_dir)
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Progress saved.")
        sys.exit(2)


if __name__ == "__main__":
    main()
