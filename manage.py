"""
Space Debris ML - Project Manager
=====================================
Central dashboard for checking pipeline status, cleaning up raw data,
and managing credentials. Run this anytime after initial setup.

Usage:
    python manage.py                  # Interactive menu
    python manage.py --status         # Pipeline status overview
    python manage.py --cleanup        # Show what can be deleted
    python manage.py --credentials    # Check/set up credentials
"""

__version__ = "2.0.0"

import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.paths import (
    RAW_MMT9, RAW_SDLCD, RAW_DISCOS, RAW_TLE_BULK,
    PROC_MMT9, PROC_SDLCD, PROC_DISCOS, PROC_TLE,
    TRAINING_DIR, CREDENTIALS_FILE, CHECKPOINT_DIR,
    ensure_dirs,
)
from common.menu import (
    print_header, prompt_choice, prompt_confirm,
    print_section, print_summary_table, print_status_bar,
)
from common.manifest import DataManifest



#  HELPERS


def _count_files(directory: Path, pattern: str = "*") -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.glob(pattern) if f.is_file())


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def _dir_size_mb(directory: Path, pattern: str = "*") -> float:
    if not directory.exists():
        return 0.0
    return sum(f.stat().st_size for f in directory.glob(pattern) if f.is_file()) / (1024 * 1024)


def _extract_year(filename: str):
    match = re.search(r'(20\d{2}|19\d{2})', filename)
    return int(match.group(1)) if match else None


def _check_items(checks: list) -> bool:
    """Print check results. checks: list of (label, ok, detail). Returns all_ok."""
    all_ok = True
    for label, ok, detail in checks:
        status = "  OK" if ok else "MISS"
        print(f"    [{status}]  {label:<35s}  {detail}")
        if not ok:
            all_ok = False
    return all_ok



#  PIPELINE STATUS


def run_pipeline_status():
    """Check all pipeline stages and report completeness."""

    stages_ok = []

    # ── Credentials ──
    print_section("CREDENTIALS")
    cred_ok = CREDENTIALS_FILE.exists()
    cred_detail = ""
    if cred_ok:
        try:
            with open(CREDENTIALS_FILE) as f:
                creds = json.load(f)
            services = []
            if creds.get("spacetrack", {}).get("username"):
                services.append("Space-Track")
            if creds.get("discos", {}).get("token"):
                services.append("DISCOS")
            cred_detail = ", ".join(services) if services else "empty"
        except Exception:
            cred_detail = "malformed"
    else:
        cred_detail = "run: python manage.py --credentials"

    print(f"    [{'  OK' if cred_ok else 'MISS'}]  credentials.json"
          f"{'':>20s}  {cred_detail}")

    # ── Stage 1: Raw Data ──
    print_section("STAGE 1: RAW DATA (1_acquire/)")

    n_mmt9 = _count_files(RAW_MMT9, "*.txt")
    n_sdlcd_db = 1 if (RAW_SDLCD / "database.json").exists() else 0
    n_sdlcd_lc = _count_files(RAW_SDLCD / "lightcurves", "*.txt")
    n_discos = 1 if (RAW_DISCOS / "discos_catalogue.csv").exists() else 0
    discos_mb = _file_size_mb(RAW_DISCOS / "discos_catalogue.csv")
    n_tle_bulk = _count_files(RAW_TLE_BULK, "*.txt")

    s1 = _check_items([
        ("MMT-9 raw files", n_mmt9 > 0, f"{n_mmt9} .txt files"),
        ("SDLCD database.json", n_sdlcd_db > 0,
         "found" if n_sdlcd_db else "run sdlcd_downloader.py"),
        ("SDLCD light curves", n_sdlcd_lc > 0, f"{n_sdlcd_lc} files"),
        ("DISCOS catalogue", n_discos > 0,
         f"{discos_mb:.1f} MB" if n_discos else "run discos_downloader.py"),
        ("TLE bulk files", n_tle_bulk > 0,
         f"{n_tle_bulk} yearly files" if n_tle_bulk else "(optional)"),
    ])
    stages_ok.append(("Stage 1", s1))

    # ── Stage 2: Processed Data ──
    print_section("STAGE 2: PROCESSED DATA (2_process/)")

    mmt9_cat = PROC_MMT9 / "catalogue.csv"
    mmt9_lc = _count_files(PROC_MMT9 / "lightcurves", "*.parquet")
    sdlcd_cat = PROC_SDLCD / "catalogue.csv"
    sdlcd_lc = _count_files(PROC_SDLCD / "lightcurves", "*.parquet")
    discos_proc = (PROC_DISCOS / "discos_catalogue.csv").exists()
    tle_index = (PROC_TLE / "tle_index.csv").exists()
    tle_histories = _count_files(PROC_TLE / "tle_histories", "*.parquet")

    s2 = _check_items([
        ("MMT-9 catalogue", mmt9_cat.exists(),
         f"{mmt9_lc} satellite CSVs" if mmt9_cat.exists() else "run mmt9_processor.py"),
        ("SDLCD catalogue", sdlcd_cat.exists(),
         f"{sdlcd_lc} satellite CSVs" if sdlcd_cat.exists() else "run sdlcd_processor.py"),
        ("DISCOS processed", discos_proc,
         "found" if discos_proc else "run discos_downloader.py"),
        ("TLE index", tle_index,
         f"{tle_histories} satellite histories" if tle_index else "run tle_manager.py"),
    ])
    stages_ok.append(("Stage 2", s2))

    # ── Stage 3: Training Data ──
    print_section("STAGE 3: TRAINING DATA (3_prepare/)")

    # Check for metadata - new format first, then legacy
    meta_path = TRAINING_DIR / "normalisation_params.json"
    if not meta_path.exists():
        meta_path = TRAINING_DIR / "metadata.json"

    has_train = (TRAINING_DIR / "train.parquet").exists()
    has_val = (TRAINING_DIR / "val.parquet").exists()
    has_test = (TRAINING_DIR / "test.parquet").exists()
    has_full = (TRAINING_DIR / "full_dataset.parquet").exists()

    train_detail = ""
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            # New format stores total_tracks; legacy stores n_train/n_val/n_test
            if "total_tracks" in meta:
                train_detail = f"{meta['total_tracks']:,} tracks total"
            elif "n_train" in meta:
                train_detail = (f"train={meta['n_train']}, val={meta['n_val']}, "
                                f"test={meta['n_test']}")
            else:
                train_detail = "found"
        except Exception:
            train_detail = "metadata unreadable"
    else:
        train_detail = "run training_pipeline.py"

    # Count tracks in parquet if available
    split_details = {}
    for split_name in ["train", "val", "test"]:
        pq = TRAINING_DIR / f"{split_name}.parquet"
        if pq.exists():
            try:
                n = len(pd.read_parquet(pq, columns=["norad_id"]))
                split_details[split_name] = f"{n:,} tracks"
            except Exception:
                split_details[split_name] = "found"
        else:
            split_details[split_name] = "missing"

    s3 = _check_items([
        ("Pipeline metadata", meta_path.exists(), train_detail),
        ("Train split", has_train, split_details.get("train", "missing")),
        ("Val split", has_val, split_details.get("val", "missing")),
        ("Test split", has_test, split_details.get("test", "missing")),
        ("Full dataset", has_full, "found" if has_full else "(optional)"),
    ])
    stages_ok.append(("Stage 3", s3))

    # ── Stage 4: Trained Models ──
    print_section("STAGE 4: TRAINED MODELS (4_train/)")

    models_dir = TRAINING_DIR / "models"
    model_checks = []
    for mt in ["fusion", "lc_only", "orbital_only"]:
        md = models_dir / mt
        has_best = (md / "best_model.pt").exists()
        has_final = (md / "final_model.pt").exists()
        has_results = (md / "test_results.json").exists()
        found = has_best or has_final

        detail = ""
        if has_results:
            try:
                with open(md / "test_results.json") as f:
                    res = json.load(f)
                r2 = res.get("regression", {}).get("r2", "?")
                f1 = res.get("classification", {}).get("macro_f1",
                     res.get("classification", {}).get("accuracy", "?"))
                detail = f"R\u00b2={r2}, F1={f1}"
            except Exception:
                detail = "trained"
        elif found:
            detail = "trained, not yet evaluated"
        else:
            detail = "not trained"

        model_checks.append((f"{mt} model", found, detail))

    s4 = _check_items(model_checks)
    stages_ok.append(("Stage 4", s4))

    # ── Summary ──
    print(f"\n  {'═' * 54}")
    print(f"  SUMMARY")
    print(f"  {'═' * 54}")

    all_complete = True
    for stage_name, ok in stages_ok:
        marker = "\u2713" if ok else "\u2717"
        status = "OK" if ok else "INCOMPLETE"
        print(f"    {marker}  {stage_name:<12s}  {status}")
        if not ok:
            all_complete = False

    if all_complete:
        print("\n  All pipeline stages complete!")
    else:
        for stage_name, ok in stages_ok:
            if not ok:
                stage_num = stage_name.split()[1]
                print(f"\n  Next step: complete {stage_name}")
                hints = {
                    "1": "  Run the downloaders in 1_acquire/",
                    "2": "  Run the processors in 2_process/",
                    "3": "  Run: python 3_prepare/training_pipeline.py --prepare",
                    "4": "  Run: python 4_train/train.py",
                }
                print(hints.get(stage_num, ""))
                break

    print()
    return all_complete



#  DISK USAGE & CLEANUP


CLEANUP_SOURCES = {
    "mmt9": {
        "name": "MMT-9 raw light curve dumps",
        "raw_dir": RAW_MMT9,
        "pattern": "*.txt",
        "manifest_name": "mmt9_raw",
        "processed_dir": PROC_MMT9 / "lightcurves",
        "explanation": (
            "These raw .txt dump files contain mixed satellite observations.\n"
            "  They have been split into individual per-satellite CSV files in\n"
            "  data/processed/mmt9/lightcurves/ by the MMT-9 processor.\n"
            "  All observation data has been extracted - the raw files are only\n"
            "  needed if you want to reprocess from scratch."
        ),
    },
    "sdlcd": {
        "name": "SDLCD raw light curve files",
        "raw_dir": RAW_SDLCD / "lightcurves",
        "pattern": "*.txt",
        "manifest_name": "sdlcd_raw",
        "processed_dir": PROC_SDLCD / "lightcurves",
        "explanation": (
            "These raw SDLCD _DATA.txt files have been converted to\n"
            "  MMT-9 compatible per-satellite CSVs in data/processed/sdlcd/lightcurves/.\n"
            "  The converted files contain all the original data plus standardised\n"
            "  columns for the ML pipeline."
        ),
    },
    "tle_bulk": {
        "name": "Bulk TLE yearly files",
        "raw_dir": RAW_TLE_BULK,
        "pattern": "*.txt",
        "manifest_name": None,
        "processed_dir": PROC_TLE / "tle_histories",
        "explanation": (
            "These are the large yearly TLE text files downloaded from\n"
            "  Space-Track's Data Cloud (e.g., tle2014.txt, tle2015.txt).\n"
            "  The TLE manager has already extracted the relevant satellites\n"
            "  into individual per-satellite CSV files in data/processed/tle/tle_histories/.\n"
            "  The bulk files can be very large (several GB each) and are safe\n"
            "  to delete once the TLE manager has finished processing them."
        ),
    },
}


def _scan_cleanup_source(source_key: str) -> dict:
    """Scan a data source and return cleanup info."""
    source = CLEANUP_SOURCES[source_key]
    raw_dir = source["raw_dir"]
    pattern = source["pattern"]

    result = {
        "name": source["name"],
        "raw_dir": raw_dir,
        "files_on_disk": [],
        "total_size_mb": 0.0,
        "deletable": [],
        "deletable_size_mb": 0.0,
        "not_deletable": [],
        "by_year": defaultdict(list),
        "explanation": source["explanation"],
    }

    if not raw_dir.exists():
        return result

    files = sorted(f for f in raw_dir.glob(pattern) if f.is_file())
    result["files_on_disk"] = files
    result["total_size_mb"] = sum(f.stat().st_size for f in files) / (1024 * 1024)

    if source_key == "tle_bulk":
        tle_dir = source["processed_dir"]
        has_processed_tles = tle_dir.exists() and len(list(tle_dir.glob("*.parquet"))) > 50

        bulk_progress_path = PROC_TLE / "tle_histories" / ".bulk_progress.json"
        completed_bulk = set()
        if bulk_progress_path.exists():
            try:
                completed_bulk = set(json.loads(bulk_progress_path.read_text()))
            except Exception:
                pass

        for f in files:
            year = _extract_year(f.name)
            if year:
                result["by_year"][year].append(f)
            if f.name in completed_bulk or has_processed_tles:
                result["deletable"].append(f)
            else:
                result["not_deletable"].append(f)
    else:
        manifest = DataManifest(source["manifest_name"])
        for f in files:
            year = _extract_year(f.name)
            if year:
                result["by_year"][year].append(f)
            if manifest.was_processed(f.name):
                result["deletable"].append(f)
            else:
                result["not_deletable"].append(f)

    result["deletable_size_mb"] = sum(
        f.stat().st_size for f in result["deletable"]
    ) / (1024 * 1024)

    return result


def _print_disk_overview(all_info: dict):
    """Print summary overview of all sources."""
    print_section("DISK USAGE OVERVIEW")

    total_files = 0
    total_mb = 0.0
    total_deletable = 0
    total_deletable_mb = 0.0

    for key, info in all_info.items():
        n = len(info["files_on_disk"])
        d = len(info["deletable"])
        if n > 0:
            print(f"  {info['name']:<40s}  {n:>5,} files  {info['total_size_mb']:>8.1f} MB  "
                  f"({d:,} deletable, {info['deletable_size_mb']:.1f} MB)")
        total_files += n
        total_mb += info["total_size_mb"]
        total_deletable += d
        total_deletable_mb += info["deletable_size_mb"]

    if total_files > 0:
        print(f"  {'─' * 70}")
        print(f"  {'TOTAL':<40s}  {total_files:>5,} files  {total_mb:>8.1f} MB  "
              f"({total_deletable:,} deletable, {total_deletable_mb:.1f} MB)")
    else:
        print(f"  No raw data files found.")
    print()

    return total_deletable, total_deletable_mb


def _delete_files(files: list, source_name: str):
    """Delete a list of files with confirmation."""
    if not files:
        print(f"  No files to delete.\n")
        return

    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

    print(f"\n  About to delete {len(files)} file(s) ({total_mb:.1f} MB)")
    print(f"  from: {source_name}")

    if prompt_confirm(f"\n  Proceed with deletion?"):
        deleted = 0
        for f in files:
            try:
                f.unlink()
                deleted += 1
            except OSError as e:
                print(f"  [WARN] Could not delete {f.name}: {e}")
        print(f"  Deleted {deleted} file(s), freed {total_mb:.1f} MB.\n")
    else:
        print(f"  No files deleted.\n")


def run_cleanup_menu(all_info: dict):
    """Run the cleanup sub-menu."""
    while True:
        options = [("overview", "Refresh disk usage overview")]

        for key, info in all_info.items():
            if info["deletable"]:
                mb = info["deletable_size_mb"]
                options.append((
                    key,
                    f"Clean {info['name']} ({len(info['deletable'])} files, {mb:.1f} MB)"
                ))

        has_years = any(info["by_year"] for info in all_info.values() if info["deletable"])
        if has_years:
            options.append(("by_year", "Delete by year (across all sources)"))

        total_deletable = sum(len(i["deletable"]) for i in all_info.values())
        total_deletable_mb = sum(i["deletable_size_mb"] for i in all_info.values())
        if total_deletable > 0:
            options.append(("all", f"Delete ALL safe files ({total_deletable} files, "
                                   f"{total_deletable_mb:.1f} MB)"))

        options.append(("back", "Back to main menu"))

        choice = prompt_choice("Cleanup options:", options)

        if choice is None or choice == "back":
            break

        elif choice == "overview":
            all_info.update({key: _scan_cleanup_source(key) for key in CLEANUP_SOURCES})
            _print_disk_overview(all_info)

        elif choice in CLEANUP_SOURCES:
            info = all_info[choice]

            print_section(info["name"].upper())
            print_summary_table([
                ("Location:",          str(info["raw_dir"])),
                ("Files on disk:",     f"{len(info['files_on_disk']):,}"),
                ("Total size:",        f"{info['total_size_mb']:.1f} MB"),
                ("Safe to delete:",    f"{len(info['deletable']):,} ({info['deletable_size_mb']:.1f} MB)"),
                ("Not yet processed:", f"{len(info['not_deletable']):,}"),
            ])

            if info["by_year"]:
                print(f"\n  By year:")
                for year in sorted(info["by_year"].keys()):
                    files = info["by_year"][year]
                    ymb = sum(f.stat().st_size for f in files) / (1024 * 1024)
                    n_del = sum(1 for f in files if f in info["deletable"])
                    status = "all processed" if n_del == len(files) else f"{n_del}/{len(files)} processed"
                    print(f"    {year}:  {len(files)} file(s), {ymb:.1f} MB  ({status})")

            if info["deletable"]:
                print(f"\n  WHY THESE ARE SAFE TO DELETE:")
                print(f"  {info['explanation']}")
                print()

                if info["by_year"] and len(info["by_year"]) > 1:
                    year_opts = []
                    for year in sorted(info["by_year"].keys()):
                        yf = [f for f in info["by_year"][year] if f in info["deletable"]]
                        if yf:
                            ymb = sum(f.stat().st_size for f in yf) / (1024 * 1024)
                            year_opts.append((str(year), f"{year}: {len(yf)} files ({ymb:.1f} MB)"))
                    year_opts.append(("all_src", f"All {len(info['deletable'])} deletable files"))
                    year_opts.append(("skip", "Don't delete anything"))

                    yr = prompt_choice("Delete by year or all?", year_opts)
                    if yr and yr == "all_src":
                        _delete_files(info["deletable"], info["name"])
                    elif yr and yr != "skip":
                        yf = [f for f in info["by_year"][int(yr)] if f in info["deletable"]]
                        _delete_files(yf, f"{info['name']} ({yr})")
                else:
                    _delete_files(info["deletable"], info["name"])

                all_info[choice] = _scan_cleanup_source(choice)
            else:
                print()

        elif choice == "by_year":
            year_groups = defaultdict(list)
            for key, info in all_info.items():
                for f in info["deletable"]:
                    year = _extract_year(f.name)
                    if year:
                        year_groups[year].append((f, info["name"]))

            if not year_groups:
                print(f"\n  No year-grouped files available.\n")
                continue

            year_opts = []
            for year in sorted(year_groups.keys()):
                files = year_groups[year]
                mb = sum(f.stat().st_size for f, _ in files) / (1024 * 1024)
                srcs = ", ".join(sorted(set(n for _, n in files)))
                year_opts.append((str(year), f"{year}: {len(files)} files ({mb:.1f} MB) - {srcs}"))
            year_opts.append(("skip", "Back"))

            yr = prompt_choice("Select year to delete:", year_opts)
            if yr and yr != "skip":
                files = [f for f, _ in year_groups[int(yr)]]
                _delete_files(files, f"all sources ({yr})")
                all_info.update({key: _scan_cleanup_source(key) for key in CLEANUP_SOURCES})

        elif choice == "all":
            print(f"\n  This will delete ALL processed raw files across all sources:")
            for key, info in all_info.items():
                if info["deletable"]:
                    print(f"    {info['name']}: {len(info['deletable'])} files "
                          f"({info['deletable_size_mb']:.1f} MB)")

            print(f"\n  WHY THESE ARE SAFE:")
            print(f"  Each file has been fully processed by the corresponding")
            print(f"  processor script. The processed data (per-satellite CSVs,")
            print(f"  catalogues, TLE histories) is stored separately and is")
            print(f"  not affected by deleting raw files. The manifest system")
            print(f"  remembers every file that existed, so features like")
            print(f"  'check for new files' continue to work correctly.")

            if prompt_confirm(f"\n  Delete all {total_deletable} files "
                              f"({total_deletable_mb:.1f} MB)?"):
                for key, info in all_info.items():
                    for f in info["deletable"]:
                        try:
                            f.unlink()
                        except OSError:
                            pass
                    if info["deletable"]:
                        print(f"  Cleaned: {info['name']}")
                print(f"\n  Freed {total_deletable_mb:.1f} MB total.\n")
                all_info.update({key: _scan_cleanup_source(key) for key in CLEANUP_SOURCES})
            else:
                print(f"  No files deleted.\n")



#  CREDENTIALS


def run_credentials():
    """Check and interactively set up credentials."""
    print_section("CREDENTIALS")
    try:
        from common.credentials import check_credentials_status, setup_credentials_interactive
        status = check_credentials_status()

        for service, ok in status.items():
            label = {"spacetrack": "Space-Track", "discos": "DISCOS"}.get(service, service)
            st = "  OK" if ok else "MISS"
            print(f"  [{st}]  {label}")

        if not all(status.values()):
            print()
            setup_credentials_interactive()
        else:
            print(f"\n  All credentials are configured.\n")

    except ImportError:
        print(f"  Could not load credential module. Run setup.py first.\n")



#  INTERACTIVE MENU


def run_interactive():
    """Main interactive menu."""
    print_header("Project Manager", __version__,
                 "Pipeline status, data cleanup, and credentials")

    while True:
        options = [
            ("status",      "Check pipeline status (all stages)"),
            ("cleanup",     "Manage disk space (clean up raw data)"),
            ("credentials", "Check / set up API credentials"),
            ("quit",        "Quit"),
        ]

        choice = prompt_choice("What would you like to do?", options)

        if choice is None or choice == "quit":
            print("\n  Goodbye!\n")
            break

        elif choice == "status":
            run_pipeline_status()

        elif choice == "cleanup":
            print(f"\n  Scanning data directories...\n")
            all_info = {key: _scan_cleanup_source(key) for key in CLEANUP_SOURCES}
            _print_disk_overview(all_info)
            run_cleanup_menu(all_info)

        elif choice == "credentials":
            run_credentials()



#  CLI MODE


def run_cli(args):
    """Run in CLI mode."""
    print_header("Project Manager", __version__)

    if args.status:
        run_pipeline_status()

    elif args.cleanup:
        all_info = {key: _scan_cleanup_source(key) for key in CLEANUP_SOURCES}
        _print_disk_overview(all_info)
        for key, info in all_info.items():
            if info["files_on_disk"]:
                print_section(info["name"].upper())
                print_summary_table([
                    ("Files on disk:",   f"{len(info['files_on_disk']):,}"),
                    ("Total size:",      f"{info['total_size_mb']:.1f} MB"),
                    ("Safe to delete:",  f"{len(info['deletable']):,} ({info['deletable_size_mb']:.1f} MB)"),
                ])
                print()

    elif args.credentials:
        run_credentials()



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="Space Debris ML - Project Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage.py                  # Interactive menu
  python manage.py --status         # Pipeline status overview
  python manage.py --cleanup        # Show what can be deleted
  python manage.py --credentials    # Check/set up API credentials
        """,
    )
    parser.add_argument("--status", action="store_true",
                        help="Show pipeline status")
    parser.add_argument("--cleanup", action="store_true",
                        help="Show disk usage and cleanup options")
    parser.add_argument("--credentials", action="store_true",
                        help="Check and set up API credentials")

    args = parser.parse_args()

    ensure_dirs()

    has_action = args.status or args.cleanup or args.credentials

    try:
        if has_action:
            run_cli(args)
        else:
            run_interactive()
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        sys.exit(2)


if __name__ == "__main__":
    main()
