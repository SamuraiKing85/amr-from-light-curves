"""
Centralised Path Configuration
================================
All scripts import from here so folder names are defined exactly once.
Override any path via environment variables prefixed with SDML_.

Usage:
    from config.paths import *

    # Use paths directly
    raw_files = list(RAW_MMT9.glob("*.txt"))

    # Ensure all directories exist before running a pipeline
    ensure_dirs()
"""

from pathlib import Path
import os

__all__ = [
    "PROJECT_ROOT",
    "RAW_DIR", "RAW_MMT9", "RAW_SDLCD", "RAW_DISCOS", "RAW_TLE_BULK",
    "PROCESSED_DIR", "PROC_MMT9", "PROC_SDLCD", "PROC_DISCOS", "PROC_TLE",
    "TRAINING_DIR", "CHECKPOINT_DIR", "LOG_DIR",
    "CREDENTIALS_FILE", "CREDENTIALS_EXAMPLE",
    "ensure_dirs", "print_paths", "find_active_catalogues",
]


# ═══════════════════════════════════════════════════════════════
#  PROJECT ROOT
# ═══════════════════════════════════════════════════════════════

# Resolve to the repo root (parent of config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════
#  RAW DATA (Stage 1 output — downloads land here)
# ═══════════════════════════════════════════════════════════════

RAW_DIR      = Path(os.getenv("SDML_RAW_DIR",      str(PROJECT_ROOT / "data" / "raw")))
RAW_MMT9     = RAW_DIR / "mmt9"
RAW_SDLCD    = RAW_DIR / "sdlcd"
RAW_DISCOS   = RAW_DIR / "discos"
RAW_TLE_BULK = RAW_DIR / "tle_bulk"


# ═══════════════════════════════════════════════════════════════
#  PROCESSED DATA (Stage 2 output — catalogues + lightcurves)
# ═══════════════════════════════════════════════════════════════

PROCESSED_DIR = Path(os.getenv("SDML_PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed")))
PROC_MMT9     = PROCESSED_DIR / "mmt9"
PROC_SDLCD    = PROCESSED_DIR / "sdlcd"
PROC_DISCOS   = PROCESSED_DIR / "discos"
PROC_TLE      = PROCESSED_DIR / "tle"


# ═══════════════════════════════════════════════════════════════
#  TRAINING DATA (Stage 3 output — ML-ready datasets)
# ═══════════════════════════════════════════════════════════════

TRAINING_DIR = Path(os.getenv("SDML_TRAINING_DIR", str(PROJECT_ROOT / "data" / "training")))


# ═══════════════════════════════════════════════════════════════
#  CHECKPOINTS & LOGS
# ═══════════════════════════════════════════════════════════════

CHECKPOINT_DIR = Path(os.getenv("SDML_CHECKPOINT_DIR", str(PROJECT_ROOT / "data" / "checkpoints")))
LOG_DIR        = Path(os.getenv("SDML_LOG_DIR",        str(PROJECT_ROOT / "data" / "logs")))


# ═══════════════════════════════════════════════════════════════
#  CREDENTIALS
# ═══════════════════════════════════════════════════════════════

CREDENTIALS_FILE    = PROJECT_ROOT / "config" / "credentials.json"
CREDENTIALS_EXAMPLE = PROJECT_ROOT / "config" / "credentials.example.json"


# ═══════════════════════════════════════════════════════════════
#  DIRECTORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════

# Every directory the pipeline expects to exist
_ALL_DIRS = [
    RAW_MMT9,
    RAW_SDLCD,
    RAW_DISCOS,
    RAW_TLE_BULK,
    PROC_MMT9 / "lightcurves",
    PROC_SDLCD / "lightcurves",
    PROC_DISCOS,
    PROC_TLE / "tle_histories",
    TRAINING_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
]


def ensure_dirs() -> None:
    """Create all data directories if they don't exist.

    Safe to call multiple times. Called automatically at the start
    of every pipeline script.
    """
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def print_paths() -> None:
    """Print all configured paths and whether they exist. Useful for debugging."""
    pairs = [
        ("Project root",   PROJECT_ROOT),
        ("Raw MMT-9",      RAW_MMT9),
        ("Raw SDLCD",      RAW_SDLCD),
        ("Raw DISCOS",     RAW_DISCOS),
        ("Raw TLE bulk",   RAW_TLE_BULK),
        ("Processed MMT-9", PROC_MMT9),
        ("Processed SDLCD", PROC_SDLCD),
        ("Processed DISCOS", PROC_DISCOS),
        ("Processed TLE",  PROC_TLE),
        ("Training data",  TRAINING_DIR),
        ("Checkpoints",    CHECKPOINT_DIR),
        ("Logs",           LOG_DIR),
        ("Credentials",    CREDENTIALS_FILE),
    ]
    max_label = max(len(label) for label, _ in pairs)
    for label, path in pairs:
        exists = path.exists()
        marker = "OK" if exists else "  "
        print(f"  [{marker:>2s}]  {label:<{max_label}s}  {path}")


if __name__ == "__main__":
    # Running directly prints diagnostic info
    print("\n  Space Debris ML — Path Configuration")
    print("  " + "=" * 50)
    print_paths()
    print()

    # Check for env var overrides
    overrides = [k for k in os.environ if k.startswith("SDML_")]
    if overrides:
        print(f"  Active overrides: {', '.join(overrides)}")
    else:
        print("  No environment variable overrides active.")
    print()
    
def find_active_catalogues() -> list[Path]:
    """Auto-detect existing catalogue.csv files from processed directories."""
    catalogues = []
    
    mmt9_cat = PROC_MMT9 / "catalogue.csv"
    if mmt9_cat.exists():
        catalogues.append(mmt9_cat)
        
    sdlcd_cat = PROC_SDLCD / "catalogue.csv"
    if sdlcd_cat.exists():
        catalogues.append(sdlcd_cat)
        
    return catalogues
