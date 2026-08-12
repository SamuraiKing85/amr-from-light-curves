"""
Space Debris ML - Setup & Dependency Checker
==============================================
Run this FIRST after cloning the repository.

Uses ONLY the Python standard library - no external dependencies required.

Usage:
    python setup.py                  # Interactive: check everything, prompt to install
    python setup.py --install        # Auto-install all missing packages
    python setup.py --check          # Check only, don't install anything
    python setup.py --gpu            # Also run GPU/CUDA diagnostics
    python setup.py --stage acquire  # Check only stage-specific dependencies
"""

__version__ = "2.0.0"

import sys
import subprocess
import importlib
import shutil
import os
import json
import platform
import argparse
from pathlib import Path


#  PYTHON VERSION


MIN_PYTHON = (3, 9)


#  REQUIREMENTS DEFINITION

#
#  (import_name, pip_name, min_version, stages[], required)
#
#  import_name:  what Python imports (e.g. "bs4")
#  pip_name:     what pip installs (e.g. "beautifulsoup4")
#  min_version:  minimum version string, or None for any
#  stages:       which pipeline stages need this package
#  required:     True = pipeline won't work without it
#                False = optional enhancement
#

REQUIREMENTS = [
    # ── Core (all stages) ──
    ("numpy",       "numpy",          "1.24.0",  ["core"],               True),
    ("pandas",      "pandas",         "2.0.0",   ["core"],               True),
    ("pyarrow",     "pyarrow",        "12.0.0",  ["core"],               True),

    # ── Stage 1: Acquire ──
    ("requests",    "requests",       "2.28.0",  ["acquire"],            True),
    ("bs4",         "beautifulsoup4", "4.12.0",  ["acquire"],            True),
    ("tqdm",        "tqdm",           "4.65.0",  ["acquire"],            True),

    # ── Stage 3-4: Prepare & Train ──
    ("torch",       "torch",          "2.0.0",   ["prepare", "train"],   True),
    ("sklearn",     "scikit-learn",   "1.3.0",   ["prepare", "train"],   True),
    ("matplotlib",  "matplotlib",     "3.7.0",   ["train", "explorer"],  True),
    ("seaborn",     "seaborn",        "0.12.0",  ["train"],              False),

    # ── Explorer GUI ──
    ("openpyxl",    "openpyxl",       "3.1.0",   ["explorer"],           False),
]

# Map import names to pip distribution names for metadata lookup
_DIST_MAP = {
    "bs4":     "beautifulsoup4",
    "sklearn": "scikit-learn",
    "cv2":     "opencv-python",
    "PIL":     "Pillow",
}



#  VERSION UTILITIES


def parse_version(v):
    """Parse '1.24.3' into (1, 24, 3) for comparison."""
    parts = []
    for p in v.split("."):
        # Handle versions like '2.0.0a1' - strip non-numeric suffix
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts)


def get_installed_version(import_name):
    """Get a package's installed version without heavy imports."""
    # Try importlib.metadata first (fast, no side effects)
    try:
        from importlib.metadata import version, PackageNotFoundError
        dist_name = _DIST_MAP.get(import_name, import_name)
        try:
            return version(dist_name)
        except PackageNotFoundError:
            pass
    except ImportError:
        pass

    # Fallback: actually import and check __version__
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None



#  SYSTEM CHECKS


def check_python():
    """Verify Python version meets minimum requirement."""
    current = sys.version_info[:2]
    ok = current >= MIN_PYTHON
    status = "  OK" if ok else "FAIL"
    print(f"  [{status}]  Python {current[0]}.{current[1]}"
          f"  (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    return ok


def check_pip():
    """Verify pip is available."""
    ok = shutil.which("pip") is not None or shutil.which("pip3") is not None
    status = "  OK" if ok else "FAIL"
    if ok:
        # Get pip version
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True, text=True, timeout=10
            )
            ver = result.stdout.split()[1] if result.returncode == 0 else "?"
            print(f"  [{status}]  pip {ver}")
        except Exception:
            print(f"  [{status}]  pip (version unknown)")
    else:
        print(f"  [{status}]  pip NOT FOUND")
        print("         Install: https://pip.pypa.io/en/stable/installation/")
    return ok


def check_git():
    """Check if git is available (optional)."""
    ok = shutil.which("git") is not None
    status = "  OK" if ok else "    "
    if ok:
        try:
            result = subprocess.run(
                ["git", "--version"], capture_output=True, text=True, timeout=10
            )
            ver = result.stdout.strip().replace("git version ", "") if result.returncode == 0 else "?"
            print(f"  [{status}]  git {ver}")
        except Exception:
            print(f"  [{status}]  git (version unknown)")
    else:
        print(f"  [{status}]  git not found (optional - recommended for version control)")
    return ok



#  PACKAGE CHECKS


def check_packages(stage_filter=None):
    """Check all required packages.

    Args:
        stage_filter: If set, only check packages for this stage
                      (e.g., "acquire", "train"). None checks all.

    Returns:
        (installed, outdated, missing) - lists of package info tuples.
    """
    installed = []
    outdated = []
    missing = []

    # Deduplicate by import_name
    seen = set()

    for import_name, pip_name, min_ver, stages, required in REQUIREMENTS:
        if import_name in seen:
            continue

        # Stage filter
        if stage_filter and stage_filter != "all":
            if stage_filter not in stages and "core" not in stages:
                continue

        seen.add(import_name)
        ver = get_installed_version(import_name)

        if ver is None:
            tag = "MISS" if required else "SKIP"
            req_label = "required" if required else "optional"
            print(f"  [{tag:>4s}]  {import_name:<15s}  not installed ({req_label})")
            missing.append((import_name, pip_name, min_ver, stages, required))

        elif min_ver and ver != "unknown" and parse_version(ver) < parse_version(min_ver):
            print(f"  [ OLD]  {import_name:<15s}  {ver:<12s}  (need >= {min_ver})")
            outdated.append((import_name, ver, min_ver, pip_name, stages))

        else:
            ver_display = ver if ver != "unknown" else "installed"
            print(f"  [  OK]  {import_name:<15s}  {ver_display}")
            installed.append((import_name, ver, stages))

    return installed, outdated, missing



#  GPU CHECK


def check_gpu():
    """Check GPU availability and CUDA/MPS support."""
    info = {"available": False, "type": "cpu", "name": None, "details": {}}

    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["type"] = "cuda"
            info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            mem = mem / (1024 ** 3)
            info["details"] = {
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "memory_gb": round(mem, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            }
            print(f"  [  OK]  CUDA GPU: {info['name']}")
            print(f"          {mem:.1f} GB VRAM | CUDA {torch.version.cuda} "
                  f"| Compute {props.major}.{props.minor}")

            if torch.cuda.device_count() > 1:
                print(f"          {torch.cuda.device_count()} GPU(s) available")

            # Check mixed precision support
            if props.major >= 7:
                print(f"          Mixed precision (float16): supported")
            else:
                print(f"          Mixed precision (float16): not efficient on this GPU")

            # Check cuDNN
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                try:
                    print(f"          cuDNN: {torch.backends.cudnn.version()}")
                except RuntimeError:
                    print(f"          cuDNN: incompatible with this GPU (not a problem for CPU training)")

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["available"] = True
            info["type"] = "mps"
            info["name"] = "Apple Silicon (MPS)"
            print(f"  [  OK]  Apple Silicon MPS available")
            print(f"          Mixed precision (float16): supported")

        else:
            print(f"  [INFO]  No GPU detected - CPU will be used for training")
            print(f"          Training will be slower but fully functional")

    except ImportError:
        print(f"  [SKIP]  GPU check skipped (PyTorch not installed yet)")
        print(f"          Install PyTorch first, then re-run with --gpu")

    return info



#  CREDENTIALS CHECK


def check_credentials():
    """Check and optionally set up credentials interactively."""
    # Import the shared credential module (stdlib only - always works)
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from common.credentials import check_credentials_status, setup_credentials_interactive
        status = check_credentials_status()

        all_ok = all(status.values())
        for service, ok in status.items():
            label = {"spacetrack": "Space-Track", "discos": "DISCOS"}.get(service, service)
            st = "  OK" if ok else "MISS"
            print(f"  [{st}]  {label}")

        if not all_ok:
            try:
                answer = input("\n  Set up missing credentials now? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
                print()

            if answer not in ("n", "no"):
                setup_credentials_interactive()

        return all_ok

    except ImportError:
        # Fallback: just check if file exists
        cred_path = Path(__file__).parent / "config" / "credentials.json"
        if cred_path.exists():
            print(f"  [  OK]  credentials.json exists")
            return True
        else:
            print(f"  [MISS]  credentials.json not found")
            return False



#  DIRECTORY CHECK


def check_directories():
    """Report which data directories exist and have content."""
    project_root = Path(__file__).parent

    dirs_to_check = [
        ("data/raw/mmt9",                "MMT-9 raw downloads"),
        ("data/raw/sdlcd",               "SDLCD raw downloads"),
        ("data/raw/discos",              "DISCOS raw data"),
        ("data/raw/tle_bulk",            "Bulk TLE files"),
        ("data/processed/mmt9",          "Processed MMT-9"),
        ("data/processed/sdlcd",         "Processed SDLCD"),
        ("data/processed/discos",        "Processed DISCOS"),
        ("data/processed/tle",           "Processed TLE"),
        ("data/training",                "Training datasets"),
    ]

    for rel_path, label in dirs_to_check:
        full = project_root / rel_path
        if full.exists():
            items = list(full.iterdir())
            # Don't count hidden files or empty subdirectories
            real_items = [i for i in items if not i.name.startswith(".")]
            if real_items:
                print(f"  [  OK]  {label:<22s}  ({len(real_items)} items)")
            else:
                print(f"  [    ]  {label:<22s}  (empty)")
        else:
            print(f"  [    ]  {label:<22s}  (not created yet)")



#  DIRECTORY CREATION


def create_directories():
    """Create the full data directory structure."""
    project_root = Path(__file__).parent

    dirs = [
        "data/raw/mmt9",
        "data/raw/sdlcd",
        "data/raw/discos",
        "data/raw/tle_bulk",
        "data/processed/mmt9/lightcurves",
        "data/processed/sdlcd/lightcurves",
        "data/processed/discos",
        "data/processed/tle/tle_histories",
        "data/training",
        "data/checkpoints",
        "data/logs",
    ]

    created = 0
    for d in dirs:
        full = project_root / d
        if not full.exists():
            full.mkdir(parents=True, exist_ok=True)
            created += 1

    if created > 0:
        print(f"  Created {created} data directories")
    else:
        print(f"  All directories already exist")



#  PEP 668 DETECTION


def _check_externally_managed():
    """Check if Python is externally managed (PEP 668, Debian/Ubuntu).

    These systems block pip install without --break-system-packages.
    Returns True if the flag is needed.
    """
    import sysconfig
    marker = Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED"
    return marker.exists()



#  INSTALLER


def install_packages(packages, upgrade=False):
    """Install packages using pip.

    Args:
        packages: list of (pip_name, min_version) tuples.
        upgrade:  if True, upgrade existing packages.

    Returns:
        True if all installations succeeded.
    """
    if not packages:
        return True

    # Separate torch from everything else
    torch_pkgs = [(n, v) for n, v in packages if n == "torch"]
    other_pkgs = [(n, v) for n, v in packages if n != "torch"]

    success = True

    # Detect PEP 668 externally-managed environments (Debian/Ubuntu)
    break_flag = []
    if _check_externally_managed():
        break_flag = ["--break-system-packages"]
        print(f"  [INFO]  Externally-managed Python detected - using --break-system-packages")

    # Base pip command used for all installs
    pip_base = [sys.executable, "-m", "pip", "install"] + break_flag
    if upgrade:
        pip_base.append("--upgrade")

    # ── Install non-torch packages ──
    if other_pkgs:
        specs = []
        for name, ver in other_pkgs:
            specs.append(f"{name}>={ver}" if ver else name)

        print(f"\n  Installing: {', '.join(specs)}")
        print(f"  {'─' * 50}")

        result = subprocess.run(pip_base + specs)
        if result.returncode != 0:
            success = False
            print(f"\n  [WARN] Some packages may have failed to install")

    # ── Handle PyTorch separately ──
    if torch_pkgs:
        print(f"\n  PyTorch Installation")
        print(f"  {'─' * 50}")
        print(f"  PyTorch requires a platform-specific wheel.")
        print(f"  Detecting best option...\n")
        print(f"  Python version: {sys.version_info.major}.{sys.version_info.minor}")

        # Detect GPU
        has_nvidia = shutil.which("nvidia-smi") is not None
        is_apple_arm = (platform.system() == "Darwin" and platform.machine() == "arm64")

        if has_nvidia:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    print(f"  NVIDIA driver detected: {result.stdout.strip()}")
            except Exception:
                pass

            cuda_indices = [
                "https://download.pytorch.org/whl/cu126",
                "https://download.pytorch.org/whl/cu124",
                "https://download.pytorch.org/whl/cu121",
            ]

            print(f"  Installing PyTorch with CUDA support...")
            installed_torch = False

            for index_url in cuda_indices:
                cuda_tag = index_url.rsplit("/", 1)[-1]
                print(f"\n  Trying {cuda_tag}...")
                cmd = pip_base + ["torch", "torchvision",
                                  "--index-url", index_url]
                print(f"  Command: {' '.join(cmd)}\n")
                result = subprocess.run(cmd)

                if result.returncode == 0:
                    installed_torch = True
                    print(f"\n  Successfully installed PyTorch ({cuda_tag})")
                    break
                else:
                    print(f"  {cuda_tag} failed - trying next index...")

            if not installed_torch:
                print(f"\n  All CUDA indices failed. Trying default PyPI...")
                cmd = pip_base + ["torch", "torchvision"]
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    print(f"\n  Installed PyTorch from PyPI (may be CPU-only).")
                    print(f"  For CUDA support, visit: https://pytorch.org/get-started/locally/")
                else:
                    print(f"\n  [WARN] Automatic PyTorch installation failed.")
                    print(f"         Install manually from: https://pytorch.org/get-started/locally/")
                    print(f"         Select: Python {sys.version_info.major}.{sys.version_info.minor}, "
                          f"your CUDA version, and your OS.")
                    success = False

        elif is_apple_arm:
            print(f"  Apple Silicon detected - MPS backend included by default\n")
            cmd = pip_base + ["torch", "torchvision"]
            print(f"  Command: {' '.join(cmd)}\n")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\n  [WARN] Automatic PyTorch installation failed.")
                print(f"         Install manually from: https://pytorch.org/get-started/locally/")
                success = False

        else:
            print(f"  No GPU detected - installing CPU-only version\n")
            cmd = pip_base + ["torch", "torchvision",
                              "--index-url", "https://download.pytorch.org/whl/cpu"]
            print(f"  Command: {' '.join(cmd)}\n")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\n  [WARN] Automatic PyTorch installation failed.")
                print(f"         Install manually from: https://pytorch.org/get-started/locally/")
                success = False

    return success



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="Space Debris ML - Setup & Dependency Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                  # Interactive check + install prompt
  python setup.py --install        # Auto-install everything missing
  python setup.py --check          # Check only, no changes
  python setup.py --gpu            # Include GPU diagnostics
  python setup.py --stage acquire  # Check only acquisition dependencies
        """
    )
    parser.add_argument("--install", action="store_true",
                        help="Auto-install missing packages without prompting")
    parser.add_argument("--check", action="store_true",
                        help="Check only - do not install anything")
    parser.add_argument("--gpu", action="store_true",
                        help="Also run GPU/CUDA diagnostics")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["acquire", "process", "prepare", "train", "explorer", "all"],
                        help="Check dependencies for a specific stage only")
    args = parser.parse_args()

    # ── Header ──
    print()
    print("  ╔════════════════════════════════════════════════════════╗")
    print(f"  ║  Space Debris ML - Setup & Diagnostics  v{__version__:<13s}║")
    print("  ╚════════════════════════════════════════════════════════╝")

    # ── Section 1: System ──
    print(f"\n  SYSTEM")
    print(f"  {'─' * 50}")
    python_ok = check_python()
    pip_ok = check_pip()
    check_git()
    print(f"  [INFO]  Platform: {platform.system()} {platform.machine()}")
    print(f"  [INFO]  Working directory: {os.getcwd()}")

    if not python_ok:
        current = sys.version_info[:2]
        print(f"\n  Your Python version ({current[0]}.{current[1]}) is too old.")
        print(f"  Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required.")
        print()

        # Detect platform and provide targeted instructions
        if platform.system() == "Windows":
            download_url = "https://www.python.org/downloads/windows/"
            print(f"  To upgrade on Windows:")
            print(f"    1. Download the latest Python from:")
            print(f"       {download_url}")
            print(f"    2. Run the installer (check 'Add Python to PATH')")
            print(f"    3. Restart your terminal and re-run this script")
            print()
            try:
                answer = input("  Open the download page in your browser? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
                print()

            if answer not in ("n", "no"):
                try:
                    import webbrowser
                    webbrowser.open(download_url)
                    print(f"  Opened {download_url}")
                except Exception:
                    print(f"  Could not open browser. Visit the URL above manually.")

        elif platform.system() == "Darwin":
            print(f"  To upgrade on macOS:")
            print(f"    brew install python@3.13")
            print(f"    Or download from: https://www.python.org/downloads/macos/")

        else:
            print(f"  To upgrade on Linux:")
            print(f"    sudo apt update && sudo apt install python3.13  (Ubuntu/Debian)")
            print(f"    Or: sudo dnf install python3.13                 (Fedora)")
            print(f"    Or download from: https://www.python.org/downloads/")

        print()
        sys.exit(1)

    if not pip_ok:
        print(f"\n  pip is required to install packages. Aborting.")
        sys.exit(1)

    # ── Section 2: Packages ──
    print(f"\n  PACKAGES")
    print(f"  {'─' * 50}")
    installed, outdated, missing = check_packages(stage_filter=args.stage)

    # ── Section 3: GPU ──
    torch_installed = any(name == "torch" for name, _, _ in installed)
    if args.gpu or torch_installed:
        print(f"\n  GPU")
        print(f"  {'─' * 50}")
        gpu_info = check_gpu()

    # ── Section 4: Credentials ──
    print(f"\n  CREDENTIALS")
    print(f"  {'─' * 50}")
    check_credentials()

    # ── Section 5: Data directories ──
    print(f"\n  DATA DIRECTORIES")
    print(f"  {'─' * 50}")
    check_directories()

    # ── Summary ──
    required_missing = [(n, p, v, s, r) for n, p, v, s, r in missing if r]
    optional_missing = [(n, p, v, s, r) for n, p, v, s, r in missing if not r]

    print(f"\n  {'═' * 54}")
    print(f"  SUMMARY")
    print(f"  {'═' * 54}")
    print(f"  Installed:           {len(installed)}")
    if outdated:
        print(f"  Outdated:            {len(outdated)}")
    if required_missing:
        print(f"  Missing (required):  {len(required_missing)}")
    if optional_missing:
        print(f"  Missing (optional):  {len(optional_missing)}")

    # ── Build install list ──
    need_install = [(p, v) for _, p, v, _, _ in required_missing]
    need_upgrade = [(p, v) for _, _, v, p, _ in outdated]
    all_to_install = need_install + need_upgrade

    if not all_to_install:
        print(f"\n  All required dependencies are installed.")

        if optional_missing:
            opt_names = ", ".join(p for _, p, _, _, _ in optional_missing)
            print(f"  Optional: pip install {opt_names}")

        # Create directories
        print()
        create_directories()

        print(f"\n  You're ready to go!")
        print(f"  Next: run the acquisition scripts in 1_acquire/")
        print()
        sys.exit(0)

    # ── Install decision ──
    if args.check:
        names = ", ".join(p for p, _ in all_to_install)
        print(f"\n  Missing: {names}")
        print(f"  Run `python setup.py --install` to install them.")
        sys.exit(1)

    if not args.install:
        # Interactive prompt
        names = ", ".join(p for p, _ in all_to_install)
        print(f"\n  Packages to install/upgrade: {names}")

        try:
            answer = input("\n  Install now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
            print()

        if answer in ("n", "no"):
            print("\n  Skipped. Run `python setup.py --install` later.")
            sys.exit(1)

    # ── Install ──
    print(f"\n  INSTALLING")
    print(f"  {'─' * 50}")
    ok = install_packages(all_to_install)

    # ── Verify ──
    # Run verification in a fresh subprocess so it sees newly installed packages.
    # importlib.metadata caches package info at startup - a same-process re-check
    # would still show packages as missing.
    print(f"\n  VERIFICATION")
    print(f"  {'─' * 50}")

    verify_script = (
        "import importlib, importlib.metadata, sys\n"
        "sys.path.insert(0, __import__('pathlib').Path(__file__).parent.__str__() "
        "if '__file__' in dir() else '.')\n"
    )
    # Build a quick inline check
    check_names = set()
    for imp_name, pip_name, min_ver, stages, required in REQUIREMENTS:
        if imp_name not in check_names:
            check_names.add(imp_name)

    verify_code = [
        "import importlib.metadata",
        "DIST_MAP = " + repr(_DIST_MAP),
        "results = []",
    ]
    for imp_name in sorted(check_names):
        pip_name = _DIST_MAP.get(imp_name, imp_name)
        required = any(r for n, _, _, _, r in REQUIREMENTS if n == imp_name)
        verify_code.append(f"""
try:
    v = importlib.metadata.version({pip_name!r})
    print(f'  [  OK]  {imp_name:<15s}  {{v}}')
    results.append(True)
except Exception:
    tag = '{"MISS" if required else "SKIP"}'
    print(f'  [{{tag:>4s}}]  {imp_name:<15s}  not installed')
    results.append({'True' if not required else 'False'})
""")
    verify_code.append("import sys; sys.exit(0 if all(results) else 1)")

    full_code = "\n".join(verify_code)
    verify_result = subprocess.run(
        [sys.executable, "-c", full_code],
        cwd=str(Path(__file__).parent),
    )

    if verify_result.returncode == 0:
        # Create directories
        print()
        create_directories()

        print(f"\n  Setup complete!")
        print(f"  Next: run the acquisition scripts in 1_acquire/")
        print()
        sys.exit(0)
    else:
        print(f"\n  Some required packages are still missing after installation.")
        print(f"  Check the errors above and install manually.")
        print(f"  For PyTorch: https://pytorch.org/get-started/locally/")
        sys.exit(1)


if __name__ == "__main__":
    main()
