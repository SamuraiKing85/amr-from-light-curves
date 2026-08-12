"""
Shared Interactive Menu Utilities
====================================
Provides consistent text-based menu UI across all pipeline scripts.
Extracted from tle_manager.py and standardised.

Usage:
    from common.menu import print_header, prompt_choice, prompt_path, prompt_confirm

    print_header("MMT-9 Downloader", "2.0.0")

    choice = prompt_choice("What would you like to do?", [
        ("download", "Download all files"),
        ("status",   "Check download status"),
        ("quit",     "Quit"),
    ])
"""

from pathlib import Path
from typing import Optional

__all__ = [
    "print_header",
    "prompt_choice",
    "prompt_path",
    "prompt_confirm",
    "prompt_input",
    "print_status_bar",
    "print_summary_table",
    "print_section",
    "print_warning",
    "print_error",
    "print_success",
]



#  HEADER & SECTION


def print_header(title: str, version: str = "", subtitle: str = "") -> None:
    """Print a consistent box-drawing header.

    Example output:
        ╔══════════════════════════════════════════════════════════╗
        ║            MMT-9 Downloader v2.0.0                      ║
        ║  Download raw light curve files from MMT-9 archive      ║
        ╚══════════════════════════════════════════════════════════╝
    """
    width = 58
    inner = width - 2  # Space inside the box

    ver_str = f" v{version}" if version else ""
    title_line = f"{title}{ver_str}"

    print()
    print(f"  \u2554{'═' * inner}\u2557")
    print(f"  \u2551  {title_line:<{inner - 2}s}\u2551")
    if subtitle:
        print(f"  \u2551  {subtitle:<{inner - 2}s}\u2551")
    print(f"  \u255a{'═' * inner}\u255d")
    print()


def print_section(title: str) -> None:
    """Print a section header within a menu."""
    print(f"\n  {title}")
    print(f"  {'─' * 50}")



#  PROMPTS


def prompt_choice(prompt: str, options: list[tuple[str, str]]) -> Optional[str]:
    """Display numbered options and get user choice.

    Args:
        prompt:  Question text (e.g. "What would you like to do?")
        options: List of (key, label) tuples. The key is returned on selection.

    Returns:
        The key of the selected option, or None on Ctrl+C / EOF.

    Example:
        choice = prompt_choice("Select mode:", [
            ("full",   "Run full pipeline"),
            ("quick",  "Quick scan only"),
            ("quit",   "Exit"),
        ])
    """
    print(f"\n  {prompt}")
    for i, (key, label) in enumerate(options, 1):
        print(f"    [{i}] {label}")
    print()

    while True:
        try:
            raw = input("  Enter choice: ").strip()
            if not raw:
                continue

            # Try numeric selection
            try:
                idx = int(raw)
                if 1 <= idx <= len(options):
                    return options[idx - 1][0]
                print(f"  Please enter a number between 1 and {len(options)}")
                continue
            except ValueError:
                pass

            # Try key match (case-insensitive)
            for key, _ in options:
                if raw.lower() == key.lower():
                    return key

            print(f"  Please enter a number between 1 and {len(options)}")

        except (EOFError, KeyboardInterrupt):
            print()
            return None


def prompt_path(prompt: str, must_exist: bool = False,
                default: Optional[str] = None,
                is_dir: bool = True) -> Optional[Path]:
    """Prompt user for a filesystem path with validation.

    Args:
        prompt:     Description of what path is needed.
        must_exist: If True, rejects paths that don't exist.
        default:    Default value shown in brackets. Enter accepts it.
        is_dir:     If True, validates the path is a directory (not a file).

    Returns:
        Path object, or None on Ctrl+C / empty input with no default.
    """
    default_str = f" [{default}]" if default else ""
    try:
        raw = input(f"  {prompt}{default_str}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not raw:
        if default:
            return Path(default)
        return None

    # Strip quotes (Windows users paste paths with quotes)
    raw = raw.strip('"').strip("'")
    path = Path(raw)

    if must_exist:
        if is_dir and not path.is_dir():
            print(f"    \u2717 Directory not found: {path}")
            return None
        elif not is_dir and not path.is_file():
            print(f"    \u2717 File not found: {path}")
            return None

    print(f"    \u2713 {path}")
    return path


def prompt_confirm(prompt: str, default: bool = False) -> bool:
    """Ask a yes/no question.

    Args:
        prompt:  Question text.
        default: What Enter with no input means (False = No, True = Yes).

    Returns:
        True for yes, False for no.
    """
    hint = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {prompt} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if not raw:
        return default

    return raw in ("y", "yes")


def prompt_input(prompt: str, default: Optional[str] = None) -> Optional[str]:
    """Prompt for free-text input.

    Args:
        prompt:  Question text.
        default: Default value shown in brackets.

    Returns:
        User input string, default if empty, or None on Ctrl+C.
    """
    default_str = f" [{default}]" if default else ""
    try:
        raw = input(f"  {prompt}{default_str}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    return raw if raw else default



#  STATUS DISPLAY


def print_status_bar(label: str, done: int, total: int, width: int = 30) -> None:
    """Print a labelled progress bar.

    Example output:
        Coverage: [████████████████████░░░░░░░░░░] 67.3%
    """
    if total <= 0:
        pct = 0.0
    else:
        pct = done / total * 100

    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {label}: [{bar}] {pct:.1f}%  ({done:,}/{total:,})")


def print_summary_table(rows: list[tuple[str, str]], indent: int = 2) -> None:
    """Print aligned key-value pairs.

    Args:
        rows: List of (label, value) tuples.
        indent: Number of leading spaces.

    Example output:
        Total objects:          89,054
        With mass data:         34,210
        With cross-section:     28,445
    """
    if not rows:
        return
    max_label = max(len(label) for label, _ in rows)
    pad = " " * indent
    for label, value in rows:
        print(f"{pad}{label:<{max_label}s}  {value}")



#  MESSAGE HELPERS


def print_warning(msg: str) -> None:
    """Print a warning message."""
    print(f"  [WARN] {msg}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"  [ERROR] {msg}")


def print_success(msg: str) -> None:
    """Print a success message."""
    print(f"  [OK] {msg}")
