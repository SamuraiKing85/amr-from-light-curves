"""
Credential Management
========================
Loads, validates, prompts for, and saves API credentials.

If credentials are missing or empty, the user is prompted interactively
rather than the script dying with an error.

Credentials are stored in config/credentials.json. The file is created
automatically from the template if it doesn't exist.

Usage:
    from common.credentials import get_credentials

    # Returns dict with populated fields, prompting if needed
    creds = get_credentials("spacetrack")  # {"username": "...", "password": "..."}
    creds = get_credentials("discos")      # {"token": "..."}
"""

import json
import sys
from pathlib import Path
from typing import Optional

__all__ = ["get_credentials", "check_credentials_status", "setup_credentials_interactive"]

# Service definitions: what fields each service needs and how to prompt for them
_SERVICE_DEFS = {
    "spacetrack": {
        "fields": [
            ("username", "Space-Track email",    "Register at https://www.space-track.org/auth/createAccount"),
            ("password", "Space-Track password",  None),
        ],
        "name": "Space-Track",
    },
    "discos": {
        "fields": [
            ("token", "DISCOS API token", "Get yours at https://discosweb.esoc.esa.int/tokens"),
        ],
        "name": "ESA DISCOS",
    },
}



#  FILE I/O


def _get_paths() -> tuple:
    """Get credential file paths."""
    from config.paths import CREDENTIALS_FILE, CREDENTIALS_EXAMPLE
    return CREDENTIALS_FILE, CREDENTIALS_EXAMPLE


def _load_file(cred_path: Path) -> dict:
    """Load credentials.json, returning empty dict on any error."""
    if not cred_path.exists():
        return {}
    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_file(cred_path: Path, data: dict) -> None:
    """Save credentials.json atomically."""
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cred_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    import os
    if os.name == "nt" and cred_path.exists():
        cred_path.unlink()
    os.replace(str(tmp), str(cred_path))


def _ensure_file_exists(cred_path: Path, example_path: Path) -> dict:
    """Create credentials.json from template if it doesn't exist. Returns contents."""
    if cred_path.exists():
        return _load_file(cred_path)

    # Create from template
    if example_path.exists():
        template = _load_file(example_path)
    else:
        # Build minimal template
        template = {}
        for service, sdef in _SERVICE_DEFS.items():
            template[service] = {field: "" for field, _, _ in sdef["fields"]}

    _save_file(cred_path, template)
    return template



#  INTERACTIVE PROMPTING


def _prompt_field(field_name: str, prompt_text: str, hint: str = None,
                  is_secret: bool = False) -> Optional[str]:
    """Prompt the user for a single credential field."""
    if hint:
        print(f"    {hint}")
    try:
        if is_secret:
            # Try to use getpass for passwords (hides input)
            try:
                import getpass
                value = getpass.getpass(f"    {prompt_text}: ").strip()
            except (ImportError, EOFError):
                value = input(f"    {prompt_text}: ").strip()
        else:
            value = input(f"    {prompt_text}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    return value if value else None


def _prompt_for_service(service: str) -> Optional[dict]:
    """Prompt user for all fields of a service. Returns dict or None if cancelled."""
    sdef = _SERVICE_DEFS.get(service)
    if not sdef:
        return None

    print(f"\n  {sdef['name']} credentials required.")
    values = {}

    for field, prompt_text, hint in sdef["fields"]:
        is_secret = "password" in field.lower() or "token" in field.lower()
        value = _prompt_field(field, prompt_text, hint, is_secret=is_secret)
        if value is None:
            print("    Cancelled.")
            return None
        values[field] = value

    return values



#  PUBLIC API


def get_credentials(service: str, allow_prompt: bool = True) -> dict:
    """Get credentials for a service, prompting interactively if missing.

    Args:
        service:      One of "spacetrack" or "discos".
        allow_prompt: If True (default), prompt the user for missing credentials.
                      If False, return None for missing fields.

    Returns:
        Dict with service-specific credential fields (all populated).

    Raises:
        SystemExit: If credentials are needed but user cancels the prompt.
    """
    cred_path, example_path = _get_paths()
    all_creds = _ensure_file_exists(cred_path, example_path)

    sdef = _SERVICE_DEFS.get(service)
    if not sdef:
        print(f"  [ERROR] Unknown service: {service}")
        print(f"          Available: {', '.join(_SERVICE_DEFS.keys())}")
        sys.exit(1)

    # Get current values for this service
    service_creds = all_creds.get(service, {})

    # Strip help fields
    service_creds = {k: v for k, v in service_creds.items() if not k.startswith("_")}

    # Check which fields are missing or empty
    missing = [field for field, _, _ in sdef["fields"]
               if not service_creds.get(field)]

    if not missing:
        return service_creds

    # Fields are missing - prompt if allowed
    if not allow_prompt:
        return service_creds

    prompted = _prompt_for_service(service)
    if prompted is None:
        print(f"\n  [ERROR] {sdef['name']} credentials are required to continue.")
        print(f"          Edit config/credentials.json manually if preferred.")
        sys.exit(1)

    # Merge and save
    if service not in all_creds:
        all_creds[service] = {}
    all_creds[service].update(prompted)
    _save_file(cred_path, all_creds)
    print(f"    Credentials saved to config/credentials.json\n")

    return prompted


def check_credentials_status() -> dict:
    """Check status of all services without prompting.

    Returns:
        {"spacetrack": True/False, "discos": True/False}
    """
    cred_path, _ = _get_paths()
    all_creds = _load_file(cred_path)
    status = {}

    for service, sdef in _SERVICE_DEFS.items():
        service_creds = all_creds.get(service, {})
        missing = [field for field, _, _ in sdef["fields"]
                   if not service_creds.get(field)]
        status[service] = len(missing) == 0

    return status


def setup_credentials_interactive() -> None:
    """Run full interactive credential setup for all services.

    Used by setup.py. Prompts for each service that has missing fields.
    """
    cred_path, example_path = _get_paths()
    all_creds = _ensure_file_exists(cred_path, example_path)

    any_prompted = False

    for service, sdef in _SERVICE_DEFS.items():
        service_creds = all_creds.get(service, {})
        service_creds = {k: v for k, v in service_creds.items() if not k.startswith("_")}

        missing = [field for field, _, _ in sdef["fields"]
                   if not service_creds.get(field)]

        if not missing:
            print(f"  [  OK]  {sdef['name']}")
            continue

        print(f"  [MISS]  {sdef['name']} - {len(missing)} field(s) missing")

        try:
            answer = input(f"          Set up now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
            print()

        if answer in ("n", "no"):
            print(f"          Skipped. Edit config/credentials.json later.\n")
            continue

        prompted = _prompt_for_service(service)
        if prompted:
            if service not in all_creds:
                all_creds[service] = {}
            all_creds[service].update(prompted)
            any_prompted = True
            print(f"    Saved.\n")

    if any_prompted:
        _save_file(cred_path, all_creds)
        print(f"  Credentials saved to: {cred_path}\n")
