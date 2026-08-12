"""
Shared Network Utilities
===========================
Provides HTTP session management, retry logic, rate limiting,
and credential loading used by all data acquisition scripts.

Usage:
    from common.network import create_session, fetch_with_retry, load_credentials

    creds = load_credentials("spacetrack")
    session = create_session("MMT9-Downloader/2.0")

    response = fetch_with_retry(session, url, max_retries=3)
"""

import json
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Callable

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

__all__ = [
    "load_credentials",
    "create_session",
    "fetch_with_retry",
    "RateLimiter",
    "check_requests_available",
]

logger = logging.getLogger(__name__)



#  DEPENDENCY CHECK


def check_requests_available() -> None:
    """Raise a clear error if requests is not installed."""
    if not HAS_REQUESTS:
        print("\n  [ERROR] The 'requests' library is required for network operations.")
        print("          Run: python setup.py --install")
        print("          Or:  pip install requests")
        sys.exit(1)



#  CREDENTIALS


def load_credentials(service: str) -> dict:
    """Load credentials for a specific service, prompting if missing.

    Args:
        service: One of "spacetrack" or "discos".

    Returns:
        Dict with service-specific credential fields.
    """
    from common.credentials import get_credentials
    return get_credentials(service)



#  SESSION MANAGEMENT


def create_session(user_agent: str = "SpaceDebrisML/2.0",
                   extra_headers: Optional[dict] = None) -> "requests.Session":
    """Create a requests Session with standard configuration.

    Args:
        user_agent:    User-Agent header value.
        extra_headers: Additional headers to set on every request.

    Returns:
        Configured requests.Session.
    """
    check_requests_available()

    session = requests.Session()
    session.headers.update({
        "User-Agent": f"{user_agent} (University Research)",
    })
    if extra_headers:
        session.headers.update(extra_headers)

    return session



#  RETRY LOGIC


def fetch_with_retry(
    session: "requests.Session",
    url: str,
    method: str = "GET",
    max_retries: int = 3,
    base_delay: float = 10.0,
    rate_limit_delay: float = 60.0,
    timeout: int = 60,
    on_401: Optional[Callable] = None,
    **kwargs,
) -> "requests.Response":
    """Fetch a URL with exponential backoff retry logic.

    Handles:
        - 429 (Too Many Requests): waits rate_limit_delay seconds
        - 401 (Unauthorized): calls on_401 callback if provided (e.g., re-login)
        - 500 (Server Error): retries with exponential backoff
        - Connection errors / timeouts: retries with exponential backoff

    Args:
        session:          requests.Session to use.
        url:              Target URL.
        method:           HTTP method ("GET" or "POST").
        max_retries:      Maximum number of retry attempts.
        base_delay:       Initial delay in seconds (doubles each retry).
        rate_limit_delay: How long to wait on a 429 response.
        timeout:          Request timeout in seconds.
        on_401:           Optional callback for re-authentication.
                          Called with (session,) as argument. Should raise
                          on failure.
        **kwargs:         Extra arguments passed to session.request()
                          (e.g., data=, json=, params=, headers=).

    Returns:
        requests.Response on success.

    Raises:
        requests.exceptions.HTTPError: After exhausting all retries.
        requests.exceptions.ConnectionError: After exhausting all retries.
    """
    check_requests_available()

    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)

            # ── Rate limited ──
            if resp.status_code == 429:
                logger.warning(f"Rate limited (429), waiting {rate_limit_delay:.0f}s...")
                print(f" rate limited, waiting {rate_limit_delay:.0f}s...",
                      end="", flush=True)
                time.sleep(rate_limit_delay)
                continue

            # ── Auth expired ──
            if resp.status_code == 401:
                if on_401 and attempt < max_retries:
                    logger.info("Session expired (401), re-authenticating...")
                    print(f" session expired, re-authenticating...",
                          end="", flush=True)
                    on_401(session)
                    continue
                resp.raise_for_status()

            # ── Server error ──
            if resp.status_code >= 500:
                if attempt < max_retries:
                    delay = base_delay * attempt
                    logger.warning(f"Server error ({resp.status_code}), "
                                   f"retrying in {delay:.0f}s ({attempt}/{max_retries})")
                    print(f" server error ({resp.status_code}), retrying in {delay:.0f}s "
                          f"({attempt}/{max_retries})...", end="", flush=True)
                    time.sleep(delay)
                    continue
                resp.raise_for_status()

            # ── Success or client error ──
            resp.raise_for_status()
            return resp

        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * attempt
                logger.warning(f"Timeout, retrying in {delay:.0f}s ({attempt}/{max_retries})")
                print(f" timeout, retrying in {delay:.0f}s ({attempt}/{max_retries})...",
                      end="", flush=True)
                time.sleep(delay)
            else:
                logger.error(f"Timeout after {max_retries} retries: {url}")

        except requests.exceptions.ConnectionError as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * attempt * 1.5
                logger.warning(f"Connection error, retrying in {delay:.0f}s "
                               f"({attempt}/{max_retries})")
                print(f" connection error, retrying in {delay:.0f}s "
                      f"({attempt}/{max_retries})...", end="", flush=True)
                time.sleep(delay)
            else:
                logger.error(f"Connection failed after {max_retries} retries: {url}")

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise requests.exceptions.ConnectionError(
        f"Failed after {max_retries} retries: {url}"
    )



#  RATE LIMITER


class RateLimiter:
    """Token bucket rate limiter for API requests.

    Usage:
        limiter = RateLimiter(requests_per_minute=20)
        for url in urls:
            limiter.wait()   # Blocks if too fast
            response = session.get(url)

    Args:
        requests_per_minute: Maximum request rate.
    """

    def __init__(self, requests_per_minute: float = 20.0):
        self._min_interval = 60.0 / requests_per_minute
        self._last_request = 0.0

    @property
    def min_interval(self) -> float:
        """Minimum seconds between requests."""
        return self._min_interval

    def wait(self) -> float:
        """Wait until the next request is allowed.

        Returns:
            Actual seconds waited (0.0 if no wait needed).
        """
        now = time.monotonic()
        elapsed = now - self._last_request
        wait_time = self._min_interval - elapsed

        if wait_time > 0:
            time.sleep(wait_time)
            self._last_request = time.monotonic()
            return wait_time
        else:
            self._last_request = now
            return 0.0

    def reset(self) -> None:
        """Reset the timer (e.g., after a long pause)."""
        self._last_request = 0.0
