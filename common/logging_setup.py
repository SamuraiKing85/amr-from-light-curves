"""
Shared Logging Configuration
================================
Provides dual-output logging: concise to console, detailed to file.

Usage:
    from common.logging_setup import setup_logging

    logger = setup_logging("mmt9_downloader")
    logger.info("Starting download...")
    logger.warning("Rate limit approaching")
    logger.error("Connection failed")

Log files are written to data/logs/{name}_{date}.log with timestamps
and full module paths for debugging. Console output is kept clean.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

__all__ = ["setup_logging"]



#  FORMATTERS


class CleanConsoleFormatter(logging.Formatter):
    """Minimal console output - no timestamps, no module paths.

    Output looks like:
        [INFO]  Starting download...
        [WARN]  Rate limit approaching
        [ERROR] Connection failed
    """

    LEVEL_TAGS = {
        logging.DEBUG:    "[DBG] ",
        logging.INFO:     "[INFO] ",
        logging.WARNING:  "[WARN] ",
        logging.ERROR:    "[ERROR]",
        logging.CRITICAL: "[CRIT] ",
    }

    def format(self, record: logging.LogRecord) -> str:
        tag = self.LEVEL_TAGS.get(record.levelno, "[????]")
        return f"  {tag} {record.getMessage()}"


class DetailedFileFormatter(logging.Formatter):
    """Verbose file output with timestamps for audit trails.

    Output looks like:
        2025-02-24 10:30:45 [INFO ] common.network  - Starting download...
        2025-02-24 10:30:46 [WARN ] common.network  - Rate limit approaching
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        level = f"{record.levelname:<5s}"
        module = record.name
        return f"{ts} [{level}] {module:<25s} - {record.getMessage()}"



#  SETUP


def setup_logging(
    name: str,
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_file: bool = True,
) -> logging.Logger:
    """Configure logging for a pipeline script.

    Args:
        name:          Script identifier (e.g., "mmt9_downloader").
                       Used as both the logger name and the log filename.
        log_dir:       Directory for log files. Defaults to config.paths.LOG_DIR.
        console_level: Minimum level for console output (default: INFO).
        file_level:    Minimum level for file output (default: DEBUG).
        enable_file:   Whether to write log files at all.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if called again
    if logger.handlers:
        return logger

    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    # ── Console handler ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(CleanConsoleFormatter())
    logger.addHandler(console)

    # ── File handler ──
    if enable_file:
        if log_dir is None:
            try:
                from config.paths import LOG_DIR
                log_dir = LOG_DIR
            except ImportError:
                # Fallback if paths module isn't available
                log_dir = Path("./logs")

        log_dir.mkdir(parents=True, exist_ok=True)

        date_str = time.strftime("%Y%m%d")
        log_path = log_dir / f"{name}_{date_str}.log"

        file_handler = logging.FileHandler(
            log_path, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(DetailedFileFormatter())
        logger.addHandler(file_handler)

        logger.debug(f"Log file: {log_path}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name (does not reconfigure).

    Use this in sub-modules that don't need their own file handler
    but want to log under the parent script's logger.
    """
    return logging.getLogger(name)
