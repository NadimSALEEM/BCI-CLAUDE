"""Centralised logging configuration.

One call to :func:`configure_logging` early in startup wires up a console
handler and (optionally) a rotating file handler. Every module obtains its
logger with ``logging.getLogger(__name__)`` and never configures handlers
itself.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_CONFIGURED = False

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: str = "INFO",
    to_file: bool = True,
    directory: str | Path = "logs",
) -> Path | None:
    """Configure the root logger. Safe to call more than once (idempotent).

    Returns the path of the log file if file logging is enabled.
    """

    global _CONFIGURED
    root = logging.getLogger()
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(numeric)

    if _CONFIGURED:
        return None

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_path: Path | None = None
    if to_file:
        log_dir = Path(directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "neurobci.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Logging configured (level=%s).", level)
    return log_path
