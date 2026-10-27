from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "experiment",
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """Configure and return a standardized Logger instance.

    Args:
        name: Logger hierarchy name.
        log_file: Optional file path to write log output (e.g., training.log).
        level: Logging level (default logging.INFO).
        console_output: Whether to attach stdout stream handler.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent propagation to root logger to avoid duplicate lines
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Check existing handlers to prevent duplicates
    has_console = False
    existing_file_paths = set()
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            has_console = True
        elif isinstance(h, logging.FileHandler):
            existing_file_paths.add(Path(h.baseFilename).resolve())

    # Attach Console Handler
    if console_output and not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Attach File Handler if requested
    if log_file is not None:
        file_path = Path(log_file).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path not in existing_file_paths:
            fh = logging.FileHandler(str(file_path), mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def close_logger_handlers(logger: logging.Logger) -> None:
    """Close and flush all file handlers associated with a logger."""
    for handler in list(logger.handlers):
        handler.flush()
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

