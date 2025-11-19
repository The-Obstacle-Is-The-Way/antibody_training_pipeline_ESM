"""
Centralized logging configuration for preprocessing scripts.

Usage:
    from preprocessing.logging_config import setup_logger

    logger = setup_logger(__name__)
    logger.info("Processing started")
    logger.debug("Detailed progress info")
"""

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Path | None = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Set up a logger with consistent formatting for preprocessing scripts.

    Args:
        name: Logger name (use __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        fmt: Log message format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Convert level string to int if needed
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)

    return logger


def add_logging_args(parser: Any) -> Any:
    """Add standard logging arguments to argparse parser."""
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )
    return parser
