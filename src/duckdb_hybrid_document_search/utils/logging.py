"""Logging utilities."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "duckdb_hybrid_document_search",
    level: int = logging.INFO,
    console: Optional[Console] = None,
) -> logging.Logger:
    """Set up a logger with Rich formatting.

    Args:
        name: Logger name
        level: Logging level
        console: Rich console instance (creates new one if None)

    Returns:
        Configured logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console = Console(stderr=True)

    # Create Rich handler
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=False,
    )
    handler.setLevel(level)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def get_logger(name: str = "duckdb_hybrid_document_search") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
