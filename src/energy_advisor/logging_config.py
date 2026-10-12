"""Logging setup shared by the CLI and library code."""

import logging

from rich.logging import RichHandler

PACKAGE_LOGGER = "energy_advisor"


def configure_logging(level: str = "INFO") -> None:
    """Our package logs at ``level``; third-party libraries (httpx, openai, chromadb) only from WARNING up."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    logging.getLogger(PACKAGE_LOGGER).setLevel(level.upper())
