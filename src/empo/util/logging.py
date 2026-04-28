"""Logging utilities for EMPO."""

import logging
import os
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "%(levelname)s:%(name)s:%(message)s"


def _normalize_level(level: Optional[Union[int, str]]) -> Union[int, str]:
    if level is None:
        level = os.environ.get("EMPO_LOG_LEVEL")
    if level is None:
        return logging.WARNING
    if isinstance(level, str):
        level_name = level.strip().upper()
        if hasattr(logging, level_name):
            return getattr(logging, level_name)
        raise ValueError(f"Invalid log level: {level}")
    return level


def configure_logging(level: Optional[Union[int, str]] = None, fmt: Optional[str] = None) -> None:
    """Configure root logging for EMPO scripts and examples."""
    resolved_level = _normalize_level(level)
    logging.basicConfig(level=resolved_level, format=fmt or DEFAULT_LOG_FORMAT)
    logging.getLogger().setLevel(resolved_level)
