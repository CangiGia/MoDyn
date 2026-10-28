"""Centralized logging configuration for MoDyn.

MoDyn does not call :func:`logging.basicConfig` on import — that is the
application's responsibility. Use :func:`configure_logging` from a script
or notebook to get a sensible default. Every module obtains its own logger
via :func:`get_logger`.
"""

from __future__ import annotations

import logging

__all__ = ["configure_logging", "get_logger"]

_LOGGER_NAME = "modyn"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    fmt: str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
    propagate: bool = False,
) -> logging.Logger:
    """Configure the package-level logger and return it.

    Idempotent: calling it twice does not duplicate handlers.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = propagate
    if not any(getattr(h, "_modyn_default", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handler._modyn_default = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of ``modyn`` (or the root one if ``name`` is None)."""
    if name is None or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
