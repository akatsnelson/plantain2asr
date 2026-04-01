from __future__ import annotations

import logging
from typing import Union


_ROOT_LOGGER_NAME = "plantain2asr"


def get_logger(name: str | None = None) -> logging.Logger:
    if not name or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(level: Union[int, str] = logging.INFO) -> logging.Logger:
    """
    Configure the package logger once.

    The library no longer prints operational progress directly. Call this helper if
    you want info-level progress messages in scripts or notebooks.
    """
    logger = get_logger()
    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_optional_dependency_warning(
    logger: logging.Logger,
    package: str,
    *,
    hint: str,
    error: Exception | None = None,
) -> None:
    message = f"Optional dependency group '{package}' is unavailable. {hint}"
    if error is not None:
        message = f"{message} Original error: {error}"
    logger.warning(message)
