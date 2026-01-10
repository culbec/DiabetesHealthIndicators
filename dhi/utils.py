import logging
import pathlib
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List

from dhi.constants import (
    DHI_LOGGING_BACKUP_COUNT,
    DHI_LOGGING_FORMAT,
    DHI_LOGGING_LOG_LEVEL,
    DHI_LOGGING_MAX_BYTES,
    DHI_SUPPORTED_EXT_FILE_TYPE,
)

ParamGridType = Dict[str, List[Any]] | List[Dict[str, List[Any]]]


def is_valid_param_grid(param_grid: Any) -> bool:
    """
    Check if param_grid has a valid format.

    Valid formats:
    1. Single grid (dict): {"param": [value1, value2], ...}
    2. List of grids (list[dict]): [{"param": [value1]}, {"param": [value2]}]

    :param param_grid: Value to validate.
    :return: True if valid, False otherwise.
    """
    if isinstance(param_grid, dict):
        return all(isinstance(v, list) for v in param_grid.values())

    if isinstance(param_grid, list):
        return all(isinstance(grid, dict) and all(isinstance(v, list) for v in grid.values()) for grid in param_grid)

    return False


def validate_param_grid(param_grid: Any, name: str = "param_grid") -> ParamGridType:
    """
    Validate and return param_grid, raising AssertionError if invalid.

    :param param_grid: Value to validate.
    :param name: Name for error messages.
    :return: The validated param_grid.
    :raises AssertionError: If param_grid format is invalid.
    """
    assert is_valid_param_grid(param_grid), (
        f"{name} must be a dictionary or a list of dictionaries, "
        f"with all values being lists. Got: {type(param_grid).__name__}"
    )
    return param_grid


def get_logger(name: str, level: int = DHI_LOGGING_LOG_LEVEL, log_path: pathlib.Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(level or DHI_LOGGING_LOG_LEVEL)
    formatter = logging.Formatter(DHI_LOGGING_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level or DHI_LOGGING_LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path:
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        rot_file_handler = RotatingFileHandler(
            log_path,
            maxBytes=DHI_LOGGING_MAX_BYTES,
            backupCount=DHI_LOGGING_BACKUP_COUNT,
        )
        rot_file_handler.setLevel(level or DHI_LOGGING_LOG_LEVEL)
        rot_file_handler.setFormatter(formatter)
        logger.addHandler(rot_file_handler)

    return logger


def get_filetype(file_path: pathlib.Path) -> str | None:
    if not file_path.exists():
        return None

    return DHI_SUPPORTED_EXT_FILE_TYPE.get(file_path.suffix.lower().lstrip("."))
