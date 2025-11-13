import logging
import pathlib
from logging.handlers import RotatingFileHandler

from dhi.const import (
    DHI_LOGGING_BACKUP_COUNT,
    DHI_LOGGING_FORMAT,
    DHI_LOGGING_LOG_LEVEL,
    DHI_LOGGING_MAX_BYTES,
    DHI_SUPPORTED_EXT_FTYPE,
)


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

    return DHI_SUPPORTED_EXT_FTYPE.get(file_path.suffix.lower().lstrip("."))
