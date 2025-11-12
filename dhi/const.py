import logging
import pathlib

import numpy as np

##############################
# GENERAL CONSTANTS
##############################

# Sizes in bytes
DHI_DEFAULT_SIZE_UNIT: str = "MB"
DHI_SIZES_BYTES: dict[str, int] = {
    "KB": 1 << 10,
    "MB": 1 << 20,
    "GB": 1 << 30,
}

DHI_SUPPORTED_EXT_FTYPE: dict[str, str] = {
    "csv": "csv",
    "xlsx": "excel",
    "xls": "excel",
    "json": "json",
    "parquet": "parquet",
}

##############################
# LOGGING CONSTANTS
##############################

DHI_LOGGING_LOG_LEVEL: int = logging.INFO
DHI_LOGGING_FORMAT: str = "[%(asctime)s] - %(name)s - [%(levelname)s] - %(message)s"
DHI_LOGGING_MAX_BYTES: int = 10 * (1 << 20)  # 10 MB
DHI_LOGGING_BACKUP_COUNT: int = 3
DHI_LOGGING_LOG_PATH: pathlib.Path = pathlib.Path().cwd() / ".." / "logs" / "dhi.log"

##############################
# NUMERICAL CONSTANTS
##############################

DHI_NUMPY_INT_RANGES: list[tuple[int, int, type]] = [
    (np.iinfo(np.int8).min, np.iinfo(np.int8).max, np.int8),
    (np.iinfo(np.int16).min, np.iinfo(np.int16).max, np.int16),
    (np.iinfo(np.int32).min, np.iinfo(np.int32).max, np.int32),
    (np.iinfo(np.int64).min, np.iinfo(np.int64).max, np.int64),
]

DHI_NUMPY_FLOAT_RANGES: list[tuple[np.floating, np.floating, type]] = [
    (np.finfo(np.float16).min, np.finfo(np.float16).max, np.float16),
    (np.finfo(np.float32).min, np.finfo(np.float32).max, np.float32),
    (np.finfo(np.float64).min, np.finfo(np.float64).max, np.float64),
]
