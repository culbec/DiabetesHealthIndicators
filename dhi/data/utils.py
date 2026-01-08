import logging
import pandas as pd

import dhi.constants as dconst
from dhi.decorators import time_func


@time_func
def reduce_memory_usage(
    df: pd.DataFrame, logger: logging.Logger | None = None, memory_usage_unit: str = "MB"
) -> pd.DataFrame:
    """
    Reduces the memory usage of a pandas DataFrame
    by downcasting numerical columns to more efficient types.

    :param pd.DataFrame df: The DataFrame to optimize
    :param logging.Logger | None logger: The logger to use for logging, defaults to None
    :param str memory_usage_unit: The unit for logging memory usage, defaults to "MB"
    :return pd.DataFrame: The optimized DataFrame with reduced memory usage
    """
    reporter_info = logger.info if logger else print
    reporter_debug = logger.debug if logger else print
    reporter_warning = logger.warning if logger else print

    memory_usage_unit = memory_usage_unit.upper()
    if memory_usage_unit not in dconst.DHI_SIZES_BYTES:
        memory_usage_unit = dconst.DHI_DEFAULT_SIZE_UNIT
    munit = dconst.DHI_SIZES_BYTES[memory_usage_unit]

    start_mem = df.memory_usage().sum() / munit
    reporter_info(f"Initial memory usage of dataframe is {start_mem:.2f} {memory_usage_unit}")

    df_optimized = df.copy()

    for col in df_optimized.columns:
        if pd.api.types.is_numeric_dtype(df_optimized[col]):
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()

            if pd.api.types.is_integer_dtype(df_optimized[col]):
                for min_val, max_val, dtype in dconst.DHI_NUMPY_INT_RANGES:
                    if col_min >= min_val and col_max <= max_val:
                        df_optimized[col] = df_optimized[col].astype(dtype)
                        reporter_debug(f"Column '{col}' downcasted to {dtype}")
                        break
            else:
                for min_val, max_val, dtype in dconst.DHI_NUMPY_FLOAT_RANGES:
                    if col_min >= min_val and col_max <= max_val:
                        df_optimized[col] = df_optimized[col].astype(dtype)
                        reporter_debug(f"Column '{col}' downcasted to {dtype}")
                        break
        else:
            # TODO: research how to optimize non-numeric columns (e.g. categories)
            reporter_warning(f"Column '{col}' is not numeric and was not downcasted")

    end_mem = df_optimized.memory_usage().sum() / munit
    reporter_info(f"Final memory usage of dataframe is {end_mem:.2f} {memory_usage_unit}")
    reporter_info(f"Decreased memory usage by {(start_mem - end_mem) / start_mem * 100:.2f}%")

    return df_optimized