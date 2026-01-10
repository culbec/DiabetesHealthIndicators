import logging

import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Optional, Dict, Any

from numpy.typing import ArrayLike

import dhi.constants as dconst
from dhi.decorators import time_func


@time_func
def reduce_memory_usage(
    df: pd.DataFrame,
    logger: logging.Logger | None = None,
    memory_usage_unit: str = "MB",
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
    reporter_debug(f"Initial memory usage of dataframe is {start_mem:.8f} {memory_usage_unit}")

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
    reporter_debug(f"Final memory usage of dataframe is {end_mem:.8f} {memory_usage_unit}")

    decrease_mem = (start_mem - end_mem) / start_mem * 100
    (
        reporter_debug(f"Decreased memory usage by {decrease_mem:.2f}%")
        if np.isclose(decrease_mem, 0.0)
        else reporter_info(f"Decreased memory usage by {decrease_mem:.2f}%")
    )

    return df_optimized


def ensure_df(X: ArrayLike) -> tuple[pd.DataFrame, bool]:
    """
    Ensures X is pandas DataFrame, required for data preprocessing.

    If X is already a DataFrame, returns it as-is with was_converted=False.
    Otherwise, converts to DataFrame with columns named Feature_0, Feature_1, etc.

    :param X: Input data (array-like or DataFrame)
    :return: Tuple of (pandas DataFrame, was_converted flag)
    """
    if isinstance(X, pd.DataFrame):
        return X, False

    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    columns = pd.Index([f"Feature_{i}" for i in range(X_arr.shape[1])])
    return pd.DataFrame(X_arr, columns=columns), True


def infer_preprocessor_config_from_df(
    df: pd.DataFrame,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds a preprocessor config based on the DataFrame's actual column dtypes.

    Detects numerical and categorical columns and populates the config accordingly.

    :param df: The DataFrame to analyze
    :param base_config: Optional base config to merge with (for scaler/encoder settings)
    :return: Preprocessor config dict with numerical_features and categorical_features correctly set
    """
    config: Dict[str, Any] = deepcopy(base_config) if base_config else {}

    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    config.setdefault("numerical_features", {})
    config.setdefault("categorical_features", {})
    config["numerical_features"].setdefault("scaler", {})
    config["categorical_features"].setdefault("encoder", {})

    config["numerical_features"]["features"] = numerical_cols
    config["categorical_features"]["features"] = categorical_cols

    return config
