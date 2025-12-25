import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd

import dhi.constants as dconst
from dhi.decorators import time_func
from dhi.utils import get_filetype, get_logger


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


class Loader(object):
    def __init__(self, **kwargs) -> None:
        self.init(**kwargs)

    def _process_dataset_path(self, dataset_path: Any) -> pathlib.Path:
        """
        Processes the dataset path such that:
        * the dataset path is set and not None
        * the dataset path is str
        * the dataset path is converted to absolute pathlib.Path and validated to exist

        :param Any dataset_path: The dataset path to process
        :return pathlib.Path: The processed dataset path
        :raises ValueError: If the dataset path is not set or does not exist
        """
        if dataset_path is None:
            self.logger.error("No dataset path provided.")
            raise ValueError("dataset_path is required")

        if not isinstance(dataset_path, str):
            self.logger.error("dataset_path must be a string.")
            raise TypeError("dataset_path must be a string")

        dataset_path = pathlib.Path(dataset_path).resolve()

        if not dataset_path.exists():
            self.logger.error(f"Dataset path does not exist: {dataset_path}")
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        return dataset_path

    def init(self, **kwargs) -> None:
        self.logger = get_logger(self.__class__.__name__)

        # TODO: maybe add a more configurable config parser?
        # NOTE: currently the config allows missing values, but is strict with their default values in this init method
        # NOTE: either create a config parser base class and inherit from it, or use pydantic for config management

        self.dataset_path = self._process_dataset_path(kwargs.get("dataset_path", None))

        # Renames the columns of the dataset to the specified values,
        # if provided in the config and if the columns are of the same size as the provided new names
        self.column_names = kwargs.get("column_names", [])
        assert isinstance(self.column_names, list), "column_names must be a list"
        self.column_names = list(set(self.column_names))

        # Separator between columns and instances
        self.separator = kwargs.get("separator", ",")
        assert isinstance(self.separator, str), "separator must be a string"

        # The decimal marker
        self.decimal = kwargs.get("decimal", ".")
        assert isinstance(self.decimal, str), "decimal must be a string"

        # Markers used to indicate missing values in the dataset
        self.missing_markers = kwargs.get("missing_markers", [])
        assert isinstance(self.missing_markers, list), "missing_markers must be a list"
        self.missing_markers = list(set(self.missing_markers))

        # Whether to remove unnamed columns from the dataset
        self.remove_unnamed = kwargs.get("remove_unnamed", True)
        assert isinstance(self.remove_unnamed, bool), "remove_unnamed must be a boolean"

        # Whether to reduce memory usage when loading the dataset
        self.reduce_memory_usage = kwargs.get("reduce_memory_usage", True)
        assert isinstance(self.reduce_memory_usage, bool), "reduce_memory_usage must be a boolean"

        # Label columns in the dataset
        self._label_columns = kwargs.get("label_columns", [])
        assert isinstance(self._label_columns, list), "label_columns must be a list"
        assert all(isinstance(col, str) for col in self._label_columns), "all label_columns must be strings"
        self._label_columns = list(set(self._label_columns))

    @property
    def label_columns(self) -> list[str]:
        return self._label_columns

    @label_columns.setter
    def label_columns(self, value) -> None:
        if isinstance(value, str):
            self._label_columns = [value]
        elif isinstance(value, list) and all(isinstance(col, str) for col in value):
            self._label_columns = list(set(value))
        else:
            raise TypeError("label_columns must be a string or a list of strings")

    def _load_dataset(self) -> pd.DataFrame:
        """
        Internal method used to load the dataset from its path,
        considering the filetype.

        Falls back to reading CSV if the filetype is not supported.

        :return pd.DataFrame: The read dataset.
        """
        ftype = get_filetype(self.dataset_path)
        if not ftype:
            self.logger.warning(f"Could not determine filetype for {self.dataset_path}, falling back to CSV reader!")
            self.logger.info(f"Supported filetypes are: {set(dconst.DHI_SUPPORTED_EXT_FTYPE.values())}")
            ftype = "csv"

        try:
            match ftype:
                case "csv":
                    df = pd.read_csv(
                        self.dataset_path,
                        sep=self.separator,
                        decimal=self.decimal,
                        na_values=self.missing_markers,
                    )
                case "excel":
                    df = pd.read_excel(
                        self.dataset_path,
                        sheet_name=None,
                        na_values=self.missing_markers,
                        decimal=self.decimal,
                    )
                    # Combining all sheets into a single DataFrame
                    df = pd.concat(df.values(), ignore_index=True)
                case "json":
                    df = pd.read_json(self.dataset_path)
                case "parquet":
                    df = pd.read_parquet(self.dataset_path)
                case _:
                    self.logger.warning(f"Unsupported filetype '{ftype}', falling back to CSV reader!")
                    self.logger.info(f"Supported filetypes are: {set(dconst.DHI_SUPPORTED_EXT_FTYPE.values())}")
                    df = pd.read_csv(
                        self.dataset_path,
                        sep=self.separator,
                        decimal=self.decimal,
                        na_values=self.missing_markers,
                    )

            return df
        except Exception as e:
            self.logger.error(f"An error occurred while loading the dataset: {e}")
            raise RuntimeError(f"Failed to load dataset from {self.dataset_path}: {e}")

    @time_func
    def load(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified path.

        Only accepts a limited number of filetypes, most common ones used in Data Analysis.

        :return pd.DataFrame: The loaded dataset.
        :raises ValueError: if the filetype is not supported.
        :raises RuntimeError: if something wrong happens when loading the data.
        """
        self.logger.info(f"Loading dataset from path: {self.dataset_path}")
        df = self._load_dataset()

        if self.column_names and len(self.column_names) == len(df.columns):
            self.logger.info("Renaming dataset columns as per provided column names")
            df.columns = self.column_names

        if self.remove_unnamed:
            self.logger.info("Removing unnamed columns from the dataset")
            unnamed_cols = df.columns[df.columns.str.contains("^Unnamed")]
            if unnamed_cols is None or unnamed_cols.empty:
                self.logger.info("No unnamed columns found")
            else:
                df.drop(columns=list(unnamed_cols), inplace=True)

        if self.missing_markers:
            self.logger.info(f"Replacing missing markers {self.missing_markers} with NaN")
            df.replace(self.missing_markers, np.nan, inplace=True)

        self.logger.info("Removing instances with all NaN values")
        initial_shape = df.shape
        df.dropna(axis=0, how="all", inplace=True)
        if df.shape != initial_shape:
            self.logger.info(f"Dropped {initial_shape[0] - df.shape[0]} rows with all NaN values")
            df.reset_index(drop=True, inplace=True)

        if self.reduce_memory_usage:
            self.logger.info("Reducing memory usage of the dataset")
            df = reduce_memory_usage(df, logger=self.logger)

        return df
