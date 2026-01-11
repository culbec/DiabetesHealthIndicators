from typing import Dict, Any

from dhi.data.loader._base import DataLoader
from dhi.data.loader.dhi_loader import DHIDataLoader

from dhi.data.preprocessing._base import DataPreprocessor
from dhi.data.preprocessing.dhi_preprocessor import DHIDataPreprocessor


SUPPORTED_DATASET_TYPES: set[str] = {"dhi"}


def build_loader(config: Dict[str, Any]) -> DataLoader:
    """
    Factory function to build a data loader based on the provided configuration, depending on dataset.

    :param config: A dictionary containing configuration parameters for the loader.
    :return: An instance of a DataLoader.
    """
    if config is None or not isinstance(config, dict):
        raise TypeError("Loader config must be provided and of type dictionary")

    loader_type = config.get("dataset_type", None)
    if loader_type is None:
        raise ValueError(
            f"Loader config missing required 'dataset_type' key. Supported values: {SUPPORTED_DATASET_TYPES}"
        )

    # NOTE: can be extended in the future to support more loader types, based on datasets used
    if loader_type == "dhi":
        return DHIDataLoader(**config)
    else:
        raise ValueError(f"Unsupported loader type: {loader_type}. Supported types: {SUPPORTED_DATASET_TYPES}.")


def build_preprocessor(config: Dict[str, Any]) -> DataPreprocessor:
    """
    Factory function to build a data preprocessor based on the provided configuration, depending on dataset.

    :param config: A dictionary containing configuration parameters for the preprocessor.
    :return: An instance of a DataPreprocessor.
    """
    if config is None or not isinstance(config, dict):
        raise TypeError("Preprocessor config must be provided and of type dictionary")

    preprocessor_type = config.get("dataset_type", None)
    if preprocessor_type is None:
        raise ValueError(
            f"Preprocessor config missing required 'dataset_type' key. Supported values: {SUPPORTED_DATASET_TYPES}"
        )

    # NOTE: can be extended in the future to support more preprocessor types, based on datasets used
    if preprocessor_type == "dhi":
        return DHIDataPreprocessor(**config)
    else:
        raise ValueError(
            f"Unsupported preprocessor type: {preprocessor_type}. Supported types: {SUPPORTED_DATASET_TYPES}"
        )
