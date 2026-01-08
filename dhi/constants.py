import logging
import pathlib
from typing import Any, Mapping, Tuple, Type, TypeAlias, Union

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVR

from dhi.models.random_forest import RandomForest_ as RF_SCRATCH
from dhi.models.svr import SVR_ as SVR_SCRATCH

##############################
# GENERAL CONSTANTS
##############################

# Sizes in bytes
DHI_DEFAULT_SIZE_UNIT: str = "MB"
DHI_SIZES_BYTES: Mapping[str, int] = {
    "KB": 1 << 10,
    "MB": 1 << 20,
    "GB": 1 << 30,
}

DHI_SUPPORTED_EXT_FILE_TYPE: Mapping[str, str] = {
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

DHI_NUMPY_INT_RANGES: list[Tuple[int, int, type]] = [
    (np.iinfo(np.int8).min, np.iinfo(np.int8).max, np.int8),
    (np.iinfo(np.int16).min, np.iinfo(np.int16).max, np.int16),
    (np.iinfo(np.int32).min, np.iinfo(np.int32).max, np.int32),
    (np.iinfo(np.int64).min, np.iinfo(np.int64).max, np.int64),
]

DHI_NUMPY_FLOAT_RANGES: list[Tuple[np.floating, np.floating, type]] = [
    (np.finfo(np.float16).min, np.finfo(np.float16).max, np.float16),
    (np.finfo(np.float32).min, np.finfo(np.float32).max, np.float32),
    (np.finfo(np.float64).min, np.finfo(np.float64).max, np.float64),
]

##############################
# PLOT CONSTANTS
##############################

DHI_PLOT_COLOR_CONTINUOUS_SCALE: str = "PiYG"
DHI_PLOT_WIDTH: int = 1280
DHI_PLOT_HEIGHT: int = 720

DHI_PLOT_HISTOGRAM_DEFAULT_BINS: int = 10
DHI_PLOT_HISTOGRAM_MARKER: Mapping = {"line": {"width": 0.8, "color": "black"}}
DHI_PLOT_HISTOGRAM_WIDTH: int = 800
DHI_PLOT_HISTOGRAM_HEIGHT: int = 600

DHI_PLOT_DISTPLOT_KDE_LINE: Mapping = {"color": "red", "width": 2}

DHI_PLOT_SUBPLOT_COLS_PER_ROW: int = 3

##############################
# FEATURE SELECTION CONSTANTS
##############################

DHI_FEATURE_SELECTION_MODES: Mapping[str, Mapping[str, Any]] = {
    "percentile": {
        "param": 20,
    },
    "kbest": {
        "param": 10,
    },
    "fpr": {
        "param": 0.05,
    },
    "fdr": {
        "param": 0.05,
    },
    "fwe": {
        "param": 0.05,
    },
}

DHI_FEATURE_SELECTION_DEFAULT_MODE: str = "percentile"
DHI_FEATURE_SELECTION_DEFAULT_RELIEF_N_FEATURES: int = 10
DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS: int = 10
DHI_FEATURE_SELECTION_DEFAULT_VARIANCE_THRESHOLD: float = 0.2

##############################
# COMPONENT REDUCTION CONSTANTS
##############################

DHI_COMPONENT_REDUCTION_DEFAULT_N_COMPONENTS: int = 2

##############################
# EXPLAINABILITY CONSTANTS
##############################

DHI_SHAP_SUBSAMPLE_SIZE: int = 20
DHI_SHAP_WATERFALL_MAX_DISPLAY: int = 10
DHI_SHAP_BACKGROUND_SAMPLES: int = 50

DHI_DEFAULT_F_BETA_SCORE_BETA: float = 2.0

##############################
# ML CONSTANTS
##############################

ModelType: TypeAlias = SVR | SVR_SCRATCH | RF | RF_SCRATCH

# "model_name": (model_class, task_type)
DHI_ML_MODEL_REGISTRY: Mapping[str, Tuple[Type[ModelType], str]] = {
    "svr_sklearn": (SVR, "regression"),
    "svr_scratch": (SVR_SCRATCH, "regression"),
    "rf_sklearn": (RF, "classification"),
    "rf_scratch": (RF_SCRATCH, "classification"),
}

# "model_name": {"class": explainer_class, "kwargs": explainer_kwargs}
DHI_ML_EXPLAINER_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "svr_sklearn": {"class": shap.Explainer, "kwargs": {}},
    "svr_scratch": {"class": shap.Explainer, "kwargs": {}},
    "rf_sklearn": {"class": shap.Explainer, "kwargs": {}},
    # NOTE: SHAP TreeExplainer is incompatible with from-scratch RF implementation due to per-tree feature subspacing,
    # as compared to sklearn's built-in RF which uses per-split subspacing instead. Therefore, KernedExplainer will be used here
    "rf_scratch": {
        "class": shap.Explainer,  # NOTE: SHAP requires a more complex RF implementation to accept the scratch implementation
        "kwargs": {},
    },
}
