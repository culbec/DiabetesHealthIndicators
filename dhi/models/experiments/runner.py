import json
import pathlib
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError

import dhi.constants as dconst
from dhi.decorators import time_func
from dhi.data.factory import build_preprocessor
from dhi.utils import get_logger, validate_param_grid

from dhi.models.eval.scorer import Scorer
from dhi.models.selection.grid_search import GridSearchCVOptimizer


class ExperimentRunner:
    def __init__(self, model_config: Dict[str, Any], preprocessor_config: Dict[str, Any]) -> None:
        self._init_model(**model_config)

        self._preprocessor = None # TODO: handle case where model is loaded from pretrained path and preprocessor may remain undefined and unfitted on training data
        self._preprocessor_config = preprocessor_config

    def _init_model(self, **kwargs) -> None:
        self._is_fitted = False

        self._model_name = kwargs.get("model_name", None)
        assert self._model_name is not None and isinstance(self._model_name, str), "model_name must be a string"
        assert (
            self._model_name in dconst.DHI_ML_MODEL_REGISTRY
        ), f"model_name must be one of {list(dconst.DHI_ML_MODEL_REGISTRY.keys())}"
        self._model_cls, self._task_type = dconst.DHI_ML_MODEL_REGISTRY[self._model_name]

        self._save_path = kwargs.get("save_path", None)
        assert self._save_path is None or isinstance(
            self._save_path, (str, pathlib.Path)
        ), "save_path must be a string or pathlib.Path"
        self._save_path = (
            (pathlib.Path(self._save_path) / f"{self._model_name}.pkl") if self._save_path is not None else None
        )

        self._metrics_path = kwargs.get("metrics_path", None)
        assert self._metrics_path is None or isinstance(
            self._metrics_path, (str, pathlib.Path)
        ), "metrics_path must be a string or pathlib.Path"
        self._metrics_path = (
            (pathlib.Path(self._metrics_path) / f"{self._model_name}_metrics.json")
            if self._metrics_path is not None
            else None
        )

        self._cv_statistics_path = kwargs.get("cv_statistics_path", None)
        assert self._cv_statistics_path is None or isinstance(
            self._cv_statistics_path, (str, pathlib.Path)
        ), "cv_statistics_path must be a string or pathlib.Path"
        self._cv_statistics_path = (
            (pathlib.Path(self._cv_statistics_path) / f"{self._model_name}_cv_statistics.json")
            if self._cv_statistics_path is not None
            else None
        )

        self._load_path = kwargs.get("load_path", None)
        assert self._load_path is None or isinstance(
            self._load_path, (str, pathlib.Path)
        ), "load_path must be a string or pathlib.Path"
        self._load_path = (
            (pathlib.Path(self._load_path) / f"{self._model_name}.pkl") if self._load_path is not None else None
        )

        self._params = kwargs.get("params", {})
        assert isinstance(self._params, dict), "params must be a dictionary"

        self._cv_params = deepcopy(kwargs.get("cv_params", {}))
        assert isinstance(self._cv_params, dict), "cv_params must be a dictionary"

        self._param_grid = self._cv_params.pop("param_grid", {})
        # Validate param_grid format (supports dict or list of dicts for kernel-specific params)
        validate_param_grid(self._param_grid, "param_grid")

        # Initializing the model instance
        self._model: dconst.ModelType = self._model_cls(**self._params)

        self.logger = get_logger(f"{self.__class__.__name__}.{self._model.__class__.__name__}")

        if self._load_path is not None:
            self._load_model()

        required_methods = ("fit", "predict")
        for method in required_methods:
            fn = getattr(self._model, method, None)
            if not callable(fn):
                raise TypeError(f"The model {self._model_name} must implement the method '{method}()'")

    @property
    def model_name(self) -> str:
        return str(self._model_name)

    @property
    def model(self) -> dconst.ModelType:
        return self._model

    @time_func
    def _save_model(self) -> None:
        if not self._save_path:
            self.logger.warning("No save path provided, model will not be saved")
            return

        if not self._save_path.parent.exists():
            self.logger.warning(f"Save path parent directory does not exist, creating it: {self._save_path.parent}")
            self._save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(self._model, self._save_path)
            self.logger.info(f"Model saved to {self._save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {self._save_path}: {e}")

    @time_func
    def _load_model(self) -> None:
        if not (self._load_path is not None and self._load_path.exists()):
            raise FileNotFoundError(f"Model file not found at {self._load_path}")

        try:
            self._model = joblib.load(self._load_path)
            self.logger.info(f"Model loaded from {self._load_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {self._load_path}: {e}")
            raise

    @time_func
    def score(self, X: ArrayLike, y: ArrayLike) -> Mapping[str, Optional[float]]:
        y_true = np.asarray(y).ravel()
        y_pred, y_proba = self.predict(X), self.predict_proba(X)

        scorer = Scorer(self._model, self._task_type)
        try:
            metrics = scorer.score(y_true, y_pred, y_proba)
        except Exception as e:
            self.logger.error(f"Failed to compute evaluation metrics: {e}")
            metrics = {}

        self.logger.info(f"Metrics of {self._model_name}: {metrics}")

        if metrics and self._metrics_path:
            if not self._metrics_path.parent.exists():
                self.logger.warning(
                    f"Metrics path parent directory does not exist, creating it: {self._metrics_path.parent}"
                )
                self._metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Evaluation metrics dumped to {self._metrics_path}")

        return metrics

    @time_func
    def fit(self, X: ArrayLike, y: ArrayLike, with_cv: bool = False) -> None:
        """
        Fits the instance model to the passed data.

        The `with_cv` parameter indicates if the model should be fitted with cross-validation.
        The cross-validation is performed together with hyperparameter optimization using Grid Search.

        :param ArrayLike X: The input data
        :param ArrayLike y: The target data
        :param bool with_cv: Whether to perform cross-validation, defaults to False
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input data X must be a pandas DataFrame for preprocessing"
            )  # Enforcing DataFrame type for now for preprocessing consistency

        if with_cv:
            self.logger.info(
                f"Fitting {self._model_name} on data with shape {X.shape} and target with shape {y.shape} with cross-validation and hyperparameter optimization"
            )
            self.logger.info(
                f"Performing cross-validation with params: {self._cv_params}\nand hyperparameter optimization grid: {self._param_grid}"
            )

            optimizer = GridSearchCVOptimizer(
                model_cls=self._model_cls,
                task_type=self._task_type,
                preprocessor_config=self._preprocessor_config,
                base_params=self._params,
                param_grid=self._param_grid,
                cv_params=self._cv_params,
            )
            optimizer.fit(X, y)

            self.logger.info(f"Best hyperparameters: {optimizer.best_params_}")
            self.logger.info(f"Best cross-validation {optimizer.refit_metric_} score: {optimizer.best_score_}")

            assert (
                optimizer.best_estimator_ is not None
            ), f"The best estimator returned by {GridSearchCVOptimizer.__name__} cannot be None"

            if not isinstance(optimizer.best_estimator_, dconst.ModelType):
                raise ValueError(
                    f"The best estimator returned by {GridSearchCVOptimizer.__name__} has an unsupported type"
                )
            self._model = optimizer.best_estimator_
            self._preprocessor = optimizer.fitted_preprocessor_

            cv_statistics_report = optimizer.get_statistics_report()
            self.logger.info(f"Cross-validation statistics report: {cv_statistics_report}")
            if self._cv_statistics_path:
                with open(self._cv_statistics_path, "w") as f:
                    json.dump(cv_statistics_report, f, indent=4)
                self.logger.info(f"Cross-validation statistics dumped to {self._cv_statistics_path}")
        else:
            self.logger.info(
                f"Fitting {self._model_name} on data with shape {X.shape} and target with shape {y.shape} without cross-validation"
            )

            self._preprocessor = build_preprocessor(self._preprocessor_config)

            X_ = np.asarray(self._preprocessor.fit_transform(X))
            y_ = np.asarray(y)

            self._model.fit(X_, y_)
            self.logger.info(f"Fitted model: {self._model}")

        self._is_fitted = True
        self._save_model()

    @time_func
    def predict(self, X: ArrayLike) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError(f"Model {self._model_name} is not fitted yet. Call 'fit' before using the model.")

        X_ = np.asarray(self._preprocessor.transform(X))
        return self._model.predict(X_)

    @time_func
    def predict_proba(self, X: ArrayLike) -> Optional[np.ndarray]:
        if not self._is_fitted:
            raise NotFittedError(f"Model {self._model_name} is not fitted yet. Call 'fit' before using the model.")

        predict_proba = getattr(self._model, "predict_proba", None)
        if predict_proba is None or not callable(predict_proba):
            self.logger.warning(
                f"Model {self._model_name} is not compatible with expected method 'predict_proba'. Unable to perform prediction."
            )
            return None

        X_ = np.asarray(self._preprocessor.transform(X))
        pred = predict_proba(X_)
        return np.asarray(pred) if pred is not None else pred
