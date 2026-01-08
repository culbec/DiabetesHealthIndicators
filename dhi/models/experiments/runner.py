import json
import joblib
import pathlib
import numpy as np

from copy import deepcopy
from typing import Optional
from numpy.typing import ArrayLike

import dhi.constants as dconst

from dhi.utils import get_logger
from dhi.decorators import time_func
from dhi.models.eval.scorer import Scorer
from dhi.models.selection.grid_search import GridSearchCVOptimizer


class ExperimentRunner:
    def __init__(self, **kwargs) -> None:
        self._init_from_kwargs(**kwargs)

    def _init_from_kwargs(self, **kwargs) -> None:
        self.model_name = kwargs.get("model_name", None)
        assert isinstance(self.model_name, str), "model_name must be a string"
        assert (
                self.model_name in dconst.DHI_ML_MODEL_REGISTRY
        ), f"model_name must be one of {list(dconst.DHI_ML_MODEL_REGISTRY.keys())}"
        self.model_cls, self.task_type = dconst.DHI_ML_MODEL_REGISTRY[self.model_name]

        self.save_path = kwargs.get("save_path", None)
        assert self.save_path is None or isinstance(self.save_path, (str, pathlib.Path)), "save_path must be a string or pathlib.Path"
        self.save_path = (
                pathlib.Path(self.save_path) / f"{self.model_name}.pkl"
        ) if self.save_path is not None else None

        self.metrics_path = kwargs.get("metrics_path", None)
        assert self.metrics_path is None or isinstance(self.metrics_path, (str, pathlib.Path)), "metrics_path must be a string or pathlib.Path"
        self.metrics_path = (
                pathlib.Path(self.metrics_path) / f"{self.model_name}_metrics.json"
        ) if self.metrics_path is not None else None

        self.load_path = kwargs.get("load_path", None)
        assert self.load_path is None or isinstance(self.load_path, (str, pathlib.Path)), "load_path must be a string or pathlib.Path"
        self.load_path = (
                pathlib.Path(self.load_path) / f"{self.model_name}.pkl"
        ) if self.load_path is not None else None

        self.params = kwargs.get("params", {})
        assert isinstance(self.params, dict), "params must be a dictionary"

        self.cv_params = deepcopy(kwargs.get("cv_params", {}))
        assert isinstance(self.cv_params, dict), "cv_params must be a dictionary"

        self.param_grid = self.cv_params.pop("param_grid", {})
        assert isinstance(self.param_grid, dict), "param_grid must be a dictionary"

        # Initializing the model instance
        self.model = self.model_cls(**self.params)
        if self.load_path is not None:
            self._load_model()

        required_methods = ("fit", "predict")
        for method in required_methods:
            fn = getattr(self.model, method, None)
            if not callable(fn):
                raise TypeError(f"The model {self.model_name} must implement the method '{method}()'")

        self.logger = get_logger(self.model.__class__.__name__)

    @time_func
    def _save_model(self) -> None:
        if not self.save_path:
            self.logger.warning("No save path provided, model will not be saved")
            return

        if not self.save_path.parent.exists():
            self.logger.warning(f"Save path parent directory does not exist, creating it: {self.save_path.parent}")
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(self.model, self.save_path)
            self.logger.info(f"Model saved to {self.save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {self.save_path}: {e}")

    @time_func
    def _load_model(self) -> None:
        if not self.load_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.load_path}")

        self.model = joblib.load(self.load_path)

    @time_func
    def score(self, X: ArrayLike, y: ArrayLike) -> dict[str, float]:
        y_true = np.asarray(y).ravel()
        y_pred, y_proba = self.predict(X), self.predict_proba(X)

        scorer = Scorer(self.model, self.task_type)
        try:
            metrics = scorer.score(y_true, y_pred, y_proba)
        except Exception as e:
            self.logger.error(f"Failed to compute evaluation metrics: {e}")
            metrics = {}

        self.logger.info(f"Metrics of {self.model_name}: {metrics}")

        if metrics and self.metrics_path:
            if not self.metrics_path.parent.exists():
                self.logger.warning(
                    f"Metrics path parent directory does not exist, creating it: {self.metrics_path.parent}"
                )
                self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Evaluation metrics dumped to {self.metrics_path}")

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
        if with_cv:
            self.logger.info(
                f"Fitting {self.model_name} on data with shape {X.shape} and target with shape {y.shape} with cross-validation and hyperparameter optimization"
            )
            self.logger.info(
                f"Performing cross-validation with params: {self.cv_params}\nand hyperparameter optimization grid: {self.param_grid}")

            optimizer = GridSearchCVOptimizer(
                model_cls=self.model_cls,
                task_type=self.task_type,
                base_params=self.params,
                param_grid=self.param_grid,
                cv_params=self.cv_params,
            )
            optimizer.fit(X, y)

            self.logger.info(f"Best hyperparameters: {optimizer.best_params_}")
            self.logger.info(f"Best cross-validation {optimizer.refit_metric_} score: {optimizer.best_score_}")
            self.model = optimizer.best_estimator_
        else:
            self.logger.info(
                f"Fitting {self.model_name} on data with shape {X.shape} and target with shape {y.shape} without cross-validation"
            )
            self.model.fit(X, y)
            self.logger.info(f"Fitted model: {self.model}")

        self._save_model()

    @time_func
    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.model.predict(X)

    @time_func
    def predict_proba(self, X: ArrayLike) -> Optional[np.ndarray]:
        predict_proba = getattr(self.model, "predict_proba", None)
        if callable(predict_proba):
            return self.model.predict_proba(X)
        else:
            self.logger.warning(
                f"Model {self.model_name} is not compatible with expected method 'predict_proba'. Unable to perform prediction.")
            return None
