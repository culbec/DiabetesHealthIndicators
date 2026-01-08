import json
import pathlib
from copy import deepcopy
from typing import Mapping, Optional

import joblib
import numpy as np
from numpy.typing import ArrayLike

import dhi.constants as dconst
from dhi.decorators import time_func
from dhi.models.eval.scorer import Scorer
from dhi.models.selection.grid_search import GridSearchCVOptimizer
from dhi.utils import get_logger


class ExperimentRunner:
    def __init__(self, **kwargs) -> None:
        self._init_from_kwargs(**kwargs)

    def _init_from_kwargs(self, **kwargs) -> None:
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
        assert isinstance(self._param_grid, dict), "param_grid must be a dictionary"

        # Initializing the model instance
        self._model: dconst.ModelType = self._model_cls(**self._params)
        if self._load_path is not None:
            self._load_model()

        required_methods = ("fit", "predict")
        for method in required_methods:
            fn = getattr(self._model, method, None)
            if not callable(fn):
                raise TypeError(f"The model {self._model_name} must implement the method '{method}()'")

        self.logger = get_logger(self._model.__class__.__name__)

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

        self._model = joblib.load(self._load_path)

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
        X_, y_ = np.asarray(X), np.asarray(y)

        if with_cv:
            self.logger.info(
                f"Fitting {self._model_name} on data with shape {X_.shape} and target with shape {y_.shape} with cross-validation and hyperparameter optimization"
            )
            self.logger.info(
                f"Performing cross-validation with params: {self._cv_params}\nand hyperparameter optimization grid: {self._param_grid}"
            )

            optimizer = GridSearchCVOptimizer(
                model_cls=self._model_cls,
                task_type=self._task_type,
                base_params=self._params,
                param_grid=self._param_grid,
                cv_params=self._cv_params,
            )
            optimizer.fit(X_, y_)

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
        else:
            self.logger.info(
                f"Fitting {self._model_name} on data with shape {X_.shape} and target with shape {y_.shape} without cross-validation"
            )
            self._model.fit(X_, y_)
            self.logger.info(f"Fitted model: {self._model}")

        self._save_model()

    @time_func
    def predict(self, X: ArrayLike) -> np.ndarray:
        return self._model.predict(X)

    @time_func
    def predict_proba(self, X: ArrayLike) -> Optional[np.ndarray]:
        predict_proba = getattr(self._model, "predict_proba", None)
        if predict_proba is None or not callable(predict_proba):
            self.logger.warning(
                f"Model {self._model_name} is not compatible with expected method 'predict_proba'. Unable to perform prediction."
            )
            return None

        pred = predict_proba(X)
        return np.asarray(pred) if pred is not None else pred
