import json
import pandas as pd
import pathlib
import numpy as np
import skops.io as sio

from sklearn.model_selection import GridSearchCV

import dhi.constants as dconst
from dhi.decorators import time_func
from dhi.utils import get_logger
from dhi.ml.scorer import Scorer


class ModelWrapper(object):
    def __init__(self, **kwargs) -> None:
        self.init(**kwargs)

    def init(self, **kwargs) -> None:
        self.model_name = kwargs.get("model_name", None)
        assert isinstance(self.model_name, str), "model_name must be a string"
        assert (
            self.model_name in dconst.DHI_ML_MODEL_MAPPING
        ), f"model must be one of {list(dconst.DHI_ML_MODEL_MAPPING.keys())}"
        self.model, self.task_type = dconst.DHI_ML_MODEL_MAPPING[self.model_name]

        self.save_path = kwargs.get("save_path", None)
        assert isinstance(self.save_path, (str, None)), "save_path must be a string"
        self.save_path = (
            pathlib.Path(self.save_path) / f"{self.model_name}.pkl" if self.save_path else None
        )

        self.metrics_path = kwargs.get("metrics_path", None)
        assert isinstance(self.metrics_path, (str, None)), "metrics_path must be a string"
        self.metrics_path = (
            pathlib.Path(self.metrics_path) / f"{self.model_name}_metrics.json"
            if self.metrics_path
            else None
        )

        self.params = kwargs.get("params", {})
        assert isinstance(self.params, dict), "params must be a dictionary"

        self.cv_params = kwargs.get("cv_params", {})
        assert isinstance(self.cv_params, dict), "cv_params must be a dictionary"
        
        self.param_grid = self.cv_params.pop("param_grid", {})
        assert isinstance(self.param_grid, dict), "param_grid must be a dictionary"

        # Initializing the model
        self.model = self.model(**self.params)

        self.logger = get_logger(self.model.__class__.__name__)

    @time_func
    def _save_model(self) -> None:
        if not self.save_path:
            self.logger.warning("No save path provided, model will not be saved")
            return

        if not self.save_path.parent.exists():
            self.logger.warning(f"Save path parent directory does not exist, creating it: {self.save_path.parent}")
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

        sio.dump(self.model, self.save_path)
        self.logger.info(f"Model saved to {self.save_path}")

    @time_func
    def fit(self, X: pd.DataFrame, y: pd.Series, with_cv: bool = False) -> None:
        """
        Fits the instance model to the passed data.

        The `with_cv` parameter indicates if the model should be fitted with cross-validation.

        The cross-validation is performed using `GridSearchCV`, attempting to also
        optimize the hyperparameters of the model.

        :param pd.DataFrame X: The input data
        :param pd.Series y: The target data
        :param bool with_cv: Whether to perform cross-validation, defaults to False
        """
        if with_cv:
            self.logger.info(
                f"Fitting {self.model_name} on data with shape {X.shape} and target with shape {y.shape} with cross-validation"
            )
            self.logger.info(f"Performing cross-validation with params: {self.cv_params}\nand param_grid: {self.param_grid}")
            
            grid = GridSearchCV(self.model, self.param_grid, **self.cv_params)
            grid.fit(X, y)

            self.logger.info(f"Best estimator: {grid.best_estimator_}")
            self.model = grid.best_estimator_
        else:
            self.logger.info(
                f"Fitting {self.model_name} on data with shape {X.shape} and target with shape {y.shape} without cross-validation"
            )
            self.model.fit(X, y)

        self._save_model()

    @time_func
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    @time_func
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    @time_func
    def score(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        y_pred = self.predict(X)

        scorer = Scorer(self.model, self.task_type)
        metrics = scorer.score(y, y_pred)

        self.logger.info(f"Metrics of {self.model_name}: {metrics}")

        if self.metrics_path:
            if not self.metrics_path.parent.exists():
                self.logger.warning(
                    f"Metrics path parent directory does not exist, creating it: {self.metrics_path.parent}"
                )
                self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Metrics dumped to {self.metrics_path}")

        return metrics
