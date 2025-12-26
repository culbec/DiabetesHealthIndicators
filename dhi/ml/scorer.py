import pandas as pd
import numpy as np

import sklearn.metrics as skm

from sklearn.base import BaseEstimator

from dhi.utils import get_logger


class Scorer(object):
    def __init__(self, model: BaseEstimator, task_type: str) -> None:
        self.logger = get_logger(self.__class__.__name__)

        self.model = model
        self.task_type = task_type

    def _regression_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_proba: np.ndarray | None = None) -> dict[str, float | None]:
        # Source: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        metrics = {}

        # Some metrics are only relevant for positive values
        if np.all(y_true >= 0) and np.all(y_pred >= 0):
            metrics["mean_squared_log_error"] = skm.mean_squared_log_error(y_true, y_pred)
            metrics["root_mean_squared_log_error"] = skm.root_mean_squared_log_error(y_true, y_pred)

        metrics.update(
            {
                "max_error": skm.max_error(y_true, y_pred),
                "mean_absolute_error": skm.mean_absolute_error(y_true, y_pred),
                "mean_squared_error": skm.mean_squared_error(y_true, y_pred),
                "median_absolute_error": skm.median_absolute_error(y_true, y_pred),
                "root_mean_squared_error": skm.root_mean_squared_error(y_true, y_pred),
                "r2": skm.r2_score(y_true, y_pred),
                "explained_variance": skm.explained_variance_score(y_true, y_pred),
            }
        )

        return metrics

    def _classification_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_proba: np.ndarray | None = None) -> dict[str, float | None]:
        # Source: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        is_multi_class = len(np.unique(y_true)) > 2

        return {
            "accuracy": skm.accuracy_score(y_true, y_pred),
            "average_precision": skm.average_precision_score(y_true, y_proba) if not is_multi_class and y_proba is not None else None,
            "f1": skm.f1_score(y_true, y_pred) if not is_multi_class else None,
            "f1_micro": skm.f1_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "f1_macro": skm.f1_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "f1_weighted": skm.f1_score(y_true, y_pred, average="weighted") if is_multi_class else None,
            "log_loss": skm.log_loss(y_true, y_pred),
            "precision": skm.precision_score(y_true, y_pred) if not is_multi_class else None,
            "precision_micro": skm.precision_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "precision_macro": skm.precision_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "precision_weighted": skm.precision_score(y_true, y_pred, average="weighted") if is_multi_class else None,
            "recall": skm.recall_score(y_true, y_pred) if not is_multi_class else None,
            "recall_micro": skm.recall_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "recall_macro": skm.recall_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "recall_weighted": skm.recall_score(y_true, y_pred, average="weighted") if is_multi_class else None,
            "roc_auc": skm.roc_auc_score(y_true, y_proba) if y_proba is not None else None,
        }

    def score(self, y_true: pd.Series, y_pred: pd.Series, y_proba: np.ndarray | None = None) -> dict[str, float]:
        if self.task_type == "regression":
            return self._regression_metrics(y_true, y_pred, y_proba)
        elif self.task_type == "classification":
            return self._classification_metrics(y_true, y_pred, y_proba)
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
