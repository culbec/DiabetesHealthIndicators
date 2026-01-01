import pandas as pd
import numpy as np
from typing import Optional

import sklearn.metrics as skm
from sklearn.base import BaseEstimator

from dhi.utils import get_logger


class Scorer:
    def __init__(self, model: BaseEstimator, task_type: str) -> None:
        self.logger = get_logger(self.__class__.__name__)

        self.model = model
        self.task_type = task_type

    def _compute_regression_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | None]:
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

    def _compute_classification_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_proba: Optional[np.ndarray] = None) -> dict[str, float | None]:
        # Source: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        is_multi_class = len(np.unique(y_true)) > 2

        metrics = {
            "accuracy": skm.accuracy_score(y_true, y_pred),
            "f1": skm.f1_score(y_true, y_pred) if not is_multi_class else None,
            "f1_micro": skm.f1_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "f1_macro": skm.f1_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "f1_weighted": skm.f1_score(y_true, y_pred, average="weighted") if is_multi_class else None,
            "precision": skm.precision_score(y_true, y_pred) if not is_multi_class else None,
            "precision_micro": skm.precision_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "precision_macro": skm.precision_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "precision_weighted": skm.precision_score(y_true, y_pred, average="weighted") if is_multi_class else None,
            "recall": skm.recall_score(y_true, y_pred) if not is_multi_class else None,
            "recall_micro": skm.recall_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "recall_macro": skm.recall_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "recall_weighted": skm.recall_score(y_true, y_pred, average="weighted") if is_multi_class else None,
        }

        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            metrics.update({
                "log_loss": skm.log_loss(y_true, y_proba),
            })

            if not is_multi_class:
                y_score = y_proba[:, 1] if (y_proba.ndim == 2 and y_proba.shape[1] == 2) else y_proba
                metrics.update({
                    "roc_auc": skm.roc_auc_score(y_true, y_score),
                    "average_precision": skm.average_precision_score(y_true, y_score),
                })
            else:
                metrics.update({
                    "roc_auc": skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
                    "average_precision": None # Average precision for multi-class is optional; can be added later via label binarization
                })

        return metrics


    # TODO: implement evaluation metrics from scratch
    def score(self, y_true: pd.Series, y_pred: pd.Series, y_proba: Optional[np.ndarray] = None) -> dict[str, float]:
        if self.task_type == "regression":
            return self._compute_regression_metrics(y_true, y_pred)
        elif self.task_type == "classification":
            return self._compute_classification_metrics(y_true, y_pred, y_proba)
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
