from typing import Mapping, Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

import dhi.constants as dconst
import dhi.models.eval.metrics as em
from dhi.utils import get_logger


class Scorer:
    """Scorer class for computing regression and classification metrics."""

    def __init__(self, model: dconst.ModelType, task_type: str) -> None:
        """
        Initializes the Scorer.

        :param BaseEstimator model: The model to score.
        :param str task_type: The type of task, either 'regression' or 'classification'.
        """
        self.logger = get_logger(self.__class__.__name__)

        self.model = model
        self.task_type = task_type

    def _compute_regression_metrics(self, y_true: ArrayLike, y_pred: ArrayLike) -> Mapping[str, Optional[float]]:
        """
        Computes all regression metrics.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return Mapping[str, Optional[float]]: Dictionary containing all regression metrics.
        """
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

        metrics = {
            "max_error": em.max_error(y_true, y_pred),
            "mean_absolute_error": em.mean_absolute_error(y_true, y_pred),
            "mean_absolute_percentage_error": em.mean_absolute_percentage_error(y_true, y_pred),
            "mean_squared_error": em.mean_squared_error(y_true, y_pred),
            "median_absolute_error": em.median_absolute_error(y_true, y_pred),
            "root_mean_squared_error": em.root_mean_squared_error(y_true, y_pred),
            "normalized_root_mean_squared_error": em.normalized_root_mean_squared_error(y_true, y_pred),
            "r2": em.r2_score(y_true, y_pred),
            "explained_variance": em.explained_variance_score(y_true, y_pred),
        }

        if np.all(y_true >= 0) and np.all(y_pred >= 0):
            metrics["mean_squared_log_error"] = em.mean_squared_log_error(y_true, y_pred)
            metrics["root_mean_squared_log_error"] = em.root_mean_squared_log_error(y_true, y_pred)

        return metrics

    def _compute_classification_metrics(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_proba: Optional[ArrayLike] = None,
        beta: float = dconst.DHI_DEFAULT_F_BETA_SCORE_BETA,
    ) -> Mapping[str, Optional[float]]:
        """
        Computes all classification metrics.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :param Optional[ArrayLike] y_proba: The predicted probabilities, defaults to None.
        :param float beta: The beta parameter for the F-beta score, defaults to DHI_DEFAULT_F_BETA_SCORE_BETA.
        :return Mapping[str, Optional[float]]: Dictionary containing all classification metrics.
        """
        is_multi_class = len(np.unique(y_true)) > 2
        beta = max(beta, 0.0)

        y_true, y_pred, y_proba = (
            np.asarray(y_true),
            np.asarray(y_pred),
            np.asarray(y_proba) if y_proba is not None else None,
        )

        metrics = {
            "accuracy": em.accuracy_score(y_true, y_pred),
            "f1": em.f1_score(y_true, y_pred) if not is_multi_class else None,
            "f1_micro": (em.f1_score(y_true, y_pred, average="micro") if is_multi_class else None),
            "f1_macro": (em.f1_score(y_true, y_pred, average="macro") if is_multi_class else None),
            "f1_weighted": (em.f1_score(y_true, y_pred, average="weighted") if is_multi_class else None),
            "precision": (em.precision_score(y_true, y_pred) if not is_multi_class else None),
            "precision_micro": (em.precision_score(y_true, y_pred, average="micro") if is_multi_class else None),
            "precision_macro": (em.precision_score(y_true, y_pred, average="macro") if is_multi_class else None),
            "precision_weighted": (em.precision_score(y_true, y_pred, average="weighted") if is_multi_class else None),
            "recall": em.recall_score(y_true, y_pred) if not is_multi_class else None,
            "recall_micro": (em.recall_score(y_true, y_pred, average="micro") if is_multi_class else None),
            "recall_macro": (em.recall_score(y_true, y_pred, average="macro") if is_multi_class else None),
            "recall_weighted": (em.recall_score(y_true, y_pred, average="weighted") if is_multi_class else None),
        }

        if beta > 0.0 and not np.isclose(beta, 1.0):
            metrics["fbeta"] = em.fbeta_score(y_true, y_pred, beta) if not is_multi_class else None
            metrics["fbeta_micro"] = em.fbeta_score(y_true, y_pred, beta, average="micro") if is_multi_class else None
            metrics["fbeta_macro"] = em.fbeta_score(y_true, y_pred, beta, average="macro") if is_multi_class else None
            metrics["fbeta_weighted"] = (
                em.fbeta_score(y_true, y_pred, beta, average="weighted") if is_multi_class else None
            )

        if y_proba is not None:
            metrics["log_loss"] = em.log_loss(y_true, y_proba)

            if not is_multi_class:
                y_score = y_proba[:, 1] if (y_proba.ndim == 2 and y_proba.shape[1] == 2) else y_proba
                metrics["roc_auc"] = em.roc_auc_score(y_true, y_score)
                metrics["average_precision"] = em.average_precision_score(y_true, y_score)
            else:
                unique_classes = np.unique(y_true)
                roc_auc_scores = []
                for i, cls in enumerate(unique_classes):
                    y_true_binary = (y_true == cls).astype(int)
                    y_score_cls = y_proba[:, i]
                    roc_auc_scores.append(em.roc_auc_score(y_true_binary, y_score_cls))
                metrics["roc_auc"] = float(np.mean(roc_auc_scores))
                metrics["average_precision"] = None

        return metrics

    def score(
        self,
        y_true: pd.Series | ArrayLike,
        y_pred: pd.Series | ArrayLike,
        y_proba: Optional[pd.Series | ArrayLike] = None,
        beta: float = dconst.DHI_DEFAULT_F_BETA_SCORE_BETA,
    ) -> Mapping[str, Optional[float]]:
        """
        Computes all metrics based on the task type.

        :param pd.Series | ArrayLike y_true: The true labels.
        :param pd.Series | ArrayLike y_pred: The predicted labels.
        :param Optional[pd.Series | ArrayLike] y_proba: The predicted probabilities, defaults to None.
        :param float beta: The beta parameter for the F-beta score, defaults to DHI_DEFAULT_F_BETA_SCORE_BETA.
        :return Mapping[str, Optional[float]]: Dictionary containing all computed metrics.
        :raises ValueError: If task_type is not 'regression' or 'classification'.
        """
        y_true, y_pred, y_proba = (
            np.asarray(y_true).ravel(),
            np.asarray(y_pred).ravel(),
            np.asarray(y_proba) if y_proba is not None else None,
        )

        if self.task_type == "regression":
            return self._compute_regression_metrics(y_true, y_pred)
        elif self.task_type == "classification":
            return self._compute_classification_metrics(y_true, y_pred, y_proba, beta)
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
