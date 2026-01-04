import pandas as pd
import numpy as np
from typing import Literal, Optional
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator

import dhi.constants as dconst
from dhi.utils import get_logger


class Scorer:
    """Scorer class for computing regression and classification metrics."""

    def __init__(self, model: BaseEstimator, task_type: str) -> None:
        """
        Initializes the Scorer.

        :param BaseEstimator model: The model to score.
        :param str task_type: The type of task, either 'regression' or 'classification'.
        """
        self.logger = get_logger(self.__class__.__name__)

        self.model = model
        self.task_type = task_type

    def _max_error(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the maximum absolute error between true and predicted values.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return float: The maximum absolute error.
        """
        return float(np.max(np.abs(y_true - y_pred)))

    def _mean_absolute_error(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the mean absolute error between true and predicted values.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return float: The mean absolute error.
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    def _mean_squared_error(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the mean squared error between true and predicted values.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return float: The mean squared error.
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def _median_absolute_error(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the median absolute error between true and predicted values.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return float: The median absolute error.
        """
        return float(np.median(np.abs(y_true - y_pred)))

    def _root_mean_squared_error(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the root mean squared error between true and predicted values.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return float: The root mean squared error.
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _mean_squared_log_error(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the mean squared logarithmic error between true and predicted values.

        :param ArrayLike y_true: The true labels (must be non-negative).
        :param ArrayLike y_pred: The predicted labels (must be non-negative).
        :return float: The mean squared logarithmic error.
        """
        return float(np.mean((np.log(1 + y_true) - np.log(1 + y_pred)) ** 2))

    def _root_mean_squared_log_error(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the root mean squared logarithmic error between true and predicted values.

        :param ArrayLike y_true: The true labels (must be non-negative).
        :param ArrayLike y_pred: The predicted labels (must be non-negative).
        :return float: The root mean squared logarithmic error.
        """
        return float(np.sqrt(np.mean((np.log(1 + y_true) - np.log(1 + y_pred)) ** 2)))

    def _mean_absolute_percentage_error(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the mean absolute percentage error between true and predicted values.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return float: The mean absolute percentage error.
        """
        return float(np.mean(np.abs(y_true - y_pred) / np.max(np.finfo(float).eps, np.abs(y_true))))

    def _r2_score(self, y_true: ArrayLike, y_pred: ArrayLike, force_finite: bool = True) -> float:
        """
        Computes the R-squared (coefficient of determination) score.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :param bool force_finite: If True, returns -inf when denominator is 0; otherwise returns nan.
        :return float: The R-squared score.
        """
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)

        if not force_finite:
            return float("nan") if denominator == 0 else float(1 - numerator / denominator)
        else:
            return float("-inf") if denominator == 0 else float(1 - numerator / denominator)

    def _explained_variance_score(self, y_true: ArrayLike, y_pred: ArrayLike, force_finite: bool = True) -> float:
        """
        Computes the explained variance score.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :param bool force_finite: If True, returns -inf when denominator is 0; otherwise returns nan.
        :return float: The explained variance score.
        """
        numerator = np.var(y_true - y_pred)
        denominator = np.var(y_true)

        if not force_finite:
            return float("nan") if denominator == 0 else float(1 - numerator / denominator)
        else:
            return float("-inf") if denominator == 0 else float(1 - numerator / denominator)

    def _compute_regression_metrics(self, y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float | None]:
        """
        Computes all regression metrics.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return dict[str, float | None]: Dictionary containing all regression metrics.
        """
        metrics = {
            "max_error": self._max_error(y_true, y_pred),
            "mean_absolute_error": self._mean_absolute_error(y_true, y_pred),
            "mean_squared_error": self._mean_squared_error(y_true, y_pred),
            "median_absolute_error": self._median_absolute_error(y_true, y_pred),
            "root_mean_squared_error": self._root_mean_squared_error(y_true, y_pred),
            "r2": self._r2_score(y_true, y_pred),
            "explained_variance": self._explained_variance_score(y_true, y_pred),
        }

        if np.all(y_true >= 0) and np.all(y_pred >= 0):
            metrics["mean_squared_log_error"] = self._mean_squared_log_error(y_true, y_pred)
            metrics["root_mean_squared_log_error"] = self._root_mean_squared_log_error(y_true, y_pred)

        return metrics

    def _accuracy_score(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Computes the accuracy score.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :return float: The accuracy score.
        """
        return float(np.mean(y_true == y_pred))

    def _f1_score(
        self, y_true: ArrayLike, y_pred: ArrayLike, average: Literal["binary", "micro", "macro", "weighted"] = "binary"
    ) -> float:
        """
        Computes the F1 score.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :param Literal["binary", "micro", "macro", "weighted"] average: The averaging method.
        :return float: The F1 score.
        :raises ValueError: If average is 'binary' for multi-class data.
        """
        unique_classes = np.unique(y_true)

        if len(unique_classes) > 2 and average not in ["micro", "macro", "weighted"]:
            raise ValueError(
                "F1 score on multi-class data cannot be 'binary' solved, use 'micro', 'macro', or 'weighted' average instead"
            )

        y_true, y_pred = np.astype(y_true, int), np.astype(y_pred, int)

        if average == "binary":
            tp = np.sum((y_true == unique_classes[0]) & (y_pred == unique_classes[0]))
            fp = np.sum((y_true == unique_classes[0]) & (y_pred == unique_classes[1]))
            fn = np.sum((y_true == unique_classes[1]) & (y_pred == unique_classes[0]))

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0

            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            return float(f1)
        elif average in ["micro", "macro", "weighted"]:
            f1_scores, supports = [], []
            for cls in unique_classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))

                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                f1_scores.append(f1)
                supports.append(np.sum(y_true == cls))

            f1_scores, supports = np.asarray(f1_scores), np.asarray(supports)
            if average == "micro":
                tp = np.sum([(y_true == cls).astype(int) & (y_pred == cls).astype(int) for cls in unique_classes])
                fp = np.sum([((y_true != cls).astype(int) & (y_pred == cls).astype(int)) for cls in unique_classes])
                fn = np.sum([((y_true == cls).astype(int) & (y_pred != cls).astype(int)) for cls in unique_classes])
                return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            elif average == "macro":
                return float(np.mean(f1_scores))
            elif average == "weighted":
                if np.sum(supports) == 0:
                    return 0.0
                return float(np.average(f1_scores, weights=supports))
        else:
            raise ValueError(f"Invalid average: {average}")

    def _fbeta_score(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        beta: float,
        average: Literal["binary", "micro", "macro", "weighted"] = "binary",
    ) -> float:
        """
        Computes the F-beta score.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :param float beta: The beta parameter weighting recall vs precision.
        :param Literal["binary", "micro", "macro", "weighted"] average: The averaging method.
        :return float: The F-beta score.
        :raises ValueError: If average is 'binary' for multi-class data.
        """
        unique_classes = np.unique(y_true)

        if len(unique_classes) > 2 and average not in ["micro", "macro", "weighted"]:
            raise ValueError(
                "F1 score on multi-class data cannot be 'binary' solved, use 'micro', 'macro', or 'weighted' average instead"
            )

        y_true, y_pred = np.astype(y_true, int), np.astype(y_pred, int)

        if average == "binary":
            return float(self._f1_score(y_true, y_pred))

        elif average in ["micro", "macro", "weighted"]:
            f_beta_scores, supports = [], []
            for cls in unique_classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                f_beta = (
                    (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
                    if precision + recall > 0
                    else 0.0
                )
                f_beta_scores.append(f_beta)
                supports.append(np.sum(y_true == cls))
            f_beta_scores, supports = np.asarray(f_beta_scores), np.asarray(supports)
            if average == "micro":
                tp = np.sum([(y_true == cls).astype(int) & (y_pred == cls).astype(int) for cls in unique_classes])
                fp = np.sum([((y_true != cls).astype(int) & (y_pred == cls).astype(int)) for cls in unique_classes])
                fn = np.sum([((y_true == cls).astype(int) & (y_pred != cls).astype(int)) for cls in unique_classes])

                numerator = (1 + beta**2) * tp
                denominator = (1 + beta**2 * tp) + fp + (beta**2 * fn)
                return float(numerator / denominator) if denominator > 0 else 0.0
            elif average == "macro":
                return float(np.mean(f_beta_scores))
            elif average == "weighted":
                if np.sum(supports) == 0:
                    return 0.0
                return float(np.average(f_beta_scores, weights=supports))
        else:
            raise ValueError(f"Invalid average: {average}")

    def _precision_score(
        self, y_true: ArrayLike, y_pred: ArrayLike, average: Literal["binary", "micro", "macro", "weighted"] = "binary"
    ) -> float:
        """
        Computes the precision score.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :param Literal["binary", "micro", "macro", "weighted"] average: The averaging method.
        :return float: The precision score.
        :raises ValueError: If average is 'binary' for multi-class data.
        """
        unique_classes = np.unique(y_true)

        if len(unique_classes) > 2 and average not in ["micro", "macro", "weighted"]:
            raise ValueError(
                "Precision score on multi-class data cannot be 'binary' solved, use 'micro', 'macro', or 'weighted' average instead"
            )

        y_true, y_pred = np.astype(y_true, int), np.astype(y_pred, int)

        if average == "binary":
            tp = np.sum((y_true == unique_classes[0]) & (y_pred == unique_classes[0]))
            fp = np.sum((y_true == unique_classes[0]) & (y_pred == unique_classes[1]))
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            return float(precision)
        elif average in ["micro", "macro", "weighted"]:
            precision_scores, supports = [], []
            for cls in unique_classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                precision_scores.append(precision)
                supports.append(np.sum(y_true == cls))
            precision_scores, supports = np.asarray(precision_scores), np.asarray(supports)
            if average == "micro":
                tp = np.sum([(y_true == cls).astype(int) & (y_pred == cls).astype(int) for cls in unique_classes])
                fp = np.sum([((y_true != cls).astype(int) & (y_pred == cls).astype(int)) for cls in unique_classes])
                return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            elif average == "macro":
                return float(np.mean(precision_scores))
            elif average == "weighted":
                if np.sum(supports) == 0:
                    return 0.0
                return float(np.average(precision_scores, weights=supports))
        else:
            raise ValueError(f"Invalid average: {average}")

    def _recall_score(
        self, y_true: ArrayLike, y_pred: ArrayLike, average: Literal["binary", "micro", "macro", "weighted"] = "binary"
    ) -> float:
        """
        Computes the recall score.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :param Literal["binary", "micro", "macro", "weighted"] average: The averaging method.
        :return float: The recall score.
        :raises ValueError: If average is 'binary' for multi-class data.
        """
        unique_classes = np.unique(y_true)

        if len(unique_classes) > 2 and average not in ["micro", "macro", "weighted"]:
            raise ValueError(
                "Recall score on multi-class data cannot be 'binary' solved, use 'micro', 'macro', or 'weighted' average instead"
            )

        y_true, y_pred = np.astype(y_true, int), np.astype(y_pred, int)

        if average == "binary":
            tp = np.sum((y_true == unique_classes[0]) & (y_pred == unique_classes[0]))
            fn = np.sum((y_true == unique_classes[0]) & (y_pred == unique_classes[1]))
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            return float(recall)
        elif average in ["micro", "macro", "weighted"]:
            recall_scores, supports = [], []
            for cls in unique_classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                recall_scores.append(recall)
                supports.append(np.sum(y_true == cls))
            recall_scores, supports = np.asarray(recall_scores), np.asarray(supports)
            if average == "micro":
                tp = np.sum([(y_true == cls).astype(int) & (y_pred == cls).astype(int) for cls in unique_classes])
                fn = np.sum([((y_true == cls).astype(int) & (y_pred != cls).astype(int)) for cls in unique_classes])
                return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            elif average == "macro":
                return float(np.mean(recall_scores))
            elif average == "weighted":
                if np.sum(supports) == 0:
                    return 0.0
                return float(np.average(recall_scores, weights=supports))
        else:
            raise ValueError(f"Invalid average: {average}")

    def _log_loss(self, y_true: ArrayLike, y_proba: ArrayLike, eps: float = 1e-15) -> float:
        """
        Computes the log loss (cross-entropy loss).

        Binary formula: -1/n * sum(yi * log(pi) + (1-yi) * log(1-pi))
        Multi-class formula: -1/n * sum(sum(yij * log(pij)))

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_proba: The predicted probabilities.
        :param float eps: Small value to avoid log(0), defaults to 1e-15.
        :return float: The log loss.
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        y_proba = np.clip(y_proba, eps, 1 - eps)

        unique_classes = np.unique(y_true)
        n_samples = len(y_true)

        if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 1):
            y_proba = y_proba.ravel()
            loss = -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))
        elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
            loss = -np.mean(y_true * np.log(y_proba_pos) + (1 - y_true) * np.log(1 - y_proba_pos))
        else:
            n_classes = len(unique_classes)
            y_true_onehot = np.zeros((n_samples, n_classes))
            for i, cls in enumerate(unique_classes):
                y_true_onehot[y_true == cls, i] = 1
            loss = -np.sum(y_true_onehot * np.log(y_proba)) / n_samples

        return float(loss)

    def _roc_auc_score(self, y_true: ArrayLike, y_score: ArrayLike) -> float:
        """
        Computes the ROC AUC score using the trapezoidal rule.

        ROC AUC is the area under the curve of TPR vs FPR.
        TPR = TP / (TP + FN), FPR = FP / (FP + TN)

        :param ArrayLike y_true: The true binary labels.
        :param ArrayLike y_score: The predicted scores/probabilities for the positive class.
        :return float: The ROC AUC score.
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.0

        tpr_list = [0.0]
        fpr_list = [0.0]

        tp, fp = 0, 0
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        tpr_arr = np.array(tpr_list)
        fpr_arr = np.array(fpr_list)
        auc = np.trapz(tpr_arr, fpr_arr)

        return float(auc)

    def _average_precision_score(self, y_true: ArrayLike, y_score: ArrayLike) -> float:
        """
        Computes the average precision score.

        AP = sum((Rn - Rn-1) * Pn), the area under the precision-recall curve.

        :param ArrayLike y_true: The true binary labels.
        :param ArrayLike y_score: The predicted scores/probabilities for the positive class.
        :return float: The average precision score.
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)

        if n_pos == 0:
            return 0.0

        tp = 0
        precision_list = []
        recall_list = []

        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            precision = tp / (i + 1)
            recall = tp / n_pos
            precision_list.append(precision)
            recall_list.append(recall)

        precision_arr = np.array(precision_list)
        recall_arr = np.array(recall_list)

        recall_diff = np.diff(np.concatenate([[0], recall_arr]))
        ap = np.sum(recall_diff * precision_arr)

        return float(ap)

    def _compute_classification_metrics(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_proba: Optional[ArrayLike] = None,
        beta: float = dconst.DHI_DEFAULT_F_BETA_SCORE_BETA,
    ) -> dict[str, float | None]:
        """
        Computes all classification metrics.

        :param ArrayLike y_true: The true labels.
        :param ArrayLike y_pred: The predicted labels.
        :param Optional[ArrayLike] y_proba: The predicted probabilities, defaults to None.
        :param float beta: The beta parameter for the F-beta score, defaults to DHI_DEFAULT_F_BETA_SCORE_BETA.
        :return dict[str, float | None]: Dictionary containing all classification metrics.
        """
        is_multi_class = len(np.unique(y_true)) > 2
        beta = max(beta, 0.0)

        metrics = {
            "accuracy": self._accuracy_score(y_true, y_pred),
            "f1": self._f1_score(y_true, y_pred) if not is_multi_class else None,
            "f1_micro": self._f1_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "f1_macro": self._f1_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "f1_weighted": self._f1_score(y_true, y_pred, average="weighted") if is_multi_class else None,
            "precision": self._precision_score(y_true, y_pred) if not is_multi_class else None,
            "precision_micro": self._precision_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "precision_macro": self._precision_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "precision_weighted": (
                self._precision_score(y_true, y_pred, average="weighted") if is_multi_class else None
            ),
            "recall": self._recall_score(y_true, y_pred) if not is_multi_class else None,
            "recall_micro": self._recall_score(y_true, y_pred, average="micro") if is_multi_class else None,
            "recall_macro": self._recall_score(y_true, y_pred, average="macro") if is_multi_class else None,
            "recall_weighted": self._recall_score(y_true, y_pred, average="weighted") if is_multi_class else None,
        }

        if beta != 1.0:
            if metrics > 0.0:
                metrics["fbeta"] = self._fbeta_score(y_true, y_pred, beta) if not is_multi_class else None
                metrics["fbeta_micro"] = (
                    self._fbeta_score(y_true, y_pred, beta, average="micro") if is_multi_class else None
                )
                metrics["fbeta_macro"] = (
                    self._fbeta_score(y_true, y_pred, beta, average="macro") if is_multi_class else None
                )
                metrics["fbeta_weighted"] = (
                    self._fbeta_score(y_true, y_pred, beta, average="weighted") if is_multi_class else None
                )

        if y_proba is not None:
            metrics["log_loss"] = self._log_loss(y_true, y_proba)

            if not is_multi_class:
                y_score = y_proba[:, 1] if (y_proba.ndim == 2 and y_proba.shape[1] == 2) else y_proba
                metrics["roc_auc"] = self._roc_auc_score(y_true, y_score)
                metrics["average_precision"] = self._average_precision_score(y_true, y_score)
            else:
                unique_classes = np.unique(y_true)
                roc_auc_scores = []
                for i, cls in enumerate(unique_classes):
                    y_true_binary = (y_true == cls).astype(int)
                    y_score_cls = y_proba[:, i]
                    roc_auc_scores.append(self._roc_auc_score(y_true_binary, y_score_cls))
                metrics["roc_auc"] = float(np.mean(roc_auc_scores))
                metrics["average_precision"] = None

        return metrics

    def score(
        self,
        y_true: pd.Series | ArrayLike,
        y_pred: pd.Series | ArrayLike,
        y_proba: Optional[pd.Series | ArrayLike] = None,
        beta: float = dconst.DHI_DEFAULT_F_BETA_SCORE_BETA,
    ) -> dict[str, float]:
        """
        Computes all metrics based on the task type.

        :param pd.Series | ArrayLike y_true: The true labels.
        :param pd.Series | ArrayLike y_pred: The predicted labels.
        :param Optional[pd.Series | ArrayLike] y_proba: The predicted probabilities, defaults to None.
        :param float beta: The beta parameter for the F-beta score, defaults to DHI_DEFAULT_F_BETA_SCORE_BETA.
        :return dict[str, float]: Dictionary containing all computed metrics.
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
