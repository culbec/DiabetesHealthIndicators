from itertools import product
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from typing import Optional, List, Dict, Any

from dhi.models.eval.scorer import Scorer
from dhi.models.selection.cross_validation import KFoldCVSplitter, StratifiedKFoldCVSplitter


def _metric_optimization_direction(task_type: str, metric: str, explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)

    metric_name = metric.lower()
    alias_map = {
        "rmse": "root_mean_squared_error",
        "mse": "mean_squared_error",
        "mae": "mean_absolute_error",
    }  # Support for common aliases if using different scoring methods
    metric_name = alias_map.get(metric_name, metric_name)

    minimize = {
        # Regression metrics where lower is better
        "max_error",
        "mean_absolute_error",
        "mean_squared_error",
        "median_absolute_error",
        "root_mean_squared_error",
        "mean_squared_log_error",
        "root_mean_squared_log_error",
        # Classification metrics where lower is better
        "log_loss",
    }

    maximize = {
        # Regression metrics where higher is better
        "r2",
        "explained_variance",
        # Classification metrics where higher is better
        "accuracy",
        "f1",
        "precision",
        "recall",
        "fbeta",
        "roc_auc",
        "average_precision",
    }

    if metric_name in minimize:
        return False
    if metric_name in maximize:
        return True

    # Fallback assumption based on task type
    if task_type == "regression":
        return False
    elif task_type == "classification":
        return True
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _expand_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not isinstance(param_grid, dict):
        raise TypeError("param_grid must be a dictionary")

    keys = list(param_grid.keys())
    values = []
    for k in keys:
        v = param_grid[k]
        if not isinstance(v, list):
            raise TypeError(f"param_grid['{k}'] must be a list")
        values.append(v)

    combos: List[Dict[str, Any]] = []
    for items in product(*values):
        combos.append(dict(zip(keys, items)))
    return combos


def _as_index_array(idxs: ArrayLike) -> np.ndarray:
    arr = np.asarray(idxs)
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    return arr.astype(int, copy=False)


@dataclass
class CVParamSearchResult:
    params: Dict[str, Any]
    fold_scores: List[float]
    mean_score: float
    std_score: float
    fold_metrics: List[Dict[str, float]]


class GridSearchCVOptimizer:
    def __init__(self,
                 *,
                 model_cls: Any,
                 base_params: Dict[str, Any],
                 param_grid: Dict[str, List[Any]],
                 cv_params: Optional[Dict[str, Any]] = None,
                 task_type: str = "classification"
                 ):
        self.model_cls = model_cls
        self.task_type = task_type
        self.base_params = dict(base_params or {})
        self.param_grid = dict(param_grid or {})
        self.cv_params = dict(cv_params or {})

        self.refit_metric_ = str(self.cv_params.get("refit_metric", "f1" if task_type == "classification" else "rmse"))
        self.greater_is_better_ = _metric_optimization_direction(task_type, self.refit_metric_,
                                                                 self.cv_params.get("greater_is_better"))

        self.n_splits = int(self.cv_params.get("n_splits", 5))
        self.shuffle = bool(self.cv_params.get("shuffle", True))
        self.random_state = self.cv_params.get("random_state", None)

        self.cv_results_: List[CVParamSearchResult] = []
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = float('-inf') if self.greater_is_better_ else float('inf')
        self.best_estimator_: Optional[BaseEstimator] = None

        if self.task_type not in {"classification", "regression"}:
            raise ValueError(
                f"Unsupported task_type: {self.task_type}. Supported types are 'classification' and 'regression'.")

    def _build_splitter(self) -> KFoldCVSplitter | StratifiedKFoldCVSplitter:
        if self.task_type == "classification":
            return StratifiedKFoldCVSplitter(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )

        return KFoldCVSplitter(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

    def _build_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        all_params = {**self.base_params, **params}
        estimator = self.model_cls(**all_params)
        return estimator

    def _is_better_score(self, score: float, best: float) -> bool:
        return score > best if self.greater_is_better_ else score < best

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GridSearchCVOptimizer":
        # TODO: add data preprocessing step here, so scalers would be fitted on training fold only and applied to validation fold
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        splitter = self._build_splitter()
        candidates = _expand_param_grid(self.param_grid)

        for params in candidates:
            fold_scores: List[float] = []
            fold_metrics: List[Dict[str, float]] = []

            split_iter = splitter.split(X_arr, y_arr)
            for train_idx, val_idx in split_iter:
                tr = _as_index_array(train_idx)
                val = _as_index_array(val_idx)

                estimator = self._build_estimator(params)
                estimator.fit(X_arr[tr], y_arr[tr])

                y_pred = estimator.predict(X_arr[val])
                y_proba = estimator.predict_proba(X_arr[val]) if callable(
                    getattr(estimator, "predict_proba", None)) else None

                scorer = Scorer(estimator, self.task_type)
                metrics = scorer.score(np.asarray(y_arr[val]).ravel(), np.asarray(y_pred).ravel(), y_proba)

                if self.refit_metric_ not in metrics:
                    raise KeyError(
                        f"Refit metric '{self.refit_metric_}' not found in computed metrics: {metrics.keys()}")

                score = float(metrics[self.refit_metric_])
                fold_scores.append(score)
                fold_metrics.append(metrics)

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

            self.cv_results_.append(
                CVParamSearchResult(
                    params=params,
                    fold_scores=fold_scores,
                    mean_score=mean_score,
                    std_score=std_score,
                    fold_metrics=fold_metrics
                )
            )

            if self._is_better_score(mean_score, self.best_score_):
                self.best_score_ = mean_score
                self.best_params_ = dict(params)

        self.best_estimator_ = self._build_estimator(self.best_params_)
        self.best_estimator_.fit(X_arr, y_arr)

        return self

    def flatten_cv_results(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for run in self.cv_results_:
            results.append({
                "params": run.params,
                "mean_score": run.mean_score,
                "std_score": run.std_score,
                "fold_scores": run.fold_scores,
            })

        return results
