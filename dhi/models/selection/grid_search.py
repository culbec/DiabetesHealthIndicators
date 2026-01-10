import json
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Set, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

import dhi.constants as dconst
from dhi.utils import get_logger
from dhi.models.eval.scorer import Scorer
from dhi.models.selection.cross_validation import (
    KFoldCVSplitter,
    StratifiedKFoldCVSplitter,
)
from dhi.data.factory import build_preprocessor
from dhi.data.preprocessing._base import DataPreprocessor

METRIC_ALIAS_MAP: Mapping[str, str] = {
    "rmse": "root_mean_squared_error",
    "mse": "mean_squared_error",
    "mae": "mean_absolute_error",
}

METRIC_MINIMIZE_SET: Set[str] = {
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

METRIC_MAXIMIZE_SET: Set[str] = {
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


def _metric_optimization_direction(task_type: str, metric: str, explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)

    if metric in METRIC_MINIMIZE_SET:
        return False
    if metric in METRIC_MAXIMIZE_SET:
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
    def __init__(
        self,
        *,
        model_cls: Any,
        task_type: str,
        preprocessor_config: Optional[Dict[str, Any]],
        base_params: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        cv_params: Optional[Dict[str, Any]] = None,
    ):
        self.logger = get_logger(self.__class__.__name__)

        self.model_cls = model_cls
        self.task_type = task_type
        self.base_params = dict(base_params or {})
        self.param_grid = dict(param_grid or {})
        self.cv_params = dict(cv_params or {})

        default_refit_metric = "f1" if task_type == "classification" else "rmse"
        self.refit_metric_ = str(self.cv_params.get("refit_metric", default_refit_metric)).lower()
        self.refit_metric_ = METRIC_ALIAS_MAP.get(self.refit_metric_, self.refit_metric_)
        self.greater_is_better_ = _metric_optimization_direction(
            task_type, self.refit_metric_, self.cv_params.get("greater_is_better")
        )

        self.n_splits = int(self.cv_params.get("n_splits", 5))
        self.shuffle = bool(self.cv_params.get("shuffle", True))
        self.random_state = self.cv_params.get("random_state", None)
        self.n_jobs = self.cv_params.get("n_jobs", 1)

        self._preprocessor_config = preprocessor_config

        self.cv_results_: List[CVParamSearchResult] = []
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = float("-inf") if self.greater_is_better_ else float("inf")
        self.best_estimator_: Optional[dconst.ModelType] = None
        self.fitted_preprocessor_: Optional[DataPreprocessor] = None

        if self.task_type not in {"classification", "regression"}:
            raise ValueError(
                f"Unsupported task_type: {self.task_type}. Supported types are 'classification' and 'regression'."
            )

    def _build_splitter(self) -> KFoldCVSplitter | StratifiedKFoldCVSplitter:
        if self.task_type == "classification":
            return StratifiedKFoldCVSplitter(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

        return KFoldCVSplitter(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

    def _build_estimator(self, params: Dict[str, Any]) -> dconst.ModelType:
        all_params = {**self.base_params, **params}
        estimator = self.model_cls(**all_params)
        return estimator

    def _is_better_score(self, score: float, best: float) -> bool:
        return score > best if self.greater_is_better_ else score < best

    def _evaluate_candidate(
        self,
        params: Dict[str, Any],
        candidate_idx: int,
        total_candidates: int,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> CVParamSearchResult:
        """
        Evaluates a single parameter configuration using cross-validation.

        This method is designed to be called in parallel by joblib.

        :param params: Parameter configuration to evaluate
        :param candidate_idx: Index of the current candidate (1-based, for logging)
        :param total_candidates: Total number of candidates (for logging)
        :param X: Input data as pandas DataFrame (for preprocessor compatibility)
        :param y: Target array
        :return: CVParamSearchResult with fold scores and metrics
        """
        self.logger.info("Evaluating candidate %d/%d parameters:", candidate_idx, total_candidates)
        self.logger.info("Candidate parameters: %s", params)

        X_arr, y_arr = np.asarray(X), np.asarray(y)

        splitter = self._build_splitter()
        fold_scores: List[float] = []
        fold_metrics: List[Dict[str, float]] = []

        split_iter = splitter.split(X_arr, y_arr)
        for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
            tr = _as_index_array(train_idx)
            val = _as_index_array(val_idx)

            self.logger.info(
                "Fold %d/%d: training on %d samples, validating on %d samples",
                fold_idx,
                self.n_splits,
                len(tr),
                len(val),
            )

            preprocessor = build_preprocessor(self._preprocessor_config)

            X_tr = X.iloc[tr].copy()
            X_val = X.iloc[val].copy()

            X_tr_fold = np.asarray(preprocessor.fit_transform(X_tr))
            X_val_fold = np.asarray(preprocessor.transform(X_val))

            y_tr_fold, y_val_fold = y_arr[tr], y_arr[val]

            estimator = self._build_estimator(params)
            estimator.fit(X_tr_fold, y_tr_fold)

            y_pred = estimator.predict(X_val_fold)

            predict_proba = getattr(estimator, "predict_proba", None)
            y_proba = (
                np.asarray(predict_proba(X_val_fold))
                if predict_proba is not None and callable(predict_proba)
                else None
            )

            scorer = Scorer(estimator, self.task_type)
            metrics = scorer.score(y_val_fold.ravel(), y_pred.ravel(), y_proba)

            if self.refit_metric_ not in metrics:
                raise KeyError(f"Refit metric '{self.refit_metric_}' not found in computed metrics: {metrics.keys()}")

            refit_score = metrics.get(self.refit_metric_)
            if refit_score is None:
                continue
            score = float(refit_score)
            fold_scores.append(score)
            safe_metrics: Dict[str, float] = {k: float(v) for k, v in metrics.items() if v is not None}
            fold_metrics.append(safe_metrics)

            self.logger.info("Fold %d metrics: %s", fold_idx, json.dumps(safe_metrics, indent=2, sort_keys=True))
            self.logger.info("Fold %d %s score: %.6f", fold_idx, self.refit_metric_, score)

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

        # Use print for real-time output in parallel workers (logger doesn't work across processes)
        if self.n_jobs != 1:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(
                f"  [Candidate {candidate_idx}/{total_candidates}] {params_str} "
                f": {self.refit_metric_}={mean_score:.6f} +/- {std_score:.6f}"
            )
        else:
            self.logger.info(
                "Candidate %d mean %s score: %.6f +/- %.6f",
                candidate_idx,
                self.refit_metric_,
                mean_score,
                std_score,
            )

        return CVParamSearchResult(
            params=params,
            fold_scores=fold_scores,
            mean_score=mean_score,
            std_score=std_score,
            fold_metrics=fold_metrics,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GridSearchCVOptimizer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input data X must be a pandas DataFrame for preprocessing"
            )  # Enforcing DataFrame type for now for preprocessing consistency

        candidates = _expand_param_grid(self.param_grid)
        total_candidates = len(candidates)

        self.logger.info(
            "Starting Grid Search optimization: %d candidate parameter sets, %d-fold cross-validation, n_jobs=%s",
            total_candidates,
            self.n_splits,
            self.n_jobs,
        )

        # Parallel evaluation of parameter candidates using joblib
        # n_jobs = -1 uses all available CPUs, n_jobs = 1 runs sequentially
        # verbose = 10 shows progress for each completed task
        parallel_verbose = 10 if self.n_jobs != 1 else 0
        results = cast(
            List[CVParamSearchResult],
            Parallel(n_jobs=self.n_jobs, verbose=parallel_verbose)(
                delayed(self._evaluate_candidate)(params, idx, total_candidates, X, np.asarray(y))
                for idx, params in enumerate(candidates, start=1)
            ),
        )

        # Collect results and find the best parameters
        # Log results here since worker process logs may not be visible
        for idx, result in enumerate(results, start=1):
            self.cv_results_.append(result)

            self.logger.info(
                "Candidate %d/%d: params=%s, mean_%s=%.6f +/- %.6f",
                idx,
                total_candidates,
                result.params,
                self.refit_metric_,
                result.mean_score,
                result.std_score,
            )

            if self._is_better_score(result.mean_score, self.best_score_):
                self.best_score_ = result.mean_score
                self.best_params_ = dict(result.params)

        self.logger.info(
            "Grid Search completed. Best %s score: %.6f with parameters: %s",
            self.refit_metric_,
            self.best_score_,
            self.best_params_,
        )

        self.fitted_preprocessor_ = build_preprocessor(self._preprocessor_config)

        X_arr = np.asarray(self.fitted_preprocessor_.fit_transform(X))
        y_arr = np.asarray(y)

        self.best_estimator_ = self._build_estimator(self.best_params_)
        self.best_estimator_.fit(X_arr, y_arr)

        self.logger.info("Best estimator refitted on full dataset: X = %s, y = %s", X_arr.shape, y_arr.shape)

        return self

    def flatten_cv_results(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for run in self.cv_results_:
            results.append(
                {
                    "params": run.params,
                    "mean_score": run.mean_score,
                    "std_score": run.std_score,
                    "fold_scores": run.fold_scores,
                }
            )

        return results
