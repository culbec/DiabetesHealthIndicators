"""
Grid Search Cross-Validation Optimizer.

This module implements exhaustive hyperparameter search with k-fold cross-validation.
It supports both single parameter grids and lists of grids for kernel-specific
parameter configurations. Parallel evaluation is supported via joblib.

The optimizer integrates with the statistics module to provide comprehensive
statistical analysis of cross-validation results, including confidence intervals
and normality testing.
"""

import json
import os
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
from dhi.statistics.metrics import CVFoldStatistics, analyze_cv_fold_scores

# Mapping of common metric aliases to their canonical names
METRIC_ALIAS_MAP: Mapping[str, str] = {
    "rmse": "root_mean_squared_error",
    "mse": "mean_squared_error",
    "mae": "mean_absolute_error",
}

# Metrics where lower values indicate better performance
METRIC_MINIMIZE_SET: Set[str] = {
    # Regression metrics
    "max_error",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "root_mean_squared_error",
    "mean_squared_log_error",
    "root_mean_squared_log_error",
    # Classification metrics
    "log_loss",
}

# Metrics where higher values indicate better performance
METRIC_MAXIMIZE_SET: Set[str] = {
    # Regression metrics
    "r2",
    "explained_variance",
    # Classification metrics
    "accuracy",
    "f1",
    "precision",
    "recall",
    "fbeta",
    "roc_auc",
    "average_precision",
}


def _metric_optimization_direction(task_type: str, metric: str, explicit: Optional[bool]) -> bool:
    """
    Determine whether a metric should be maximized or minimized.

    The optimization direction is determined by:
    1. Explicit override if provided
    2. Known metric sets (METRIC_MINIMIZE_SET, METRIC_MAXIMIZE_SET)
    3. Fallback based on task type (regression minimizes, classification maximizes)

    :param str task_type: Either "regression" or "classification".
    :param str metric: Name of the metric to optimize.
    :param Optional[bool] explicit: Optional explicit override (True=maximize, False=minimize).
    :return bool: True if metric should be maximized, False if minimized.
    :raises ValueError: If task_type is unknown and no explicit direction given.
    """
    if explicit is not None:
        return bool(explicit)

    if metric in METRIC_MINIMIZE_SET:
        return False
    if metric in METRIC_MAXIMIZE_SET:
        return True

    # Fallback: regression typically minimizes error, classification maximizes accuracy
    if task_type == "regression":
        return False
    elif task_type == "classification":
        return True
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _expand_single_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expand a single parameter grid into all combinations (Cartesian product).

    Given {"a": [1, 2], "b": [3, 4]}, produces:
    [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]

    :param Dict[str, List[Any]] param_grid: Dictionary mapping parameter names to lists of values.
    :return List[Dict[str, Any]]: List of dictionaries, each representing one parameter combination.
    :raises TypeError: If param_grid is not a dict or values are not lists.
    """
    if not isinstance(param_grid, dict):
        raise TypeError("param_grid must be a dictionary")

    keys = list(param_grid.keys())
    values = []
    for k in keys:
        v = param_grid[k]
        if not isinstance(v, list):
            raise TypeError(f"param_grid['{k}'] must be a list")
        values.append(v)

    # Generate Cartesian product of all parameter values
    combos: List[Dict[str, Any]] = []
    for items in product(*values):
        combos.append(dict(zip(keys, items)))
    return combos


def _expand_param_grid(
    param_grid: Dict[str, List[Any]] | List[Dict[str, List[Any]]],
) -> List[Dict[str, Any]]:
    """
    Expand parameter grid(s) into candidate parameter combinations.

    Supports two input formats:
    1. Single grid (dict): Standard Cartesian product expansion
    2. List of grids (list[dict]): Each grid expanded separately, results combined

    The list format is useful for kernel-specific parameters where certain
    parameters only apply to specific kernel types (e.g., degree for poly kernel).

    :param Dict[str, List[Any]] | List[Dict[str, List[Any]]] param_grid: Single dict or list of dicts defining the search space.
    :return List[Dict[str, Any]]: List of all parameter combinations to evaluate.
    """
    if isinstance(param_grid, list):
        # Combine results from multiple grids (union of search spaces)
        all_combos: List[Dict[str, Any]] = []
        for grid in param_grid:
            all_combos.extend(_expand_single_param_grid(grid))
        return all_combos

    return _expand_single_param_grid(param_grid)


def _as_index_array(idxs: ArrayLike) -> np.ndarray:
    """
    Convert index array to integer indices, handling boolean masks.

    :param ArrayLike idxs: Array-like of indices (integers or boolean mask).
    :return np.ndarray: Integer numpy array of indices.
    """
    arr = np.asarray(idxs)
    if arr.dtype == bool:
        # Convert boolean mask to integer indices
        return np.flatnonzero(arr)
    return arr.astype(int, copy=False)


@dataclass
class CVParamSearchResult:
    """
    Result container for a single parameter configuration evaluation.

    Stores all information from cross-validation of one hyperparameter
    combination, including per-fold scores, aggregate statistics, and
    comprehensive statistical analysis.

    Attributes:
    ------------------------------
        params: The hyperparameter configuration that was evaluated.
        fold_scores: List of scores from each CV fold for the refit metric.
        mean_score: Arithmetic mean of fold scores.
        std_score: Sample standard deviation of fold scores.
        fold_metrics: Complete metrics dictionary for each fold.
        statistics: Comprehensive statistical analysis from the statistics module.
    """

    params: Dict[str, Any]
    fold_scores: List[float]
    mean_score: float
    std_score: float
    fold_metrics: List[Dict[str, float]]
    statistics: Optional[CVFoldStatistics] = None

    def asdict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        :return: Dictionary representation of the result.
        """
        result = {
            "params": self.params,
            "fold_scores": self.fold_scores,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "fold_metrics": self.fold_metrics,
        }
        if self.statistics is not None:
            result["statistics"] = self.statistics.asdict()
        return result


class GridSearchCVOptimizer:
    """
    Exhaustive hyperparameter search with cross-validation.

    Evaluates all combinations of hyperparameters using k-fold cross-validation,
    tracking performance metrics and statistical analysis for each configuration.
    Supports parallel evaluation via joblib for improved performance.

    The optimizer automatically selects the best parameters based on the
    specified refit metric and refits the model on the full dataset.

    Attributes:
        model_cls: The model class to instantiate for each evaluation.
        task_type: Either "regression" or "classification".
        base_params: Default parameters passed to every model instance.
        param_grid: Search space (dict or list of dicts).
        cv_params: Cross-validation configuration (n_splits, shuffle, etc.).
        cv_results_: List of results for all evaluated configurations.
        best_params_: Parameters of the best-performing configuration.
        best_score_: Score achieved by the best configuration.
        best_estimator_: Model fitted with best parameters on full data.
    """

    def __init__(
        self,
        *,
        model_cls: Any,
        task_type: str,
        preprocessor_config: Optional[Dict[str, Any]],
        base_params: Dict[str, Any],
        param_grid: Dict[str, List[Any]] | List[Dict[str, List[Any]]],
        cv_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the grid search optimizer.

        :param Any model_cls: Model class to instantiate (must have fit/predict).
        :param str task_type: Either "regression" or "classification".
        :param Dict[str, Any] base_params: Default parameters for all model instances.
        :param Dict[str, List[Any]] | List[Dict[str, List[Any]]] param_grid: Parameter search space.
        :param Optional[Dict[str, Any]] cv_params: CV configuration (n_splits, shuffle, etc.).
        :param Optional[Dict[str, Any]] preprocessor_config: Optional preprocessor configuration.
        """
        self.logger = get_logger(self.__class__.__name__)

        self.model_cls = model_cls
        self.task_type = task_type
        self.base_params = dict(base_params or {})
        self.param_grid = param_grid if param_grid else {}
        self.cv_params = dict(cv_params or {})

        # Resolve the metric to optimize and its direction
        default_refit_metric = "f1" if task_type == "classification" else "rmse"
        self.refit_metric_ = str(self.cv_params.get("refit_metric", default_refit_metric)).lower()
        self.refit_metric_ = METRIC_ALIAS_MAP.get(self.refit_metric_, self.refit_metric_)
        self.greater_is_better_ = _metric_optimization_direction(
            task_type, self.refit_metric_, self.cv_params.get("greater_is_better")
        )

        # CV configuration
        self.n_splits = int(self.cv_params.get("n_splits", 5))
        self.shuffle = bool(self.cv_params.get("shuffle", True))
        self.random_state = self.cv_params.get("random_state", None)
        self.n_jobs = self.cv_params.get("n_jobs", 1)

        self._preprocessor_config = preprocessor_config

        # Reserve one CPU core when using all available cores
        # This prevents system unresponsiveness during long searches
        if self.n_jobs == -1:
            cpu_count = os.cpu_count()
            self.n_jobs = max(1, cpu_count - 1) if cpu_count is not None else self.n_jobs

        # Initialize result storage
        self.cv_results_: List[CVParamSearchResult] = []
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = float("-inf") if self.greater_is_better_ else float("inf")
        self.best_estimator_: Optional[dconst.ModelType] = None
        self.fitted_preprocessor_: Optional[DataPreprocessor] = None

        # Validate task type
        if self.task_type not in {"classification", "regression"}:
            raise ValueError(
                f"Unsupported task_type: {self.task_type}. Supported types are 'classification' and 'regression'."
            )

    def _build_splitter(self) -> KFoldCVSplitter | StratifiedKFoldCVSplitter:
        """
        Create the appropriate CV splitter based on task type.

        Uses stratified splitting for classification to maintain class
        distribution across folds, standard k-fold for regression.

        :return KFoldCVSplitter | StratifiedKFoldCVSplitter: Configured CV splitter instance.
        """
        if self.task_type == "classification":
            return StratifiedKFoldCVSplitter(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

        return KFoldCVSplitter(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

    def _build_estimator(self, params: Dict[str, Any]) -> dconst.ModelType:
        """
        Create a model instance with merged parameters.

        Combines base_params with the candidate params, with candidate
        params taking precedence in case of conflicts.

        :param Dict[str, Any] params: Candidate hyperparameters to evaluate.
        :return dconst.ModelType: Configured model instance.
        """
        all_params = {**self.base_params, **params}
        estimator = self.model_cls(**all_params)
        return estimator

    def _is_better_score(self, score: float, best: float) -> bool:
        """
        Compare scores according to optimization direction.

        :param float score: Candidate score to evaluate.
        :param float best: Current best score.
        :return bool: True if score is better than best.
        """
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
        Evaluate a single parameter configuration using cross-validation.

        This method is designed for parallel execution via joblib. It performs
        complete k-fold CV for one hyperparameter combination and computes
        comprehensive statistical analysis of the results.

        :param Dict[str, Any] params: Hyperparameter configuration to evaluate.
        :param int candidate_idx: 1-based index for progress logging.
        :param int total_candidates: Total number of configurations for logging.
        :param pd.DataFrame X: Features as DataFrame (for preprocessor compatibility).
        :param np.ndarray y: Target values as numpy array.
        :return CVParamSearchResult: With all evaluation data.
        """
        self.logger.info("Evaluating candidate %d/%d parameters:", candidate_idx, total_candidates)
        self.logger.info("Candidate parameters: %s", params)

        X_arr, y_arr = np.asarray(X), np.asarray(y)

        splitter = self._build_splitter()
        fold_scores: List[float] = []
        fold_metrics: List[Dict[str, float]] = []

        # Iterate through CV folds
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

            # Fit preprocessor on training data only to prevent data leakage
            preprocessor = build_preprocessor(self._preprocessor_config)

            X_tr = X.iloc[tr].copy()
            X_val = X.iloc[val].copy()

            X_tr_fold = np.asarray(preprocessor.fit_transform(X_tr))
            X_val_fold = np.asarray(preprocessor.transform(X_val))

            y_tr_fold, y_val_fold = y_arr[tr], y_arr[val]

            # Train and evaluate model
            estimator = self._build_estimator(params)
            estimator.fit(X_tr_fold, y_tr_fold)

            y_pred = estimator.predict(X_val_fold)

            # Get probability predictions if available (for classification metrics)
            predict_proba = getattr(estimator, "predict_proba", None)
            y_proba = (
                np.asarray(predict_proba(X_val_fold))
                if predict_proba is not None and callable(predict_proba)
                else None
            )

            # Compute all metrics for this fold
            scorer = Scorer(estimator, self.task_type)
            metrics = scorer.score(
                y_val_fold.ravel(),
                y_pred.ravel(),
                y_proba,
            )

            # Validate that the refit metric was computed
            if self.refit_metric_ not in metrics:
                raise KeyError(f"Refit metric '{self.refit_metric_}' not found in computed metrics: {metrics.keys()}")

            refit_score = metrics.get(self.refit_metric_)
            if refit_score is None:
                continue

            score = float(refit_score)
            fold_scores.append(score)

            # Store all metrics, filtering out None values
            safe_metrics: Dict[str, float] = {k: float(v) for k, v in metrics.items() if v is not None}
            fold_metrics.append(safe_metrics)

            self.logger.info(
                "Fold %d metrics: %s",
                fold_idx,
                json.dumps(safe_metrics, indent=2, sort_keys=True),
            )
            self.logger.info("Fold %d %s score: %.6f", fold_idx, self.refit_metric_, score)

        # Compute comprehensive statistical analysis using the statistics module
        # This includes confidence intervals, normality testing, and descriptive stats
        if fold_scores:
            cv_statistics = analyze_cv_fold_scores(fold_scores, confidence_level=0.95)
            mean_score = cv_statistics.descriptive.mean
            std_score = cv_statistics.descriptive.std
        else:
            cv_statistics = None
            mean_score = float("-inf")
            std_score = 0.0

        # Output progress: use print for parallel workers (logger not visible)
        if self.n_jobs != 1:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            ci_str = ""
            if cv_statistics is not None:
                ci = cv_statistics.confidence_interval
                ci_str = f", 95% CI: [{ci.lower:.6f}, {ci.upper:.6f}]"
            print(
                f"  [Candidate {candidate_idx}/{total_candidates}]"
                f": {self.refit_metric_}={mean_score:.6f} +/- {std_score:.6f}{ci_str}\nparams=({params_str})"
            )
        else:
            self.logger.info(
                "Candidate %d mean %s score: %.6f +/- %.6f",
                candidate_idx,
                self.refit_metric_,
                mean_score,
                std_score,
            )
            if cv_statistics is not None:
                ci = cv_statistics.confidence_interval
                self.logger.info(
                    "Candidate %d/%d  %s=%0.6f +/- %0.6f, 95%% CI: [%.6f, %.6f]\nparams=%s",
                    candidate_idx,
                    total_candidates,
                    self.refit_metric_,
                    mean_score,
                    std_score,
                    ci.lower,
                    ci.upper,
                    params,
                )

        return CVParamSearchResult(
            params=params,
            fold_scores=fold_scores,
            mean_score=mean_score,
            std_score=std_score,
            fold_metrics=fold_metrics,
            statistics=cv_statistics,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GridSearchCVOptimizer":
        """
        Run grid search cross-validation and fit the best model.

        Evaluates all parameter combinations using k-fold CV, selects the
        best configuration based on the refit metric, and trains a final
        model on the complete dataset.

        :param ArrayLike X: Feature matrix (array-like or DataFrame).
        :param ArrayLike y: Target vector (array-like).
        :return GridSearchCVOptimizer: Self, for method chaining.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input data X must be a pandas DataFrame for preprocessing"
            )  # Enforcing DataFrame type for now for preprocessing consistency

        # Expand parameter grid to get all candidates
        candidates = _expand_param_grid(self.param_grid)
        total_candidates = len(candidates)

        self.logger.info(
            "Starting Grid Search optimization: %d candidate parameter sets, %d-fold cross-validation, n_jobs=%s",
            total_candidates,
            self.n_splits,
            self.n_jobs,
        )

        # Parallel evaluation using joblib
        # n_jobs = -1 uses all available CPUs, n_jobs = 1 runs sequentially
        # verbose=10 provides per-task progress updates
        parallel_verbose = 10 if self.n_jobs != 1 else 0
        results = cast(
            List[CVParamSearchResult],
            Parallel(n_jobs=self.n_jobs, verbose=parallel_verbose)(
                delayed(self._evaluate_candidate)(params, idx, total_candidates, X, np.asarray(y))
                for idx, params in enumerate(candidates, start=1)
            ),
        )

        # Collect results and identify best configuration
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

        # Refit best model on full dataset
        self.fitted_preprocessor_ = build_preprocessor(self._preprocessor_config)

        X_arr = np.asarray(self.fitted_preprocessor_.fit_transform(X))
        y_arr = np.asarray(y)

        self.best_estimator_ = self._build_estimator(self.best_params_)
        self.best_estimator_.fit(X_arr, y_arr)

        self.logger.info(
            "Best estimator refitted on full dataset: X = %s, y = %s",
            X_arr.shape,
            y_arr.shape,
        )

        return self

    def get_best_statistics(self) -> Optional[CVFoldStatistics]:
        """
        Retrieve statistical analysis for the best parameter configuration.

        :return Optional[CVFoldStatistics]: CVFoldStatistics for best params, or None if not available.
        """
        for run in self.cv_results_:
            if run.params == self.best_params_:
                return run.statistics
        return None

    def get_statistics_report(self) -> Dict[str, Any]:
        """
        Generate a human-readable report of the best model's CV statistics.

        Includes descriptive statistics, confidence intervals, and normality
        test results formatted for easy reading.

        :return Dict[str, Any]: Dictionary containing the statistics report.
        """
        best_stats = self.get_best_statistics()
        if best_stats is None:
            return {}

        result = {
            "refit_metric": self.refit_metric_,
            "best_params": self.best_params_,
            "best_statistics": best_stats.asdict(),
        }

        return result

    def flatten_cv_results(self, include_statistics: bool = True) -> List[Dict[str, Any]]:
        """
        Convert CV results to a list of dictionaries for serialization.

        Useful for exporting results to JSON or other formats.

        :param bool include_statistics: Whether to include full statistical analysis.
        :return List[Dict[str, Any]]: List of result dictionaries.
        """
        results: List[Dict[str, Any]] = []
        for run in self.cv_results_:
            result_dict: Dict[str, Any] = {
                "params": run.params,
                "mean_score": run.mean_score,
                "std_score": run.std_score,
                "fold_scores": run.fold_scores,
            }
            if include_statistics and run.statistics is not None:
                result_dict["statistics"] = run.statistics.asdict()
            results.append(result_dict)

        return results
