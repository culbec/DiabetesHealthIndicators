# Partial credits to: https://github.com/nihil21/custom-svm/blob/master/src/svm.py

import gc
import logging
from typing import Any, Callable, Optional, Union

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

DHI_SVR_KERNEL_TYPES: list[str] = ["linear", "poly", "rbf", "sigmoid"]
DHI_SVR_DEFAULT_KERNEL: str = "auto"
DHI_SVR_DEFAULT_C: float = 1.0
DHI_SVR_DEFAULT_EPSILON: float | str = 1e-1
DHI_SVR_EPSILON_SCALE_FACTOR: float = 0.1  # Multiplier for auto epsilon: epsilon = factor * std(y)
DHI_SVR_DEFAULT_TOL: float = 1e-3
DHI_SVR_DEFAULT_MAX_ITER: int = 1000
DHI_SVR_DEFAULT_MAX_PASSES: int = 50
DHI_SVR_GAMMA_OPTIONS: list[str] = ["scale", "auto"]
DHI_SVR_DEFAULT_GAMMA: str = "scale"
DHI_SVR_DEFAULT_DEGREE: int = 3
DHI_SVR_DEFAULT_COEF0: float | None = None
DHI_SVR_DEFAULT_SHRINKING: bool = True
DHI_SVR_DEFAULT_SHRINKING_INTERVAL: int = 50
DHI_SVR_DEFAULT_KERNEL_BATCH_SIZE: int = 5000
# Auto-disable kernel caching above this sample count to prevent OOM
DHI_SVR_MAX_SAMPLES_FOR_CACHE: int = 15000
# Memory limits for chunked kernel computation
DHI_SVR_MAX_CHUNK_BYTES: int = 500 * 1024 * 1024  # 500 MB per chunk
DHI_SVR_MIN_CHUNK_BATCH_SIZE: int = 500
DHI_SVR_MAX_CHUNK_BATCH_SIZE: int = 5000

DHI_SVR_X_NEAR_ZERO: float = 1e-12


class SVRValidationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class SVRKernelError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class SVR_(BaseEstimator, RegressorMixin):
    """
    Class implementing a Support Vector Machine for the regression task.

    Maximizes the dual formulation of the SVM problem, relying on sample similarities,
    rather than explicit feature mappings from the primal formulation.

    Parameters
    ------------------------------
    kernel: {"linear", "poly", "rbf", "sigmoid"}
        The kernel to use for the SVM.
    C: float
        The regularization parameter.
    epsilon: float
        The epsilon-insensitive loss parameter.
    gamma: {"scale", "auto"} or float
        Kernel coefficient for rbf, poly, and sigmoid kernels.
        - "scale" (default): gamma = 1 / (n_features * X.var())
        - "auto": gamma = 1 / n_features
        - float: explicit gamma value
    degree: int
        The degree of the polynomial kernel. Only used if the kernel is "poly".
    coef0: float
        Independent term in kernel function. Only used if the kernel is "poly" or "sigmoid".
    epsilon: float
        The epsilon-insensitive loss parameter.
    tol: float
        The tolerance for the optimization algorithm.
    max_iter: int
        The maximum number of iterations for the optimization algorithm.
    max_passes: int
        The maximum number of passes for the optimization algorithm.
    cache_kernel: bool
        Whether to cache the kernel matrix.
    shrinking: bool
        Whether to use a shrinking heuristic in SMO (working-set reduction; speed only).
    shrinking_interval: int
        How often (in SMO iterations) to apply the shrinking heuristic. Only used if shrinking=True.
    kernel_batch_size: int
        Batch size for chunked kernel matrix computation when cache_kernel=False.
        Controls memory usage during final error computation. Lower values use less memory.
    random_state: int | None
        The random state for the optimization algorithm.
    verbose: bool
        Whether to print verbose output.

    Attributes
    ------------------------------
    X_: ArrayLike
        The input data.
    y_: ArrayLike
        The target data.
    n_samples_: int
        The number of samples in the input data.
    n_features_in_: int
        The number of features in the input data.
    kernel_func_: Callable
        The kernel function.
    kernel_matrix_: ArrayLike
        The kernel matrix, if cached.
    gamma_: float
        The resolved gamma parameter.
    degree_: int
        The resolved degree parameter.
    coef0_: float
        The resolved coef0 parameter.
    a_: ArrayLike
        Contiguous array of dual variables, positive and negative, alpha and alpha_star.
    sign_: ArrayLike
        The sign of the dual variables.
    b_: float
        The bias term.
    phi_cache_: ArrayLike
        The phi function (objective function) cache.
    errors_: ArrayLike
        Computed errors.
    alpha_: ArrayLike
        The positive coefficients.
    alpha_star_: ArrayLike
        The negative coefficients.
    dual_coef_: ArrayLike
        The difference between the dual coefficients alpha and alpha_star.
    support_: ArrayLike
        The indices of the support vectors.
    support_vectors_: ArrayLike
        The support vectors.
    kernel_stats_: dict[str, Any]
        The statistics of the kernel matrix.
    passes_: int
        The number of passes of the SMO algorithm.
    iters_: int
        The number of iterations of the SMO algorithm.
    ok_steps_: int
        The number of successful steps of the SMO algorithm.
    fail_steps_: int
        The number of failed steps of the SMO algorithm.
    active_set_: ArrayLike
        Boolean mask over dual variables used by the shrinking heuristic in SMO.
    """

    def __init__(
        self,
        *,
        C: float = DHI_SVR_DEFAULT_C,
        kernel: str = DHI_SVR_DEFAULT_KERNEL,
        gamma: float | str = DHI_SVR_DEFAULT_GAMMA,
        degree: int = DHI_SVR_DEFAULT_DEGREE,
        coef0: float | None = DHI_SVR_DEFAULT_COEF0,
        epsilon: float | str = DHI_SVR_DEFAULT_EPSILON,
        tol: float = DHI_SVR_DEFAULT_TOL,
        max_iter: int = DHI_SVR_DEFAULT_MAX_ITER,
        max_passes: int = DHI_SVR_DEFAULT_MAX_PASSES,
        cache_kernel: bool = True,
        shrinking: bool = DHI_SVR_DEFAULT_SHRINKING,
        shrinking_interval: int = DHI_SVR_DEFAULT_SHRINKING_INTERVAL,
        kernel_batch_size: int = DHI_SVR_DEFAULT_KERNEL_BATCH_SIZE,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__()

        self.C = C
        self.kernel = "linear" if kernel == "auto" else kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.epsilon = epsilon
        self.tol = tol
        self.max_iter = max_iter
        self.max_passes = max_passes
        self.cache_kernel = cache_kernel
        self.shrinking = shrinking
        self.shrinking_interval = shrinking_interval
        self.kernel_batch_size = kernel_batch_size
        self.random_state = random_state
        self.verbose = verbose

        from dhi.utils import get_logger

        self.logger = get_logger(self.__class__.__name__, level=logging.DEBUG if verbose else logging.INFO)

        self.X_: np.ndarray = np.asarray([])
        self.y_: np.ndarray = np.asarray([])

        self.n_samples_: int = 0
        self.n_features_in_: int = 0

        self.kernel_func_: Optional[
            Callable[[Union[np.number, ArrayLike], Union[np.number, ArrayLike]], Union[np.number, ArrayLike]]
        ] = None
        self.kernel_matrix_: Optional[np.ndarray] = None

        self.gamma_: Optional[float] = None
        self.degree_: Optional[int] = None
        self.coef0_: Optional[float] = None
        self.epsilon_: float = float(epsilon) if isinstance(epsilon, (int, float)) else 0.1

        self.a_: np.ndarray = np.asarray([])
        self.sign_: np.ndarray = np.asarray([])
        self.b_: float = 0.0

        self.phi_cache_: np.ndarray = np.asarray([])
        self.errors_: np.ndarray = np.asarray([])

        self.alpha_: np.ndarray = np.asarray([])
        self.alpha_star_: np.ndarray = np.asarray([])
        self.dual_coef_: np.ndarray = np.asarray([])

        self.support_: np.ndarray = np.asarray([])
        self.support_vectors_: np.ndarray = np.asarray([])
        self.support_dual_coef_: np.ndarray = np.asarray([])
        self._fitted_no_sv_: Optional[bool] = None  # None means not fitted

        self.kernel_stats_: Optional[dict[str, Any]] = None
        self.passes_: Optional[int] = None
        self.iters_: Optional[int] = None
        self.ok_steps_: Optional[int] = None
        self.fail_steps_: Optional[int] = None
        self.active_set_: Optional[np.ndarray] = None

    def __getstate__(self) -> dict:
        """
        Excludes the logger from serialization as it contains an RLock that cannot be pickled.

        :return dict: The object's state without the logger
        """
        state = self.__dict__.copy()
        # Remove logger if it exists (it contains an RLock which cannot be serialized)
        if "logger" in state:
            del state["logger"]
        # Do not serialize lambda-based kernel functions (not picklable).
        # It can be re-created from (kernel, gamma_, degree_, coef0_) after deserialization.
        if "kernel_func_" in state:
            del state["kernel_func_"]
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restores the object's state and re-initializes the logger after deserialization.

        :param dict state: The object's state
        """
        self.__dict__.update(state)
        from dhi.utils import get_logger

        self.logger = get_logger(
            self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )
        self.kernel_func_ = None

    def _resolve_gamma(self, X: Optional[ArrayLike]) -> float:
        """
        Resolve the gamma parameter based on its value.

        Supports three modes (similar to sklearn):
        - "scale": gamma = 1 / (n_features * X.var())
        - "auto": gamma = 1 / n_features
        - float: use the provided value directly

        :param X: The input data used to compute gamma for "scale" and "auto" modes.
        :return: The resolved gamma value as a float.
        """
        if X is None:
            return 0.0

        X_ = np.asarray(X, dtype=np.float64)
        n_features = X_.shape[1]

        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)

        gamma_str = str(self.gamma).lower()

        if gamma_str == "auto":
            # auto: gamma = 1 / n_features
            gamma = 1.0 / n_features
            self.logger.debug(f"Using 'auto' gamma: 1 / {n_features} = {gamma}")

        elif gamma_str == "scale":
            # scale: gamma = 1 / (n_features * X.var())
            X_var = float(np.var(X_, dtype=np.float64))
            self.logger.debug(f"Input data variance: {X_var}")

            # Prevent division by zero for very small variance
            if X_var < DHI_SVR_X_NEAR_ZERO:
                self.logger.warning(f"Data variance ({X_var}) is near zero, using gamma=1.0")
                return 1.0

            gamma = 1.0 / (n_features * X_var)
            self.logger.debug(f"Using 'scale' gamma: 1 / ({n_features} * {X_var}) = {gamma}")

        else:
            raise SVRValidationError(
                f"Invalid gamma value: '{self.gamma}'. Expected 'scale', 'auto', or a numeric value."
            )

        return float(np.clip(gamma, DHI_SVR_X_NEAR_ZERO, 1 / DHI_SVR_X_NEAR_ZERO))

    def _resolve_epsilon(self, y: ArrayLike) -> float:
        """
        Resolve the epsilon parameter.

        Supports:
        - "auto": epsilon = DHI_SVR_EPSILON_SCALE_FACTOR * std(y)
        - float: use the provided value directly

        :param y: Target values used to compute epsilon for "auto" mode.
        :return: The resolved epsilon value.
        """
        if isinstance(self.epsilon, (int, float)):
            return float(self.epsilon)

        if str(self.epsilon).lower() == "auto":
            y_std = float(np.std(np.asarray(y)))
            epsilon = DHI_SVR_EPSILON_SCALE_FACTOR * y_std
            self.logger.debug(f"Using 'auto' epsilon: {DHI_SVR_EPSILON_SCALE_FACTOR} * {y_std:.4f} = {epsilon:.4f}")
            return max(epsilon, DHI_SVR_X_NEAR_ZERO)

        raise SVRValidationError(f"Invalid epsilon value: '{self.epsilon}'. Expected 'auto' or a numeric value.")

    def _resolve_kernel_func(self) -> Callable:
        """
        Resolves the kernel function according to the kernel type.

        :return Callable: The resolved kernel function
        :raises SVRKernelError: If the kernel type is invalid
        """
        # Resolve coef0 to 0.0 if None (required for poly and sigmoid kernels)
        self.coef0_ = self.coef0 if self.coef0 is not None else 0.0
        if self.gamma_ is None:
            self.gamma_ = self._resolve_gamma(self.X_)
            self.logger.debug(f"Gamma was resolved to: {self.gamma_}")

        self.degree_ = self.degree

        match self.kernel:
            case "linear":
                self.gamma = DHI_SVR_DEFAULT_GAMMA
                self.coef0 = DHI_SVR_DEFAULT_COEF0
                self.degree = DHI_SVR_DEFAULT_DEGREE
                self.gamma_ = None
                self.coef0_ = DHI_SVR_DEFAULT_COEF0
                self.degree_ = DHI_SVR_DEFAULT_DEGREE

                def linear_kernel(x, y):
                    x, y = np.atleast_2d(x), np.atleast_2d(y)
                    return x.T @ y

                self.logger.debug("Using a linear kernel")
                return linear_kernel
            case "poly":
                if self.gamma_ is None:
                    raise SVRKernelError("Gamma parameter must be resolved before using the poly kernel.")

                gamma_, coef0_, degree_ = self.gamma_, self.coef0_, self.degree_
                max_base = np.float64(1e300) ** (1.0 / degree_) if degree_ > 0 else np.inf

                def poly_kernel(x, y):
                    x, y = np.atleast_2d(x), np.atleast_2d(y)
                    base = gamma_ * (x.T @ y) + coef0_
                    base = np.clip(base, -max_base, max_base)
                    return base**degree_

                self.logger.debug(
                    f"Using a polynomial kernel with the following parameters: gamma={self.gamma_}, coef0={self.coef0_}, degree={self.degree_}"
                )
                return poly_kernel
            case "rbf":
                self.coef0, self.coef0_ = DHI_SVR_DEFAULT_COEF0, DHI_SVR_DEFAULT_COEF0
                self.degree, self.degree_ = DHI_SVR_DEFAULT_DEGREE, DHI_SVR_DEFAULT_DEGREE

                if self.gamma_ is None:
                    raise SVRKernelError("Gamma parameter must be resolved before using the RBF kernel.")

                gamma_ = self.gamma_

                def rbf_kernel(x, y):
                    x = np.atleast_2d(x).astype(np.float64, copy=False)
                    y = np.atleast_2d(y).astype(np.float64, copy=False)
                    # Vectorized: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
                    x_sq = np.sum(x * x, axis=1, keepdims=True, dtype=np.float64)
                    y_sq = np.sum(y * y, axis=1, keepdims=True, dtype=np.float64)
                    sq_dist = x_sq + y_sq.T - 2.0 * (x @ y.T)
                    # Clamp negative values due to numerical precision
                    sq_dist = np.maximum(sq_dist, 0.0)
                    return np.exp(-gamma_ * sq_dist)

                self.logger.debug(f"Using a RBF kernel with the following parameters: gamma={self.gamma_}")
                return rbf_kernel
            case "sigmoid":
                self.degree, self.degree_ = DHI_SVR_DEFAULT_DEGREE, DHI_SVR_DEFAULT_DEGREE

                if self.gamma_ is None:
                    raise SVRKernelError("Gamma parameter must be resolved before using the sigmoid kernel.")

                gamma_, coef0_ = self.gamma_, self.coef0_

                def sigmoid_kernel(x, y):
                    x, y = np.atleast_2d(x), np.atleast_2d(y)
                    return np.tanh(gamma_ * (x.T @ y) + coef0_)

                self.logger.debug(
                    f"Using a sigmoid kernel with the following parameters: gamma={self.gamma_}, coef0={self.coef0_}"
                )
                return sigmoid_kernel
            case _:
                raise SVRKernelError(f"Invalid kernel type: {self.kernel}")

    def _validate_data(self, X: ArrayLike, y: ArrayLike) -> bool:
        """
        Validates the input and target data.

        :param ArrayLike X: The input data
        :param ArrayLike y: The target data
        :return bool: Whether the data is valid or not
        """
        X, y = np.asarray(X), np.asarray(y).ravel()

        if X.ndim != 2:
            self.logger.error(f"X must be a 2D array, got {X.ndim}D")
            return False
        if y.ndim != 1:
            self.logger.error(f"y must be a 1D array, got {y.ndim}D")
            return False

        if X.shape[0] != y.shape[0]:
            self.logger.error(f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}")
            return False

        if X.shape[0] == 0:
            self.logger.error("Data cannot be empty")
            return False

        return True

    def _compute_kernel_matrix(self, A: ArrayLike, B: ArrayLike) -> np.ndarray:
        """
        Computes the kernel matrix for the input data.

        :param ArrayLike A: The first input data (n_a samples x d features)
        :param ArrayLike B: The second input data (n_b samples x d features)
        :return np.ndarray: The kernel matrix (n_a x n_b) in float32 to reduce memory
        """
        if self.kernel_func_ is None:
            self.kernel_func_ = self._resolve_kernel_func()

        return np.asarray(self.kernel_func_(A, B), dtype=np.float32)

    def _K(self, i: int, j: int) -> float:
        """
        Returns the kernel value for the given indices.

        :param int i: The first index
        :param int j: The second index
        :return float: The kernel value
        """
        if self.kernel_matrix_ is not None:
            return float(self.kernel_matrix_[i, j])

        if self.kernel_func_ is None:
            self.kernel_func_ = self._resolve_kernel_func()

        xi, xj = self.X_[i], self.X_[j]
        result = self.kernel_func_(xi, xj)

        # Kernel returns (1,1) array for pairwise; extract scalar
        return float(np.asarray(result).ravel()[0])

    def _output_for_index(self, idx: int) -> float:
        """
        Provides the output of the current model for the given index.

        :param int idx: The index
        :return float: The output of the current model
        """
        beta = self.a_[: self.n_samples_] - self.a_[self.n_samples_ :]
        if self.kernel_matrix_ is not None:
            return float(beta @ self.kernel_matrix_[:, idx] + self.b_)

        return float(sum(beta[i] * self._K(i, idx) for i in range(self.n_samples_)) + self.b_)

    def _update_error_cache(self, idx_i: int, idx_j: int, delta_ai: float, delta_aj: float, delta_b: float) -> None:
        """
        Efficiently updates the error cache without recomputing the objective function.

        :param int idx_i: The first index
        :param int idx_j: The second index
        :param float delta_ai: The change in the first dual variable
        :param float delta_aj: The change in the second dual variable
        :param float delta_b: The change in the bias term
        """
        ii = idx_i if idx_i < self.n_samples_ else idx_i - self.n_samples_
        ij = idx_j if idx_j < self.n_samples_ else idx_j - self.n_samples_
        sign_i, sign_j = self.sign_[idx_i], self.sign_[idx_j]

        if self.kernel_matrix_ is not None:
            self.errors_ += (
                sign_i * delta_ai * self.kernel_matrix_[:, ii]
                + sign_j * delta_aj * self.kernel_matrix_[:, ij]
                + delta_b
            )
            return

        if self.kernel_func_ is None:
            self.kernel_func_ = self._resolve_kernel_func()

        # Vectorized: compute K(all_samples, sample_ii) and K(all_samples, sample_ij)
        K_ii = np.asarray(self.kernel_func_(self.X_, self.X_[ii : ii + 1]), dtype=np.float32).ravel()
        K_ij = np.asarray(self.kernel_func_(self.X_, self.X_[ij : ij + 1]), dtype=np.float32).ravel()
        self.errors_ += (
            np.float32(sign_i * delta_ai) * K_ii + np.float32(sign_j * delta_aj) * K_ij + np.float32(delta_b)
        )

    def _phi(self, idx: int) -> float:
        """
        Computes the gradient (phi) for the dual variable at the given index.

        For epsilon-SVR dual maximization, the gradients are:
          - For alpha_i:      dL/d(alpha_i)      = -err_i - epsilon
          - For alpha_star_i: dL/d(alpha_star_i) = err_i - epsilon
        where err = f(x) - y is the prediction error.

        :param int idx: The index (0..n-1 for alpha, n..2n-1 for alpha_star)
        :return float: The gradient value
        """
        sample_idx = idx if idx < self.n_samples_ else idx - self.n_samples_
        err = self.errors_[sample_idx]

        if idx < self.n_samples_:
            return float(-err - self.epsilon_)
        else:
            return float(err - self.epsilon_)

    def _violates_kkt_for_index(self, idx: int) -> bool:
        """
        Checks if the KKT conditions are violated for the given index.

        For epsilon-SVR dual maximization, KKT conditions at optimum are:
          - If alpha < C: gradient <= 0 (otherwise we could increase alpha)
          - If alpha > 0: gradient >= 0 (otherwise we could decrease alpha)

        A violation occurs when:
          - alpha < C and gradient > tol (should increase)
          - alpha > 0 and gradient < -tol (should decrease)

        :param int idx: The index
        :return bool: Whether the KKT conditions are violated or not
        """
        gradient = self.phi_cache_[idx]
        alpha = self.a_[idx]

        # Violation: can increase (low alpha, positive gradient)
        if alpha < self.C - np.finfo(float).eps and gradient > self.tol:
            self.logger.debug(f"Violation: can increase (low alpha, positive gradient) for index {idx}")
            return True
        # Violation: can decrease (high alpha, negative gradient)
        if alpha > np.finfo(float).eps and gradient < -self.tol:
            self.logger.debug(f"Violation: can decrease (high alpha, negative gradient) for index {idx}")
            return True
        return False

    def _select_first_index(self, ignore_indices: set[int] | None = None) -> int | None:
        """
        Selects the first index to be optimized by the SMO algorithm,
        one that violates the KKT conditions.

        For epsilon-SVR dual maximization:
          - Low alpha (< C) with positive gradient can increase (violation)
          - High alpha (> 0) with negative gradient can decrease (violation)

        :param set[int] | None ignore_indices: Indices to ignore
        :return int | None: The selected index, or None if no index is found
        """
        alpha = self.a_
        gradient = self.phi_cache_

        mask_low = alpha < (self.C - np.finfo(float).eps)
        mask_high = alpha > (np.finfo(float).eps)

        viol = np.zeros_like(gradient)
        # Low alpha with positive gradient: can increase (violation magnitude = gradient)
        viol[mask_low] = np.maximum(0.0, gradient[mask_low])
        # High alpha with negative gradient: can decrease (violation magnitude = -gradient)
        viol[mask_high] = np.maximum(0.0, -gradient[mask_high])

        # Shrinking heuristic: ignore indices that are very unlikely to violate KKT conditions.
        if self.shrinking and (self.active_set_ is not None):
            viol[~self.active_set_] = 0.0

        if ignore_indices:
            viol[list(ignore_indices)] = 0.0

        max_viol = float(np.max(viol))
        if max_viol <= self.tol and np.all(np.abs(self.errors_) <= self.epsilon_ + self.tol):
            self.logger.debug("No optimizable KKT violations found, algorithm converged")
            return None

        return int(np.argmax(viol))

    def _apply_shrinking(self) -> None:
        """
        Applies the SMO shrinking heuristic (working-set reduction).

        It is safe to temporarily shrink variable i when it's at a bound and the gradient
        is pushing it deeper into that bound (not causing a KKT violation):
          - alpha near 0 and gradient <= 0
          - alpha near C and gradient >= 0
        """
        if (not self.shrinking) or (self.active_set_ is None):
            self.logger.debug("Shrinking is not enabled or the active set is None")
            return

        bound_eps = max(float(self.tol), float(np.finfo(float).eps))

        for i in range(2 * self.n_samples_):
            if not bool(self.active_set_[i]):
                continue

            alpha = float(self.a_[i])
            gradient = float(self.phi_cache_[i])

            if (alpha <= bound_eps and gradient <= 0.0) or (alpha >= float(self.C) - bound_eps and gradient >= 0.0):
                self.active_set_[i] = False

    def _select_second_index(self, idx: int) -> int | None:
        """
        Selects the second index to be optimized by the SMO algorithm.

        For epsilon-SVR SMO, we pair variables with opposite signs from different samples.
        Same-sample pairs do not work because their gradients always sum to -2*epsilon.

        :param int idx: The first index to be optimized
        :return int | None: The selected index, or None if no index is found
        """
        sign_idx = self.sign_[idx]
        gradient_idx = self.phi_cache_[idx]

        # Build mask for opposite-sign indices from different samples
        opposite_sign_mask = self.sign_ != sign_idx
        opposite_sign_mask[idx] = False

        same_sample_twin = idx + self.n_samples_ if idx < self.n_samples_ else idx - self.n_samples_
        opposite_sign_mask[same_sample_twin] = False

        # Apply shrinking mask if active
        if self.shrinking and (self.active_set_ is not None):
            opposite_sign_mask = opposite_sign_mask & self.active_set_

        if not np.any(opposite_sign_mask):
            return None

        gradients = self.phi_cache_
        if gradient_idx > 0:
            # First index wants to increase; pick second with largest positive gradient
            masked = np.where(opposite_sign_mask, gradients, -np.inf)
            best_second = int(np.argmax(masked))
            # Only proceed if the best second also has positive gradient
            if gradients[best_second] <= 0:
                return None
        else:
            # First index wants to decrease; pick second with smallest (most negative) gradient
            masked = np.where(opposite_sign_mask, gradients, np.inf)
            best_second = int(np.argmin(masked))
            # Only proceed if the best second also has negative gradient
            if gradients[best_second] >= 0:
                return None

        if best_second == idx:
            return None

        return best_second

    def _bounds_for_index_pair(self, idx_i: int, idx_j: int) -> tuple[float, float]:
        """
        Computes the box bounds for the given index pair.

        :param int idx_i: The first index
        :param int idx_j: The second index
        :return tuple[float, float]: The box bounds
        """
        alpha_i, alpha_j = self.a_[idx_i], self.a_[idx_j]
        sign_i, sign_j = self.sign_[idx_i], self.sign_[idx_j]

        if sign_i == sign_j:
            W = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        else:
            W = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + (alpha_j - alpha_i))

        return W, H

    def _take_step(self, idx_i: int, idx_j: int) -> bool:
        """
        Takes a step in the direction of the gradient for a pair of dual variables.

        For epsilon-SVR SMO, we pair variables with opposite signs to maintain
        the equality constraint sum(alpha - alpha_star) = 0.

        :param int idx_i: The first index
        :param int idx_j: The second index
        :return bool: If progress was made in the optimization algorithm or not
        """
        if idx_i == idx_j:
            return False

        sign_i, sign_j = self.sign_[idx_i], self.sign_[idx_j]

        ii = idx_i if idx_i < self.n_samples_ else idx_i - self.n_samples_
        ij = idx_j if idx_j < self.n_samples_ else idx_j - self.n_samples_

        alpha_i_old, alpha_j_old = self.a_[idx_i], self.a_[idx_j]

        W, H = self._bounds_for_index_pair(idx_i, idx_j)
        if H - W < np.finfo(float).eps:
            return False

        Kii = self._K(ii, ii)
        Kjj = self._K(ij, ij)
        Kij = self._K(ii, ij)

        eta = Kii + Kjj - 2 * Kij
        if eta < np.finfo(float).eps:
            return False

        if sign_i * sign_j < 0:
            alpha_j_new = alpha_j_old + (self.phi_cache_[idx_i] + self.phi_cache_[idx_j]) / eta
        else:
            alpha_j_new = alpha_j_old + (self.phi_cache_[idx_j] - self.phi_cache_[idx_i]) / eta

        alpha_j_new = max(W, min(H, alpha_j_new))
        alpha_i_new = alpha_i_old + sign_i * sign_j * (alpha_j_old - alpha_j_new)
        alpha_i_new = max(0.0, min(self.C, alpha_i_new))

        change_i = abs(alpha_i_new - alpha_i_old)
        change_j = abs(alpha_j_new - alpha_j_old)
        threshold = 1e-8
        if change_i < threshold and change_j < threshold:
            return False

        self.a_[idx_i] = alpha_i_new
        self.a_[idx_j] = alpha_j_new

        delta_i = alpha_i_new - alpha_i_old
        delta_j = alpha_j_new - alpha_j_old

        b_old = self.b_
        eps_bound = np.finfo(float).eps
        i_free = (alpha_i_new > eps_bound) and (alpha_i_new < self.C - eps_bound)
        j_free = (alpha_j_new > eps_bound) and (alpha_j_new < self.C - eps_bound)

        if i_free:
            target_err = self.epsilon_ if idx_i < self.n_samples_ else -self.epsilon_
            self.b_ = b_old + target_err - self.errors_[ii] - sign_i * delta_i * Kii - sign_j * delta_j * Kij
        elif j_free:
            target_err = self.epsilon_ if idx_j < self.n_samples_ else -self.epsilon_
            self.b_ = b_old + target_err - self.errors_[ij] - sign_i * delta_i * Kij - sign_j * delta_j * Kjj

        delta_b = self.b_ - b_old
        self._update_error_cache(idx_i, idx_j, delta_i, delta_j, delta_b)

        return True

    def _compute_phi_cache(self) -> np.ndarray:
        """
        Computes the phi cache.

        :return np.ndarray: The phi cache (float32 to reduce memory)
        """
        return np.array([self._phi(i) for i in range(2 * self.n_samples_)], dtype=np.float32)

    def _smo_optimize(self) -> None:
        """
        Optimizes the SVR model using the Sequential Minimal Optimization (SMO) algorithm.
        """
        rng = np.random.default_rng(self.random_state)

        iters, passes = 0, 0
        ok_steps, fail_steps = 0, 0
        shrinking_interval = max(DHI_SVR_DEFAULT_SHRINKING_INTERVAL, int(self.shrinking_interval))

        # Keep track of indices that failed to produce a valid step
        # to prevent infinite loops on the same index
        failed_indices: set[int] = set()

        self.phi_cache_ = self._compute_phi_cache()

        if self.shrinking:
            self.active_set_ = np.ones(2 * self.n_samples_, dtype=bool)
        else:
            self.active_set_ = None

        while passes < self.max_passes and iters < self.max_iter:
            # Unshrink near iteration limit to catch late violations
            if self.shrinking and (self.active_set_ is not None) and (self.max_iter - iters <= shrinking_interval):
                self.active_set_[:] = True

            if self.shrinking and iters > 0 and (iters % shrinking_interval == 0):
                self._apply_shrinking()

            i = self._select_first_index(ignore_indices=failed_indices)
            if i is None:
                # Before declaring convergence, unshrink and re-check all variables
                if self.shrinking and (self.active_set_ is not None) and (not bool(np.all(self.active_set_))):
                    self.active_set_[:] = True
                    failed_indices.clear()
                    self.phi_cache_ = self._compute_phi_cache()
                    continue

                self.logger.info("No optimizable KKT violations found, algorithm converged")
                break

            j = self._select_second_index(i)
            if j is not None and self._take_step(i, j):
                ok_steps += 1
                passes = 0
                failed_indices.clear()
                iters += 1
                self.phi_cache_ = self._compute_phi_cache()
                if self.shrinking and (self.active_set_ is not None):
                    self.active_set_[i] = True
                    self.active_set_[j] = True
                continue

            # Try other indices if heuristic fails
            start_offset = rng.integers(0, 2 * self.n_samples_)
            found_step = False

            for k_offset in range(2 * self.n_samples_):
                k = (start_offset + k_offset) % (2 * self.n_samples_)
                if k == i:
                    continue
                if self.shrinking and (self.active_set_ is not None) and (not bool(self.active_set_[k])):
                    continue

                if self._take_step(i, int(k)):
                    ok_steps += 1
                    passes = 0
                    failed_indices.clear()
                    found_step = True
                    break

            if found_step:
                iters += 1
                self.phi_cache_ = self._compute_phi_cache()
                continue

            fail_steps += 1
            passes += 1
            failed_indices.add(i)
            iters += 1

        if passes >= self.max_passes:
            self.logger.debug(f"SMO algorithm did not converge after {self.max_passes} passes")
        if iters >= self.max_iter:
            self.logger.debug(f"SMO algorithm did not converge after {self.max_iter} iterations")

        self.logger.debug(f"SMO algorithm OK steps: {ok_steps}, fail steps: {fail_steps}")

        # Free SMO working arrays before final error computation to reduce memory pressure
        del self.phi_cache_
        self.phi_cache_ = np.asarray([])
        if self.active_set_ is not None:
            del self.active_set_
            self.active_set_ = None

        self.passes_ = passes
        self.iters_ = iters
        self.ok_steps_ = ok_steps
        self.fail_steps_ = fail_steps

    def _check_fitted(self) -> None:
        """
        Checks if the model is fitted for safe usage before prediction.

        :raises NotFittedError: If the model is not fitted
        """
        # Check if fit() was called by looking for the _fitted_no_sv_ flag (None means not fitted)
        if self._fitted_no_sv_ is None:
            raise NotFittedError(
                "This SVR_ instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SVR_":
        """
        Fits the SVR model to the training data, returning the instance itself.

        :param ArrayLike X: The input data
        :param ArrayLike y: The target data
        :return SVR_: The fitted SVR model
        """
        if not self._validate_data(X, y):
            raise SVRValidationError("Provided data cannot be fed to the SVR model")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        self.X_, self.y_ = X, y
        self.n_samples_, self.n_features_in_ = X.shape[0], X.shape[1]

        self.logger.debug(f"Input data shape: {self.X_.shape}, target data shape: {self.y_.shape}")

        effective_cache_kernel = self.cache_kernel
        if self.n_samples_ > DHI_SVR_MAX_SAMPLES_FOR_CACHE and self.cache_kernel:
            kernel_mem_gb = (self.n_samples_**2 * 4) / (1024**3)  # float32
            self.logger.warning(
                f"Dataset too large ({self.n_samples_} samples) for kernel caching "
                f"(would require ~{kernel_mem_gb:.1f} GB). Disabling cache_kernel."
            )
            effective_cache_kernel = False

        self.gamma_ = self._resolve_gamma(self.X_)
        self.logger.debug(f"Resolved gamma: {self.gamma_}")

        self.epsilon_ = self._resolve_epsilon(y)
        self.logger.debug(f"Resolved epsilon: {self.epsilon_}")

        self.kernel_func_ = self._resolve_kernel_func()

        # The dual variables: (alpha, alpha_star), stored in a single array of shape (2 * n_samples_)
        # * alpha_: positive support vectors, raises the prediction above the error tube
        # * alpha_star_: negative support vectors, lowers the prediction below the error tube
        self.a_ = np.zeros(2 * self.n_samples_, dtype=np.float32)

        # The sign vector: its purpose is to map the sign of the dual variables to the corresponding side of the error tube
        self.sign_ = np.ones(2 * self.n_samples_, dtype=np.int8)
        self.sign_[self.n_samples_ :] = -1.0

        # The bias term
        self.b_ = 0.0

        if effective_cache_kernel:
            self.kernel_matrix_ = self._compute_kernel_matrix(X, X).astype(np.float32)
            self.logger.debug("Kernel matrix cached")
        else:
            self.kernel_matrix_ = None

        self.errors_ = -y.copy().astype(np.float32)

        self._smo_optimize()

        self.alpha_, self.alpha_star_ = (
            self.a_[: self.n_samples_].copy(),
            self.a_[self.n_samples_ :].copy(),
        )
        self.dual_coef_ = (self.alpha_ - self.alpha_star_).astype(np.float32)

        # Counting the support vectors as the most closer ones to the error tube
        self.support_ = np.nonzero(np.abs(self.dual_coef_) > self.epsilon)[0]

        # Handle edge case: no support vectors found (SMO didn't converge or trivial solution)
        if len(self.support_) == 0:
            self.logger.warning(
                "No support vectors found (all dual coefficients below epsilon). "
                "Model will predict bias only. Consider increasing max_iter or adjusting epsilon."
            )
            # Keep empty arrays but mark model as fitted
            self.support_vectors_ = np.asarray([], dtype=np.float32).reshape(0, self.n_features_in_)
            self.support_dual_coef_ = np.asarray([], dtype=np.float32)
            self._fitted_no_sv_ = True
        else:
            self.support_vectors_ = self.X_[self.support_].astype(np.float32)
            self.support_dual_coef_ = self.dual_coef_[self.support_].astype(np.float32)
            self._fitted_no_sv_ = False

        # Kernel stats (compute on float32 to avoid memory doubling)
        self.kernel_stats_ = {}
        if self.kernel_matrix_ is not None and np.all(np.isfinite(self.kernel_matrix_)):
            self.kernel_stats_ = {
                "min": float(np.min(self.kernel_matrix_)),
                "max": float(np.max(self.kernel_matrix_)),
                "mean": float(np.mean(self.kernel_matrix_)),
                "std": float(np.std(self.kernel_matrix_)),
                "var": float(np.var(self.kernel_matrix_)),
                "median": float(np.median(self.kernel_matrix_)),
            }

        # Free large arrays no longer needed for prediction to reduce memory footprint
        # Prediction only needs: support_vectors_, support_dual_coef_, b_, kernel params
        del self.a_, self.sign_, self.errors_
        self.a_ = np.asarray([])
        self.sign_ = np.asarray([])
        self.errors_ = np.asarray([])

        # Free alpha arrays (we have dual_coef_ and support_dual_coef_)
        del self.alpha_, self.alpha_star_
        self.alpha_ = np.asarray([])
        self.alpha_star_ = np.asarray([])

        # Free kernel matrix cache (will recompute during prediction if needed)
        if self.kernel_matrix_ is not None:
            del self.kernel_matrix_
            self.kernel_matrix_ = None

        # Free full training data - only keep support vectors for prediction
        del self.X_, self.y_
        self.X_ = np.asarray([])
        self.y_ = np.asarray([])

        self.logger.debug(
            f"Memory optimized: keeping {len(self.support_)} support vectors "
            f"out of {self.n_samples_} training samples ({100*len(self.support_)/self.n_samples_:.1f}%)"
        )

        gc.collect()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the output of the SVR model for the given input data.

        Uses only support vectors for prediction (memory efficient).
        Uses chunked kernel computation for large test sets to limit memory usage.

        :param ArrayLike X: The input data
        :return np.ndarray: The predicted output
        """
        self._check_fitted()

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise SVRValidationError("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise SVRValidationError(f"X must have {self.n_features_in_} features")

        n_test = X.shape[0]

        # Handle edge case: no support vectors (return bias only)
        if self._fitted_no_sv_:
            return np.full(n_test, self.b_, dtype=np.float32)

        n_sv = len(self.support_vectors_)

        batch_size = max(
            DHI_SVR_MIN_CHUNK_BATCH_SIZE,
            min(DHI_SVR_MAX_CHUNK_BATCH_SIZE, DHI_SVR_MAX_CHUNK_BYTES // (n_sv * 4)),
        )

        if n_test <= batch_size:
            K = self._compute_kernel_matrix(X, self.support_vectors_)
            return (K @ self.support_dual_coef_ + self.b_).astype(np.float32)

        predictions = np.empty(n_test, dtype=np.float32)

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            K_chunk = self._compute_kernel_matrix(X[start:end], self.support_vectors_)
            predictions[start:end] = (K_chunk @ self.support_dual_coef_ + self.b_).astype(np.float32)
            del K_chunk

        return predictions


def plot_svr_predictions(
    estimator,
    X: ArrayLike,
    y: ArrayLike,
    *,
    feature_indices: tuple[int, int] = (0, 1),
    feature_names: Optional[tuple[str, str]] = None,
    grid_resolution: int = 100,
    colorscale: str = "RdYlBu_r",
    opacity: float = 0.7,
    show_training_data: bool = True,
    support_vectors: bool = True,
    max_sv_display: Optional[int] = None,
    show_contours: bool = True,
    n_contours: int = 10,
    show_epsilon_tube: bool = False,
    epsilon: Optional[float] = None,
    title: Optional[str] = None,
    show_colorbar: bool = True,
    scatter_size: int = 10,
    sv_size: int = 22,
    width: int = 800,
    height: int = 600,
    show: bool = True,
) -> go.Figure:
    """
    Plots the SVR prediction surface with support vectors for 2D visualization using Plotly.

    Works with both sklearn.svm.SVR and custom SVR_ estimator.
    For datasets with more than 2 features, specify which 2 features to use
    for the visualization via `feature_indices`.

    :param estimator: A fitted SVR model (sklearn SVR or custom SVR_).
    :param ArrayLike X: Training data of shape (n_samples, n_features).
    :param ArrayLike y: Target values of shape (n_samples,).
    :param tuple[int, int] feature_indices: Indices of the two features to use for 2D visualization.
    :param Optional[tuple[str, str]] feature_names: Names of the two features for axis labels.
    :param int grid_resolution: Resolution of the prediction grid.
    :param str colorscale: Plotly colorscale for the prediction surface.
    :param float opacity: Transparency of the prediction surface (0-1).
    :param bool show_training_data: Whether to show training data points.
    :param bool support_vectors: Whether to highlight support vectors.
    :param Optional[int] max_sv_display: Maximum number of support vectors to display. If None, shows all.
        When limited, selects SVs with highest dual coefficient magnitude (most influential).
    :param bool show_contours: Whether to show prediction contour lines (decision boundaries).
    :param int n_contours: Number of contour lines to display when show_contours=True.
    :param bool show_epsilon_tube: Whether to show epsilon-insensitive tube contours.
    :param Optional[float] epsilon: Epsilon value for the tube. If None, tries to get from estimator.
    :param Optional[str] title: Plot title. If None, auto-generates based on estimator.
    :param bool show_colorbar: Whether to show colorbar.
    :param int scatter_size: Size of data point markers.
    :param int sv_size: Size of support vector markers.
    :param int width: Figure width in pixels.
    :param int height: Figure height in pixels.
    :param bool show: Whether to display the figure immediately.
    :return go.Figure: The Plotly figure object.
    :raises ValueError: If X is not a 2D array or feature_indices are out of bounds.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    _, n_features = X.shape
    f1, f2 = feature_indices

    if f1 >= n_features or f2 >= n_features:
        raise ValueError(f"feature_indices ({f1}, {f2}) out of bounds for data with {n_features} features")

    if feature_names is None:
        x_label, y_label = f"Feature {f1}", f"Feature {f2}"
    else:
        x_label, y_label = feature_names

    X_2d = X[:, [f1, f2]]

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )

    if n_features > 2:
        grid_points = np.zeros((xx.ravel().shape[0], n_features))
        grid_points[:, f1] = xx.ravel()
        grid_points[:, f2] = yy.ravel()
        for i in range(n_features):
            if i not in (f1, f2):
                grid_points[:, i] = X[:, i].mean()
    else:
        grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = estimator.predict(grid_points).reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=np.linspace(x_min, x_max, grid_resolution),
            y=np.linspace(y_min, y_max, grid_resolution),
            z=Z,
            colorscale=colorscale,
            opacity=opacity,
            zmin=y.min(),
            zmax=y.max(),
            showscale=show_colorbar,
            colorbar={"title": "Predicted Value"} if show_colorbar else None,
            hovertemplate=f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>Prediction: %{{z:.3f}}<extra></extra>",
        )
    )

    if show_contours:
        contour_step = (Z.max() - Z.min()) / (n_contours + 1)
        fig.add_trace(
            go.Contour(
                x=np.linspace(x_min, x_max, grid_resolution),
                y=np.linspace(y_min, y_max, grid_resolution),
                z=Z,
                contours={
                    "start": float(Z.min() + contour_step),
                    "end": float(Z.max() - contour_step),
                    "size": float(contour_step),
                    "showlabels": True,
                    "labelfont": {"size": 9, "color": "black"},
                },
                line={"width": 1.5, "color": "black"},
                showscale=False,
                opacity=0.9,
                hoverinfo="skip",
                name="Contours",
            )
        )

    if show_epsilon_tube:
        eps = epsilon
        if eps is None:
            if hasattr(estimator, "epsilon_"):
                eps = estimator.epsilon_
            elif hasattr(estimator, "epsilon"):
                eps = estimator.epsilon

        if eps is not None and eps > 0:
            z_mean = float(Z.mean())
            fig.add_trace(
                go.Contour(
                    x=np.linspace(x_min, x_max, grid_resolution),
                    y=np.linspace(y_min, y_max, grid_resolution),
                    z=Z,
                    contours={
                        "start": z_mean - eps,
                        "end": z_mean + eps,
                        "size": eps,
                        "showlabels": True,
                        "labelfont": {"size": 10, "color": "black"},
                    },
                    line={"width": 2, "color": "darkred", "dash": "dash"},
                    showscale=False,
                    opacity=0.8,
                    hoverinfo="skip",
                    name="ε-tube",
                )
            )

    if show_training_data:
        fig.add_trace(
            go.Scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                mode="markers",
                marker={
                    "size": scatter_size,
                    "color": y,
                    "colorscale": colorscale,
                    "cmin": y.min(),
                    "cmax": y.max(),
                    "line": {"width": 1, "color": "black"},
                    "showscale": False,
                },
                hovertemplate=f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>Target: %{{marker.color:.3f}}<extra></extra>",
                name="Training Data",
            )
        )

    if support_vectors and hasattr(estimator, "support_vectors_"):
        sv = estimator.support_vectors_
        if sv is not None and len(sv) > 0:
            n_total_sv = len(sv)
            is_limited = max_sv_display is not None and n_total_sv > max_sv_display

            all_sv_predictions = estimator.predict(sv)

            if is_limited and max_sv_display is not None:
                rng = np.random.default_rng(seed=42)
                n_bins = max(n_contours, 5)
                sv_per_bin = max(1, max_sv_display // n_bins)

                pred_min, pred_max = all_sv_predictions.min(), all_sv_predictions.max()
                bin_edges = np.linspace(pred_min, pred_max + 1e-9, n_bins + 1)
                bin_indices = np.digitize(all_sv_predictions, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                selected_indices = []
                for bin_idx in range(n_bins):
                    bin_mask = bin_indices == bin_idx
                    bin_sv_indices = np.nonzero(bin_mask)[0]

                    if len(bin_sv_indices) == 0:
                        continue

                    n_select = min(sv_per_bin, len(bin_sv_indices))

                    if hasattr(estimator, "dual_coef_"):
                        coef_magnitude = np.abs(estimator.dual_coef_).ravel()
                        if len(coef_magnitude) == n_total_sv:
                            bin_coefs = coef_magnitude[bin_sv_indices]
                            top_in_bin = bin_sv_indices[np.argsort(bin_coefs)[-n_select:]]
                            selected_indices.extend(top_in_bin)
                        else:
                            selected_indices.extend(rng.choice(bin_sv_indices, n_select, replace=False))
                    else:
                        selected_indices.extend(rng.choice(bin_sv_indices, n_select, replace=False))

                selected_indices = np.array(selected_indices)
                sv = sv[selected_indices]
                sv_predictions = all_sv_predictions[selected_indices]
            else:
                sv_predictions = all_sv_predictions

            if sv.shape[1] > 2:
                sv_2d = sv[:, [f1, f2]]
            else:
                sv_2d = sv

            sv_label = f"Support Vectors (n={len(sv)}"
            if is_limited:
                sv_label += f" of {n_total_sv}, stratified by prediction"
            sv_label += ")"

            fig.add_trace(
                go.Scatter(
                    x=sv_2d[:, 0],
                    y=sv_2d[:, 1],
                    mode="markers",
                    marker={
                        "size": sv_size,
                        "symbol": "circle",
                        "color": sv_predictions,
                        "colorscale": colorscale,
                        "cmin": y.min(),
                        "cmax": y.max(),
                        "line": {"width": 2.5, "color": "black"},
                        "showscale": False,
                    },
                    hovertemplate=f"Support Vector<br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>Prediction: %{{marker.color:.3f}}<extra></extra>",
                    name=sv_label,
                )
            )

    if title is None:
        kernel = getattr(estimator, "kernel", "unknown")
        title = f"{repr(estimator)} Predictions ({kernel} kernel)"

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis={"title": x_label, "range": [x_min, x_max]},
        yaxis={"title": y_label, "range": [y_min, y_max]},
        width=width,
        height=height,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        hovermode="closest",
    )

    if show:
        fig.show()

    return fig


def plot_compare_svr_models(
    estimators: list[tuple[str, Any]],
    X: ArrayLike,
    y: ArrayLike,
    *,
    feature_indices: tuple[int, int] = (0, 1),
    feature_names: Optional[tuple[str, str]] = None,
    grid_resolution: int = 100,
    width: int = 500,
    height: int = 450,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """
    Compares multiple SVR models side-by-side using Plotly subplots.

    :param list[tuple[str, Any]] estimators: List of (name, fitted_estimator) tuples.
    :param ArrayLike X: Training data of shape (n_samples, n_features).
    :param ArrayLike y: Target values of shape (n_samples,).
    :param tuple[int, int] feature_indices: Indices of the two features to use for 2D visualization.
    :param Optional[tuple[str, str]] feature_names: Names of the two features for axis labels.
    :param int grid_resolution: Resolution of the prediction grid.
    :param int width: Width per subplot in pixels.
    :param int height: Height of the figure in pixels.
    :param bool show: Whether to display the figure immediately.
    :param kwargs: Additional arguments (colorscale, opacity, scatter_size, sv_size, support_vectors).
    :return go.Figure: The Plotly figure object with subplots.
    """
    n_models = len(estimators)

    fig = make_subplots(
        rows=1,
        cols=n_models,
        subplot_titles=[name for name, _ in estimators],
        horizontal_spacing=0.08,
    )

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    _, n_features = X.shape
    f1, f2 = feature_indices

    if feature_names is None:
        x_label, y_label = f"Feature {f1}", f"Feature {f2}"
    else:
        x_label, y_label = feature_names

    X_2d = X[:, [f1, f2]]

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )

    if n_features > 2:
        grid_points = np.zeros((xx.ravel().shape[0], n_features))
        grid_points[:, f1] = xx.ravel()
        grid_points[:, f2] = yy.ravel()
        for i in range(n_features):
            if i not in (f1, f2):
                grid_points[:, i] = X[:, i].mean()
    else:
        grid_points = np.c_[xx.ravel(), yy.ravel()]

    colorscale = kwargs.get("colorscale", "RdYlBu_r")
    opacity = kwargs.get("opacity", 0.7)
    scatter_size = kwargs.get("scatter_size", 10)
    sv_size = kwargs.get("sv_size", 18)
    show_support_vectors = kwargs.get("support_vectors", True)

    for col_idx, (name, estimator) in enumerate(estimators, start=1):
        Z = estimator.predict(grid_points).reshape(xx.shape)

        fig.add_trace(
            {
                "type": "heatmap",
                "x": np.linspace(x_min, x_max, grid_resolution).tolist(),
                "y": np.linspace(y_min, y_max, grid_resolution).tolist(),
                "z": Z.tolist(),
                "colorscale": colorscale,
                "opacity": opacity,
                "zmin": float(y.min()),
                "zmax": float(y.max()),
                "showscale": col_idx == n_models,
                "colorbar": {"title": "Predicted"} if col_idx == n_models else None,
                "hovertemplate": f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>Pred: %{{z:.3f}}<extra>{name}</extra>",
            },
            row=1,
            col=col_idx,
        )

        fig.add_trace(
            {
                "type": "scatter",
                "x": X_2d[:, 0].tolist(),
                "y": X_2d[:, 1].tolist(),
                "mode": "markers",
                "marker": {
                    "size": scatter_size,
                    "color": y.tolist(),
                    "colorscale": colorscale,
                    "cmin": float(y.min()),
                    "cmax": float(y.max()),
                    "line": {"width": 0.5, "color": "black"},
                    "showscale": False,
                },
                "hovertemplate": f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>Target: %{{marker.color:.3f}}<extra>Data</extra>",
                "name": "Data",
                "showlegend": col_idx == 1,
            },
            row=1,
            col=col_idx,
        )

        if show_support_vectors and hasattr(estimator, "support_vectors_"):
            sv = estimator.support_vectors_
            if sv is not None and len(sv) > 0:
                sv_2d = sv[:, [f1, f2]] if sv.shape[1] > 2 else sv
                fig.add_trace(
                    {
                        "type": "scatter",
                        "x": sv_2d[:, 0].tolist(),
                        "y": sv_2d[:, 1].tolist(),
                        "mode": "markers",
                        "marker": {
                            "size": sv_size,
                            "color": "rgba(0,0,0,0)",
                            "line": {"width": 2, "color": "black"},
                        },
                        "hovertemplate": f"SV<br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra>{name}</extra>",
                        "name": f"SVs (n={len(sv)})",
                        "showlegend": col_idx == 1,
                    },
                    row=1,
                    col=col_idx,
                )

        fig.update_xaxes(title_text=x_label, range=[x_min, x_max], row=1, col=col_idx)
        fig.update_yaxes(title_text=y_label if col_idx == 1 else "", range=[y_min, y_max], row=1, col=col_idx)

    fig.update_layout(
        width=width * n_models,
        height=height,
        title={"text": "SVR Model Comparison", "x": 0.5, "xanchor": "center"},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )

    if show:
        fig.show()

    return fig
