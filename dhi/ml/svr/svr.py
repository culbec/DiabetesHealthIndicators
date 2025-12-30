# Partial credits to: https://github.com/nihil21/custom-svm/blob/master/src/svm.py

import logging
import numpy as np

from numpy.typing import ArrayLike
from typing import Callable, Any
from sklearn.base import BaseEstimator, RegressorMixin

##############################
# SVR CONSTANTS
##############################

DHI_SVR_KERNEL_TYPES: list[str] = ["linear", "poly", "rbf", "sigmoid"]
DHI_SVR_DEFAULT_KERNEL: str = "auto"
DHI_SVR_DEFAULT_C: float = 1.0
DHI_SVR_DEFAULT_EPSILON: float = 1e-3
DHI_SVR_DEFAULT_TOL: float = 1e-3
DHI_SVR_DEFAULT_MAX_ITER: int = 1000
DHI_SVR_DEFAULT_MAX_PASSES: int = 50
DHI_SVR_DEFAULT_GAMMA: float | None = None
DHI_SVR_DEFAULT_DEGREE: int = 3
DHI_SVR_DEFAULT_COEF0: float | None = None


class SVR_ValidationException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class SVR_KernelException(Exception):
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
    gamma: float
        The kernel coefficient; if None, it will be computed automatically during the fit.
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
    kernal_stats: dict[str, Any]
        The statistics of the kernel matrix.
    passes_: int
        The number of passes of the SMO algorithm.
    iters_: int
        The number of iterations of the SMO algorithm.
    ok_steps_: int
        The number of successful steps of the SMO algorithm.
    fail_steps_: int
        The number of failed steps of the SMO algorithm.
    """

    def __init__(
        self,
        *,
        C: float = DHI_SVR_DEFAULT_C,
        kernel: str = DHI_SVR_DEFAULT_KERNEL,
        gamma: float | None = DHI_SVR_DEFAULT_GAMMA,
        degree: int = DHI_SVR_DEFAULT_DEGREE,
        coef0: float | None = DHI_SVR_DEFAULT_COEF0,
        epsilon: float = DHI_SVR_DEFAULT_EPSILON,
        tol: float = DHI_SVR_DEFAULT_TOL,
        max_iter: int = DHI_SVR_DEFAULT_MAX_ITER,
        max_passes: int = DHI_SVR_DEFAULT_MAX_PASSES,
        cache_kernel: bool = True,
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
        self.random_state = random_state
        self.verbose = verbose

        from dhi.utils import get_logger

        self.logger = get_logger(self.__class__.__name__, level=logging.DEBUG if verbose else logging.INFO)

        self.X_: ArrayLike = None
        self.y_: ArrayLike = None

        self.n_samples_: int = None
        self.n_features_in_: int = None

        self.kernel_func_: Callable = None
        self.kernel_matrix_: ArrayLike = None

        self.gamma_: float = None
        self.degree_: int = None
        self.coef0_: float = None

        self.a_: ArrayLike = None
        self.sign_: ArrayLike = None
        self.b_: float = None

        self.phi_cache_: ArrayLike = None
        self.errors_: ArrayLike = None

        self.alpha_: ArrayLike = None
        self.alpha_star_: ArrayLike = None
        self.dual_coef_: ArrayLike = None

        self.support_: ArrayLike = None
        self.support_vectors_: ArrayLike = None
        
        self.kernel_stats_: dict[str, Any] = None
        self.passes_: int = None
        self.iters_: int = None
        self.ok_steps_: int = None
        self.fail_steps_: int = None

    def __getstate__(self) -> dict:
        """
        Excludes the logger from serialization as it contains an RLock that cannot be pickled.

        :return dict: The object's state without the logger
        """
        state = self.__dict__.copy()
        # Remove logger if it exists (it contains RLock which can't be serialized)
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restores the object's state and reinitializes the logger after deserialization.

        :param dict state: The object's state
        """
        self.__dict__.update(state)
        # Reinitialize the logger after deserialization
        from dhi.utils import get_logger

        self.logger = get_logger(self.__class__.__name__, level=logging.DEBUG if self.verbose else logging.INFO)

    def _resolve_gamma(self, X: ArrayLike) -> float:
        """
        Resolves the `gamma` parameter according to the `scale` approach,
        if it was not provided during the initialization.

        If the `gamma` parameter was provided during the initialization, it is returned as is, float scaled.

        The `gamma` parameter is computed as follows:
        gamma = 1 / (n_features * var(X))

        :param ArrayLike X: The input data
        :return float: The resolved `gamma` parameter
        """
        if self.gamma is not None:
            return float(self.gamma)

        X = np.asarray(X)
        X_var = X.var()
        return 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0

    def _resolve_kernel_func(self) -> Callable:
        """
        Resolves the kernel function according to the kernel type.

        :return Callable: The resolved kernel function
        :raises SVR_KernelException: If the kernel type is invalid
        """
        # Resolve coef0 to 0.0 if None (required for poly and sigmoid kernels)
        self.coef0_ = self.coef0 if self.coef0 is not None else 0.0
        if not hasattr(self, "gamma_"):
            self.gamma_ = self._resolve_gamma(self.X_)

        self.degree_ = self.degree

        match self.kernel:
            case "linear":
                self.gamma, self.coef0, self.degree = (
                    DHI_SVR_DEFAULT_GAMMA,
                    DHI_SVR_DEFAULT_COEF0,
                    DHI_SVR_DEFAULT_DEGREE,
                )
                self.gamma_, self.coef0_, self.degree_ = (
                    DHI_SVR_DEFAULT_GAMMA,
                    DHI_SVR_DEFAULT_COEF0,
                    DHI_SVR_DEFAULT_DEGREE,
                )
                return lambda x, y: np.dot(x, y)
            case "poly":
                return lambda x, y: (self.gamma_ * np.dot(x, y) + self.coef0_) ** self.degree_
            case "rbf":
                self.coef0, self.degree = DHI_SVR_DEFAULT_COEF0, DHI_SVR_DEFAULT_DEGREE
                self.coef0_, self.degree_ = DHI_SVR_DEFAULT_COEF0, DHI_SVR_DEFAULT_DEGREE
                return lambda x, y: np.exp(-self.gamma_ * np.linalg.norm(x - y) ** 2)
            case "sigmoid":
                self.degree = DHI_SVR_DEFAULT_DEGREE
                self.degree_ = DHI_SVR_DEFAULT_DEGREE
                return lambda x, y: np.tanh(self.gamma_ * np.dot(x, y) + self.coef0_)
            case _:
                raise SVR_KernelException(f"Invalid kernel type: {self.kernel}")

    def _validate_data(self, X: ArrayLike, y: ArrayLike) -> bool:
        """
        Validates the input and target data.

        :param ArrayLike X: The input data
        :param ArrayLike y: The target data
        :return bool: Whether the data is valid or not
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()

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

        :param ArrayLike A: The first input data
        :param ArrayLike B: The second input data
        :return np.ndarray: The kernel matrix
        """
        if not hasattr(self, "kernel_func_"):
            self.kernel_func_ = self._resolve_kernel_func()

        A_, B_ = np.asarray(A, dtype=float), np.asarray(B, dtype=float)

        m, n = A_.shape[0], B_.shape[0]
        K = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                K[i, j] = self.kernel_func_(A_[i], B_[j])
        return K

    def _K(self, i: int, j: int) -> float:
        """
        Returns the kernel value for the given indices.

        :param int i: The first index
        :param int j: The second index
        :return float: The kernel value
        """
        if self.kernel_matrix_ is not None:
            return float(self.kernel_matrix_[i, j])

        return float(self.kernel_func_(self.X_[i], self.X_[j]))

    def _output_for_index(self, idx: int) -> float:
        """
        Provides the output of the current model for the given index.

        :param int idx: The index
        :return float: The output of the current model
        """
        beta = self.a_[: self.n_samples_] - self.a_[self.n_samples_ :]
        if self.kernel_matrix_ is not None:
            return float(beta @ self.kernel_matrix_[:, idx] + self.b_)

        res = 0.0
        for i in range(self.n_samples_):
            res += beta[i] * self._K(i, idx)
        return float(res + self.b_)

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

        for k in range(self.n_samples_):
            self.errors_[k] += sign_i * delta_ai * self.kernel_func_(self.X_[k], self.X_[ii])
            self.errors_[k] += sign_j * delta_aj * self.kernel_func_(self.X_[k], self.X_[ij])
            self.errors_[k] += delta_b

    def _phi(self, idx: int) -> float:
        """
        Computes the phi function for the given index.

        :param int idx: The index
        :return float: The phi function
        """
        sample_idx = idx if idx < self.n_samples_ else idx - self.n_samples_

        # Computing the error with respect to the epsilon-insensitive loss function and the side of the error tube
        err = self.errors_[sample_idx]
        if idx < self.n_samples_:
            return float(err - self.epsilon)
        return float(err + self.epsilon)

    def _violates_kkt_for_index(self, idx: int) -> bool:
        """
        Checks if the KKT conditions are violated for the given index.

        :param int idx: The index
        :return bool: Whether the KKT conditions are violated or not
        """
        phi = self.phi_cache_[idx]
        alpha = self.a_[idx]

        if alpha < self.C - np.finfo(float).eps and phi < -self.tol:
            return True
        if alpha > np.finfo(float).eps and phi > self.tol:
            return True
        return False

    def _select_first_index(self, rng: np.random.Generator, ignore_indices: set[int] | None = None) -> int | None:
        """
        Selects the first index to be optimized by the SMO algorithm,
        one that violates the KKT conditions.

        :param np.random.Generator rng: The random number generator
        :param set[int] | None ignore_indices: Indices to ignore
        :return int | None: The selected index, or None if no index is found
        """
        alpha = self.a_
        phi = self.phi_cache_ * self.sign_

        mask_low = alpha < (self.C - np.finfo(float).eps)
        mask_high = alpha > (np.finfo(float).eps)

        viol = np.zeros_like(phi)
        viol[mask_low] = np.maximum(0.0, -phi[mask_low])
        viol[mask_high] = np.maximum(0.0, phi[mask_high])

        # Ignoring indices by setting their violation to 0
        if ignore_indices:
            viol[list(ignore_indices)] = 0.0

        max_viol = float(np.max(viol))
        if max_viol <= self.tol and np.all(np.abs(self.errors_) <= self.epsilon + self.tol):
            return None

        return int(np.argmax(viol))

    def _select_second_index(self, idx: int) -> int | None:
        """
        Selects the second index to be optimized by the SMO algorithm.

        The second index is selected according to the passed first index.
        Pairs variables from opposite sides of the error tube (alpha with alpha_star).

        This forms the dual problem, optimizing two variables at a time.

        :param int idx: The first index to be optimized
        :return int | None: The selected index, or None if no index is found
        """
        phi_idx = self.phi_cache_[idx]

        # Choosing indices on the opposite side of the error tube
        if phi_idx > 0:
            best_second = int(np.argmin(self.phi_cache_))
        else:
            best_second = int(np.argmax(self.phi_cache_))

        # Cannot optimize if we have the same index
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

        # Ensuring the bounds are of the same sign if the dual variables are on opposite sides of the error tube
        if sign_i == sign_j:
            W = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        else:
            W = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + (alpha_j - alpha_i))

        return W, H

    def _take_step(self, idx_i: int, idx_j: int) -> bool:
        """
        Takes a step in the direction of the gradient for the given index pair.

        :param int idx_i: The first index
        :param int idx_j: The second index
        :return bool: If progress was made in the optimization algorithm or not
        """
        # False if we have the same indices
        if idx_i == idx_j:
            return False

        sign_i, sign_j = self.sign_[idx_i], self.sign_[idx_j]

        # Ensuring we are within the box bounds
        ii = idx_i if idx_i < self.n_samples_ else idx_i - self.n_samples_
        ij = idx_j if idx_j < self.n_samples_ else idx_j - self.n_samples_

        alpha_i_old, alpha_j_old = self.a_[idx_i], self.a_[idx_j]

        W, H = self._bounds_for_index_pair(idx_i, idx_j)

        if H - W < np.finfo(float).eps:
            return False

        phi_i, phi_j = self.phi_cache_[idx_i], self.phi_cache_[idx_j]

        Kii = self._K(ii, ii)
        Kjj = self._K(ij, ij)
        Kij = self._K(ii, ij)

        # Computing the denominator of the step, avoiding division by zero
        eta = Kii + Kjj - 2 * Kij
        # Scale by kernel magnitude to handle cases where kernel values are legitimately small
        kernel_scale = max(abs(Kii), abs(Kjj), abs(Kij), 1.0)
        # Use tolerance-scaled threshold relative to kernel scale, with machine epsilon as minimum
        eta_threshold = max(self.tol * kernel_scale * 1e-6, np.finfo(float).eps)
        if eta < eta_threshold:
            return False

        # Use phi values for the step calculation, which account for epsilon
        alpha_j_new = alpha_j_old + sign_j * (phi_i - phi_j) / eta
        if alpha_j_new > H:
            alpha_j_new = H
        if alpha_j_new < W:
            alpha_j_new = W

        alpha_i_new = (sign_i * alpha_i_old + sign_j * alpha_j_old - sign_j * alpha_j_new) / sign_i

        # The bounds [W, H] should ensure both alphas stay in [0, C], but check for numerical issues
        # If alpha_i_new is out of bounds due to numerical errors, clip it and adjust alpha_j_new
        if alpha_i_new < 0:
            alpha_i_new = 0.0
            alpha_j_new = (sign_i * alpha_i_old + sign_j * alpha_j_old - sign_i * alpha_i_new) / sign_j
            alpha_j_new = max(W, min(H, alpha_j_new))
        elif alpha_i_new > self.C:
            alpha_i_new = self.C
            alpha_j_new = (sign_i * alpha_i_old + sign_j * alpha_j_old - sign_i * alpha_i_new) / sign_j
            alpha_j_new = max(W, min(H, alpha_j_new))

        # Final bounds check, should not be necessary if bounds are correct
        alpha_i_new = max(0.0, min(self.C, alpha_i_new))
        alpha_j_new = max(0.0, min(self.C, alpha_j_new))

        # Check if the change is significant
        # Threshold should be relative to C and account for numerical precision
        change_i = abs(alpha_i_new - alpha_i_old)
        change_j = abs(alpha_j_new - alpha_j_old)
        threshold = self.tol * max(1.0, self.C) * 1e-2
        if change_i < threshold and change_j < threshold:
            return False

        self.a_[idx_i] = alpha_i_new
        self.a_[idx_j] = alpha_j_new

        delta_ai = alpha_i_new - alpha_i_old
        delta_aj = alpha_j_new - alpha_j_old

        # Biases for both indices
        b_old = self.b_
        b1 = b_old - self.errors_[ii] - sign_i * delta_ai * Kii - sign_j * delta_aj * Kij
        b2 = b_old - self.errors_[ij] - sign_i * delta_ai * Kij - sign_j * delta_aj * Kjj

        # Determining which index is free to update the bias
        eps_bound = np.finfo(float).eps
        i_free = (self.a_[idx_i] > eps_bound) and (self.a_[idx_i] < self.C - eps_bound)
        j_free = (self.a_[idx_j] > eps_bound) and (self.a_[idx_j] < self.C - eps_bound)

        if i_free:
            self.b_ = b1
        elif j_free:
            self.b_ = b2
        else:
            self.b_ = (b1 + b2) / 2.0

        delta_b = self.b_ - b_old

        self._update_error_cache(idx_i, idx_j, delta_ai, delta_aj, delta_b)

        return True

    def _compute_phi_cache(self) -> np.ndarray:
        """
        Computes the phi cache.

        :return np.ndarray: The phi cache
        """
        return np.array([self._phi(i) for i in range(2 * self.n_samples_)], dtype=float)

    def _smo_optimize(self) -> None:
        """
        Optimizes the SVR model using the Sequential Minimal Optimization (SMO) algorithm.
        """
        rng = np.random.default_rng(self.random_state)

        iters, passes = 0, 0
        ok_steps, fail_steps = 0, 0

        # Keep track of indices that failed to produce a valid step
        # to prevent infinite loops on the same index
        failed_indices: set[int] = set()

        self.phi_cache_ = self._compute_phi_cache()

        while passes < self.max_passes and iters < self.max_iter:
            i = self._select_first_index(rng, ignore_indices=failed_indices)
            if i is None:
                self.logger.info("No optimizable KKT violations found, algorithm converged")
                break

            j = self._select_second_index(i)
            if j is not None and self._take_step(i, j):
                # OK step using the heuristic for choosing the second index
                ok_steps += 1
                passes = 0
                failed_indices.clear()
                iters += 1
                self.phi_cache_ = self._compute_phi_cache()
                continue

            # If the heuristic for choosing the second index fails,
            # try to take a random step with any other index
            start_offset = rng.integers(0, 2 * self.n_samples_)
            found_step = False

            for k_offset in range(2 * self.n_samples_):
                k = (start_offset + k_offset) % (2 * self.n_samples_)
                if k == i:
                    continue

                if self._take_step(i, k):
                    ok_steps += 1
                    passes = 0
                    failed_indices.clear()
                    found_step = True
                    break

            if found_step:
                iters += 1
                self.phi_cache_ = self._compute_phi_cache()
                continue

            # Failure if we have tried all other indices and still no step could be taken
            fail_steps += 1
            passes += 1
            failed_indices.add(i)
            iters += 1

        if passes >= self.max_passes:
            self.logger.warning(f"SMO algorithm did not converge after {self.max_passes} passes")
        if iters >= self.max_iter:
            self.logger.warning(f"SMO algorithm did not converge after {self.max_iter} iterations")

        self.logger.info(f"SMO algorithm OK steps: {ok_steps}, fail steps: {fail_steps}")

        if self.kernel_matrix_ is not None:
            beta = self.a_[: self.n_samples_] - self.a_[self.n_samples_ :]
            f = beta @ self.kernel_matrix_ + self.b_
            self.errors_ = f - self.y_
        else:
            for i in range(self.n_samples_):
                self.errors_[i] = self._output_for_index(i) - self.y_[i]
                
        self.passes_ = passes
        self.iters_ = iters
        self.ok_steps_ = ok_steps
        self.fail_steps_ = fail_steps

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SVR_":
        """
        Fits the SVR model to the training data, returning the instance itself.

        :param ArrayLike X: The input data
        :param ArrayLike y: The target data
        :return SVR_: The fitted SVR model
        """
        if not self._validate_data(X, y):
            raise SVR_ValidationException("Provided data cannot be fed to the SVR model")

        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
        self.X_, self.y_ = X, y
        self.n_samples_, self.n_features_in_ = X.shape[0], X.shape[1]

        self.logger.debug(f"Input data shape: {X.shape}, target data shape: {y.shape}")

        # Resolving the kernel coefficient and the kernel function
        self.gamma_ = self._resolve_gamma(X)
        self.logger.debug(f"Resolved gamma: {self.gamma_}")

        self.kernel_func_ = self._resolve_kernel_func()

        # The dual variables: (alpha, alpha_star), stored in a single array of shape (2 * n_samples_)
        # * alpha_: positive support vectors, raises the prediction above the error tube
        # * alpha_star_: negative support vectors, lowers the prediction below the error tube
        self.a_ = np.zeros(2 * self.n_samples_, dtype=float)

        # The sign vector: its purpose is to map the sign of the dual variables to the corresponding side of the error tube
        self.sign_ = np.ones(2 * self.n_samples_, dtype=float)
        self.sign_[self.n_samples_ :] = -1.0

        # The bias term
        self.b_ = 0.0

        # Caching the kernel matrix, if specified
        if self.cache_kernel:
            self.kernel_matrix_ = self._compute_kernel_matrix(X, X)
            self.logger.debug("Kernel matrix cached")
        else:
            self.kernel_matrix_ = None
            self.logger.debug("Not caching kernel matrix")

        # Initial errors are the negative of the target data
        self.errors_ = -y.copy()

        # Sequential Minimal Optimization (SMO) algorithm step
        self._smo_optimize()

        # Enforcing epsilon-insensitive sparsity to prune unused dual variables
        for i in range(self.n_samples_):
            if abs(self.errors_[i]) < self.epsilon - self.tol:
                self.a_[i] = 0.0
                self.a_[i + self.n_samples_] = 0.0

        self.alpha_, self.alpha_star_ = self.a_[: self.n_samples_].copy(), self.a_[self.n_samples_ :].copy()
        self.dual_coef_ = self.alpha_ - self.alpha_star_

        # Counting the support vectors as the most closer ones to the error tube
        self.support_ = np.where(np.abs(self.dual_coef_) > self.epsilon)[0]
        self.support_vectors_ = self.X_[self.support_]
        
        # Kernel stats
        self.kernel_stats_ = {}
        if self.kernel_matrix_ is not None:
            self.kernel_stats_ = {
                "min": np.min(self.kernel_matrix_),
                "max": np.max(self.kernel_matrix_),
                "mean": np.mean(self.kernel_matrix_),
                "std": np.std(self.kernel_matrix_),
                "var": np.var(self.kernel_matrix_),
                "median": np.median(self.kernel_matrix_)
            }

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the output of the SVR model for the given input data.

        :param ArrayLike X: The input data
        :return np.ndarray: The predicted output
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise SVR_ValidationException("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise SVR_ValidationException(f"X must have {self.n_features_in_} features")

        K = self._compute_kernel_matrix(X, self.X_)
        return K @ self.dual_coef_ + self.b_
