import numpy as np

from numpy.typing import ArrayLike
from typing import Optional, Tuple, List, Iterator


class KFoldCVSplitter:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    def split(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X_arr = np.asarray(X)
        n_samples = int(X_arr.shape[0])
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        folds = np.array_split(indices, self.n_splits)
        for i in range(self.n_splits):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train_idx, val_idx


class StratifiedKFoldCVSplitter:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    def split(self, X: ArrayLike, y: ArrayLike) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_samples = int(X_arr.shape[0])

        if y_arr.shape[0] != n_samples:
            raise ValueError(f"X and y must have the same number of samples, got {n_samples} and {y_arr.shape[0]}")

        classes, y_inv = np.unique(y_arr, return_inverse=True)
        n_classes = int(classes.shape[0])
        if n_classes < 2:
            raise ValueError("Stratified K-Fold requires at least 2 classes in the target variable")

        rng = np.random.default_rng(self.random_state)

        per_class: List[np.ndarray] = []
        for c in range(n_classes):
            idx = np.flatnonzero(y_inv == c)
            if self.shuffle:
                rng.shuffle(idx)
            per_class.append(idx)

        fold_bins: List[List[int]] = [[] for _ in range(self.n_splits)]
        for idx in per_class:
            for j, sample_idx in enumerate(idx.tolist()):
                fold_bins[j % self.n_splits].append(sample_idx)

        all_idx = np.arange(n_samples)
        for i in range(self.n_splits):
            val_idx = np.array(fold_bins[i], dtype=int)
            val_idx.sort()
            mask = np.ones(n_samples, dtype=bool)
            mask[val_idx] = False
            train_idx = all_idx[mask]
            yield train_idx, val_idx

