import numpy as np

from typing import Tuple
from .node import Node

class Tree:
    """
    Decision Tree implementation using binary splits for classification tasks.

    Supports both binary and multi-class classification.
    Designed as a base weak estimator for ensemble methods like Random Forest.
    """
    def __init__(self,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 impurity_metric: str = "gini"):
        self.root = None
        self.is_fitted_ = False

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.impurity_metric = impurity_metric

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        assert x.shape[0] == y.shape[0], "Number of samples in training data and labels must be the same"
        assert y.shape[1] == 1, "Labels array must be of shape (n_samples, 1)"

        self.root = Node(data=x,
                         labels=y,
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         min_samples_leaf=self.min_samples_leaf,
                         impurity_metric=self.impurity_metric)
        self.is_fitted_ = True

    def predict(self, x: np.ndarray ) -> Tuple[int, float]:
        assert self.is_fitted_, "The tree must be trained before prediction"
        assert x.ndim == 1, "Input sample must be of shape (n_features,), batch predictions are not supported"
        return self._get_node_prediction(x)

    def _get_node_prediction(self, x: np.ndarray) -> Tuple[int, float]:
        current = self.root
        while not current.is_leaf:
            if x[current.split_dim] <= current.split_threshold:
                current = current.left
            else:
                current = current.right

        return current.best_label, current.best_percentage