import numpy as np

from typing import Tuple
from .node import Node

class Tree:
    def __init__(self,
                 max_depth: int = 10,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 3):
        self.head = None
        self.is_trained = False
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        assert x.shape[0] == y.shape[0], "Number of samples in training data and labels must be the same"
        assert y.shape[1] == 1, "Labels array must be of shape (n_samples, 1)"

        self.head = Node(data=x,
                         labels=y,
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         min_samples_leaf=self.min_samples_leaf)
        self.is_trained = True

    def predict(self, x: np.ndarray ) -> Tuple[int, float]:
        assert self.is_trained, "The tree must be trained before prediction"
        assert x.ndim == 1, "Input sample must be of shape (n_features,), batch predictions are not supported"
        return self._get_node_prediction(x)

    def _get_node_prediction(self, x: np.ndarray) -> Tuple[int, float]:
        current = self.head
        while not current.is_leaf:
            if x[current.split_dim] <= current.split_threshold:
                current = current.left
            else:
                current = current.right

        return current.best_label, current.best_percentage