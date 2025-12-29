import numpy as np

from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin

from dhi.models.random_forest.tree.tree import Tree
from dhi.models.random_forest.sampling.bagging import BaggingSampler


@dataclass
class BootstrappedTree:
    features: np.ndarray
    tree: Tree

    @classmethod
    def train_from_bag(cls,
                       x_bag: np.ndarray,
                       y_bag: np.ndarray,
                       features: np.ndarray,
                       max_depth: int,
                       min_samples_split: int,
                       min_samples_leaf: int) -> 'BootstrappedTree':
        tree = Tree(max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf)
        tree.fit(x_bag, y_bag)
        return cls(features=features, tree=tree)

    def predict_one(self, x: np.ndarray) -> Tuple[int, float]:
        x = np.asarray(x)
        x_sub = x[self.features]
        return self.tree.predict(x_sub)


class RandomForest(BaseEstimator, ClassifierMixin):
    """
    RandomForest binary classifier implementation compatible with Sklearn API.

    - __init__() stores only hyperparameters
    - fit(X, y) trains and sets the learned attributes
    - predict(X) makes class labels predictions
    - predict_proba(X) makes class probabilities predictions
    """

    def __init__(self,
                 n_trees: int,
                 max_features: int,
                 bootstrap_features: bool,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 vote: str = "hard",
                 threshold: float = 0.5,
                 seed: Optional[int] = None):
        # TODO: add default values to all hyperparameters?
        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.vote = vote.lower()
        self.threshold = threshold

        self.seed = seed

        if self.vote not in {"hard", "soft"}:
            raise ValueError("vote must be 'hard' or 'soft'")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        X = np.asarray(X)
        y = np.asarray(y)

        assert X.ndim == 2, "Input data must be a 2D array (n_samples, n_features)"
        self.n_features_in_ = X.shape[1]
        # allow y to be passed as batched (n,1) or unbatched (n, ) input shape
        if y.ndim == 2 and y.shape[1] == 1:
            y_1d = y[:, 0].astype(int)
            y_col = y.astype(int)
        elif y.ndim == 1:
            y_1d = y.astype(int)
            y_col = y_1d.reshape(-1, 1)
        else:
            raise ValueError(f"Output label data must be of shape (n,) or (n,1). Got {y.shape}")

        assert y_col.shape[0] == X.shape[0], "Number of samples in data and expected labels must be the same size"

        # ensure only binary classification tasks are attempted
        classes = np.unique(y_1d)
        if set(classes.tolist()) != {0, 1}:
            raise ValueError(
                f"Model supports binary classification input only, with expected labels {{0, 1}}. Got {classes.tolist()}")

        self.classes_ = np.array([0, 1], dtype=int) # or use the more generic classes variable, computed using np.unique
        self.n_classes_ = 2

        self.sampler_ = BaggingSampler(
            data=X,
            labels=y_col.astype(int),
            n_bags=self.n_trees,
            max_features=self.max_features,
            bootstrap_features=self.bootstrap_features,
            seed=self.seed,
            oob=False  # TODO: can be turned on later for certain performance evaluation experiments
        )

        self.trees_ = []
        for i in range(self.n_trees):
            x_bag, y_bag, features = self.sampler_.get_bag(i)
            self.trees_.append(
                BootstrappedTree.train_from_bag(
                    x_bag=x_bag,
                    y_bag=y_bag,
                    features=features,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf
                )
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with {self.n_features_in_} features")

        preds = [self._predict_one(X[i])[0] for i in range(X.shape[0])]
        return np.asarray(preds, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with {self.n_features_in_} features")

        p1 = np.array([self._p1_one(X[i]) for i in range(X.shape[0])], dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def _p1_one(self, x: np.ndarray) -> float:
        """
        Return the forest-averaged probability P(y=1|x) for a single sample.
        Aggregation is done by averaging the per-tree predicted probabilities.
        """
        p1_sum = 0.0
        for cls, prob in (tb.predict_one(x) for tb in self.trees_):
            prob = float(prob)
            p1_sum += prob if cls == 1 else (1.0 - prob)
        return p1_sum / len(self.trees_)

    def _predict_one(self, x: np.ndarray) -> Tuple[int, float]:
        if self.vote == "hard":
            return self._predict_one_hard(x)
        elif self.vote == "soft":
            return self._predict_one_soft(x)
        else:
            raise ValueError("Voting strategy must be either 'hard' or 'soft'")

    def _predict_one_hard(self, x: np.ndarray) -> Tuple[int, float]:
        votes = [tb.predict_one(x)[0] for tb in self.trees_]
        tally = Counter(votes)
        pred, count = tally.most_common(1)[0]
        confidence = count / len(votes)
        return pred, confidence

    def _predict_one_soft(self, x: np.ndarray) -> Tuple[int, float]:
        p1 = self._p1_one(x)
        pred = 1 if p1 >= self.threshold else 0
        confidence = p1 if pred == 1 else (1.0 - p1)
        return pred, confidence

    def _check_fitted(self):
        if not hasattr(self, "trees_") or self.trees_ is None or len(self.trees_) == 0:
            raise NotFittedError("Estimator not fitted. "
                                 "Call fit with appropriate input data before using this estimator.")
        if not hasattr(self, "n_features_in_") or not hasattr(self, "classes_"):
            raise NotFittedError("Estimator not fitted and missing metadata. "
                                 "Call fit with appropriate input data before using this estimator.")
