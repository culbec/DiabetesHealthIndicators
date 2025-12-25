import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

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

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        x = np.asarray(x)
        x_sub = x[self.features]
        return self.tree.predict(x_sub)


class RandomForest:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 n_trees: int,
                 max_features: int,
                 bootstrap_features: bool,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 seed: Optional[int] = None):
        self.data = np.asarray(data)
        self.labels = np.asarray(labels)

        assert self.data.ndim == 2, "Data must be a 2D array"
        assert self.labels.shape[0] == self.data.shape[0], "Number of samples in data and labels must be the same"

        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.sampler = BaggingSampler(
            data=self.data,
            labels=self.labels,
            n_bags=self.n_trees,
            max_features=self.max_features,
            bootstrap_features=self.bootstrap_features,
            seed=seed,
            oob=False   # TODO: can be turned on later for certain performance evaluation experiments
        )

        self.trees: List[BootstrappedTree] = []
        for i in range(self.n_trees):
            x_bag, y_bag, features = self.sampler.get_bag(i)
            self.trees.append(
                BootstrappedTree.train_from_bag(
                x_bag=x_bag,
                y_bag=y_bag,
                features=features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
        )

    def predict(self, x: np.ndarray, vote: str = 'hard') -> Tuple[int, float]:
        vote = vote.lower()
        if vote == 'hard':
            return self._predict_hard(x)
        elif vote == 'soft':
            return self._predict_soft(x)
        else:
            raise ValueError("voting strategy must be either 'hard' or 'soft'")

    def _predict_hard(self, x: np.ndarray) -> Tuple[int, float]:
        votes = [tb.predict(x)[0] for tb in self.trees]
        tally = Counter(votes)
        predicted_class, count = tally.most_common(1)[0]
        confidence = count / len(votes)
        return predicted_class, confidence

    def _predict_soft(self, x: np.ndarray) -> Tuple[int, float]:
        p1_sum = 0.0
        for cls, prob in (tb.predict(x) for tb in self.trees):
            prob = float(prob)
            if cls == 1:
                p1_sum += prob
            else:
                p1_sum += (1.0 - prob)

        p1 = p1_sum / len(self.trees)
        # TODO: make threshold configurable in case of class imbalance?
        predicted_class = 1 if p1 >= 0.5 else 0
        confidence = p1 if predicted_class == 1 else (1.0 - p1)
        return predicted_class, confidence