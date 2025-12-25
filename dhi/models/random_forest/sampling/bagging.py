import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class _Bag:
    indices: np.ndarray
    features: np.ndarray
    oob_indices: Optional[np.ndarray] = None


class BaggingSampler:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 n_bags: int,
                 max_features: int,
                 bootstrap_features: bool = False,
                 bag_size: Optional[int] = None,
                 seed: Optional[int] = None,
                 oob: bool = False):
        data = np.asarray(data)
        labels = np.asarray(labels)

        assert data.ndim == 2, "Data must be a 2D array"
        assert labels.shape[0] == data.shape[0], "Number of samples in data and labels must be the same"

        self.data = data
        self.labels = labels
        self.n_bags = n_bags
        self.num_features = data.shape[1]
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features
        self.seed = seed
        self.oob = oob

        self.n = data.shape[0]
        self.bag_size = bag_size if bag_size is not None else self.n

        assert max_features >= 1, "max_features must be at least 1"
        assert (self.num_features >= max_features) or self.bootstrap_features, \
            "max_features cannot exceed the total number of features unless bootstrap_features is True"
        assert 1 <= self.bag_size <= self.n, "bag_size must be between 1 and the number of samples"

        self.rng = np.random.default_rng(self.seed)

        self.bags = [self._make_bag() for _  in range(self.n_bags)]

    def _make_bag(self) -> _Bag:
        # feature subspace
        n_features = self.rng.integers(1, self.max_features + 1)
        features = self.rng.choice(
            self.num_features,
            size=n_features,
            replace=self.bootstrap_features
        )
        # bootstrap rows
        indices = self.rng.choice(
            self.n,
            size=self.bag_size,
            replace=True
        )

        oob_indices = None
        if self.oob:
            in_bag = np.zeros(self.n, dtype=bool)
            in_bag[indices] = True
            oob_indices = np.where(~in_bag)[0]

        return _Bag(indices=indices, features=features, oob_indices=oob_indices)

    def get_bag(self, bag_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert 0 <= bag_index < len(self.bags), "Bag index out of range"
        bag = self.bags[bag_index]

        bag_data = self.data[np.ix_(bag.indices, bag.features)]
        bag_labels = self.labels[bag.indices]

        return bag_data, bag_labels, bag.features

    def get_oob_indices(self, bag_index: int) -> Optional[np.ndarray]:
        assert 0 <= bag_index < len(self.bags), "Bag index out of range"
        return self.bags[bag_index].oob_indices
