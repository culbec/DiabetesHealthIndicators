import numpy as np

from .impurity import compute_impurity


class Node:
    """
    Implementation of a CART-style decision tree node.

    Each node either:
    - Acts as a decision node (internal): splits data based on the best feature and threshold
    - Acts as a leaf node: holds the predicted class and class distribution for samples reaching this node; returns the majority class during prediction
    """
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 impurity_metric: str = 'gini',
                 depth: int = 0,
                 max_depth: int = 10,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 3):
        """
        Initialize the node and recursively grow the tree by finding the best splits

        :param data: the samples from the original data points reaching this node, with shape (n_samples, n_features)
        :param labels: the associated classes for each data point
        :param impurity_metric: 'gini' or 'entropy', metric used for deciding the best splits
        :param depth: current depth of the node in the tree
        :param max_depth: regularization parameter, maximum allowed recursion depth of the tree
        :param min_samples_split: regularization parameter, minimum samples required at the node to attempt splitting further
        :param min_samples_leaf: regularization parameter, minimum samples required in each child after the split
        """
        self.data = data
        self.labels = labels
        self.impurity_metric = impurity_metric
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.n = data.shape[0]
        self.split_dim = None
        self.split_threshold = None
        self.gain = None
        self.split_cost = None

        self.is_leaf = False
        self.left = None
        self.right = None

        self.class_labels, self.class_counts = np.unique(self.labels, return_counts=True)
        self.class_count_dict = {l: c for l, c in zip(self.class_labels, self.class_counts)}

        self.best_label = max(self.class_count_dict, key=self.class_count_dict.get)
        self.best_percentage = self.class_count_dict[self.best_label] / sum(self.class_counts)

        self.impurity = compute_impurity(self.class_counts, self.impurity_metric)

        # early stopping for node splitting if a node is already 100% pure (no need to instantiate node children)
        if self.impurity == 0.0:
            self.is_leaf = True
            return

        if self.depth >= self.max_depth or self.n < self.min_samples_split:
            self.is_leaf = True
            return

        # TODO: separate this training logic method call from the init?
        self.split_dim, self.split_threshold, self.split_cost = self._split_node() # grow the tree recursively

        if any(x is None for x in (self.left, self.right, self.split_dim, self.split_threshold)):
            self.is_leaf = True

    def _split_node(self):
        # find the best dimension (feature) to split on,
        # the threshold that will reduce the impurity the most on that dimension,
        # and the resulting impurity (split_cost) after the split
        split_dimension, split_threshold, split_cost = self._find_best_split()
        if split_threshold is None:
            return None, None, None

        self.split_threshold = split_threshold
        self.split_dim = split_dimension
        # information gain achieved by the split, representing the reduction in impurity from parent to children (higher is better)
        self.gain = self.impurity - split_cost

        left_indices = np.argwhere(self.data[:, split_dimension] <= split_threshold)
        left_data = self.data[left_indices[:, 0], :]
        left_labels = self.labels[left_indices[:, 0], 0]
        left_labels = np.atleast_2d(left_labels).T  # can be replaced with reshape(-1, 1)

        right_indices = np.argwhere(self.data[:, split_dimension] > split_threshold)
        right_data = self.data[right_indices[:, 0], :]
        right_labels = self.labels[right_indices[:, 0], 0]
        right_labels = np.atleast_2d(right_labels).T

        if len(left_indices) >= self.min_samples_leaf and len(right_indices) >= self.min_samples_leaf:
            self.left = self._create_child_node(left_data, left_labels)
            self.right = self._create_child_node(right_data, right_labels)

        return split_dimension, split_threshold, split_cost

    def _find_best_split(self):
        # best weighted impurity after the split, as the average of child node impurities (lower is better)
        best_split_cost = 1.
        best_threshold = None
        best_dimension = None

        sorted_indices = np.argsort(self.data, axis=0)

        for dim in range(sorted_indices.shape[1]):
            dim_indices = np.atleast_2d(sorted_indices[:, dim]).T
            current_split_cost, current_threshold = self._find_best_split_for_dim(dim, dim_indices)
            if current_split_cost < best_split_cost:
                best_split_cost = current_split_cost
                best_threshold = current_threshold
                best_dimension = dim

        return best_dimension, best_threshold, best_split_cost

    def _find_best_split_for_dim(self, dim: int, indices: np.ndarray):
        left_label_counts = {l: 0 for l in self.class_labels}
        right_label_counts = {l: c for l, c in zip(self.class_labels, self.class_counts)}

        best_threshold = None
        best_impurity = 1.

        for i in range(1, self.n):
            left_val = self.data[indices[i - 1, 0], dim]
            right_val = self.data[indices[i, 0], dim]

            if left_val == right_val:
                continue

            left_label_counts[self.labels[indices[i - 1, 0], 0]] += 1
            right_label_counts[self.labels[indices[i - 1, 0], 0]] -= 1

            left_counts = np.array(list(left_label_counts.values()))
            right_counts = np.array(list(right_label_counts.values()))

            g_left = compute_impurity(left_counts, self.impurity_metric)
            g_right = compute_impurity(right_counts, self.impurity_metric)

            total = sum(left_counts) + sum(right_counts)
            cost = (sum(left_counts) / total) * g_left + (sum(right_counts) / total) * g_right

            if cost < best_impurity and self.min_samples_leaf <= i <= self.n - self.min_samples_leaf:
                best_impurity = cost
                best_threshold = (left_val + right_val) / 2.

        return best_impurity, best_threshold

    def _create_child_node(self, data: np.ndarray, labels: np.ndarray):
        return Node(data=data,
                    labels=labels,
                    impurity_metric=self.impurity_metric,
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf
        )
