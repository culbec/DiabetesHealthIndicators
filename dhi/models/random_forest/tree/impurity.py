import numpy as np


def gini_impurity(class_counts: np.ndarray) -> float:
    total = np.sum(class_counts)
    return 1. - np.sum(np.square(class_counts / total)) if total > 0 else 0.0


def entropy(class_counts: np.ndarray) -> float:
    total = np.sum(class_counts)
    if total == 0:
        return 0.0

    probs = class_counts / total
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))


def compute_impurity(class_counts: np.ndarray, metric: str) -> float:
    if metric == 'gini':
        return gini_impurity(class_counts)
    elif metric == 'entropy':
        return entropy(class_counts)
    else:
        raise ValueError(f"Unknown impurity metric: {metric}")
