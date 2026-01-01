from graphviz import Digraph

from .tree import Tree
from .node import Node


def build_graph(tree: Tree, filename: str, path: str, view: bool = True):
    node_attr = [
        ('shape', 'box'),
        ('style', 'filled,rounded'),
        ('fontname', 'helvetica')
    ];

    graph = Digraph('DTree',
                    filename=filename,
                    directory=path,
                    format='png',
                    node_attr=node_attr)

    _traverse_tree(graph, tree.root)

    if view:
        graph.view()

    return graph


def _traverse_tree(graph: Digraph, node: Node):
    if node is None:
        return

    node_id = _node_id(node)
    graph.node(node_id, label=_format_node_label(node))

    if node.left is not None:
        left_id = _node_id(node.left)
        graph.node(left_id, label=_format_node_label(node.left))
        graph.edge(node_id, left_id, label='<=')
        _traverse_tree(graph, node.left)

    if node.right is not None:
        right_id = _node_id(node.right)
        graph.node(right_id, label=_format_node_label(node.right))
        graph.edge(node_id, right_id, label='>')
        _traverse_tree(graph, node.right)


def _node_id(node: Node):
    return str(id(node))


def _format_node_label(node: Node):
    impurity = getattr(node, 'impurity', None)
    samples = getattr(node, 'n', None)
    count_dict = getattr(node, 'class_count_dict', None)

    if isinstance(count_dict, dict) and len(count_dict) > 0:
        labels = list(count_dict.keys())
        counts = list(count_dict.values())
        predicted_class = max(count_dict, key=count_dict.get)
        total = float(sum(count_dict.values()))
        prob = float(count_dict[predicted_class]) / total if total > 0 else 0.0
    else:
        labels, counts, predicted_class, prob = [], [], None, 0.0

    feat_index = getattr(node, 'split_dim', None)
    threshold = getattr(node, 'split_threshold', None)

    lines = []
    if impurity is not None:
        lines.append(f"Impurity: {impurity:.4f}")
    if samples is not None:
        lines.append(f"Samples: {samples}")
    if labels:
        lines.append(f"Labels: {labels}")
        lines.append(f"Counts: {counts}")

    if feat_index is not None and threshold is not None:
        lines.append(f"Split: X[{feat_index}] <= {threshold:.4f}")

    if predicted_class is not None:
        lines.append(f"Predict: {predicted_class} ({prob * 100:.1f}%)")

    return '\\n'.join(lines)
