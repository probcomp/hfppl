"""Utility functions"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def logsumexp(nums):
    m = np.max(nums)
    return np.log(np.sum(np.exp(nums - m))) + m
    
def log_softmax(nums):
    """Compute log(softmax(nums)).
    
    Args:
        nums: a vector or numpy array of unnormalized log probabilities.
    
    Returns:
        np.array: an array of log (normalized) probabilities.
    """
    return nums - logsumexp(nums)

def softmax(nums):
    return np.exp(log_softmax(nums))

def build_graph(LLM, node=None, path=None, graph=None, level=0):
    if graph is None:
        graph = nx.DiGraph()
    if node is None:
        node = LLM.cache
    if path is None:
        path = []

    # Add node to graph
    node_id = '->'.join([str(token_id) for token_id in path])
    node_label = LLM.tokenizer.decode([path[-1]]) if path else 'ROOT'
    graph.add_node(node_id, label=node_label, level=level)

    # Add edge to graph
    if path:
        parent_id = '->'.join([str(token_id) for token_id in path[:-1]])
        graph.add_edge(parent_id, node_id)

    # Recurse on children
    for token_id, child in node.children.items():
        build_graph(LLM, child, path + [token_id], graph, level + 1)

    return graph

def draw_graph(graph):
    pos = nx.multipartite_layout(graph, subset_key="level")  # Position nodes at different levels
    labels = {node: data['label'] for node, data in graph.nodes(data=True)}
    nx.draw(graph, pos, labels=labels, arrows=True)
    plt.show()

def show_graph(LLM):
    graph = build_graph(LLM)
    draw_graph(graph)
