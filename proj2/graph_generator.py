import math
import networkx as nx
import numpy as np
import random as rnd
from scipy.special import zeta


def configuration_model(n, gamma, max_deg_f=lambda n, g: n, seed=None):
    if seed:
        np.random.seed(seed)

    max_deg = max_deg_f(n, gamma)
    sum_degrees = 0
    deg_dist = np.zeros(max_deg)
    expected_degrees = []
    c = 1 / zeta(gamma)

    for i in range(max_deg):
        deg_dist[i] = c * ((i + 1) ** -gamma)
        add_nodes = int(round(n * deg_dist[i]))
        sum_degrees += (i + 1) * add_nodes
        expected_degrees += [i+1] * add_nodes

    if sum_degrees % 2 == 1:  # Sum of degrees has to be even
        expected_degrees[0] += 1
        sum_degrees += 1

    g = nx.configuration_model(
        expected_degrees, 
        seed=seed, 
        create_using=nx.Graph
    )

    g.remove_edges_from(nx.selfloop_edges(g))
    g.add_nodes_from(range(len(g), n))
    cum_deg_dist = np.cumsum(deg_dist)
    cum_deg_dist[-1] = 1
    return _connect_isolated_nodes(g, cum_deg_dist, max_deg)

def _connect_isolated_nodes(g, cum_deg_dist, max_deg):
    for node in g.nodes():
        if g.degree(node) > 0:
            continue

        rnd = np.random.uniform()
        i = 0

        while rnd > cum_deg_dist[i]:
            i += 1

        for _ in range(i + 1):
            neigh = node

            while g.degree(neigh) >= max_deg or neigh == node:
                neigh = np.random.randint(0, len(g))

            g.add_edge(node, neigh)
        
    return g

def dms(n, seed=None):
    if seed:
        np.random.seed(seed)

    g = nx.Graph()
    g.add_nodes_from(range(3))
    edges = [(0, 1), (1, 2), (2, 0)]
    g.add_edges_from(edges)

    for _ in range(n - 3):
        idx = np.random.randint(0, len(edges))
        new_edges = [(len(g), edges[idx][0]), (len(g), edges[idx][1])]
        g.add_node(len(g))
        g.add_edges_from(new_edges)
        edges += new_edges

    return g

