import gmpy2 as gm
import math
import networkx as nx
import numpy as np
import random as rnd
from scipy.special import zeta


def barabasi_albert_naive(n, avg_deg, seed=None):
    if seed:
        rnd.seed(seed)

    g = nx.barabasi_albert_graph(n, int(math.ceil(avg_deg / 2)), seed)
    curr_deg = 2 * g.size() / g.order()

    while abs(avg_deg - curr_deg) > 1e-2:
        g.remove_edge(*list(rnd.choice(g.edges())))
        curr_deg = 2 * g.size() / g.order()

def barabasi_albert(n, avg_deg, seed=None):
    if seed:
        rnd.seed(seed)

    edges_seq = _get_seq_of_edges_to_add(avg_deg)
    g, i = _initial_graph()
    seq_idx = 0

    while g.order() < n:
        g.add_node(i)
        _add_edges_with_pref_attach(g, i, edges_seq[seq_idx])

    return g

def _get_seq_of_edges_to_add(avg_deg):
    aux = gm.mpq(avg_deg)
    seq_len = int(min(aux.numerator, aux.denominator))
    seq_val = int(max(aux.numerator, aux.denominator) / seq_len)
    edges_seq = [seq_val for i in range(seq_len)]
    edges_seq[0] += 1
    return edges_seq

def _initial_graph():
    g = nx.Graph()
    g.add_nodes_from([0, 1])
    g.add_edge(0, 1)
    i = 2
    return g, i

def _add_edges_with_pref_attach(g, node, m):
    for _ in range(m):
        to_node = np.random.choice(
            g.order(), 
            p=[deg for node, deg in g.degree()] / g.order()
        )

        g.add_edge(node, to_node)

def configuration_model(n, gamma, max_deg_f=lambda n, g: n, seed=None):
    if seed:
        np.random.seed(seed)

    max_deg = max_deg_f(n, gamma)
    deg_dist = np.zeros(max_deg)
    expected_degrees = []
    c = 1 / zeta(gamma)

    for i in range(max_deg):
        deg_dist[i] = c * ((i + 1) ** -gamma)
        expected_degrees += [i+1] * int(round(n * deg_dist[i]))

    g = nx.expected_degree_graph(
        expected_degrees, 
        seed=seed, 
        selfloops=False
    )

    cum_deg_dist = np.cumsum(deg_dist)
    cum_deg_dist[-1] = 1

    while len(g) < n:
        rnd = np.random.uniform()
        i = 0

        while rnd > cum_deg_dist[i]:
            i += 1

        g.add_node(len(g))

        for _ in range(i):
            neigh = np.random.randint(0, len(g) - 1)

            while g.degree(neigh) >= max_deg:
                neigh = np.random.randint(0, len(g) - 1)

            g.add_edge((len(g) - 1, neigh))
        
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

