import gmpy2 as gm
import math
import networkx as nx
import numpy as np
import random as rnd


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
