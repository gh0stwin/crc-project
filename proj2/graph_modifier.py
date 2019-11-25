import numpy as np


def modify_deg_dist(g, expected_deg_dist, min_diff=1e-2):
    edges_array = np.array(g.edges)
    current_deg_dist = g.size() / len(g)

    while abs(expected_deg_dist - current_deg_dist) > min_diff:
        change_idx = np.random.randint(0, len(edges_array))
        current_edge = edges_array[change_idx]
        edges_array[change_idx] = (
            np.random.randint(0, len(g)),
            np.random.randint(0, len(g))
        )

        new_deg_dist = compute_array_deg_dist(edges_array)

        if (
            abs(expected_deg_dist - current_deg_dist) < 
            abs(expected_deg_dist - new_deg_dist)
        ):
            edges_array[change_idx] = current_edge
        else:
            current_deg_dist = new_deg_dist

def compute_array_deg_dist(len_g, array):
    nodes_deg = np.zeros((len_g, ))

    for i in nodes_deg:
        nodes_deg[i] = np.count_nonzero(array == i)

