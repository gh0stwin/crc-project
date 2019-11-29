import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class RandomWalk(VaccinationProtocol):
    def __init__(self, g, f, state='state', alpha=3):
        super(RandomWalk, self).__init__(g, f, state)
        self._acronym = 'RW'
        self._alpha = alpha

    def vaccinate_network(self, **kwargs):
        self._random_walk(kwargs.get('m', 0))

    def _random_walk(self, m):
        visit_node, visit_node_deg = None, None
        best_nodes, best_degrees = self._get_random_nodes_and_degs(
            self._n_nodes_to_vacc
        )

        for _ in range(m + 1):
            visit_node, visit_node_deg = self._choose_node(
                visit_node, 
                visit_node_deg
            )

            if visit_node_deg > best_degrees[-1]:
                best_nodes[-1] = visit_node
                best_degrees[-1] = visit_node_deg
                idxs = np.argsort(best_degrees)[::-1]
                best_nodes = best_nodes[idxs]
                best_degrees = best_degrees[idxs]

        self._vaccinate_nodes(best_nodes)
        return m * (1 - self._alpha / (self._g.size() + self._alpha))

    def _get_random_nodes_and_degs(self, n):
        best_nodes = np.array(self._get_n_random_nodes(n), dtype=int)
        best_degrees = np.array(
            [self._g.degree[node] for node in best_nodes],
            dtype=int
        )

        idxs = np.argsort(best_degrees)[::-1]
        return best_nodes[idxs], best_degrees[idxs]


    def _choose_node(self, current_node, current_node_deg):
        visit_node = None

        if current_node == None:
                visit_node = np.random.randint(0, len(self._g))
        else:
            if np.random.uniform() < self._alpha / (
                current_node_deg + self._alpha
            ):
                visit_node = np.random.randint(0, len(self._g))
            else:
                current_node_neighs = list(
                    self._g.neighbors(current_node)
                )

                visit_node = current_node_neighs[
                    np.random.randint(0, len(current_node_neighs))
                ]

        return visit_node, self._g.degree[visit_node]

    def _vaccinate_nodes(self, nodes):
        for node in nodes:
            self._g.nodes[node][self._state] = SirState.VACCINATED
