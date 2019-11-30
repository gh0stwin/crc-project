import numpy as np

from sir.sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class RandomWalk(VaccinationProtocol):
    def __init__(self):
        super(RandomWalk, self).__init__()
        self._acronym = 'RW'

    def vaccinate_network(self, g, f, state='state', **kwargs):
        alpha = kwargs.get('alpha', 3)
        m = kwargs.get('m', 0)
        n_nodes_to_vacc = int(round(len(g) * f))

        visit_node, visit_node_deg = None, None
        best_nodes, best_degrees = self._get_random_nodes_and_degs(
            g,
            n_nodes_to_vacc
        )

        for _ in range(m + 1):
            visit_node, visit_node_deg = self._choose_node(
                g,
                alpha,
                visit_node, 
                visit_node_deg
            )

            if visit_node_deg > best_degrees[-1]:
                best_nodes[-1] = visit_node
                best_degrees[-1] = visit_node_deg
                idxs = np.argsort(best_degrees)[::-1]
                best_nodes = best_nodes[idxs]
                best_degrees = best_degrees[idxs]

        self._vaccinate_nodes(g, best_nodes, state)
        return m * (1 - alpha / (2 * g.size() / len(g) + alpha))

    def _get_random_nodes_and_degs(self, g, n):
        best_nodes = np.array(self._get_n_random_nodes(g, n), dtype=int)
        best_degrees = np.array(
            [g.degree[node] for node in best_nodes],
            dtype=int
        )

        idxs = np.argsort(best_degrees)[::-1]
        return best_nodes[idxs], best_degrees[idxs]


    def _choose_node(self, g, alpha, current_node, current_node_deg):
        visit_node = None

        if current_node == None:
                visit_node = np.random.randint(0, len(g))
        else:
            if np.random.uniform() < alpha / (
                current_node_deg + alpha
            ):
                visit_node = np.random.randint(0, len(g))
            else:
                current_node_neighs = list(
                    g.neighbors(current_node)
                )

                visit_node = current_node_neighs[
                    np.random.randint(0, len(current_node_neighs))
                ]

        return visit_node, g.degree[visit_node]

    def _vaccinate_nodes(self, g, nodes, state):
        for node in nodes:
            g.nodes[node][state] = SirState.VACCINATED
