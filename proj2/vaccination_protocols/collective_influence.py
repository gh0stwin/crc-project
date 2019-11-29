import networkx as nx
import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class CollectiveInfluence(VaccinationProtocol):
    def __init__(self, g, f, state='state'):
        super(CollectiveInfluence, self).__init__(g, f, state)
        self._acronym = 'CI'
        self._l = 3

    def vaccinate_network(self, **kwargs):
        self._compute_ci()
        return len(self._g)

    def _compute_ci(self):
        graph_cp = self._g.copy()

        for _ in range(self._n_nodes_to_vacc):
            max_ci = -1
            node_with_max_ci = None

            for node in graph_cp:
                s = 0
                k_neighs = nx.single_source_shortest_path_length(
                    graph_cp,
                    node,
                    self._l
                )

                for neigh in k_neighs.keys():
                    if neigh == node:
                        continue

                    s += graph_cp.degree(neigh) - 1

                current_ci = (graph_cp.degree(node) - 1) * s

                if current_ci > max_ci:
                    node_with_max_ci = node 
                    max_ci = current_ci

            graph_cp.remove_node(node_with_max_ci)
            self._g.nodes[node_with_max_ci][self._state] = \
                SirState.VACCINATED
