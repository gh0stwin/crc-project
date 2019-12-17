import networkx as nx
import numpy as np

from sir.sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class CollectiveInfluence(VaccinationProtocol):
    def __init__(self):
        super(CollectiveInfluence, self).__init__()
        self._acronym = 'CI'

    def vaccinate_network(self, g, f, state='state', **kwargs):
        l = kwargs.get('l', 3)
        self._compute_ci(g, f, l, state)
        return len(g)

    def _compute_ci(self, g, f, l, state):
        n_nodes_to_vacc = int(round(len(g) * f))
        graph_cp = g.copy()

        for _ in range(n_nodes_to_vacc):
            max_ci = -1
            node_with_max_ci = None

            for node in graph_cp:
                s = 0
                k_neighs = nx.single_source_shortest_path_length(
                    graph_cp,
                    node,
                    l
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
            g.nodes[node_with_max_ci][state] = \
                SirState.VACCINATED
