import networkx as nx
import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class TwoStepHeuristic(VaccinationProtocol):
    def __init__(self, g, f, state='state'):
        super(TwoStepHeuristic, self).__init__(g, f, state)
        self._acronym = 'TS'

    def vaccinate_network(self, **kwargs):
        return self._tsh(kwargs.get('n', 0))

    def _tsh(self, n):
        reduced_g = self._get_subgraph(n)
        high_deg_nodes = self._high_deg_nodes_in_subgraph(reduced_g, n)

        for node, _ in high_deg_nodes:
            self._g.nodes[node][self._state] = SirState.VACCINATED

        return 2 * n

    def _get_subgraph(self, n):
        n1_nodes = self._get_n_random_nodes(n)
        reduced_graph_nodes = list(n1_nodes)

        for node in n1_nodes:
            reduced_graph_nodes += [n for n in self._g.neighbors(node)]

        return self._g.subgraph(reduced_graph_nodes)

    def _high_deg_nodes_in_subgraph(self, reduced_graph, n):
        high_deg_nodes_reduced_g = sorted(
            list(reduced_graph.degree), 
            key=lambda el: el[1], 
            reverse=True
        )[:n]

        high_deg_nodes = sorted(
            list(self._g.degree(high_deg_nodes_reduced_g)),
            key=lambda el: el[1],
            reverse=True
        )[:self._n_nodes_to_vacc]

        diff = len(high_deg_nodes) - self._n_nodes_to_vacc

        if diff < 0:
            high_deg_nodes += self._get_n_random_nodes(abs(diff))

        return high_deg_nodes

