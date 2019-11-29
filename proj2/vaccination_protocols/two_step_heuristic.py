import networkx as nx
import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class TwoStepHeuristic(VaccinationProtocol):
    def __init__(self, g, f, state='state'):
        super(TwoStepHeuristic, self).__init__(g, f, state)
        self._acronym = 'TS'

    def vaccinate_network(self, **kwargs):
        n = kwargs.get('n', 0)
        nodes_to_vacc = int(round(len(self._g) * self._f))

        if n < nodes_to_vacc:
            n = nodes_to_vacc

        return self._tsh(n)

    def _tsh(self, n):
        reduced_g = self._get_subgraph(n)
        high_deg_nodes = self._high_deg_nodes_in_subgraph(reduced_g, n)

        for node, _ in high_deg_nodes:
            self._g.nodes[node][self._state] = SirState.VACCINATED

        return 2 * n

    def _get_subgraph(self, n):
        n1_nodes = np.arange(len(self._g))
        np.random.shuffle(n1_nodes)
        n1_nodes = n1_nodes[:n]
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

        return sorted(
            list(self._g.degree(high_deg_nodes_reduced_g)),
            key=lambda el: el[1],
            reverse=True
        )[:self._nodes_to_vacc]

