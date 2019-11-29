import networkx as nx
import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class TwoStepHeuristic(VaccinationProtocol):
    def __init__(self, g, f, state='state', n=0):
        super(TwoStepHeuristic, self).__init__(g, f, state)
        self._acronym = 'TS'
        self._n = n

    def vaccinate_network(self, g, f, state='state', **kwargs):
        n = kwargs.get('n', 0)
        reduced_g = self._get_subgraph(g, n)
        high_deg_nodes = self._high_deg_nodes_in_subgraph(
            reduced_g, 
            n
        )

        for node, _ in high_deg_nodes:
            g.nodes[node][state] = SirState.VACCINATED

        return 2 * n

    def _get_subgraph(self, g, n):
        n1_nodes = self._get_n_random_nodes(n)
        reduced_graph_nodes = list(n1_nodes)

        for node in n1_nodes:
            reduced_graph_nodes += [n for n in g.neighbors(node)]

        return g.subgraph(reduced_graph_nodes)

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

