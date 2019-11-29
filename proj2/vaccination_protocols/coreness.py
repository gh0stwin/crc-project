import networkx as nx
import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Coreness(VaccinationProtocol):
    def __init__(self, g, f, state='state'):
        super(Coreness, self).__init__(g, f, state)
        self._acronym = 'CO'

    def vaccinate_network(self, **kwargs):
        high_core_nodes = sorted(
            nx.core_number(self._g).items(),
            key=lambda el: el[1],
            reverse=True
        )[:self._n_nodes_to_vacc]

        for node, _ in high_core_nodes:
            self._g.nodes[node][self._state] = SirState.VACCINATED

        return len(self._g)
