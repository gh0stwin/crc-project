import networkx as nx
import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Degree(VaccinationProtocol):
    def __init__(self, g, f, state='state'):
        super(Degree, self).__init__(g, f, state)
        self._acronym = 'DE'

    def vaccinate_network(self, **kwargs):
        high_deg_nodes = sorted(
            list(self._g.degree),
            key=lambda el: el[1],
            reverse=True
        )[:self._nodes_to_vacc]

        for node, _ in high_deg_nodes:
            self._g.nodes[node][self._state] = SirState.VACCINATED

        return len(self._g)
