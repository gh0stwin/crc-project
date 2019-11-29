import networkx as nx
import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Coreness(VaccinationProtocol):
    def __init__(self):
        super(Coreness, self).__init__()
        self._acronym = 'CO'

    def vaccinate_network(self, g, f, state='state'):
        n_nodes_to_vacc = int(round(len(g) * f))
        high_core_nodes = sorted(
            nx.core_number(g).items(),
            key=lambda el: el[1],
            reverse=True
        )[:n_nodes_to_vacc]

        for node, _ in high_core_nodes:
            g.nodes[node][state] = SirState.VACCINATED

        return len(g)
