import networkx as nx
import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Degree(VaccinationProtocol):
    def __init__(self):
        super(Degree, self).__init__()
        self._acronym = 'DE'

    def vaccinate_network(self, g, f, state='state', **kwargs):
        n_nodes_to_vacc = int(round(len(g) * f))
        high_deg_nodes = sorted(
            list(g.degree),
            key=lambda el: el[1],
            reverse=True
        )[:n_nodes_to_vacc]

        for node, _ in high_deg_nodes:
            g.nodes[node][state] = SirState.VACCINATED

        return len(g)
