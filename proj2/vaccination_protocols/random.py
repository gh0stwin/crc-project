import numpy as np

from sir.sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Random(VaccinationProtocol):
    def __init__(self):
        super(Random, self).__init__()
        self._acronym = 'RA'

    def vaccinate_network(self, g, f, state='state', **kwargs):
        n_nodes_to_vacc = int(round(len(g) * f))

        for node in self._get_n_random_nodes(g, n_nodes_to_vacc):
            g.nodes[node][state] = SirState.VACCINATED

        return 0
