import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Random(VaccinationProtocol):
    def __init__(self, g, f, state='state'):
        super(Random, self).__init__(g, f, state)
        self._acronym = 'RA'

    def vaccinate_network(self, **kwargs):
        for node in self._get_random_n_nodes(self._n_nodes_to_vacc):
            self._g.nodes[node][self._state] = SirState.VACCINATED

        return 0
