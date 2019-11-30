import networkx as nx
import numpy as np

from sir.sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Bfs(VaccinationProtocol):
    def __init__(self):
        super(Bfs, self).__init__()
        self._acronym = 'BF'

    def vaccinate_network(self, g, f, state='state', **kwargs):
        n_nodes_to_vacc = int(round(len(g) * f))
        nodes = nx.bfs_predecessors(
            g, np.random.randint(0, len(g))
        ).keys()


        for i in range(n_nodes_to_vacc):
            g.nodes[nodes[i]][state] = SirState.VACCINATED
            i += 1

        return 0
