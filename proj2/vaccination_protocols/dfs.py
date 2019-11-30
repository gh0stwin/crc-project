import networkx as nx
import numpy as np

from sir.sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Dfs(VaccinationProtocol):
    def __init__(self):
        super(Dfs, self).__init__()
        self._acronym = 'DF'

    def vaccinate_network(self, g, f, state='state', **kwargs):
        n_nodes_to_vacc = int(round(len(g) * f))
        nodes = nx.dfs_predecessors(
            g, np.random.randint(0, len(g))
        ).keys()


        for i in range(n_nodes_to_vacc):
            g.nodes[nodes[i]][state] = SirState.VACCINATED
            i += 1

        return 0
