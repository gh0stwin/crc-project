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
        vacc_nodes = 0
        nodes = None

        while vacc_nodes < n_nodes_to_vacc:
            nodes = list(nx.dfs_predecessors(
                g, np.random.randint(0, len(g))
            ).keys())

            for node in nodes:
                if g.nodes[node][state] == SirState.SUSCEPTIBLE:
                    g.nodes[node][state] = SirState.VACCINATED
                    vacc_nodes += 1

                if vacc_nodes >= n_nodes_to_vacc:
                    break

        return len(g) * f
