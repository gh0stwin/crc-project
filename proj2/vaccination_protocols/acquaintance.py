import numpy as np

from sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Acquaintance(VaccinationProtocol):
    def __init__(self, g, f, state='state'):
        super(Acquaintance, self).__init__(g, f, state)
        self._acronym = 'AC'

    def vaccinate_network(self, **kwargs):
        nodes = list(range(len(self._g)))

        for _ in range(int(round(len(self._g) * self._f))):
            self._vaccinate_acquaintance(
                nodes, 
                self._g[np.random.randint(0, len(self._g))]
            )

        return len(self._g) * self._f


    def _vaccinate_acquaintance(self, nodes, picked_node):
        neighs = []
        node_to_vacc = None

        for i in self._g.neighbors(picked_node):
            if self._g.nodes[i][self._state] == SirState.SUSCEPTIBLE:
                neighs.append(i)

        if len(neighs) > 0:
            node_to_vacc = neighs[np.random.randint(0, len(neighs))]
            nodes.remove(node_to_vacc)
        else:
            node_to_vacc = nodes.pop(np.random.randint(0, len(nodes)))
        
        self._g.nodes[node_to_vacc][self._state] = SirState.VACCINATED
