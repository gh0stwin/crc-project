import numpy as np

from sir.sir import SirState
from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class Acquaintance(VaccinationProtocol):
    def __init__(self):
        super(Acquaintance, self).__init__()
        self._acronym = 'AC'

    def vaccinate_network(self, g, f, state='state', **kwargs):
        unvacc_nodes = list(range(len(g)))
        n_nodes_to_vacc = int(round(len(g) * f))

        for _ in range(n_nodes_to_vacc):
            self._vaccinate_acquaintance(
                g,
                unvacc_nodes, 
                np.random.randint(0, len(g)),
                state
            )

        return len(g) * f


    def _vaccinate_acquaintance(
        self, 
        g, 
        unvacc_nodes, 
        picked_node, 
        state
    ):
        neighs = []
        node_to_vacc = None

        for i in g.neighbors(picked_node):
            if g.nodes[i][state] == SirState.SUSCEPTIBLE:
                neighs.append(i)

        if len(neighs) > 0:
            node_to_vacc = neighs[np.random.randint(0, len(neighs))]
            unvacc_nodes.remove(node_to_vacc)
        else:
            node_to_vacc = unvacc_nodes.pop(
                np.random.randint(0, len(unvacc_nodes))
            )
        
        g.nodes[node_to_vacc][state] = SirState.VACCINATED
