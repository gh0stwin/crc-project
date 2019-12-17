from abc import ABC, abstractmethod
import numpy as np


class VaccinationProtocol(ABC):
    def __init__(self):
        self._acronym = ''

    @property
    def acronym(self):
        return self._acronym

    @abstractmethod
    def vaccinate_network(self, g, f, **kwargs):
        pass

    def _get_n_random_nodes(self, g, n, replace=False):
        nodes = list(range(len(g)))
        selected_nodes = []

        for _ in range(n):
            node_idx = np.random.randint(0, len(nodes))

            if replace:
                node = node[node_idx]
            else:
                node = nodes.pop(node_idx)

            selected_nodes.append(node)

        return selected_nodes


