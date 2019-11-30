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

    def _get_n_random_nodes(self, g, n):
        nodes = list(range(len(g)))
        selected_nodes = []

        for _ in range(n):
            node_idx = np.random.randint(0, len(nodes))
            selected_nodes.append(nodes.pop(node_idx))

        return selected_nodes


