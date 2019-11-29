from abc import ABC, abstractmethod
import numpy as np


class Vaccination_protocol(ABC):
    def __init__(self, state='state'):
        self._state = state
        self._acronym = ''

    @property
    def acronym(self):
        return self._acronym

    @abstractmethod
    def vaccinate_network(self, g, f, **kwargs):
        pass

    def _get_random_n_nodes(self, n):
        nodes = list(range(len(self._g)))
        selected_nodes = []

        for _ in range(n):
            node_idx = np.random.randint(0, len(nodes))
            selected_nodes.append(nodes.pop(node_idx))

        return selected_nodes


