from abc import ABC, abstractmethod

from sir.sir import Sir


class Simulator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _get_vacc_prot_and_args(self, **kwargs):
        pass

    def simulate_sir(self, file, g, beta, f, seed):
        vacc_prot, vacc_prot_args = self._get_vacc_prot_and_args()
        res, c = Sir(g.copy(), beta, vacc_prot, {}, f, seed).simulate()
        file.write(
            '{},{},{},{}'.format(seed, res[-1][0], vacc_prot.acronym, c)
        )

        return seed
