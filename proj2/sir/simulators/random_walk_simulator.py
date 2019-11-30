from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.random_walk import RandomWalk


class RandomWalkSimulator(Simulator):
    def __init__(self, alpha=3, m_f=lambda n: range(0, n, int(n / 10))):
        super(RandomWalkSimulator, self).__init__()
        self._alpha = 3
        self._m_func = m_f

    def _get_vacc_prot_and_args(self, **kwargs):
        g = kwargs['g']
        vacc_prots_and_args = []

        for m in self._m_func(len(g)):
            vacc_prots_and_args.append([RandomWalk(), {'m': m}])

        return vacc_prots_and_args

    def simulate_sir(self, file, g, beta, f, seed):
        g = g.copy()

        for vacc_prot, vacc_args in self._get_vacc_prot_and_args(g=g):
            res, c = Sir(g, beta, vacc_prot, {}, f, seed).simulate()
            file.write('{},{},{},{},{}'.format(
                seed,
                res[-1][0], 
                vacc_prot.acronym, 
                c,
                vacc_args['m']
            ))

            seed += 1

        return seed - 1
