from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.two_step_heuristic import TwoStepHeuristic


class TwoStepHeuristicSimulator(Simulator):
    def __init__(self, n_func=lambda n: range(0, n, int(n / 10))):
        super(TwoStepHeuristicSimulator, self).__init__()
        self._n_func = n_func

    def _get_vacc_prot_and_args(self, **kwargs):
        g = kwargs['g']
        vacc_prots_and_args = []

        for n in self._n_func(len(g)):
            vacc_prots_and_args.append([TwoStepHeuristic(), {'n': n}])

        return vacc_prots_and_args

    def simulate_sir(self, file, g, beta, f, seed):
        for vacc_prot, vacc_args in self._get_vacc_prot_and_args(g=g):
            res = Sir(
                g.copy(), beta, vacc_prot, {}, f, seed
            ).simulate()

            file.write('{},{},{},{}\n'.format(
                seed,
                res[-1], 
                vacc_prot.acronym, 
                vacc_args['n']
            ))

            seed += 1

        return seed - 1
