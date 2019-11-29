import networkx as nx
import pathlib as pl
import sys
import time

from sir import Sir
from vaccination_protocols.random import Random
from vaccination_protocols.acquaintance import Acquaintance
from vaccination_protocols.random_walk import RandomWalk
from vaccination_protocols.two_step_heuristic import TwoStepHeuristic
from vaccination_protocols.degree import Degree
from vaccination_protocols.coreness import Coreness
from vaccination_protocols.collective_influence import CollectiveInfluence

class SirHandler(object):
    def __init__(
        self, 
        path='./results/', 
        vacc_protocols=[
            Random(),
            Acquaintance(),
            RandomWalk(),
            TwoStepHeuristic(),
            Degree(),
            Coreness(),
            CollectiveInfluence()
        ],
        vacc_protocol_kwargs=[
            {},
            {},
            {'m': range(0, n, int(n / 10))},
            {'n': range(0, n, int(n / 10))},
            {},
            {},
            {}
        ]
    ):
        self._store_path = pl.Path(path)
        self._vacc_protocols = vacc_protocols
        self._vacc_protocols_kwargs = vacc_protocol_kwargs

    def simulate(self, files, betas, fs, times_per_beta_f, seed=0):
        self._seed = seed

        for file in files:
            g = nx.convert_node_labels_to_integers(nx.read_gml(file))

            for beta in betas:
                for f in fs:
                    self._simulate_for_net_beta_f(
                        self._get_res_filename(file, beta, f),
                        g, 
                        beta, 
                        f, 
                        times_per_beta_f
                    )

    def _get_res_filename(self, file, beta, f):
        return self._store_path.joinpath(
            (
                pl.Path(file).stem.rsplit('_', 1)[0] + 
                '_{}_{}'.format(beta, f).replace('.', '') + '.out'
            )
        )

    def _simulate_for_net_beta_f(self, res_nm, g, beta, f, iters):
        res_f = open(res_nm, 'a')

        for i in range(iters):
            self._simulate_all_vacc_protocols(
                g, 
                beta, 
                f, 
                res_nm, 
                res_f, 
                i
            )
        
        res_f.close()

    def _simulate_all_vacc_protocols(
        self, 
        g, 
        beta, 
        f, 
        res_nm, 
        res_f, 
        i
    ):
        i = 0
        
        for vacc_protocol in self._vacc_protocols:
            if vacc_protocol.__name__ == 'RandomWalk':
                for value in self._vacc_protocols_kwargs[i]['m']:
                    g_aux = g.copy()
                    vacc_prot_args = {'m': value}

                    vacc_prot_args
                    self._simulate_graph(
                        g, 
                        beta, 
                        f, 
                        vacc_protocol,
                        vacc_prot_args,
                        res_nm, 
                        res_f, 
                        i
                    )
                    
            elif vacc_protocol.__name__ == 'TwoStepHeuristic':
                for value in self._vacc_protocols[i]['n']:
                    g_aux = g.copy()
                    vacc_prot_args = {'n': value}

                    self._simulate_graph(
                        g, 
                        beta, 
                        f, 
                        vacc_protocol,
                        vacc_prot_args,
                        res_nm, 
                        res_f, 
                        i
                    )

            else:
                g_aux = g.copy()

                self._simulate_graph(
                    g, 
                    beta, 
                    f, 
                    vacc_protocol,
                    self._vacc_protocol_kwargs[i], 
                    res_nm, 
                    res_f, 
                    i
                )

            i += 1

    def _simulate_graph(
        self, 
        g, 
        beta, 
        f, 
        vacc,
        vacc_args, 
        res_nm, 
        res_f, 
        i
    ):
        sys.stdout.write('\rnetwork: {}, it: {}'.format(res_nm, i))
        sys.stdout.flush()
        r = Sir(
            g.copy(), beta, f, vacc, vacc_args, self._seed
        ).simulate()

        if 'm' in vacc_args:
            res_f.write('{},{},{},{}\n'.format(
                self._seed, 
                r[-1][0], 
                vacc.acronym,
                vacc_args['m']
            ))
        elif 'n' in vacc_args:
            res_f.write('{},{},{},{}\n'.format(
                self._seed, 
                r[-1][0], 
                vacc.acronym,
                vacc_args['n']
            ))
        else:
            res_f.write('{},{},{}\n'.format(
                self._seed,
                r[-1][0], 
                vacc.acronym
            ))

        self._seed += 1
