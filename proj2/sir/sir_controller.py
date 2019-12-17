import networkx as nx
import pathlib as pl
import sys
import time

from sir.simulators.acquaintance_simulator import AcquaintanceSimulator
from sir.simulators.bfs_simulator import BfsSimulator
from sir.simulators.collective_influence_simulator import CollectiveInfluenceSimulator
from sir.simulators.coreness_simulator import CorenessSimulator
from sir.simulators.degree_simulator import DegreeSimulator
from sir.simulators.dfs_simulator import DfsSimulator
from sir.simulators.no_vaccination_simulator import NoVaccinationSimulator
from sir.simulators.random_simulator import RandomSimulator
from sir.simulators.random_walk_simulator import RandomWalkSimulator
from sir.simulators.two_step_heuristic_simulator import TwoStepHeuristicSimulator
from sir.sir import Sir


class SirController(object):
    def __init__(
        self, 
        path='./results/', 
        simulators=[
            RandomSimulator(),
            BfsSimulator(),
            DfsSimulator(),
            AcquaintanceSimulator(),
            RandomWalkSimulator(),
            TwoStepHeuristicSimulator(),
            DegreeSimulator(),
            CorenessSimulator(),
            # CollectiveInfluenceSimulator()
        ]
    ):
        self._store_path = pl.Path(path)
        self._simulators = simulators

    def simulate(self, files, betas, fs, times_per_beta_f, seed=0):
        self._seed = seed

        for file in files:
            g = nx.convert_node_labels_to_integers(nx.read_gml(file))

            for beta in betas:
                self._simulate_no_vacc(
                    g, 
                    beta, 
                    self._get_res_filename(file, beta, 0), 
                    times_per_beta_f
                )

                for f in fs:
                    if f == 0:
                        continue

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
        for simulator in self._simulators:
            sys.stdout.write('\rnetwork: {}, it: {}'.format(res_nm, i))
            sys.stdout.flush()
            self._seed = simulator.simulate_sir(
                res_f,
                g, 
                beta, 
                f, 
                self._seed
            )

            self._seed += 1

    def _simulate_no_vacc(self, g, beta, res_nm, iters):
        res_f = open(res_nm, 'a')

        for i in range(iters):
            sys.stdout.write('\rnetwork: {}, it: {}'.format(res_nm, i))
            sys.stdout.flush()
            self._seed = NoVaccinationSimulator().simulate_sir(
                res_f,
                g, 
                beta, 
                0, 
                self._seed
            )

            self._seed += 1
