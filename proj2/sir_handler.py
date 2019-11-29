import networkx as nx
import pathlib as pl
import sys
import time

from sir import Sir


class SirHandler(object):
    def __init__(self, path='./results/'):
        self._store_path = pl.Path(path)

    def simulate(self, files, betas, fs, times_per_beta_f, seed=0):
        for file in files:
            g = nx.convert_node_labels_to_integers(nx.read_gml(file))

            for beta in betas:
                for f in fs:
                    self._simulate_for_net_beta_f(
                        self._get_res_filename(file, beta, f),
                        g, 
                        beta, 
                        f, 
                        times_per_beta_f, 
                        seed
                    )

    def _get_res_filename(self, file, beta, f):
        return self._store_path.joinpath(
            (
                pl.Path(file).stem.rsplit('_', 1)[0] + 
                '_{}_{}'.format(beta, f).replace('.', '') + '.out'
            )
        )

    def _simulate_for_net_beta_f(self, res_nm, g, beta, f, iters, seed):
        res_f = open(res_nm, 'w')

        for i in range(iters):
            sys.stdout.write('\rnetwork: {}, it: {}'.format(res_nm, i))
            sys.stdout.flush()
            r = Sir(g.copy(), beta, f, seed).simulate()
            res_f.write('{},{}\n'.format(r[-1][0], seed))
            seed += 1
        
        res_f.close()
