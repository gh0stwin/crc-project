import networkx as nx
import pathlib as pl

from sir import Sir


class SirHandler(object):
    def __init__(self):
        self._store_path = pl.Path('./simulations/')

    def simulate(self, net_path, betas, fs, times_per_beta_f, seed=0):
        g = nx.read_gml(net_path)

        for beta in betas:
            for f in fs:
                for i in range(times_per_beta_f):
                    net_props = net_path.split('/')[-1].split('_')[0:3]
                    print(
                        (
                            'network: {}, nodes: {}, index: {}, '
                            'beta: {}, f: {}, seed: {}'.format(
                                *net_props, beta, f, seed 
                            )
                        ),
                        end='\r'
                    )
                    res = Sir(g.copy(), beta, f, seed).simulate()
                    write_in_file(res, net_path, beta, f, seed)
                    seed += 1

    def write_in_file(self, result, net_path, beta, f, seed):
        file_path = self._store_path.joinpath(
            (
                pl.Path(net_path).stem + 
                '_{}_{}_{}.out'.format(beta, f, seed).replace('.', '')
            ), 
        )

        f = open(file_path, 'w')
        f.write('{},{}'.format(result[-1][0], result[-1][1]))
        f.close()
