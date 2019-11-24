import networkx as nx
import pathlib as pl

from sir import Sir


class SirHandler(object):
    def __init__(self):
        self._store_path = pl.Path('./simulations/')

    def simulate(self, network_path, beta, f, seed):
        g = nx.read_gml(network_path)
        sir = Sir(g, beta, f, seed)
        res = sir.simulate()
        file_path = self._store_path.joinpath(
            (
                pl.Path(network_path).stem + 
                '_{}_{}_{}.out'.format(
                    sir.beta, sir.f, seed
                ).replace('.', '')
            ), 
        )

        f = open(file_path, 'w')
        f.write('{},{}'.format(res[-1][0], res[-1][1]))
        f.close()
