import networkx as nx
import pathlib as pl

from sir import Sir


class SirHandler(object):
    def __init__(self):
        self._store_path = pl.Path('./simulations/')

    def simulate(self, files, betas, fs, times_per_beta_f, seed=0):
        for file in files:
            g = nx.read_gml(file)

            for beta in betas:
                for f in fs:
                    self._simulate_for_net_beta_f(
                        self._create_res_filename(file, beta, f),
                        g, 
                        beta, 
                        f, 
                        times_per_beta_f, 
                        seed
                    )

    def _create_res_filename(self, file, beta, f):
        return self._store_path.joinpath(
            (
                pl.Path(file).stem.rsplit('_', 1)[0] + 
                '_{}_{}'.format(beta, f).replace('.', '')
            ), 
            '.out'
        )

    def _simulate_for_net_beta_f(self, res_nm, g, beta, f, iters, seed):
        res_f = open(res_nm, 'w')

        for i in range(iters):
            print('network: {}, iter: {}'.format(res_f, i), end='\r')
            r = Sir(g.copy(), beta, f, seed).simulate()
            res_f.write('{},{},{}\n'.format(seed, r[-1][0], r[-1][0]))
            seed += 1
        
        res_f.close()

    def _write_in_file(self, result, net_path, beta, f, seed):
        file_path = self._store_path.joinpath(
            (
                pl.Path(net_path).stem.rsplit('_', 1)[0] + 
                '_{}_{}'.format(beta, f).replace('.', '')
            ), 
            '.out'
        )

        f = open(file_path, 'w')
        f.write('{},{},{}'.format(seed, result[-1][0], result[-1][1]))
        f.close()

if __name__ == '__main__':
    sh = SirHandler()
    sh.simulate('./networks/ba_625_1_0.gml', 0.03125, 0.1, 300000)
