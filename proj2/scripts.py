import networkx as nx
import os
import pathlib as pl

from graph_generator import configuration_model, dms
from sir_handler import SirHandler


def create_networks(
    nodes,
    gen_graph_function,
    gen_graph_args,
    path,
    graphs_per_node=100,
    seed=0
):
    for n in nodes:
        for i in range(graphs_per_node):
            g = gen_graph_function(*((n,) + gen_graph_args + (seed,)))
            file_path = (path + '_' + str(n) + '_' + str(i + 1) + '_' + 
                str(seed) + '.gml'
            )

            nx.write_gml(g, file_path)
            seed += 1

if __name__ == '__main__':
    # create_networks(
    #     [625, 1250, 2500, 5000, 10000],
    #     configuration_model,
    #     (2.5, lambda n, g: int(round(n ** (1 / (g - 1))))),
    #     './networks/configuration_model',
    #     100,
    #     0
    # )

    # create_networks(
    #     [625, 1250, 2500, 5000, 10000],
    #     dms,
    #     tuple(),
    #     './networks/dms',
    #     100,
    #     0
    # )

    # create_networks(
    #     [625, 1250, 2500, 5000, 10000],
    #     nx.barabasi_albert_graph,
    #     (2,),
    #     './networks/ba',
    #     100,
    #     0
    # )

    files = sorted(pl.Path('.').glob('**/*.gml'))
    betas = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    fs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    sh = SirHandler()
    sh.simulate(files, betas, fs, 300000, 0)
