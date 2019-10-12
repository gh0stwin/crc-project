from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.generators.classic import dorogovtsev_goltsev_mendes_graph
import gc


BARABASI_FILE_NAME = '../results/results-barabsi.out'
DGM_FILE_NAME = '../results-dgm.out'

def graph_analyser(
    min_vertices, 
    max_vertices, 
    step, 
    samples, 
    file_names=None
):
    if file_names is None:
        file_names[0] = BARABASI_FILE_NAME
        file_names[1] = DGM_FILE_NAME

    results_barabasi = open(file_names[0], 'w+')
    results_dgm = open(file_names[1], 'w+')

    for vertices in range(min_vertices, max_vertices, step):
        edges = (vertices - 1) * 2 + 1
        
        for _ in range(samples):
            graph_metrics(barabasi_albert_graph(vertices, edges))

            gc.collect()
            graph_metrics(dorogovtsev_goltsev_mendes_graph(vertices))

        gc.collect()

def graph_metrics(g, f):
    # Average Degree
    write_in_csv_file(f, 2 * g.number_of_edges() / g.number_of_nodes())

    # 


def write_in_csv_file(f, value, first=False):
    f.write((',' if not first else '') + str(value))