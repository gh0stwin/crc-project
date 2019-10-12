import graph_tool.all as gt
import graph_tool.stats as gt_stats
import math
from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.cluster import transitivity
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from networkx.readwrite.gml import write_gml
import numpy as np

def avg_degree(g):
    return gt_stats.vertex_average(g, 'total')[0]

def max_degree(g):
    return len(gt_stats.vertex_hist(g, 'total')[0])

def avg_path_length(g):
    return np.sum(
        gt_stats.vertex_average(g, gt.shortest_distance(g))[0]
    ) / (g.num_vertices() - 1)

def gb_clus_coef(g):
    return gt.global_clustering(g)[0]

def lcl_clus_coef(g):
    return np.sum(
        gt_stats.vertex_average(g, gt.local_clustering(g))
    ) / g.num_vertices()

def diameter_approx(g):
    return gt.pseudo_diameter(g)[0]

if __name__ == '__main__':
    nx_g = barabasi_albert_graph(int(1e3), 1)
    nx_apl = average_shortest_path_length(nx_g)
    nx_ad = 2 * nx_g.number_of_edges() / nx_g.number_of_nodes()
    nx_gcc = transitivity(nx_g)
    nx_lcc = average_clustering(nx_g)

    write_gml(nx_g, './graph.gml')
    gt_g = gt.load_graph('./graph.gml')

    gt_apl = avg_path_length(gt_g)
    gt_ad = avg_degree(gt_g)
    gt_gcc = gb_clus_coef(gt_g)
    gt_lcc = lcl_clus_coef(gt_g)

    assert math.isclose(nx_apl, gt_apl) == True
    assert math.isclose(nx_ad, gt_ad) == True
    assert math.isclose(nx_gcc, gt_gcc) == True
    assert math.isclose(nx_lcc, gt_lcc) == True