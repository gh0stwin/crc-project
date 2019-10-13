import graph_tool.all as gt
import graph_tool.stats as gt_stats
import math

from networkx.algorithms.assortativity import average_degree_connectivity, average_neighbor_degree, degree_pearson_correlation_coefficient, k_nearest_neighbors
from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.cluster import transitivity
from networkx.algorithms.components import connected_component_subgraphs
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from networkx.classes.function import degree_histogram
from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.readwrite.gml import write_gml
import numpy as np
import powerlaw as pl


def assortativity(g):
    return gt.scalar_assortativity(g, 'total')[0]

def avg_degree(g):
    return gt_stats.vertex_average(g, 'total')[0]

def avg_neighbor_corr(g):
    return gt.avg_neighbor_corr(g, 'total', 'total')

def avg_path_length(g):
    return np.sum(
        gt_stats.vertex_average(g, gt.shortest_distance(g))[0]
    ) / (g.num_vertices() - 1)

def degree_ratio_of_giant_comp(g):
    return gt.extract_largest_component(g).num_vertices() / \
        g.num_vertices()

def diameter_approx(g):
    return gt.pseudo_diameter(g)[0]

def gb_clus_coef(g):
    return gt.global_clustering(g)[0]

def lcl_clus_coef(g):
    return gt_stats.vertex_average(g, gt.local_clustering(g))[0]

def deg_powerlaw_low_high_sat(g):
    pl_fit = pl.Fit(
        gt_stats.vertex_hist(g, 'total')[0] / g.num_vertices()
    )
    return pl_fit.xmin, pl_fit.xmax

def cum_deg_powerlaw_low_high_sat(g):
    deg_hist = gt_stats.vertex_hist(g, 'total')[0] / g.num_vertices()
    pl_fit = pl.Fit(
        np.flip(np.flip(deg_hist, 0).cumsum(), 0),
        verbose=False
    )

    return (
        pl_fit.power_law.alpha, 
        pl_fit.power_law.xmin, 
        pl_fit.power_law.xmax
    )

def max_degree(g):
    return gt_stats.vertex_hist(g, 'total')[1][-2]

def variance(g):
    degree_hist = gt_stats.vertex_hist(g, 'total')[0]
    second_m = np.sum(degree_hist * (np.arange(len(degree_hist)) ** 2))
    return math.sqrt(second_m - avg_degree(g) ** 2)

if __name__ == '__main__':
    nx_g = barabasi_albert_graph(int(1e3), 2)
    nx_apl = average_shortest_path_length(nx_g)
    nx_ad = 2 * nx_g.number_of_edges() / nx_g.number_of_nodes()
    nx_gcc = transitivity(nx_g)
    nx_lcc = average_clustering(nx_g)
    nx_md = len(degree_histogram(nx_g)) - 1
    nx_drogc = max(connected_component_subgraphs(nx_g), key=len).number_of_nodes() / nx_g.number_of_nodes()
    second_m = np.sum(np.array(degree_histogram(nx_g)) * (np.arange(len(degree_histogram(nx_g))) ** 2))
    nx_v = math.sqrt(second_m - nx_ad ** 2)
    nx_ap = degree_pearson_correlation_coefficient(nx_g)
    nx_aknn = np.array(average_degree_connectivity(nx_g).values())

    write_gml(nx_g, './graph.gml')
    gt_g = gt.load_graph('./graph.gml')

    gt_apl = avg_path_length(gt_g)
    gt_ad = avg_degree(gt_g)
    gt_gcc = gb_clus_coef(gt_g)
    gt_lcc = lcl_clus_coef(gt_g)
    gt_md = max_degree(gt_g)
    gt_drogc = degree_ratio_of_giant_comp(gt_g)
    gt_v = variance(gt_g)
    gt_ap = assortativity(gt_g)
    gt_aknn = avg_neighbor_corr(gt_g)

    assert math.isclose(nx_apl, gt_apl) == True
    assert math.isclose(nx_ad, gt_ad) == True
    assert math.isclose(nx_gcc, gt_gcc) == True
    assert math.isclose(nx_lcc, gt_lcc) == True
    assert nx_md == gt_md
    assert math.isclose(nx_drogc, gt_drogc) == True
    assert math.isclose(nx_v, gt_v) == True
    print(cum_deg_powerlaw_low_high_sat(gt_g))
    print(nx_aknn)
    print(gt_aknn[0])
    assert math.isclose(nx_ap, gt_ap) == True
    assert np.array_equal(nx_aknn, gt_aknn[0]) == True