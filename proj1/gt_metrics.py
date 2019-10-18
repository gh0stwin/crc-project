import graph_tool.all as gt
import graph_tool.stats as gt_stats
import math
from networkx.algorithms.assortativity import average_degree_connectivity, average_neighbor_degree, degree_pearson_correlation_coefficient, k_nearest_neighbors
from networkx.algorithms.centrality import degree_centrality as nx_degree_centrality, harmonic_centrality as nx_harmonic_centrality, betweenness_centrality as nx_betweenness_centrality
from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.cluster import transitivity
from networkx.algorithms.components import connected_component_subgraphs
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from networkx.classes.function import degree_histogram
from networkx.generators.random_graphs import barabasi_albert_graph, erdos_renyi_graph
from networkx.readwrite.edgelist import read_edgelist
from networkx.readwrite.gml import write_gml
import numpy as np
import powerlaw as pl
import random as rnd


def assortativity(g):
    return gt.scalar_assortativity(g, 'total')[0]

def avg_degree(g):
    return gt_stats.vertex_average(g, 'total')[0]

def avg_neighbor_corr(g):
    return gt.avg_neighbor_corr(g, 'total', 'total')[0]

def avg_path_length(g):
    return np.sum(
        gt_stats.vertex_average(g, gt.shortest_distance(g))[0]
    ) / (g.num_vertices() - 1)

def betweenness_centrality(g):
    return gt.betweenness(g)[0].get_array()

def cum_degree_dist(g):
    return np.flip(np.flip(degree_dist(g), 0).cumsum(), 0)

def cum_deg_powerlaw_low_high_sat(g):
    deg_hist = gt_stats.vertex_hist(g, 'total')[0] / g.num_vertices()
    pl_fit = pl.Fit(
        np.flip(np.flip(deg_hist, 0).cumsum(), 0),
        verbose=False
    )

    return (
        pl_fit.alpha, 
        pl_fit.xmin, 
        pl_fit.xmax
    )

def degree_centrality(g):
    return g.get_total_degrees(g.get_vertices())

def degree_dist(g):
    deg_array = g.get_total_degrees(g.get_vertices())
    max_val = np.max(deg_array)
    

def degree_centrality_normalized(g):
    return degree_centrality(g) / (g.num_vertices() - 1)

def degree_dist(g):
    return gt_stats.vertex_hist(g, 'total')[0] / g.num_vertices()

def degree_ratio_of_giant_comp(g):
    return gt.extract_largest_component(g).num_vertices() / \
        g.num_vertices()

def deg_powerlaw_low_high_sat(g):
    pl_fit = pl.Fit(
        gt_stats.vertex_hist(g, 'total')[0] / g.num_vertices(),
        verbose=False
    )

    return (
        pl_fit.alpha, 
        pl_fit.xmin, 
        pl_fit.xmax
    )

def density(g):
    return g.num_edges() / (g.num_vertices() ** 2 - g.num_vertices())

def dgm_network(n, seed=None):
    if seed:
        rnd.seed(seed)

    g = gt.Graph(directed=False)
    v1 = g.add_vertex()
    v2 = g.add_vertex()
    v3 = g.add_vertex()
    g.add_edge(v1, v2)
    g.add_edge(v2, v3)
    g.add_edge(v3, v1)
    edges = 2

    for _ in range(n - 3):
        new_v = g.add_vertex()
        e = gt.find_edge(g, g.edge_index, rnd.randint(0, edges))[0]
        g.add_edge(e.source(), new_v)
        g.add_edge(new_v, e.target())
        edges += 1

    return g

def diameter_approx(g):
    return gt.pseudo_diameter(g)[0]

def erdos_renyi_network(n, p, seed=None):
    return nx2gt(erdos_renyi_graph(n, p, seed))

def gb_clus_coef(g):
    return gt.global_clustering(g)[0]

def harmonic_centrality(g):
    return gt.closeness(g, harmonic=True).get_array()

def kcore(g):
    return gt_stats.vertex_hist(g, gt.kcore_decomposition(g))[0]

def lcl_clus_coef(g):
    return gt_stats.vertex_average(g, gt.local_clustering(g))[0]

def lcl_clus_coef_dist(g):
    return gt.local_clustering(g).get_array()

def max_degree(g):
    return gt_stats.vertex_hist(g, 'total')[1][-2]
    
def page_rank(g):
    # return gt_stats.vertex_hist(g, gt.pagerank(g))
    return gt.pagerank(g).get_array()

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
    nx_aknn = np.flip(np.array(
        [it[1] for it in sorted(
            average_degree_connectivity(nx_g).items(), reverse=True
        )]
    ))

    nx_dh = np.array(degree_histogram(nx_g)) / nx_g.number_of_nodes()
    nx_cdh = np.flip(np.flip(
        (np.array(degree_histogram(nx_g)) / nx_g.number_of_nodes())
        , 0
    ).cumsum(),0)

    nx_dc = np.array(
            [it[1] for it in sorted(nx_degree_centrality(nx_g).items())]
    )

    nx_pr = np.array([val for val in pagerank(nx_g).values()])
    nx_hc = np.array(
        [val for val in nx_harmonic_centrality(nx_g).values()]
    ) / nx_g.number_of_nodes()

    nx_bc = np.array(
        [val for val in nx_betweenness_centrality(nx_g).values()]
    )

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
    gt_dh = degree_dist(gt_g)
    gt_cdh = cum_degree_dist(gt_g)
    gt_dc = degree_centrality(gt_g)
    gt_hc = harmonic_centrality(gt_g)
    gt_pr = page_rank(gt_g)
    gt_bc = betweenness_centrality(gt_g)

    assert math.isclose(nx_apl, gt_apl)
    assert math.isclose(nx_ad, gt_ad)
    assert math.isclose(nx_gcc, gt_gcc)
    assert math.isclose(nx_lcc, gt_lcc)
    assert nx_md == gt_md
    assert math.isclose(nx_drogc, gt_drogc)
    assert math.isclose(nx_v, gt_v)
    print(cum_deg_powerlaw_low_high_sat(gt_g))
    assert math.isclose(nx_ap, gt_ap)
    assert np.array_equal(
        nx_aknn, gt_aknn[np.isfinite(gt_aknn)]
    )

    assert np.array_equal(nx_dh, gt_dh)
    assert np.array_equal(nx_cdh, gt_cdh)
    assert np.allclose(nx_dc, gt_dc)
    assert np.allclose(nx_pr, gt_pr, atol=1e-05)
    assert np.allclose(nx_hc, gt_hc, atol=1e-03)
    assert np.allclose(nx_bc, gt_bc)


########################################################################
def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, unicode):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, unicode):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Actually add all the nodes and vertices with their properties
    # Add the nodes
    for _ in range(len(nxG.nodes)):
        gtG.add_vertex()

    # Add the edges
    for src, dst in nxG.edges:
        gtG.add_edge(gtG.vertex(src), gtG.vertex(dst))

    # Done, finally!
    return gtG
########################################################################