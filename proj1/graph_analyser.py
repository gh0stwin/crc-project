import gc
import graph_tool.all as gt
import graph_tool.stats as gt_stats
from scipy.stats import poisson

from gt_metrics import *

P = 0.15

def graph_analyser(min_vertices, max_vertices, step, samples):
    metrics = open('./results/metrics.out', 'a')
    graph_header(metrics)

    for vertices in range(min_vertices, max_vertices, step):
        vert_dep_metrics = open(
            './results/' + str(vertices) + '-vertices-metrics.out', 
            'a'
        )

        print('for ' + str(vertices) + ':')
        vertices_header(vert_dep_metrics)

        for i in range(samples):
            print('sample: ' + str(i))
            # print('er')
            # g_name = 'er-' + str(vertices) + '-' + str(i)
            # compute_all_metrics(
            #     erdos_renyi_network(vertices, P),
            #     g_name,
            #     metrics, 
            #     vert_dep_metrics
            # )

            print('ba')
            g_name = 'ba-' + str(vertices) + '-' + str(i)
            compute_all_metrics(
                gt.price_network(vertices, 2, directed=False), 
                g_name,
                metrics, 
                vert_dep_metrics
            )

            print('dgm')
            g_name = 'dgm-' + str(vertices) + '-' + str(i)
            compute_all_metrics(
                dgm_network(vertices),
                g_name,
                metrics, 
                vert_dep_metrics
            )

        vert_dep_metrics.close()

    metrics.close()

def compute_all_metrics(g, g_name, g_file, v_file):
    g_file.write(g_name)
    graph_metrics(g, g_file)
    vertices_metrics(g, v_file, g_name)
    g = None
    gc.collect()

def graph_header(f):
    f.write(
        'Average Degree,Max Degree,Variance,Average Path Length,' + 
        'Pseudo Diameter,Global Clustering Coefficient,' + 
        'Local Clustering Coefficient,Giant Component Ratio,' + 
        'Powerlaw Parameter alpha,Powerlaw Parameter xmin,' +
        'Powerlaw Parameter xmax,Cumulative Powerlaw Parameter alpha,' +
        'Cumulative Powerlaw Parameter xmin,' + 
        'Cumulative Powerlaw Parameter xmax,Degree Assortativity,' + 
        'Average Neighbour Correlation\n'
    )

def vertices_header(f):
    f.write(
        'Degree Centrality,Page Rank,Harmonic Centrality,' + 
        'Betweenness Centrality,Degree Distribution,' + 
        'Cumulative Degree Distribution\n'
    )

def graph_metrics(g, f):
    # Average Degree
    write_in_csv_file(f, avg_degree(g))

    # Max Degree
    write_in_csv_file(f, max_degree(g))

    # Variance
    write_in_csv_file(f, variance(g))

    # Average Path Length
    write_in_csv_file(f, avg_path_length(g))

    # Pseudo Diameter
    write_in_csv_file(f, diameter_approx(g))

    # Global Clustering Coefficient
    write_in_csv_file(f, gb_clus_coef(g))

    # Local Clustering Coefficient
    write_in_csv_file(f, lcl_clus_coef(g))

    # Giant Component Ratio
    write_in_csv_file(f, degree_ratio_of_giant_comp(g))

    # Powerlaw Parameters
    d = deg_powerlaw_low_high_sat(g)
    write_in_csv_file(f, d[0])
    write_in_csv_file(f, d[1])
    write_in_csv_file(f, d[2])

    # Cumulative Powerlaw Parameters
    d = cum_deg_powerlaw_low_high_sat(g)
    write_in_csv_file(f, d[0])
    write_in_csv_file(f, d[1])
    write_in_csv_file(f, d[2])

    # Degree Assortativity
    write_in_csv_file(f, assortativity(g))

    write_in_csv_file(f, '\n')

def vertices_metrics(g, f, first_el):
    # Degree Centrality
    f.write(first_el + '-dc,')
    np.savetxt(f, degree_centrality(g), newline=',')
    f.write('\n' + first_el + '-pr,')

    # Page Rank
    np.savetxt(f, page_rank(g), delimiter=',', newline=',')
    f.write('\n' + first_el + '-hc,')

    # Harmonic Centrality 
    np.savetxt(f, harmonic_centrality(g), delimiter=',', newline=',')
    f.write('\n' + first_el + '-bc,')

    # Betweenness Centrality
    np.savetxt(f, betweenness_centrality(g), delimiter=',', newline=',')
    f.write('\n' + first_el + '-anc,')

    # Average Neighbour Correlation
    np.savetxt(f, avg_neighbor_corr(g), delimiter=',', newline=',')
    f.write('\n' + first_el + '-dd,')

    # Degree Distribution
    np.savetxt(f, degree_dist(g), delimiter=',', newline=',')
    f.write('\n' + first_el + '-cdd,')

    # Cumulative Degree Distribution
    np.savetxt(f, cum_degree_dist(g), delimiter=',', newline=',')
    f.write('\n')

def write_in_csv_file(f, value, first=False):
    f.write((',' if not first else '') + str(value))

if __name__ == '__main__':
    graph_analyser(100, 1001, 100, 40)