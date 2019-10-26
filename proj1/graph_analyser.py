import gc
import graph_tool.all as gt
import graph_tool.stats as gt_stats
import sys

from gt_metrics import *

P = 0.15

def graph_analyser(min_vertices, max_vertices, step, samples, print_header=False):
    metrics = open('./results/metrics.out', 'a')

    if print_header:
        graph_header(metrics)

    metrics.close()

    for vertices in range(min_vertices, max_vertices, step):
        metrics = open('./results/metrics.out', 'a')
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

            print('dms')
            g_name = 'dms-' + str(vertices) + '-' + str(i)
            compute_all_metrics(
                dms_network(vertices),
                g_name,
                metrics, 
                vert_dep_metrics
            )

        vert_dep_metrics.close()
        metrics.close()

def compute_all_metrics(g, g_name, g_file, v_file):
    g_file.write(g_name)
    graph_metrics(g, g_file)
    # vertices_metrics(g, v_file, g_name)
    g = None
    gc.collect()

def graph_header(f):
    f.write(
        ',Number of Nodes,Number of Edges,Density,Average Degree,' + 
        'Max Degree,Variance,Average Path Length,Pseudo Diameter,' + 
        'Global Clustering Coefficient,Local Clustering Coefficient,' + 
        'Giant Component Ratio,Powerlaw Parameter alpha,' +
        'Powerlaw Parameter xmin,Powerlaw Parameter xmax,' + 
        'Cumulative Powerlaw Parameter alpha,' + 
        'Cumulative Powerlaw Parameter xmin,' + 
        'Cumulative Powerlaw Parameter xmax,Degree Assortativity\n'
    )

def vertices_header(f):
    f.write(
        ',Degree Centrality,Page Rank,Harmonic Centrality,' + 
        'Betweenness Centrality,Average Neighbour Correlation,' + 
        'K Core,Local Culstering Coefficient Distribution\n'
    )

def graph_metrics(g, f):
    # # Number of Nodes
    write_in_csv_file(f, g.num_vertices())

    # # Number of Edges
    write_in_csv_file(f, g.num_edges())

    # # Density
    write_in_csv_file(f, density(g))

    # # Average Degree
    write_in_csv_file(f, avg_degree(g))

    # # Max Degree
    write_in_csv_file(f, max_degree(g))

    # Variance
    write_in_csv_file(f, variance(g))

    # # Average Path Length
    write_in_csv_file(f, avg_path_length(g))

    # # Pseudo Diameter
    write_in_csv_file(f, diameter_approx(g))

    # # Global Clustering Coefficient
    write_in_csv_file(f, gb_clus_coef(g))

    # # Local Clustering Coefficient
    write_in_csv_file(f, lcl_clus_coef(g))

    # # Giant Component Ratio
    write_in_csv_file(f, degree_ratio_of_giant_comp(g))

    # # Powerlaw Parameters
    d = deg_powerlaw_low_high_sat(g)
    write_in_csv_file(f, d[0])
    write_in_csv_file(f, d[1])
    write_in_csv_file(f, d[2])

    # # Cumulative Powerlaw Parameters
    d = cum_deg_powerlaw_low_high_sat(g)
    write_in_csv_file(f, d[0])
    write_in_csv_file(f, d[1])
    write_in_csv_file(f, d[2])

    # # Degree Assortativity
    write_in_csv_file(f, assortativity(g))

    write_in_csv_file(f, '\n')

def vertices_metrics(g, f, first_el):
    f.write(first_el + '-dc,')

    # Degree Centrality
    np.savetxt(f, degree_centrality(g), fmt='%i', delimiter=',', newline=',')
    f.write('\n' + first_el + '-dcn,')

    # # Degree Centrality Normalized
    np.savetxt(f, degree_centrality_normalized(g), newline=',')
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

    # # Degree Distribution
    np.savetxt(f, degree_dist(g), delimiter=',', newline=',')
    f.write('\n' + first_el + '-cdd,')

    # # Cumulative Degree Distribution
    np.savetxt(f, cum_degree_dist(g), delimiter=',', newline=',')
    f.write('\n' + first_el + '-kcore')

    # K-Core
    np.savetxt(f, kcore(g), delimiter=',', newline=',')
    f.write('\n' + first_el + '-lccd,')

    # Local Culstering Coefficient Distribution
    np.savetxt(f, lcl_clus_coef_dist(g), delimiter=',', newline=',')
    f.write('\n')

def write_in_csv_file(f, value, first=False):
    f.write((',' if not first else '') + str(value))

if __name__ == '__main__':
    graph_analyser(
        int(sys.argv[1]), 
        int(sys.argv[2]), 
        int(sys.argv[3]), 
        int(sys.argv[4])
    )