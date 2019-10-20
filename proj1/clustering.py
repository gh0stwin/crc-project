from networkx.generators.random_graphs import barabasi_albert_graph

from networkx.algorithms.cluster import clustering
import matplotlib.pyplot as plt
import networkx as nx
import math
import random
import numpy as np
import gc
from scipy.optimize import curve_fit
import statistics
import argparse
from create_graphs import create_DMS

def nx_average_clustering_per_k(g):
    coefficients = [None] * g.number_of_nodes()
    for k in range(len(coefficients)):
        coefficients[k] = [0,0]
    clustering_coefficient = clustering(g) # dict of (vertex, cc)
    all_degrees = g.degree() # list of (vertex, degree)
    for deg in all_degrees:
        coefficients[deg[1]][0] += 1
        coefficients[deg[1]][1] += clustering_coefficient[deg[0]]
    # average cc
    ck = []
    for coef in coefficients:
        if coef[0] == 0:
            ck.append(0)
        else:
            ck.append(coef[1]/coef[0])
    return ck

def main():
    # Deal with arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N', 
        action="store",
        type=int, 
        help='nodes in BA and DMS'
    )
    parser.add_argument(
        'n_samples', 
        action="store",
        nargs='?',
        default=10,
        type=int, 
        help='number of samples to run'
    )
    args = parser.parse_args()

    N = args.N
    SAMPLES = args.n_samples
    clustersBA = []
    clustersDMS = []
    for n in range(SAMPLES):
        print(str(n) + ' sample started')
        g = barabasi_albert_graph(N, 2)
        print('\tba created')
        cluster1 = nx_average_clustering_per_k(g)
        clustersBA.append(cluster1)
        print('\tba averaged')
        g = create_DMS(N)
        print('\tdms created')
        cluster2 = nx_average_clustering_per_k(g)
        clustersDMS.append(cluster2)
        gc.collect()
        print('\tdms averaged')

    ba = np.asarray(clustersBA)
    dms = np.asarray(clustersDMS)

    np.savetxt('./results/clustering_coefficient/'+ str(N) + '-ba.out', ba, delimiter=',')
    np.savetxt('./results/clustering_coefficient/'+ str(N) + '-dms.out', dms, delimiter=',')

if __name__ == '__main__':
    main()
