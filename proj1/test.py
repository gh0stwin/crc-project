from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.generators.classic import dorogovtsev_goltsev_mendes_graph, empty_graph

from networkx.algorithms.cluster import clustering
import matplotlib.pyplot as plt

import math
import random
import numpy

def nx_average_clustering_per_k(g):
    coefficients = [None] * g.number_of_nodes()
    for k in range(len(coefficients)):
        coefficients[k] = [0,0]
    clustering_coefficient = clustering(g) # dict of (vertex, cc)
    all_degrees = g.degree() # list of (vertex, degree)
    for deg in all_degrees:
        #print(coefficients[deg[1]])
        coefficients[deg[1]][0] += 1
        coefficients[deg[1]][1] += clustering_coefficient[deg[0]]
    # print(clustering_coefficient)
    # print(coefficients)
    # average cc
    ck = []
    for coef in coefficients:
        if coef[0] == 0:
            ck.append(0)
        else:
            ck.append(coef[1]/coef[0])
    return ck

def create_DMS(n):
    G = empty_graph(0)
    # add beggining 2 nodes and edges
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1,2)

    # loop
    for k in range(3,n+1):
        edge = random.sample(G.edges(), 1)[0]
        G.add_node(k)
        G.add_edge(k, edge[0])
        G.add_edge(k, edge[1])
    # 
    return G

N = 2000
SAMPLES = 10
clustersBA = []
clustersDMS = []
for n in range(SAMPLES):
    g = barabasi_albert_graph(N, 2)
    cluster1 = nx_average_clustering_per_k(g)
    clustersBA.append(cluster1)
    # g = dorogovtsev_goltsev_mendes_graph(int(math.log(10)))
    # print(g.number_of_nodes())
    # print(nx_average_clustering_per_k(g))

    g = create_DMS(N)
    cluster2 = nx_average_clustering_per_k(g)
    clustersDMS.append(cluster2)
    print(str(n) + ' done')

ba = numpy.asarray(clustersBA)
dms = numpy.asarray(clustersDMS)
degreesBA = []
ba_avg_ck = []
degreesDMS = []
dms_avg_ck = []
for j in range(ba.shape[1]):
    auxBA = []
    auxDMS = []
    for i in range(ba.shape[0]):
        if ba[i,j] != 0:
            auxBA.append(ba[i,j])
        if dms[i,j] != 0:
            auxDMS.append(dms[i,j])
    if len(auxBA) != 0:
        degreesBA.append(j+1)
        ba_avg_ck.append(sum(auxBA)/len(auxBA))
    if len(auxDMS) != 0:
        degreesDMS.append(j+1)
        dms_avg_ck.append(sum(auxDMS)/len(auxDMS))

degreesBA = numpy.asarray(degreesBA)
ba_avg_ck = numpy.asarray(ba_avg_ck)
degreesDMS = numpy.asarray(degreesDMS)
dms_avg_ck = numpy.asarray(dms_avg_ck)
    
# print(ba)
# print(ba.shape)
# print(dms.shape)
# print(ba[:,:] != 0)
# ba = ba[ba[:,:] != 0, :]
# dms = dms[dms[:,:] != 0, :]
# print(ba.shape)
# print(dms.shape)
#ba_avg_ck = numpy.mean(ba, axis=0)
#dms_avg_ck = numpy.mean(dms, axis=0)

#degrees = numpy.arange(0, N)
#degrees = numpy.reshape(degrees, (N,1))
#ba_avg_ck = numpy.reshape(ba_avg_ck, (N,1))
#dms_avg_ck = numpy.reshape(dms_avg_ck, (N,1))

#ba_np = numpy.concatenate((ba_avg_ck, degrees), axis=1)
#dms_np = numpy.concatenate((dms_avg_ck, degrees), axis=1)

#ba_np = ba_np[ba_np[:,0] != 0]
#dms_np = dms_np[dms_np[:,0] != 0]

#ba_avg_ck = [x for x in ba_np[:,0].tolist()]
#dms_avg_ck = [x for x in dms_np[:,0].tolist()]

#degreesBA = ba_np[:,1]
#degreesDMS = dms_np[:,1]

trendBA = numpy.polyfit(numpy.log(degreesBA), numpy.log(ba_avg_ck),1)
trendpolyBA = numpy.poly1d(trendBA) 

plt.loglog(degreesDMS, dms_avg_ck, 'ro', color='blue', markersize=5)
yfit = lambda x: numpy.exp(trendpolyBA(numpy.log(x)))
y = yfit(degreesBA)

X = degreesBA - degreesBA.mean()
Y = y - y.mean()

slope = (X.dot(Y)) / (X.dot(X))
print('slope ' + str(slope))
plt.loglog(degreesBA, y)

print(ba.shape)
# deg = numpy.arange(1,N+1)
# deg = deg.reshape((N,1))
# deg = numpy.repeat(deg,10,axis=1).T 
# ba2 = ba.reshape((10,N,1))
# deg = deg.reshape((10,N,1))
# baDeg = numpy.concatenate((ba2,deg),axis=2)
# # nsample, N, (CK, degree)
# baDeg = baDeg[baDeg[:,:,0] != 0]
x = []
y = []

for j in range(ba.shape[1]):
    for i in range(ba.shape[0]):
        if ba[i,j] != 0:
            x.append(j+1)
            y.append(ba[i,j])

plt.loglog(x, y, 'ro', color='black', markersize=5)
plt.loglog(degreesBA, ba_avg_ck, 'ro', color='red', markersize=5)

plt.show()




