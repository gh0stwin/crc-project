from networkx.generators.random_graphs import barabasi_albert_graph

import matplotlib.pyplot as plt
import networkx as nx
import math
import numpy as np
import gc
from scipy.optimize import curve_fit
import statistics

import argparse
from create_graphs import create_DMS

def degree_distribution(g):
    deg_hist = [0] * g.number_of_nodes()
    all_degrees = g.degree() # list of (vertex, degree)
    for deg in all_degrees:
        deg_hist[deg[1]] += 1
    # average dist
    deg_dist = np.asarray(deg_hist) / g.number_of_nodes()
    return deg_dist

N = 100000
print('Networks of size ' + str(N)+ '\n')
SAMPLES = 10
distsBA = []
distsDMS = []
for n in range(SAMPLES):
    print(str(n) + ' sample started')
    g = barabasi_albert_graph(N, 2)
    print('\tba created')
    dist1 = degree_distribution(g)
    distsBA.append(dist1)
    print('\tba averaged')
    g = create_DMS(N)
    print('\tdms created')
    dist2 = degree_distribution(g)
    distsDMS.append(dist2)
    print('\tdms averaged')
    gc.collect()
    
ba = np.asarray(distsBA)
dms = np.asarray(distsDMS)

degreesBA = []
ba_avg_dist = []
degreesDMS = []
dms_avg_dist = []
for i in range(ba.shape[0]):
    auxBA = []
    auxDMS = []
    auxDBA = []
    auxDDMS = []
    for j in range(ba.shape[1]):
        if ba[i,j] != 0:
            auxBA.append(ba[i,j])
            auxDBA.append(j)
        if dms[i,j] != 0:
            auxDMS.append(dms[i,j])
            auxDDMS.append(j)
    if len(auxBA) != 0:
        ba_avg_dist.append(auxBA)
        degreesBA.append(auxDBA)
    if len(auxDMS) != 0:
        dms_avg_dist.append(auxDMS)
        degreesDMS.append(auxDDMS)

# Powerlaw 
def f(x, N, a):
    return N*np.power(x,-a)

# Powerlaw of 3
def f2(x, N, a):
    return N*np.power(x,-3.0)

fig, axes = plt.subplots(figsize=(3,3))

# Plot each samples distribution of BA
for k in range(len(degreesBA)):
    if k == len(degreesBA) - 1:
        axes.loglog(degreesBA[k], ba_avg_dist[k], 'o', color='black', markersize=2, label="BA distribution")
    else:
        axes.loglog(degreesBA[k], ba_avg_dist[k], 'o', color='black', markersize=2)

# Plot each samples distribution of DMS
for k in range(len(degreesDMS)):
    if k == len(degreesDMS) - 1:
        axes.loglog(degreesDMS[k], dms_avg_dist[k], 'o', mfc='none', color='black', markersize=2, label="DMS distribution")
    else:
        axes.loglog(degreesDMS[k], dms_avg_dist[k], 'o', mfc='none', color='black', markersize=2)

# get mean of distributions for fitting powerlaw
degreesBA = []
ba_avg_dist = []
degreesDMS = []
dms_avg_dist = []
low_cutoff = 20
high_cutoff = int(N/1000)
# Cutoff values found by hand for N=10^5
for j in range(low_cutoff, high_cutoff):#ba.shape[1]):
    auxBA = []
    auxDMS = []
    for i in range(ba.shape[0]):
        if ba[i,j] != 0:
            auxBA.append(ba[i,j])
        if dms[i,j] != 0:
            auxDMS.append(dms[i,j])
    if len(auxBA) != 0:
        degreesBA.append(j)
        auxBA.sort()
        ba_avg_dist.append(statistics.harmonic_mean(auxBA))
    if len(auxDMS) != 0:
        degreesDMS.append(j)
        auxDMS.sort()
        dms_avg_dist.append(statistics.harmonic_mean(auxDMS))
    
# fit BA powerlaw
popt, pcov = curve_fit(f, degreesBA, ba_avg_dist)
axes.loglog(range(2,110), f(range(2,110), *popt), 'red', linestyle='-', linewidth="1",
    label="BA fit, %1.2f*k^-%1.2f" % (popt[0],popt[1]))
# power of BA
# print(popt[1])

# fit DMS powerlaw
popt, pcov = curve_fit(f, degreesDMS, dms_avg_dist)
axes.loglog(range(2,110), f(range(2,110), *popt), 'red', linestyle='--', linewidth="1", 
    label="DMS fit, %1.2f*k^-%1.2f" % (popt[0],popt[1]))
# power of DMS
# print(popt[1])

# fit expected powerlaw
#popt, pcov = curve_fit(f2, degreesBA, ba_avg_dist)
#plt.loglog(degreesBA, f2(degreesBA, *popt), 'cyan', linewidth="2", label="" + "{:1.2f}".format(popt[0]) + "*k^-3")

plt.xlabel(r"$k$")
plt.ylabel(r"$P_k$")
plt.tight_layout()
plt.show()



