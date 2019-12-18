import sir
import graph_generator as gg 
import graph_vacinator as gv
import matplotlib.pyplot as plt
import graph_modifier as gm
import networkx as nx
import time
import math

# Files are created with the following rules in folder values
#   filename is the name of the method .txt
#   Each column follows:
#       Recovered individuals
#       Size of graph
#       Probability of infection, Beta value
#       Fraction of Vaccinated
#       Number of samples
#       Max infected
#       Apogee

SUSC = 'S'
INF = 'I'
REC = 'R'
VAC = 'V'

network_dict = {
    'ba': lambda n: nx.barabasi_albert_graph(n,2),
    'dms':  gg.create_DMS
}
networks = ['dms']
betas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
#betas1 = [1/8, 1/2, 2.0, 8.0, 32.0]
#betas2 = [1/32, 1/16, 1/4, 1.0, 4.0, 16.0]
Ns = [625] #625, 1250, 2500, 5000, 10000]
samples = [300]#, 10000, 100000]
frac_vacs = [0.1]#[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
vaccination_methods = [
    'hub',
    'realbfs',
    'dfs',
    'bfs',
    'rnd',
    'rdw',
    'acq'
]
gamma = 3

def run_sir(n, m, frac, method, beta, network):
    sims = []
    
    for _ in range(m):
        #start_time = time.time()
        g = network(n)
        #gm.modify_deg_dist(g, 2.5)##
        nx.classes.function.set_node_attributes(g, SUSC, 'state')
        # vaccinate
        g = gv.methods[method](g, frac)
        # TEST
        color_map = []
        for node in g:
            if g.nodes[node]['state'] == VAC:
                color_map.append('green')
            elif g.nodes[node]['state'] == INF:
                color_map.append('red')
            elif g.nodes[node]['state'] == REC:
                color_map.append('yellow')
            else:
                color_map.append('blue')
        nx.draw_spring(
            g,
            edgecolors='black',
            node_color = color_map,
            node_size=100,
            edge_color='#525252'
        )
        plt.show()
        exit()
        #print("\t\t\t\tVACC %2.3f seconds" % (time.time() - start_time))
        # sir
        #start_time = time.time()
        sims.append(sir.sir_simulation(g, beta))
        #print("\t\t\t\tSIR %2.3f seconds" % (time.time() - start_time))

    cum_report = [0,0,0]

    for report in sims:
        cum_report[0] += report[0] # Recovered
        cum_report[1] += report[1] # Max infectious
        cum_report[2] += report[2] # Apogee of infection
        
    #print(cum_infected_frac / m)
    return [cum_report[0]/ m, cum_report[1]/ m, cum_report[2]/ m]

def simple():
    # Method
    for network in networks:
        print('Network:%s' % (network))
        for method in vaccination_methods:
            print('METHOD %s' % (method))
            f = open("values/new_dms/%s.txt" % (method), "w+")
            # N
            for n in Ns:
                print('\tN: %i' % (n))
                # Beta
                for beta in betas:
                    print('\t\tBeta: %f' % (beta))
                    # Fraction Vaccinated
                    for frac_vac in frac_vacs:
                        print('\t\t\tFV: %.2f' % (frac_vac))
                        # Samples
                        for n_samples in samples:
                            print('\t\t\t\tSamples: %i' % (n_samples))
                            # run sir
                            report = run_sir(n, n_samples, frac_vac, method, beta, network_dict[network])
                            # get recovered
                            f.write('%f %i %f %.2f %i %f %f\n' % (report[0], n, beta, frac_vac, n_samples, report[1], report[2]))
            f.close()
                    
            
def compare():
    Ns = [1250, 2500]
    method = 'dfs'
    beta = 16
    for network in ['ba', 'dms']:
        print('\n\nNETWORK: %s' % (network))
        for n in Ns:
            print('\nSIZE: %i' % (n))
            f = open("values/complexity/beta16-%s-%i.txt" % (network, n), 'w+')
            for n_samples in [10, 100, 300, 500]:
                total_time = []
                reports = []
                for k in range(7):
                    start_time = time.time()
                    report = run_sir(n, n_samples, 0.2, method, beta, network_dict[network])
                    finish_time = time.time() - start_time
                    total_time.append(finish_time)
                    reports.append(report)
                    print(report)
                # statistics
                med_sum = reports[0]
                values = [[reports[0][0]], [reports[0][1]], [reports[0][2]]]
                max_val = [reports[0][0],reports[0][1],reports[0][2]]
                min_val = [reports[0][0],reports[0][1],reports[0][2]]
                for k in range(1, len(reports)):
                    # values
                    values[0].append(reports[k][0])
                    values[1].append(reports[k][1])
                    values[2].append(reports[k][2])
                    # median
                    med_sum[0] += reports[k][0]
                    med_sum[1] += reports[k][1]
                    med_sum[2] += reports[k][2]
                    # Recovered
                    if reports[k][0] > max_val[0]:
                        max_val[0] = reports[k][0]
                    if reports[k][0] < min_val[0]:
                        min_val[0] = reports[k][0]
                    # Max_val infectious
                    if reports[k][1] > max_val[1]:
                        max_val[1] = reports[k][1]
                    if reports[k][1] < min_val[1]:
                        min_val[1] = reports[k][1]
                    # Apogee of infection
                    if reports[k][2] > max_val[2]:
                        max_val[2] = reports[k][2]
                    if reports[k][2] < min_val[2]:
                        min_val[2] = reports[k][2]
                # get median
                medians = [
                    med_sum[0] / len(reports),
                    med_sum[1] / len(reports),
                    med_sum[2] / len(reports)
                ]
                # deviation
                deviations = [
                    deviation(values[0], medians[0]),
                    deviation(values[1], medians[1]),
                    deviation(values[2], medians[2])
                ]
                # median_time
                med_time = sum(total_time)/len(reports)
                print('medians:')
                print(medians)
                print('max:')
                print(max_val)
                print('min:')
                print(min_val)
                print('deviation:')
                print(deviations)
                print(med_time)
                # samples time //deviations// recovered max_inf apogee_inf
                print('\noutput:%i %f %f %f %f\n' % (n_samples, med_time, deviations[0], deviations[1], deviations[2]))
                f.write('%i %f %f %f %f\n' % (n_samples, med_time, deviations[0], deviations[1], deviations[2]))
                print('done: %i\n' % (n_samples))

def deviation(values, median):
    sum_x = 0
    for val in values:
        sum_x += (val - median)**2
    return math.sqrt( sum_x / ( len(values) - 1 ) )

def remove_vacs(g):
    nodes_to_remove = []
    edges_to_remove = []
    for node in reversed(list(g.nodes)):
        if g.nodes[node]['state'] == VAC:
            nodes_to_remove.append(node)
    g.remove_nodes_from(nodes_to_remove)
    g = nx.convert_node_labels_to_integers(g)
    return g

#compare()
simple()
