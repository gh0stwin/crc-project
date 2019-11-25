import sir
import graph_generator as gg 
import graph_vacinator as gv
import graph_modifier as gm
import networkx as nx
import time

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
networks = ['ba']
#betas = [1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
betas1 = [1/8, 1/2, 2.0, 8.0, 32.0]
betas2 = [1/32, 1/16, 1/4, 1.0, 4.0, 16.0]
Ns = [625, 1250, 2500, 5000, 10000]
samples = [1]#, 10000, 100000]
frac_vacs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
vaccination_methods = [
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
            f = open("values/%s/%s.txt" % (network, method), "w+")
            # N
            for n in Ns:
                print('\tN: %i' % (n))
                # Beta
                for beta in betas1: # PEDRO MUDA AQUI TOTO
                    print('\t\tBeta: %f' % (beta))
                    # Fraction Vaccinated
                    for frac_vac in frac_vacs:
                        print('\t\t\tFV: %.1f' % (frac_vac))
                        # Samples
                        for n_samples in samples:
                            print('\t\t\t\tSamples: %i' % (n_samples))
                            # run sir
                            report = run_sir(n, n_samples, frac_vac, method, beta, network_dict[network])
                            # get recovered
                            f.write('%f %i %f %.1f %i %f %f\n' % (report[0], n, beta, frac_vac, n_samples, report[1], report[2]))
            f.close()
                    
            
def compare():
    n = 2500
    method = 'dfs'
    beta = 32
    for n_samp in [300, 300, 300, 300, 300, 300, 300]:
        start_time = time.time()
        report = run_sir(n, n_samp, 0.2, method, beta, network_dict['dms'])
        print("%2.3f seconds" % (time.time() - start_time))
        print(report)
        print('\n')

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
