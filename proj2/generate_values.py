import sir
import graph_generator as gg 
import graph_vacinator as gv
import graph_modifier as gm
import networkx as nx

# Files are created with the following rules in folder values
#   filename is the name of the method .txt
#   Each column follows:
#       Recovered individuals
#       Size of graph
#       Probability of infection, Beta value
#       Fraction of Vaccinated
#       Number of samples

SUSC = 'S'
INF = 'I'
REC = 'R'
VAC = 'V'

betas = [1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
Ns = [625, 1250, 2500, 5000, 10000]
samples = [10]#, 10000, 100000]
frac_vacs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
vaccination_methods = [
    'dfs',
    'bfs',
    'rnd',
    'rdw',
    'acq'
]
gamma = 2.5

def run_sir(n, m, frac, method, beta):
    sims = []
    
    for _ in range(m):
        g = nx.barabasi_albert_graph(n, 2)
        #gm.modify_deg_dist(g, 2.5)##
        nx.classes.function.set_node_attributes(g, SUSC, 'state')
        g = gv.methods[method](g, frac)
        sims.append(sir.sir_simulation(g, beta))

    cum_infected_frac = 0
    cum_recovered = 0

    for report in sims:
        cum_recovered += report[-1][2]

    # recovered
    return cum_recovered / m

# Method
for method in vaccination_methods:
    print('METHOD %s' % (method))
    f = open("values/%s.txt" % (method), "w+")
    # N
    for n in Ns:
        print('\tN: %i' % (n))
        # Beta
        for beta in betas:
            print('\t\tBeta: %f' % (beta))
            # Fraction Vaccinated
            for frac_vac in frac_vacs:
                print('\t\t\tFV: %.1f' % (frac_vac))
                # Samples
                for n_samples in samples:
                    print('\t\t\t\tSamples: %i' % (n_samples))
                    # run sir
                    recovered = run_sir(n, n_samples, frac_vac, method, beta)
                    # get recovered
                    f.write('%i %i %f %i %i\n' % (recovered, n, beta, frac_vac, n_samples))
    f.close()


