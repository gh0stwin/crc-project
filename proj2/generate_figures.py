import sir
import graph_generator as gg 
import graph_vacinator as gv
import matplotlib.pyplot as plt
import graph_modifier as gm
import networkx as nx
import time
import math

SUSC = 'S'
INF = 'I'
REC = 'R'
VAC = 'V'

network_dict = {
    'ba': lambda n: nx.barabasi_albert_graph(n,2),
    'dms':  gg.create_DMS
}
networks = ['ba'] #['ba', 'dms']
betas = [2]#[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
Ns = [625] #625, 1250, 2500, 5000, 10000]
frac_vacs = [0.2]#[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
vaccination_methods = [
#    'hub',
#    'realbfs',
#    'dfs',
#    'bfs',
    'rnd',
    'rdw',
    'acq'
]
gamma = 3

def draw_graph(g, my_pos, filename):
    d = dict(g.degree)
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
    plt.figure(figsize=(10,10))
    nx.draw(
        g,
        pos = my_pos,
        edgecolors='black',
        node_color = color_map,
        node_size=200,
        edge_color='#525252',
        #node_size=[v * 50 for v in d.values()]
    )
    plt.savefig('figures/%s.png' % (filename))
    plt.clf()

# RUN SIR ONCE
def run_sir(n, frac, method, beta, network, network_name):    
    g = network(n)
    my_pos = nx.kamada_kawai_layout(g)#, seed = 100)
    nx.classes.function.set_node_attributes(g, SUSC, 'state')
    # vaccinate
    g = gv.methods[method](g, frac)
    # draw after vaccination
    draw_graph(g, my_pos, '%s/%s-before' % (network_name, method))
    print('DRAW VAC')
    # sir
    sir.sir_simulation(g, beta, my_pos, draw_graph)
    # draw after sir
    draw_graph(g, my_pos, '%s/%s-after' % (network_name, method))
    print('DRAW SIR')
    exit()

def draw_all():
    for network in networks:
        print('Network:%s' % (network))
        for method in vaccination_methods:
            print('METHOD %s' % (method))
            # N
            for n in Ns:
                print('\tN: %i' % (n))
                # Beta
                for beta in betas:
                    print('\t\tBeta: %f' % (beta))
                    # Fraction Vaccinated
                    for frac_vac in frac_vacs:
                        print('\t\t\tFV: %.2f' % (frac_vac))
                        # run sir
                        report = run_sir(n, frac_vac, method, beta, network_dict[network], network)


draw_all()
