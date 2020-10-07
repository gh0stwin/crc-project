import sir
import graph_generator as gg 
import graph_vacinator as gv
import matplotlib.pyplot as plt
import graph_modifier as gm
import networkx as nx
import time
import math
import json
import pickle
import matplotlib as mpl
from matplotlib.lines import Line2D
mpl.rc('font',family='monospace')

SUSC = 'S'
INF = 'I'
REC = 'R'
VAC = 'V'

network_dict = {
    'ba': lambda n: nx.barabasi_albert_graph(n,2),
    'dms':  gg.create_DMS
}
NETWORK = 'dms'
networks = [NETWORK] #['ba', 'dms']
betas = [2]#[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
Ns = [10000] #625, 1250, 2500, 5000, 10000]
frac_vacs = [0.2]#[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
vaccination_methods = [
    'hub',
#    'realbfs',
#    'dfs',
#    'bfs',
#    'rnd',
#    'rdw',
#    'acq'
]
gamma = 3

def draw_graph(g, my_pos, filename, iter_sim = -1):
    d = dict(g.degree)
    color_map = []
    states = [0,0,0,0] # VAC INF REC SUSC
    for node in g:
        if g.nodes[node]['state'] == VAC:
            color_map.append('green')
            states[0] +=1
        elif g.nodes[node]['state'] == INF:
            color_map.append('red')
            states[1] +=1
        elif g.nodes[node]['state'] == REC:
            color_map.append('yellow')
            states[2] +=1
        else:
            #color_map.append('blue')
            color_map.append('black')
            states[3] +=1
    edge_color_map = []
    for edge in g.edges:
        if ((g.nodes[edge[0]]['state'] == SUSC and g.nodes[edge[1]]['state'] == INF) or
           (g.nodes[edge[1]]['state'] == SUSC and g.nodes[edge[0]]['state'] == INF)):
            edge_color_map.append('red')
        else:
            edge_color_map.append('#525252')
    plt.figure(figsize=(10,10))
    nx.draw(
        g,
        pos = my_pos,
        edgecolors='black',
        node_color = color_map,
        node_size = 200,
        edge_color = edge_color_map,
        #node_size=[v * 50 for v in d.values()]
    )
    legend_els = [
            Line2D(
                [0], 
                [0], 
                marker='o', 
                color='w', 
                label='S (' + str(states[3]) + ')',
                markerfacecolor='blue', 
                markersize=15
            ),
            Line2D(
                [0], 
                [0], 
                marker='o', 
                color='w', 
                label='I (' + str(states[1]) + ')',
                markerfacecolor='red', 
                markersize=15
            ),
            Line2D(
                [0], 
                [0], 
                marker='o', 
                color='w', 
                label='R (' + str(states[2]) + ')',
                markerfacecolor='yellow', 
                markersize=15
            ),
            Line2D(
                [0], 
                [0], 
                marker='o', 
                color='w', 
                label='V (' + str(states[0]) + ')',
                markerfacecolor='green', 
                markersize=15
            ),
        ]
    # plt.gca().legend(
    #         handles=legend_els, 
    #         loc=2, 
    #         prop={'size': 15},
    #         labelspacing=0.90,
    #         title_fontsize=15,
    #         title='Simulation %i' %(iter_sim+1)
    # )
    plt.savefig('figures/%s.pdf' % (filename))
    plt.clf()

# RUN SIR ONCE
def run_sir(n, frac, method, beta, network, network_name):    
    g = network(n)
    my_pos = nx.kamada_kawai_layout(g)#, seed = 100)

    # write
    with open(NETWORK + '.txt', 'wb') as handle:
      pickle.dump(my_pos, handle)
    nx.write_gml(g, NETWORK)
    print('Written')

    # load
    #g = nx.convert_node_labels_to_integers(nx.read_gml(NETWORK))
    #with open(NETWORK + '.txt', 'rb') as handle:
    #    my_pos = pickle.loads(handle.read())
    #print('Loaden')

    nx.classes.function.set_node_attributes(g, SUSC, 'state')
    # vaccinate
    g = gv.methods[method](g, frac)
    # draw after vaccination
    draw_graph(g, my_pos, '%s/%s-before' % (network_name, method))
    print('DRAW VAC')
    # sir
    filename = 0
    for k in range(20):
        print(filename)
        filename = sir.sir_simulation(g, beta, my_pos, draw_graph, k, filename)
        # clean states
        for node in g:
            if g.nodes[node]['state'] == REC:
                g.nodes[node]['state'] = SUSC
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


#draw_all()
ba = nx.barabasi_albert_graph(50,2)
dms = gg.create_DMS(50)
erdos = nx.binomial_graph(50, 0.1)
arr = [ba, dms, erdos]
names = ['ba', 'dms', 'erdos']

for i, net in enumerate(arr):
    my_pos = nx.kamada_kawai_layout(net)
    nx.classes.function.set_node_attributes(net, SUSC, 'state')
    draw_graph(net, my_pos, '%sTHESIS' %(names[i]))


