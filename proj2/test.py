import graph_generator as gg 
import graph_vacinator as gv
import networkx as nx

SUSC = 'S'
 
ba = nx.barabasi_albert_graph(100, 2)
nx.classes.function.set_node_attributes(ba, SUSC, 'state')

print(ba.edges(0))
#vac_ba = gv.dfs_vaccinated_graph(ba, 0.5)
#vac_ba = gv.bfs_vaccinated_graph(ba, 0.5)
#vac_ba = gv.rnd_vaccinated_graph(ba, 0.5)
#vac_ba = gv.acq_vaccinated_graph(ba, 0.5)
vac_ba = gv.rnd_walk_vaccinated_graph(ba, 0.5)
#vac_ba = gv.dfs_vaccinated_graph_iter(ba, 0.5)


for node in vac_ba.nodes:
    print(vac_ba.nodes[node]['state'])
