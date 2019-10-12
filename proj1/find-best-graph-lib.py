import graph_tool
import graph_tool.all as gt
from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from networkx.readwrite.gml import write_gml
import numpy as np
import time

if __name__ == '__main__':
    g = barabasi_albert_graph(int(1e4), 1)
    start_t = time.process_time()
    average_shortest_path_length(g)
    print(time.process_time() - start_t)
    write_gml(g, './graph.gml')
    g = gt.load_graph('./graph.gml')
    start_t = time.process_time()
    all_sp = gt.shortest_distance(g)
    vertex_avgs = graph_tool.stats.vertex_average(g, all_sp)
    avg_path = np.sum(vertex_avgs[0]) / (g.num_vertices() - 1)
    print(time.process_time() - start_t)
    start_t = time.process_time()
    sum([sum(i) for i in gt.shortest_distance(g)]) / (g.num_vertices() ** 2 - g.num_vertices()) 
    print(time.process_time() - start_t)