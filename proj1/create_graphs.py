from networkx.generators.classic import empty_graph
import random

def create_DMS(n):
    G = empty_graph(0)
    # add beggining 2 nodes and edges
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1,2)
    G.add_node(3)
    G.add_edge(3,1)
    G.add_edge(3,2)

    edges = [(1,2),(3,1),(3,2)]
    # loop
    for k in range(4,n+1):
        edge = edges[random.randint(0, 2*k-6)]
        G.add_node(k)
        G.add_edge(k, edge[0])
        G.add_edge(k, edge[1])
        edges.append((k,edge[0]))
        edges.append((k,edge[1]))
    # 
    return G