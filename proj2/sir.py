import networkx as nx
import networkx.classes
import networkx.classes.function
import random as rnd
import sys


SUSC = 'S'
INF = 'I'
REC = 'R'

def sir_simulation(g, beta, seed=None):
    if seed:
        rnd.seed(seed)

    iter_data = [[len(g) - 1, 1]]
    nx.classes.function.set_node_attributes(g, SUSC, 'state')
    infected_node_edges, num_s_i_edges = _first_infect_event(g)
    infected_nodes = list(infected_node_edges.keys())
    return _sir_simulation_cycle(
        g, 
        beta, 
        infected_nodes, 
        infected_node_edges, 
        num_s_i_edges, 
        iter_data
    )

def _sir_simulation_cycle(
    g, 
    beta, 
    infected_nodes, 
    infected_node_edges, 
    num_s_i_edges, 
    iter_data
):
    while infected_nodes:
        inf_r = beta * num_s_i_edges

        if rnd.random() < 1 / (inf_r + len(infected_nodes)):
            iter_data.append([iter_data[-1][0], iter_data[-1][1]])

        if rnd.random() < (inf_r / (inf_r + len(infected_nodes))):
            iter_data[-1][1] += 1
            iter_data[-1][0] -= 1
            num_s_i_edges = _infect_event(
                g, 
                infected_nodes, 
                infected_node_edges, 
                num_s_i_edges
            )
        else:
            iter_data[-1][1] -= 1
            num_s_i_edges = _recover_event(
                g, 
                infected_nodes, 
                infected_node_edges, 
                num_s_i_edges
            )
        
    return iter_data

def _neighbour_s_i_edges(g, i_node):
    s_i_edges = []

    for edge in g.edges(i_node):
        if g.nodes[edge[0]]['state'] == SUSC:
            s_i_edges.append(edge[0])
        elif g.nodes[edge[1]]['state'] == SUSC:
            s_i_edges.append(edge[1])

    return s_i_edges
    
def _rm_s_i_edges_of_new_infected(g, i_node, infected_node_edges):
    edges_removed = 0

    for neigh in g.neighbors(i_node):
        if g.nodes[neigh]['state'] != INF:
            continue

        for idx in range(len(infected_node_edges[neigh])):
            if (infected_node_edges[neigh][idx] == i_node):
                infected_node_edges[neigh].pop(idx)
                edges_removed += 1
                break

    return edges_removed

def _first_infect_event(g):
    node_to_infect = rnd.randint(0, len(g) - 1)
    g.nodes[node_to_infect]['state'] = INF
    s_i_edges = _neighbour_s_i_edges(g, node_to_infect)
    return {node_to_infect: s_i_edges}, len(s_i_edges)

def _infect_event(g, infected_nodes, infected_node_edges, n_s_i_edges):
    infected_node, node_to_infect = _select_s_i_edge(
        infected_node_edges, 
        n_s_i_edges
    )

    g.nodes[node_to_infect]['state'] = INF
    infected_nodes.append(node_to_infect)
    new_s_i_edges = _neighbour_s_i_edges(g, node_to_infect)
    infected_node_edges[node_to_infect] = new_s_i_edges
    removed_s_i_edges = _rm_s_i_edges_of_new_infected(
        g, 
        node_to_infect, 
        infected_node_edges
    )

    return n_s_i_edges - removed_s_i_edges + len(new_s_i_edges)

def _select_s_i_edge(infected_node_edges, n_s_i_edges):
    idx = rnd.randint(0, n_s_i_edges - 1)
    edge_count = 0

    for node in infected_node_edges:
        lcl_edge_count = len(infected_node_edges[node])
        
        if lcl_edge_count <= 0:
            continue
            
        if idx >= edge_count + lcl_edge_count:
            edge_count += lcl_edge_count
            continue

        return node, infected_node_edges[node][idx - edge_count]

def _recover_event(g, infected_nodes, infected_node_edges, n_s_i_edges):
    node_to_recover_idx = rnd.randint(0, len(infected_nodes) - 1)
    node_to_recover = infected_nodes.pop(node_to_recover_idx)
    g.nodes[node_to_recover]['state'] = REC
    n_rem_edges = infected_node_edges.pop(node_to_recover, None)
    return n_s_i_edges - len(n_rem_edges)

def _max_infected(report):
    max_val = -1

    for it in report:
        if it[1] > max_val:
            max_val = it[1]

    return max_val

if __name__ == '__main__':
    print(*sys.argv)

    sims = []
    n = int(sys.argv[1])
    m = int(sys.argv[3])
    
    for _ in range(m):
        sims.append(sir_simulation(nx.barabasi_albert_graph(n, 2), float(sys.argv[2])))
    
    cum_infected_frac = 0

    for report in sims:
        cum_infected_frac += (n - _max_infected(report)) / n

    print(cum_infected_frac / m)
