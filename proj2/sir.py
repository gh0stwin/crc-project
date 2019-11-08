import networkx as nx
import random as rnd


def sir_simulation(g, beta, seed=None):
    if seed:
        rnd.seed(seed)
    iters_data = [[len(g) - 1, 1]]
    nx.classes.function.set_node_attributes(g, 'S', 'state')
    infected_nodes, num_s_i_edges = first_infect_event(g)
    while infected_nodes:
        infected_ratio = beta * num_s_i_edges
        if rnd.random() < (
            infected_ratio / (infected_ratio + len(infected_nodes))
        ):
            num_s_i_edges = infect_event(g, infected_nodes)
            iters_data[-1][1] += 1
            iters_data[-1][0] -= 1
        else:
            num_s_i_edges = recover_event(g, infected_nodes)
            iters_data[-1][1] -= 1
        if rnd.random() < 1 / (infected_ratio + len(infected_nodes)):
            iters_data.append([iters_data[-1][0], iters_data[-1][1]])

def neighbour_s_i_edges(g, i_node):
    s_i_edges = []
    for edge in g.edges(i_node):
        if edge[0] != i_node and edge[0]['label'] == 'S':
            s_i_edges.append(edge)
        elif edge[1] != i_node and edge[1]['label'] == 'S':
            s_i_edges.append(edge)

def first_infect_event(g):
    node_to_infect = rnd.randint(0, len(g) - 1)
    g.nodes[node_to_infect]['state'] = 'I'
    s_i_edges = neighbour_s_i_edges(g, node_to_infect)
    return {node_to_infect: s_i_edges}, len(s_i_edges)

def infect_event(g, infected_nodes):
    infected_node = infected_nodes[rnd.choice(infected_nodes.keys)]
    selected_edge_idx = rnd.randint(
        0, 
        len(infected_nodes[infected_node]) - 1
    )
    s_i_edge = infected_nodes[infected_node][selected_edge_idx]
    infected_nodes[infected_node].pop(selected_edge_idx)
    node_to_infect = s_i_edge[0]
    if s_i_edge[1]['state'] == 'S':
        node_to_infect = s_i_edge[1]
    infected_nodes[node_to_infect] = neighbour_s_i_edges(
        g, 
        node_to_infect
    )
    g.nodes[node_to_infect]['state'] = 'I'

def recover_event(g, infected_nodes):
    node_to_recover = rnd.choice(infected_nodes.keys())
    g.nodes[node_to_recover]['state'] = 'R'
    infected_nodes.pop(node_to_recover, None)