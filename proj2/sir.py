import networkx as nx
import networkx.classes
import networkx.classes.function
import random as rnd
import sys
from graph_vacinator import methods

SUSC = 'S'
INF = 'I'
REC = 'R'
VAC = 'V'

def sir_simulation(g, beta, my_pos, draw, seed=None):
    if seed:
        rnd.seed(seed)

    infected_node_edges, num_s_i_edges = _first_infect_event(g)
    infected_nodes = list(infected_node_edges.keys())
    iter_data = [
        [_count_type_nodes(g, SUSC), 1, 0, _count_type_nodes(g, VAC)],
        [_count_type_nodes(g, SUSC), 1, 0, _count_type_nodes(g, VAC)]
    ]
    return _sir_simulation_cycle(
        g, 
        beta, 
        infected_nodes, 
        infected_node_edges, 
        num_s_i_edges, 
        iter_data,
        my_pos,
        draw
    )

def _sir_simulation_cycle(
    g, 
    beta, 
    infected_nodes, 
    infected_node_edges, 
    num_s_i_edges, 
    iter_data,
    my_pos,
    draw
):
    k = 0
    inf = 0
    rec = 0
    inf_on = False
    changed = 0 
    while infected_nodes:
        inf_r = beta * num_s_i_edges
        #draw(g, my_pos, 'test/%s' % (k))
        k += 1
        # if new cycle, record data
        if rnd.random() < 1 / (inf_r + len(infected_nodes)):
            # SUSC INF REC VAC
            iter_data.append([iter_data[-1][0], iter_data[-1][1], iter_data[-1][2], iter_data[-1][3]])

        # if infect event, infect one susceptible
        if rnd.random() < (inf_r / (inf_r + len(infected_nodes))):
            if not inf_on:
                print('REC %i' % (rec))
                inf_on = True
                rec = 0
                changed += 1
                draw(g, my_pos, 'test/%s' % (changed))
            inf += 1
            iter_data[-1][1] += 1 # Add to INF count
            iter_data[-1][0] -= 1 # Reduce SUSC count
            num_s_i_edges = _infect_event(
                g, 
                infected_nodes, 
                infected_node_edges, 
                num_s_i_edges
            )
        else:
            if inf_on:
                print('INF %i' % (inf))
                inf_on = False
                inf = 0
                changed += 1
                draw(g, my_pos, 'test/%s' % (changed))
            rec += 1
            iter_data[-1][1] -= 1 # Reduce INF count
            iter_data[-1][2] += 1 # Add to REC count
            num_s_i_edges = _recover_event(
                g, 
                infected_nodes, 
                infected_node_edges, 
                num_s_i_edges
            )
    print('CHANGED %i' % (changed))
    return sim_report(iter_data)

def _neighbour_s_i_edges(g, i_node):
    s_i_edges = []

    for edge in g.edges(i_node):
        #if g.nodes[edge[0]]['state'] == SUSC:
        #    s_i_edges.append(edge[0])
        if g.nodes[edge[1]]['state'] == SUSC:
            s_i_edges.append(edge[1])

    return s_i_edges
    
def _rm_s_i_edges_of_new_infected(g, i_node, infected_node_edges):
    '''
    Remove edges between susceptible and infected nodes for the new infected node.
    '''
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
    '''
    Infect patient zero and get his edges.
    '''
    while True:
        node_to_infect = rnd.randint(0, len(g) - 1)
        if g.nodes[node_to_infect]['state'] != VAC:
            break
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

def _get_vaccinated_graph(g, frac_vac, method):
    '''
    Get vaccinated graph by using the requested method with the given 
    graph g and fraction frac_vac
    '''
    return methods[method](g, frac_vac)
    
def _count_type_nodes(g, type_node):
    '''
    Count susceptible nodes.
    '''
    n_s = 0
    for node in g.nodes:
        if g.nodes[node]['state'] == type_node:
            n_s += 1
    return n_s

def sim_report(iter_data):
    '''
    '''
    max_inf = -1
    max_k = 0
    k = 0
    for data in iter_data:
        # SUSC INF REC VAC
        if data[1] > max_inf:
            max_inf = data[1]
            max_k = k
        k+=1
    return [iter_data[-1][2], max_inf, max_k]

if __name__ == '__main__':
    '''
    Arguments:
        1: Number of nodes per graph
        2: Beta (Probability of Infection)
        3: Number of samples to run
        4: Vaccinated fraction
        5: Vaccination method
    '''
    print(*sys.argv)

    sims = []
    n = int(sys.argv[1])
    m = int(sys.argv[3])
    frac = float(sys.argv[4])
    method = sys.argv[5]
    
    for _ in range(m):
        g = nx.barabasi_albert_graph(n, 2)
        nx.classes.function.set_node_attributes(g, SUSC, 'state')
        g = _get_vaccinated_graph(g, frac, method)
        sims.append(sir_simulation(g, float(sys.argv[2])))

    cum_report = [0,0,0]

    for report in sims:
        cum_report[0] += report[0] # Recovered
        cum_report[1] += report[1] # Max infectious
        cum_report[2] += report[2] # Apogee of infection
        
    #print(cum_infected_frac / m)
    print(str(cum_report[0]/ m))
    print(str(cum_report[1]/ m))
    print(str(cum_report[2]/ m))

