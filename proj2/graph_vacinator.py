import math
import networkx as nx
import numpy as np
import random as rnd
from operator import itemgetter

VAC = 'V'

def dfs_vaccinated_graph_iter(g, frac_vac):
    stack = []
    nodes = len(g)
    node = rnd.randint(0, nodes - 1)
    vac = 1
    g.nodes[node]['state'] = VAC
    neighbor_iter = 0
    while vac / nodes < frac_vac:
        neighbors = list(g.edges(node))
        # Get next non vaccinated neighbor
        while neighbor_iter < len(neighbors) and g.nodes[neighbors[neighbor_iter][1]]['state'] == VAC:
            neighbor_iter += 1

        # No more neighbors
        if neighbor_iter > len(neighbors) - 1 :
            if len(stack) == 0:
                node = rnd.randint(0, nodes - 1)
                while g.nodes[node]['state'] == VAC:
                    node = rnd.randint(0, nodes - 1)
                neighbor_iter = 0
                g.nodes[node]['state'] = VAC
                vac += 1
                continue
            node, neighbor_iter = stack.pop()
            continue

        # Go to next neighbor
        stack.append([node, neighbor_iter + 1])
        node = neighbors[neighbor_iter][1]
        neighbor_iter = 0
        g.nodes[node]['state'] = VAC
        vac += 1
    return g

def dfs_vaccinated_graph(g, frac_vac):
    '''
    Depth first vaccination of graph g with fraction frac_vac
    '''
    vac = 0
    nodes = len(g)
    while vac / nodes < frac_vac:
        random_node = rnd.randint(0, nodes - 1)
        vac = _dfs_recursion(g, frac_vac, vac, random_node)
    return g


def _dfs_recursion(g, frac_vac, vac, node):
    '''
    Recursion method of Depth First Vaccination.
    '''
    if g.nodes[node]['state'] == VAC:
        return vac

    local_vac = vac + 1
    g.nodes[node]['state'] = VAC
    nodes = len(g)

    for neighbor in g.edges(node):
        if local_vac / nodes >= frac_vac:
            break
        local_vac = _dfs_recursion(g, frac_vac, local_vac, neighbor[1])
    
    return local_vac

def bfs_vaccinated_graph(g, frac_vac):
    '''
    Breadth first search vaccination of graph g with fraction frac_vac
    '''
    vac = 0
    nodes = len(g)
    while vac / nodes < frac_vac:
        node = rnd.randint(0, nodes - 1)

        if g.nodes[node]['state'] == VAC:
            continue

        vac += 1
        g.nodes[node]['state'] = VAC

        for neighbor in g.edges(node):
            if vac / nodes >= frac_vac:
                break
            g.nodes[neighbor[1]]['state'] = VAC
            vac += 1
    return g

def rnd_vaccinated_graph(g, frac_vac):
    '''
    Randomly vacinate graph g given fraction frac_vac
    '''
    for index in rnd.sample(g.nodes, int(len(g) * frac_vac)):
        g.nodes[index]['state'] = VAC

    return g

def rnd_walk_vaccinated_graph(g, frac_vac, m = 0):
    '''
    Random walk vaccination of graph g given fraction frac_vac
    '''
    # default
    if m == 0:
        m = int(len(g) * (frac_vac + 0.1))
    # Get highest degree nodes
    list_highest = [] # list of lists of (node_id, degree)
    max_list_size = int(frac_vac * len(g))
    current_node = rnd.randint(0, len(g) - 1)#random node
    alpha = 3

    for _ in range(m):
        degree_node = g.degree(current_node)
        # add to highest if still not max size
        if len(list_highest) < max_list_size:
            list_highest.append([current_node, degree_node])
        else:
            # add to highest and sort
            list_highest.append([current_node, degree_node])
            list_highest = sorted(list_highest, key=itemgetter(1), reverse=True) # sort by degree

            # remove smallest
            list_highest.pop()

        # get probability for node
        jump_prob = alpha / (degree_node + alpha)
        prob = rnd.random()

        if prob < jump_prob:
            # jump
            current_node = rnd.randint(0, len(g) - 1)#random node
            continue

        # neighbor
        neighbors = list(g.edges(current_node))
        current_node = neighbors[rnd.randint(0, len(neighbors)-1)][1]

    # Vaccinate highest
    for node in list_highest:
        g.nodes[node[0]]['state'] = VAC
    return g

def acq_vaccinated_graph(g, frac_vac):
    '''
    Acquaintance vaccination of graph g given fraction frac_vac
    '''
    vac = 0
    nodes = len(g)
    while vac / nodes < frac_vac: 
        node = rnd.randint(0, nodes - 1)
        unvac = []
        for neighbor in g.edges(node):
            if g.nodes[neighbor[1]]['state'] == VAC:
                continue
            unvac.append(neighbor[1])
        
        unvac_size = len(unvac)
        if unvac_size == 0:
            continue
        node = rnd.randint(0, unvac_size - 1)
        g.nodes[unvac[node]]['state'] = VAC
        vac += 1

    return g

methods = {
    'dfs': dfs_vaccinated_graph_iter,
    'bfs': bfs_vaccinated_graph,
    'rnd': rnd_vaccinated_graph,
    'acq': acq_vaccinated_graph,
    'rdw': rnd_walk_vaccinated_graph
}
