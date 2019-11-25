import enum
import networkx as nx
import networkx.classes
import networkx.classes.function
import numpy as np
import sys


class SirState(enum.Enum):
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3
    VACCINATED = 4

class Sir(object):
    def __init__(self, g, beta, f=0, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._state = 'state'
        self.g = g
        self.beta = beta
        self.f = f
        self._num_s_i_edges = 0
        self._infected_nodes = []
        self._infected_node_edges = {}
        self._iterations_info = []

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, g):
        self._g = g

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if not 0 <= beta <= 1:
            raise ValueError(
                'beta should be a proportion (0 <= beta <= 1)'
            )
        
        self._beta = beta

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        if not 0 <= f <= 1:
            raise ValueError(
                'f should be a proportion (0 <= f <= 1)'
            )

        self._f = f

    def _initialize_sir_network(self):
        nx.classes.function.set_node_attributes(
            self._g, 
            SirState.SUSCEPTIBLE, 
            self._state
        )

        self._vaccinate_before_sim()
        aux = self._first_infect_event()
        self._infected_node_edges, self._num_s_i_edges = aux
        self._infected_nodes = list(self._infected_node_edges.keys())
        self._iterations_info = [[
            len(self._g) - len(self._infected_nodes), 
            len(self._infected_nodes)
        ]]

    def _first_infect_event(self):
        susc_nodes = [
            node for node, state in nx.get_node_attributes(
                self._g, self._state
            ).items() if state == SirState.SUSCEPTIBLE
        ]

        infected_node_edges, s_i_edges = {}, []

        if len(susc_nodes):
            node_to_inf = susc_nodes[
                np.random.randint(0, len(susc_nodes))
            ]

            self._g.nodes[node_to_inf][self._state] = SirState.INFECTED
            s_i_edges = self._neighbour_s_i_edges(node_to_inf)
            infected_node_edges[node_to_inf] = s_i_edges

        return infected_node_edges, len(s_i_edges)

    def _vaccinate_before_sim(self):
        nodes = list(range(len(self._g)))

        for _ in range(int(round(len(self._g) * self._f))):
            idx = np.random.randint(0, len(nodes))
            node = nodes.pop(idx)
            self._g.nodes[node][self._state] = SirState.VACCINATED

    def simulate(self):
        self._initialize_sir_network()
        
        while self._infected_nodes:
            inf_r = self._beta * self._num_s_i_edges
            time_inc_r = 1 / (inf_r + len(self._infected_nodes))
            prob_to_infect = inf_r / (inf_r + len(self._infected_nodes))
            self._check_new_iteration(time_inc_r)
            self._perform_infection_or_recovery_event(prob_to_infect)

        return self._iterations_info

    def _check_new_iteration(self, time_inc_ratio):
        if np.random.uniform() < time_inc_ratio:
            self._iterations_info.append([
                self._iterations_info[-1][0], 
                self._iterations_info[-1][1]
            ])

    def _perform_infection_or_recovery_event(self, prob_to_infect):
        if np.random.uniform() < prob_to_infect:
            self._iterations_info[-1][1] += 1
            self._iterations_info[-1][0] -= 1
            self._infect_event()
        else:
            self._iterations_info[-1][1] -= 1
            self._recover_event()

    def _infect_event(self):
        infected_node, node_to_infect = self._select_s_i_edge()
        self._g.nodes[node_to_infect][self._state] = SirState.INFECTED
        self._infected_nodes.append(node_to_infect)
        new_s_i_edges = self._neighbour_s_i_edges(node_to_infect)
        self._infected_node_edges[node_to_infect] = new_s_i_edges
        removed_s_i_edges = self._rm_s_i_edges_of_new_infected(
            node_to_infect
        )

        self._num_s_i_edges += (-removed_s_i_edges + len(new_s_i_edges))

    def _recover_event(self):
        rec_node_idx = np.random.randint(0, len(self._infected_nodes))
        node_to_recover = self._infected_nodes.pop(rec_node_idx)
        self._g.nodes[node_to_recover][self._state] = SirState.RECOVERED
        rem_edges = self._infected_node_edges.pop(node_to_recover, None)
        self._num_s_i_edges -= len(rem_edges)

    def _rm_s_i_edges_of_new_infected(self, inf_node):
        edges_removed = 0

        for neigh in self._g.neighbors(inf_node):
            if self._g.nodes[neigh][self._state] != SirState.INFECTED:
                continue

            inf_node_links = self._infected_node_edges[neigh]

            for idx in range(len(inf_node_links) - 1, -1, -1): 
                if (inf_node_links[idx] == inf_node):
                    inf_node_links.pop(idx)
                    edges_removed += 1
                    break

        return edges_removed

    def _select_s_i_edge(self):
        i = np.random.randint(0, self._num_s_i_edges)
        edge_count = 0

        for node in self._infected_node_edges:
            lcl_edge_count = len(self._infected_node_edges[node])
            
            if lcl_edge_count <= 0:
                continue
                
            if i >= edge_count + lcl_edge_count:
                edge_count += lcl_edge_count
                continue

            return node, self._infected_node_edges[node][i - edge_count]

    def _neighbour_s_i_edges(self, node):
        s_i_edges = []
        state = self._state

        for edge in self._g.edges(node):
            if self._g.nodes[edge[0]][state] == SirState.SUSCEPTIBLE:
                s_i_edges.append(edge[0])
            elif self._g.nodes[edge[1]][state] == SirState.SUSCEPTIBLE:
                s_i_edges.append(edge[1])

        return s_i_edges
