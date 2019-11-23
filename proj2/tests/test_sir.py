import networkx as nx
import numpy as np
import unittest
import unittest.mock as mock

from sir import Sir, SirState

class TestSir(unittest.TestCase):
    def setUp(self):
        self._state = 'state'
        self._create_graph()
        self._beta = 0.5
        self._f = 0.2
        self._f_to_int = 1
        self._seed = 1
        self._sir = Sir(self._g, self._beta, self._f)

    def test_sir_constructor(self):
        with mock.patch('sir.np.random.seed') as mock_seed:
            sir = Sir(self._g, self._beta, self._f, self._seed)
            mock_seed.assert_called_once_with(self._seed)
            self._assert_public_atts(sir, self._g, self._beta, self._f)

        with mock.patch('sir.np.random.seed') as mock_seed:
            sir = Sir(None, 1)
            mock_seed.assert_not_called()
            self._assert_public_atts(sir, None, 1, 0)

    def test_sir_g_property(self):
        self.assertEqual(self._sir.g, self._g)
        self._sir.g = None
        self.assertEqual(self._sir.g, None)
        self._sir.g = g = object()
        self.assertEqual(self._sir.g, g)

    def test_sir_beta_property(self):
        self.assertEqual(self._sir.beta, self._beta)
        self._sir.beta = 0
        self.assertEqual(self._sir.beta, 0)
        self._sir.beta = 1e0
        self.assertEqual(self._sir.beta, 1)
        self._sir.beta = 0.0001
        self.assertEqual(self._sir.beta, 0.0001)
        self._sir.beta = 0.999
        self.assertEqual(self._sir.beta, 0.999)

    def test_sir_beta_property_exception(self):
        with self.assertRaises(ValueError):
            self._sir.beta = -.1

        with self.assertRaises(ValueError):
            self._sir.beta = -2

        with self.assertRaises(ValueError):
            self._sir.beta = 1.1

    def test_sir_f_property(self):
        self.assertEqual(self._sir.f, self._f)
        self._sir.f = 0
        self.assertEqual(self._sir.f, 0)
        self._sir.f = 1e0
        self.assertEqual(self._sir.f, 1)
        self._sir.f = 0.0012
        self.assertEqual(self._sir.f, 0.0012)
        self._sir.f = 0.998
        self.assertEqual(self._sir.f, 0.998)

    def test_sir_f_property_exception(self):
        with self.assertRaises(ValueError):
            self._sir.f = -.001

        with self.assertRaises(ValueError):
            self._sir.f = -22

        with self.assertRaises(ValueError):
            self._sir.f = 1.00011

    def test_sir_initialize_sir_network(self):
        with mock.patch('sir.np.random.randint') as mock_rnd:
            randint_returns = [0, 0]
            mock_rnd.side_effect = list(randint_returns)
            self._sir._initialize_sir_network()
            s, i, r, v = self._assert_state_and_return_values()
            self.assertEqual(s + i + r + v, len(self._sir.g))
            self.assertEqual(i, 1)
            self.assertEqual(v, 1)
            self.assertEqual(r, 0)
            self.assertEqual(s, len(self._sir.g) - i - v)
            self.assertEqual(
                mock_rnd.mock_calls, 
                [
                    mock.call(0, len(self._sir.g)), 
                    mock.call(0, len(self._sir.g) - 1) 
                ]
            )

            self.assertEqual(self._sir._iterations_info, [])
            self.assertEqual(self._sir._num_s_i_edges, 1)
            self.assertEqual(self._sir._infected_nodes, [1])
            self.assertEqual(
                self._sir._infected_node_edges, 
                {1: [5]}
            )

        self._sir.f = 1

        with mock.patch('sir.np.random.randint') as mock_rnd:
            randint_returns = [0] * len(self._sir.g)
            mock_rnd.side_effect = list(randint_returns)
            self._sir._initialize_sir_network()
            s, i, r, v = self._assert_state_and_return_values()
            self.assertEqual(s, 0)
            self.assertEqual(i, 0)
            self.assertEqual(r, 0)
            self.assertEqual(v, len(self._sir.g))
            self.assertEqual(
                mock_rnd.mock_calls, 
                [mock.call(0, len(self._sir.g) - i) for i in range(
                    len(randint_returns)
                )]
            )
            
            self.assertEqual(self._sir._iterations_info, [])
            self.assertEqual(self._sir._num_s_i_edges, 0)
            self.assertEqual(self._sir._infected_nodes, [])
            self.assertEqual(self._sir._infected_node_edges, {})

    def test_sir_simulate(self):
        event_manager = self._prepare_simulation_case(1)

        with mock.patch('sir.np.random') as mock_random:
            uniform_returns = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
            randint_returns = [0, 0, 0, 1, 0]
            mock_random.uniform.side_effect = list(uniform_returns)
            mock_random.randint.side_effect = list(randint_returns)
            self._sir.simulate()
            self._sir._initialize_sir_network.assert_called_once_with()
            self.assertEqual(
                self._sir._check_new_iteration.call_count,
                len(uniform_returns) // 2
            )

            self.assertEqual(
                event_manager.mock_calls,
                [
                    mock.call.inf(), 
                    mock.call.rec(), 
                    mock.call.inf(), 
                    mock.call.rec(), 
                    mock.call.rec()
                ]
            )

            self.assertEqual(len(self._sir._iterations_info), 2)

        event_manager = self._prepare_simulation_case(2)
   
        with mock.patch('sir.np.random') as mock_random:
            uniform_returns = [1, 1, 1, 0, 1, 0, 1, 0]
            randint_returns = [0, 0, 0, 0]
            mock_random.uniform.side_effect = list(uniform_returns)
            mock_random.randint.side_effect = list(randint_returns)
            self._sir.simulate()
            self._sir._initialize_sir_network.assert_called_once_with()
            self.assertEqual(
                self._sir._check_new_iteration.call_count, 
                len(uniform_returns) // 2
            )

            self.assertEqual(
                event_manager.mock_calls,
                [
                    mock.call.rec(), 
                    mock.call.inf(), 
                    mock.call.rec(), 
                    mock.call.rec()
                ]
            )

            self.assertEqual(len(self._sir._iterations_info), 1)

    def test_sir_infect_event(self):
        pass

    def _assert_state_and_return_values(self):
        susc = 0
        inf = 0
        rec = 0
        vacc = 0
        g = self._sir.g
        
        for node in g:
            self.assertIn(self._state, g.nodes[node])

            if g.nodes[node][self._state] == SirState.SUSCEPTIBLE:
                susc += 1
            elif g.nodes[node][self._state] == SirState.INFECTED:
                inf += 1
            elif g.nodes[node][self._state] == SirState.RECOVERED:
                rec += 1
            elif g.nodes[node][self._state] == SirState.VACCINATED:
                vacc += 1

        return susc, inf, rec, vacc

    def _assert_public_atts(self, sir, g, beta, f):
        self.assertEqual(sir.g, g)
        self.assertEqual(sir.beta, beta)
        self.assertEqual(sir.f, f)

    def _create_graph(self):
        self._g = nx.Graph()
        self._g.add_nodes_from(range(6))
        self._g.add_edges_from(
            [(0, 1), (0, 2), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5)]
        )

    def _prepare_simulation_case(self, case):
        self._sir._initialize_sir_network = mock.Mock()
        self._sir._check_new_iteration = mock.Mock(
            wraps=self._sir._check_new_iteration
        )

        self._sir._infect_event = mock.Mock(
            wraps=self._sir._infect_event
        )

        self._sir._recover_event = mock.Mock(
            wraps=self._sir._recover_event
        )

        self._sir._iterations_info = [[1, 1]]
        nx.classes.function.set_node_attributes(
            self._sir.g, 
            SirState.SUSCEPTIBLE, 
            self._state
        )

        event_manager = mock.Mock()
        event_manager.attach_mock(self._sir._infect_event, 'inf')
        event_manager.attach_mock(self._sir._recover_event, 'rec')

        if case == 1:
            self._give_state_to_nodes([0, 1, 3], [4, 5], [2], [])
            self._sir._num_s_i_edges = 1
            self._sir._infected_nodes = [2]
            self._sir._infected_node_edges = {2: [5]}
        elif case == 2:
            self._give_state_to_nodes([0, 3, 4], [1], [2, 5], [])
            self._sir._num_s_i_edges = 1
            self._sir._infected_nodes = [2, 5]
            self._sir._infected_node_edges = {2: [], 5: [1]}

        return event_manager

    def _prepare_for_infect_event(self, case):
        nx.classes.function.set_node_attributes(
            self._sir.g, 
            SirState.SUSCEPTIBLE, 
            self._state
        )

        if case == 1:
            self._give_state_to_nodes([5, 0], [1, 2, 4], [3])
            self._sir._num_s_i_edges = 5
            self._sir._infected_nodes = [1, 2, 4]
            self._sir._infected_node_edges = {1: [0, 5], 2: [0], 4: [0]}
        elif case == 2:
            self._give_state_to_nodes([0], [1, 2], [3, 4, 5])
            self._sir._num_s_i_edges = 5
            self._sir._infected_nodes = [1, 2]
            self._sir._infected_node_edges = {1: [0], 2: [0]}

    def _give_state_to_nodes(self, susc, inf, rec, vac):
        for i in susc:
                self._sir.g.nodes[i][self._state] = SirState.RECOVERED

        for i in inf:
            self._sir.g.nodes[i][self._state] = SirState.SUSCEPTIBLE
        
        for i in rec:
            self._sir.g.nodes[i][self._state] = SirState.INFECTED

        for i in vac:
            self._sir.g.nodes[i][self._state] = SirState.VACCINATED
