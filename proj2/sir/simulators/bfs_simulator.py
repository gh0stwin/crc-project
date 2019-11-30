from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.bfs import Bfs


class BfsSimulator(Simulator):
    def __init__(self):
        super(BfsSimulator, self).__init__()

    def _get_vacc_prot_and_args(self, **kwargs):
        return Bfs(), {}        
