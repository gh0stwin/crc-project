from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.dfs import Dfs


class DfsSimulator(Simulator):
    def __init__(self):
        super(DfsSimulator, self).__init__()

    def _get_vacc_prot_and_args(self, **kwargs):
        return Dfs(), {}        
