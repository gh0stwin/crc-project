from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.coreness import Coreness


class CorenessSimulator(Simulator):
    def __init__(self):
        super(CorenessSimulator, self).__init__()

    def _get_vacc_prot_and_args(self):
        return Coreness(), {}        
