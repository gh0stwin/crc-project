from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.random import Random


class RandomSimulator(Simulator):
    def __init__(self):
        super(RandomSimulator, self).__init__()

    def _get_vacc_prot_and_args(self, **kwargs):
        return Random(), {}        
