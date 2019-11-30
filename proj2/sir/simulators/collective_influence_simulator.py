from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.collective_influence import CollectiveInfluence


class CollectiveInfluenceSimulator(Simulator):
    def __init__(self):
        super(CollectiveInfluenceSimulator, self).__init__()

    def _get_vacc_prot_and_args(self):
        return CollectiveInfluence(), {} 
