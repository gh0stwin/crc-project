from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.degree import Degree


class DegreeSimulator(Simulator):
    def __init__(self):
        super(DegreeSimulator, self).__init__()

    def _get_vacc_prot_and_args(self):
        return Degree(), {}        
