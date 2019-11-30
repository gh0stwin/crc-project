from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.acquaintance import Acquaintance


class AcquaintanceSimulator(Simulator):
    def __init__(self):
        super(AcquaintanceSimulator, self).__init__()

    def _get_vacc_prot_and_args(self, **kwargs):
        return Acquaintance(), {}
