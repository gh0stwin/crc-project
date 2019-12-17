from sir.simulators.simulator import Simulator
from sir.sir import Sir
from vaccination_protocols.no_vaccination import NoVaccination


class NoVaccinationSimulator(Simulator):
    def __init__(self):
        super(NoVaccinationSimulator, self).__init__()

    def _get_vacc_prot_and_args(self, **kwargs):
        return NoVaccination(), {}        
