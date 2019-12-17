from vaccination_protocols.vaccination_protocol import VaccinationProtocol


class NoVaccination(VaccinationProtocol):
    def __init__(self):
        super(NoVaccination, self).__init__()
        self._acronym = 'NO'

    def vaccinate_network(self, g, f, **kwargs):
        return 0
