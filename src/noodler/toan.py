from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from .utils.circuits import ElectricalComponent, CircuitMixin, Pot, Resistance, Capacitance


@_dataclass
class ToanBleeder(ElectricalComponent):
    resistance: float
    capacitance: float
    name: str = _field(default="tb")

    def __post_init__(self):
        from lcapy import R, C

        self.elements = [Resistance(self.resistance, self.name), Capacitance(self.capacitance, self.name)]
        self.circuit = R(self.elements[0].name) | C(self.elements[1].name)


class Toans(CircuitMixin):
    def __init__(
        self,
        volume_pot=250e3,
        toan_bleed_cap=.0015e-6,
        toan_bleed_res=220e3,
    ):
        import numpy as np

        self.volume_pot = Pot(volume_pot)
        self.toan_bleeder = ToanBleeder(toan_bleed_res, toan_bleed_cap)
        self._make_circuit_model()

        self._audible_frequencies = np.logspace(0, np.log10(20000), 200)

    def _make_circuit_model(self):
        from lcapy import Circuit

        self.circuit = Circuit((self.volume_pot | self.toan_bleeder).netlist())

    def _find_last_element(self):
        for e in reversed(self.circuit.elements):
            if ("WW" not in e) and ("V" not in e):
                return e

    def plot_impedance_vs_frequency(self):
        from lcapy import f
        from notebook_helpers import plot_log_x

        # what the output see
        plot_log_x(
            self.circuit.oneport(Np="5", Nm="5_1")
            .Z(f)
            .evaluate(self._audible_frequencies),
            fs=40e3,
            y_range=[70, 120],
            title=f"Impedance @ Output",
        )
        # what the pickup see
        plot_log_x(
            self.circuit.oneport(Np="1", Nm="0")
            .Z(f)
            .evaluate(self._audible_frequencies),
            fs=40e3,
            y_range=[70, 120],
            title=f"Impedance the Pickup See's",
        )
