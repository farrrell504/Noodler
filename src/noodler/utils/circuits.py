from dataclasses import dataclass as _dataclass
from dataclasses import field as _field


class ElectricalComponent:
    def __add__(self, circuit):
        return self.circuit + circuit.circuit

    def __or__(self, circuit):
        return self.circuit | circuit.circuit


class CircuitMixin:
    def _get_tf(self):
        from lcapy import Circuit, j, omega, s

        test_circuit = self.circuit.copy()
        test_circuit.add("Rload 0 00; down")
        test_circuit.add("W 00 01; left")
        test_circuit.add("Vi 1 01 step; down")

        self.H=(test_circuit.Rload.V(s) / test_circuit.Vi.V(s)).simplify()
        self.tf = self.H(j * omega)
        return self.tf

    def draw(self):
        self.circuit.draw()

@_dataclass
class Pot(ElectricalComponent):
    value: float
    name: str = _field(default="Rpot")
    current_value: float = _field(default=0.0)
    _turn_ratio: float = _field(default=0.0)

    def __post_init__(self):
        from lcapy import R
        self._update_circuit()
        self.circuit = R(self.name)   

    @property
    def turn_ratio(self):
        return self._turn_ratio

    @turn_ratio.setter
    def turn_ratio(self, new_turn_ratio):
        self._turn_ratio = new_turn_ratio
        self._update_circuit()

    def _update_circuit(self):
        self.current_value = int(self.value * self.turn_ratio)

@_dataclass
class Resistance(ElectricalComponent):
    value: float
    name: str = _field(default="R")

    def __post_init__(self):
        if "R" not in self.name:
            self.name = "R_" + self.name

@_dataclass
class Capacitance(ElectricalComponent):
    value: float
    name: str = _field(default="C")

    def __post_init__(self):
        if "C" not in self.name:
            self.name = "C_" + self.name
