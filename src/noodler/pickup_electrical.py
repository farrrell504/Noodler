class Pickup_Electrical:
    def __init__(
        self,
        name='generic',
        L=2.5,
        Rp = 15e3,
        C = 120e-12,
        R_eddy = 200e3,
    ):
        import numpy as np

        self.name = name
        self.L = L
        self.Rp = Rp
        self.Cp = C
        self.R_eddy = R_eddy
        self._make_circuit_model()

        self._audible_frequencies = np.logspace(0, np.log10(20000), 200)

    def _make_circuit_model(self):
        from lcapy import Circuit
        import warnings

        warnings.filterwarnings("ignore")

        self.circuit_model = Circuit()
        # self.circuit_model.add("V42 1 0_1 {u(t)};down")
        # self.circuit_model.add("V42 1 0 step;down")
        self.circuit_model.add("L1 1 2;right, i={i_L}")
        self.circuit_model.add("Re 2 21;down")
        self.circuit_model.add("W 0 21;right")

        self.circuit_model.add("L2 2 3;right, i={i_L}")
        self.circuit_model.add("Rpickup 3 4;right")
        self.circuit_model.add("Cpickup 4 41;down,v=vcpickup")
        self.circuit_model.add("W 21 41;right")
        self.circuit_model.add("W 4 5;right")
        self.circuit_model.add("W 41 51;right")

        self._get_tf()

    def _get_tf(self):
        from lcapy import j, omega, s

        test_circuit = self.circuit_model.copy()
        test_circuit.add("Vi 1 0 step; down")
        test_circuit.add("Rload 5 51; down")
        print('put method to detect if Voltage source already exists, or have one with one without or something?')
        self.H=(test_circuit.Rload.V(s) / test_circuit.Vi.V(s)).simplify()
        self.tf = self.H(j * omega)
        return self.tf

    def substitute(self, expression=None):
        if expression is None:
            expression = self.circuit_model.copy()
        return expression.subs({'Re':self.R_eddy, "Rpickup":self.Rp, "L1":self.L, "L2":self.L, "Cpickup":self.Cp})

    def draw(self):
        self.circuit_model.draw()

    def plot_impedance_vs_frequency(self):
        from lcapy import f
        from notebook_helpers import plot_log_x

        # what the controls see
        plot_log_x(
            self.circuit_model.oneport(Np="5", Nm="5_1")
            .Z(f)
            .evaluate(self._audible_frequencies),
            fs=40e3,
            y_range=[70, 120],
            title=f"Pickup: {self.name}, Impedance the Controls See",
        )
        # what the "strings" see
        plot_log_x(
            self.circuit_model.oneport(Np="0_1", Nm="0")
            .Z(f)
            .evaluate(self._audible_frequencies),
            fs=40e3,
            y_range=[70, 120],
            title=f"Pickup: {self.name}, Impedance the 'String' See's",
        )

    def plot_tf_their_way(self):
        from lcapy import Circuit, j, omega, s
        tf = self.H.subs('Rpickup',self.Rp).subs('Cpickup',self.Cp).subs('Re',self.R_eddy).subs("L1",self.L).subs("L2",self.L)

        from numpy import logspace
        w = logspace(1, 6, 500)
        ax = tf(j * omega).dB.plot(w, log_frequency=True)
        return ax