import numpy as np
import pennylane as qml

class QuantumReservoir:
    def __init__(self, n_qubits=6, n_layers=3, input_scale=np.pi, seed=0):
        self.n_qubits, self.n_layers, self.input_scale = n_qubits, n_layers, input_scale
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0, 2*np.pi, size=(n_layers, n_qubits, 2))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, window):
        for layer in range(self.n_layers):
            angle = self.input_scale * window[layer]
            for w in range(self.n_qubits):
                qml.RY(angle, wires=w)
            for w in range(self.n_qubits):
                qml.RX(self.weights[layer, w, 0], wires=w)
                qml.RZ(self.weights[layer, w, 1], wires=w)
            for w in range(self.n_qubits):
                qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

    def run(self, inputs):
        padded = np.concatenate([np.zeros(self.n_layers - 1), inputs])
        feats = np.empty((len(inputs), self.n_qubits))
        for t in range(len(inputs)):
            feats[t] = np.asarray(self._qnode(padded[t:t + self.n_layers]))
        return feats
