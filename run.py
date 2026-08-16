import numpy as np
import matplotlib.pyplot as plt

from data import hindmarsh_rose_trace, make_supervised
from reservoir import QuantumReservoir
from readout import RidgeReadout

def nrmse(pred, true):
    return float(np.sqrt(np.mean((pred - true)**2)) / (true.std() or 1.0))

def r2(pred, true):
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - true.mean())**2) or 1.0
    return float(1 - ss_res / ss_tot)

def split(x, y, frac=0.7):
    n = int(len(x) * frac)
    return x[:n], y[:n], x[n:], y[n:]

def linear_baseline(inputs, targets, n_lags, frac=0.7):
    padded = np.concatenate([np.zeros(n_lags - 1), inputs])
    feats = np.stack([padded[t:t + n_lags] for t in range(len(inputs))])
    xtr, ytr, xte, yte = split(feats, targets, frac)
    ro = RidgeReadout().fit(xtr, ytr)
    pred = ro.predict(xte)
    return nrmse(pred, yte), r2(pred, yte)

if __name__ == "__main__":
    series = hindmarsh_rose_trace(n_steps=2000, seed=0)
    inputs, targets = make_supervised(series, horizon=20)

    res = QuantumReservoir(n_qubits=8, n_layers=3, seed=0)
    feats = res.run(inputs)

    xtr, ytr, xte, yte = split(feats, targets, 0.7)
    ro = RidgeReadout().fit(xtr, ytr)
    pred = ro.predict(xte)

    b_nrmse, b_r2 = linear_baseline(inputs, targets, 3)

    print("quantum: ", {"nrmse": round(nrmse(pred, yte), 3), "r2": round(r2(pred, yte), 3)})
    print("baseline:", {"nrmse": round(b_nrmse, 3), "r2": round(b_r2, 3)})

    plt.figure(figsize=(10, 3.2))
    plt.plot(yte[:300], label="true")
    plt.plot(pred[:300], "--", label="predicted")
    plt.xlabel("timestep"); plt.ylabel("membrane potential (norm.)")
    plt.legend(); plt.tight_layout()
    plt.savefig("prediction.png", dpi=130)
    plt.show()
