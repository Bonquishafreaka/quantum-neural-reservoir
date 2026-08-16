# Quantum Neural Reservoir

Simulating the temporal dynamics of a biological neuron with a **quantum reservoir** —
a fixed quantum circuit whose rich internal dynamics act as a computational substrate,
with only a cheap classical linear layer trained on top.

This is *not* a claim that neurons are quantum mechanical. It asks a narrower question:
**can the quantum formalism serve as a useful simulation substrate for the probabilistic,
dynamical behaviour neurons exhibit?**

## The idea

A reservoir computer feeds a signal into a fixed high-dimensional dynamical system and
reads out predictions with a trained linear map. The reservoir itself is never trained, so
there's no variational optimisation of the circuit — training is a single ridge-regression
solve. We drive a small (6–8 qubit) circuit with a simulated neuron's membrane potential,
one timestep at a time, and predict its future activity. Runs on CPU; no GPU, no quantum
hardware.

## Results

Predicting a bursting Hindmarsh–Rose neuron *k* steps ahead (R², higher is better):

| Horizon | Quantum reservoir | Linear lag baseline |
|--------:|------------------:|--------------------:|
| 1       | 0.82              | **1.00**            |
| 5       | 0.76              | **0.98**            |
| 10      | 0.77              | 0.81                |
| 20      | **0.93**          | 0.25                |

At short horizons a trivial baseline wins — consecutive samples barely change. The
reservoir earns its keep at **long horizons**, where its fading memory captures temporal
structure raw lags can't. That crossover is the point of the study.

![prediction](prediction.png)

## Run it

```bash
pip install -r requirements.txt
python run.py
```

## Files

data.py neuron models (Izhikevich, Hindmarsh-Rose) + supervised framing
reservoir.py the fixed quantum reservoir circuit
readout.py ridge-regression linear readout (the only trained part)
run.py end-to-end pipeline + linear baseline + plot

## Swapping in real data

Replace the generators in `data.py` with a function returning a 1-D normalised array — an
EEG channel, an intracellular recording, a binned spike train — and the rest is unchanged.

## Caveats

- Small qubit counts only; simulation cost doubles per qubit (~20 qubits is where a laptop slows).
- The reservoir is fixed random — no attempt to optimise its dynamics.
- Single-neuron traces. Population dynamics is the natural next step.
- No claim of quantum *advantage*; that needs a head-to-head against a classical echo-state network.
