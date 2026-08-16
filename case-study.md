# Can a Quantum Circuit Simulate a Neuron? A Reservoir-Computing Case Study

*A small, honest experiment in using quantum dynamics as a simulation substrate for
biological neural activity — no quantum hardware, no GPU, no claims that brains are quantum.*

## Why this project exists

"Quantum neural network" gets used for two completely different things: a quantum-flavoured
*model* that plays the role of an artificial neural net with no biology involved, and a
quantum method used as a *tool to study real neural processes*. This project is the second
kind, and it narrows the question further. It does not claim neurons are quantum mechanical
— that's contested Penrose–Hameroff territory and not what this is about. Instead:

> Can the quantum formalism act as a useful *simulation substrate* for the probabilistic,
> dynamical behaviour neurons exhibit?

The vehicle is reservoir computing, and the answer is a qualified, interesting *yes*.

## The one idea you need: reservoir computing

Most machine learning trains every parameter. Reservoir computing takes a **fixed**
high-dimensional dynamical system (the *reservoir*), drives it with your input so its
internal state becomes a rich nonlinear transformation of the signal's recent history, and
trains **only** a linear readout on top. The reservoir is never trained — which is exactly
why a quantum version is cheap: no variational optimisation, no barren-plateau gradients.
Training collapses to a single ridge-regression solve.

A quantum system is a natural reservoir. An n-qubit circuit lives in a 2ⁿ-dimensional state
space with intrinsically nonlinear, entangling dynamics. Feed a signal in, measure a few
expectation values out, and you have your feature vector.

## The experiment

**The neuron.** We simulate a single biological neuron two ways. The Izhikevich model gives
clean regular spiking; the Hindmarsh–Rose model gives richer *bursting* dynamics — bursts of
spikes separated by quiet stretches — which is the harder, more interesting target. Both
produce a membrane-potential trace normalised to roughly [-1, 1].

**The task.** Predict the neuron's activity *k* timesteps ahead. Small *k* is easy; large
*k* demands genuine memory of past dynamics.

**The reservoir.** Six to eight qubits. At each step we angle-encode the input onto every
qubit, then apply fixed random single-qubit rotations and a ring of CNOTs to entangle
everything. To give it memory, we re-encode a short window of recent inputs across the
layers, so past values leave a fading trace. We read out ⟨Z⟩ on each qubit.

**The readout.** Ridge regression. One matrix solve.

**The baseline — and this matters.** We compare against a linear model fed the same raw
lagged inputs. Without a baseline any result looks impressive; with one you find out whether
the quantum dynamics bought you anything.

## Results

Next-value prediction on the bursting neuron is a humbling start:

| Horizon | Quantum reservoir (R²) | Linear baseline (R²) |
|--------:|-----------------------:|---------------------:|
| 1       | 0.82                   | **1.00**             |
| 5       | 0.76                   | **0.98**             |

At short horizons the dumb baseline crushes the reservoir — and that's correct. Consecutive
samples of a smooth trace barely differ, so "predict roughly the last value" is nearly
perfect. No amount of entanglement improves on trivial.

The story flips as we look further ahead:

| Horizon | Quantum reservoir (R²) | Linear baseline (R²) |
|--------:|-----------------------:|---------------------:|
| 10      | 0.77                   | 0.81                 |
| 20      | **0.93**               | 0.25                 |

At a 20-step horizon the baseline collapses to 0.25 — raw lagged inputs don't carry enough
about where a bursting neuron is headed. The quantum reservoir holds at 0.93. Its fading
memory has encoded enough of the dynamical trajectory to see the next burst coming.

![Predicted vs. true membrane potential, 20 steps ahead](prediction.png)

That crossover is the entire finding. The reservoir isn't magic; it's a memory machine. It
doesn't help when the problem needs no memory, and it shines when the problem needs a lot.

## What this does and doesn't show

**It shows** a small fixed quantum circuit can serve as a legitimate temporal-simulation
substrate for nonlinear neural dynamics, and that its advantage is specifically about memory
over time — precisely what makes neural processes hard to model.

**It does not show** any quantum *advantage* over classical reservoir computing. A classical
echo-state network might match or beat this; establishing advantage needs that head-to-head.

**It does not show** anything about whether real brains use quantum mechanics. Nothing here
depends on that.

## Running it yourself

The whole thing runs on CPU — an 8-qubit state vector is a few thousand complex numbers.
GPUs only matter if you later switch to *training* a large variational circuit, which the
fixed-reservoir design avoids.

```bash
pip install -r requirements.txt
python run.py
```

## Where it goes next

- **Real data.** Swap the neuron simulator for an EEG channel or intracellular recording.
- **Classical showdown.** Benchmark against an echo-state network to test for real advantage.
- **Populations.** Model coupled neurons and look for synchronisation.
- **Tune the reservoir.** Input scaling, entangling topology, and layer count are unexplored.
