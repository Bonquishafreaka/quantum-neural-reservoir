import numpy as np

def izhikevich_trace(n_steps=2000, dt=0.5, a=0.02, b=0.2, c=-65.0, d=8.0, I=10.0, seed=0):
    rng = np.random.default_rng(seed)
    v, u = -65.0, b * -65.0
    out = np.empty(n_steps)
    for t in range(n_steps):
        v += dt * (0.04*v*v + 5*v + 140 - u + I + rng.normal(0, 1))
        u += dt * (a * (b*v - u))
        if v >= 30:
            out[t] = 30; v = c; u += d
        else:
            out[t] = v
    return _norm(out)

def hindmarsh_rose_trace(n_steps=2000, dt=0.05, a=1, b=3, c=1, d=5, r=0.006, s=4, x_rest=-1.6, I=3.2, seed=0):
    x, y, z = -1.5, -10.0, 2.0
    out = np.empty(n_steps)
    for t in range(n_steps):
        x += dt * (y - a*x**3 + b*x**2 - z + I)
        y += dt * (c - d*x**2 - y)
        z += dt * (r * (s*(x - x_rest) - z))
        out[t] = x
    return _norm(out)

def make_supervised(series, horizon=1, washout=100):
    x, y = series[:-horizon], series[horizon:]
    return x[washout:], y[washout:]

def _norm(a):
    a = a - a.mean()
    return a / (np.max(np.abs(a)) or 1.0)
