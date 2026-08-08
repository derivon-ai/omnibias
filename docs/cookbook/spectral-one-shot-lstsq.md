# One-shot least-squares (no GD dynamics)

Spectral bias is a statement about gradient-descent dynamics on the NTK: low
eigenmodes learn first. Freeze the hidden features so
``u = sum_j c_j phi_j`` is linear in the readout ``c``. Closed-form
``sigma^(n)`` makes ``L[phi_j]`` exact, so ``L[u]`` stays linear in ``c``.
Assemble interior + boundary rows once and solve. Skipping the dynamics
removes the NTK bias; what remains is feature bandwidth and conditioning
(the f=64 capacity falsification in
[`spectral_bias_fbpinn.json`](../benchmarks/spectral_bias_fbpinn.json)).

## PDE path (preferred)

```python
import math
import numpy as np
import torch
import omnibias.pinn.solver as pde
import omnibias.pinn.solver.torch as pt

torch.set_default_dtype(torch.float64)
dom = pde.Domain(("x",), ((0.0, 1.0),))

def source(c):
    xp = pde.array_namespace(c)
    return -(math.pi**2) * xp.sin(math.pi * c[:, 0])

system = pde.poisson(dom, source=source, boundary=0.0)
sol = pt.solve_least_squares(
    system,
    hidden=48,
    seed=0,
    collocation=pde.CollocationSpec(n_interior=40, n_boundary=8),
    hard_conditions="auto",
)
pts = np.linspace(0.02, 0.98, 40).reshape(-1, 1)
u = sol.evaluate(pts, "u").detach().numpy()
ustar = np.sin(math.pi * pts[:, 0])
rel = float(np.linalg.norm(u - ustar) / np.linalg.norm(ustar))
assert rel < 1e-2, rel
assert sol.diagnostics["hard_absorbed"] == 2
```

``hard_conditions="auto"`` absorbs every condition the planner can certify
into the ansatz and drops those rows, so the boundary is exact by construction
and the least-squares problem is smaller.

## Honesty on speed and memory

From the instrumented spectral benchmark (median over seeds/freqs, float64 CPU):

- Accuracy: capacity-rich ``lstsq`` clears ``5e-6`` through f=16; a
  parameter-matched ``lstsq_matched`` arm is weaker but still beats plain GD.
- Speed: one-shot wall time is typically ~20x below an equal-budget Adam arm
  at this scale (see ``median_wall_seconds`` in the JSON). Complexity is one
  feature build ``O(N H)`` plus one SVD ``O(N H^2)`` vs Adam ``O(S N H)``.
- Memory: structural loss. Adam holds ``O(P)``; one-shot materialises a dense
  ``N x H`` design matrix. Trivial at benchmark scale; binding at
  ``N ~ 10^6``, ``H ~ 10^4``.

Use ``solve_least_squares`` only on ``Linearity.LINEAR`` systems. Nonlinear
problems still need ``solve_optimize`` (or marching / cages).
