# Piecewise hybrid automaton

A switched ODE is recovered as a `HybridAutomaton`: one sparse equation per
region plus the hardened `if ... then` switch. Three drivers, with different
honesty:

1. **Oracle partition** -- you supply the split; STLSQ fits each cell
   (`fit_piecewise_law`). The control.
2. **Learned gates** -- `fit_learned_piecewise_ode` trains soft weights, hardens,
   then STLSQ-polishes (it calls `_refine_split_threshold`).
3. **Tab-head harden** -- a SoftTree / Arrangement trained on the trajectory's
   finite-difference `du` is hardened from the **fitted** split
   (`tree_params` / `arrangement_params`). Arrangement is **unplanted**
   (constructor random `W`, no `e_0`); that path does **not** call
   `_refine_split_threshold`. STLSQ still uses the field jet.

STLSQ polish is numpy / non-differentiable. Gate hardening is `beta -> inf`
(feasibility / temperature), never the founding `delta -> 0` bias collapse.
The full neural-field pipeline lives in
[`docs/examples/piecewise_discovery.py`](../examples/piecewise_discovery.py).

## Oracle partition (algebraic control)

Sample the state space directly -- no neural field, no 400-point fit.

```python
import numpy as np
from omnibias.partition import PartitionConfig
from omnibias.partition._core.params import PartitionParams
from omnibias.symbolic.piecewise import (
    fit_learned_piecewise_ode,
    fit_piecewise_law,
    polynomial_value_library,
    polynomial_vector_library,
)

rng = np.random.default_rng(0)
n = 80
x = rng.uniform(-2.0, 2.0, size=n)
u = rng.uniform(-1.0, 2.0, size=n)
du = np.where(x < 0.0, 1.0 - u, -2.0 * u)   # A: du = 1 - u ; B: du = -2 u
design, names = polynomial_value_library(u, degree=2)
cfg = PartitionConfig(
    n_features=1, depth=1, split_kind="axis", beta_final=32.0, anneal_steps=1
)
partition = PartitionParams(cfg, W=np.array([[1.0]]), t=np.array([0.0]))
oracle = fit_piecewise_law(
    partition, x.reshape(-1, 1), design, du, names,
    lhs_name="du", alpha=1e-12, threshold=1e-5,
)
laws = {law.region: law for law in oracle.laws}
assert set(laws) == {0, 1}
assert abs(laws[0].equation.coefficients[0] - (-1.0)) < 1e-2
assert abs(laws[1].equation.coefficients[0] - (-2.0)) < 1e-2
```

## Vector systems: `report()` prints `k` formulas

Shared gates; two left-hand sides in one `report()` block.

```python
u0 = rng.uniform(-1.0, 2.0, size=n)
u1 = rng.uniform(-1.0, 1.0, size=n)
U = np.stack([u0, u1], axis=1)
du_vec = np.stack(
    [np.where(x < 0.0, 1.0 - u0, -2.0 * u0), np.where(x < 0.0, u1, -u1)],
    axis=1,
)
design_v, names_v = polynomial_vector_library(U, degree=1)
vector = fit_piecewise_law(
    partition, x.reshape(-1, 1), design_v, du_vec, names_v,
    lhs_names=("du0", "du1"), alpha=1e-12, threshold=1e-5,
)
report = vector.report()
assert "du0" in report and "du1" in report
assert report.count("if [") == 2
```

## Learned gates (API smoke)

A few Adam steps on the same algebraic samples. This is not the recovery
gate -- that lives in the example and the test suite.

```python
learned, state = fit_learned_piecewise_ode(
    x.reshape(-1, 1), u, du,
    n_gates=1, degree=1, steps=40, seed=0,
    alpha=1e-12, threshold=1e-5,
)
assert "if [" in learned.report()
assert "W" in state
```

## Tab-head harden is unplanted

Train a depth-1 SoftTree or `H=1` Arrangement on finite-difference `du` of a
**trajectory** (kinked), then pass the fitted `W` / `t` to `tree_params` /
`arrangement_params` and run `fit_piecewise_law` on a field jet. Do not plant
`W = e_0`. Do not call `_refine_split_threshold` on that path. See
[`docs/examples/piecewise_discovery.py`](../examples/piecewise_discovery.py)
and `packages/omnibias-symbolic/tests/test_piecewise.py`.
