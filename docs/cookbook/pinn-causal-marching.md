# Causal time marching

Regime matters. On linear heat with a hard IC/BC cage, a whole-interval solve
already reaches skill near 1, so marching pays fragmentation and handoff cost.
On Krishnapriyan's stiff reaction ``u_t = rho u(1-u)`` at ``rho = 12``,
whole-interval fails causality and gated marching wins. Absolute gates and
both families live in
[`docs/benchmarks/pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).

Supply the IC as ``ic_fn`` evaluated on the marcher's own slice points -- never
as a linspace-ordered ``ic_values`` vector (that contaminates the seam metric).

```python
import math
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.constrained import HardCondition, dirichlet
from omnibias.pinn._core.marching import TimeWindowSchedule
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.cage import ConstrainedExpressionField
from omnibias.pinn.torch.fields import OneLayerVectorField
from omnibias.pinn.train.torch import march_solve

torch.set_default_dtype(torch.float64)
cs = CoordinateSpec(("t", "x"), domain=((0.0, 0.25), (0.0, 1.0)), time_axis="t")
base = OneLayerVectorField(
    coordinate_spec=cs,
    components=ComponentSpec(("u",)),
    hidden=16,
    base="tanh",
)
field = ConstrainedExpressionField(
    base=base,
    conditions=[
        HardCondition("u", 1, dirichlet(0.0), 0.0),
        HardCondition("u", 1, dirichlet(1.0), 0.0),
        HardCondition("u", 0, dirichlet(0.0), lambda c: torch.sin(math.pi * c[:, 1])),
    ],
)

def residual_fn(fld, coords):
    st = fld(coords)
    return tops.derivative(st, "u", axis="t", order=1) - tops.derivative(
        st, "u", axis="x", order=2
    )

def ic_fn(coords):
    return torch.sin(math.pi * coords[:, 1])

result = march_solve(
    field,
    residual_fn,
    cs,
    TimeWindowSchedule(
        0.0, 0.25, n_windows=2, n_time_bins=4, epsilon=1.0, tolerance=1.0
    ),
    steps_per_window=30,
    max_steps_per_window=30,
    lr=1e-2,
    per_bin=4,
    n_slice=16,
    ic_fn=ic_fn,
    ic_mode="hard",
    value_fn=lambda fld, c: tops.value(fld(c), "u"),
    advance_policy="gate",
    seed=0,
)
assert len(result.windows) >= 1
assert result.windows[0].seam_mse is not None
assert result.windows[0].seam_mse < 1e-20  # hard cage: seam is machine zero
# With advance_policy="gate", an unconverged first window stays put -- that is
# the honesty of the gate, not a failure of the cage.
```

For the stiff reaction regime, drop the hard cage, use soft IC with
``ic_weight=10``, ``epsilon=0.5``, and compare marched vs whole-interval at
equal step budget -- that is the gate in ``benchmarks/causal_marching.py``.
