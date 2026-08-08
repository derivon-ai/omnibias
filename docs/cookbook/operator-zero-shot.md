# Operator zero-shot with parameter conditioning

A DeepONet that sees only sensor values of ``u_0`` cannot distinguish two
diffusivities that share the same IC. Multi-head conditioning adds a scalar
parameter head; for ``n_parameters=1`` LayerNorm is skipped (width-1 LayerNorm
zeros its input). The acceptance gate requires conditioned median rel-L2 to
beat both the unconditioned ablation and a per-instance residual PINN retrain,
with an ETDRK4 / exact mode-1 heat reference that respects the maximum
principle. See
[`docs/benchmarks/pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).

```python
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.operator import ConditioningSpec
from omnibias.pinn.operator.torch import build_deeponet

torch.set_default_dtype(torch.float64)
cond = ConditioningSpec(n_function_sensors=8, n_parameters=1)
op = build_deeponet(
    coordinate_spec=CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    ),
    components=ComponentSpec(("u",)),
    n_sensors=8,
    trunk_width=4,
    trunk_hidden=8,
    trunk_depth=2,
    branch_hidden=8,
    branch_depth=2,
    conditioning=cond,
)
sensors = torch.randn(1, 8).expand(3, -1).contiguous()
params = torch.tensor([[0.05], [0.15], [0.35]])
coeffs, _bias = op.branch(sensors=sensors, parameters=params)
d01 = float((coeffs[0] - coeffs[1]).detach().norm())
d02 = float((coeffs[0] - coeffs[2]).detach().norm())
assert d01 > 1e-8, d01
assert d02 > 1e-8, d02
```

Train once across a diffusivity sweep; evaluate zero-shot on held-out ``nu``.
Do not march an explicit RK4 heat reference without a maximum-principle check
-- that path produced ``max|u| ~ 1e9`` while still looking ``isfinite``.
