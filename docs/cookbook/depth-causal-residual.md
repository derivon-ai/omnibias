# Depth-causal residual

A layer can take a Gauss–Newton step on the **PDE residual of its own
decoded field** before the next layer sees the activations. Spec 08-05
is a depth march, not time marching and not CCF stretch. Bias collapse
(`delta -> 0`) supplies the 2-jet.

## Manufactured 1-D Poisson: constant field, exact residual

`u = x(1-x)` has `u'' = -2`. A network that decodes the constant `1`,
wrapped by the hard Dirichlet factor `x(1-x)`, therefore has residual
zero against `f = 2`.

```python
import torch
from omnibias.pinn.train import DepthResidualConfig
from omnibias.pinn.train.torch.depth_residual import (
    field_tower,
    poisson_1d_residual,
)

torch.set_default_dtype(torch.float64)

xs = torch.linspace(0.1, 0.9, 9)
layers = [(torch.zeros(1, 1), torch.ones(1), None)]
decode = (torch.ones(1, 1), None)
cfg = DepthResidualConfig(n_directions=1, jet_order=2, damping=0.0)
tower = field_tower(layers, decode, xs, config=cfg)
residual = poisson_1d_residual(torch.full_like(xs, 2.0))(tower)
assert float(residual.abs().max()) <= 1e-12
assert tower[2].allclose(torch.full_like(xs, -2.0))
```

## One depth sweep, honesty keys stay sealed

```python
import math
import torch
from omnibias.pinn.train import DepthResidualConfig
from omnibias.pinn.train.torch.depth_residual import (
    depth_residual_sweep,
    poisson_1d_residual,
)

torch.set_default_dtype(torch.float64)

xs = torch.linspace(0.15, 0.85, 7)
force = (math.pi**2) * torch.sin(math.pi * xs)
layers = [(0.3 * torch.ones(3, 1), torch.zeros(3), "tanh")]
decode = (0.2 * torch.ones(1, 3), torch.zeros(1))
cfg = DepthResidualConfig(n_directions=1, jet_order=2, damping=1e-3)
_new, _dec, report = depth_residual_sweep(
    layers,
    decode,
    poisson_1d_residual(force),
    xs,
    config=cfg,
)
assert report.navier_stokes_proof_claim is False
assert report.stretch_cleared is False
assert report.hilbert_not_in_scope is True
assert report.greedy_only_claimed_optimal is False
assert report.n_gn_steps == 1
```

`n_directions >= n_params` raises `DepthResidualForbidden` unless
`allow_full=True`. Last-layer-only is the G1 control and spends the
same GN step budget on the last block. Hilbert is out of scope.
