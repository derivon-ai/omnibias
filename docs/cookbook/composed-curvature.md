# Composed-curvature joint Newton

A two-layer residual can sit at a last-layer least-squares minimum
while the joint Hessian of `(W_{ell-1}, W_ell)` is indefinite. The
order-2 chain rule keeps the `sigma''` term that Gauss–Newton drops.
This is theory spec 08-02: a local escape from a *slice* critical
point, not a global solver and not a CCF stretch fix.

## Scalar nest (spec §5)

```python
from omnibias.core.composed_curvature import scalar_nest_hessian

h_ww, h_wv, h_vv = scalar_nest_hessian(0.2, 0.1, 1.0)
assert h_vv > 0.0
assert abs(h_wv) > 0.0
```

`L = 1/2 (v tanh(w) - 1)^2`. The mixed block is the coupling; it is
not Gauss–Newton.

## Joint step on a torch residual

```python
import torch
from omnibias.core.composed_curvature import ComposedCurvatureConfig
from omnibias.torch.optim_composed import composed_curvature_step

torch.set_default_dtype(torch.float64)

def residual(w, v):
    return (v * torch.tanh(w) - 1.0).reshape(1)

cfg = ComposedCurvatureConfig(n_directions=2, allow_full=True)
new_w, new_v, report = composed_curvature_step(
    residual,
    torch.tensor(0.2),
    torch.tensor(0.1),
    config=cfg,
)
assert report.step_norm >= 0.0
assert isinstance(report.escaped, bool)
```

Two parameters need `allow_full=True`. A larger net should keep
`n_directions << n_params`. Length along the joint direction uses the
03-12 jet line search with `verify=True`.
