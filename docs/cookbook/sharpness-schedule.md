# Sharpness-scheduled cubic step

Exact Hessian-vector products give a Ritz `lambda_max`. That value
sets the cubic penalty (or a learning rate). This is theory spec
08-06: a step-size signal, not a generalization claim and not CCF
stretch.

## Stiff quadratic (spec §5)

```python
import torch
from omnibias.torch.optim_sharpness import (
    SharpnessSchedule,
    sharpness_lambda_max,
    sharpness_scheduled_minimize,
)

torch.set_default_dtype(torch.float64)

def loss(params):
    p = params.reshape(-1)
    return 0.5 * (p[0] * p[0] + 1.0e4 * p[1] * p[1])

ell = sharpness_lambda_max(loss, torch.tensor([1.0, 1.0]), n_lanczos=4)
assert abs(ell - 1.0e4) / 1.0e4 < 1e-10

schedule = SharpnessSchedule(n_lanczos=4, c=1e-3, ell_min=1e-6)
params, losses, ells = sharpness_scheduled_minimize(
    loss, torch.tensor([1.0, 1.0]), schedule=schedule, steps=20
)
assert all(v == v and v not in (float("inf"), float("-inf")) for v in losses)
assert losses[-1] < 1e-8
assert max(ells) > 1.0
```

`c = 1e-3` is the named constant recorded in the smoke artifact. The
API default `c = 1.0` is more conservative and stays finite; it is
slower to drive this quadratic under `1e-8` in twenty steps. A
non-positive scheduled value refuses the step instead of taking an
unscaled update.

## Hook on CubicNewton

```python
import torch
from omnibias.torch.optim import CubicRegularizedNewton, SharpnessSchedule

torch.set_default_dtype(torch.float64)

def loss(params):
    p = params.reshape(-1)
    return 0.5 * (p[0] * p[0] + 1.0e4 * p[1] * p[1])

opt = CubicRegularizedNewton(
    krylov_dim=4,
    sharpness=SharpnessSchedule(n_lanczos=4, c=1e-3, ell_min=1e-6),
)
new, info = opt.step(loss, torch.tensor([1.0, 1.0]))
assert opt.last_ell_k is not None
assert abs(opt.last_ell_k - 1.0e4) / 1.0e4 < 1e-10
assert info.sigma > 0.0
assert bool(torch.isfinite(new).all())
```
