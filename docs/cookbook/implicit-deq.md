# Implicit / DEQ Newton

A tiny equilibrium `u = tanh(W u + x)` is solved by Newton and
differentiated with exact `sigma'`. Spec 08-08 is the implicit-function
theorem at a fixed point, not unrolled BPTT and not CCF stretch.
Bias collapse (`delta -> 0`) supplies `sigma'`.

## Spec §5: scalar fixed point and IFT slope

```python
import torch
from omnibias.core.implicit import DEQConfig
from omnibias.torch.implicit import deq_solve, deq_vjp

torch.set_default_dtype(torch.float64)
cfg = DEQConfig(solver="newton", tol=1e-12)
w = torch.tensor([[0.2]])
x = torch.tensor([0.5])
result = deq_solve(w, x, "tanh", config=cfg)
u = float(result.u.reshape(-1)[0])
assert 0.542 < u < 0.544
assert result.residual <= 1e-10
assert result.spectral_radius_bound < 1.0

g = deq_vjp(w, x, "tanh", torch.tensor([1.0]), config=cfg)
eps = 1e-6
up = float(deq_solve(w + eps, x, "tanh", config=cfg).u.reshape(-1)[0])
um = float(deq_solve(w - eps, x, "tanh", config=cfg).u.reshape(-1)[0])
fd = (up - um) / (2.0 * eps)
rel = abs(float(g.reshape(-1)[0]) - fd) / max(abs(fd), 1e-12)
assert rel <= 1e-8
```

A bound `>= 1` with `require_contraction=True` raises
`DEQNotContractive` instead of silently unrolling. JAX uses
`lax.while_loop` for the default Newton loop; see
[implicit.md](../api/implicit.md).
