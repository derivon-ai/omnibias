# omnibias-score

**Status: Alpha (0.1.0a1).**

Score-based / SDE operators composed from the `omnibias-fields` closed-form
gradient / Hessian primitives (no new low-level kernels):

- `score(state, name)` -- the score `grad log p = grad p / p`.
- `ito_generator(state, name, *, drift, diffusion)` -- the Ito generator
  `L f = b . grad f + 1/2 tr(a hess f)`.
- `fokker_planck(state, name, *, drift, diffusion, drift_divergence)` -- the
  Fokker-Planck adjoint `L* p = -div(b p) + 1/2 a_ij d_i d_j p` (constant
  diffusion `a`).

```python
import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.score.torch import ops as sde

field = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("x", "y")),
    components=ComponentSpec(("f",)), hidden=16, base="tanh",
)
x = torch.randn(8, 2, dtype=torch.float64)
state = field(x)

# Ornstein-Uhlenbeck generator on a field component "f": drift is per-point (B, D).
b, a = -x, torch.eye(2, dtype=torch.float64)
Lf = sde.ito_generator(state, "f", drift=b, diffusion=a)   # (B,)
```

## Validation

On the Ornstein-Uhlenbeck process `dX = -theta X dt + sigma dW` the stationary
density `N(0, sigma^2/(2 theta))` satisfies `L* p_inf = 0`; the generator on
monomials and the score of the Gaussian match their closed forms; torch and jax
agree to `rtol=1e-12`.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
