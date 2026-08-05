# QPINN cookbook: dark soliton of the 1D Gross-Pitaevskii equation

The defocusing 1D Gross-Pitaevskii equation

\[
    i\hbar\,\partial_t\psi
    = -\frac{\hbar^2}{2m}\,\partial_x^2\psi
    + g\,|\psi|^2\,\psi
\]

with :math:`g > 0` admits the stationary dark-soliton solution

\[
    \psi_{\text{soliton}}(x, t)
    = \sqrt{\rho_0}\,\tanh\!\Big(\frac{x}{\sqrt{2}\,\xi}\Big)\,
      e^{-i\mu t / \hbar},
\]

with chemical potential :math:`\mu = g\rho_0` and healing length
:math:`\xi = \hbar / \sqrt{2 m g \rho_0}`.

## Pythonic recipe (torch backend)

<!-- docs-test: slow -->
```python
import torch
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.equations import nls

torch.manual_seed(0)
coord = CoordinateSpec(("x", "t"))
spec = make_psi_components(name="psi")
field = OneLayerVectorField(
    coordinate_spec=coord, components=spec,
    hidden=64, base="tanh", dtype=torch.float64,
)

xs = torch.linspace(-6.0, 6.0, 41, dtype=torch.float64)
ts = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
grid = torch.cartesian_prod(xs, ts)

# Initial condition: dark soliton at t=0.
rho_0, xi = 1.0, 1.0
def initial_re(x):
    return (rho_0 ** 0.5) * torch.tanh(x / (xi * (2 ** 0.5)))

optim = torch.optim.Adam(field.parameters(), lr=5e-4)
for step in range(20_000):
    optim.zero_grad()
    state_pde = field(grid)
    nl_out = nls(state_pde, g=1.0, hbar=1.0, mass=1.0)
    pde_loss = (nl_out.residual ** 2).sum(dim=-1).mean()
    # IC: psi_re(x, 0) = tanh(...), psi_im(x, 0) = 0.
    ic_pts = torch.stack((xs, torch.zeros_like(xs)), dim=-1)
    state_ic = field(ic_pts)
    psi_re_ic = state_ic.ops.value(state_ic, "psi_re")
    psi_im_ic = state_ic.ops.value(state_ic, "psi_im")
    ic_loss = (
        (psi_re_ic - initial_re(xs)) ** 2 + psi_im_ic ** 2
    ).mean()
    loss = pde_loss + 1e2 * ic_loss
    loss.backward()
    optim.step()
```

The dark-soliton solution has a phase that rotates uniformly in time
but a static amplitude profile. The trained network should recover
both; useful diagnostics are the *density* :math:`|\psi|^2` and the
*phase* :math:`\arg\psi`:

```python
from omnibias.qpinn._core.complex import psi_density, psi_phase

state = field(grid)
density = psi_density(state)
phase = psi_phase(state, atan2=torch.atan2)
```

## Why qpinn is the natural choice here

The Gross-Pitaevskii equation has two second-order spatial derivatives
*and* a nonlinearity that couples them through :math:`|\psi|^2`. Any
PINN approach is hostage to the stability of the second-derivative
estimator. `omnibias-qpinn` uses the closed-form ``sigma''(z)`` from
`omnibias-core`, so the second-derivative path is numerically identical
to its forward pass -- there is no double-backprop or finite-difference
estimator anywhere in the loop.

## Compute discipline

This recipe runs in ~5 minutes on CPU and is captured by
[`tests/integration/test_dark_soliton_gp.py`][test_soliton]. The
full-fidelity benchmark (comparison to the analytic profile + the
4th-order split-step reference at multiple healing lengths) lives in
the internal benchmark archive (not shipped publicly; see
[`benchmarks.md`](../benchmarks.md)).

[test_soliton]: https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-qpinn/tests/integration/test_dark_soliton_gp.py
