# QPINN cookbook: free-particle Gaussian wavepacket (TDSE)

This cookbook propagates a Gaussian wavepacket on :math:`x \in
\mathbb{R}, t \ge 0` under the free-particle TDSE and verifies the
omnibias-qpinn output against the closed-form analytic propagator.

## The physics

The free-particle time-dependent Schrodinger equation in 1D is

\[
    i\hbar\,\partial_t\psi(x, t)
    = -\frac{\hbar^2}{2m}\,\partial_x^2\,\psi(x, t),
\]

with the initial Gaussian wavepacket :math:`\psi(x, 0) = (\pi
\alpha^2)^{-1/4}\,\exp[-(x - x_0)^2/(2\alpha^2) + i p_0 (x - x_0)/\hbar]`.
The exact solution at time :math:`t` is

\[
    \psi(x, t) = \frac{1}{(\pi)^{1/4}\sqrt{\alpha + i\hbar t/(m\alpha)}}\,
                \exp\!\Big[-\frac{(x - x_0 - p_0 t / m)^2}{2(\alpha^2 + i\hbar t/m)}
                + \frac{i p_0 (x - x_0)}{\hbar}
                - \frac{i p_0^2 t}{2 m \hbar}\Big].
\]

## Pythonic recipe (torch backend)

<!-- docs-test: slow -->
```python
import torch
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.equations import tdse

torch.manual_seed(0)
coord = CoordinateSpec(("x", "t"))
spec = make_psi_components(name="psi")
field = OneLayerVectorField(
    coordinate_spec=coord, components=spec,
    hidden=64, base="gaussian", dtype=torch.float64,
)

# Initial condition (t=0) and PDE collocation grid.
xs = torch.linspace(-4.0, 4.0, 51, dtype=torch.float64)
ts = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
grid = torch.cartesian_prod(xs, ts)

def initial_psi(x):
    return torch.exp(-x * x / 2.0)  # real-valued, alpha = 1, p_0 = 0

optim = torch.optim.Adam(field.parameters(), lr=1e-3)
for step in range(10_000):
    optim.zero_grad()
    state_pde = field(grid)
    pde_out = tdse(state_pde, hbar=1.0, mass=1.0)
    pde_loss = (pde_out.residual ** 2).sum(dim=-1).mean()
    # Initial condition: psi(x, 0) = exp(-x^2/2), psi_im(x, 0) = 0.
    ic_pts = torch.stack((xs, torch.zeros_like(xs)), dim=-1)
    state_ic = field(ic_pts)
    psi_re_ic = state_ic.ops.value(state_ic, "psi_re")
    psi_im_ic = state_ic.ops.value(state_ic, "psi_im")
    ic_loss = (
        (psi_re_ic - initial_psi(xs)) ** 2
        + psi_im_ic ** 2
    ).mean()
    loss = pde_loss + 10.0 * ic_loss
    loss.backward()
    optim.step()
```

## Continuity-equation diagnostic

The trained network can be checked against the conservation law
:math:`\partial_t\rho + \nabla\cdot\vec j = 0`:

```python
from omnibias.qpinn.torch.diagnostics import continuity_residual

state = field(grid)
r = continuity_residual(state, hbar=1.0, mass=1.0)
print(f"max |continuity residual| = {float(r.detach().abs().max())}")
```

A well-trained free-particle wavefunction should satisfy this exactly
in continuum (and to convergence in the network).

## Compute discipline

This recipe runs in <2 minutes on CPU and is captured by
[`tests/integration/test_gaussian_wavepacket.py`][test_wave]. The
full-fidelity comparison to the analytic propagator at multiple
:math:`t` values lives in the internal benchmark archive (not shipped
publicly; see [`benchmarks.md`](../benchmarks.md)) and runs on a GPU
cluster outside the public package tree.

[test_wave]: https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-qpinn/tests/integration/test_gaussian_wavepacket.py
