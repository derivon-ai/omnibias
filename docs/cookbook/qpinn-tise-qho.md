# QPINN cookbook: 1D harmonic-oscillator ground state (TISE)

This cookbook trains a small `omnibias-qpinn` network to recover the
ground state :math:`\psi_0(x) = \pi^{-1/4} \exp(-x^2/2)` of the
quantum harmonic oscillator (QHO) and its energy
:math:`E_0 = \hbar\omega/2 = 1/2` (atomic units, :math:`\hbar = m =
\omega = 1`).

## The physics

The time-independent Schrodinger equation in 1D is

\[
    \hat H\,\psi(x) = E\,\psi(x),
    \qquad
    \hat H = -\frac{1}{2}\frac{d^2}{dx^2} + \frac{1}{2}\,x^2.
\]

The eigenfunctions are the Hermite-Gaussians; the ground state is

\[
    \psi_0(x) = \pi^{-1/4}\,e^{-x^2/2},
    \qquad
    E_0 = \tfrac{1}{2}.
\]

We train a `OneLayerVectorField` with `gaussian` activations and the
`make_psi_components` encoding, minimising the residual
:math:`\|(\hat H - E)\,\psi\|_2^2` plus a soft norm-conservation term.

## Pythonic recipe (torch backend)

<!-- docs-test: slow -->
```python
import torch

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.cage import norm_loss
from omnibias.qpinn.torch.equations import tise

torch.manual_seed(0)
coord = CoordinateSpec(("x",))
spec = make_psi_components(name="psi")
field = OneLayerVectorField(
    coordinate_spec=coord, components=spec,
    hidden=32, base="gaussian", dtype=torch.float64,
)

# Quadrature grid: trapezoidal weights for a [-4, 4] box.
xs = torch.linspace(-4.0, 4.0, 401, dtype=torch.float64).unsqueeze(-1)
ws = torch.full((401,), 8.0 / 401, dtype=torch.float64)

def V(state):
    return 0.5 * state.coords[..., 0] ** 2

optim = torch.optim.Adam(field.parameters(), lr=5e-3)
for step in range(5_000):
    optim.zero_grad()
    state = field(xs)
    out = tise(state, energy=0.5, potential=V, quadrature_weights=ws)
    residual_loss = (out.residual ** 2).sum(dim=-1).mean()
    norm_pen = norm_loss(state, quadrature_weights=ws, target_norm=1.0)
    loss = residual_loss + 10.0 * norm_pen
    loss.backward()
    optim.step()
    if step % 1000 == 0:
        E = float(out.energy_estimate.detach())
        print(f"step {step:5d}  loss={float(loss):.3e}  E_est={E:.6f}")
```

Setting `energy=0.5` pins the eigenvalue (we already know what we
want). If you want to *learn* :math:`E_0`, pass a 0-d trainable
`torch.nn.Parameter` and include it in the optimiser.

## Diagnostics

`omnibias.qpinn.torch.diagnostics` exposes the variational
expectation values used to verify convergence:

```python
from omnibias.qpinn.torch.diagnostics import expected_energy, norm_squared

state = field(xs)
E = expected_energy(state, potential=V, quadrature_weights=ws)
N = norm_squared(state, quadrature_weights=ws)
print(f"<H> = {float(E):.6f}, ||psi||^2 = {float(N):.6f}")
```

For the trained QHO ground state we expect `<H>` to converge to
:math:`0.5` and :math:`\|\psi\|^2` to converge to :math:`1`.

## Compute discipline

The example above runs in about 60 seconds on CPU and is mirrored by the package integration smoke check in [`tests/integration/test_qho_eigenstates.py`][test_qho]. Full-fidelity sweeps are kept in the internal benchmark archive (not shipped publicly; see [`benchmarks.md`](../benchmarks.md)), not in the public package tree.

[test_qho]: https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-qpinn/tests/integration/test_qho_eigenstates.py
