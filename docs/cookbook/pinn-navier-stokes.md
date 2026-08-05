# PINN: 2D / 3D Navier-Stokes via the new `omnibias.pinn` API

The headline `omnibias-pinn` v0.1 demo is incompressible Navier-Stokes
solved with closed-form spatial derivatives via `SpectralVectorField`,
hard incompressibility via the `VectorPotentialField` cage, and the
prebuilt `NavierStokes` residual.

This cookbook keeps the package-level recipe. Full benchmark drivers and heavy GPU outputs live in the internal benchmark archive (not shipped publicly; see [`benchmarks.md`](../benchmarks.md)).

## 2D vorticity-streamfunction

```python
import torch

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields import SpectralVectorField
from omnibias.pinn.torch.equations import NavierStokes

field = SpectralVectorField(
    coordinate_spec=CoordinateSpec(axes=("x", "y", "t"), time_axis="t"),
    components=ComponentSpec(("psi",)),
    K=32, time_hidden=128, time_depth=3, activation="tanh",
)

equation = NavierStokes(
    viscosity=1e-2, form="vorticity_stream_2d", streamfunction="psi",
)

coords = torch.rand(4096, 3, dtype=torch.float64) * 6.28
state = field(coords)
out = equation(state)
loss = (out.residual ** 2).mean()
```

The spatial derivatives in the residual ($\psi_{xx}$, $\psi_{yy}$,
$\Delta\psi$, $(\Delta\psi)_x$, $(\Delta\psi)_y$, $\Delta^2\psi$) are
all **diagonal in the Fourier basis** -- a single
$\mathcal{O}(B\,K^2)$ matmul per residual evaluation, *independent of
order*. The time derivative goes through the omnibias temporal MLP
fastpath when `time_depth=1`, and through `torch.func.jacrev`+`vmap`
when `time_depth > 1`.

## 3D primitive variables with hard incompressibility

```python
import torch

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields import SpectralVectorField
from omnibias.pinn.torch.cage import VectorPotentialField
from omnibias.pinn.torch.equations import NavierStokes

base = SpectralVectorField(
    coordinate_spec=CoordinateSpec(axes=("x", "y", "z", "t"), time_axis="t"),
    components=ComponentSpec(("A1", "A2", "A3", "p")),
    K=8, time_hidden=128, time_depth=3, activation="tanh",
)

cage = VectorPotentialField(
    base=base,
    A_components=("A1", "A2", "A3"),
    velocity_names=("u", "v", "w"),
    passthrough_names=("p",),
)

equation = NavierStokes(
    viscosity=1e-2, form="primitive_3d",
    velocity=("u", "v", "w"), pressure="p",
    incompressibility="hard",   # cage already enforces div(u) = 0
)

coords = torch.rand(4096, 4, dtype=torch.float64) * 6.28
state = cage(coords)
out = equation(state)
loss = (out.residual ** 2).mean()      # 3-component momentum residual
```

Why this is fast:

* The cage produces $u = \nabla\times A$ and exposes only the
  derivatives that the residual actually needs (curl-compatible
  combinations of derivatives of $A$, all of which are diagonal in
  the Fourier basis).
* Continuity $\nabla\cdot u = 0$ is **identically zero** by
  construction, so we don't pay the soft `mean(div u)^2` penalty.
* The vector Laplacian $\Delta u$ is built from the spectral
  $\Delta A$ via the same single-matmul shortcut.

## Sobolev + causal loss

The plain MSE residual loss is sometimes ill-conditioned (high-frequency
components are over-weighted). Ramp it down with the Sobolev
preconditioner:

Both of these losses act on a **gridded** residual tensor rather than a scattered
collocation set: the Sobolev weight is applied in the Fourier basis over the
spatial axes, and the causal weighting reduces along a leading time axis.

```python
import torch
from omnibias.pinn.torch.losses import sobolev_residual_loss

res = torch.randn(8, 16, 16, dtype=torch.float64)   # (n_t, n_x, n_y) residual grid
loss = sobolev_residual_loss(res, sobolev_p=0.5, L=2.0 * torch.pi, spatial_axes=(1, 2))
```

For long-horizon problems also stage the loss along time with
Wang-Perdikaris causal weighting:

```python
from omnibias.pinn.torch.losses import causal_residual_loss

# The leading axis is time; `epsilon` sets how sharply later times are damped
# until the earlier ones are solved. `sobolev_p > 0` composes both preconditioners.
loss = causal_residual_loss(res, epsilon=1.0, sobolev_p=0.5, L=2.0 * torch.pi)
```

## End-to-end benchmark discipline

The public package tree contains the reusable field, cage, equation, and loss APIs. GPU job scripts, multi-seed aggregate reports, reference trajectories, predictions, and checkpoints are maintained in the internal benchmark archive (not shipped publicly; see [`benchmarks.md`](../benchmarks.md)).
