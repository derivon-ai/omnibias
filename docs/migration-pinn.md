# Migration guide: DeepXDE / NVIDIA Modulus -> `omnibias-pinn`

This guide is for users coming from
[DeepXDE](https://github.com/lululxvi/deepxde) or
[NVIDIA Modulus (Sym)](https://github.com/NVIDIA/modulus-sym) who want
to take advantage of `omnibias-pinn`'s closed-form n-th derivative
operators, hard-conservation cages, and bit-stable cross-backend
numerics.

## Why migrate

| Feature | DeepXDE | Modulus | `omnibias-pinn` |
| --- | --- | --- | --- |
| Closed-form spatial derivatives | none (autograd only) | none (autograd only) | yes (Fourier / Chebyshev / one-layer fields) |
| Hard incompressibility cage | no | partial (geometry-only) | yes (`StreamfunctionField`, `VectorPotentialField`) |
| O(1) memory for high-order derivatives | no | no | yes |
| Cross-backend numerical parity | partial | no | yes (Torch + JAX, bit-identical) |
| Equation registry typed outputs | partial | partial | yes (`NamedTuple` outputs with diagnostics) |
| Wang-Perdikaris causal weighting | manual | none | first-class |
| Sobolev preconditioning | manual | none | first-class |
| NTK rebalance | none | none | first-class |

For 4th-order PDEs (Navier-Stokes vorticity-stream, Cahn-Hilliard,
Kuramoto-Sivashinsky), the autograd cost in DeepXDE / Modulus grows
exponentially with derivative order. `omnibias-pinn` makes the cost
*independent* of order.

## API mapping

### Geometry / domain

| DeepXDE | omnibias-pinn |
| --- | --- |
| `dde.geometry.Interval(0, L)` | `CoordinateSpec(axes=("x",), periodicity=(True,))` |
| `dde.geometry.Rectangle((0,0), (L,L))` | `CoordinateSpec(axes=("x", "y"), periodicity=(True, True))` |
| `dde.geometry.Cuboid((0,0,0), (L,L,L))` | `CoordinateSpec(axes=("x", "y", "z"), periodicity=(True, True, True))` |
| `dde.geometry.TimeDomain(0, T)` | extra axis: `axes=("x", ..., "t"), time_axis="t"` |

### Network

| DeepXDE | omnibias-pinn |
| --- | --- |
| `dde.nn.FNN([D] + [H]*L + [C], "tanh", "Glorot uniform")` | `OneLayerVectorField(coordinate_spec=..., components=..., hidden=H, base="tanh")` for L=1 |
| -- (stack OMBU layers) | `SpectralVectorField(K=..., time_hidden=H, time_depth=L, activation="tanh")` |
| -- | `ChebyshevVectorField(N=..., ...)` (non-periodic) |

### Residual / loss

DeepXDE pattern:

<!-- docs-test: skip reason="DeepXDE-side comparison code; DeepXDE is not an omnibias dependency" -->
```python
def pde_residual(x, y):
    u = y[:, 0:1]
    du_dt = dde.grad.jacobian(y, x, i=0, j=1)
    du_dx = dde.grad.jacobian(y, x, i=0, j=0)
    d2u_dx2 = dde.grad.hessian(y, x, i=0, j=0)
    return du_dt + u * du_dx - 0.01 * d2u_dx2
```

`omnibias-pinn` equivalent:

```python
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields import SpectralVectorField
from omnibias.pinn.torch.equations import Burgers

equation = Burgers(nu=0.01, form="scalar", component="u")

def loss(state):
    out = equation(state)
    return (out.residual ** 2).mean()

field = SpectralVectorField(
    coordinate_spec=CoordinateSpec(axes=("x", "t"), time_axis="t"),
    components=ComponentSpec(("u",)),
    K=8, time_hidden=32, time_depth=1, activation="tanh",
)
coords = torch.rand(64, 2, dtype=torch.float64) * 6.28
out = equation(field(coords))
```

The `Burgers` class encapsulates the full residual; you don't write
the derivative chain by hand. The same pattern works for
`Heat`, `Burgers`, `KuramotoSivashinsky`, `CahnHilliard`,
`Biharmonic`, `NavierStokes`.

### Boundary conditions

| DeepXDE | omnibias-pinn |
| --- | --- |
| `dde.icbc.DirichletBC(geom, lambda x: ..., on_boundary)` | `HardCondition(component, axis, dirichlet(face), value)` on a box, or `HardBoundaryField(base, distance_fn, boundary_fn, components)` for arbitrary geometry |
| `dde.icbc.NeumannBC(...)` / `RobinBC(...)` | `HardCondition(component, axis, neumann(face))` / `robin(face, alpha=, beta=)` |
| `dde.icbc.PeriodicBC(...)` | Solver: `BoundaryCondition(component, kind="periodic", axis=...)` (default `periodic_orders=(0, 1, 2)`), or cage: `HardCondition(component, axis, periodic(lo, hi, order=n))` for each order; or `basis="spectral"` / `SpectralVectorField` when the ansatz already carries spatial periodicity |
| `dde.icbc.IC(geom, lambda x: ..., lambda x: x[1] == 0)` | `HardCondition(component, time_axis, dirichlet(t0), value)`; `derivative_at(t0, 1)` for an initial velocity |

DeepXDE imposes all of these as penalty terms. `ConstrainedExpressionField`
collects the `HardCondition`s above into an ansatz that satisfies them
identically, so the corresponding loss terms disappear rather than being
weighted -- and on the solver, `hard_conditions="auto"` builds that ansatz for
you and drops the rows it absorbed.

#### Periodic domains (migration)

Marking an axis `periodicity=True` on a `Domain` / `CoordinateSpec` does **not**
by itself emit a periodic boundary condition into the residual. The six
canonical builders take `periodic_boundary=False` by default; pass
`periodic_boundary=True` to append one periodic `BoundaryCondition` per
component per periodic spatial axis. Hand-built `System`s still need an
explicit periodic BC (or `basis="spectral"` on a time-dependent solve, where
spatial periodicity lives in the Fourier ansatz).

Default seam orders moved from `(0, 1)` to `PERIODIC_ORDERS = (0, 1, 2)`
(value, slope, and second derivative) after a C¹-seam sweep under a second-order
operator. Override with `periodic_orders=(...)` on the `BoundaryCondition` if
you need the old subset.

### Training loop

DeepXDE:

<!-- docs-test: skip reason="DeepXDE-side comparison code; DeepXDE is not an omnibias dependency" -->
```python
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)
```

`omnibias-pinn`: write your own loop (typically ~30 LOC). See the
[cookbook](cookbook/pinn-navier-stokes.md) for a worked example with
Sobolev / causal / IC losses.

### Causal weighting (Wang-Perdikaris 2022)

DeepXDE: requires a custom callback or manual reweighting.
`omnibias-pinn`:

```python
from omnibias.pinn.torch.losses import causal_residual_loss

# The residual is reduced on a grid whose leading axis is time.
res_grid = out.residual.reshape(8, 8)
loss = causal_residual_loss(res_grid, epsilon=1.0)
```

### Sobolev preconditioning

DeepXDE: not built-in (write a custom loss). `omnibias-pinn`:

```python
from omnibias.pinn.torch.losses import sobolev_residual_loss
loss = sobolev_residual_loss(res_grid, sobolev_p=0.5, L=2.0 * torch.pi, spatial_axes=(1,))
```

The Sobolev weight matrix is precomputed once; per-iteration cost is
just an FFT and a multiply.

## Modulus -> omnibias-pinn

Modulus's `nodes`, `arch`, and `constraint` system maps less directly
because it operates at a higher level of abstraction. The general
pattern is:

* `Node` -> `FieldBase` (with `ComponentSpec` for the output channels)
* `Arch` (FullyConnectedArch / FourierNetArch) ->
  `OneLayerVectorField` / `SpectralVectorField`
* `Constraint(geom, eq)` -> `equation(state)` evaluation in the
  training loop, optionally wrapped in a cage for hard constraints.

Modulus's *built-in* PDEs map directly onto `omnibias.pinn.torch.equations`:

| Modulus equation | omnibias-pinn |
| --- | --- |
| `NavierStokes(...)` | `equations.NavierStokes(form="primitive_3d")` |
| `Diffusion(...)` | `equations.Heat(...)` |
| `WaveEquation(...)` | (manual; compose `derivative` ops on `Biharmonic`) |
| `KuramotoSivashinsky(...)` | `equations.KuramotoSivashinsky(...)` |

## Numerical parity

If you want to verify that an `omnibias-pinn` solver matches a known
DeepXDE / Modulus baseline:

1. Initialise both networks with the *same* random seed.
2. Generate the *same* collocation grid.
3. Evaluate both residuals at the same points.
4. Compare per-point with a tight tolerance.

For closed-form vs autograd derivatives, `omnibias-pinn` matches
autograd to ~`1e-12` in float64 (verified by
`packages/omnibias-pinn/tests/torch/test_torch_*.py`). Cross-backend
parity (Torch <-> JAX) is enforced to the same tolerance by
`packages/omnibias-pinn/tests/cross_backend/`.

## Common gotchas

* `omnibias-pinn` defaults to `float64` (DeepXDE/Modulus default to
  `float32`). Set `dtype=torch.float32` explicitly if you want
  parity with a `float32` baseline.
* `SpectralVectorField` requires a *time axis* on the
  `CoordinateSpec` (the spatial axes go through Fourier; time goes
  through the omnibias temporal MLP). For steady-state problems use
  `OneLayerVectorField` or `ChebyshevVectorField` instead.
* The `VectorPotentialField` cage produces a divergence-free velocity
  by construction. Use `incompressibility="hard"` on `NavierStokes`
  to skip the soft `mean (div u)^2` penalty.
