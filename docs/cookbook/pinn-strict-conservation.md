# PINN: strict-conservation cages

`omnibias.pinn.torch.cage` provides architectural layers that
enforce physical invariants **by construction**, eliminating the need
for soft penalty terms (and their tuning). Every cage is a thin
wrapper around an underlying `FieldBase` and exposes a transformed
component view; all derivatives propagate through the cage's algebraic
identity, preserving the closed-form fast-path.

## Streamfunction (2D incompressible)

For 2D incompressible flow on $\mathbb{R}^2$:

$$
u = \partial_y \psi, \qquad v = -\partial_x \psi
\;\Longrightarrow\;
\partial_x u + \partial_y v \equiv 0.
$$

```python
import torch

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields import SpectralVectorField
from omnibias.pinn.torch.cage import StreamfunctionField


def spectral(axes, names, K=8):
    """A small spectral base field, used as the caged `base` throughout."""
    return SpectralVectorField(
        coordinate_spec=CoordinateSpec(axes=axes, time_axis="t"),
        components=ComponentSpec(names),
        K=K, time_hidden=32, time_depth=1, activation="tanh",
    )


cage = StreamfunctionField(
    base=spectral(("x", "y", "t"), ("psi", "p")),   # exposes a "psi" component
    psi="psi",
    velocity_names=("u", "v"),
    passthrough_names=("p",),
)

coords = torch.rand(64, 3, dtype=torch.float64) * 6.28
state = cage(coords)
print(float(state.velocity.div.abs().max()))   # exactly 0.0
print(state.components.names)                  # ('u', 'v', 'p') -- psi is consumed
```

The caged state exposes the velocity it derives, not the streamfunction it derives
it *from*: `u` is $\partial_y\psi$ by construction, so `psi` no longer appears as a
component.

## Vector potential (3D incompressible)

For 3D incompressible flow,
$u = \nabla\times A \;\Longrightarrow\; \nabla\cdot u \equiv 0$.

```python
from omnibias.pinn.torch.cage import VectorPotentialField

cage3d = VectorPotentialField(
    base=spectral(("x", "y", "z", "t"), ("A1", "A2", "A3", "p")),
    A_components=("A1", "A2", "A3"),
    velocity_names=("u", "v", "w"),
    passthrough_names=("p",),
)

coords3d = torch.rand(32, 4, dtype=torch.float64) * 6.28
state3d = cage3d(coords3d)
assert torch.allclose(
    state3d.velocity.div, torch.zeros_like(state3d.velocity.div), atol=1e-12,
)
```

The Coulomb-gauge constraint $\nabla\cdot A = 0$ may be enforced as a
soft term. The gauge losses take the *cage* and the coordinates (they need the
pre-curl potential, which the caged state no longer exposes):

```python
from omnibias.pinn.torch.cage import coulomb_gauge_loss
loss_gauge = coulomb_gauge_loss(cage3d, coords3d)
```

## Helmholtz projection (soft + parameterised)

When a learned pressure is preferred over a vector potential, project
the predicted velocity:

$$
u = u_{\text{pred}} - \nabla\phi
\;\Longrightarrow\;
\Delta\phi = \nabla\cdot u_{\text{pred}}\;\;
\text{(Poisson)}.
$$

```python
from omnibias.pinn.torch.cage import HelmholtzProjectionField, helmholtz_gauge_loss

cage_h = HelmholtzProjectionField(
    base=spectral(("x", "y", "z", "t"), ("u_pred", "v_pred", "w_pred", "phi", "p")),
    u_pred_components=("u_pred", "v_pred", "w_pred"),
    phi="phi",
    velocity_names=("u", "v", "w"),
    passthrough_names=("p",),
)
state_h = cage_h(coords3d)
loss_gauge = helmholtz_gauge_loss(cage_h, coords3d)
```

## Energy / enstrophy conservation (skew-symmetric advection)

`EnergyConserving.advection(state, vel)` returns the skew-symmetric
discretisation of $(u\cdot\nabla)v$ that conserves
$\int |v|^2 \,dx$ to round-off, even for non-divergence-free $u$
(useful when the cage is soft or absent):

```python
from omnibias.pinn.torch.cage import energy_conserving_advection
adv = energy_conserving_advection(state3d, velocity=("u", "v", "w"))
```

The 2D enstrophy-conserving variant plays the same role for the vorticity
equation, taking the vorticity component alongside the velocity:

```python
from omnibias.pinn.torch.cage import enstrophy_conserving_advection

vort = spectral(("x", "y", "t"), ("omega", "u", "v"))(coords)
adv_omega = enstrophy_conserving_advection(
    vort, velocity=("u", "v"), vorticity="omega",
)
```

## Hard boundary conditions

Two cages impose boundary data exactly, and the choice is made by geometry.

`HardBoundaryField` handles a boundary of **any shape**, but only Dirichlet
data, and it does not compose -- wrapping it around a derivative condition
breaks whichever constraint ends up inside.
[`ConstrainedExpressionField`](#hard-neumann-robin-and-initial-conditions)
needs an axis-aligned **box**, and in exchange covers Dirichlet, Neumann, Robin
and initial conditions uniformly, composes exactly across axes and kinds, and
does not blow up at corners.

### Dirichlet on an arbitrary boundary

`HardBoundaryField` enforces $u\big|_{\partial\Omega} = g(x, t)$ via a
distance-function ansatz $u = g + d(x) \cdot \tilde u$:

```python
from omnibias.pinn.torch.cage import HardBoundaryField

def distance(x): return x[..., 0] * (1.0 - x[..., 0])   # vanishes at x=0,1

def boundary_value(x): return {"u": torch.zeros_like(x[..., 0])}

cage_bc = HardBoundaryField(
    base=spectral(("x", "y", "t"), ("u", "p")),
    distance_fn=distance,
    boundary_value_fn=boundary_value,
    bounded_names=("u",),
    passthrough_names=("p",),
)

edge = torch.zeros(4, 3, dtype=torch.float64)            # x = 0 exactly
edge[:, 1] = torch.rand(4, dtype=torch.float64)
print(float(cage_bc(edge).u.value.abs().max()))          # 0.0, for any parameters
```

`boundary_value_fn` receives the coordinate tensor and returns one entry per
bounded component; `bounded_names` selects which components the ansatz wraps.

The cage interferes only with the values and derivatives of the wrapped
components -- pass-through components (e.g. pressure) flow through
unchanged. All derivative paths use the product rule on
$d \cdot \tilde u$ and the chain rule on $g(x, t)$, so the closed-form
fastpath is preserved end-to-end.

### Hard Neumann, Robin and initial conditions

On a box, `ConstrainedExpressionField` embeds any set of linear conditions
$C_k[u] = t_k$ using switching functions $\varphi_i$ with
$C_k[\varphi_i] = \delta_{ki}$:

$$u = g + \sum_k \varphi_k \bigl(t_k - C_k[g]\bigr),$$

which satisfies every condition for **any** free function $g$. Dirichlet,
Neumann, Robin and an initial value or velocity are all one `LinearConstraint`
type with different terms -- and so is periodicity, as the *relative* constraint
$\partial^n u(hi) - \partial^n u(lo) = 0$, since a linear functional may
reference several points.

For a **time-dependent** problem whose spatial axes are already periodic, prefer
a `SpectralVectorField` (or the solver's `basis="spectral"` on
`build_field` / `solve_least_squares`): spatial periodicity is free in the
Fourier ansatz, so you do not need to spend cage degrees of freedom on the
seam. Keep `ConstrainedExpressionField` + `periodic(...)` for steady problems,
MLP ansatze, or when you want algebraic seam matching on top of a non-spectral
base.

```python
from omnibias.pinn._core.constrained import (
    HardCondition, derivative_at, dirichlet, neumann, periodic,
)
from omnibias.pinn.torch.cage import ConstrainedExpressionField

wave_cage = ConstrainedExpressionField(
    base=spectral(("x", "y", "t"), ("u", "p")),
    conditions=[
        HardCondition("u", 0, dirichlet(0.0), 0.0),          # u(0, y, t) = 0
        HardCondition("u", 0, neumann(1.0), 0.0),            # u_x(1, y, t) = 0
        HardCondition("u", 1, periodic(0.0, 1.0, order=0), 0.0),  # seam in y
        HardCondition("u", 1, periodic(0.0, 1.0, order=1), 0.0),  # ... and slope
        # Initial state x(2 - x): zero at x = 0 and flat at x = 1, so it agrees
        # with the two conditions above where the axes meet. Data that does not
        # is refused at construction rather than half-satisfied.
        HardCondition("u", 2, dirichlet(0.0), lambda c: c[:, 0] * (2.0 - c[:, 0])),
        HardCondition("u", 2, derivative_at(0.0, 1), 0.0),   # u_t(x, y, 0) = 0
    ],
    bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    passthrough_names=("p",),
)
face = torch.rand(4, 3, dtype=torch.float64)
face[:, 0] = 0.0
print(float(wave_cage(face).u.value.abs().max()) < 1e-14)    # True, untrained
print(wave_cage.projection_cost)                             # 18 base evaluations
```

Three constrained axes cost `3 * 3 * 2 = 18` base evaluations per pass -- the
product over axes of `1 + #projection points`, not the sum. That is the price of
exact edges and corners, and `projection_cost` reports it so the decision to
absorb a third axis is made on a number.

The support matrix behind the switching functions has to be invertible, and
that is **certified** rather than assumed: `support_certificates()` seals a
hash-verifiable enclosure of `lambda_min(M^T M) > 0`, and a linearly dependent
condition set is refused at construction. Condition data on different axes must
also agree where those axes meet, and construction refuses data that does not,
naming the two conditions that clash; `compatibility_residual` keeps the size of
the disagreement visible, which is order one when it is real.

## When to choose which cage

| Cage | Constraint | Cost | Notes |
| --- | --- | --- | --- |
| `StreamfunctionField` | 2D `div u = 0` | free | Optimal for 2D NS |
| `VectorPotentialField` | 3D `div u = 0` | small (1 extra tensor) | Headline 3D NS choice |
| `HelmholtzProjectionField` | soft `div u = 0` via Poisson | medium (extra `phi` field) | Use when a learned pressure is required |
| `HardBoundaryField` | Dirichlet `u = g` | free | Any geometry; Dirichlet only, does not compose |
| `ConstrainedExpressionField` | Dirichlet / Neumann / Robin / initial / periodic | product over axes of `1 + #faces` | Box geometry; composes across axes and kinds, certified. For time-dependent periodic *spatial* axes, prefer `SpectralVectorField` / `basis="spectral"` (periodicity free in the Fourier base) |
| `MassFluxPotentialField` | compressible `rho u = curl Psi` | small | For variable-density flows |

All cages are duals across `omnibias.pinn.torch.cage` and
`omnibias.pinn.jax.cage`; cross-backend tests in
`packages/omnibias-pinn/tests/cross_backend/` assert bit-identical
behaviour.
