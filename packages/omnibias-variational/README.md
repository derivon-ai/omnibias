# omnibias-variational

**Status: Alpha (0.1.0a1).**

The Least Action Principle for omnibias. Hamilton's principle says the physical
trajectory makes the action stationary,

```
S[q] = integral L(q, qdot, t) dt,     delta S = 0   <=>   d/dt (dL/dqdot) - dL/dq = 0,
```

and this package builds that end-to-end on top of the `omnibias-fields`
substrate: the trajectory / field derivatives (`qdot`, `qddot`, `d_mu phi`,
`d_mu d_nu phi`) are the **closed-form** sigma-tower derivatives, the action is a
**quadrature** of the field, and the Lagrangian's own partials are **autodiff**
of the user callable.

## What's here (torch + jax twins)

- `action(state, lagrangian, *, rule)` -- the action `S = integral L dt`.
- `euler_lagrange_residual(state, lagrangian)` -- `d/dt(dL/dqdot) - dL/dq`
  (dispatches to the arbitrary-order Euler-Poisson operator for `order >= 2`).
- `functional_derivative(state, lagrangian)` -- the variational derivative
  `delta S/delta q` (Euler-Poisson; `== -euler_lagrange_residual`), for a
  `Lagrangian` of any `order` (Pais-Uhlenbeck, Euler-Bernoulli beam).
- `first_variation(state, lagrangian, perturbation, *, rule)` -- the exact weak
  Gateaux variation `delta S[q; eta]` (boundary terms retained).
- `conjugate_momentum`, `hamiltonian`, `energy`, `hamiltons_equations_residual`.
- **Forward dynamics (Lagrangian Neural Networks):** `mass_matrix`,
  `generalized_force`, `acceleration` (solve `M qddot = F` -- the equations of
  motion), `dynamics_rhs`, `inverse_dynamics`, `predicted_acceleration`, and
  `losses.lagrangian_dynamics_loss` (learn `L_theta` from observed accelerations).
- **Legendre / Hamiltonian bridge:** `momentum`, `velocity_from_momentum`,
  `legendre_transform` (the true `H(q, p, t)`), `hamiltonian_from_lagrangian`
  (a `Hamiltonian`), and `canonical_equations` (`qdot = dH/dp`, `pdot = -dH/dq`).
- `noether_charge(state, lagrangian, generator)` -- conserved charge of a symmetry.
- `action_density`, `field_euler_lagrange_residual`, `stress_energy_tensor`,
  `field_functional_derivative`, `first_variation_density` -- classical field
  theory from a `LagrangianDensity`.
- `constrained_euler_lagrange_residual(state, lagrangian, constraint, multipliers)`
  -- holonomic `g(q, t) = 0` via Lagrange multipliers (a `Constraint`); and
  `augmented_lagrangian(lagrangian, constraint, multiplier)` for isoperimetric
  `int g dt = C` (bead on a circle, catenary).
- `metric_lagrangian(manifold)`, `geodesic_action` -- geodesics as least action
  (optional `omnibias-geometry` bridge; the metric Lagrangian's Euler-Lagrange
  residual reproduces `omnibias.geometry` `geodesic_rhs`).
- `discrete_euler_lagrange_residual`, `stormer_verlet_step` -- symplectic /
  variational integrators.
- `losses.action_minimization_loss`, `losses.euler_lagrange_loss` -- drop-in PINN
  losses for the direct and indirect methods.
- `omnibias.variational.verified` -- rigorous `action_enclosure` /
  `euler_lagrange_enclosure` / `acceleration_enclosure` via
  `omnibias.core.verified` (pure Python).

```python
import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as var

# 1-D harmonic oscillator L = 1/2 qdot^2 - 1/2 w^2 q^2
lag = Lagrangian(lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * (q**2).sum(-1),
                 dof=("q",), time_axis="t")

# any omnibias-fields FieldState carrying the dof "q"; evaluate it *on the
# quadrature nodes* so the action integral and the field agree pointwise
traj = OneLayerVectorField(coordinate_spec=CoordinateSpec(("t",), time_axis="t"),
                           components=ComponentSpec(("q",)), hidden=32, base="tanh")
rule = gauss_legendre(((0.0, 1.0),), 16)
state = traj(torch.as_tensor(rule.nodes, dtype=torch.float64))

S = var.action(state, lag, rule=rule)              # scalar action
res = var.euler_lagrange_residual(state, lag)      # (B, 1), ~0 on a solution
```

## Honesty labels

- Trajectory / field derivatives (including the outer `d^k/dt^k` of the
  Euler-Poisson operator and the perturbation derivatives): **closed form** (the
  omnibias sigma-tower).
- Lagrangian / constraint partials `dL/dq^(k)`, `dg/dq`, second partials:
  **autodiff** of the user callable (unless it is itself an omnibias field/jet).
- Action integral and the first-variation pairing: **quadrature** (Gauss-Legendre
  exact to degree `2n-1`).
- Forward dynamics: `mass_matrix` / `generalized_force` are **autodiff** of `L`;
  `acceleration` is an **exact linear solve** of `M qddot = F` (order-1 only).
  `velocity_from_momentum` / `legendre_transform` use a **Newton solve** (exact in
  one step for velocity-quadratic `L`, numerical otherwise).
- Metric-derived quantities in the geodesic bridge: inherit the
  `omnibias-geometry` label (autodiff of the analytic metric).
- `omnibias.variational.verified`: rigorous outward-rounded enclosures.

## Validation

Harmonic oscillator (`EL ~ 0`, energy conserved), free particle (momentum
conserved via Noether), Klein-Gordon (`field EL == wave / d'Alembertian`),
geodesics (metric-Lagrangian `EL == geodesic_rhs` on the sphere / hyperbolic
plane), and a symplectic integrator with bounded long-horizon energy drift.
Higher-order Euler-Poisson (Pais-Uhlenbeck, Euler-Bernoulli beam), the first
variation (finite-difference cross-check and `delta S = 0` on solutions), and
constraints (bead on the unit circle, catenary) are validated analytically.
Forward dynamics (`acceleration == -w^2 q`, `inverse_dynamics` inverts it and
equals the Euler-Lagrange residual on a trajectory) and the Legendre transform
(momentum round trip, `canonical_equations` reproduce the forward dynamics, and
the involution recovers `L`) are checked against closed forms, and a small
Lagrangian Neural Network recovers the oscillator frequency. Torch and jax agree
to `rtol=1e-12`.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
