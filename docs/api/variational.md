# omnibias-variational

The **Least Action Principle** for omnibias. Hamilton's principle says the
physical trajectory makes the action stationary,

\[
S[q]=\int_{t_0}^{t_1} L(q,\dot q,t)\,dt,\qquad
\delta S=0 \;\Longrightarrow\;
\frac{d}{dt}\frac{\partial L}{\partial\dot q^i}-\frac{\partial L}{\partial q^i}=0 .
\]

It is built from the pieces omnibias already has: path / field derivatives come
from the closed-form `sigma^(n)` tower (via `omnibias-fields`), and the action
integral reuses the field quadrature surface. Everything ships as bit-identical
PyTorch and JAX twins.

## Two methods

Given a `Lagrangian` and an omnibias `FieldState` carrying the trajectory
`q(t)`:

- **Direct (Ritz):** minimize `action(state, lagrangian, rule=...)` with respect
  to the trajectory parameters using ordinary autograd (`.backward()` /
  `jax.grad`). The integrand's `q`, `qdot` are closed form.
- **Indirect (PINN):** drive `euler_lagrange_residual(state, lagrangian)` (or
  `euler_lagrange_loss`) to zero as a physics-informed loss.

```python
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as var

# L = 1/2 qdot^2 - 1/2 w^2 q^2  (a 1-DOF harmonic oscillator)
lag = Lagrangian(lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * (q**2).sum(-1), dof=("q",))

# `state` is any omnibias FieldState whose components include the dof "q": a neural
# field for training (below), or an analytic field for validation.
traj = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("t",), time_axis="t"),
    components=ComponentSpec(("q",)), hidden=32, base="tanh",
)
state = traj(torch.linspace(0.0, 1.0, 8, dtype=torch.float64)[:, None])

residual = var.euler_lagrange_residual(state, lag)   # (B, 1); ~0 on the true path
```

See the runnable, self-contained scripts under
[`docs/examples/`](https://github.com/derivon-ai/omnibias/tree/main/docs/examples):
`variational_harmonic_oscillator.py`, `variational_brachistochrone.py`,
`variational_geodesic.py`, `variational_klein_gordon.py`, and
`variational_learned_lagrangian.py` (a Lagrangian Neural Network).

## Theory note

`euler_lagrange_residual` expands the total time derivative with the chain rule,
so the outer `d/dt` is taken by the **closed-form** higher derivatives rather
than by differentiating through a black box:

\[
\text{EL}_i=\sum_j \frac{\partial^2 L}{\partial\dot q_i\partial q_j}\,\dot q_j
+\sum_j\frac{\partial^2 L}{\partial\dot q_i\partial\dot q_j}\,\ddot q_j
+\frac{\partial^2 L}{\partial\dot q_i\partial t}
-\frac{\partial L}{\partial q_i}.
\]

Here `qdot`, `qddot` are supplied by the sigma-tower (`derivative(..., order=1|2)`)
and the `L`-partials / Hessian in `(q, qdot)` come from backend autodiff of the
callable. For `L = 1/2 m qdot^2 - V(q)` this collapses to the familiar
`m qddot + V'(q)`. The classical field-theory path
(`field_euler_lagrange_residual`) is the direct analogue with `d_mu d_nu phi`
closed form.

## Functional derivative, first variation, and constraints

The **variational (functional) derivative** is the arbitrary-order Euler-Poisson
operator -- the generalisation of Euler-Lagrange to a Lagrangian
`L(q, q', ..., q^(n), t)` of any `order = n`:

\[
\frac{\delta S}{\delta q_i}
  =\sum_{k=0}^{n}(-1)^k\frac{d^k}{dt^k}\frac{\partial L}{\partial q_i^{(k)}} .
\]

`functional_derivative(state, lagrangian)` evaluates it, with the outer total
time derivatives `d^k/dt^k` riding on the **closed-form** trajectory derivatives
`q^(k)` (needed up to order `2n`, all from the sigma-tower) and the `L`-partials
by autodiff. The sign convention is
`functional_derivative == -euler_lagrange_residual`; for the first-order default
`euler_lagrange_residual` keeps its bit-identical fast path, and for `order >= 2`
it returns the Euler-Poisson residual (e.g. the Pais-Uhlenbeck oscillator or the
static Euler-Bernoulli beam, whose equation is `EI y'''' - rho`).

The **first (Gateaux) variation** `first_variation(state, lagrangian,
perturbation, *, rule)` is the exact weak pairing against a perturbation field
`eta` (itself an omnibias field on the same nodes, so its derivatives `eta^(k)`
are closed form):

\[
\delta S[q;\eta]=\int \sum_{k=0}^{n}\frac{\partial L}{\partial q^{(k)}}\cdot\eta^{(k)}\,dt .
\]

No integration by parts is performed, so boundary terms are retained --
`first_variation` is exactly `d/deps S[q + eps*eta]` at `eps = 0`, and it
vanishes on a solution for every endpoint-vanishing `eta` (Hamilton's principle).
The classical-field analogues are `field_functional_derivative`
(`delta S / delta phi = -field_euler_lagrange_residual`) and
`first_variation_density`.

**Constrained** stationarity comes in two forms. For a *holonomic* constraint
`g(q, t) = 0` (a `Constraint`), `constrained_euler_lagrange_residual` returns the
Lagrange-multiplier residual `euler_lagrange_residual - sum_a lambda_a dg_a/dq`
together with the constraint values (zero on a constrained solution -- e.g. a
bead on the unit circle). For an *isoperimetric* constraint `int g dt = C`,
`augmented_lagrangian(lagrangian, constraint, multiplier)` returns the augmented
Lagrangian `L' = L - lambda g` (the constraint `g` is itself a `Lagrangian`
integrand and may depend on `qdot`, e.g. arc length), whose Euler-Lagrange
extremals are the isoperimetric solutions -- e.g. the catenary
`y = lambda + a cosh(x/a)`.

### Honesty labels

- Trajectory / field derivatives (`qdot`, `qddot`, `q^(k)`, `eta^(k)`,
  `d_mu phi`, `d_mu d_nu phi`): **closed form** (the omnibias sigma-tower). This
  includes the outer `d^k/dt^k` in the Euler-Poisson operator and the constraint
  perturbation derivatives.
- Lagrangian / density / constraint partials (`dL/dq^(k)`, `dg/dq`, and the
  second partials): **autodiff** of the user callable (unless the callable is
  itself an omnibias field / jet).
- The action integral and the first-variation pairing: **quadrature**
  (Gauss-Legendre is exact to degree `2n-1`).
- The rigorous register (`omnibias.variational.verified`): outward-rounded
  **interval** arithmetic -- guaranteed enclosures over a stated local scope,
  never a global claim.

## Forward dynamics (Lagrangian Neural Networks)

The residual above *checks* the equations of motion on a supplied trajectory; the
**forward** direction *solves* them for the acceleration. Expanding the total
time derivative in Euler-Lagrange gives `M qddot = F` with

\[
M=\frac{\partial^2 L}{\partial\dot q\,\partial\dot q},\qquad
F=\frac{\partial L}{\partial q}
   -\frac{\partial^2 L}{\partial\dot q\,\partial q}\,\dot q
   -\frac{\partial^2 L}{\partial\dot q\,\partial t},
\]

so `acceleration(lag, q, qdot, t) = M^{-1} F` turns a Lagrangian into its
equations of motion. These are **array-level** ops on state samples (`q`, `qdot`
of shape `(B, n)`, `t` of `(B, 1)`) -- the natural interface for a **Lagrangian
Neural Network**: parametrize `L_theta` and minimize `lagrangian_dynamics_loss`
so its predicted acceleration matches observed data (energy-conserving by
construction).

```python
import torch
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as var

lag = Lagrangian(lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * (q**2).sum(-1), dof=("q",))
q, qd, t = torch.tensor([[0.3], [-1.0]]), torch.tensor([[1.0], [0.5]]), torch.zeros(2, 1)
a   = var.acceleration(lag, q, qd, t)          # == -q for this oscillator
M   = var.mass_matrix(lag, q, qd, t)           # == I
tau = var.inverse_dynamics(lag, q, qd, a, t)   # ~ 0: the force that realises `a`
```

The surface is `mass_matrix`, `generalized_force`, `acceleration`, `dynamics_rhs`
(the `(qdot, qddot)` ODE right-hand side for rollouts), `inverse_dynamics`
(`tau = M qddot - F`, robotics inverse dynamics -- identically the Euler-Lagrange
residual evaluated with an explicit `qddot`), and `predicted_acceleration(state,
lag)` (a `FieldState` wrapper). Forward dynamics is implemented for `order == 1`
Lagrangians (higher-order Euler-Poisson forward solves raise
`NotImplementedError`). The runnable `variational_learned_lagrangian.py` fits a
structured `L_theta = 1/2 a qdot^2 - V_theta(q)` to pendulum data and shows the
learned model conserves energy on a long rollout where a black-box acceleration
network drifts.

## Legendre transform and the Hamiltonian bridge

The Legendre transform sends `L(q, qdot, t)` to the phase-space Hamiltonian
`H(q, p, t)`. The conjugate momentum is `p = dL/dqdot`; inverting it for
`qdot(q, p, t)` (a Newton solve whose Jacobian is the mass matrix `M`) gives

\[
H(q,p,t)=p\cdot\dot q(p)-L\big(q,\dot q(p),t\big).
\]

`momentum`, `velocity_from_momentum` (Newton inversion), `legendre_transform`
(the true `H`), and `hamiltonian_from_lagrangian` (a `Hamiltonian` whose callable
is the Legendre transform) build the bridge; `canonical_equations(ham, q, p, t)`
returns Hamilton's flow `(qdot, pdot) = (dH/dp, -dH/dq)`. Unlike
`hamiltonian(state, lag)`, which evaluates the energy *along a supplied
trajectory*, `legendre_transform` is the genuine `H` on phase space, and its
canonical equations reproduce the Lagrangian forward dynamics.

### Honesty labels (forward dynamics & Legendre)

- `mass_matrix`, `generalized_force`, and the momentum `p = dL/dqdot`:
  **autodiff** of the user callable (same as the rest of the package).
- `acceleration` / `inverse_dynamics`: an **exact linear solve** of `M qddot = F`
  (torch/jax agree to `rtol=1e-12`; `M` must be invertible -- positive definite
  for a physical Lagrangian).
- `velocity_from_momentum` / `legendre_transform`: a **Newton solve** -- exact in
  one step for a velocity-quadratic `L`, convergent (numerical) for a general
  convex-in-velocity `L`. `canonical_equations`: autodiff of the `H` callable.
- `verified.acceleration_enclosure(force, *, mass)`: the rigorous scalar /
  constant-mass `qddot = F/m` enclosure (the general positive-definite matrix
  solve is a noted follow-up).

## Schemas

::: omnibias.variational
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - Lagrangian
        - LagrangianDensity
        - Constraint
        - Hamiltonian

## Ops (torch)

::: omnibias.variational.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## Rigorous enclosures (pure Python)

::: omnibias.variational.verified
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.variational.jax.ops`) is the bit-identical twin of
the torch surface. Every op is validated analytically and for torch/jax parity
(`allclose(rtol=1e-12, atol=1e-12)` in float64): the harmonic oscillator
(`action`, `EL ~ 0`, conserved energy), the free particle (conserved momentum),
the Klein-Gordon field (`EL == d'Alembertian` residual), the brachistochrone
(direct minimization recovers the cycloid), forward dynamics (`acceleration`,
`inverse_dynamics`, and the Legendre transform / `canonical_equations`), and --
through the optional `omnibias-geometry` bridge -- geodesics
(`euler_lagrange_residual` of the metric Lagrangian equals `geodesic_rhs` on the
sphere and hyperbolic plane).
