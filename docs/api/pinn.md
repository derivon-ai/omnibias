# omnibias-pinn

Physics-informed neural networks (PINNs) with **closed-form n-th
derivative operators**, built on top of `omnibias-core`. Cross-backend
(PyTorch + JAX) typed fields, ops, hard-conservation cages, prebuilt
PDE residuals, and diagnostics.

The package surfaces three layers:

| Layer | Purpose | Example |
| --- | --- | --- |
| **Fields** (`fields/`) | Structural backends with closed-form derivatives. | `OneLayerVectorField`, `SpectralVectorField`, `ChebyshevVectorField` |
| **Ops** (`ops/`) | User-facing operator surface (functional kernel). | `derivative`, `gradient`, `laplacian`, `biharmonic`, `curl`, `advection` |
| **Cage** (`cage/`) | Strict-conservation layers (hard physical constraints). | `StreamfunctionField`, `VectorPotentialField`, `HelmholtzProjectionField` |

Plus equation-aware modules:

* **Losses** (`losses/`): Sobolev preconditioning, Wang-Perdikaris
  causal weighting, NTK rebalance, entropy-consistent residual, and
  **asymptotic / removable boundary conditions** (`asymptotic_ratio`,
  `asymptotic_bc_loss`, `far_field_decay_loss`) -- the differentiable jet
  `lim` operator surfaced as trainable losses. Plus the *stateful* half:
  `LossWeighter` and friends (EMA + update cadence over the per-term weights),
  `self_adaptive_loss` (trained pointwise weights), and `TimeMarcher` (causal
  time-window marching).
* **Equations** (`equations/`): prebuilt PDE residuals
  (NavierStokes, Burgers, Heat, KuramotoSivashinsky, CahnHilliard,
  Biharmonic) returning :class:`NamedTuple` outputs with diagnostics, plus the
  **nonlocal** ones -- CordobaCordobaFontelos (Hilbert transform) and the
  Fredholm / Volterra integral equations.
* **Diagnostics** (`diagnostics/`): backend-agnostic
  ``relative_l2_per_time``, ``forecast_horizon``, ``spectral_fidelity``,
  plus field-level ``derivative_stability`` and ``autograd_phase_check``.

## Alpha submodules

Four alpha submodules of Beta `omnibias-pinn` host the gated research surface
(not separate distributions — see [`packages.md`](../packages.md)):

| Submodule | Role | Docs |
| --- | --- | --- |
| `omnibias.pinn.solver` | Mesh-free PDE solver, stiff ETDRK4 / Rosenbrock, least-squares collocation | [pinn-solver.md](pinn-solver.md) |
| `omnibias.pinn.train` | Causal `march_solve`, causality / trivial-solution diagnostics | [pinn-train.md](pinn-train.md) |
| `omnibias.pinn.domain` | SDF / R-function geometry + hard curved BCs | [pinn-domain.md](pinn-domain.md) |
| `omnibias.pinn.operator` | DeepONet / FNO + multi-head conditioning | [pinn-operator.md](pinn-operator.md) |

Four-gap acceptance matrix (smoke vs `--full`):
[`benchmarks/pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).

## Pythonic DSL

The canonical user surface is **attribute-based** (Option 1 in the
design memo). Every :class:`FieldState` exposes per-component views
that route to the underlying functional ops:

```python
import torch

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields import SpectralVectorField

field = SpectralVectorField(
    coordinate_spec=CoordinateSpec(axes=("x", "y", "z", "t"), time_axis="t"),
    components=ComponentSpec(("u", "v", "w"), groups={"velocity": ("u", "v", "w")}),
    K=8, time_hidden=32, time_depth=1, activation="tanh",
)

coords = torch.rand(16, 4, dtype=torch.float64) * 6.28
state = field(coords)            # FieldState
state.u.dt                       # ∂u/∂t
state.u.dx                       # ∂u/∂x
state.u.lap                      # Δu
state.u.biharm                   # Δ²u
state.velocity.div               # ∇·u
state.velocity.curl              # ∇×u  (3D)
state.velocity.advect()          # (u·∇) u  -- a method, unlike the properties above
```

The same operations are also available as functions
(`omnibias.pinn.torch.ops.derivative(state, "u", axis="t")` etc.) and
the DSL is a thin wrapper -- both routes share a single fastpath
implementation per backend.

## Top-level API

Both the PyTorch and JAX backends expose the same public surface
under their own root namespaces:

* `omnibias.pinn.torch.{fields, ops, cage, losses, equations, diagnostics}`
* `omnibias.pinn.jax.{fields,   ops, cage, losses, equations, diagnostics}`

The shared *schemas* (`CoordinateSpec`, `ComponentSpec`, `FieldState`)
live in `omnibias.pinn._core`.

## Fields

Every field turns `coords` into a `FieldState`; what distinguishes them is *how*
the derivative tower is evaluated.

| Field | Architecture | Derivative path |
| --- | --- | --- |
| `OneLayerVectorField` | one hidden layer | closed-form single-layer `sigma`-tower contraction |
| `JetMLPVectorField` | deep MLP, any depth | closed-form multivariate jet (`mlp_jet_mv`) |
| `FourierFeatureVectorField` | random Fourier front end + deep body | closed-form multivariate jet |
| `AdaptiveJetMLPVectorField` | deep MLP with a trainable activation slope | closed-form multivariate jet |
| `MscaleVectorField` | MscaleDNN band mixture | closed-form multivariate jet, one per band |
| `AttentionVectorField` | deep encoder + softmax mixture over a trainable memory | closed-form multivariate jet (`jet_attention`) |
| `SpectralVectorField` / `ChebyshevVectorField` | basis expansion | closed-form basis derivatives |

### Readout-independence invariant

The frozen-feature linear solver (`solve_least_squares`) builds a collocation
plan once and reuses the same `FieldState` caches while sweeping the readout
weights. That is sound only when every cached quantity is independent of those
weights and every op re-reads the readout live.

Fields that honour the contract set the class attribute named by
`omnibias.fields.READOUT_INDEPENDENT_ATTR` (`"_omnibias_readout_independent"`)
to `True`. Affine cages (`ConstrainedExpressionField`, streamfunction / flux-form
/ hard-boundary, …) recurse through `base` and declare when the base does.
Fields that are *nonlinear* in the readout (`IntegralConservationField`,
`NormConservationField`) leave the marker unset; the driver refuses them with
`ReadoutDependentError` rather than silently miscomputing -- use
`solve_optimize` instead.

Spectral / Chebyshev / jet caches store the readout-*independent* factor (hidden
temporal features, or the jet through `layers[:-1]`) and apply the final affine
readout per call, so a state remains coherent under a readout sweep.

### Deep and Fourier-feature fields (`jet_mlp`)

`JetMLPVectorField` and `FourierFeatureVectorField` lift the arbitrary-depth
architectures in `omnibias.torch.architectures.pinn` onto the field substrate, so
a deep network reaches the full operator surface, the attribute DSL, the
conservation cages and the prebuilt PDE residuals. `make_siren_vector_field`
builds a SIREN the same way.

Derivatives are **closed form, not autodiff**: one call to
`omnibias.torch.jet_mv.mlp_jet_mv` yields every mixed partial `D^alpha u` up to
total order `N` as `alpha! c_alpha`, at any depth. There is no
`torch.autograd.grad` in the differential operator (contrast `PartitionedField`
below, which must fall back to the autodiff product rule).

The *hidden* jet (layers before the final affine readout) is memoised in
`FieldState.extra`, and the live readout is applied on every call, so a whole
second-order residual -- gradient, Laplacian, Hessian, divergence, mixed
partials -- costs **one** jet evaluation (a value-only term takes the plain
forward path and never pays for a jet). `jet_order` is the planning knob: the
jet is built at `max(requested_order, jet_order)`, so set it to the highest
derivative order in your residual. `polylaplacian` reads `Delta^k` off a single
order-`2k` jet via the multinomial expansion of `(sum_i d_i^2)^k`, so its cost
is independent of `k`.

`FourierFeatureVectorField` is the spectral-bias cure. The encoding
`gamma(x) = [cos(B x), sin(B x)]` is a single `sin` layer (because
`cos(z) = sin(z + pi/2)`), and `sin^(n)(z) = sin(z + n pi/2)` is exact at every
order -- so breaking spectral bias costs nothing in the derivative tower. Passing
a *sequence* of `frequency_scale` values concatenates bands into a multi-scale
encoding, and `trainable_features=True` makes the frequencies learnable.

See [`docs/examples/pinn_fourier_features.py`](https://github.com/derivon/omnibias/blob/main/docs/examples/pinn_fourier_features.py)
for a runnable two-scale Poisson comparison.

### Multi-scale fields: the frequency knob inside the network

A Fourier encoding widens the *input basis* with a fixed random draw. The two
fields below instead put the frequency knob inside the network, and both stay on
the same `jet_mlp` path -- same hidden-jet cache, same operator surface, same
exactness.

`AdaptiveJetMLPVectorField` gives every hidden layer a **trainable slope** `a` and
computes `sigma(n a z)` (Jagtap et al. 2020), so the network tunes its own
frequency content instead of inheriting the fixed spectrum of a plain `tanh` MLP.
`n` is a fixed amplification factor: it scales the gradient reaching `a` by
exactly `n` without changing the initial function. `granularity="neuron"` gives
one slope per hidden unit instead of one per layer.

This is the construction that most obviously *ought* to break closed-form
differentiation, since the frequency moves every optimiser step. It does not: the
slope is the temperature of `omnibias.core.spec.tempered`, so the layer is a real
`ActivationSpec` whose tower

    d^k/dz^k sigma(n a z) = (n a)^k sigma^(k)(n a z)

is inherited from the base activation at every order, and the kernel reads `a` at
call time. `field.slopes()` reports the current effective slopes -- values well
above 1 mean the solution carries more high-frequency content than the base
activation supplies on its own.

`MscaleVectorField` is the MscaleDNN band mixture `u(x) = sum_j f_j(alpha_j x)`
(Liu, Cai & Xu 2020). Each subnetwork sees the input pre-scaled by its band factor,
so a feature oscillating at frequency `k` looks like `k / alpha_j` to band `j`: the
high bands turn the hard, high-frequency part of the target into the easy,
low-frequency part a plain MLP learns quickly. Scaling the input is the same map as
scaling the first weight matrix, so each band is an ordinary layer chain and the
mixture's jet is the *sum* of the per-band jets -- one exact `mlp_jet_mv` call per
band. `hidden` is the total width, split evenly across bands, so the mixture costs
about as much as a single MLP of that width. `adaptive=True` puts a trainable slope
inside every band as well.

### Choosing the bands from data

Band scales are normally guessed as the ladder `1, 2, 4, 8` (`geometric_bands`).
`suggest_frequency_bands` measures them instead: it reads the power spectrum of a
sampled field, splits it into equal-**energy** segments, and returns each segment's
centroid converted to cycles per unit length (`s = j / L`). Equal-energy rather than
equal-width splitting is what makes it useful -- a spectrum with a strong low peak
and a weak high tail gets one band on each, where equal-width bins would spend every
band on the peak.

```python
import numpy as np
from omnibias.pinn.torch.diagnostics import suggest_frequency_bands

x = np.linspace(0.0, 1.0, 256, endpoint=False)
u_sample = np.sin(2 * np.pi * 3 * x) + 0.5 * np.sin(2 * np.pi * 20 * x)
bands = suggest_frequency_bands(u_sample[None, :], L=1.0, n_bands=2)
print(bands)  # (3.0, 20.0) -- the two tones, recovered
```

The tuple goes straight into `MscaleVectorField(scales=bands)` or
`FourierFeatureVectorField(frequency_scale=bands)`.

See [`docs/examples/pinn_multiscale_feedback.py`](https://github.com/derivon/omnibias/blob/main/docs/examples/pinn_multiscale_feedback.py)
for the loop end to end: on a target whose two modes are twenty times apart the
guessed ladder is no better than a plain MLP, the measured bands find the
oscillation, and the adaptive slopes -- which need no spectrum at all -- do best
while reporting the frequency they had to reach.

### Non-local fields: attention with closed-form `d/dx`

Every field above is *local*: `u(x)` is a chain of elementwise activations over
affine maps of `x`, so its jet is the plain Faà di Bruno recursion.
`AttentionVectorField` is not. It routes the coordinates through a softmax
mixture over a trainable memory `(K, V)`,

    u(x) = W_o [ softmax(beta q(x) K^T) V + q(x) ] + b_o,    q(x) = MLP(x)

so every output couples to every memory slot through the shared denominator.
Read as a PINN construction it is a **learned soft partition of the domain**
with one local model `V_j` per region -- a differentiable, globally-coupled
relative of a domain decomposition, and the reason this field is worth having
next to the local ones. `field.attention_weights(coords)` returns the per-point
partition of unity, which is what makes it interpretable.

`omnibias.hopfield` already differentiates this block's log-sum-exp core in
closed form -- but with respect to the **scores**, which is the wrong variable
for a PDE. The missing `d/dx` is supplied by four jet primitives added to both
backends:

| Primitive | What it composes | Tower |
| --- | --- | --- |
| `jet_exp` | `exp(u)` | the registered `exp` activation (`exp^(k) = exp`) |
| `jet_reciprocal` | `1 / u` | `(-1)^k k! u^(-(k+1))` |
| `jet_softmax` | `softmax(s)` over the last axis | the two above + a sum + a jet product |
| `jet_attention` | `softmax(beta q K^T) V` | `jet_softmax` between two affine maps |

`compose_jet_mv` alone reaches only *elementwise* maps, so this genuinely widens
the closed-form class to rational and coupled ones. The block's value agrees with
`omnibias.hopfield.torch.ops.attention` exactly; what is new is that
`D^alpha u(x)` at arbitrary order comes out of the same single jet, with no
nested autodiff. Note the temperature collapse this shares with the rest of the
repo: as `beta -> inf` the mixture hardens into a crisp assignment -- the
*feasibility* sense of collapse, not the founding `delta -> 0` one.

```python
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch import equations, ops
from omnibias.pinn.torch.fields import build_attention_vector_field

field = build_attention_vector_field(
    coordinate_spec=CoordinateSpec(("x", "t")),
    components=ComponentSpec(("u",)),
    hidden=16,
    depth=2,
    memory=8,      # how many regions the field may specialise to
    beta=1.0,      # softmax sharpness
    jet_order=2,   # the residual below is second order
    seed=0,
)
coords = torch.randn(32, 2, dtype=torch.float64)
state = field(coords)

residual = equations.burgers(state, nu=0.01).residual   # one jet, closed form
weights = field.attention_weights(coords)               # (32, 8), rows sum to 1
print(residual.shape, weights.shape, float(weights.sum(-1).mean().detach()))
```

The `residual=True` skip (on by default) keeps a local path alongside the
non-local one; without it the readout is confined to the convex hull of the
value slots. The JAX twin is `make_attention_vector_field`, bit-identical to
float64 round-off.

::: omnibias.pinn.torch.fields
    options:
      show_root_heading: false
      heading_level: 3

## Ops

::: omnibias.pinn.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## Cage

A cage enforces a constraint **by construction** rather than by penalty, so the
corresponding loss term disappears from the objective entirely. Six are shipped:

| Cage | Constraint | How it holds |
| --- | --- | --- |
| `StreamfunctionField` | `div u = 0` in 2-D | `u = (d_y psi, -d_x psi)` |
| `VectorPotentialField` | `div u = 0` in 3-D | `u = curl A` |
| `FluxFormField` | `div G = 0` in any dimension, space *or* space-time | `G^i = sum_j d_j A^ij`, `A` antisymmetric |
| `IntegralConservationField` | `int sum_c u_c^p dx = C` | global rescaling by `lambda = (C / I)^(1/p)` |
| `HardBoundaryField` | `u = g` on a boundary of any shape | `u = g + d * f` with a user distance function `d` |
| `ConstrainedExpressionField` | linear Dirichlet / Neumann / Robin / initial / periodic conditions on a box | `u = g + sum_k phi_k (t_k - C_k[g])`, switching functions from a certified support matrix |

The last two both impose boundary data exactly; pick by geometry. `HardBoundaryField`
takes any shape but only Dirichlet data, and does not compose -- wrapping it around a
derivative condition breaks whichever constraint ends up inside.
`ConstrainedExpressionField` needs an axis-aligned box, and in exchange covers value,
normal-derivative, Robin, initial and periodic conditions uniformly, composes exactly
across axes and kinds, and is better behaved at corners (its switching functions are
polynomials, so nothing blows up where two faces meet -- the known failure mode of
distance-function ansaetze under second-order operators).

### Divergence form: the finite-volume cage

A conservation law `d_t rho + div F = 0` says one space-time vector
`G = (rho, F)` is divergence-free. For an antisymmetric potential
`A^ij = -A^ji`, setting `G^i = sum_j d_j A^ij` makes `div G = sum_ij d_i d_j A^ij`
vanish **identically**: `d_i d_j` is symmetric in `(i, j)` while `A^ij` is
antisymmetric, so the double sum cancels term by term. No quadrature, no
tolerance.

`FluxFormField` is that statement in any dimension, and it subsumes the two
incompressibility cages exactly: in 2-D the single potential is the
streamfunction, in 3-D the three potentials are the vector potential and
`G = curl A`. What it unlocks is the case neither covers -- letting `t` be one of
the axes, so the divergence-free object is a space-time flux and the identity is
a *conservation law* rather than incompressibility. On any control volume the
divergence theorem then gives the cell balance
`d/dt int_V rho = -oint_dV F . n` for **every** volume simultaneously.

Adding a residual penalty for `div G` on top of this cage is not merely
redundant but actively harmful: it contributes only round-off-scale gradient
noise.

### Conserved integrals: the rescaling cage

A conserved *integral* couples the whole domain, so no pointwise rearrangement
can enforce it. The one lever that does is a global rescaling, which works
whenever the density is homogeneous of degree `p`:

    I[lambda u] = lambda^p I[u]   =>   lambda = (C / I[u])^(1/p)

`IntegralConservationField` is that cage, with `degree=1` for a mass / charge /
probability density (`lambda = C / I`) and `degree=2` for a squared `L^2` norm
(`lambda = sqrt(C / I)`). The `degree=2`, two-component case is exactly
`omnibias.qpinn`'s `NormConservationField`; this generalises away both the
hard-wired `|psi|^2` density and the hand-rolled quadrature -- the rule is a
`QuadratureSpec`, so the same cage works on a Gauss-Legendre box in any
dimension, a Gauss-Hermite weight on an unbounded domain, or a seeded
Monte-Carlo sample.

Because `lambda` is one scalar with no `x` dependence, every derivative is
scaled by the same factor and the closed-form tower survives intact:
`D^alpha u = lambda D^alpha(u~)`.

**Honesty: the constraint holds to quadrature accuracy, not to machine
precision.** What is exact is `sum_q w_q rho(u(x_q)) == C`; the continuum
integral differs by the rule's own error. Use a rule that resolves the field and
check it on a finer one. (`FluxFormField`, by contrast, is exact pointwise.)

```python
import torch
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch.cage import IntegralConservationField
from omnibias.pinn.torch.fields import build_jet_mlp_vector_field

bounds = ((-4.0, 4.0),)
inner = build_jet_mlp_vector_field(
    coordinate_spec=CoordinateSpec(("x",), domain=bounds),
    components=ComponentSpec(("psi_re", "psi_im")),
    hidden=16,
    depth=2,
    seed=0,
)
caged = IntegralConservationField(
    base=inner,
    rule=gauss_legendre(bounds, 128),
    conserved=("psi_re", "psi_im"),
    total=1.0,
    degree=2,           # int (psi_re^2 + psi_im^2) dx = 1
    dtype=torch.float64,
)
state = caged(torch.linspace(-3.0, 3.0, 16, dtype=torch.float64).reshape(-1, 1))
print(round(float(caged.integral(state).detach()), 12))  # 1.0, by construction
```

### Hard boundary / initial conditions: the constrained-expression cage

For linear conditions `C_k[u] = t_k` on an axis-aligned box, pick support
functions `s_j`, form the **support matrix** `M_kj = C_k[s_j]`, and define
switching functions `phi_i = sum_j (M^-1)_ji s_j`, which satisfy
`C_k[phi_i] = delta_ki`. Then

    u = g + sum_k phi_k (t_k - C_k[g])

satisfies every condition for **any** free function `g` -- this is the Theory of
Functional Connections (Mortari 2017; Leake and Mortari, *Mathematics*
8(8):1303, 2020). Applying it once per axis embeds conditions on several axes at
once; the cross terms that make the corners come out right are generated by the
recursion rather than special-cased.

`ConstrainedExpressionField` is that cage. Dirichlet, Neumann, Robin and an
initial value or velocity are all the same `LinearConstraint` type with
different terms, which is why there is no per-kind branching. **Periodicity is
the same type too**, as a *relative* constraint `d^n u(hi) - d^n u(lo) = 0`,
because a linear functional may reference more than one point. On the cage /
`HardCondition` side, `periodic(lo, hi, order=n)` matches one derivative order
at a time. On the solver side, a periodic `BoundaryCondition` carries
`periodic_orders` (default `PERIODIC_ORDERS = (0, 1, 2)`): value, slope *and*
second derivative across the seam. The default rose from `(0, 1)` after a C¹-seam
sweep under a second-order operator -- matching only value and slope left `u''`
free to jump.

What that buys is matching **at the declared orders and nothing above**. The
first unmatched order still jumps across the seam, by roughly the magnitude of
that derivative itself, and both cage test suites pin it so the claim cannot
drift from "exact at three orders" to "smooth". The sweep stops at `(0, 1, 2)`
because the fourth order gains only 1.30x against the third's 3.34x on a smooth
manufactured solution -- which would reward extra orders indefinitely -- while a
higher default would over-smooth seams near steep gradients. If you need a seam
closed at *every* order, that is what the spectral basis below gives you. See
[benchmarks](../benchmarks.md#hard-vs-soft-boundary-initial-conditions).

### Periodic domains: spectral basis vs periodic cage

Two routes close a periodic seam, and they are not substitutes:

| Route | When | How |
| --- | --- | --- |
| `basis="spectral"` on `build_field` / `solve_least_squares` | Time-dependent problem; spatial axes are periodic | `SpectralVectorField` -- Fourier modes make spatial periodicity free in the ansatz; no periodic BC rows required for that axis |
| Periodic cage / BC | Steady or MLP ansatz; any axis you want matched by algebra | `ConstrainedExpressionField` with `periodic(...)` / a periodic `BoundaryCondition`, absorbed by `hard_conditions="auto"` |

`basis="spectral"` needs a time axis (the spectral field is a space-time Fourier
ansatz). For a steady Poisson seam, use the cage. For a time-dependent periodic
heat / Burgers / RD problem you can pick either: free periodicity in the Fourier
base, or an MLP wrapped in the constrained-expression cage.

The six problem builders (`poisson`, `heat`, `wave`, `burgers`,
`reaction_diffusion`, `advection_diffusion`) accept `periodic_boundary: bool =
False`. When `True`, each appends one periodic `BoundaryCondition` per component
per periodic *spatial* axis (after existing BCs, so absorbed indices stay
stable). The default stays off: measured on manufactured periodic Burgers and
reaction-diffusion, emit helped RD but hurt Burgers' interior fit, so
auto-emit is opt-in. Hand-built `System`s still need an explicit periodic BC.

Cost is one base evaluation per *distinct combination* of projection points
across the constrained axes -- the product over axes of `1 + #projection
points`, reported by `projection_cost`. A face carrying both a value and a slope
costs one, but a second constrained axis multiplies rather than adds. That is
the price of exact corners, and it is why absorbing every face of a 3-D box is a
deliberate choice.

Two preconditions are checked rather than assumed:

* **The support matrix must be invertible.** `certify_support_matrix` builds `M`
  entrywise in outward-rounded interval arithmetic, encloses
  `lambda_min(M^T M) > 0` (the Gram, because `M` is not symmetric), widens the
  enclosure by the Weyl bound so the statement is about the *exact* Gram rather
  than its float image, and seals a hash-verifiable certificate. A linearly
  dependent condition set is **refused**, not silently approximated. It is a
  finite rational obligation, so it is in scope for the Lean kernel.
* **Data on different axes must agree where those axes meet.** Construction
  refuses data that does not, naming the two conditions that clash; the gate
  covers every *pair* of axes rather than consecutive ones, on a deterministic
  lattice so a refusal is reproducible across backends. `compatibility_residual`
  keeps the number visible: order one when the clash is real, round-off when it
  is not. This is a statement about the data -- physically, the initial state has
  to satisfy the boundary condition at `t0` -- and no ansatz can repair it.

The cage is **closed form in the network**: every value and derivative of `g` it
needs, including at the projected face points, comes from the sigma-tower.
Autodiff appears only when a *user-supplied target callable* has to be
differentiated along another axis; constant targets need none.

```python
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.constrained import HardCondition, derivative_at, dirichlet, neumann
from omnibias.pinn.torch.cage import ConstrainedExpressionField
from omnibias.pinn.torch.fields import OneLayerVectorField

torch.manual_seed(0)
box = ((0.0, 1.0), (0.0, 1.0))
free = OneLayerVectorField(
    coordinate_spec=CoordinateSpec(("t", "x"), domain=box, time_axis="t"),
    components=ComponentSpec(("u",)),
    hidden=16,
    base="tanh",
    dtype=torch.float64,
)
hard = ConstrainedExpressionField(
    base=free,
    conditions=[
        HardCondition("u", 1, dirichlet(0.0), 0.0),          # u(t, 0) = 0
        HardCondition("u", 1, neumann(1.0), 0.0),            # u_x(t, 1) = 0
        # Initial state x(2 - x): zero at x = 0 and flat at x = 1, so it agrees
        # with the two conditions above where the axes meet.
        HardCondition("u", 0, dirichlet(0.0), lambda c: c[:, 1] * (2.0 - c[:, 1])),
        HardCondition("u", 0, derivative_at(0.0, 1), 0.0),   # u_t(0, x) = 0
    ],
)
face = torch.rand(8, 2, dtype=torch.float64)
face[:, 1] = 0.0                                   # the x = 0 face
state = hard(face)
print(float(state.ops.value(state, "u").detach().abs().max()) < 1e-14)  # True
```

Nothing above was trained: the conditions are an algebraic identity, so they
hold for every parameter value.

::: omnibias.pinn.torch.cage
    options:
      show_root_heading: false
      heading_level: 3

## Losses

### Adaptive weighting: the EMA and the cadence

A multi-term loss `L = sum_k lambda_k L_k` trains badly with fixed `lambda_k`,
because the terms' gradients differ by orders of magnitude -- the *gradient
pathology*. Every published cure has the same shape: measure something, turn it
into a target `lhat_k`, and smooth it with an EMA. Only the measurement differs,
so only the measurement is written per backend.

`LossWeighter` (in `omnibias.pinn._core.weighting`, re-exported from both
backends' `losses`) owns the rest: the EMA `lambda <- alpha lambda + (1 - alpha)
lhat`, the update *cadence* (each estimate costs an extra backward pass per term,
so refreshing every step is usually waste), the clamping, and `combine()`. It
holds host-side floats, never tensors, which is why the torch and jax weights are
bit-identical by construction rather than by test.

| Weighter | Target | Measure with |
| --- | --- | --- |
| `GradNormWeighter` | `max|dL_r/dtheta| / mean|dL_k/dtheta|` (Wang, Teng & Perdikaris 2021) | `grad_stats` |
| `NTKWeighter` | `exp(mean_j log T_j - log T_k)` -- the geometric-mean NTK balance | `ntk_trace_stats` |
| `ConstantWeighter` | fixed; the ablation baseline | -- |

```python
from omnibias.pinn.torch.losses import GradNormWeighter, grad_stats

weighter = GradNormWeighter(["pde", "bc"], reference="pde", alpha=0.9, every=10)
# ... inside the training loop, with terms = {"pde": ..., "bc": ...}:
#   weighter.update(grad_stats(terms, field.parameters()))
#   weighter.combine(terms).backward()
```

`GradNormWeighter` is the *annealing* variant: it compares the reference's
largest gradient entry against each other term's average, which is deliberately
more aggressive on a stiff residual than the L2-norm ratio of
`omnibias.torch.optim.GradNormBalancer`. The reference term's own weight stays
pinned at 1, so the weights say what they mean.

### Self-adaptive pointwise weights

`self_adaptive_loss` and `SelfAdaptiveWeights` are the pointwise counterpart:
one weight *per collocation point*, `L = mean_k m(lambda_k) r_k^2`, minimised
over the network and **maximised** over `lambda` (McClenny & Braga-Neto 2020).
The maximisation is the mechanism -- a point the network fits badly grows its own
weight, so the optimiser is pulled toward shock fronts and boundary layers
instead of averaging them away.

`ascent=True` (the default) reverses the gradient reaching the mask, so a single
`loss.backward()` and a single optimiser holding both `theta` and `lambda`
performs the whole minimax. That is exact, not an approximation: `2 x_detached -
x` is exactly `x` in IEEE-754 and differentiates to `-1`, and since Adam's update
is `m / sqrt(v)`, flipping the gradient's sign flips the step's sign, which is
what `maximize=True` does.

The mask `m` is an omnibias activation (`"sigmoid"`, `"softplus"`, any
`ActivationSpec`) -- so it comes from the shared dictionary and is bit-identical
across backends -- or `"identity"` / `"square"` for the paper's polynomial masks.
A bounded mask caps how far one point can dominate; `"identity"` does not.

### Causal time marching

`causal_residual_loss` implements the Wang-Perdikaris weight
`w_i = exp(-eps sum_{j<i} L_j)`, which stops a PINN fitting late times before
early ones. On its own that is half the recipe: it reweights whatever points it
is handed, over the whole interval, for the whole run. The other half is to solve
a short window and *march*, which is `TimeWindowSchedule` and `TimeMarcher`
(`omnibias.pinn._core.marching`, pure numpy, re-exported by both backends).

* `TimeWindowSchedule` -- window bounds (with optional `overlap` so window `k+1`
  re-fits the seam), the time bins causality is enforced at, the annealed
  sharpness `epsilon_at(k) = epsilon * growth^k`, and the advance criterion
  `is_converged`, which is Wang et al.'s `min_i w_i >= tolerance` rule; because
  the weights are non-increasing, the last bin's weight measures how much of the
  window has unlocked.
* `window_points` -- collocation stratified *by time bin*, returned already
  shaped `(n_bins, per_bin, D)`. Uniform sampling would leave the bin counts to
  chance, and an empty bin makes the cumulative causal weight meaningless.
* `TimeMarcher` -- the driver, including the **warm start**: `handoff_points()`
  are the next window's opening slice, and the values the trained field takes
  there become that window's initial condition, so window `k+1` starts from a
  real state rather than from noise.

The marcher owns no tensors and no optimiser; it answers *which points, which
epsilon, may I advance yet*, so it drives a torch loop, a jax loop, or the
solver.

Two things worth knowing before you reach for it, both measured rather than
asserted:

* **Scale `epsilon` to the residual you actually have.** The causal weights are
  `exp(-eps sum_j L_j)`, a function of the residual's *absolute* magnitude, not
  its shape. Non-dimensionalising a residual -- dividing `u_t - rho u(1-u)`
  through by `rho`, say -- shrinks the per-bin losses by `rho^2` and quietly
  drives every weight to 1, switching the causal filter off while leaving the
  code looking correct. `causal_residual_loss(..., return_weights=True)` and a
  glance at `weights.min()` is the check.
* **Marching is not a free win -- and not a free loss.** Regime decides.
  On linear heat with a hard IC/BC cage
  ([`causal_marching.json`](../benchmarks/causal_marching.json) family
  `heat`), `whole_interval` is best (median rel-L2 ≈ 9.4e-3, skill ≈ 0.9999);
  marching pays budget fragmentation and cannot improve on an already-easy
  problem. On Krishnapriyan's stiff reaction
  ``u_t = rho u(1-u)`` at ``rho = 12`` (family `reaction`, soft IC weight 10),
  whole-interval collapses (median rel-L2 ≈ 0.99) and gated
  `causal_marching` wins (≈ 8.4e-2). Supply the IC as `ic_fn` on the
  marcher's own slice points -- a linspace-ordered `ic_values` vector
  contaminates the seam metric. Advertised equal step budgets may be
  exceeded by gated retries; read `total_steps_*` in the summary. Full
  matrix: [`pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).

See
[`docs/examples/pinn_causal_marching.py`](https://github.com/derivon/omnibias/blob/main/docs/examples/pinn_causal_marching.py)
for a short `march_solve` smoke, and
[`docs/cookbook/pinn-causal-marching.md`](../cookbook/pinn-causal-marching.md)
for the hard-cage + `ic_fn` pattern. The acceptance artifact is
`benchmarks/causal_marching.py` (both families).

### Alpha submodules

| Submodule | Role |
| --- | --- |
| [`omnibias.pinn.train`](pinn-train.md) | Causal `march_solve`, causality / trivial-solution diagnostics |
| [`omnibias.pinn.domain`](pinn-domain.md) | SDF / R-function geometry, `DistanceConstrainedField` |
| [`omnibias.pinn.operator`](pinn-operator.md) | DeepONet / FNO + multi-head conditioning, ETDRK4 references |
| [`omnibias.pinn.interface`](interface.md) | Gated transmission PINN (02-05); `alpha -> inf` is sharpening. **Not** the XPINN seam glue in `omnibias.pinn._core.interface` |

### Interface residuals: gluing subdomains back together

Domain decomposition (XPINN / cPINN) splits a hard problem into easy patches.
The glue lives on a codimension-1 seam, and it is **two** conditions:

\[
[\![u]\!] = u_+ - u_- = 0,
\qquad
[\![k \partial_n u]\!] = k_+ \nabla u_+ \cdot n - k_- \nabla u_- \cdot n = 0.
\]

Driving only the first is the classic mistake. Value continuity is cheap to
satisfy and says nothing about whether the two pieces exchange the right amount
of flux, so an assembled field can be perfectly continuous and still fail to
solve the equation across the seam. The flux condition is what makes the
decomposition conservative, and the one that carries a genuine kink when the
material coefficients differ.

The geometry is backend-free numpy in `omnibias.pinn._core.interface`, so both
backends sample *the same* points:

* `Interface` — an oriented hyperplane `{x : n.x = c}` with the normal stored
  unit-length, so `signed_distance` is a true distance and `unit_normal` is
  directly the `n` in `d/dn`. `from_axis` / `from_spec` name it by axis;
  `from_split` reads a `PartitionedField` gate's zero set, so the seam you sample
  on is the seam the partition of unity blends across.
* `interface_points` — points drawn **on** the seam, in its own tangent
  coordinates and mapped back, so `n.x - c` is zero to round-off. This is the
  part that is easy to get subtly wrong: sampling the box and keeping what is
  "close to" the interface gives points that are *near* it, and the residual then
  measures the jump plus however much the solution varies over the gap — a floor
  no amount of training removes.
* `split_by_interface` — the complement, routing ordinary collocation to the
  patch that owns it.

```python
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch.fields import build_jet_mlp_vector_field
from omnibias.pinn.torch.losses import (
    Interface, InterfaceSpec, interface_loss, interface_residual, interface_points,
)

cs, comps = CoordinateSpec(("x", "y")), ComponentSpec(("u",))
seam = Interface.from_spec(cs, axis="x", value=0.5)
pts = torch.as_tensor(
    interface_points(seam, ((0.0, 1.0), (0.0, 1.0)), n_points=32, seed=0),
    dtype=torch.float64,
)

left = build_jet_mlp_vector_field(coordinate_spec=cs, components=comps, hidden=8, seed=0)
right = build_jet_mlp_vector_field(coordinate_spec=cs, components=comps, hidden=8, seed=1)

spec = InterfaceSpec(seam, conductivity=(2.0, 1.0), weights=(1.0, 0.1))
out = interface_residual(left(pts), right(pts), spec)
loss = interface_loss(out, weights=spec.weights)
sorted(out.diag)
```

`InterfaceSpec` carries the material pair `(k_+, k_-)` and the value / flux
weights alongside the geometry, so a problem with several seams carries its own
balance. `interface_residual` also takes `residuals=(r_+, r_-)` for XPINN's third
condition, the PDE residual's own continuity.

Everything routes through `state.ops`, so the ops work for *every* field type and
each side may be a **different** one — which is the point of decomposing at all:
a stiff patch can afford a bigger network than its quiet neighbour. See
[`docs/examples/pinn_xpinn_stiff.py`](https://github.com/derivon/omnibias/blob/main/docs/examples/pinn_xpinn_stiff.py)
for a two-patch solve with a genuine conductivity contrast, and the stiff
integrators on [the solver page](pinn-solver.md#stiff-time-stepping).

Honesty: `d/dn` is exactly as exact as the field it is taken on — closed form for
the `sigma`-tower and `jet_mlp` families, autodiff for the partitioned and cage
families. The op adds no approximation of its own; it contracts the gradient the
substrate already provides, never a finite difference across the seam.

::: omnibias.pinn.torch.losses
    options:
      show_root_heading: false
      heading_level: 3

## Equations

::: omnibias.pinn.torch.equations
    options:
      show_root_heading: false
      heading_level: 3

### Nonlocal residuals: integral equations

Every other equation here is **local** -- a residual at \(x\) reads the field and
its derivatives at \(x\) and nowhere else. `Fredholm` and `Volterra`
(`omnibias.pinn.{torch,jax}.equations.integral`) are not. They are the residuals
of an integral equation of the second kind,

\[
u(x) = f(x) + \lambda \int_\Omega K(x, t)\, u(t)\, d\mu(t),
\]

which couples every point to every other. That changes what an evaluation costs
and what the network must be able to do: it has to be evaluable at the quadrature
nodes, not only at the collocation points. In omnibias that is free, because a
field *is* a function -- `state.field(nodes)` re-evaluates it anywhere.

The two differ in one way that decides which is affordable:

| | Domain | Extra field evaluations per residual |
| --- | --- | --- |
| `Fredholm` | fixed \(\Omega\) | `n_nodes`, shared across the whole batch |
| `Volterra` | \([a, x]\), moves with the point | `batch * n_nodes`, nothing shareable |

`Volterra` stays mesh-free by pulling each interval back to a reference one,
\(t = a + (x - a)s\), and reusing a single fixed rule on \(s \in [0,1]\). That
buys the rule's own convergence order -- Gauss-Legendre, so spectral on a smooth
integrand -- instead of the second order a fixed cumulative-trapezoid grid would
give, which is what keeps `n_nodes` small enough for the cost above to be
tolerable. Its `axis` argument names the causal coordinate and every other one is
frozen at the collocation point's own value, so in a space-time problem it is a
memory term \(\int_0^t K(t,s)\, u(x, s)\, ds\) at fixed \(x\).

Only the integral is quadrature; local terms go through `state.ops.*` and stay
exact closed form. Both residuals are differentiable in the field parameters, the
kernel (so a **learned** kernel is a first-class case) and \(\lambda\), and both
return the nonlocal term alongside the residual, since it is the expensive and
the only approximated half.

Requires the quadrature from `omnibias-measure`:
`pip install "omnibias-pinn[integral]"`. When the equation is *not* coupled to a
PDE and nodal values are enough, the direct solvers in
[`omnibias.measure.{_core,torch,jax}.integraleq`](measure.md) are cheaper and
carry a Fredholm-alternative guard; see
[`docs/examples/measure_integraleq.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/measure_integraleq.py),
which validates both sides against the same analytic oracle.

::: omnibias.pinn.torch.equations.integral
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - Fredholm
        - Volterra
        - fredholm
        - volterra
        - fredholm_residual_samples
        - volterra_residual_samples

## Proof Prep

::: omnibias.pinn.certified
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - build_ns_cap_bundle
        - ns_cap_schema_errors
        - candidate_upgrade_gates
        - vorticity_residual_periodic
        - interval_arithmetic_metadata
        - interval_from_bounds
        - interval_add
        - interval_sub
        - interval_mul
        - interval_div
        - interval_square
        - interval_sqrt
        - interval_trapezoid_bound
        - compactification_map_interval
        - coefficient_interval_boxes
        - certified_tail_bounds_from_artifact
        - continuum_residual_certificates
        - finite_energy_tail_certificate
        - certified_clm_blowup
        - certified_clm_blowup_schema_errors
        - certified_clm_multizero_first_blowup
        - certified_clm_multizero_first_blowup_schema_errors
        - certified_ccf_selfsimilar_blowup_attempt
        - certified_ccf_selfsimilar_blowup_attempt_schema_errors
        - certified_ccf_hardy_wholeline_blowup_attempt
        - certified_ccf_hardy_wholeline_blowup_attempt_schema_errors
        - certified_ccf_linearized_operator_bound
        - certified_ccf_linearized_operator_bound_schema_errors
        - certified_fractional_dissipation_threshold
        - refine_ccf_hardy_profile
        - certified_euler2d_steady_vortex
        - certified_euler2d_steady_vortex_schema_errors
        - taylor_green_vortex
        - kolmogorov_flow
        - certified_taylor_green_residual
        - certified_kolmogorov_residual
        - certified_periodic_flow_residual
        - certified_periodic_flow_residual_schema_errors
        - periodic_residual_digest_ok
        - beltrami_abc_flow
        - shear_streamfunction
        - cellular_streamfunction
        - streamfunction_from_descriptor
        - certified_streamfunction_residual
        - certified_shear_streamfunction_residual
        - streamfunction_residual_schema_errors
        - fourier_mode_vorticity
        - vorticity_from_descriptor
        - integrate_vorticity_2d
        - certified_rollout_diagnostics
        - rollout_diagnostics_schema_errors
        - certified_sqg_steady_vortex
        - certified_sqg_steady_vortex_schema_errors
        - certified_sqg_selfsimilar_blowup_attempt
        - certified_sqg_selfsimilar_blowup_attempt_schema_errors
        - certified_sqg_linearized_coercivity_attempt
        - certified_sqg_linearized_coercivity_attempt_schema_errors
        - refine_ccf_selfsimilar_profile
        - radii_polynomial_closure
        - default_ccf_collocation_nodes
        - certified_gclm_selfsimilar_blowup
        - certified_gclm_selfsimilar_blowup_schema_errors
        - certified_gclm_gradient_amplification
        - certified_gclm_gradient_amplification_schema_errors
        - axisymmetric_axis_smoothness_certificate
        - build_axisymmetric_interval_report
        - certified_candidate_refinement_report
        - axisymmetric_function_space_metadata
        - assemble_axisymmetric_linearized_operator
        - operator_theoretic_invertibility_certificate
        - componentwise_radii_polynomial_certificate
        - radii_polynomial_certificate
        - norm_divergence_certificate
        - theorem_grade_function_space_contract
        - continuum_banach_invertibility_attempt
        - theorem_grade_radii_polynomial_attempt
        - exact_profile_norm_divergence_attempt
        - regularity_all_data_proof_attempt
        - build_theorem_grade_closure_attempt
        - exact_navier_stokes_equation_contracts
        - theorem_grade_function_space_definitions
        - interval_cap_backend_contract
        - blowup_route_lemma_package
        - regularity_route_lemma_package
        - proof_obligation_bundle
        - blowup_proof_obligation_bundles
        - regularity_proof_obligation_bundles
        - theorem_verifier_record
        - ingest_theorem_verifier_bundle
        - lean_formalization_package
        - external_review_gate
        - build_ns_proof_program_report
        - external_verification_record
        - verify_external_proof_package
        - theorem_claim_gate
        - build_axisymmetric_blowup_closure_report
        - build_blowup_closure_report
        - build_regularity_closure_report
        - build_regularity_inequality_report
        - regularity_counterexample_sweep
        - build_analytic_closure_report
        - build_formal_proof_package
        - build_certificate_manifest

## Diagnostics

::: omnibias.pinn.torch.diagnostics
    options:
      show_root_heading: false
      heading_level: 3

## Discontinuity-capturing PINN (partition)

A single smooth activation network cannot represent a kink / shock / phase
boundary; a **soft partition of unity of smooth sub-solutions** can.
`omnibias.pinn.partition` (a bridge on the [`omnibias-partition`](partition.md)
keystone) provides `PartitionedField`, a genuine PINN field
`u(x) = Σ_l w_l(x) u_l(x)` that plugs into the existing ops and develops an
interface between regions as the gate sharpness `beta -> ∞`. The conservative
(cPINN) demo enforces the PDE per region with an interface-continuity penalty,
beating a single `OneLayerVectorField` on interface error.

### Heterogeneous patches

The sub-solutions need not be the same *type* or the same *size*. Anything that
answers `forward_values(coords) -> (B, C)` can be a patch — a
`OneLayerVectorField`, a deep `JetMLPVectorField`, a Fourier-feature or Mscale
field — and they can be mixed freely, provided they agree on the coordinate and
component specs, which is checked rather than assumed.

```python
from omnibias.pinn.partition.torch import build_partitioned_field
from omnibias.pinn.torch.fields import OneLayerVectorField

one_d = CoordinateSpec(("x",))

def patch(region):
    if region == 0:  # the seam-side region gets the depth
        return build_jet_mlp_vector_field(
            coordinate_spec=one_d, components=comps, hidden=16, depth=3, seed=0,
        )
    return OneLayerVectorField(coordinate_spec=one_d, components=comps, hidden=4)

field = build_partitioned_field(
    coordinate_spec=one_d,
    components=comps,
    split_dirs=torch.tensor([[1.0]], dtype=torch.float64),
    split_thresh=torch.tensor([0.0], dtype=torch.float64),
    subfield_factory=patch,
)
[type(sub).__name__ for sub in field.subfields]
```

`hidden` and `base` also accept one entry per region for the common case where
only the *size* differs; `subfield_factory(region_index)` is the general escape
hatch. This is what makes decomposition worth the bookkeeping: the region holding
a boundary layer or a shock gets a bigger, higher-frequency network instead of
paying that capacity everywhere. Far from the seam the gate is saturated, so the
blend *is* that patch — a deep patch's exact closed-form derivatives survive into
the composite to float64 round-off on its own side.

Whether a skewed budget *fits better* is a benchmark question that depends on the
problem and the optimiser, and is not claimed here; what the library guarantees
is that the capacity really moves and every patch still receives a gradient.

**Honesty label.** The blended field's derivatives use the **autodiff product
rule** (the closed-form `sigma`-tower does not cover products of sigmoids); the
sound, certified quantity is the *soft->hard partition gap*
(`omnibias.partition.certify_partition_gap`). The `beta -> ∞` hardening is the
feasibility / temperature sense of "collapse", never the founding `delta -> 0`
bias collapse.

::: omnibias.pinn.partition.torch
    options:
      show_root_heading: false
      heading_level: 3

## Extensions

Opt-in helpers that wire model-level operators into the
`omnibias.fields.ops_registry` extension point. Importing the package registers
nothing; call the `register_*` helpers to opt in. `register_lim_along` exposes
the closed-form jet `lim` operator as `state.<component>.lim_along`.

::: omnibias.pinn.extensions
    options:
      show_root_heading: false
      heading_level: 3

## Core schemas

::: omnibias.pinn._core
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend has the same module layout under
`omnibias.pinn.jax`. All cross-backend tests in
`packages/omnibias-pinn/tests/cross_backend/` assert *bit-identical*
results between the two backends (typical tolerances: rtol/atol=1e-12
in float64).
