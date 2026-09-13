# Neuromanifold geometry and certified collision transitions

This alpha extension treats a network as a realization map from parameters to
explicitly declared observations. Its central collision chart replaces divergent
opposite neuron coefficients with bounded moments. The public numerical modules
are `omnibias.geometry.neuromanifold`, `omnibias.{torch,jax}.realization`,
`omnibias.{torch,jax}.confluence`, and `omnibias.{torch,jax}.confluent_bank`.
Certified consumers live in `omnibias.verify.neuromanifold`; exact polynomial
images and finite proof replay live in existing core/formal modules.

The [operator surface](../operator-surface.md) remains the capability contract.
Bias collapse produces activation derivatives. Temperature collapse relaxes a
finite architecture choice. Enclosure collapse contracts a **sound** enclosure.
They are distinct operations. All six `OperatorBlock` roles remain available,
including the antiderivative window `S(z+b_hi)-S(z+b_lo)` with `S'=sigma`.

## Scope and derivatives

- Hidden parameter dependence was missing from input jets → joint input/weight
  Cauchy products retain trainable mixed derivatives →
  `omnibias.{torch,jax}.realization.parameter_jet`, also exported through
  `omnibias.{torch,jax}.jet_mv.parameter_jet`. Pre-seeded directional and
  multivariate towers enter through `jet.mlp_joint_jet` and
  `jet_mv.mlp_joint_jet_mv`; all three delegate to the same realization kernels.
- A finite probe vector could be confused with a function → explicit parameter,
  realization, observation, and integral descriptions retain the equality scope →
  `omnibias.core.realization`.
- Derivative orders repeated activation work → an optional registered
  `ActivationSpec.tower(z, max_order)` shares one base evaluation and the existing
  coefficient/evaluation conventions → activation registries and MultiPack.
  Custom providers without a tower use a labeled per-order fallback.

`ParameterLayout` fixes row-major parameter blocks and names. `LayerSpec`
expresses dense affine layers followed by identity/grad/laplacian/derivative,
activation bands, or integral windows. Window endpoints, and the alternative
center plus raw softplus width, stay live. `operator_block_jet` adapts the actual
Torch block storage; `operator_parameter_jet` is its explicit Torch/JAX twin.
Unsupported supplied models are rejected instead of replaced by a manufactured
example. The historical core `autodiff` spelling is a deprecated finite-difference
alias; new backend autodiff callbacks perform actual autodiff.

```python
import torch
from omnibias.core.realization import LayerSpec, ParameterLayout, RealizationSpec
from omnibias.torch.jet_mv import parameter_jet
from omnibias.geometry.neuromanifold.torch import dense_realization, dense_jacobian

layout = ParameterLayout.from_shapes([
    ("W0", (2, 1)), ("b0", (2,)), ("W1", (1, 2)), ("b1", (1,)),
])
spec = RealizationSpec(
    (LayerSpec(1, 2, activation="tanh", weight_name="W0", bias_name="b0"),
     LayerSpec(2, 1, activation=None, weight_name="W1", bias_name="b1")),
    layout, input_dim=1, output_dim=1,
)
x = torch.linspace(-1, 1, 9, dtype=torch.float64).reshape(-1, 1)
theta = torch.tensor([1., .7, .1, -.2, .8, -.4, .2],
                     dtype=torch.float64, requires_grad=True)
basis = torch.eye(layout.size, dtype=theta.dtype)[:, :2]
jet = parameter_jet(x, theta, spec, basis, 4)
assert jet.shape == (15, 9, 1)  # two directions, total order <= 4
jet.square().sum().backward()
assert theta.grad is not None
realization = dense_realization(x, spec)
jacobian = dense_jacobian(realization, theta)
metric_times_v = realization.metric_action(theta, torch.ones_like(theta), lambda v: v / 9)
torch.testing.assert_close(metric_times_v, jacobian.T @ jacobian @ torch.ones_like(theta) / 9)
```

Jets contain Taylor-normalized coefficients and require an explicit `P x q`
direction basis. Full parameter seeds require explicit opt-in in the underlying
jet APIs and a coefficient budget. `JVP` uses the supplied jet; `VJP` uses
first-order backend reverse autodiff. These tensors remain connected to training
graphs. Native transcendental evaluations retain their backend accuracy; exact
rational witnesses belong to the rigorous/formal registers, not float equality.

`ObservationSpec` distinguishes complete coefficients, finite values,
derivatives, residuals, and integrals. Its `integral_kind` distinguishes an
activation window from domain quadrature and measure-weighted quadrature. A
coefficient-map label requires an actual determining representation to support
global equality. A quadrature label does not supply a continuum error estimate.

`ChartSpec(jacobian=..., derivative_provenance=...)` lets existing pullback
geometry consume a supplied chart derivative. Without it, the existing labeled
forward-autodiff path remains in use. Geometry beyond the supplied derivative
continues to differentiate the analytic metric; it is not automatically a
closed-form metric-curvature tower.

## The centered collision chart

For `rho=h² >= 0`, define

\[
 A=\tfrac12[\sigma(z+h\eta)+\sigma(z-h\eta)],\qquad
 B=\frac{\sigma(z+h\eta)-\sigma(z-h\eta)}{2h}.
\]

The ordinary pair equals `m0*A+m1*B`, where `m0=a_plus+a_minus` and
`m1=h*(a_plus-a_minus)`. At zero spread, `A=sigma(z)` and `B=eta*sigma'(z)`.
An affine `eta=v*x+c` also supports weight-direction collisions.

```python
from omnibias.torch.confluence import centered_pair, polynomial_pair

z = torch.tensor([-.4, .3], dtype=torch.float64)
rho = torch.tensor(0., dtype=torch.float64, requires_grad=True)
eta = torch.tensor([.7, -.2], dtype=torch.float64)
a, b = centered_pair(z, rho, eta)
s = torch.sigmoid(z)
torch.testing.assert_close(a, s)
torch.testing.assert_close(b, eta * s * (1-s))
(a + b).sum().backward()
assert torch.isfinite(rho.grad)
# A polynomial control terminates exactly, including at zero spread.
pa, pb = polynomial_pair(z, rho, eta, torch.tensor([0., 0., 1.], dtype=z.dtype))
torch.testing.assert_close(pa, z*z)
torch.testing.assert_close(pb, 2*z*eta)
```

`centered_pair` supports sigmoid/tanh stable finite-spread formulas. Near zero it
uses a declared even polynomial in `rho`, with safe inactive branches and
`pair_series_error` bounds. The analytic pair is smooth; the finite branch switch
is a numerical approximation. A whole-domain spatial-derivative transition
certificate refuses a domain crossing that switch. The series settings are
included in bank snapshots. Backend floating-point evaluation rounding is
separate from the analytic truncation and parameter-conversion bounds; use the
rigorous register when the evaluated number itself needs an enclosure.

`initialize_moments` computes common-weight bias-cluster moments using exact
arithmetic on stored dyadic parameters and returns conversion-error intervals.
`moment_atom` evaluates their finite derivative expansion. `cluster_remainder`
includes Taylor truncation and moment conversion; a finite cluster expansion is
not an exact coordinate change. Initializing moments cannot recover information
lost before parameters were stored.

## Transactions and optimizer state

`ConfluentPackBank` opts into preallocated ordinary, pair, and moment slots.
Parameter objects and shapes survive commits. Stable IDs travel with exact
permutations. Legacy bank checkpoints migrate to ordinary mode. Torch has a
module interface; JAX has an immutable PyTree, `parameters`/`with_parameters`,
portable checkpoints, and functional commit results.

```python
from omnibias.core.refine import RefinedPack
from omnibias.core.verified.interval import Interval
from omnibias.torch.confluent_bank import ConfluentPackBank, commit_transition
from omnibias.verify.neuromanifold import certify_transition

bank = ConfluentPackBank([
    RefinedPack(center=-.001, weight=.7, scale=1., order=0),
    RefinedPack(center=.001, weight=-.2, scale=1., order=0),
], max_packs=4, max_order=4, dtype=torch.float64)
optimizer = torch.optim.Adam(bank.parameters(), lr=.001)
proposal = bank.propose_pair(0, 1, error_budget=1e-8)


def accept(source, proposed):
    return certify_transition(source.portable_snapshot(), proposed,
                              domain=Interval(-1, 1), spatial_orders=(0, 4)).accepted


parameter_ids = [id(p) for p in bank.parameters()]
assert commit_transition(bank, proposal, optimizer, between_steps=True, acceptance=accept)
assert [id(p) for p in bank.parameters()] == parameter_ids
```

Generate proposals eagerly, then commit **between optimizer steps**, after
`zero_grad(set_to_none=True)`. Source hashes bind all parameters, modes,
configuration, and the version. Stale proposals are rejected. Failed acceptance
or postconditions leave the original model intact; failed postconditions restore
optimizer state too.

`propose_pair`, `propose_derivative`, `propose_cluster`, `propose_remove_zero`,
and `propose_birth` cover the supported structural transitions. Cluster/birth
updates initialize complete preallocated rows. Nonlinear transitions reset
changed elementwise moments while preserving unaffected slots and shared scalar
step counters. Reused slots reset. Coupled LBFGS history is cleared for nonlinear
changes; unknown optimizer structures require an explicit adapter. Exact
permutation and ordinary-tanh sign relabelings have dedicated first-/second-
moment transport. Unsafe inherited `sync_from` writes are refused.

`detect_transitions` ranks compatible proposals by a caller's measured
observation-conditioning improvement, using stable-ID ties. Numerical proposals
remain permissive. `verify.neuromanifold.selection` reuses the shipped structured
log-sum-exp/softmax operator for finite architecture costs. Its
`log(N)/beta` gap applies only to that fixed catalogue. `decode_architecture`
requires a source-bound certificate for the actual decoded transition.

Spatial derivative budgets compose through `linear_residual_error` and
`nonlinear_residual_error`; `integral_error` requires explicit activation-window,
domain-quadrature, or measure semantics. Input/PDE derivative preservation does
not imply preservation of every parameter gradient after an architecture change.

## Fibers, regular geometry, and minima

- Parameter relabelings obscured equivalent functions → explicit permutation,
  tanh parity, sigmoid complement, and declared monomial scaling preserve the
  supported atoms → `neuromanifold.symmetry`.
- A zero singular value did not identify its cause → numerical rank reports,
  exact stored-matrix witnesses, determining-observation comparisons, and exact
  monomial-curve controls separate supported cases → `neuromanifold.strata`.
- Ambient loss curvature hid realization curvature → weighted projections,
  normal acceleration, second fundamental forms, and the full least-squares
  Hessian are available on regular charts → `neuromanifold.geometry`.

`affine_quotient` replays a rational affine map's factorization, right inverse,
and complete kernel. `homogeneous_anchor_chart` fixes nonzero monomial anchors
and invalidates them when their margins fail. An empirical nullspace and damping
alone do not construct a quotient. Numerical equality of a sampled Jacobian to
an affine matrix does not prove a nonlinear realization has that fiber.

`ObservationMetric` records empirical/quadrature/Sobolev/residual/measure
provenance. `Realization.metric_action` computes `J.T W J v` without allocating a
parameter matrix. A dense observation Jacobian requires a budget. These providers
compose with the existing `omnibias.torch.optim.NaturalGradient` and JAX natural
steps. `quotient_step` supplies reduced-coordinate backtracking; the caller must
already have justified those coordinates.

`monomial_curve_stratum((2,))`, `((3,))`, and `((2,3))` distinguish a real-image
boundary, a singular parametrization of a smooth line, and an intrinsic algebraic
cusp. General nonlinear maps return finite-order diagnostics. Higher-order
visibility and escape candidates are directional evidence, not complete global
critical-point coverage.

```python
from omnibias.geometry.neuromanifold import affine_quotient, monomial_curve_stratum
from omnibias.verify._core.param_loss import HyperDual
from omnibias.verify.neuromanifold import (
    IntervalObjective, certify_quotient_minimum, minimum_replay_certificates,
)
from omnibias.core.proof.realization_replay import verify_replay_certificate

assert monomial_curve_stratum((3,)).intrinsic_class == "smooth_curve"
chart = affine_quotient([[1, 1]])


def reduced_loss(q):
    residual = q[0] - HyperDual.constant(2)
    return residual * residual


objective = IntervalObjective(reduced_loss, "sum_parameter_quadratic", ("sum",))
minimum = certify_quotient_minimum(chart, objective, (Interval(1.9, 2.1),), morse_bott=True)
assert minimum.status == "proved"
assert minimum.stationary_box[0].contains(2)
assert all(verify_replay_certificate(c) for c in minimum_replay_certificates(minimum))
```

The last example **defines** the full loss as `ell(chart.projection @ theta)`.
That supplies exact invariance, a valid slice, constant rank, and the complete
affine critical family over the certified reduced box. `certify_slice_minimum` makes the weaker slice claim for
an arbitrary supplied coordinate restriction. Both use whole-box interval
hyperdual derivatives, Krawczyk inclusion, and uniformly positive reduced
Hessians. The reported box encloses the actual stationary point, not merely the
floating center. Center Taylor coefficients with only a scalar remainder are
not derivative enclosures.

`formalize_minimum(minimum, mathlib=True)` runs both real builds on the source
factorization, recomputed Krawczyk products/root enclosure, and interval LDL.
`formalize_confluence` checks source affine-bias and moment identities, proposed
stored-coordinate conversion bounds, and reconstructed error budgets. Formal flags are
earned only by the respective build; absent toolchains leave them false. Taylor,
root-existence, and geometric coverage theorems remain explicit dependencies
when the finite Lean obligation does not prove those analytic statements.

## Complete implementation map

| Gap | Working public capability | Acceptance scope |
|---|---|---|
| 1 | Shared realization/observation descriptions and numerical realizations | Explicit finite versus determining observations |
| 2 | [Exact coefficient maps and scoped membership](realization-algebra.md) | Rational/algebraic witnesses, real image versus closure, bounded searches |
| 3 | Live joint input/weight jets; all six roles | Conditioned mixed derivatives through order four; coefficient budgets |
| 4 | Centered pairs and bounded cluster moments | Zero/tiny/finite spread, affine directions, declared remainders |
| 5 | Explicit symmetry actions and regular charts | Declared affine/monomial families; duplicate/inactive fibers separate |
| 6 | Rank witnesses, visibility, scoped strata controls | Exact minor/factorization evidence and honest unresolved cases |
| 7 | Dense and matrix-free observation metrics | Fixed weight provenance, regular reduced coordinates |
| 8 | Extrinsic geometry and higher-order escape candidates | Regular charts; finite-order singular diagnostics |
| 9 | Slice/quotient/Morse–Bott minimum certificates | Actual root enclosure, whole-box Hessian, explicit family |
| 10 | Versioned structural commits and finite temperature selection | State reset/transport, stale rejection, rollback, source-bound budgets |
| 11 | [General identifiability](neuromanifold-science.md) | Fixed covariance, nuisance profiling, local interval information |
| 12 | [Experimental design](neuromanifold-science.md) | D/A/E numerical objectives; guarantees only for eligible additive PSD D-design |
| 13 | [Continuation and Bratu](neuromanifold-science.md) | Supported fold/Hopf systems, validated overlaps, analytic PDE reference |
| 14 | [Boundary exploration and law reduction](neuromanifold-science.md) | Finite grammar/callbacks, uniform derivative errors, nonuniform refusal |
| 15 | [Matrix-free quantum geometry and SR](neuromanifold-science.md) | Complex waves, real parameters, null projection, explicit convergence |
| 16 | [Operand-bound finite replay](realization-algebra.md) | Both actual Lean routes; analytic assumptions remain visible |

Reproduction and measured limits:
[collision benchmark](../benchmarks/neuromanifold-collisions.md),
[scientific benchmark](../benchmarks/neuromanifold-science.md),
[acceptance map](../benchmarks/neuromanifold-acceptance.md).
The implementation is a research instrument. Famous-problem solutions or new
scientific discoveries require their own mathematical or experimental evidence.
