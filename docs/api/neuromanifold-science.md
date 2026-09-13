# Neuromanifold science integrations

Finite observations connect a model's parameter directions to quantities a
scientific experiment can measure. The new integrations use that map for
observation design, continuation, effective models, and quantum geometry. They
build on the [operator surface](../operator-surface.md): closed-form input jets
can supply observation derivatives, while derivatives with respect to arbitrary
model parameters still need an explicit analytic or autodiff provider. These
are different derivative contracts.

The numerical producers live in `geometry.continuation`, `pinn.inverse`,
`symbolic.reduction`, `curvature.operators`, and `ferminet.operator_sr`.
Rigorous consumers live in `dynamics.continuation` and
`verify.neuromanifold.scientific`. The permissive producers do not import the
copyleft consumers. Existing inverse-location and dense SR interfaces remain
available with their previous defaults.

## Observations, nuisance parameters, and design

`ObservationModel` accepts arbitrary vector observations and their parameter
Jacobian. By default, `observation_information` computes the finite Fisher
matrix `J.T @ covariance^-1 @ J` for fixed covariance; a scalar denotes
variance. An explicit likelihood-score provider includes information carried
by a parameter-dependent noise law. A callable covariance requires this
provider and is still checked for valid positive covariance.

```python
import numpy as np
from omnibias.pinn.inverse.observation import (
    ObservationModel, candidate_information, identifiability_report,
    observation_information, profile_information,
)
from omnibias.pinn.inverse.design import design_score, select_design

def observations(theta, times):
    amplitude, rate, offset = theta
    return amplitude * np.exp(-rate * times) + offset

def observation_jacobian(theta, times):
    amplitude, rate, offset = theta
    decay = np.exp(-rate * times)
    return np.stack((decay, -amplitude * times * decay, np.ones_like(times)), axis=-1)

model = ObservationModel(observations, observation_jacobian, ("amplitude", "rate", "offset"))
theta = np.array([2.0, 0.7, 0.2])
times = np.array([0.0, 0.5, 1.0, 2.0])
info = observation_information(model, theta, times, covariance=0.01)
assert identifiability_report(info).rank == 3
profile = profile_information(info.whitened_jacobian, target=[1], nuisance=[0, 2])
assert profile.fisher[0, 0] > 0
blocks = candidate_information(model, theta, times, covariance=0.01)
score = design_score(blocks, np.ones(4), np.eye(3), criterion="D")
assert np.isfinite(score)
selection = select_design(blocks, np.eye(3), 3, criterion="A",
                          method="exhaustive", max_subsets=10)
assert selection.status == "selected" and selection.enumeration_complete
assert len(selection.indices) == 3
```

`LikelihoodScores` contains one parameter-score vector per sampled outcome of
the whole declared experiment, nonnegative weights, and expectation provenance
(`exact_finite`, `quadrature`, or `monte_carlo`). Weights are normalized by their
positive sum. Fisher is the weighted sum of score outer products, which need
not equal the whitened mean-Jacobian Gram matrix. Providers are responsible for
supplying the actual derivatives of the log likelihood. The provenance label
does not certify their derivation or the numerical expectation.

For `Y ~ Normal(mu, exp(2*eta))`, the log-scale score is `z² - 1` and carries
information even though the mean has zero derivative with respect to `eta`.
Three-node Gauss-Hermite quadrature integrates these degree-four score products
exactly in real arithmetic; the following evaluation uses floating arithmetic.

```python
from omnibias.pinn.inverse.observation import LikelihoodScores

def gaussian_likelihood_scores(theta, design):
    nodes, weights = np.polynomial.hermite.hermgauss(3)
    z = np.sqrt(2.0) * nodes
    scores = np.column_stack((z * np.exp(-theta[1]), z*z - 1.0))
    return LikelihoodScores(scores, weights, "quadrature", "Gauss-Hermite, three nodes")

gaussian = ObservationModel(
    lambda theta, design: np.array([theta[0]]),
    lambda theta, design: np.array([[1.0, 0.0]]),
    ("mean", "log_scale"),
    score_provider=gaussian_likelihood_scores,
)
noise_theta = np.array([0.7, 0.4])
noise_info = observation_information(
    gaussian, noise_theta, np.ones(1),
    covariance=lambda theta, design: np.exp(2.0 * theta[1]),
)
assert np.allclose(noise_info.fisher, np.diag([np.exp(-0.8), 2.0]))
noise_profile = profile_information(noise_info.information_factor, target=[1], nuisance=[0])
assert np.allclose(noise_profile.fisher, [[2.0]])
assert identifiability_report(noise_info).rank == 2
```

`information_factor` selects the weighted score factor when supplied and the
whitened mean Jacobian otherwise. Pass it to `profile_information` to retain
noise-law information during nuisance projection. `whitened_jacobian` retains
its original meaning. Score providers may also be passed explicitly to
`observation_information` and `candidate_information`; candidate blocks still
assume independent experiments.

`ObservationModel.directional_jet` optionally supplies ordinary directional
derivatives through a requested finite order, including order zero. It returns
an array shaped `(order + 1, *observation.shape)`, not Taylor-normalized
coefficients. `observation_visibility` validates its value and first derivative
against the model and reports the first visible higher derivative.

```python
import math
from omnibias.pinn.inverse.observation import observation_visibility

def cubic_jet(theta, design, direction, order):
    return np.array([
        math.factorial(3) / math.factorial(3-k) * theta[0]**(3-k) * direction[0]**k
        if k <= 3 else 0.0
        for k in range(order + 1)
    ])

cubic = ObservationModel(
    lambda theta, design: np.asarray(theta[0]**3),
    lambda theta, design: np.array([3*theta[0]**2]),
    ("t",), directional_jet=cubic_jet,
)
visibility = observation_visibility(cubic, np.zeros(1), np.ones(1), np.ones(1), max_order=4)
assert visibility.first_visible_order == 3
assert not visibility.certified
```

A tower that vanishes through the requested order is
`inconclusive_at_supplied_order`. These reports concern the local observation
map along the supplied direction; they do not establish global identification,
classify intrinsic image singularities, or analyze an entire likelihood law.

`profile_information` projects out nuisance tangents, including redundant
nuisance columns. A deficient first-order Jacobian is a diagnostic; it does
not prove global nonidentifiability. `certify_identifiability` requires an
outward-rounded whitened-Jacobian callback over the declared parameter box.
Its sufficient condition proves full rank of all selected physical and
nuisance columns. A redundant nuisance block can therefore be numerically
profiled but still receive `Inconclusive` from this certificate.

`differentiable_design_score` provides torch/JAX D-, A-, and E-objectives on supplied
information blocks. E uses the smallest eigenvalue and is differentiable where
that eigenvalue is simple. At a repeated minimum it is nonsmooth;
`design_gradient(..., criterion="E")` refuses to report a unique gradient.
`select_design` provides numerical greedy or exhaustive subset selection for
all three criteria, with deterministic index ties and an explicit evaluation
budget. Exhaustive completion means every requested-cardinality subset was
numerically compared. Greedy completion only means the requested number of
candidates was selected. A preflight `budget_exceeded` result carries no partial
design; neither method attaches a rigorous optimality proof or approximation
factor.

For eligible D-selection with a separate submodular guarantee,
`omnibias.submodular.design.information_design_problem` accepts fixed independent
observation factors and an SPD prior. Its normalized log-determinant objective
fits the existing cardinality-constrained submodular algorithms. That guarantee
does not extend automatically to nuisance-profiled or robust objectives, or to
correlated candidate blocks. Weighted log determinant is not the multilinear
extension of the corresponding set function.

## Continuation and a PDE fold

`ImplicitFamily` accepts `F(y)`, its Jacobian, and optional second/third derivative
tensors, where the last coordinate of `y` is the continuation parameter. Generic
pseudo-arclength correction follows regular branches through parameter folds.
`locate_fold` and `locate_hopf` return numerical normal-form diagnostics; the
Hopf first Lyapunov coefficient requires the supplied third derivatives.

`geometry.continuation_directional.directional_residual_family` adapts callbacks
for `D F[v]`, `D^2 F[u,v]`, and `D^3 F[u,v,w]` to these correctors. The callbacks
return actual derivatives; Taylor coefficients need their factorial factors.
This adapter checks the total dense coefficient budget before allocating a basis
or calling a model. It introduces neither finite differences nor a claim of
matrix-free event localization.

```python
from omnibias.geometry.continuation_directional import directional_residual_family
from omnibias.geometry.continuation import locate_fold

directional_fold = directional_residual_family(
    lambda y: np.array([y[0]**2 - y[1]]),
    lambda y, v: np.array([2*y[0]*v[0] - v[1]]),
    state_dimension=1,
    second=lambda y, u, v: np.array([2*u[0]*v[0]]),
    third=lambda y, u, v, w: np.zeros(1),
)
assert locate_fold(directional_fold, np.array([0.1, 0.02])).nondegenerate
```

For the Bratu boundary-value problem `u'' + lambda exp(u) = 0`, `u(0)=u(1)=0`,
the analytic solution family
`u(x) = 2 log(cosh(a/2) / cosh(a(x-1/2)))` reduces to
`g(a, lambda) = lambda cosh(a/2)^2 - 2a^2 = 0`. The following encloses its fold
using the augmented equations `(g, g_a) = 0`.

```python
from omnibias.core.verified.interval import Interval as I
from omnibias.core.verified.transcend import cosh_iv, sinh_iv
from omnibias.dynamics.continuation import certify_event

def bratu_fold(z):
    a, lam = z
    return [lam * cosh_iv(a / 2)**2 - 2*a*a, lam * sinh_iv(a)/2 - 4*a]

def bratu_fold_jacobian(z):
    a, lam = z
    return [[lam*sinh_iv(a)/2 - 4*a, cosh_iv(a/2)**2],
            [lam*cosh_iv(a)/2 - 4, sinh_iv(a)/2]]

def bratu_conditions(z):
    a, lam = z
    return {"transversality": cosh_iv(a/2)**2,
            "quadratic": (lam*cosh_iv(a)/2 - 4)/2,
            "complement_gap": I.point(1)}

fold = certify_event("fold", bratu_fold, bratu_fold_jacobian,
                     [2.3993572805, 3.5138307191], bratu_conditions, radius=1e-6)
assert fold.certified
```

The complement condition here is vacuous for the one-dimensional state equation.
The result concerns this explicit analytic family; it does not classify all PDE
solutions. For general problems, the event callback must supply sound
transversality, normal-form, spectral separation, and (for Hopf) resonance
conditions at the enclosed root. `certify_event` checks those supplied
enclosures; it does not derive general interval normal forms automatically.
Discretized PDEs additionally need continuum existence and discretization-error
arguments. `certify_segment` proves a root for every parameter in an interval;
`certify_join` requires a common endpoint uniqueness enclosure, not box overlap.

`geometry.continuation_switch.switch_equilibrium_branch` switches from a known
parent equilibrium branch at a supported simple branch point. It requires a
two-dimensional null space of the augmented state/parameter Jacobian, a parent
tangent with a nonzero parameter component, a nonzero mixed quadratic crossing,
and supplied second/third derivatives. A transverse phase condition and scaled
Newton correction produce an off-parent seed. The following pitchfork control
switches from `x=0` to the independently known branch `lambda=x**2`, then continues
along it.

```python
from omnibias.geometry.continuation import ImplicitFamily, continue_branch
from omnibias.geometry.continuation_switch import switch_equilibrium_branch

def pitchfork_third(z):
    tensor = np.zeros((1, 2, 2, 2))
    tensor[0, 0, 0, 0] = -6
    return tensor

family = ImplicitFamily(
    lambda z: np.array([z[0]*(z[1]-z[0]**2)]),
    lambda z: np.array([[z[1]-3*z[0]**2, z[0]]]),
    lambda z: np.array([[[-6*z[0], 1.], [1., 0.]]]),
    pitchfork_third,
)
switched = switch_equilibrium_branch(family, np.zeros(2), np.array([0., 1.]), side=1)
assert switched.status == "switched" and switched.seed is not None
branch = continue_branch(family, switched.seed, direction=switched.tangent, n_steps=12)
assert branch.status == "complete" and branch.points[-1, 0] > 0.1
assert np.allclose(branch.points[:, 1], branch.points[:, 0]**2, atol=1e-9)
```

The switch is a numerical result. The declared parent branch and derivative
callbacks must describe the same model. Higher-codimension rank loss, a
degenerate mixed crossing, missing derivatives, and failed correctors return
`inconclusive`; this API does not construct periodic branches or classify every
degenerate bifurcation.

## Effective models with explicit reconstruction

`ParameterReduction` describes fixing, coalescence, and integer-power scaling
of parameters. A singular scaling requires an explicit reduced model. The
following vanishing quadratic coefficient has an exact reduced affine law.

```python
from omnibias.symbolic.reduction import (
    ParameterReduction, ParameterTerm, evaluate_reduction, reduction_candidate,
)
from omnibias.verify.neuromanifold.scientific import certify_reduction

def full_law(parameters, x):
    return parameters[0] + parameters[1]*x + parameters[2]*x*x

transform = ParameterReduction(
    (ParameterTerm(0), ParameterTerm(1), ParameterTerm(None, power=1)), 2)
candidate = reduction_candidate(full_law, transform)
report = evaluate_reduction(candidate, [0.1, 0.01, 0.001],
                            np.array([1.0, 2.0]), np.linspace(-1, 1, 21))
assert np.allclose(report.observed_orders, 1)

def reduction_error(box):
    epsilon, x = box
    return {0: epsilon*x*x, 1: 2*epsilon*x, 2: 2*epsilon}

reduction = certify_reduction(
    reduction_error, [I(0, 0.001), I(-1, 1)], orders=(0, 1, 2), error_budget=0.0021,
    provider_assumption="exact polynomial value and x-derivative error expressions")
assert reduction.certified and not reduction.unresolved_regions
```

The numerical report covers evaluated inputs only. The rigorous result covers
all accepted boxes and requested derivative orders, conditional on the supplied
error callback being sound. Any unresolved region prevents a whole-domain
certificate. For example, `epsilon/(epsilon+x)` on a box containing `(0,0)`
cannot earn a small uniform-error certificate from its pointwise limit at
fixed positive `x`.

`geometry.continuation.follow_model_boundary` proposes reduction directions by
integrating a geodesic using the observation Jacobian and its directional second
derivative. Rank loss, domain exit, and a large parameter norm stop the numerical
path. None of these stops establishes the limiting effective law. Model-manifold
boundary reduction has substantial [prior literature](https://pmc.ncbi.nlm.nih.gov/articles/PMC4425275/).

## Quantum geometry without a parameter-square matrix

`qgt_operator` computes centered real quantum-geometric-tensor actions from JAX
JVP/VJP operations on a live real parameter pytree. Complex log amplitudes are
supported through their real and imaginary score components. Chunking avoids
materializing the full sample-by-parameter score table. `matrixfree_sr_step`
uses `curvature.operators.pcg_solve`; a failed convergence check leaves parameters
unchanged. The implicit solve derivative is meaningful only for a converged
solve. This is a numerical solver, and sampling metadata is not a proof.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.ferminet.operator_sr import matrixfree_sr_step, qgt_operator

def log_psi(parameters, x):
    return parameters["offset"] + jnp.dot(parameters["weights"], x)

parameters = {"offset": jnp.asarray(0.0), "weights": jnp.array([0.2, -0.1])}
samples = jnp.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
operator = qgt_operator(log_psi, parameters, samples, chunk_size=2)
assert operator.matvec(jnp.ones(3)).shape == (3,)
step = matrixfree_sr_step(log_psi, parameters, samples, jnp.array([0., 1., 2., 1.]),
                         damping=0.1, chunk_size=2, rtol=1e-10)
assert step.accepted
```

For a declared finite distribution with exact rational scores and weights,
`certify_quantum_geometry` instead computes and replays exact centered
covariance, complete null space, and positive geometry on a complement.

```python
from omnibias.verify.neuromanifold.scientific import (
    certify_quantum_geometry, replay_quantum_geometry,
)
quantum = certify_quantum_geometry([[1, 0, 0], [1, 1, 1], [1, 2, 2]], [1, 2, 1])
assert quantum.certified and quantum.rank == 1
assert replay_quantum_geometry(quantum.certificate)
```

This proves a finite rational matrix statement for the supplied operands.
Their correspondence to the intended wavefunction and the completeness of the
finite distribution remain explicit scientific obligations. Floating Monte Carlo
scores cannot be relabeled as this proof. Neither the numerical QGT nor these
certificates assert a ground-state energy, a new physical phase, or a formal
Lean result. See the [benchmark ledger](../benchmarks/neuromanifold-science.md)
for measured acceptance workloads and remaining limitations.
