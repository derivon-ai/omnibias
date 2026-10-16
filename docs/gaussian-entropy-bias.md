# Certified Gaussian bias and target sensitivities

The [stable-face target certificate](gaussian-entropy-sensitivity.md) extends
to the logarithmic mixture weights of the original Gaussian heat family.
It encloses derivatives of the exact, locally smooth **global decision**,
even when the Gaussian potential is nonconvex outside its certified region.
These biases are the parameters \(b_j\) in positive weights \(e^{b_j}\).
They are not arbitrary hidden-neuron translations, and no activation
interchangeability follows.

Centers, viscosity \(\nu\), heat age \(s\), and the polyhedron remain
fixed. Only the target \(v\) and original mixture biases vary. A stable
face proof is required; uniqueness alone does not justify an ordinary
derivative at a constraint switch.

## Joint derivative and pullback

At the certified optimizer \(x_*\), write

\[
\pi_j=\frac{\exp(b_j-\|x_*-a_j\|^2/(4\nu s))}
{\sum_k\exp(b_k-\|x_*-a_k\|^2/(4\nu s))},\qquad
\mu=\sum_j\pi_j a_j,\qquad U=(x-\mu)/s.
\]

Let \(R\) be the stable face's orthogonal tangent projector and
\(J=\partial x_*/\partial v\) the certified symmetric target Jacobian.
For bias direction \(\beta\), differentiation of the normalized weights
at fixed \(x\) gives

\[
D_b\pi_j[\beta]=\pi_j(\beta_j-\mathbb E_\pi\beta),\qquad
D_bU[\beta]=-\operatorname{Cov}_\pi(a,\beta)/s.
\]

Differentiating tangent stationarity therefore yields

\[
\boxed{D_{(v,b)}x_*[\xi,\beta]
=J\left(\xi+\operatorname{Cov}_\pi(a,\beta)/s\right)}.
\]

The bias Jacobian columns are

\[
\boxed{\partial_{b_j}x_*=
\frac{\pi_j}{s}J(a_j-\mu)}.
\]

For a fixed cotangent \(g\), set \(z=Jg\). The pullback is

\[
\boxed{\nabla_v(g^\top x_*)=z},\qquad
\boxed{\partial_{b_j}(g^\top x_*)=
\frac{\pi_j}{s}(a_j-\mu)^\top z}.
\]

These formulas use the original posterior at the exact optimizer; no rational
posterior is fitted. Log-partition differentiation and its covariance
identities are classical; see [Wainwright–Jordan, Section 3.5](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf).

## Exact invariances and interval bounds

A common bias change multiplies the heat function by one positive scalar,
leaving its velocity and optimizer unchanged. Hence a constant direction
\(\beta=c\mathbf1\) has exactly zero bias response, and every mathematical
bias pullback sums to zero.

Queries subtract \(\beta_0\) exactly before numerical conversion. Spatial
centers are likewise shifted and projected exactly:
\(u_j=R(a_j-a_0)\). Thus common-bias directions and constraint-normal
cotangents have exact zero responses, including when their entries are much
larger than binary64 can represent. Ordinary coordinate interval boxes need
not display the zero-sum identity when their endpoints are added; that is
not evidence that the true pullback violates it.

A useful covariance identity is

\[
\operatorname{Cov}_\pi(a,\beta)
=\tfrac12\sum_{j,k}\pi_j\pi_k(a_j-a_k)(\beta_j-\beta_k).
\]

Equivalent centered moments avoid a quadratic mode-pair loop. The joint
forcing is uncertain because the true optimizer and posterior are enclosed,
so it cannot simply be passed to an API accepting an exact cotangent. The
bias certificate propagates this interval forcing through the stored matrix
polynomial and adds the existing analytic tail bound. Dependence between
posterior moments and the Jacobian can widen the enclosure; interval
arithmetic preserves containment without assuming independence.

## Why nearby biases retain the global branch

The base point has positive tangent curvature and a strict stable-face
witness. Gaussian smoothness and the implicit-function theorem give a local
smooth branch jointly in \((v,b)\). For bounded nearby biases and targets,
the quadratic growth of the Gaussian potential remains uniformly coercive.
Uniqueness at the base point forces all nearby global minimizers into this
branch neighborhood, where the implicit solution is unique.

Consequently the derivative belongs to the unique global optimizer on some
open parameter neighborhood. This argument does not require recomputing a
posterior-domain certificate at every nearby bias. It also does not supply a
numerical neighborhood radius, finite-step accuracy, or active-set stability
for arbitrary finite bias changes.

## API and example

`certify_gaussian_entropy_bias(sensitivity)` in
`omnibias.core.verified.gaussian_entropy_bias` takes an intact
`GaussianEntropySensitivity` and returns `GaussianEntropyBias` or refuses.

| Field | Meaning |
| --- | --- |
| `sensitivity` | Source target-Jacobian and stable-face proof. |
| `projected_centers` | Exact rows \(R(a_j-a_0)\). |
| `projected_mean` | Interval enclosure of \(\sum_j\pi_j u_j\). |
| `bias_forcing` | A \(d\times m\) enclosure with columns \(\pi_j(u_j-\mathbb E u)/s\). |
| `bias_jacobian` | A \(d\times m\) enclosure of the true bias derivatives. |

`.jvp(target_direction, bias_direction)` accepts exact integer/`Fraction`
vectors of lengths \(d\) and \(m\). `.vjp(cotangent)` accepts an exact
length-\(d\) vector and returns `(target_box, bias_box)`. Cached queries
perform exact and interval arithmetic without a new transcendental evaluation,
nonlinear solve, or certificate seal. Constructing a new certificate retains
the rigorous-backend and provenance requirements of its sources.

```python
from omnibias.core.verified.gaussian_entropy_bias import (
    certify_gaussian_entropy_bias,
)
from omnibias.core.verified.gaussian_entropy_face import (
    certify_gaussian_entropy_face,
)
from omnibias.core.verified.gaussian_entropy_sensitivity import (
    certify_gaussian_entropy_sensitivity,
)
from omnibias.core.verified.gaussian_entropy_solver import solve_gaussian_entropy
from omnibias.core.verified.gaussian_entropy_unique import (
    certify_gaussian_entropy_unique,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import (
    certify_gaussian_posterior_domain,
)
from omnibias.core.verified.polyhedral_projection import (
    compile_polyhedral_projector,
)

model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
projector = compile_polyhedral_projector(2, ((1, 0), (2, 0)), (0, 0))
assert projector is not None
domain = certify_gaussian_posterior_domain(model, projector, (1, 0))
assert domain is not None
unique = certify_gaussian_entropy_unique(domain)
assert unique is not None and model.contraction_bound == 4
solution = solve_gaussian_entropy(unique, tolerance=1e-12, max_iterations=12)
assert solution is not None and solution.status == "tolerance"
face = certify_gaussian_entropy_face(solution.residual)
assert face is not None
sensitivity = certify_gaussian_entropy_sensitivity(face)
assert sensitivity is not None
bias = certify_gaussian_entropy_bias(sensitivity)
assert bias is not None

common = bias.jvp((0, 0), (10**400,)*3)
assert all(axis.lo == axis.hi == 0 for axis in common)
target_pullback, bias_pullback = bias.vjp((2**80, 0))
assert all(axis.lo == axis.hi == 0
           for axis in (*target_pullback, *bias_pullback))
# A nonconstant bias direction changes the tangential decision.
response = bias.jvp((0, 0), (0, 0, 1))
assert response[1].lo > 0
```

The cotangent is fixed input data. If it comes from a loss evaluated at an
approximate optimizer, its uncertainty or additional direct dependence on
parameters requires separate treatment. These certificates do not cover
moving centers, viscosity, age, geometry, higher derivatives, or switch
points lacking a stable-face proof.

The [finite bias-region certificates](gaussian-entropy-bias-region.md) use
an explicit bound on relative bias changes to prove global uniqueness,
bound decision displacement, and optionally preserve the same active face.
They provide a separate finite-region guarantee; the derivative above
remains a local quantity.

## Connection to heat and Burgers calculus

For the normalized heat mixture

\[
\tau(x,s)=(4\pi\nu s)^{-d/2}
\sum_j e^{b_j}e^{-\|x-a_j\|^2/(4\nu s)},
\qquad U=-2\nu\nabla\log\tau,
\]

both \(\tau\) and its fixed-bias-direction variation
\(\delta\tau=\sum_j\beta_j e^{b_j}K_j\) solve the heat equation.
Consequently

\[
\delta U=-2\nu\nabla(\delta\tau/\tau)
=-\operatorname{Cov}_\pi(a,\beta)/s
\]

satisfies the linearized viscous Burgers equation

\[
\partial_s\delta U+(U\cdot\nabla)\delta U
+(\delta U\cdot\nabla)U=\nu\Delta\delta U.
\]

This follows by differentiating the classical
[Cole–Hopf transformation](https://doi.org/10.1090/qam/42889).
The identity concerns the underlying field on physical space. Its
constrained decision derivative additionally requires the stable projection
geometry proved above; no boundary-value PDE solver or universal optimizer
is implied.
