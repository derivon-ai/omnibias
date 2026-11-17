# Certified finite target and bias regions

The [finite bias-region certificate](gaussian-entropy-bias-region.md) extends
to simultaneous target changes. Fix the Gaussian centers, viscosity
\(\nu>0\), heat age \(s>0\), and nonempty closed convex polyhedron \(C\).
The new certificate covers every finite pair

\[
\|\delta v\|_2\le V,\qquad
\operatorname{osc}(\delta b)\le\omega.
\]

Common bias shifts remain unrestricted. The result proves unique global
decisions and bounds their movement. A separate face certificate can prove
that the constraint face remains fixed. Neither result treats the original
Jacobian as a finite-step approximation.

## Expand the physical domain in the Euclidean metric

Let \(K\) be the convex physical box from the original posterior-domain
certificate. It encloses
\(P_C(sv+\operatorname{conv}\{a_j\})\), and the original posterior bounds
hold throughout \(K\). Nonexpansiveness of projection gives

\[
P_C(s(v+\delta v)+\operatorname{conv}\{a_j\})
\subset M:=K+B_2(sV).
\]

Indeed, compare the two projections using the same point of the center
hull. Every new stationary point, hence every new global minimizer, lies in
\(M\). This Minkowski sum is convex even when the projected center-hull
image is not.

For \(x\in M\), choose \(y\in K\) with \(\|x-y\|\le sV\).
The change of an original logit contrast is exactly

\[
(\ell_i(x)-\ell_j(x))-(\ell_i(y)-\ell_j(y))
=\frac{(a_i-a_j)\cdot(x-y)}{2\nu s}.
\]

Writing \(\Delta=\max_{i,j}\|a_i-a_j\|\), the spatial tilt therefore
has oscillation at most \(\eta_r=\Delta V/(2\nu)\). Adding the original
bias perturbation gives total oscillation at most \(\eta_r+\omega\).

The source uniqueness proof supplies
\(\mathbb E_{\pi_b(y)}\|a-c\|^2\le B<H=2\nu s\) on \(K\).
With \(D=\max_j\|a_j-c\|^2\), define the transferred moment bound

\[
\mathcal B(\eta)=\frac{DB}{B+(D-B)e^{-\eta}},
\]

with the direct zero limits. The bounded-tilt argument from the bias-region
page proves

\[
\operatorname{Cov}_{\pi_b(x)}(a)\preceq\mathcal B(\eta_r)I,
\qquad
\operatorname{Cov}_{\pi_{b+\delta b}(x)}(a)
\preceq\mathcal B(\eta_r+\omega)I
\quad(x\in M).
\]

If \(\mathcal B(\eta_r+\omega)<H\), the perturbed objective has Hessian
at least

\[
\mu I,\qquad
\mu=\frac{1-\mathcal B(\eta_r+\omega)/H}{s}>0
\]

on \(M\). Coercivity gives existence; all minimizers lie in \(M\), so
strong monotonicity and the convex normal cone give exactly one global
minimizer. The objective need not be convex outside \(M\).

**The coordinate box \(K+[-sV,sV]^d\) is only an outer enclosure.** Its
corners can lie outside \(M\); the curvature claim does not extend to that
whole box. Replacing the Euclidean expansion with the coordinate box in
this proof would require an expansion radius as large as \(\sqrt d\,sV\).

## Decision and projection-input movement

Let \(\rho_r=\mathcal B(\eta_r)/H\). The original posterior mean is
\(\rho_r\)-Lipschitz on the enlarged convex domain. At a fixed point,
bias oscillation \(\omega\) changes its mean by at most
\(\Delta\tanh(\omega/4)\). Here total variation means half the
\(\ell^1\) distance, as in the normalized form of
[Cohen–Fausti, Theorem 5.1](https://arxiv.org/html/2309.02413v2#S5.Thmtheorem1).

Applying projection nonexpansiveness to the two fixed-point equations gives

\[
\boxed{\|x_{v+\delta v,b+\delta b}-x_{v,b}\|,
\ \|q_{v+\delta v,b+\delta b}-q_{v,b}\|
\le\frac{sV+\Delta\tanh(\omega/4)}{1-\rho_r}},
\]

where \(q=sv+\mathbb E_{\pi_b(x)}a\). The denominator uses the original
field's contraction bound **on \(M\)**, not its smaller bound on \(K\),
and not the perturbed field's bound. At \(V=0\), this recovers the bias-only
theorem. A pure normal target change can leave \(x\) fixed while moving
\(q\), which is why the second bound is needed for face stability.

The optional face certificate enlarges the source projection-input bounds
by this drift, then replays the cached exact normal-cone witness, normal
basis and tangent projector. Positive active witness multipliers and
positive inactive slacks certify the same projection face throughout the
joint region. Optimizer boxes may be intersected with the expanded
coordinate bounds for containment, without asserting curvature at every
point of that outer box. Refusal of a face proof does not negate uniqueness.

## API and example

`certify_gaussian_entropy_joint_region(unique, bias_oscillation, target_radius)`
in `omnibias.core.verified.gaussian_entropy_joint_region` accepts exact
nonnegative integer/`Fraction` radii and returns
`GaussianEntropyJointRegion` or refuses.

Its fields include `expansion_radius`, `diameter_bound`,
`spatial_oscillation`, `total_oscillation`,
`spatial_variance_upper_bound`, `variance_upper_bound`,
`spatial_contraction_bound`, `contraction_bound`,
`strong_convexity_bound`, `uniqueness_margin`, `total_variation_bound`,
`forcing_bound`, `displacement_squared_bound` and `displacement_bound`.
Bounds are outward; in particular `diameter_bound` can exceed the exact
diameter slightly.

`expanded_coordinate_bounds` encloses \(M\). For an exact point,
`.contains_physical(point)` instead checks the true membership condition

\[
\sum_i\operatorname{dist}(x_i,[K_i^-,K_i^+])^2\le(sV)^2.
\]

`.contains_parameters(target_change, bias_change)` checks the exact
Euclidean target radius and relative-bias oscillation. These cached queries
validate their bound proof and need no new transcendental evaluation.

`certify_gaussian_entropy_joint_face(face, region)` in
`omnibias.core.verified.gaussian_entropy_joint_face` requires the same source
uniqueness proof. Its result retains the original face, region, expanded
`projection_input_bounds`, `optimizer_bounds`, `projection_multipliers`
and `slacks`.

```python
from fractions import Fraction
from omnibias.core.verified.gaussian_entropy_joint_region import (
    certify_gaussian_entropy_joint_region,
)
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector

model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
projector = compile_polyhedral_projector(2, (), ())
assert projector is not None
domain = certify_gaussian_posterior_domain(model, projector, (0, 0))
assert domain is not None and domain.physical_bounds == ((0, 4), (0, 4))
unique = certify_gaussian_entropy_unique(domain)
assert unique is not None and model.contraction_bound == 4
region = certify_gaussian_entropy_joint_region(
    unique, Fraction(1, 2), Fraction(1, 10),
)
assert region is not None and region.uniqueness_margin > 0
assert region.contains_parameters(
    (Fraction(-3, 50), Fraction(-2, 25)), (0, Fraction(1, 2), 0),
)
# A point on the rounded Euclidean boundary is included.
assert region.contains_physical((Fraction(-3, 50), Fraction(-2, 25)))
# This corner is in the outer coordinate box, but not in the certified M.
assert not region.contains_physical((Fraction(-1, 10), Fraction(-1, 10)))
```

New producers retain the rigorous backend and source-provenance requirements.
Failure can mean insufficient margins, lost numerical resolution
or exhausted budgets. It does not prove nonuniqueness. This is a finite
region for a specified Gaussian decision family, assembled from bounded
reweighting, convex projection and contraction arguments. It is not a
universal PDE solver, a new general theorem claim, or a finite Taylor error
bound for the original [bias and target Jacobians](gaussian-entropy-bias.md).
