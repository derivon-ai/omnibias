# Certified finite Gaussian bias regions

The [bias derivative certificate](gaussian-entropy-bias.md) concerns an
infinitesimal change at a stable decision. This certificate instead proves
uniqueness for a whole finite region of original Gaussian mixture biases.
A separate face certificate can prove that every decision in that region
uses the same constraint face. Centers, target, viscosity, heat age and
polyhedron remain fixed.

Only relative biases matter. Write the perturbed biases as \(b+\delta b\)
and assume

\[
\operatorname{osc}(\delta b)
=\max_j\delta b_j-\min_j\delta b_j\le\omega.
\]

The region allows arbitrarily large common shifts. It concerns logarithmic
positive mixture weights \(e^{b_j}\), not arbitrary neural-network biases.

## Transfer the existing covariance bound

Let \(C\) be the fixed nonempty closed convex polyhedron. Every constrained
global minimizer satisfies

\[
x=P_C\bigl(sv+\mathbb E_{\pi_b(x)}a\bigr).
\]

Consequently every minimizing point, for every finite bias vector, lies in
\(S=P_C(sv+\operatorname{conv}\{a_j\})\). The source posterior-domain
certificate encloses \(S\) in a convex physical box \(K\), independent of
biases. Its probability bounds hold at **every** point of \(K\).

The [uniqueness certificate](gaussian-entropy-uniqueness.md) supplies an
exact reference \(c\) and a bound

\[
\mathbb E_{\pi_b(x)}\|a-c\|^2\le B<H:=2\nu s
\quad(x\in K),\qquad D=\max_j\|a_j-c\|^2.
\]

At a fixed physical point, perturbing the biases simply reweights the
posterior by numbers whose ratio is at most \(t=e^\omega\). The sharp
universal transfer using only \(0\le X\le D\) and \(\mathbb E X\le B\) is

\[
\boxed{B_\omega=\frac{DtB}{D+(t-1)B}
=\frac{DB}{B+(D-B)e^{-\omega}}}.
\]

To prove it, set \(q=B_\omega\). For a reweighting \(1\le w\le t\),
\(w(X-q)\le\phi(X)=\max\{X-q,t(X-q)\}\). Convexity puts \(\phi\)
below its chord between \(0,D\), whose expected value is at most
\(-q+(B/D)[tD-(t-1)q]=0\). Divide by \(\mathbb E w>0\).
Two-point mass at \(0,D\), with mass \(B/D\) at \(D\) and maximum tilt
toward it, attains the bound. This probability bound is sharp; the resulting
Gaussian uniqueness test is only sufficient.

Covariance is bounded by the expected squared distance from any reference.
Thus \(B_\omega<H\) gives, throughout the same \(K\),

\[
\rho_\omega=B_\omega/H<1,\qquad
\nabla^2 F_{b+\delta b}\succeq
\mu_\omega I,\qquad
\mu_\omega=\frac{1-\rho_\omega}{s}>0.
\]

All global minimizers lie in \(K\), so this proves exactly one. It also
gives a contraction bound for the projected posterior-mean map on \(K\).
The objective can still be nonconvex outside this box.

When \(D<H\), all finite bias perturbations satisfy the sufficient test.
When \(D=H\) and \(B<H\), the mathematical margin is positive for every
finite \(\omega\), but tends to zero. When \(D>H\) and \(B>0\), the
condition is

\[
\omega<\log\frac{H(D-B)}{B(D-H)}.
\]

Zero-radius and zero-moment cases are handled directly. A common bias change
has \(\omega=0\), retains \(B\), and has zero decision displacement.

## Finite displacement and face stability

Let \(\Delta=\max_{i,j}\|a_i-a_j\|\) and
\(\rho_0=B/(2\nu s)\) be the **original** contraction bound. With total
variation defined as \(\mathrm{TV}(p,p')=\tfrac12\|p-p'\|_1\), bounded
log-weight oscillation gives

\[
\mathrm{TV}(\pi_{b+\delta b}(x),\pi_b(x))
\le\tanh(\omega/4).
\]

This is the Hilbert-distance comparison in
[Cohen–Fausti, Theorem 5.1 and Remark 13](https://arxiv.org/html/2309.02413v2#S5.Thmtheorem1).
Their total-variation norm is the full \(\ell^1\) norm; dividing their
bound by two gives the convention above. The posterior mean therefore
changes by at most \(e=\Delta\tanh(\omega/4)\) at a fixed point.

Write \(x_0,x_\delta\) for the original and perturbed minimizers, and
\(q_b=sv+\mathbb E_{\pi_b(x_b)}a\) for their projection inputs. Projection
is nonexpansive, and the original posterior mean is \(\rho_0\)-Lipschitz
on \(K\). Hence

\[
\boxed{\|x_\delta-x_0\|,\ \|q_{b+\delta b}-q_b\|
\le d_\omega:=\frac{\Delta\tanh(\omega/4)}{1-\rho_0}}.
\]

This displacement estimate holds for every perturbed global minimizer even
without the transferred uniqueness test. The certificate records outward
bounds, including the square of this radius.

The optional face proof enlarges the original certified projection-input
box by \(d_\omega\), then replays the existing exact tangent projector,
independent normal basis and full positive normal-cone witness. Strict
positivity of every active witness multiplier and every inactive slack
proves that the projection face is unchanged throughout the bias region.
Redundant tight rows and opposing rows representing equalities are allowed.
No new projection solve is needed, and no agreement of the faces is assumed
before these tests pass.

Uniform positive curvature and strict face margins give a locally smooth
global decision at each bias in the certified region. This remains an
existential local smoothness statement at each point, including the boundary
of the region. It does not make the original Jacobian correct for a finite
step or provide a Taylor remainder.

## API and example

`certify_gaussian_entropy_bias_region(unique, oscillation)` in
`omnibias.core.verified.gaussian_entropy_bias_region` accepts an intact
uniqueness certificate and a nonnegative exact integer or `Fraction`.
It returns `GaussianEntropyBiasRegion` or refuses.

| Field | Meaning |
| --- | --- |
| `oscillation` | The declared finite relative-bias radius. |
| `max_squared_radius`, `tilt_decay` | \(D\) and an enclosure of \(e^{-\omega}\). |
| `variance_upper_bound` | An outward rational upper bound on \(B_\omega\). |
| `uniqueness_margin`, `strong_convexity_bound` | Positive bounds \(H-B_\omega\), \(\mu_\omega\). |
| `total_variation_bound` | Rational upper bound on \(\tanh(\omega/4)\). |
| `displacement_squared_bound`, `displacement_bound` | Rational squared bound and outward binary64 radius for both decision and projection-input drift. |

`certify_gaussian_entropy_bias_face(face, region)` in
`omnibias.core.verified.gaussian_entropy_bias_face` requires the same source
uniqueness proof. Its result stores `projection_input_bounds`,
`optimizer_bounds`, `projection_multipliers` and `slacks` for the entire
bias region, together with its bound proof.

```python
from fractions import Fraction
from omnibias.core.verified.gaussian_entropy_bias_face import (
    certify_gaussian_entropy_bias_face,
)
from omnibias.core.verified.gaussian_entropy_bias_region import (
    certify_gaussian_entropy_bias_region,
)
from omnibias.core.verified.gaussian_entropy_face import certify_gaussian_entropy_face
from omnibias.core.verified.gaussian_entropy_solver import solve_gaussian_entropy
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector

model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -20, -20))
projector = compile_polyhedral_projector(2, ((1, 0), (2, 0)), (0, 0))
assert projector is not None
domain = certify_gaussian_posterior_domain(model, projector, (1, 0))
assert domain is not None
unique = certify_gaussian_entropy_unique(domain)
assert unique is not None and model.contraction_bound == 4
solution = solve_gaussian_entropy(unique, tolerance=1e-12, max_iterations=12)
assert solution is not None
face = certify_gaussian_entropy_face(solution.residual)
assert face is not None

region = certify_gaussian_entropy_bias_region(unique, Fraction(1, 8))
assert region is not None and region.uniqueness_margin > 0
stable = certify_gaussian_entropy_bias_face(face, region)
assert stable is not None and stable.optimizer_bounds[0] == (0, 0)
# Every finite bias change with max(delta_b)-min(delta_b) <= 1/8
# retains this same global optimizer face x[0] = 0.
common = certify_gaussian_entropy_bias_region(unique, 0)
assert common is not None and common.displacement_squared_bound == 0
```

The two producers require rigorous source provenance and seal new proofs.
Refusal can mean insufficient margins, resource limits, or numerical loss
of a positive lower bound; it does not establish nonuniqueness or a face
change. In particular, an underflowed lower bound on \(e^{-\omega}\)
may prevent certification when \(D=2\nu s\), even though the exact finite
bias margin is positive. The \(D<2\nu s\) case can retain a positive margin
without that lower bound.

These results combine bounded reweighting, covariance bounds, convex
projection and contraction arguments. They do not establish a new general
theorem, a speed advantage, an activation-independent calculus, or finite
step accuracy for the [infinitesimal bias Jacobian](gaussian-entropy-bias.md).

The [joint target and bias extension](gaussian-entropy-joint-region.md)
also covers a finite Euclidean target ball. Its curvature domain expands
with the target radius, and its displacement bound accounts for the
original field on that larger domain.
