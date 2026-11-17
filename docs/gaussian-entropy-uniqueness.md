# Uniqueness from a certified posterior domain

A Gaussian heat potential can be nonconvex on the whole space and still have
exactly one global minimizer on a polyhedron. The certified posterior domain
can prove this without requiring the global diameter contraction condition.
The test bounds a weighted squared radius over all possible optimum
posteriors, then compares it with the entropy curvature.

This certificate proves uniqueness for the declared model, target and
constraints. It does not locate the optimizer, bound its derivative or prove
that an ordinary derivative exists at a constraint switch.

## The conditional-weight box has a convex image

The [entropy reduction](gaussian-entropy-global.md) supplies a box
\(L_j\le z_j\le U_j\) of stick-breaking coordinates containing the
posterior of every global minimizer. Write

\[
T_j=\sum_{i\ge j}p_i.
\]

Its image in the probability simplex is exactly the polytope

\[
P=\left\{p\ge0:\sum_jp_j=1,\quad
L_jT_j\le p_j\le U_jT_j\quad(j<m-1)\right\}.
\]

Indeed, stick breaking gives \(p_j=z_jT_j\), which satisfies these linear
inequalities. Conversely, when \(T_j>0\), take \(z_j=p_j/T_j\).
If \(T_j=0\), all remaining weights vanish and any allowed subsequent
coordinates reproduce them. This also proves the claim for boundary or
degenerate intervals. A one-mode family has the singleton probability vector.

Thus this particular nonlinear parametrization of a box has a **convex**
image. No convex-hull relaxation is needed. Convexity of an objective in
\(p\) would still not imply convexity after nonlinear substitution into
the coordinates \(z\).

## An exact, inexpensive sufficient bound

Choose an exact rational reference point \(c\) and define

\[
d_j=\|a_j-c\|^2,\qquad
B=\max_{p\in P}\sum_jp_jd_j.
\]

The minimum and maximum of this linear expectation can be computed by a
scalar tail recurrence. Start with the last value \(d_{m-1}\). At step
\(j\), interpolate
\(z_jd_j+(1-z_j)e_{j+1}\) over its two coordinate endpoints and
the two previously computed tail endpoints. The resulting minimum and maximum
are exact: the tail and current conditional coordinate are independently
chosen, and the expression is affine in each. This uses rational arithmetic;
it does not require a generic linear optimizer or a new projection query.

The sufficient uniqueness condition is

\[
\boxed{B<2\nu s.}
\]

This weighted squared-radius bound dominates the posterior covariance:

\[
\operatorname{tr}\operatorname{Cov}_p(a)
\le\sum_jp_j\|a_j-c\|^2\le B,
\qquad \operatorname{Cov}_p(a)\preceq BI.
\]

It can be conservative. Reference choice matters, and a failed test does not
prove multiple optima. Sharper matrix covariance bounds are possible in
principle, but this API certifies the stated scalar bound.

## Proof without differentiating projection

Let \(E\) be the exact Gibbs entropy envelope and
\(x_p=\operatorname{Proj}_C(sv+\sum_jp_ja_j)\). For positive
\(p,p'\in P\), put \(u=p-p'\), \(Au=\sum_ju_ja_j\), and

\[
J=\sum_ju_j(\log p_j-\log p'_j).
\]

The envelope gradient and nonexpansiveness of Euclidean projection give

\[
u\cdot\bigl(\nabla E(p)-\nabla E(p')\bigr)
=2\nu J-\frac{Au\cdot(x_p-x_{p'})}{s}
\ge2\nu J-\frac{\|Au\|^2}{s}.
\]

For every point \(p(t)=p'+tu\) on their segment, weighted
Cauchy–Schwarz gives

\[
\|Au\|^2
=\left\|\sum_ju_j(a_j-c)\right\|^2
\le B\sum_j\frac{u_j^2}{p_j(t)}.
\]

Convexity of \(P\) is essential here. Integrating over \(0\le t\le1\)
yields \(\|Au\|^2\le BJ\), hence

\[
u\cdot\bigl(\nabla E(p)-\nabla E(p')\bigr)
\ge\left(2\nu-\frac Bs\right)J>0\quad(p\ne p').
\]

Every global optimum has a strictly positive Gibbs posterior, even when its
numerical domain touches a boundary. At such an optimum the simplex-tangent
gradient vanishes. Two different global posteriors would therefore make the
left side zero, contradicting the strict inequality. The unique pairing
between optimum posteriors and spatial optima proves uniqueness of the
original constrained global minimum.

The argument uses only projection values and nonexpansiveness. It survives
active-set switches and does not assume a projection Jacobian. It also does
not assert global convexity of the original spatial potential or of \(E\)
outside \(P\).

## API and executable examples

`certify_gaussian_entropy_unique(domain, *, reference=None)` in
`omnibias.core.verified.gaussian_entropy_unique` accepts a certified
`GaussianPosteriorDomain`. The default reference is the first original
center. An explicit reference must use exact integers or `Fraction` values.
The result, `GaussianEntropyUnique`, records:

| Field | Meaning |
| --- | --- |
| `domain`, `reference` | Bound source problem and selected exact reference. |
| `squared_radii` | Exact \(d_j=\|a_j-c\|^2\). |
| `expectation_bounds` | Exact minimum and maximum of \(\sum_jp_jd_j\) over \(P\). |
| `variance_upper_bound` | The upper endpoint \(B\). |
| `uniqueness_margin` | Positive \(2\nu s-B\). |
| `entropy_monotonicity_lower_bound` | Positive factor \(2\nu-B/s\) in the proof. |

The factory performs exact postprocessing of the existing domain. It makes
no new transcendental, projection or minimizer evaluation. Damaged source
evidence, exhausted resources or an insufficient bound returns `None`;
malformed inputs raise. Existing certificate provenance is preserved.
Refusal does not establish nonuniqueness.

The following genuinely nonconvex family has diameter contraction bound
\(q=4\). Every possible optimum nevertheless has sufficiently concentrated
posterior weights for the uniqueness test to pass.

```python
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

model = GaussianHeatMap(((0, 0), (4, 0), (0, 4)), (0, -8, -8))
projector = compile_polyhedral_projector(2, (), ())
assert projector is not None and model.contraction_bound == 4
domain = certify_gaussian_posterior_domain(model, projector, (0, 0))
assert domain is not None
unique = certify_gaussian_entropy_unique(domain)
assert unique is not None
assert unique.variance_upper_bound < 1
assert unique.uniqueness_margin > 1
```

To check the example analytically, every global optimizer lies in the triangle
spanned by the centers. The two nonreference logit contrasts are at most
\(-4\). Since a finite Taylor sum proves \(e^4>32\), their total posterior
mass is less than \(1/17\). Thus \(B<16/17<2\) with reference zero.
In contrast, at \(x=(6,0)\), the first-coordinate variance is
\(16p(1-p)\), where \(p=1/(2+e^{-12})\in(1/3,1/2)\).
The spatial Hessian entry \(1-8p(1-p)\) is negative there. Uniqueness
therefore does not follow from global spatial convexity.

A different reference can make an otherwise inconclusive bound useful:

```python
symmetric = GaussianHeatMap(((-1,), (1,)), (0, 0))
line = compile_polyhedral_projector(1, (), ())
assert line is not None
symmetric_domain = certify_gaussian_posterior_domain(symmetric, line, (0,))
assert symmetric_domain is not None
assert certify_gaussian_entropy_unique(symmetric_domain) is None
centered = certify_gaussian_entropy_unique(symmetric_domain, reference=(0,))
assert centered is not None and centered.expectation_bounds == (1, 1)
```

Boundary-touching numerical posterior bounds do not invalidate this theorem:
it applies to the positive true optimum posteriors, while the squared-radius
bound holds over the entire closed outer domain. This differs from evaluating
an entropy gradient at a numerical zero probability.

The [certified projected solver](gaussian-entropy-solver.md) uses this
uniqueness certificate to bound an exact feasible proposal's distance and
objective gap to the optimizer. Its posterior-mean iteration checks every
candidate with that residual bound, including at constraint switches.

A further [stable-face certificate](gaussian-entropy-sensitivity.md) proves
local face persistence before enclosing derivatives of the global decision.

The [finite bias-region certificate](gaussian-entropy-bias-region.md) can
transfer this uniqueness guarantee to every bounded-spread log-weight
change, without recomputing a posterior domain at each perturbed model.

## Scope and classical ingredients

The result proves uniqueness only for the supplied fixed problem. It provides
no optimizer location, perturbation radius, target Jacobian or selected-branch
derivative. A unique optimizer can still encounter a constraint switch.

The argument combines standard entropy/Fisher geometry, the Gibbs variational
identity and nonexpansive convex projection. See
[Wainwright and Jordan, Sections 3.5–3.6](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf)
and [Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
No new general theorem, universal efficiency result or external formal
theorem-prover verification is claimed.
