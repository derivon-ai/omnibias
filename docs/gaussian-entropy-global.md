# Global Gaussian decisions through an entropy reduction

The Gaussian global optimization problem has a second exact reduction to
mixture weights. It separates the positive Gaussian sum into a quadratic
projection problem and a finite entropy term. The reduced objective is
differentiable at projection active-set switches whenever all weights are
positive. Its gradient requires the projection value, not a derivative of the
projection.

The certificate bounds the original global minimum, supplies an exactly
feasible point and covers every global minimizer. Nonconvexity and ties remain
possible. This is an alternative to the value-only
[projected-simplex search](gaussian-projection-global.md), with no general
speed or accuracy advantage asserted.

## Exact Gibbs reduction

Let \(C=\{x:Gx\le h\}\) be a nonempty closed polyhedron, and let
the Gaussian mixture have centers \(a_j\), biases \(b_j\), positive
viscosity \(\nu\) and age \(s\). With the heat prefactor omitted,

\[
f_v(x)=-2\nu\log\sum_j
\exp\left(b_j-\frac{\|x-a_j\|^2}{4\nu s}\right)-v\cdot x.
\]

For \(p\) in the closed probability simplex, put

\[
G(x,p)=\sum_jp_j\frac{\|x-a_j\|^2}{2s}-v\cdot x
+2\nu\sum_jp_j(\log p_j-b_j),\qquad 0\log0=0.
\]

The Gibbs variational identity gives
\(\min_pG(x,p)=f_v(x)\). The unique minimizing weights are the
strictly positive Gaussian posterior \(\pi(x)\). For fixed \(p\),
write \(\mu(p)=\sum_jp_ja_j\) and \(q=sv+\mu(p)\).
Completing the square gives the unique constrained minimizer

\[
x_p=\operatorname{Proj}_C(q),
\]

and reduced objective

\[
E(p)=\frac{\operatorname{dist}^2(q,C)
+\sum_jp_j\|a_j\|^2-\|\mu(p)\|^2}{2s}
-v\cdot\mu(p)-\frac{s\|v\|^2}{2}
+2\nu\sum_jp_j(\log p_j-b_j).
\]

Minimizing in either order proves \(\min_pE(p)=\min_{x\in C}f_v(x)\).
More specifically,

\[
E(p)=f_v(x_p)+2\nu\,\mathrm{KL}\bigl(p\,\|\,\pi(x_p)\bigr).
\]

Thus the entropy objective majorizes the original value at the projected
point. Every global pair satisfies both \(x=x_p\) and \(p=\pi(x)\).
Global spatial minimizers and global weight minimizers correspond bijectively;
all minimizing weights are strictly positive. Boundary weights are still
valid search points because \(E\) extends continuously to the closed
simplex. The statement does not make \(E\) convex.

The [posterior covariance criterion](gaussian-entropy-uniqueness.md) can
separately prove global uniqueness on the certified posterior region, including
some models whose original spatial potential is nonconvex. It uses exact
rational postprocessing of the domain and requires no minimizer search.

## Smooth envelope across projection switches

Squared distance to a nonempty closed convex set is continuously differentiable,
with gradient \(2(q-\operatorname{Proj}_Cq)\). On the simplex interior,

\[
\partial_{p_j}E
=\frac{\|a_j\|^2}{2s}-\frac{a_j\cdot x_p}{s}
+2\nu(\log p_j+1-b_j),
\]

up to a common offset in the simplex-normal direction. The projection may
change active sets or fail to have an ordinary Jacobian; this formula remains
valid. Higher smoothness is not guaranteed. At a zero weight, the entropy
derivative is singular, so an interior gradient formula must not be applied
to a boundary-touching search box.

## A domain containing all optimum posteriors

The projection argument bounds every global spatial minimizer by the compact
image \(\operatorname{Proj}_C(sv+\operatorname{conv}\{a_j\})\).
An outward enclosure of that image therefore bounds all posterior logits
needed at global minimizers. Equal Gaussian covariances make each logit
contrast \(\ell_i(x)-\ell_j(x)\) affine in \(x\).

For the stick-breaking parametrization, the conditional coordinate is

\[
z_j=\frac{p_j}{\sum_{i\ge j}p_i}
=\frac1{1+\sum_{i>j}\exp(\ell_i-\ell_j)}.
\]

Rigorous contrast bounds give an interval for each \(z_j\) containing
its value at **every** global minimizer. Restricting the search to the product
of these intervals preserves all global optimum posteriors. Points elsewhere
in this box need not themselves be Gaussian posteriors of their projections;
their positive KL discrepancy is part of the objective.

`certify_gaussian_posterior_domain(model, projector, target)` in
`omnibias.core.verified.gaussian_posterior_domain` returns a
`GaussianPosteriorDomain`. It stores the original `model`, `projector`,
`target`, `physical_bounds`, `parameter_box`, `gradient_safe` and a
certificate. A mathematically positive posterior may be too small for a
strict binary64 bound. Such a domain remains valid, but its box can touch zero
or one; the implementation does not manufacture strict endpoints.

When `gradient_safe` is true, the optimizer propagates the envelope gradient
through the stick-breaking map with interval arithmetic. Otherwise it uses
value-only optimization and the continuous extension of \(p\log p\).
Boundary fallback is a numerical enclosure limitation, not evidence of an
exact zero posterior. A single mode has no genuine conditional coordinate;
its dummy singleton interval is marked safe and has zero parameter derivative.

## Reported normalization and global certificate

The API removes common constants exactly. Let \(c=a_0\),
\(r=x-c\), \(\delta_j=a_j-c\), and \(\mu_\delta=\sum_jp_j\delta_j\).
Its original spatial objective is

\[
F_v(x)=\frac{\|r\|^2}{2s}-v\cdot r
-2\nu\log\sum_j\exp\left(
\frac{\delta_j\cdot r}{2\nu s}+(b_j-b_0)
-\frac{\|\delta_j\|^2}{4\nu s}\right).
\]

With \(q=c+sv+\mu_\delta\), the corresponding entropy objective is

\[
E_c(p)=\frac{\operatorname{dist}^2(q,C)
+\sum_jp_j\|\delta_j\|^2-\|sv+\mu_\delta\|^2}{2s}
+2\nu\sum_jp_j\bigl(\log p_j-(b_j-b_0)\bigr).
\]

It obeys \(\min E_c=\min_C F_v\) and the same KL identity. Exact
centering prevents a large common bias or translation from being rounded
before cancellation.

The quadratic part can also be bounded without subtracting two large squared
quantities. Set \(w=sv+\mu_\delta\) and

\[
H(w)=\operatorname{dist}^2(c+w,C)-\|w\|^2
=\min_{r\in C-c}\bigl(\|r\|^2-2w\cdot r\bigr).
\]

For an exact reference \(w_0\), let
\(r_0=\operatorname{Proj}_C(c+w_0)-c\) and
\(T(w)=H(w_0)-2r_0\cdot(w-w_0)\). Then

\[
T(w)-\|w-w_0\|^2\le H(w)\le T(w).
\]

The upper bound tests the feasible point \(r_0\). For the lower bound,
projection optimality gives
\((r_0-w_0)\cdot(r-r_0)\ge0\); drop this nonnegative term and
complete the square. This bound remains valid at every projection switch and
requires no square root or projection Jacobian. Its quadratic remainder also
proves \(\nabla H(w_0)=-2r_0\) directly; continuity follows from continuity
of projection.

`certify_gaussian_entropy_minimum(domain, *, value_tolerance=1e-6,
max_boxes=255)` in `omnibias.core.verified.gaussian_entropy_minimum`
returns `GaussianEntropyMinimum` or refuses. Its records distinguish the two
objectives:

| Field | Meaning |
| --- | --- |
| `minimum` | Enclosure of the original global minimum \(\min_C F_v=\min E_c\). |
| `sample_point`, `sample_value` | Exact feasible spatial witness and its original \(F_v\) enclosure. |
| `sample_entropy` | Enclosure of \(E_c\) at the same sampled weights. |
| `candidate_boxes` | Spatial cover containing every original global minimizer. |
| `candidate_parameters` | Retained conditional-weight boxes. |
| `parameter_result` | Bounded optimizer diagnostics for the entropy objective. |
| `gradient_used` | Whether the certified interior gradient was supplied to the bounded optimizer. |
| `status`, `gap_upper` | Final outward global gap and termination status. |

The feasible original sample value can improve the upper bound because
\(F_v(x_p)\le E_c(p)\). The final gap and status use that bound. A
candidate can be pruned only when its entropy lower bound strictly exceeds a
feasible upper bound: every original global minimizer still has a surviving
posterior preimage. Spatial boxes can also contain infeasible or nonoptimal
points; the exact witness alone is asserted feasible.

`"value_tolerance"` confirms the requested global value gap.
`"box_budget"` or `"resolution_limit"` preserves sound bounds and coverage
without asserting that accuracy. Both source budgets apply. Missing rigorous
transcendental support, damaged dependencies, nonfinite evaluation and resource
limits cannot produce a partial global certificate. Refusal proves neither
infeasibility nor the absence of an optimizer.

## Executable examples

The four-center product mixture below has four tied global minima. Its
conditional posterior bounds are strictly interior, so the entropy search
can use a rigorous gradient even though the original potential is nonconvex.
The modest box budget does not promise a requested value accuracy.

```python
from fractions import Fraction as F
from itertools import product

from omnibias.core.verified.gaussian_entropy_minimum import (
    certify_gaussian_entropy_minimum,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import (
    certify_gaussian_posterior_domain,
)
from omnibias.core.verified.polyhedral_projection import (
    compile_polyhedral_projector,
)

model = GaussianHeatMap(
    tuple(product((-1, 1), repeat=2)), (0, 0, 0, 0), viscosity=F(1, 4),
)
projector = compile_polyhedral_projector(2, (), ())
assert projector is not None
domain = certify_gaussian_posterior_domain(model, projector, (0, 0))
assert domain is not None and domain.gradient_safe
minimum = certify_gaussian_entropy_minimum(domain, max_boxes=31)
assert minimum is not None and minimum.candidate_boxes
assert minimum.parameter_result.gradient_evaluations > 0
if minimum.status == "value_tolerance":
    assert minimum.gap_upper <= 1e-6
```

With an extremely small mode weight, the outward domain can touch a boundary.
The same global-value contract remains available through continuous entropy
evaluation and value-only search.

```python
small_weight = GaussianHeatMap(((0, 0), (0, 0)), (0, -10000))
target = (F(1, 3), F(-2, 5))
boundary_domain = certify_gaussian_posterior_domain(
    small_weight, projector, target,
)
assert boundary_domain is not None and not boundary_domain.gradient_safe
boundary_minimum = certify_gaussian_entropy_minimum(
    boundary_domain, max_boxes=7,
)
assert boundary_minimum is not None
assert boundary_minimum.parameter_result.gradient_evaluations == 0
assert boundary_minimum.sample_point == target
```

The integration tests independently substitute original Gaussian distances
into the joint objective and verify its KL difference from the original
spatial value. They also check the envelope gradient against an explicit
piecewise formula across a nonsmooth projection switch, original-gauge
values under large common biases, and coverage of all known tied minima.

## Scope and prior work

The construction supports the declared finite positive Gaussian family on
rational polyhedra at fixed positive viscosity and age. It does not require
collinear centers or a globally convex potential. The number of mixture
weights, projection active bases and interval boxes still controls cost.
Entropy or quadratic interval dependencies can yield loose bounds; smoother
formulas alone do not establish faster optimization.

This is a classical Gibbs variational identity and convex projection envelope,
combined with rigorous domain bounds and interval global optimization.
See [Wainwright and Jordan, Section 3.6](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf)
for exponential-family variational structure and
[Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
for convex optimality and quadratic projection. No new general theorem or
external formal theorem-prover verification is claimed. A global minimizer
cover also does not supply a differentiable selection when optima tie.
