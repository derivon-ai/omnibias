# Gaussian switching derivatives with grouped logits

An exact switching equilibrium need not have uniform Gaussian probabilities.
Equal-logit groups with a shared centroid give an exact posterior mean,
velocity and KKT classification even when the Hessian is nonrational.
The directional response can then be enclosed by a projected residual,
without solving a linear system at each query.

This extends [exact critical-cone derivatives](gaussian-polyhedral-directional.md).
The source, heat age, viscosity and constraints remain fixed; only the
optimization target changes.

The same critical-face compiler also accepts
[projected equilibria](gaussian-projected-equilibrium.md), whose group
centroids agree only after projection onto the relevant constraints.
That extension retains the between-group covariance in the full Hessian
while proving that it vanishes from the projected derivative equations.

## Exact classification with nonuniform probabilities

At an exact rational point \(X\), the original Gaussian scores are

\[
\ell_j=b_j-\frac{\|X-a_j\|^2}{4\nu s}.
\]

Group indices having exactly equal scores. Write \(\alpha_g\) for
each group's score relative to the first score, \(n_g\) for its size,
and \(c_g\) for its exact centroid. Then

\[
\mu(X)=\frac{\sum_g n_gc_g e^{\alpha_g}}
                 {\sum_g n_g e^{\alpha_g}}.
\]

If every \(c_g=c\), then \(\mu(X)=c\) exactly. The individual
probabilities generally remain transcendental; no rounded softmax is
used to establish the mean. Consequently

\[
U(X)=\frac{X-c}{s}
\]

is rational, and feasibility, stationarity, multiplier signs and zero
slacks can be checked exactly against the original model. The proposed
polyhedral face must provide valid KKT multipliers. With
\(q_{\mathrm{base}}<1\), these conditions certify the unique global
optimum over the declared polyhedron.

All tight inequalities are retained, including those omitted from the
proposed face. Positive multipliers identify strong-active rows; zero
multipliers at tight rows identify weak-active rows. This classification
uses exact equalities, not an interval that happens to contain zero.

The factory `certify_gaussian_grouped_equilibrium(face, target, point)`
in `omnibias.core.verified.gaussian_grouped_equilibrium` accepts exact
integer or `Fraction` target and point coordinates. Its
`GaussianGroupedEquilibrium` retains the original `face`, `target`,
`point`, `groups`, `group_logits`, `centroid`, `centered_centers`,
`velocity`, primal multipliers and slacks, all tight/strong/weak indices,
`strong_convexity_bound`, and its `certificate`. Groups follow first
appearance, and their logits are relative to the first original mode.
This artifact contains neither a posterior nor a Hessian. Its centroid
identity holds at the declared point, not throughout a neighborhood.

Malformed exact inputs raise. Unequal group centroids, failed KKT or
feasibility checks, failed base contraction, damaged dependencies or
exhausted budgets return `None`. The grouped equilibrium check itself
does not evaluate exponentials.

The shared-centroid condition has a precise limitation. For rational
scores and centers, it is necessary and sufficient for the **full mean**
to be algebraic. If a coordinate \(\mu_i\) were algebraic, then

\[
\sum_g n_g(c_{g,i}-\mu_i)e^{\alpha_g}=0
\]

forces every coefficient to vanish by the Lindemann--Weierstrass theorem.
Conversely shared centroids give the rational mean above. This is a
corollary of the classical independence theorem, whose precise statement
and proof appear in
[Popescu's proof of Lindemann--Weierstrass](https://arxiv.org/abs/2306.14352).
It does not characterize every exact constrained equilibrium: a projected
mean can be rational while the full mean is not, and nonrational input
biases fall outside this argument.

## Interval Hessian, exact tangent geometry

The spatial Hessian is

\[
H=D_xU(X)=\frac1s\left(I-
 \frac{\sum_j p_j(a_j-c)(a_j-c)^T}{2\nu s}\right).
\]

The exact centroid is retained in this expression. Rigorous probability
enclosures give an interval matrix containing \(H\); no rational
Hessian is assumed. On the base equality tangent its quadratic form is
bounded below by

\[
m=\frac{1-q_{\mathrm{base}}}{s}>0.
\]

The ambient Hessian may be indefinite. For target direction \(\xi\),
the true one-sided Hadamard derivative is the unique critical-cone
quadratic-program minimizer

\[
\underset{w\in K}{\arg\min}\;
\frac12w^THw-\xi^Tw,
\qquad
K=\{w:Cw=0,\ G_+w=0,\ G_0w\le0\}.
\]

Choose weak rows \(\mathcal B\) to impose as equalities and form
\(E=[C;G_+;G_{\mathcal B}]\). The selected rows must be independent.
Compile the exact orthogonal projector \(P\) onto \(\ker E\) and
the exact left inverse \(L=(EE^T)^{-1}E\). The interval Hessian is
also compiled once. The compiler forms the group moments for \(PHP\)
and \(LHP\) with exact projected centers before interval conversion,
preserving structural cancellations. It mixes group matrices as their
first value plus probability-weighted exact differences; entries common
to all groups therefore do not acquire artificial weight uncertainty.
This representation stores dense matrices; it is not a fully matrix-free
interface.

For a finite binary64 proposal \(a\), set \(\widehat w=Pa\) using
exact arithmetic on its stored binary values. Both \(\widehat w\)
and the true equality-constrained quadratic minimizer \(w_{\mathcal B}\)
lie in \(\ker E\). An outward residual enclosure gives

\[
r=P(H\widehat w-\xi),\qquad
\|w_{\mathcal B}-\widehat w\|_2
 \le \varepsilon:=\frac{\|r\|_{2,\mathrm{upper}}}{m}.
\]

Indeed \(PHw_{\mathcal B}=P\xi\); coercivity on \(\ker E\)
and Cauchy--Schwarz bound the error by its residual. Coordinate radii
can use \(\sqrt{P_{ii}}\varepsilon\), since the error is tangent.
The exact center \(\widehat w\) is not itself asserted to be the
true derivative.

The multiplier vector is enclosed from
\(L(\xi-Hw_{\mathcal B})\). Selected weak multipliers must be
nonnegative; base-equality and strong-row derivative multipliers are
unrestricted. Each unselected weak row \(g_i^T\) has slack enclosure

\[
-g_i^T\widehat w
 +[-\|Pg_i\|_2\varepsilon,\ \|Pg_i\|_2\varepsilon].
\]

This uses the tangent error directly. If \(Pg_i=0\), that slack is
exactly zero, avoiding the loss of tangency from applying a row to a
coordinate box. Once all weak signs are certified, the equality-QP
solution satisfies the full critical-cone KKT conditions and is the
true directional derivative.

No active-set search occurs. A failed sign enclosure may mean the
proposed face is wrong or the bounds are inconclusive. At a direction
boundary, a true zero multiplier need not be provable from a nonsingleton
interval. Redundant unselected weak rows are allowed; another face or
multiplier representation can succeed when this one does not.

## Compiling and querying

The compiler `compile_gaussian_grouped_critical_face(equilibrium,
active_weak_indices=())` and query
`certify_gaussian_grouped_directional(face, direction, proposal)` are in
`omnibias.core.verified.gaussian_grouped_directional`. Weak indices
refer to original inequality rows.

`GaussianGroupedCriticalFace` retains its `equilibrium`, selected weak
indices, `constraints`, exact `projector` and `left_inverse`, interval
`posterior` and `group_probabilities`, and the interval matrices
`hessian`, `projected_hessian` and `multiplier_hessian`. Their meanings
are \(H\), \(PHP\) and \(LHP\). Group masses include group
cardinalities; different group sizes are supported. The compiler uses
one exact row-Gram factorization and reused solves for its geometry.
Multiple distinct logit groups require a rigorous transcendental backend.
A single group has exact uniform probabilities and needs no exponential
evaluation.

The query accepts exact integer or `Fraction` directions and finite
binary64 proposals, including exactly representable integer entries.
Its `GaussianGroupedDirectional` result retains the input `proposal`,
exact `projected_proposal`, exact `value`, `enclosure`, Euclidean
`error_bound`, `coordinate_error_bounds`, `residual_enclosure`,
`residual_norm_bound`, and interval multiplier/slack evidence.
Normally `value` is the projected proposal. If \(P\xi=0\), the
equality-QP response is exactly zero; the query uses that exact value
regardless of the proposal, while still checking the weak signs.

`equality_multipliers` follow the compiled \(E\) row order, including
selected weak rows. `weak_multipliers` and `weak_slacks` follow every
original inequality index. Their nonweak entries are structural zeros,
not claims about original primal multipliers or slacks. Both compiled
faces and query results retain their `certificate`.

No query solve, factorization, nonlinear iteration or exponential
evaluation occurs. Invalid input shapes/types and dependent selected
rows raise. Missing rigorous backends raise `RuntimeError` when needed;
prior fallback history remains recorded. A wrong face, inconclusive
signs, damaged dependencies, exhausted resources or nonfinite bounds
return `None`, without claiming that the derivative is undefined.

## A nonuniform two-dimensional switch

Take \(\nu=s=1\), centers
\((-1,0),(1,0),(0,-1),(0,1)\), and biases
\((1/4,1/4,5/4,5/4)\). At \(X=v=0\), the two score groups
have scores \(0\) and \(1\), both centroids zero, and
\(q_{\mathrm{base}}=1/2\). Impose \(x_1+2x_2\le0\).
Its multiplier is exactly zero, while

\[
H=\operatorname{diag}\left(
 \frac{1+2e}{2(1+e)},\frac{2+e}{2(1+e)}\right).
\]

In direction \((1,0)\), the wall is active in the critical QP:

\[
DX(0;(1,0))=
 \frac{1+e}{6+9e}(8,-4)
 \approx(0.97642236,-0.48821118).
\]

The opposite direction releases the wall:

\[
DX(0;(-1,0))=
 \left(-\frac{2(1+e)}{1+2e},0\right)
 \approx(-1.15536240,0).
\]

These responses are not negatives of each other, so an ordinary
Jacobian does not exist at this equilibrium.

The following uses ordinary floating-point formulas only to propose
responses. The certificate recomputes their projected residuals using
the exact declared model and an outward Hessian enclosure.

```python
from fractions import Fraction
import math

from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_grouped_equilibrium import (
    certify_gaussian_grouped_equilibrium,
)
from omnibias.core.verified.gaussian_grouped_directional import (
    compile_gaussian_grouped_critical_face,
    certify_gaussian_grouped_directional,
)

source = GaussianHeatMap(
    ((-1, 0), (1, 0), (0, -1), (0, 1)),
    (Fraction(1, 4), Fraction(1, 4), Fraction(5, 4), Fraction(5, 4)),
    viscosity=1, time=1,
)
base = restrict_gaussian_heat(source, (), ())
assert base is not None
polyhedral_face = compile_gaussian_polyhedral_face(base, ((1, 2),), (0,))
assert polyhedral_face is not None
equilibrium = certify_gaussian_grouped_equilibrium(
    polyhedral_face, (0, 0), (0, 0),
)
assert equilibrium is not None
assert equilibrium.groups == ((0, 1), (2, 3))
assert equilibrium.group_logits == (0, 1)
assert equilibrium.centroid == (0, 0)
assert equilibrium.weak_indices == (0,)

free = compile_gaussian_grouped_critical_face(equilibrium)
wall = compile_gaussian_grouped_critical_face(equilibrium, (0,))
assert free is not None and wall is not None
h1 = (1 + 2 * math.e) / (2 * (1 + math.e))
h2 = (2 + math.e) / (2 * (1 + math.e))
denominator = 4 * h1 + h2
positive = certify_gaussian_grouped_directional(
    wall, (1, 0), (4 / denominator, -2 / denominator),
)
negative = certify_gaussian_grouped_directional(
    free, (-1, 0), (-1 / h1, 0.0),
)
assert positive is not None and negative is not None
assert positive.error_bound < Fraction(1, 10**12)
assert negative.error_bound < Fraction(1, 10**12)
assert positive.enclosure[0].lo > 0
assert negative.enclosure[0].hi < 0
```

## Scope

The new capability combines exact grouped-mean evidence with a rigorous
residual bound for a classical critical-cone sensitivity problem. See
[Jittorntrum on directional differentiability without strict complementarity](https://link.springer.com/chapter/10.1007/BFb0121215)
and [Levy and Rockafellar on solution mappings](https://sites.math.washington.edu/~rtr/papers/rtr163-GeometrySolutionMappings.pdf).
No new general theorem or performance result is claimed.

The result concerns a first target-direction derivative at a fixed
source and fixed constraints. It does not certify finite-step Taylor
remainders, an ordinary VJP at a switch, changes to constraint geometry,
or higher derivatives. Certificate checks use exact arithmetic and
outward intervals under explicit budgets and backend requirements;
they are not theorem-prover verification.
