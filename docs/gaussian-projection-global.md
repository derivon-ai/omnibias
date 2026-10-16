# Global Gaussian decisions through a projected simplex

A finite positive Gaussian heat mixture can be optimized over a closed
polyhedron even when its centers are not collinear and its potential is
nonconvex. Every global minimizer lies in the projection of one compact
polytope: the convex hull of the centers translated by the target. This gives
a bounded parameter domain for rigorous global optimization without requiring
a contraction condition or a smooth projection map.

The result encloses the global minimum value, supplies an exactly feasible
witness, and retains a cover of **all** global minimizers. It can retain broad
or redundant candidates. It does not select a unique branch or certify its
derivative.

## Why the projected hull contains every minimizer

For positive viscosity \(\nu\), age \(s\), fixed centers \(a_j\) and
real biases \(b_j\), let

\[
\tau(x,s)=(4\pi\nu s)^{-d/2}
\sum_j\exp\left(b_j-\frac{\|x-a_j\|^2}{4\nu s}\right),
\qquad \Phi_s=-2\nu\log\tau.
\]

Write \(p_j(x)\) for the positive normalized Gaussian weights and
\(\mu(x)=\sum_jp_j(x)a_j\). The objective and gradient are

\[
f_v(x)=\Phi_s(x)-v\cdot x,\qquad
\nabla f_v(x)=\frac{x-\mu(x)}s-v.
\]

For a nonempty closed polyhedron \(C=\{x:Gx\le h\}\), the objective
is coercive, so a global minimizer exists. At every such minimizer \(x_*\),
first-order optimality over the convex feasible set gives

\[
sv+\mu(x_*)-x_*\in N_C(x_*),\qquad
x_*=\operatorname{Proj}_C\bigl(sv+\mu(x_*)\bigr).
\]

The projection characterization is valid without independence of the
constraint rows, an interior feasible point, or convexity of \(f_v\).
Because \(\mu(x_*)\) belongs to the convex hull of the centers,

\[
S=\operatorname{Proj}_C\bigl(sv+\operatorname{conv}\{a_j\}\bigr)
\subset C
\]

contains every global minimizer. Projection is continuous, so \(S\) is
compact. Therefore minimizing \(f_v\) over \(S\) preserves both its
global value and all original global minimizers. Points in \(S\) need not
be stationary, and the fixed-point equation alone would not distinguish a
global minimum from another stationary point.

Crucially, projecting the convex hull is different from taking the convex hull
of the projected centers. For \(C=\mathbb R^2_+\), centers
\((-1,1),(1,-1)\) and zero target, the midpoint projects to \((0,0)\).
The projected vertices are \((0,1),(1,0)\), whose convex hull misses that
point. At \(\nu=s=1\), the original potential actually has its unique
global minimum at \((0,0)\): its axial component is the critical
two-center potential and its transverse component is a positive quadratic.
Replacing \(S\) by the convex hull of projected vertices would lose the
optimizer in this example.

## A bounded, onto parametrization

For \(m\) modes, let \(z\in[0,1]^{m-1}\) and form weights

\[
\theta_0=z_0,\qquad
\theta_j=z_j\prod_{i<j}(1-z_i)\quad(1\le j<m-1),\qquad
\theta_{m-1}=\prod_{i<m-1}(1-z_i).
\]

These nonnegative weights sum to one and cover the entire closed simplex.
To obtain any desired weights, divide each weight by the remaining mass;
when that mass becomes zero, all remaining weights are zero and the remaining
parameters may be arbitrary. Hence

\[
q(z)=sv+\sum_j\theta_j(z)a_j,\qquad
x(z)=\operatorname{Proj}_C(q(z))
\]

covers \(S\). One mode uses a degenerate dummy parameter interval.
Repeated or coincident centers are allowed; their weights remain part of
the original objective.

The exact polyhedral projector enumerates independent active normal bases.
For any rational query, exact KKT checks select a feasible projection with
nonnegative multipliers. Rank deficiency, redundant rows and equalities
encoded by opposite inequalities do not require a floating rank decision.
Compilation and queries remain subject to exact arithmetic and work budgets.

## Enclosing a nonsmooth projected image

On a parameter box, evaluate an exact rational midpoint \(z_0\), its
mixture point \(q_0\), and its exact projection \(x_0\). If rational
coordinate bounds imply \(|q_i(z)-q_{0,i}|\le e_i\), nonexpansiveness
gives

\[
\|x(z)-x_0\|_2\le\|q(z)-q_0\|_2\le\sum_i e_i.
\]

Thus an outward box around \(x_0\) encloses the whole projected image.
The bound shrinks with the parameter box, including across an active-set
switch. A value-only interval optimizer encloses the Gaussian objective on
this image box; it does not assume an ordinary derivative of the projection.
Exact rational parameter samples yield exact feasible spatial witnesses.
Spatial boxes can include infeasible points, which is harmless for an
enclosure; only the witness itself is asserted feasible.

## Objective normalization and contracts

The reported objective removes common constants. With
\(c=a_0\), \(r=x-c\), \(\delta_j=a_j-c\) and
\(\beta_j=(b_j-b_0)-\|\delta_j\|^2/(4\nu s)\), it is

\[
F_v(x)=\frac{\|r\|^2}{2s}-v\cdot r
-2\nu\log\sum_j
\exp\left(\frac{\delta_j\cdot r}{2\nu s}+\beta_j\right).
\]

This differs from \(\Phi_s(x)-v\cdot x\) by one common constant,
so the minimizers are unchanged. Center and bias differences are taken
exactly before outward numerical conversion. Objective values from another
normalization must be shifted before comparison.

`compile_polyhedral_projector(dimension, inequalities, upper_bounds, *,
budget=None)` lives in `omnibias.core.verified.polyhedral_projection`.
The optimizer is
`certify_gaussian_projection_minimum(model, projector, target, *,
value_tolerance=1e-6, max_boxes=255)` in
`omnibias.core.verified.gaussian_projection_minimum`.
Model metadata, constraints and target use exact integers or `Fraction`
values. Both model and projector budgets constrain the composed calculation.
`projector.project(point)` returns an exact projection record with `point`,
original-row `multipliers`, `active_indices` and `squared_distance`.
Compilation itself does not assert that the polyhedron is feasible. A query
returning `None` supplies no projection or infeasibility certificate.

| Result field | Meaning |
| --- | --- |
| `minimum` | Enclosure of the global minimum of \(F_v\). |
| `sample_point`, `sample_value` | An exact feasible rational witness and its objective enclosure. |
| `candidate_boxes` | A spatial cover of every global minimizer. |
| `candidate_parameters` | Retained parameter boxes before projection. |
| `parameter_result` | Underlying bounded minimizer diagnostics. |
| `gap_upper`, `status` | Outward value-gap bound and termination status. |

`"value_tolerance"` confirms the requested gap. `"box_budget"` and
`"resolution_limit"` retain sound bounds and the full minimizer cover without
asserting that accuracy. Strict pruning preserves ties. Refusal is not a proof
of infeasibility or absence of a global minimum. The rigorous transcendental
backend is required, and prior conditional fallback history is not erased.

## Executable examples

This noncollinear four-mode density factors into two one-dimensional Gaussian
mixtures. Each has two global wells, so all four sign combinations are global
minimizers. A modest box budget deliberately makes no accuracy or uniqueness
assertion.

```python
from fractions import Fraction as F
from itertools import product

from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_projection_minimum import (
    certify_gaussian_projection_minimum,
)
from omnibias.core.verified.polyhedral_projection import (
    compile_polyhedral_projector,
)

model = GaussianHeatMap(
    tuple(product((-1, 1), repeat=2)), (0, 0, 0, 0), viscosity=F(1, 4),
)
projector = compile_polyhedral_projector(2, (), ())
assert projector is not None
minimum = certify_gaussian_projection_minimum(
    model, projector, (0, 0), max_boxes=31,
)
assert minimum is not None and minimum.candidate_boxes
assert minimum.parameter_result.gradient_evaluations == 0
assert all(isinstance(x, F) for x in minimum.sample_point)
if minimum.status == "value_tolerance":
    assert minimum.gap_upper <= 1e-6
```

The following exact projections exhibit the difference between the projected
hull and the hull of projected vertices, including the genuine optimizer at
the active-set switch.

```python
orthant = compile_polyhedral_projector(2, ((-1, 0), (0, -1)), (0, 0))
assert orthant is not None
left, right = orthant.project((-1, 1)), orthant.project((1, -1))
midpoint = orthant.project((0, 0))
assert left is not None and right is not None and midpoint is not None
assert left.point == (0, 1) and right.point == (1, 0)
assert midpoint.point == (0, 0)

critical = GaussianHeatMap(((-1, 1), (1, -1)), (0, 0))
constrained = certify_gaussian_projection_minimum(
    critical, orthant, (0, 0), max_boxes=31,
)
assert constrained is not None and constrained.sample_point == (0, 0)
assert any(all(axis.lo <= 0 <= axis.hi for axis in box)
           for box in constrained.candidate_boxes)
```

Independent integration tests evaluate the original Gaussian distances and
compare all four known global wells with the returned value and spatial cover.
They also cover equality constraints, halfspaces, exact one-mode projections,
coincident modes and common-bias cancellation.

## Scope and classical context

Unlike the [collinear scalar reduction](gaussian-collinear-reduction.md),
this construction supports arbitrary finite center geometry, including one
mode, repeated centers and nonconvex regimes. Its parameter dimension is
\(m-1\), and active-basis enumeration can be combinatorial. Projection
enclosures and high-dimensional interval search can remain coarse; there is
no dimension-independent speed or requested-accuracy guarantee.

The [entropy reduction](gaussian-entropy-global.md) gives an alternative
mixture-weight objective with the same global minimum. On a certified interior
posterior domain, its gradient remains valid across projection active-set
switches and enables interval gradient pruning.

The original positive heat mixture and its logarithmic gradient still satisfy
the heat and irrotational Burgers equations. The constrained optimization here
is at one fixed age. It does not solve arbitrary PDE data, make every global
decision differentiable, or justify replacing one activation by another.

Gaussian posterior-mean fixed points are classical: see
[Carreira-Perpiñán, Equation 7](https://faculty.ucmerced.edu/mcarreira-perpinan/papers/pami07.pdf).
Projection optimality is standard convex analysis; see
[Boyd and Vandenberghe, Sections 4.2.3 and 4.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
The global coverage argument combines those facts with a bounded interval
search. No new general theorem or external formal theorem-prover verification
is claimed.
