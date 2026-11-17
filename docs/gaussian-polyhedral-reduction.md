# Polyhedral Gaussian optimization through a scalar reduction

A two-center Gaussian potential has one nonlinear spatial direction.
At each value of that direction, minimizing over the remaining coordinates
is a strictly convex quadratic program. For rational polyhedral constraints,
the transverse minimizer has finitely many exact affine pieces. This reduces
the original constrained global problem to finitely many scalar problems,
including nonconvex cases.

The reduction is exact. The subsequent interval optimizer returns a
minimum-value enclosure, a feasible witness and boxes covering **all** global
minimizers. Unresolved candidates can remain in that cover; it does not claim
to identify every tie or give a unique optimizer.

The [collinear mixture reduction](gaussian-collinear-reduction.md) extends
this construction to any finite number of centers on one affine line, with
at least two distinct positions. It replaces the two-center hyperbolic
tangent by a posterior mean with a finite support bound.

## Exact reduction of the original model

Write the two distinct original centers as \(c-a,c+a\), and let

\[
\alpha=\|a\|^2>0,\qquad
\lambda=\frac{2\nu s}{\alpha}>0,\qquad
\delta=b_+-b_-,\qquad P=I-aa^T/\alpha.
\]

This construction covers all positive viscosities and ages; it does not
require \(q=1/\lambda<1\). For a fixed target \(v\), define

\[
w=\frac{s\,a\cdot v}{\alpha}+\frac{\lambda\delta}{2},\qquad
x=c+\lambda a(y-\delta/2)+r,\qquad a\cdot r=0.
\]

The target-tilted original potential satisfies

\[
\Phi_s(x)-v\cdot x
=\frac{\|r-sPv\|^2}{2s}
 +2\nu\left(\frac{\lambda y^2}{2}-\log\cosh y-wy\right)
 +\text{constant},
\]

where the constant is independent of \(x\). The normalization of the
original heat field is included in this constant, not changed by the reduction.

For the closed polyhedron \(Cx\le d\), set

\[
g(y)=\min_{r}\left\{\frac{\|r-sPv\|^2}{2s}:
a\cdot r=0,\quad
Cr\le d-Cc-\lambda Ca(y-\delta/2)\right\}.
\]

The feasible scalar set \(I\) is the linear image of a polyhedron,
so it is a closed interval, ray, the whole line, a singleton or empty.
For every feasible \(y\), the transverse objective is strictly convex
and coercive, so it has a unique minimizer \(r_*(y)\). Consequently

\[
\min_{Cx\le d}\bigl(\Phi_s(x)-v\cdot x\bigr)
=\min_{y\in I}\left[
g(y)+2\nu\left(\frac{\lambda y^2}{2}-\log\cosh y-wy\right)
\right]+\text{constant}.
\]

The scalar objective is coercive because \(g\ge0\),
\(\lambda>0\), and \(\log\cosh y\le|y|\). A nonempty original
polyhedron therefore has at least one global minimizer, including when it is
unbounded or has empty interior.

## Finite exact quadratic pieces

Choose rational coordinates for \(a^\perp\). The transverse Hessian
in these coordinates is positive definite. For each linearly independent
subset of active constraint normals, its KKT system has an exact rational
solution affine in \(y\). Nonnegative multipliers and all original
inequalities define a closed validity interval. On that interval,

\[
x_*(y)=x_0+x_1y,\qquad
g(y)=g_0+g_1y+g_2y^2,\qquad g_2\ge0.
\]

These intervals form a finite cover of \(I\). They may overlap:
different active bases can represent the same unique transverse minimizer.
Redundant or degenerate constraints do not require a global independence
assumption. The normal-cone representation of the optimum has an independent
subset of active generators; this subset appears in the enumeration. Rows
with zero transverse normal still impose scalar feasibility conditions.
Singleton intervals and the empty active subset must be retained.

The value function \(g\) is convex, continuous on its domain, and
piecewise quadratic. It need not be differentiable at a breakpoint. For
example, with centers \(\pm(1,0)\), \(\nu=1/4\), \(s=1\),
zero biases and target, impose

\[
x_2\ge1+x_1,\qquad x_2\ge1-x_1.
\]

Then \(y=2x_1\), \(r_*(y)=(0,1+|y|/2)\), and
\(g(y)=\tfrac12(1+|y|/2)^2\). The unique constrained global
minimum is the breakpoint \(x=(0,1)\). An algorithm considering only
stationary roots inside smooth pieces would miss it.

## A finite search interval for each piece

The reduced objective derivative on a piece is

\[
J_i'(y)=2\nu\bigl(k_i y-\tanh y-h_i\bigr),\qquad
k_i=\lambda+\frac{g_2}{\nu}>0,\qquad
h_i=w-\frac{g_1}{2\nu}.
\]

Because \(-1<\tanh y<1\), this derivative is strictly negative
below \((h_i-1)/k_i\) and strictly positive above
\((h_i+1)/k_i\). Clamp both rational endpoints to the piece's validity
interval. The resulting finite closed `search_interval` contains every
minimum of that piece, including an endpoint minimum. If the validity interval
lies entirely to one side, the search interval is a singleton.

This replaces an unbounded search with finitely many bounded searches without
discarding any global minimizer. It does not rely on each piece being convex.
When \(k_i<1\), up to three stationary roots can occur. A constraint can
exclude the unconstrained winning branch and make a metastable branch the
constrained global minimum. The exact unconstrained selection rule from
[the two-center certificate](gaussian-two-center-minimum.md) is therefore
insufficient for a general polyhedron.

## Compiler and optimizer contracts

`compile_gaussian_polyhedral_reduction(model, inequalities, upper_bounds,
target)` in `omnibias.core.verified.gaussian_polyhedral_reduction` constructs
the exact finite KKT cover. Inputs use exact integers or `Fraction` values.
It preserves the original two-center model, inequalities and target.

Each piece records `active_indices`, `lower` and `upper` validity endpoints
(`None` denotes the corresponding infinity), `point_intercept`, `point_slope`,
`transverse_cost` in ordinary polynomial order \((g_0,g_1,g_2)\),
`scalar_curvature`, `scalar_field`, and its finite `search_interval`.
The reduction provides exact `lift(piece_index, y)` and
`transverse_value(piece_index, y)` queries within the selected piece. It also
records `enumerated_bases`; its `feasible` property reports whether the complete
cover is nonempty.

An empty `pieces` tuple certifies that the original polyhedron is infeasible.
Compiler refusal or resource exhaustion returns `None`, which must never be
interpreted as an infeasibility proof. Enumeration is finite but can require
combinatorially many active bases. Exact coefficient size and work limits are
part of the computational contract.

`certify_gaussian_polyhedral_reduced_minimum(reduction, *,
value_tolerance=1e-6, max_boxes_per_piece=255)` in
`omnibias.core.verified.gaussian_polyhedral_reduced_minimum` then returns a
`GaussianPolyhedralReducedMinimum` or `None`. It requires the rigorous
transcendental backend and refuses prior conditional fallback history. An
infeasible reduction, damaged certificate, exhausted aggregate resources or
unusable numerical enclosure returns `None`. Malformed arguments and a missing
required rigorous backend raise; neither outcome proves infeasibility.

The scalar optimizer maps each search interval affinely from \([0,1]\),
using exact rational endpoints, and applies rigorous interval
[branch-and-bound](global-optimization.md). Piecewise derivative callbacks
are smooth within their individual search interval, even when the assembled
objective has a corner at a shared endpoint. Bounds are combined across the
entire piece cover. Strict lower-bound pruning preserves all tied minima;
overlapping pieces can produce redundant candidate boxes.

| Result field | Meaning |
| --- | --- |
| `minimum` | Enclosure of the global reduced minimum value. |
| `sample_point`, `sample_value` | An exactly feasible rational witness and an enclosure of its reduced objective value. |
| `candidate_boxes` | Spatial boxes whose union contains every global minimizer. They may also contain infeasible or nonoptimal points. |
| `candidate_piece_indices`, `candidate_parameters` | Each candidate's piece and its retained interval in the normalized parameter \(t\in[0,1]\). |
| `piece_results` | The complete per-piece `BoxMinimum` diagnostics. |
| `gap_upper`, `status` | Outward value-gap bound and completion status. |

`status="value_tolerance"` means the global gap meets the requested tolerance.
`"box_budget"` or `"resolution_limit"` retains valid bounds and a complete
minimizer cover without asserting that accuracy. `max_boxes_per_piece` limits
each piece's evaluated boxes; aggregate resource limits are also checked before
the full computation. A failed piece cannot be silently omitted from a global
certificate.

The reported objective is the displayed reduced objective. It differs from
the original target-tilted Gaussian potential by one common additive constant.
The optimizer does not silently treat that reduced value as the normalized
original objective value. A small value gap does not imply a small location
error, uniqueness, or a differentiable choice across switches.

The [uniqueness and sensitivity certificates](gaussian-polyhedral-sensitivity.md)
can establish a unique global solution from this complete cover and then
enclose its target Jacobian when strict active-constraint and curvature
conditions hold.

## A minimum at a constraint breakpoint

The following example realizes the corner minimum derived above. Its scalar
search intervals both collapse to zero, so the spatial witness is exact.

```python
from fractions import Fraction
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    compile_gaussian_polyhedral_reduction,
)
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    certify_gaussian_polyhedral_reduced_minimum,
)

model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=Fraction(1, 4))
reduction = compile_gaussian_polyhedral_reduction(
    model, ((1, -1), (-1, -1)), (-1, -1), (0, 0),
)
assert reduction is not None and reduction.feasible
assert all(piece.search_interval == (0, 0) for piece in reduction.pieces)
minimum = certify_gaussian_polyhedral_reduced_minimum(reduction)
assert minimum is not None
assert minimum.sample_point == (0, 1)
assert minimum.minimum.lo <= 0.5 <= minimum.minimum.hi
assert any(box[0].lo <= 0 <= box[0].hi and box[1].lo <= 1 <= box[1].hi
           for box in minimum.candidate_boxes)
```

## When the unconstrained losing branch wins

For the scalar two-center family below, a positive target favors the positive
unconstrained global minimizer. The constraint \(x\le0\) excludes it.
The negative metastable stationary point has lower objective than the feasible
boundary and becomes the constrained global minimum. Optimizing all compiled
pieces preserves this possibility.

```python
scalar = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=Fraction(1, 4))
halfline = compile_gaussian_polyhedral_reduction(
    scalar, ((1,),), (0,), (Fraction(1, 100),),
)
assert halfline is not None and halfline.feasible
constrained = certify_gaussian_polyhedral_reduced_minimum(halfline)
assert constrained is not None and constrained.sample_point[0] < Fraction(-9, 10)
assert constrained.sample_point[0] <= 0

# A completed empty cover is a proof, unlike a resource refusal.
empty = compile_gaussian_polyhedral_reduction(scalar, ((0,),), (-1,), (0,))
assert empty is not None and not empty.feasible and empty.pieces == ()
assert certify_gaussian_polyhedral_reduced_minimum(empty) is None
```

## Prior work and scope

The finite affine optimizer and convex piecewise-quadratic value structure are
classical parametric quadratic programming. See Theorem 4 of
[Bemporad, Morari, Dua and Pistikopoulos, *The Explicit Solution of Model
Predictive Control via Multiparametric Quadratic Programming*](https://cse.lab.imtlucca.it/~bemporad/publications/papers/acc00-mpqp.pdf).
The Gaussian reduction binds that structure to this particular nonlinear
potential; it is not a new general parametric-QP theorem.

The construction covers exactly two distinct Gaussian centers and rational
polyhedral constraints at a fixed model and target. It does not provide a
polynomial-time guarantee, arbitrary-mixture reduction, exact classification
of unresolved objective ties, or certified parameter derivatives. Certificate
integrity checks and outward numerical bounds are not an external formal
theorem-prover verification.
