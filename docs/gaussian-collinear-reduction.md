# Global decisions for collinear Gaussian mixtures

A positive Gaussian mixture can have many spatial centers while its nonlinear
dependence remains confined to one direction. If all centers lie on an affine
line, polyhedral constrained minimization reduces exactly to finitely many
scalar problems. The remaining coordinates are eliminated by strictly convex
quadratic programs. The scalar objective can have several competing wells.

This extends the [two-center reduction](gaussian-polyhedral-reduction.md) to
any finite number of collinear modes. It provides a global minimum-value
enclosure, an exactly feasible witness and a cover of **all** global minimizers.
It does not require global convexity, identify a unique solution or assert a
differentiable choice between tied minima.

The [projected-simplex optimizer](gaussian-projection-global.md) supports
arbitrary finite center geometry through a higher-dimensional search.

## Reduction of the original Gaussian model

Let the stored centers be \(a_j=c+t_j a\), where \(c\) is the first
center and \(a\ne0\) is the first nonzero center difference. The compiler
checks this representation exactly. Set

\[
\alpha=\|a\|^2,\qquad
\lambda=\frac{2\nu s}{\alpha},\qquad
P=I-\frac{aa^T}{\alpha},\qquad
x=c+\lambda a y+r,\quad a\cdot r=0,
\]

\[
\beta_j=(b_j-b_0)-\frac{\alpha t_j^2}{4\nu s},\qquad
w=\frac{s\,a\cdot v}{\alpha}.
\]

Here \(s>0\) is the physical heat age, \(\nu>0\) is viscosity and
\(v\) is the target. For the Gaussian heat potential
\(\Phi_s=-2\nu\log\tau_s\), direct expansion gives

\[
\Phi_s(x)-v\cdot x
=\frac{\|r-sPv\|^2}{2s}
 +\nu\lambda y^2
 -2\nu\log\sum_j e^{t_jy+\beta_j}
 -2\nu wy+\text{constant}.
\]

The constant is common to all feasible points; it includes the heat
normalization and the removed common bias. Duplicate centers remain distinct
weighted modes. Removing a duplicate without combining its positive weight
would change the original model.

For constraints \(Cx\le d\), define the transverse value

\[
g(y)=\min_{r}\left\{
\frac{\|r-sPv\|^2}{2s}:
a\cdot r=0,\quad Cr\le d-Cc-\lambda Ca\,y
\right\}.
\]

The feasible scalar domain is a closed interval, ray, singleton, the whole
line or empty. At each feasible \(y\), the transverse minimizer is unique.
Enumerating independent active transverse normals gives exact affine pieces

\[
x_*(y)=x_0+x_1y,\qquad
g(y)=g_0+g_1y+g_2y^2,\qquad g_2\ge0.
\]

Nonnegative KKT multipliers and all original inequalities define each piece's
closed validity interval. The finite cover includes the empty active subset,
singleton intervals and redundant representations. No global independence of
the original constraint rows is assumed: an optimum's normal-cone vector has
an independent active representation of at most \(d-1\) transverse rows.
Overlapping pieces describe the same transverse minimum.

Thus the original global problem becomes minimization of

\[
J(y)=g(y)+\nu\lambda y^2
 -2\nu\log\sum_j e^{t_jy+\beta_j}-2\nu wy.
\]

Since \(g\ge0\), \(\lambda>0\) and log-sum-exp grows at most linearly
in \(|y|\), this objective is coercive. A nonempty closed original
polyhedron has an attained global minimum, including when it has empty
interior or is unbounded.

## Finite search cover from the posterior mean

On a quadratic piece, set

\[
k=\lambda+g_2/\nu>0,\qquad h=w-g_1/(2\nu),\qquad
p_j(y)=\frac{e^{t_jy+\beta_j}}{\sum_\ell e^{t_\ell y+\beta_\ell}}.
\]

Then

\[
J'(y)=2\nu\bigl(ky-\mu(y)-h\bigr),\qquad
\mu(y)=\sum_jp_j(y)t_j,\qquad
\mu'(y)=\operatorname{Var}_{p(y)}(t)\ge0.
\]

At least two distinct center coordinates have positive weights. Consequently
\(t_{\min}<\mu(y)<t_{\max}\) at every finite real \(y\).
The derivative is strictly negative below \((h+t_{\min})/k\) and strictly
positive above \((h+t_{\max})/k\). Clamping these two exact endpoints to
the full piece interval retains every piece minimum, including an endpoint
or breakpoint minimum. The resulting search intervals are finite even when
the feasible polyhedron is unbounded.

There is no two-center bound on the number of stationary points here: three
modes already permit three local minima. The two-center branch-selection
hyperplane does not by itself select the global branch of a general mixture.

The field `axial_scale` stores \(\lambda\). It is **not** the inverse
of `model.contraction_bound`. With this choice of coordinates,

\[
q=\frac{(t_{\max}-t_{\min})^2}{4\lambda}.
\]

The reduction and optimizer support every \(q>0\), subject to their
geometry, arithmetic and resource contracts.

## API and certified outputs

`compile_gaussian_collinear_reduction(model, inequalities, upper_bounds,
target)` in `omnibias.core.verified.gaussian_collinear_reduction` accepts
exact integers or `Fraction` values. It preserves the original model and
returns a `GaussianCollinearReduction` with `origin`, `direction`,
`projector`, `axial_scale`, `field`, `center_coordinates`, `logit_offsets`,
`mean_bounds`, `pieces` and `enumerated_bases`.

Each piece stores `active_indices`, `lower` and `upper` validity endpoints
(`None` denotes an infinite endpoint), `point_intercept`, `point_slope`,
`transverse_cost` in ordinary polynomial order, `scalar_curvature`,
`scalar_field` and a finite `search_interval`. Despite its name,
`scalar_curvature` stores \(k\), not the full curvature
\(J''(y)=2\nu[k-\operatorname{Var}_{p(y)}(t)]\).
`lift(piece_index, y)` and `transverse_value(piece_index, y)` return exact
rational results and require \(y\) to lie in that piece.

An empty completed `pieces` tuple proves infeasibility. `None` means refusal,
which can include noncollinear or all-coincident centers or a resource limit;
it is not an infeasibility proof. Active-basis enumeration can be combinatorial.

`certify_gaussian_collinear_minimum(reduction, *, value_tolerance=1e-6,
max_boxes_per_piece=255)` in
`omnibias.core.verified.gaussian_collinear_minimum` performs interval
[branch-and-bound](global-optimization.md) on every finite search interval.
Exact affine logits precede outward conversion. Rigorous shifted log-sum-exp
encloses the value, and endpoint posterior means enclose the derivative
because \(\mu\) is monotone. Only a strict lower-bound comparison prunes
a candidate, preserving ties across and within pieces.

| Result | Meaning |
| --- | --- |
| `minimum` | Enclosure of the global value of the displayed reduced objective \(J\). |
| `sample_point`, `sample_value` | An exactly feasible rational witness and its reduced value enclosure. |
| `candidate_boxes` | A spatial cover containing every global minimizer; additional nonoptimal or infeasible points may remain. |
| `candidate_piece_indices`, `candidate_parameters` | Each retained piece and interval in its normalized parameter in \([0,1]\). |
| `piece_results` | Per-piece scalar minimizer diagnostics. |
| `gap_upper`, `status` | Outward global value-gap bound and termination status. |

`status="value_tolerance"` confirms the requested gap.
`"box_budget"` and `"resolution_limit"` retain valid bounds and a complete
cover with weaker accuracy. Aggregate resources are checked as well as
`max_boxes_per_piece`; no failed piece is omitted from a global certificate.
An infeasible reduction, damaged dependency or numerical/resource refusal
returns `None`. Invalid arguments or a missing required rigorous backend
raise. Prior conditional transcendental fallback is refused, not reset.
The certificates do not claim external formal theorem-prover verification.

## Three competing wells

This mixture has local minima near \(-2,0,2\). The middle minimum has a
higher value, while symmetry gives two equal global outer minima.

```python
from fractions import Fraction as F

from omnibias.core.verified.gaussian_collinear_minimum import (
    certify_gaussian_collinear_minimum,
)
from omnibias.core.verified.gaussian_collinear_reduction import (
    compile_gaussian_collinear_reduction,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap

model = GaussianHeatMap(
    ((-2,), (0,), (2,)), (0, -1, 0), viscosity=F(1, 8), time=1,
)
reduction = compile_gaussian_collinear_reduction(model, (), (), (0,))
assert reduction is not None
assert reduction.center_coordinates == (0, 1, 2)
assert reduction.axial_scale == F(1, 16)
assert reduction.pieces[0].search_interval == (0, 32)

minimum = certify_gaussian_collinear_minimum(reduction)
assert minimum is not None and minimum.status == "value_tolerance"
assert minimum.gap_upper <= 1e-6
assert any(box[0].hi < 0 for box in minimum.candidate_boxes)
assert any(box[0].lo > 0 for box in minimum.candidate_boxes)
assert all(not (box[0].lo <= 0 <= box[0].hi)
           for box in minimum.candidate_boxes)
```

The integration tests independently bracket the original Gaussian velocity's
outer zeros, check positive curvature at all three local minima, compare
their original potential values, and verify coverage of both outer roots.
A second case retains a duplicated outer mode; its additional weight breaks
the tie and changes the winning well.

Constraints can instead select the middle well. The same compiler accepts
unbounded polyhedra, corners and equalities encoded by opposite inequalities.

```python
middle_reduction = compile_gaussian_collinear_reduction(
    model, ((1,), (-1,)), (1, 1), (0,),
)
assert middle_reduction is not None
middle = certify_gaussian_collinear_minimum(middle_reduction)
assert middle is not None
assert all(F(-1) <= x <= F(1) for x in middle.sample_point)
assert any(box[0].lo <= 0 <= box[0].hi for box in middle.candidate_boxes)
```

## PDE meaning and scope

For fixed original centers, biases and viscosity, the positive field

\[
\tau(x,s)=(4\pi\nu s)^{-d/2}
\sum_j e^{b_j-\|x-a_j\|^2/(4\nu s)}
\]

solves the heat equation. Its logarithmic gradient

\[
U(x,s)=-2\nu\nabla\log\tau
=\frac{x-\sum_jp_j(x,s)a_j}{s}
\]

solves the irrotational vector Burgers equation
\(U_s+(U\cdot\nabla)U=\nu\Delta U\).
The global decision computed here minimizes \(\Phi_s-v\cdot x\) on
the polyhedron at one fixed age. Unconstrained stationary points obey
\(U=v\); constraints give the corresponding KKT equations. A global
selection can jump at a tie, so the optimizer alone does not provide a smooth
inverse PDE solution or a derivative certificate.

The compiled \(\lambda\) and \(\beta_j\) depend on age. Freezing them
while changing \(s\) changes the family; it does not differentiate the
original heat evolution. General noncollinear mixtures, arbitrary prescribed
PDE data, certified selected-branch derivatives and universal speed guarantees
are outside this API.

The heat-to-Burgers transform is classical
([Hopf, 1950](https://doi.org/10.1002/cpa.3160030302),
[Cole, 1951](https://doi.org/10.1090/qam/42889)). Log-partition derivatives as
means and covariances are standard exponential-family identities; see
[Wainwright and Jordan, Section 3.5](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf).
Affine optimizers and piecewise quadratic value functions are classical
parametric-QP structure; see
[Bemporad et al., Theorem 4](https://cse.lab.imtlucca.it/~bemporad/publications/papers/acc00-mpqp.pdf).
The contribution here is a checked implementation of this reduction and its
global enclosure contract for the declared Gaussian family, not a claim of
a new general theorem or a general-purpose fast PDE solver.
