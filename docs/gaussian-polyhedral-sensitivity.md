# A unique global decision and its local target derivative

The [polyhedral scalar reduction](gaussian-polyhedral-reduction.md) encloses
every global minimizer of the original constrained two-center Gaussian
potential. That complete cover can support two further certificates: uniqueness
of the global minimizer, and a local target Jacobian on a stable active face.
These are separate claims. Uniqueness alone does not imply differentiability.

## Proving uniqueness across the complete scalar hull

Map every surviving candidate parameter interval back through its exact affine
map to the scalar logit coordinate \(y\). Let \(I\) be their common hull.
Every global minimizer lies in \(I\), and every feasible \(y\) has exactly
one transverse minimizer. Thus uniqueness of the reduced minimizer implies
uniqueness of the original spatial minimizer.

If \(I\) is a singleton, this already proves uniqueness. Otherwise the
reduced objective on each compiled quadratic piece satisfies

\[
J_i''(y)=2\nu\bigl(k_i-\operatorname{sech}^2y\bigr).
\]

The test includes **every compiled piece meeting the entire hull in an interval
of positive length**, including pieces removed from the candidate cover and
gaps between surviving candidates. Testing only the surviving boxes would be
unsound: two separated global minima can each have positive local curvature
with a discarded nonconvex region between them.

For a piece intersection, let \(z\ge0\) be its distance from zero.
The elementary bound

\[
\cosh z\ge1+z^2/2
\quad\Longrightarrow\quad
\sup\operatorname{sech}^2y\le\frac{1}{(1+z^2/2)^2}
\]

gives an exact rational curvature bound. A positive lower bound for
\(k_i-\operatorname{sech}^2y\) proves strong convexity on that segment.
If \(k_i\ge1\), the piece is strictly convex even if the lower bound
is zero: the second derivative can vanish only at the isolated point
\(k_i=1,y=0\). The implementation uses the simpler bound \(k_i-1\)
in this case, and the distance-dependent rational bound when \(k_i<1\).

The transverse value function \(g\) is convex, so its derivative jumps
upward at piece boundaries. Adding the smooth scalar Gaussian term preserves
these jumps. Therefore the finite strictly convex pieces join to give a
strictly convex objective on \(I\). When every bound is positive, their
minimum also gives a uniform strong-convexity bound. In either case, there is
at most one minimizer in the hull; the original complete cover and existence
of a global minimum supply the other half of the uniqueness proof.

An adversarial example makes the full-hull requirement concrete. For centers
\(\pm(1,0)\), zero biases and target, \(\nu=1/4,s=1\), impose
\(x_2\ge2|x_1|-1/2\). Then

\[
g(y)=\tfrac12\max(0,|y|-1/2)^2.
\]

The outer pieces have \(k=5/2\), and the middle piece has \(k=1/2\).
The optimizer can discard the entire middle piece while retaining two tied
outer minima. Checking only retained pieces would falsely suggest uniqueness;
checking the whole hull finds the intervening nonconvex piece and refuses.

## A stable face gives an explicit Jacobian

Fix the independent selected constraint rows \(A\), and retain the
two-center notation \(P=I-aa^T/\|a\|^2\) and
\(\lambda=2\nu s/\|a\|^2\). Define exact matrices and vectors

\[
B=AP,\qquad M=(BB^T)^{-1},\qquad
R=P-B^TMB,\qquad
\ell=\lambda a,\qquad t=\ell-B^TMA\ell.
\]

The projector \(R\) is tangent to both the selected face and the logit
hyperplane. The vector \(t\) is the compiler's `point_slope`.
For nearby targets \(v\) while the face stays active, the exact transverse
solution has the form

\[
x(y,v)=x_{\rm const}+sRv+ty,\qquad
k=\frac{\|t\|^2}{2\nu s},\qquad
h(v)=h_{\rm const}+\frac{t\cdot v}{2\nu}.
\]

At an interior stationary point,
\(ky-\tanh y-h(v)=0\). If \(k-\operatorname{sech}^2y>0\),
implicit differentiation gives

\[
\boxed{\qquad
D_vX=sR+\frac{tt^T}{2\nu\,[k-\operatorname{sech}^2y_*]}.
\qquad}
\]

This is the derivative of the original constrained optimum with respect to
the original target, not a derivative of a frozen piece polynomial. The latter
would omit the target dependence of its intercept and scalar field.

Sufficient evidence for a stable winning face is a unique global optimizer,
a root hull strictly inside one piece's validity interval, strictly positive
selected original multipliers, strictly positive inactive slacks, and a positive
denominator throughout the hull. The selected original multipliers equal the
transverse projection multipliers divided by \(s\).

These conditions make the Hessian positive on the selected tangent space and
keep the active face fixed locally. The implicit-function theorem supplies a
smooth local branch. Coercivity, uniqueness and the strict local conditions
ensure that nearby global minimizers stay on this branch. This proves existence
of a target neighborhood; the certificate does not provide its radius.

The ambient Hessian may still be indefinite. For example, take centers
\(\pm(1,0,0)\), zero biases, \(\nu=1/4,s=1\), target
\((-7,0,0)\), and inequalities

\[
-x_1+x_2+x_3\le1,\qquad -x_1-x_2-2x_3\le-3.
\]

At the winning point \((0,-1,2)\), the original multipliers are
\((4,3)\), the ambient Hessian is \(\operatorname{diag}(-1,1,1)\),
but the feasible tangent direction \((1,3,-2)\) has positive quadratic
curvature \(12\). Here \(R=0\), \(t=(1/2,3/2,-1)\), \(k=7\),
and the target Jacobian is

\[
D_vX=\frac1{12}
\begin{pmatrix}1\\3\\-2\end{pmatrix}
\begin{pmatrix}1&3&-2\end{pmatrix}.
\]

## API and executable example

`certify_gaussian_polyhedral_unique(minimum)` in
`omnibias.core.verified.gaussian_polyhedral_unique` returns a
`GaussianPolyhedralUnique` or `None`. It requires a complete certified minimum
cover, then uses exact rational arithmetic. Its fields include `minimum`,
`parameter_hull`, `method`, `curvature_lower_bound`, `checked_piece_indices`,
`piece_parameter_intervals`, `piece_curvature_lower_bounds`, and `certificate`.

The method is `"singleton"`, `"strict_convexity"`, or `"strong_convexity"`.
`curvature_lower_bound` omits the factor \(2\nu\): multiply it by
\(2\nu\) for a lower bound on the scalar objective curvature within
smooth pieces. It is `None` for singleton proofs, which assert no curvature
bound; it can be zero for strict-convexity proofs.

`certify_gaussian_polyhedral_reduced_sensitivity(unique, piece_index)` in
`omnibias.core.verified.gaussian_polyhedral_reduced_sensitivity` checks the
selected piece and returns a `GaussianPolyhedralReducedSensitivity` or `None`.
It requires rigorous transcendental evaluation and preserves prior fallback
provenance. It performs no new root search. Both factories refuse inconclusive
conditions, damaged proof dependencies and exhausted resources; such a refusal
is not a proof that the mathematical conclusion fails.

The sensitivity record retains exact `projected_constraints`, `gram_inverse`,
`tangent_projector` \(R\), `point_slope` \(t\), `scalar_curvature`,
`base_jacobian` \(sR\), and `rank_one_numerator` \(tt^T/(2\nu)\).
Its `denominator`, `target_jacobian`, `multipliers` and `slacks` are interval
enclosures. Multiplier and slack entries follow the original inequality-row
order; inactive multipliers and selected slacks are exact zeros.

The `vjp(cotangent)` query accepts exact integer or `Fraction` entries and
encloses the target VJP. It contracts the rational geometry before interval
conversion, preserving exact normal null directions. This query requires no
new transcendental evaluation or linear solve.

```python
from fractions import Fraction
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    compile_gaussian_polyhedral_reduction,
)
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    certify_gaussian_polyhedral_reduced_minimum,
)
from omnibias.core.verified.gaussian_polyhedral_unique import (
    certify_gaussian_polyhedral_unique,
)
from omnibias.core.verified.gaussian_polyhedral_reduced_sensitivity import (
    certify_gaussian_polyhedral_reduced_sensitivity,
)

model = GaussianHeatMap(
    ((-1, 0, 0), (1, 0, 0)), (0, 0), viscosity=Fraction(1, 4),
)
rows = ((-1, 1, 1), (-1, -1, -2))
reduction = compile_gaussian_polyhedral_reduction(
    model, rows, (1, -3), (-7, 0, 0),
)
assert reduction is not None
minimum = certify_gaussian_polyhedral_reduced_minimum(
    reduction, max_boxes_per_piece=63,
)
assert minimum is not None
unique = certify_gaussian_polyhedral_unique(minimum)
assert unique is not None and unique.method == "strong_convexity"
index = next(i for i, piece in enumerate(reduction.pieces)
             if piece.active_indices == (0, 1))
sensitivity = certify_gaussian_polyhedral_reduced_sensitivity(unique, index)
assert sensitivity is not None and sensitivity.denominator.lo > 0
assert sensitivity.scalar_curvature == 7
assert sensitivity.point_slope == (Fraction(1, 2), Fraction(3, 2), -1)

tangent = (1, 3, -2)
for i, row in enumerate(sensitivity.target_jacobian):
    for j, enclosure in enumerate(row):
        exact = Fraction(tangent[i]*tangent[j], 12)
        assert Fraction(enclosure.lo) <= exact <= Fraction(enclosure.hi)
for normal in rows:
    assert all(value.lo == value.hi == 0 for value in sensitivity.vjp(normal))
```

The critical case separates the two certificates:

```python
critical = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=Fraction(1, 2))
critical_reduction = compile_gaussian_polyhedral_reduction(critical, (), (), (0,))
assert critical_reduction is not None
critical_minimum = certify_gaussian_polyhedral_reduced_minimum(critical_reduction)
assert critical_minimum is not None
critical_unique = certify_gaussian_polyhedral_unique(critical_minimum)
assert critical_unique is not None
assert critical_unique.method == "strict_convexity"
assert critical_unique.curvature_lower_bound == 0
assert certify_gaussian_polyhedral_reduced_sensitivity(critical_unique, 0) is None
```

## What a refusal means

These tests are sufficient, not necessary. A unique corner minimum may fail
the face-interior condition. Redundant tight inequalities may fail a strict
inactive-slack test. A critical unique inverse may have zero curvature and an
unbounded ordinary derivative. None of these refusals disproves uniqueness
or establishes a particular alternative derivative.

At an exact two-minimum tie the global uniqueness certificate must refuse.
At a critical two-center point with \(q=1\), the objective is strictly
convex and the global minimizer is unique, but the ordinary target derivative
at the critical target is singular. The existing
[critical chart](gaussian-critical-inverse.md) describes that separate case.

The Jacobian certificate concerns fixed original centers, biases, viscosity,
age and constraint geometry. It does not certify derivatives with respect to
those parameters, a VJP through an uncertain upstream gradient, ordinary
derivatives across active-set switches, or a finite perturbation radius.

## Prior work

The face formula is the reduced-Hessian/KKT sensitivity rule specialized to a
Gaussian rank-one Hessian. Strict complementarity and a positive reduced
Hessian are classical active-set sensitivity conditions; see
[Ferreau et al., *qpOASES: a parametric active-set algorithm for quadratic
programming*](https://publications.syscop.de/Ferreau2014.pdf).
The complete global cover provides the global justification that local KKT
conditions alone cannot supply for a nonconvex objective. No new general
sensitivity theorem or external formal theorem-prover verification is claimed.
