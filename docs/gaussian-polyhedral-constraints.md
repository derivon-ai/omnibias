# Gaussian inverse decisions under polyhedral constraints

The affine Gaussian certificates can check a proposed active set for

\[
\min_x F_v(x)=\Phi(x,s)-v\cdot x
\quad\text{subject to}\quad Cx=d,\quad Gx\le h.
\]

The checker combines an exact active-face restriction with rigorous
multiplier and slack bounds. It does not search for an active set.
Its global guarantee requires convexity on the original equality space;
positive curvature on the proposed face alone is insufficient.
The model and exact affine reduction are described in
[Gaussian affine constraints](gaussian-affine-constraints.md).

## Global optimality and a stable active set

Let \(P_0\) be the orthogonal projector onto \(\ker C\), and define

\[
q_{\mathrm{base}}=
\frac{\max_{i,j}\|P_0(a_i-a_j)\|_2^2}{8\nu s}.
\]

When \(q_{\mathrm{base}}<1\), the objective on \(Cx=d\) is globally
strongly convex with modulus

\[
m=\frac{1-q_{\mathrm{base}}}{s}>0.
\]

The ambient objective may still be nonconvex. The condition concerns the
base equality space before imposing any active inequalities.

For a proposed active set \(\mathcal A\), form

\[
E=\begin{pmatrix}C\\G_{\mathcal A}\end{pmatrix},
\qquad e=\begin{pmatrix}d\\h_{\mathcal A}\end{pmatrix}.
\]

The stacked rows must be independent. The affine certificate encloses
the exact optimizer \(X\) on \(Ex=e\), and the original-objective
stationarity multipliers satisfy

\[
\nabla F_v(X)+C^T\lambda+G_{\mathcal A}^T\mu=0.
\]

Suppose all active multipliers are certified nonnegative and all inactive
slacks \(h_i-G_iX\) are certified nonnegative. Then \(X\) is the unique
global constrained optimum. Indeed, every feasible \(Y\) satisfies

\[
\begin{aligned}
F_v(Y)-F_v(X)
&\ge \nabla F_v(X)\cdot(Y-X)+\frac m2\|Y-X\|_2^2\\
&=-\mu^T(G_{\mathcal A}Y-h_{\mathcal A})
  +\frac m2\|Y-X\|_2^2\\
&\ge\frac m2\|Y-X\|_2^2.
\end{aligned}
\]

Thus explicit KKT witnesses suffice; this argument does not require an
additional Slater assumption. A multiplier or slack enclosure that
extends below zero is inconclusive, even if its midpoint is positive.

For a locally stable active set, require the stronger margins

\[
\inf\mu_j>0\quad(j\in\mathcal A),\qquad
\inf(h_i-G_iX)>0\quad(i\notin\mathcal A).
\]

Full row rank of \(E\) and positive tangent curvature make its KKT
matrix nonsingular. The implicit function theorem and the strict margins
then preserve this active set in a neighborhood of the declared data.
The certificate does not specify a radius for that neighborhood.
Nonnegative margins can certify the minimum without certifying this
derivative claim.

## Checking an active face

`compile_gaussian_polyhedral_face(base_restriction, inequalities,
upper_bounds, active_indices=())` in
`omnibias.core.verified.gaussian_polyhedral` prepares the proposed face.
The inequalities and bounds are exact integers or `Fraction` values.
Its source remains the original Gaussian family; the intermediate
projected biases are not treated as independently fixed original
parameters.

Then `certify_gaussian_polyhedral_minimum(face, target, scaled_point)`
checks an exact target and an untrusted binary64 proposal \(x/s\).
Primal multipliers are evaluated independently of a loss upstream.
Even if the active equalities fix a single point, its multiplier signs
still need to be checked.

This example has negative ambient curvature at the optimum but is
strongly convex on the base equality \(y=1\).

```python
from fractions import Fraction

from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import (
    compile_gaussian_polyhedral_face,
    certify_gaussian_polyhedral_minimum,
)

source = GaussianHeatMap(
    ((-1, -2), (1, 2)), (Fraction(9, 4), Fraction(1, 4)),
    viscosity=1, time=1,
)
base = restrict_gaussian_heat(source, ((0, 1),), (1,))
assert base is not None
assert source.contraction_bound == Fraction(5, 2)
assert base.model.contraction_bound == Fraction(1, 2)

# y = 1, x <= 0, and -x <= 2; propose the first inequality as active.
face = compile_gaussian_polyhedral_face(
    base, ((1, 0), (-1, 0)), (0, 2), active_indices=(0,),
)
assert face is not None
minimum = certify_gaussian_polyhedral_minimum(face, (1, 0), (0.0, 1.0))
assert minimum is not None
assert minimum.strong_convexity_bound == Fraction(1, 2)
assert minimum.active_set_stable
assert minimum.root_enclosure[0].lo <= 0 <= minimum.root_enclosure[0].hi
assert minimum.root_enclosure[1].lo <= 1 <= minimum.root_enclosure[1].hi
assert minimum.inequality_multipliers[0].lo > 0
assert minimum.slacks[1].lo > 0
```

The exact optimum is \((0,1)\), with equality multiplier \(-1\),
active inequality multiplier \(1\), and inactive slack \(2\).
The ambient objective Hessian there is
\(\begin{pmatrix}1/2&-1\\-1&-1\end{pmatrix}\); its restriction to
the equality tangent is the positive scalar \(1/2\).

`GaussianPolyhedralMinimum` retains `face`, its `stationarity`
certificate, `root_enclosure`, `equality_multipliers`,
`inequality_multipliers`, `slacks`, `strong_convexity_bound`,
`active_set_stable`, and its `certificate`.
Inequality multipliers are ordered like the original inequality rows,
with exact zeros for inactive rows. Active slacks are exactly zero
because the true root satisfies the compiled face equalities.
An axis-aligned root enclosure can also contain points that violate
those equalities; its claim concerns the feasible optimizer inside it.

The frozen `GaussianPolyhedralFace` retains `base_restriction`,
`inequalities`, `upper_bounds`, sorted `active_indices`, the
compiled affine adjoint `geometry`, exact `slack_directions` and
`slack_offsets`, and its `certificate`. These slack data remove
normal directions before outward evaluation, avoiding unnecessary loss
when large coefficients cancel on the face.
Compilation makes no optimality or inactive-feasibility claim and does
not impose the contraction test. Invalid input or redundant stacked rows
raise; inconsistent active equations raise `InconsistentAffineConstraints`.
Damaged dependencies or exhausted budgets return `None`.

The minimum query returns `None` when global convexity, feasibility or
multiplier signs cannot be certified, or when numerical/resource limits
prevent a finite enclosure. Invalid query input raises. No finite
accuracy tolerance is promised, and any required rigorous transcendental
backend must be available.

## Why face convexity is not enough

Take \(\nu=s=1\), \(v=0\), centers \((-2,0),(4,0)\), biases
\((1,4)\), no base equalities, and the inequality \(x_1\le0\).
On its active face \(x_1=0\), the unique face minimizer is \(X=(0,0)\).
Its posterior is exactly \((1/2,1/2)\), so \(U(X)=(-1,0)\);
the active multiplier is \(1>0\). The face tangent Hessian is \(1\)
and \(q_{\mathrm{face}}=0\).

Nevertheless,

\[
\widetilde\tau(0,0)=2,\qquad
\widetilde\tau(-2,0)=e+e^{-5}>2,
\]

where \(\widetilde\tau\) is the unnormalized mixture sum, with the
common positive heat-kernel prefactor omitted.
The feasible point \((-2,0)\) therefore has strictly smaller
\(\Phi=-2\log\widetilde\tau+\text{constant}\).
The face certificate and strict multiplier
establish a strict local constrained minimum but not a global one.
Here \(q_{\mathrm{base}}=9/2\), so the global checker must refuse.

```python
nonconvex = GaussianHeatMap(((-2, 0), (4, 0)), (1, 4))
nonconvex_base = restrict_gaussian_heat(nonconvex, (), ())
assert nonconvex_base is not None
misleading_face = compile_gaussian_polyhedral_face(
    nonconvex_base, ((1, 0),), (0,), active_indices=(0,),
)
assert misleading_face is not None
assert misleading_face.geometry.restriction.model.contraction_bound == 0
assert nonconvex_base.model.contraction_bound == Fraction(9, 2)
assert certify_gaussian_polyhedral_minimum(
    misleading_face, (0, 0), (0.0, 0.0),
) is None
```

## Derivatives on a stable face and at a switch

`certify_gaussian_polyhedral_parameter_vjp(minimum, upstream, proposal)`
requires the strict margins recorded by `minimum.active_set_stable`.
It then certifies first derivatives of \(\mathcal L=g^TX\), with the
declared upstream \(g\) fixed, by applying the affine parameter adjoint
to \(E,e\). The derivative proposal is untrusted; its residual is checked
against the true constrained inverse.

If the affine adjoint multipliers are \(\eta=(\eta_C,\eta_{\mathcal A})\)
and its tangent target adjoint is \(z\), the active inequality derivatives
are

\[
\partial_{h_i}\mathcal L=\eta_i,\qquad
\nabla_{G_i}\mathcal L=-\mu_i z^T-\eta_iX^T
\quad (i\in\mathcal A).
\]

The bound derivative is an **adjoint** multiplier, not the primal
inequality multiplier \(\mu_i\). Inactive inequality and bound derivatives
are exactly zero locally. The remaining target, equality, center, bias,
viscosity and age derivatives are those of the original Gaussian
parameter certificate. Their row order maps back to the original input
constraints.

```python
from omnibias.core.verified.gaussian_polyhedral import (
    certify_gaussian_polyhedral_parameter_vjp,
)

# The stable face fixes both coordinates. Its target adjoint is zero,
# but moving the equality or active inequality bound moves the optimum.
derivatives = certify_gaussian_polyhedral_parameter_vjp(
    minimum, (1.0, 3.0), (0.0, 0.0),
)
assert derivatives is not None
assert derivatives.equality_rhs_gradient[0].lo == derivatives.equality_rhs_gradient[0].hi == 3
assert derivatives.inequality_rhs_gradient[0].lo == derivatives.inequality_rhs_gradient[0].hi == 1
assert derivatives.inequality_rhs_gradient[1].lo == derivatives.inequality_rhs_gradient[1].hi == 0
assert all(box.lo == box.hi == 0 for box in derivatives.inequality_gradient[1])
assert derivatives.inequality_gradient[0][1].lo <= -1 <= derivatives.inequality_gradient[0][1].hi
```

`GaussianPolyhedralParameterVJP` retains `minimum`, the underlying
affine `base_vjp`, `target_gradient`, `center_gradient`,
`bias_gradient`, `viscosity_gradient`, `time_gradient`,
`equality_gradient`, `equality_rhs_gradient`,
`inequality_gradient`, `inequality_rhs_gradient`, and its
`certificate`. All row-indexed gradients follow the original input
order, irrespective of the proposed active-index order.

At a switching boundary, a global optimum can remain certifiable while
an ordinary derivative fails to exist. For a single center at zero,
\(\nu=s=1\), and \(x\le0\), the objective differs from
\(x^2/2-vx\) only by a constant. Its optimizer is

\[
X(v)=\min(v,0).
\]

At \(v=0\), the active-face root and multiplier are both zero. The
minimum is unique, but the active multiplier has no strict positive
margin. The minimum certificate can therefore succeed with
`active_set_stable=False`, while the parameter certificate returns
`None`. Neither the active-face derivative \(0\) nor the inactive-face
derivative \(1\) is an ordinary derivative of \(X\) at that point.

```python
switch_source = GaussianHeatMap(((0,),), (0,))
switch_base = restrict_gaussian_heat(switch_source, (), ())
assert switch_base is not None
switch_face = compile_gaussian_polyhedral_face(
    switch_base, ((1,),), (0,), active_indices=(0,),
)
assert switch_face is not None
switch_minimum = certify_gaussian_polyhedral_minimum(switch_face, (0,), (0.0,))
assert switch_minimum is not None
assert not switch_minimum.active_set_stable
assert switch_minimum.inequality_multipliers[0].lo == switch_minimum.inequality_multipliers[0].hi == 0
assert certify_gaussian_polyhedral_parameter_vjp(
    switch_minimum, (1.0,), (0.0,),
) is None
```

In general, `active_set_stable=False` only means the strict-margin
test did not succeed; it does not prove nondifferentiability. The explicit
piecewise formula above establishes that fact for this particular example.

Strict-margin certification asserts a locally unchanged active set,
not validity across its switches. It gives no perturbation radius,
higher-derivative enclosure, or bound for an approximate external loss
upstream. The coordinate boxes may lose correlations even when the
underlying true derivatives satisfy exact identities.

## Scope and established methods

This is convex KKT sufficiency and active-set sensitivity specialized
to the Gaussian inverse certificates. These methods are classical; see
[Boyd and Vandenberghe's convex optimization material](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf)
and [OptNet's optimization-layer differentiation](https://proceedings.mlr.press/v70/amos17a.html).
No novelty of the general method is claimed.

The supported condition is sufficient, not necessary. Failure to certify
one proposed face does not establish infeasibility, nonuniqueness or
failure of another face. There is no active-set search, general
nonconvex global optimizer, or theorem-prover verification in this
interface. Exact arithmetic budgets and the rigorous numerical backend
requirements still apply.
