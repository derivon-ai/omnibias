# Directional Gaussian decisions at active-set switches

A constrained Gaussian decision can have a unique global optimum and
well-defined one-sided directional derivatives even where its ordinary
Jacobian does not exist. This interface certifies an exact equilibrium,
compiles a proposed face of its critical-cone quadratic program, and checks
the resulting response for each target direction.

The Gaussian source, heat age, viscosity and polyhedron remain fixed.
Only the target varies. This extends the
[polyhedral minimum certificates](gaussian-polyhedral-constraints.md);
it does not infer a switching derivative from an arbitrary smooth face.

## An exact equilibrium of the declared model

For a rational point \(X\), the equilibrium checker requires exact equality
of all original Gaussian logits

\[
b_j-\frac{\|X-a_j\|^2}{4\nu s}.
\]

Consequently its posterior is exactly \(p_j=1/M\), where \(M\) is the
number of centers. The mean, velocity and spatial Hessian are rational:

\[
\mu=\frac1M\sum_j a_j,\qquad U(X)=\frac{X-\mu}{s},
\qquad
H=D_xU(X)=\frac1s\left(I-\frac{\operatorname{Cov}(a)}{2\nu s}\right).
\]

No rounded softmax probabilities or substituted model are used. The
condition binds the calculation to the supplied source model at the
supplied point. Unequal logits are outside this exact checker; this does
not imply that their optimizer or directional derivative is undefined.

The checker also verifies the proposed face equations, every inequality,
and the original stationarity equation with nonnegative inequality
multipliers. It requires \(q_{\mathrm{base}}<1\) on the base equality
space, so these KKT conditions certify the unique global optimum.
Primal multipliers are obtained from the declared independent face rows;
omitted inequality rows have multiplier zero.

All exactly tight inequalities are collected, including tight rows
omitted from the proposed face. Their full collection need not be
independent. Strictly positive multipliers identify strong-active rows;
tight rows with zero multipliers are weak-active rows. An interval that
merely contains zero would not justify this classification, which is why
exact equilibrium evidence is used here.

The factory
`certify_gaussian_polyhedral_equilibrium(face, target, point)` in
`omnibias.core.verified.gaussian_polyhedral_equilibrium` takes exact
integer or `Fraction` target and point coordinates. Its equilibrium
retains `face`, `target`, `point`, `posterior`, `velocity`,
`hessian`, `equality_multipliers`, `inequality_multipliers`,
`slacks`, `active_indices`, `strong_indices`, `weak_indices`,
`strong_convexity_bound`, and its `certificate`.
Here `active_indices` means all tight inequalities, not merely the
indices used to compile the original proposed face.

Malformed exact input raises. Unequal logits, failed feasibility or KKT
checks, negative inequality multipliers, failed base contraction, damaged
dependencies or exhausted budgets return `None`. The calculation uses
no exponentials or nonlinear iteration; the shared certificate-sealing
policy still applies.

## The critical-cone derivative

Write \(F_v(x)=\Phi(x)-v\cdot x\), and let \(\mathcal A\) contain all
inequalities tight at \(X\). The critical cone is

\[
K=\{w:Cw=0,\ G_iw\le0\ (i\in\mathcal A),\
             (\nabla\Phi(X)-v)^Tw=0\}.
\]

For any valid nonnegative KKT multiplier vector, this equals

\[
K=\{w:Cw=0,\ G_iw=0\ (i\in\mathcal A_+),\
                   G_iw\le0\ (i\in\mathcal A_0)\},
\]

where \(\mathcal A_+\) and \(\mathcal A_0\) are its strong- and
weak-active sets. Multiplier uniqueness is unnecessary: on the tangent
cone,
\((\nabla\Phi(X)-v)^Tw=-\sum_i\mu_iG_iw\) is a sum of nonnegative
terms, so it is zero exactly when all positive-multiplier terms vanish.
Any valid multiplier vector therefore describes the same critical cone.

For each target direction \(\xi\), the one-sided Hadamard derivative is

\[
\boxed{\quad
DX(v;\xi)=\lim_{t\downarrow0}\frac{X(v+t\xi)-X(v)}t
 =\underset{w\in K}{\arg\min}\,
   \left(\frac12w^THw-\xi^Tw\right).
\quad}
\]

The Hadamard statement also allows the direction to tend to \(\xi\)
as \(t\) tends to zero. The objective is strongly convex on the base
equality tangent, with modulus
\(m=(1-q_{\mathrm{base}})/s>0\), hence the quadratic program has one
solution. The ambient Hessian need not be positive definite.

To see the formula, strong monotonicity first bounds the scaled optimizer
displacements. The perturbed objective expands as

\[
F_{v+t\xi}(X+tw)-F_{v+t\xi}(X)
=t(\nabla\Phi(X)-v)^Tw
+t^2\left(\frac12w^THw-\xi^Tw\right)+o(t^2).
\]

Comparison with feasible directions in \(K\) forces every displacement
limit into \(K\), and then makes it minimize the displayed quadratic.
Uniqueness gives convergence. Uniform expansion on bounded directions
gives the Hadamard conclusion.

The response is positively homogeneous and piecewise linear in \(\xi\),
and obeys

\[
\|DX(v;\xi)\|_2\le\frac{\|P_0\xi\|_2}{m}.
\]

It need not be linear, so a directional response is not automatically
an ordinary Jacobian-vector product.

## Compiling and checking a directional face

Choose a subset \(\mathcal B\) of weak-active rows to impose as equalities
in the quadratic program, and form

\[
E_{\mathcal B}=
\begin{pmatrix}C\\G_{\mathcal A_+}\\G_{\mathcal B}\end{pmatrix}.
\]

The selected stacked rows must be independent. Let \(N\) be any exact
basis matrix for their nullspace. The equality-constrained quadratic
response is

\[
w=B\xi,\qquad
B=N(N^THN)^{-1}N^T.
\]

Only \(N^THN\) must be positive definite. Its factorization can be
compiled once; subsequent queries use rational matrix actions and sign
checks. A zero-dimensional nullspace gives \(B=0\) without a reduced
solve.

The query checks nonnegative quadratic-program multipliers for selected
weak rows and \(G_iw\le0\) for every unselected weak row. Multipliers
for the base equalities and strong-active rows have no sign restriction.
If these checks pass, the proposed response satisfies the critical-cone
KKT conditions and is the unique true directional derivative.
Non-strict signs are sufficient: this is an optimality check for one
direction, not a claim of differentiability of the response across its
own switching boundaries.

There is no search for the weak face. A failed face can return no result
while another face succeeds. The compiled matrix \(B\) is a candidate
branch response; multiplying it by an unchecked direction does not
establish that the result is the constrained directional derivative.
With redundant tight rows, a failed sign check need not mean \(B\xi\)
is wrong: another multiplier representation may certify the same value.

The compiler
`compile_gaussian_critical_face(equilibrium, active_weak_indices=())`
and query `certify_gaussian_polyhedral_directional(face, direction)`
are in `omnibias.core.verified.gaussian_polyhedral_directional`.
Weak indices refer to original inequality rows. The compiler forms both
the reduced Hessian and row-Gram factorizations once, reusing each across
its right-hand sides. The query performs no solve or active-set
enumeration.

`GaussianCriticalFace` retains `equilibrium`, sorted
`active_weak_indices`, `constraints`, `response`,
`multiplier_response`, `region_inequalities`, and its
`certificate`. If \(M_{\mathcal B}\) denotes its multiplier response,
the compiler replays

\[
E_{\mathcal B}B=0,\qquad
HB+E_{\mathcal B}^TM_{\mathcal B}=I,\qquad B^T=B
\]

exactly. Its `region_inequalities` matrix \(R\) expresses the
selected weak multiplier and unselected weak slack conditions as
\(R\xi\ge0\).

The query takes exact integer or `Fraction` direction coordinates.
Its `GaussianPolyhedralDirectional` result retains `face`,
`direction`, exact `value`, `equality_multipliers`,
`weak_multipliers`, `weak_slacks`, `quadratic_value`, and its
`certificate`. The equality multipliers follow the compiled \(E\)
row order, including selected weak rows. The weak arrays follow every
original inequality index; entries at nonweak rows are structural zeros
and do not represent primal slacks or primal multipliers.

Invalid inputs or redundant selected rows raise. Unselected redundant
tight rows are allowed and remain in the sign checks. A wrong direction
region, damaged dependency, exhausted budget or failed exact replay
returns `None`.

## A coupled Gaussian switch

Take centers \((-1,0),(1,0)\), zero biases, \(\nu=s=1\), target zero,
and the inequality \(x+y\le0\). The equilibrium is \(X=(0,0)\);
its multiplier is zero and

\[
q_{\mathrm{base}}=\tfrac12,\qquad
H=\begin{pmatrix}1/2&0\\0&1\end{pmatrix},\qquad
K=\{w:w_1+w_2\le0\}.
\]

In direction \(\xi=(1,0)\), the unconstrained quadratic response
\((2,0)\) is infeasible. Imposing the weak row gives
\(DX(v;(1,0))=(2/3,-2/3)\).
For \(\xi=(-1,0)\), the unconstrained response \((-2,0)\) is feasible,
so \(DX(v;(-1,0))=(-2,0)\).
The two derivatives are not negatives of each other: an ordinary
Jacobian at this equilibrium does not exist.

```python
from fractions import Fraction

from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)
from omnibias.core.verified.gaussian_polyhedral_directional import (
    compile_gaussian_critical_face,
    certify_gaussian_polyhedral_directional,
)

source = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=1, time=1)
base = restrict_gaussian_heat(source, (), ())
assert base is not None
# Deliberately omit the tight row from the original proposed active set.
polyhedral_face = compile_gaussian_polyhedral_face(
    base, ((1, 1),), (0,), active_indices=(),
)
assert polyhedral_face is not None
equilibrium = certify_gaussian_polyhedral_equilibrium(
    polyhedral_face, (0, 0), (0, 0),
)
assert equilibrium is not None
assert equilibrium.posterior == (Fraction(1, 2), Fraction(1, 2))
assert equilibrium.hessian == ((Fraction(1, 2), 0), (0, 1))
assert equilibrium.active_indices == equilibrium.weak_indices == (0,)
assert equilibrium.strong_indices == ()

free = compile_gaussian_critical_face(equilibrium)
wall = compile_gaussian_critical_face(equilibrium, active_weak_indices=(0,))
assert free is not None and wall is not None
positive = certify_gaussian_polyhedral_directional(wall, (1, 0))
negative = certify_gaussian_polyhedral_directional(free, (-1, 0))
assert positive is not None and negative is not None
assert positive.value == (Fraction(2, 3), Fraction(-2, 3))
assert positive.quadratic_value == Fraction(-1, 3)
assert positive.weak_multipliers == (Fraction(2, 3),)
assert negative.value == (-2, 0)
assert negative.weak_slacks == (2,)

# Each proposed face refuses a direction outside its certified region.
assert certify_gaussian_polyhedral_directional(free, (1, 0)) is None
assert certify_gaussian_polyhedral_directional(wall, (-1, 0)) is None

# Closed regions overlap at a zero-multiplier boundary; both certify
# the same unique directional derivative there.
boundary_free = certify_gaussian_polyhedral_directional(free, (1, -2))
boundary_wall = certify_gaussian_polyhedral_directional(wall, (1, -2))
assert boundary_free is not None and boundary_wall is not None
assert boundary_free.value == boundary_wall.value == (2, -2)
```

## Scope and prior work

This is classical directional sensitivity through a critical-cone
quadratic program. See
[Jittorntrum's analysis without strict complementarity](https://link.springer.com/chapter/10.1007/BFb0121215)
and [Levy and Rockafellar on solution mappings](https://sites.math.washington.edu/~rtr/papers/rtr163-GeometrySolutionMappings.pdf).
The implementation specializes that theory to exact Gaussian
equilibria and bounded rational certificates; no general novelty is
claimed.

The result concerns a first directional derivative at one fixed source,
geometry and target. It does not certify a finite-step Taylor remainder,
an active-set neighborhood radius, higher derivatives, or an ordinary
VJP at a switch. For a fixed loss upstream \(g\), \(g^TDX(v;\xi)\)
is a scalar directional derivative, but there need not be a single
vector representing it for every direction. These certificates are not
theorem-prover verification, and finite resource budgets still apply.
