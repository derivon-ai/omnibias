# Inverting a positive exponential PDE field

The positive Cole–Hopf family in [exact nonlinear PDE identities](nonlinear-pde-identities.md)
connects a nonlinear PDE solution to a finite convex inverse problem. Given a
target velocity, its spatial preimage can be found by matching the mean of a
softmax distribution. The convex geometry supplies a global uniqueness
argument; interval arithmetic can certify a numerical root.

This is an application of classical exponential-family and convex-duality
results. It does not make general nonlinear PDEs linear, make all neural
networks convex, or guarantee rapid convergence near an ill-conditioned target.

## From the PDE field to a mean map

For fixed spatial modes \(k_j\in\mathbb R^d\), finite real biases \(b_j\),
and viscosity \(\nu>0\), set

\[
\tau(x,t)=\sum_{j=1}^n
\exp\bigl(k_j\cdot x+\nu\lVert k_j\rVert^2t+b_j\bigr),
\qquad U(x,t)=-2\nu\nabla_x\log\tau(x,t).
\]

The sum is nonempty and strictly positive. Each summand satisfies the heat
equation, and the Cole–Hopf identity proves that \(U\) solves the irrotational
vector Burgers system. Biases may vary freely without breaking this identity;
the relation between the temporal and spatial weights must remain intact.

Fix time and write \(a_j=\nu\lVert k_j\rVert^2t+b_j\). Define

\[
Z(x)=\sum_j e^{k_j\cdot x+a_j},\qquad
p_j(x)=\frac{e^{k_j\cdot x+a_j}}{Z(x)},\qquad
\mu(x)=\sum_jp_j(x)k_j.
\]

All \(p_j(x)\) are strictly positive at every finite \(x\). Inverting the
velocity \(U(x,t)=-2\nu m\) is exactly the equation \(\mu(x)=m\). It is
the stationary-point equation of

\[
\Phi_m(x)=\log Z(x)-m\cdot x.
\]

Differentiating the finite sum gives

\[
\nabla\Phi_m(x)=\mu(x)-m,\qquad
C(x):=\nabla^2\Phi_m(x)
=\sum_jp_j(x)(k_j-\mu(x))(k_j-\mu(x))^T.
\]

Thus the Hessian is the covariance matrix of the modes. In particular,
\(D_xU=-2\nu C(x)\) is negative semidefinite. This is a property of this
positive exponential representation; replacing its activation with an arbitrary
function does not preserve the calculation.

For every unit vector \(v\), the scalar values \(v\cdot k_j\) lie in an
interval of length at most \(\operatorname{diam}K\), where
\(K=\operatorname{conv}\{k_1,\ldots,k_n\}\). Subtracting the interval
midpoint shows that their variance is at most one quarter of its squared
length. Consequently,

\[
\lVert C(x)\rVert_2\le\frac{\operatorname{diam}(K)^2}{4},\qquad
\lVert D_xU(x,t)\rVert_2
\le\frac{\nu\operatorname{diam}(K)^2}{2}.
\]

This is a uniform upper bound on spatial sensitivity. It is not a uniform
lower bound on the Hessian or an upper bound on inverse sensitivity.

The hyperplanes \((k_j-k_\ell)\cdot x+a_j-a_\ell=0\) locate equal softmax
weights. Bias changes move these hyperplanes while preserving the PDE family.
The inverse problem uses the whole weighted mean; intersecting equal-weight
hyperplanes alone does not generally solve a prescribed target equation.

## Exact existence and uniqueness conditions

Assume first that the differences \(k_j-k_1\) span \(\mathbb R^d\).
This is **full affine span**. Ordinary linear rank of the uncentered modes is
insufficient: \((1,0)\) and \((0,1)\) have linear rank two but affine span one.

For nonzero \(v\),
\[
v^TC(x)v=\operatorname{Var}_{p(x)}(v\cdot k)>0,
\]
because every weight is positive and the projected modes cannot all be equal.
Therefore \(\Phi_m\) is strictly convex on \(\mathbb R^d\). There are three
different target cases:

| Target | Finite preimage and optimization behavior |
| --- | --- |
| \(m\in\operatorname{int}K\) | A unique finite preimage, which is the unique global minimizer of \(\Phi_m\). |
| \(m\in\partial K\) | No finite preimage; the infimum is finite and unattained. Means can approach the target as inputs escape to infinity. |
| \(m\notin K\) | No preimage; \(\Phi_m\) is unbounded below in a separating direction. |

Here is a direct proof of existence in the first case. Choose \(r>0\) with
the ball \(B(m,r)\subset K\). The support function
\(h_K(x)=\max_j k_j\cdot x\) then obeys
\(h_K(x)-m\cdot x\ge r\lVert x\rVert\). Hence

\[
\Phi_m(x)\ge h_K(x)-m\cdot x+\min_j a_j
\ge r\lVert x\rVert+\min_j a_j.
\]

The continuous objective is coercive, so it attains a minimum. The gradient
vanishes there, and strict convexity proves uniqueness.

A finite mean cannot lie on a supporting face: a strictly positive average
attains a supporting linear maximum only when every mode lies on that face,
contradicting full affine span. For a boundary target, any feasible convex
combination gives the finite entropy lower bound below, while evaluation at a
finite input gives a finite upper bound on the infimum. No minimum can be
attained, since an attained minimum would have zero gradient. To see that every
boundary target is a limit of means, approach it through interior targets and
use their preimages. These preimages have no bounded subsequence, since
continuity would otherwise give a finite boundary preimage.

For an exterior target, strict separation gives a vector \(v\) with
\(m\cdot v>h_K(v)\). For \(s>0\),
\[
\Phi_m(sv)\le s\bigl(h_K(v)-m\cdot v\bigr)
                  +\max_j a_j+\log n\longrightarrow-\infty.
\]

If the modes have a deficient affine span, let
\(L=\operatorname{span}\{k_j-k_1\}\). The range of the mean map is
\(\operatorname{relint}K\). For a target in that relative interior there is
one solution modulo \(L^\perp\): adding a vector in \(L^\perp\) shifts all
logits equally and changes neither probabilities nor the mean. The objective
is also unchanged when \(m\in\operatorname{aff}K\). Any implementation that
returns a unique input must fix a gauge or explicitly require full affine span.
Duplicate modes are allowed; finite biases still give each duplicate positive
mass. Removing a mode, for example through an idealized bias of \(-\infty\),
changes the convex hull and is a different problem.

These statements specialize the mean-parameter results in
[Wainwright and Jordan, *Graphical Models, Exponential Families, and Variational
Inference*](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf),
Sections 3.5–3.6 and Appendix B. In that author-hosted version, Proposition 2
gives the covariance Hessian, Proposition 3 gives injectivity under minimality,
and Theorem 1 gives surjectivity onto the interior. Minimality here is exactly
the affine-span condition.

## Rational witnesses and a global objective bound

For rational modes and a rational target, a proposed vector \(q\) can be checked
using exact rational arithmetic:

\[
q_j\ge0,\qquad \sum_jq_j=1,\qquad \sum_jq_jk_j=m.
\]

This proves membership in \(K\). If every \(q_j>0\), full affine span proves
interior membership: a supporting equality at \(m\) would force equality at
every mode. An exact positive witness can therefore establish existence before
the numerical inverse solve. A zero component does not prove that the target
is on the boundary; another representation may have all positive components.
Approximate feasibility within a floating-point tolerance is not this exact
witness.

Every feasible \(q\), including those with zero components, gives

\[
B(q)=\sum_jq_ja_j-\sum_{j:q_j>0}q_j\log q_j
\le\inf_x\Phi_m(x).
\]

The omitted zero terms use \(0\log0=0\). Indeed, substituting the softmax
formula and using the exact moment constraints gives

\[
\Phi_m(x)-B(q)
=\sum_{j:q_j>0}q_j\log\frac{q_j}{p_j(x)}
=\operatorname{KL}(q\Vert p(x))\ge0.
\]

Thus an interval lower bound for \(B(q)\) and an interval upper bound for
\(\Phi_m\) at any finite witness yield a global minimum-value enclosure.
The logarithms require rigorous transcendental enclosures. Exact rational
feasibility alone does not make the entropy rational, nor does it make the
bound tight. In an interior problem equality holds when \(q=p(x_*)\); a fixed
feasible witness can remain loose even after the numerical input converges.

## Certification and conditioning

A stable numerical solve can propose an input using the gradient and covariance
matrix. Subtracting a common logit before exponentiation avoids many overflow
problems, but does not itself certify the result. Floating-point underflow can
turn a positive probability into numerical zero, and very large dot products
can overflow before the shift. Such numerical failures must not be interpreted
as a proof that a target is on a face or outside the convex hull.

The verified path encloses the residual \(F(x)=\mu(x)-m\) and its Jacobian
\(C(x)\) on a box. A successful Krawczyk test proves that a true zero exists
inside an explicit enclosure. Full affine span then upgrades this local root
claim to the unique global minimizer of \(\Phi_m\). This uses the existing
multidimensional Krawczyk primitive; it does not require a global box search.
Failure of the interval test is inconclusive. It cannot negate an exact
interior witness or establish boundary/exterior membership.

At an interior target the implicit function theorem gives the inverse
derivatives

\[
D_mx_*=C(x_*)^{-1},\qquad
\frac{\partial x_*}{\partial a_j}
=-C(x_*)^{-1}p_j(x_*)(k_j-m).
\]

For a fixed target and the heat-dispersion biases
\(a_j(t)=\nu\lVert k_j\rVert^2t+b_j\), this also gives

\[
\frac{d x_*}{dt}
=-\nu C(x_*)^{-1}
\operatorname{Cov}_{p(x_*)}(k,\lVert k\rVert^2).
\]

These are identities for the exact solution. Evaluating them at an approximate
input produces numerical sensitivities unless the input and matrix solve are
enclosed separately. No global lower eigenvalue bound follows from full affine
span: probabilities may concentrate as the input tends to infinity, making
the covariance nearly singular. A small residual alone therefore does not
provide a small distance-to-solution bound. This matters especially near the
boundary of the target hull.

## Exact model and verified API

`omnibias.core.verified.mean_map.MeanMap(modes, biases, target, budget=None)`
stores a frozen exact model. Entries are integers or `Fraction` objects, with
one bias per mode and one target coordinate per spatial dimension. Its
`dimension`, `mode_count` and `affine_rank` properties describe the checked
model; `affine_rank` is computed from the mode differences using exact
arithmetic. The optional resource budget is a `RidgeBudget`.

| Method | Contract |
| --- | --- |
| `objective(box)` | Encloses \(\Phi_m\) over the supplied interval box. |
| `residual(box)` | Encloses every coordinate of \(\mu-m\). |
| `jacobian(box)` | Encloses every covariance-matrix entry. |
| `dual_bound(weights)` | Checks exact feasibility of \(q\), then encloses \(B(q)\). Invalid mathematical witnesses raise `ValueError`. |
| `target_interior(weights)` | Checks strict positivity, exact feasibility and full affine rank. A false result is inconclusive about hull membership. |
| `float_evaluate(point)` | Returns untrusted original and common-bias-reduced objectives, a residual and a covariance for proposing an iterate. |

`target_interior` returns false for invalid mathematical witnesses or deficient
rank; malformed types, shapes and exceeded budgets still raise. A failed
witness is not a separating hyperplane. The model's transcendental interval
evaluations require the rigorous backend, available through
`omnibias-core[verified]`.

Exact common mode and bias offsets are removed before numerical evaluation.
The proposal's `reduced_objective` excludes the first bias, so a large common
constant does not erase small descent differences. The original objective
still has to fit the numerical range for `float_evaluate` to succeed. If it
does not, the iterative solver can return `"not_certified"` even when the
residual and Jacobian remain evaluable for a direct root check.

For a small example, take modes \(0,e_1,e_2\), biases \((0,-1,1)\), and
target \((1/3,1/3)\). At \(x_*=(1,-1)\) all three logits vanish, so the
uniform weights give the target exactly. The following code asks the solver
to find and certify that input, starting from its default zero proposal.

```python
from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_mean_map import inverse_mean_map
from omnibias.core.verified.linalg import interval_solve
from omnibias.core.verified.mean_map import MeanMap

third = Fraction(1, 3)
model = MeanMap(
    ((0, 0), (1, 0), (0, 1)),
    (0, -1, 1),
    (third, third),
)
q = (third, third, third)
assert model.affine_rank == model.dimension == 2
assert model.target_interior(q)

result = inverse_mean_map(model)
assert result.status == "certified"
certificate = result.certificate
assert certificate is not None
assert certificate.enclosure[0].contains(1.0)
assert certificate.enclosure[1].contains(-1.0)

# A separate global minimum-value enclosure from primal and dual witnesses.
lower = model.dual_bound(q)
upper = model.objective(tuple(Interval.point(x) for x in result.point))
minimum = Interval(lower.lo, upper.hi)
print(result.point, certificate.enclosure, minimum)

# First column of D_target x*: C(x*) inverse times the first unit vector.
column = interval_solve(model.jacobian(certificate.enclosure), (1, 0))
assert column[0].contains(6.0)
assert column[1].contains(3.0)
```

Here the exact minimum value is \(\log3\). The root certificate establishes
location and uniqueness; `minimum` is a separate value enclosure. Because
\(q=p(x_*)\) in this example, the entropy lower bound is optimal. The exact
inverse covariance is \(\begin{pmatrix}6&3\\3&6\end{pmatrix}\).
`interval_solve` verifies its own matrix enclosure conditions and raises when
they are inconclusive. A certified inverse point does not by itself guarantee
that an arbitrary-width interval sensitivity solve will succeed.

`inverse_mean_map` and `certify_mean_inverse` are imported from
`omnibias.core.verified.inverse_mean_map`.
`inverse_mean_map(model, initial=None, max_iterations=64, radii=...)` proposes
an input and tries a finite list of certificate radii. The default radii are
`(1e-8, 1e-6, 1e-4, 1e-2)`. Success stops at the first proven root enclosure.
The returned `point` remains an approximate center; no residual or position
tolerance is implied. Use `certificate.enclosure` for the proved root location.
Restricting the radius list, for example to `radii=(1e-8,)`, asks the checker
to certify a smaller tested box and can make a finite search inconclusive.
The returned result has `point`, `iterations`, `certificate`, and one of these
statuses:

| Status | Meaning |
| --- | --- |
| `"certified"` | A verified root enclosure and exact full affine rank establish the unique global minimizer. |
| `"not_certified"` | The computation did not produce that certificate. The proposed point carries no root or global-minimum guarantee. |
| `"rank_deficient"` | The modes fail the full affine-span requirement. The implementation does not choose a quotient gauge. |

The iteration budget and finite radius list do not guarantee certification of
every interior target. `"not_certified"` does not classify the target as
boundary or exterior. Likewise, `"rank_deficient"` does not prove that no
preimage exists.

For a proposed point and approximate inverse Jacobian,
`certify_mean_inverse(model, point, preconditioner, radius)` attempts the same
verified check directly. It returns a `MeanInverseCertificate` or `None`.
The certificate contains the proposal `point`, the actual tested interval
`box`, the tighter root `enclosure`, exact `affine_rank`, contraction bound
`kappa`, and a serialized `certificate`. These are interval-arithmetic claims;
they do not assert verification by the Lean kernel. The sensitivity identities
above currently have no dedicated sensitivity API.

## Scope

The mathematical model, numerical proposal, exact witness and interval root
enclosure have different roles. The root and objective claims concern the
ideal real-valued exponential model with the declared exact constants, not an
arbitrary hardware forward pass.

This inverse problem finds a spatial input at a fixed time and fixed biases.
It does not fit arbitrary PDE data, impose initial or boundary conditions, or
train an unrestricted neural network. The construction assumes the previously
verified Cole–Hopf solution family; solving its mean equation establishes the
requested velocity preimage within that family. Its convex geometry and
entropy duality are classical. The reusable capability is their integration
with exact model data, explicit witness checks and verified root enclosures.

The numerical distinction between shifted formulas and mathematical guarantees
is discussed in [Blanchard, Higham and Higham, *Accurate Computation of the
Log-Sum-Exp and Softmax Functions*](https://arxiv.org/abs/1909.03469). No universal
iteration bound, floating-point identity across backends, or formal Lean
verification is implied by this construction.
