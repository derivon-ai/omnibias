# Higher-order Gaussian paths through active-set switches

Exact Taylor coefficients can distinguish constrained optimizer branches
when the first directional derivative is zero. This interface follows a
specified polynomial path in the original Gaussian biases and target,
then checks the eventual signs of its KKT multipliers and inequality
slacks on a proposed face.

Centers, viscosity, heat age and the polyhedron remain fixed. A `valid`
result proves that the analytic face path is the unique global optimizer
for all sufficiently small positive path parameters. It supplies neither
a numerical neighborhood radius nor an error bound for evaluating its
truncated Taylor polynomial at a nonzero parameter.

## Exact paths of the original model

Write the constraints as \(Cx=d\), \(Gx\le h\), and prescribe

\[
v(t)=\sum_{n=0}^N v_nt^n,\qquad
b_j(t)=\sum_{n=0}^N b_{jn}t^n.
\]

These are ordinary Taylor coefficients: a raw order-\(n\) derivative is
\(n!\) times its coefficient. The finite input polynomials specify the
whole parameter path; the resulting optimizer and posterior series need
not terminate at order \(N\). Adding zero input rows requests more
output coefficients for the same polynomial path.

The base record is an
[exact equal-logit equilibrium](gaussian-polyhedral-directional.md) of the
declared source. All original Gaussian logits agree at its rational point
\(x_0\), so the posterior there is exactly uniform. This condition is
checked on the original model, not on rounded probabilities or a fitted
nearby root. The target at a constrained equilibrium need not equal the
unconstrained velocity.

Choose the independent rows \(E\) consisting of the base equalities and
the proposed active inequalities, with right-hand side \(e\). Let
\(P\) be their exact tangent projector and
\(L_E=(EE^T)^{-1}E\). The existing
[affine restriction](gaussian-affine-constraints.md) gives centers
\(a'_j\) and biases whose posterior agrees with the original posterior
on \(Ex=e\). Because the geometry and heat age are fixed, higher bias
coefficients pass through this restriction unchanged, up to a common
bias shift. Target coefficients become \(Pv_n\).

The sufficient condition \(q_{\mathrm{base}}<1\) ensures global strong
convexity on the base equality space for every bias along the path.
Further projection cannot increase the center diameter, so the selected
face also satisfies the contraction condition. The original ambient
Hessian may be indefinite in normal directions.

## One reusable factor for all Taylor orders

For the restricted family, define

\[
x(t)=x_0+\sum_{n\ge1}x_nt^n,\qquad
\gamma_{jn}=b_{jn}-b_{0n},\qquad
r_j=\frac{a'_j-a'_0}{2\nu s}.
\]

Equal base logits let the posterior be written using the relative
exponentials

\[
\exp\!\left(\sum_{n\ge1}\ell_{jn}t^n\right)
 =\sum_{n\ge0}e_{jn}t^n,\qquad
\ell_{jn}=\gamma_{jn}+r_j\cdot x_n,\qquad e_{j0}=1.
\]

At order \(n\), all lower coefficients are known. Differentiating the
exponential series gives its known contribution

\[
\widehat e_{jn}=\gamma_{jn}
 +\frac1n\sum_{k=1}^{n-1}k\ell_{jk}e_{j,n-k},
\qquad e_{jn}=\widehat e_{jn}+r_j\cdot x_n.
\]

Divide the weighted moment series by the weighted exponential sum to
obtain the normalized mean. Its order-\(n\) coefficient has the form

\[
\mu'_n=\widehat\mu'_n+
 \frac{\operatorname{Cov}_{p_0}(a')}{2\nu s}x_n.
\]

Consequently the implicit equation \(x=sPv+\mu'\) yields

\[
\boxed{\quad
\left(I-\frac{\operatorname{Cov}_{p_0}(a')}{2\nu s}\right)x_n
 =sPv_n+\widehat\mu'_n.
\quad}
\]

The matrix is rational and positive definite. One exact LDLT
factorization is reused for every positive Taylor order. Exponential
recurrence and normalized series division use rational arithmetic; no
exponential evaluation or nonlinear root iteration is needed.

The same posterior coefficients must then be combined with the
**original** centers to recover the original KKT multipliers:

\[
\mu_n^{\mathrm{orig}}=\sum_j p_{jn}a_j,\qquad
\lambda_n=L_E\left(v_n-
                     \frac{x_n-\mu_n^{\mathrm{orig}}}{s}\right).
\]

This formula also holds at order zero. Omitted inequality multipliers
are identically zero. Slack coefficients are \(h-Gx_0\) at order zero
and \(-Gx_n\) thereafter. The implementation replays posterior
normalization, face feasibility and original stationarity coefficient by
coefficient. The underlying analytic implicit-function theorem identifies
these exact formal coefficients with the local analytic face solution.

## What the sign classification proves

For a real analytic scalar quantity
\(f(t)=c_kt^k+O(t^{k+1})\), with \(c_k\ne0\), the sign of \(f(t)\)
for sufficiently small \(t>0\) is the sign of \(c_k\). Apply this to
selected inequality multipliers and omitted inequality slacks. Selected
slacks and omitted multipliers are already identically zero.

An all-zero finite coefficient prefix does not prove an analytic
identity. The implementation recognizes these sufficient structural
cases:

- An omitted slack is constant if its row annihilates the selected face
  tangent.
- Constant target and relative-bias paths keep the whole equilibrium
  fixed; a common bias polynomial has no effect on the optimizer.
- A selected multiplier is constant if its row of \(L_E\) annihilates
  every original center contrast and every higher target coefficient.
  Here \(L_Ex\) is constant because \(Ex=e\).

The statuses have distinct meanings:

| Status | Meaning |
| --- | --- |
| `valid` | Every required sign is positive or structurally zero. There exists a common \(\delta>0\) on which KKT holds for all \(0<t<\delta\); base strong convexity makes this the unique global optimizer. |
| `invalid` | At least one selected KKT sign is eventually negative. A negative slack proves primal infeasibility. A negative multiplier rejects this selected multiplier representation only. |
| `inconclusive` | No negative sign was found, but at least one zero coefficient prefix lacks an identity proof. More Taylor orders may resolve it. |

There are finitely many inequalities, so the individual sign
neighborhoods have a common positive radius. The certificate asserts its
existence without calculating it. When `status == "valid"`,
`eventual_active_indices` contains the selected inequalities and omitted
rows proved identically tight. The other omitted slacks are eventually
strictly positive.

## API

`compile_gaussian_polyhedral_path(equilibrium,
higher_target_coefficients, higher_bias_coefficients)` is provided by
`omnibias.core.verified.gaussian_polyhedral_path`. It accepts equally many
exact integer or `Fraction` rows for orders \(1,\ldots,N\), including
explicit zero rows. The equilibrium determines the original order-zero
target, bias values and selected face.

The returned `GaussianPolyhedralPath` retains `equilibrium`, `series`,
`target_coefficients`, `bias_coefficients`, `coefficients`,
`posterior_coefficients`, `equality_multiplier_coefficients`,
`inequality_multiplier_coefficients`, `slack_coefficients`, `sign_tests`,
`status`, `eventual_active_indices`, `order`, and `certificate`.
All coefficient matrices are order-major and include order zero.
Equality multiplier columns follow the base equality rows; inequality
multiplier and slack columns follow every original inequality.
The nested `series` is bound to the transformed face model. Its base
biases and projected targets therefore differ from the original values
retained on the outer path record; its posterior coefficients agree with
those of the original model along the face.

Each `GaussianPathSign` gives `quantity`, original `index`,
`coefficients`, `leading_order`, `leading_coefficient`, `verdict`, and
an optional structural `identity`. A `zero` verdict requires that
identity; an all-zero prefix alone gives `inconclusive`.

The underlying `gaussian_bias_inverse_series` factory in
`omnibias.core.gaussian_bias_jet` also exposes the equality-free primitive
for a supplied model and exactly equal-logit base point. Its
`GaussianBiasInverseSeries.evaluate(t)` evaluates only the finite inverse
polynomial, exactly, without adding a remainder.

Malformed inputs raise. Damaged dependencies, exhausted exact budgets or
failed identity replay return `None` from the path compiler. Such refusal
is separate from a returned `invalid` sign classification. Work, storage,
Taylor order and rational coefficient sizes are bounded; exact arithmetic
does not imply unlimited practical order. Certificate sealing does not
constitute verification by an external formal proof checker.

## Cubic contact resolves a zero first derivative

Take centers \(-1,1\), \(\nu=s=1\), initial biases
\((1/4,1/4)\), and the inequality \(x\le0\). Along

\[
v(t)=-t,\qquad b_1(t)=1/4,\qquad b_2(t)=1/4+2t,
\]

the [joint directional derivative](gaussian-joint-directional.md) is
zero. The free equality branch satisfies

\[
x-\tanh(x/2+t)=-t,
\]

and has expansion

\[
x_{\mathrm{free}}(t)
=-\frac23t^3+\frac{14}{15}t^5-\frac{538}{315}t^7+O(t^9).
\]

Its omitted slack \(-x\) is positive at leading order three. The
boundary branch \(x=0\) instead has selected multiplier

\[
\lambda_{\mathrm{bound}}(t)=\tanh t-t
=-\frac13t^3+\frac{2}{15}t^5-\frac{17}{315}t^7+O(t^9).
\]

Thus orders one and two are inconclusive, while order three accepts the
free branch and rejects the boundary representation. In this example
there is no alternative inequality representation, so the boundary
branch is not the optimizer for small positive \(t\).

```python
from fractions import Fraction

from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)
from omnibias.core.verified.gaussian_polyhedral_path import (
    compile_gaussian_polyhedral_path,
)

source = GaussianHeatMap(
    ((-1,), (1,)), (Fraction(1, 4), Fraction(1, 4)), viscosity=1, time=1,
)
base = restrict_gaussian_heat(source, (), ())
assert base is not None
free_face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), ())
bound_face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), (0,))
assert free_face is not None and bound_face is not None
free_eq = certify_gaussian_polyhedral_equilibrium(free_face, (0,), (0,))
bound_eq = certify_gaussian_polyhedral_equilibrium(bound_face, (0,), (0,))
assert free_eq is not None and bound_eq is not None

def contact_path(equilibrium, order):
    targets = ((-1,),) + ((0,),) * (order - 1)
    biases = ((0, 2),) + ((0, 0),) * (order - 1)
    result = compile_gaussian_polyhedral_path(equilibrium, targets, biases)
    assert result is not None
    return result

for order in (1, 2):
    assert contact_path(free_eq, order).status == "inconclusive"
    assert contact_path(bound_eq, order).status == "inconclusive"

free = contact_path(free_eq, 7)
bound = contact_path(bound_eq, 3)
assert free.status == "valid" and free.eventual_active_indices == ()
assert bound.status == "invalid"
assert tuple(free.coefficients[n][0] for n in (3, 5, 7)) == (
    Fraction(-2, 3), Fraction(14, 15), Fraction(-538, 315),
)
assert free.sign_tests[0].leading_order == 3
assert free.sign_tests[0].leading_coefficient == Fraction(2, 3)
assert bound.sign_tests[0].leading_coefficient == Fraction(-1, 3)
```

## A negative selected multiplier is not always a wrong optimizer

Redundant tight inequalities can permit another multiplier
representation. For example, \(x\le0\) and \(-x\le0\) force
\(x=0\). With the same Gaussian source, constant biases and target
\(v(t)=-t\), selecting only the first row gives multiplier \(-t\).
The compiler returns `invalid`, although the sole feasible point is
necessarily the optimizer: the second row could instead carry multiplier
\(t\).

```python
singleton_face = compile_gaussian_polyhedral_face(
    base, ((1,), (-1,)), (0, 0), (0,),
)
assert singleton_face is not None
singleton_eq = certify_gaussian_polyhedral_equilibrium(
    singleton_face, (0,), (0,),
)
assert singleton_eq is not None
singleton_path = compile_gaussian_polyhedral_path(
    singleton_eq, ((-1,),), ((0, 0),),
)
assert singleton_path is not None
assert singleton_path.status == "invalid"
assert singleton_path.coefficients == ((Fraction(0),), (Fraction(0),))
assert singleton_path.inequality_multiplier_coefficients[1] == (-1, 0)
assert singleton_path.slack_coefficients[1] == (0, 0)
```

## Scope and prior work

Taylor propagation and differentiation of implicitly defined functions
are established techniques; see Wagner, Walther and Schaefer,
[On the efficient computation of high-order derivatives for implicitly
defined functions](https://arxiv.org/abs/0912.2208), and Griewank and
Walther's chapters on
[Taylor coefficients](https://epubs.siam.org/doi/10.1137/1.9780898717761.ch13)
and [one-sided smooth arcs through nonsmooth
programs](https://epubs.siam.org/doi/10.1137/1.9780898717761.ch14).
The formulas here specialize those ideas to exact Gaussian posterior
series and original polyhedral KKT evidence; no new general theorem or
universal optimizer is claimed.

The certificate considers one proposed primal face, one exact polynomial
parameter path, and the positive parameter side. It performs no active-set
search and does not support arbitrary nonuniform base posteriors. A
`valid` analytic face path gives one-sided optimizer coefficients along
that path, not an ordinary multivariate Jacobian at a switch. No finite
step, convergence radius or approximation accuracy follows merely from
the stored coefficients. Explicit neighborhoods would require additional
analytic remainder bounds for both the primal path and its original KKT
quantities.
