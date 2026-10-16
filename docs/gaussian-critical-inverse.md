# A certified Gaussian inverse at critical curvature

A two-center Gaussian heat field can have a unique global inverse even
when its Jacobian vanishes in one direction. At the exact critical
parameter, a cubic change of target coordinate gives an analytic inverse
chart with rational Taylor coefficients and an explicit error bound.

This covers a case outside the sufficient \(q<1\) condition of the
[regular Gaussian inverse](gaussian-heat-inverse.md). It does not extend
that condition to arbitrary Gaussian mixtures or remove the singular
derivative in the original target coordinate.

For two-center models beyond criticality, the
[nonconvex minimum certificate](gaussian-two-center-minimum.md) instead
selects the globally minimizing branch and retains both minimizers at an
exact tie. That problem can have other stationary points, so the global
minimum certificate does not assert a unique velocity inverse.

## The universal scalar inverse

For centers \(-1,1\), equal biases, viscosity \(\nu=1/2\) and heat
age \(s=1\), the velocity is

\[
U(x)=x-\tanh x,\qquad U'(x)=\tanh^2x.
\]

The derivative is positive except at zero. Its integral over every
nonempty interval is positive, so \(U\) is strictly increasing. Also
\(U(x)\to\pm\infty\) as \(x\to\pm\infty\). Hence a unique
global real inverse \(\psi\) exists. Near zero,

\[
U(x)=\frac{x^3}{3}+O(x^5),\qquad
\psi(w)\sim\sqrt[3]{3w}.
\]

The ordinary derivative of \(\psi\) at zero is not finite. Instead
define

\[
\chi(t)=\psi(t^3/3).
\]

This function is analytic and odd near zero, with exact coefficients

\[
\chi(t)=t+\frac2{15}t^3+\frac3{175}t^5
 +\frac2{1575}t^7-\frac{16}{202125}t^9+\cdots.
\]

In particular \(\chi'(0)=1\). This finite chart derivative does not
imply differentiability with respect to \(w\), because
\(d(t^3/3)/dt=0\) at the critical point.

## Exact vector model binding and global uniqueness

Let the original model have exactly two distinct centers, in their
stored order,

\[
a_-=c-a,\qquad a_+=c+a,
\qquad \alpha=\|a\|_2^2=2\nu s>0.
\]

Set \(\delta=b_+-b_-\) and
\(P_\perp=I-aa^T/\alpha\). The center diameter is \(2\|a\|\),
so the usual sufficient contraction parameter equals exactly one.
The actual velocity is

\[
U(x)=\frac{x-c-a\tanh(z+\delta/2)}s,
\qquad z=\frac{a\cdot(x-c)}\alpha.
\]

For an arbitrary real target \(v\), let

\[
w=\frac{s\,a\cdot v}{\alpha}+\frac\delta2.
\]

The exact global inverse is

\[
\boxed{\quad
X(v)=c+sP_\perp v
 +a\left[\psi(w)-\frac\delta2\right].
\quad}
\]

This follows by projecting the velocity equation onto \(a^\perp\)
and onto \(a\). It permits arbitrary exact original bias contrast;
a common bias shift cancels. The relation \(\alpha=2\nu s\) controls
critical curvature, while the bias contrast translates its location.

The spatial Jacobian is

\[
D_xU=\frac1s\left[
 P_\perp+\frac{aa^T}{\alpha}\tanh^2(z+\delta/2)
\right].
\]

It is positive semidefinite everywhere and singular on
\(z+\delta/2=0\). Nevertheless the Gaussian potential is strictly
convex: its transverse part is quadratic, while its one-dimensional
derivative along \(a\) is strictly increasing. The potential minus
\(v\cdot x\) is coercive and has the unique global minimizer displayed
above. It has no positive uniform strong-convexity modulus.

The numerical chart uses the exact target parameterization

\[
v(t,u)=\frac as\left(\frac{t^3}{3}-\frac\delta2\right)+u,
\qquad a\cdot u=0,
\]

with inverse

\[
X(t,u)=c+su+a\left[\chi(t)-\frac\delta2\right].
\]

The transverse target is unrestricted in magnitude, subject to resource
and finite enclosure limits. Only the critical scalar coordinate needs
the local chart.

## Removing the singular coefficient equation

Write \(\chi(t)=tz(t)\), with \(z(0)=1\), and define the entire
function with its removable value at zero,

\[
A(y)=\frac{3(y\cosh y-\sinh y)}{y^3}
 =\sum_{n\ge0}\frac{6(n+1)}{(2n+3)!}y^{2n}.
\]

The equation \(\chi-\tanh\chi=t^3/3\) becomes

\[
H(t,z)=z^3A(tz)-\cosh(tz)=0.
\]

There is no singular division in its power-series representation:

\[
H(t,z)=\sum_{n\ge0}
\left[\frac{6(n+1)z^{2n+3}}{(2n+3)!}
      -\frac{z^{2n}}{(2n)!}\right]t^{2n}.
\]

Here \(H(0,z)=z^3-1\), so \(H_z(0,1)=3\). This equation establishes
the normalized analytic branch used in the convergence proof below.
A polynomial differential identity gives a cheaper coefficient recurrence.
Differentiating \(\chi-\tanh\chi=t^3/3\) gives
\(\chi'\tanh^2\chi=t^2\). Set \(u=t^2\),
\(\chi=t z(u)\), and use \(\tanh\chi=t(z-u/3)\). Cancelling
\(t^2\) and extending at zero yields

\[
\boxed{\qquad (z+2uz')(z-u/3)^2=1.\qquad}
\]

The prime in this equation differentiates with respect to \(u\).
Write \(z=\sum_{n\ge0}z_nu^n\), \(w=z-u/3\) and \(p=w^2\),
with \(z_0=w_0=p_0=1\). At order \(n\ge1\), compute

\[
h_n=\sum_{k=1}^{n-1}w_kw_{n-k},\qquad
c_n=\sum_{k=1}^{n-1}(2k+1)z_kp_{n-k},
\]

then update

\[
\boxed{\quad
z_n=\frac{\frac23\mathbf1_{n=1}-h_n-c_n}{2n+3},\qquad
w_n=z_n-\frac13\mathbf1_{n=1},\qquad
p_n=2w_n+h_n.
\quad}
\]

Indeed the coefficient of \(u^n\) in the differential identity is
\((2n+1)z_n+p_n+c_n\); substituting \(p_n\) produces the
denominator \(2n+3\). Every update uses only earlier coefficients.
All arithmetic is rational and no singular Jacobian is solved. Even
coefficients of \(\chi\) are exactly zero. The next odd coefficient
after those displayed above is \(-362/9384375\) at order eleven.

There is no free integration constant. Conversely, let
\(y=\chi-t^3/3\). The differential identity gives
\(\chi' y^2=t^2\), hence \(y'=\chi'(1-y^2)\). Since
\(y(0)=\chi(0)=0\), the formal series
\(\operatorname{atanh}(y)-\chi\) has derivative zero and constant
zero. Thus \(y=\tanh\chi\), recovering the original equation.
Equivalently, the analytic branch proved from \(H\) satisfies this
recurrence, whose initial value \(z_0=1\) determines every coefficient
uniquely.

The pure exact factory `gaussian_critical_series(order, *, budget=None)`
in `omnibias.core.gaussian_critical` returns a frozen
`GaussianCriticalSeries`. Its flat `coefficients` tuple includes every
order through the requested `order`, and `z_coefficients` stores the
coefficients in \(u=t^2\). At order zero the latter is empty.
`.evaluate(t)` evaluates the finite polynomial exactly, without a tail.
This universal artifact contains no Gaussian model or fitted root. Exact
intermediate arithmetic, coefficient sizes, retained storage and work are
budgeted. The two convolution sums use \(O(K^2)\) rational arithmetic
operations and \(O(K+N)\) storage for requested \(\chi\) order \(N\).
This counts arithmetic operations, not their growing rational bit cost,
and is not a measured runtime claim.

## An explicit analytic disk and tail

Take

\[
R=\frac14,\qquad a_0=\frac18,\qquad
|t|\le R,\quad |z-1|\le a_0.
\]

Here \(a_0\) denotes the scalar shape radius, not a Gaussian center.
For \(T(t,z)=z-H(t,z)/3\), the following estimates prove contraction
on this complex disk and shape ball. Set
\(Z=1+a_0=9/8\) and \(Y=RZ=9/32\).

The ratio of the \(y^{2n}\) coefficient of \(A\) to that of
\(\cosh\) is \(3/[(2n+1)(2n+3)]\). For \(n\ge1\) it is at
most \(1/5\). Positivity of these real series coefficients therefore
gives the absolute-value majorants

\[
A(Y)-1\le\frac{\cosh Y-1}{5},\qquad
A'(Y)\le\frac{\sinh Y}{5}.
\]

Also, for \(0\le Y<1\),

\[
\cosh Y-1\le\frac{Y^2}{2(1-Y^2)},\qquad
\sinh Y\le\frac{Y}{1-Y^2}.
\]

Differentiating \(T\) and applying these bounds yields

\[
\begin{aligned}
|T_z|
&\le2a_0+a_0^2+
 \frac{Z^2Y^2}{10(1-Y^2)}
 +\left(\frac{RZ^3}{15}+\frac R3\right)
       \frac{Y}{1-Y^2}\\
&=\frac{17}{64}+\frac{6561}{603520}+\frac{429}{13120}
 =\frac{37321}{120704}<\frac13.
\end{aligned}
\]

At the center of the shape ball,

\[
|T(t,1)-1|
\le\frac{R^2}{6(1-R^2)}=\frac1{90}
 <\frac1{12}=(1-\tfrac13)a_0.
\]

The self-map margin using contraction bound \(1/3\) is \(13/180\).
Consequently contraction gives one analytic solution with
\(|z(t)-1|\le1/8\). Its uniqueness makes \(z\) even in \(t\).
The bound \(|tz|\le9/32\) also keeps \(\cosh(tz)\) nonzero,
so the multiplied equation genuinely solves the original inverse
equation. Real global uniqueness identifies this chart with \(\psi\).

Suppose the stored scalar polynomial retains \(\chi\) through order
\(N\ge1\), and put \(K=\lfloor(N-1)/2\rfloor\). Cauchy's bound
for the even series \(z-1\) gives

\[
\boxed{\quad
\left|\chi(t)-\sum_{n=0}^Nc_nt^n\right|
\le \frac\rho8\,
 \frac{(\rho/R)^{2K+2}}{1-(\rho/R)^2},
\qquad |t|\le\rho<R.
\quad}
\]

For \(N=0\), the retained polynomial is zero and the bound is
\(9\rho/8\). An even requested order adds an exact zero coefficient;
its remainder agrees with the preceding odd order. The scalar tail lifts
to coordinate error \(|a_i|\) times that tail, or Euclidean error
\(\|a\|_2\) times the tail, before outward numerical conversion.

## Certified model and target queries

`certify_gaussian_critical_chart(model, *, order=9)` in
`omnibias.core.verified.gaussian_critical` checks the exact two-center
condition and constructs a `GaussianCriticalChart`, or returns `None`.
It retains `model`, `series`, `midpoint`, `half_difference`,
`squared_half_distance`, `orthogonal_projector`, `bias_contrast`,
`parameter_radius`, `shape_radius`, `complex_contraction_bound`, and
`certificate`. The recorded contraction bound is the sharper rational
\(37321/120704\), whose upper bound \(1/3\) suffices for the proof.

The methods use exact integer or `Fraction` inputs, excluding booleans:

- `.target(t, transverse_target=None)` returns the exact vector
  \(v(t,u)\). The optional \(u\) defaults to zero and must be exactly
  orthogonal to the center difference.
- `.enclose(t, transverse_target=None)` returns an outward box for the
  unique global optimizer at that target.
- `.tail_bound(rho)` returns the universal scalar \(\chi\) error for
  \(0\le\rho<1/4\), before the vector lift.
- `.enclose_target(v, precision=80)` accepts a supplied exact target in
  the local critical range and encloses its unique optimizer directly.

The first two methods require \(|t|<1/4\). The direct-target method
computes the exact \(w\) above and requires
\(|3w|<(1/4)^3\). It brackets the real cube root \(t=\sqrt[3]{3w}\)
on an exactly scaled dyadic grid. Integer cube comparisons certify that
bracket, including for negative and very small nonzero inputs. Since
\(\chi\) is increasing, Taylor enclosures at the lower and upper
parameter endpoints enclose \(\chi(t)\). The exact transverse part
and signed axial vector then give the final box.

`precision` controls the relative dyadic grid resolution. If a parameter
bracket reaches the analytic boundary, the method refuses rather than
extrapolating; higher precision can resolve some such cases. It uses no
floating cube-root proposal or inverse Jacobian solve. Unsupported models
and exhausted construction budgets return `None`; invalid input raises.
Query domain errors, damaged evidence, resource exhaustion and nonfinite
endpoint conversion raise explicitly.

This example includes an oblique critical direction, nonzero bias
contrast and a translated center pair:

```python
from fractions import Fraction

from omnibias.core.gaussian_critical import gaussian_critical_series
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_critical import certify_gaussian_critical_chart

assert gaussian_critical_series(11).coefficients[11] == Fraction(-362, 9384375)
model = GaussianHeatMap(
    ((0, -2), (2, 0)), (0, Fraction(1, 3)), viscosity=1, time=1,
)
assert model.contraction_bound == 1
chart = certify_gaussian_critical_chart(model, order=9)
assert chart is not None
assert chart.midpoint == (1, -1)
assert chart.half_difference == (1, 1)
assert chart.series.coefficients[3] == Fraction(2, 15)
assert chart.series.coefficients[9] == Fraction(-16, 202125)

t = Fraction(1, 16)
transverse = (Fraction(1, 5), Fraction(-1, 5))
target = chart.target(t, transverse)
box = chart.enclose(t, transverse)
direct_box = chart.enclose_target(target)
assert chart.tail_bound(t) == Fraction(1, 125829120)
assert all(max(a.lo, b.lo) <= min(a.hi, b.hi)
           for a, b in zip(box, direct_box))

critical_point = (Fraction(31, 30), Fraction(-41, 30))
critical_box = chart.enclose(0, transverse)
assert all(Fraction(iv.lo) <= value <= Fraction(iv.hi)
           for iv, value in zip(critical_box, critical_point))

# A supplied rational target need not have a rational cube-root parameter.
base_target = chart.target(0, transverse)
nearby_target = tuple(value + Fraction(1, 10000) for value in base_target)
nearby_box = chart.enclose_target(nearby_target, precision=80)
assert len(nearby_box) == 2
```

Both query methods enclose the true unique optimizer, including the
Taylor remainder and endpoint conversion. They do not merely evaluate the
stored polynomial. They certify values; no finite bound on the ordinary
axial target derivative is implied at the critical target.

## Scope and established mathematics

The Gaussian centers, biases, viscosity and heat age stay fixed within a
chart. Polynomial joint bias/target paths can be studied through their
scalar effective target \(w\), but arbitrary such paths need not become
ordinary integer-power series in their original parameter. Generically
the inverse has cube-root powers.

This is a specialized analytic desingularization. Existence and
convergence of Puiseux expansions are classical; see
[Greenblatt, *Adapted coordinates in two dimensions and a proof of
Puiseux's theorem*](https://arxiv.org/abs/0807.2287). The repository's
existing algebraic Puiseux interface accepts a finite polynomial equation
and requires a caller-justified tail-ratio hypothesis. A truncated
polynomial replacement of \(\tanh\) would not by itself certify this
Gaussian family. The explicit analytic contraction above supplies that
missing remainder proof.

The ordinary inverse hodograph PDE applies where the original spatial
Jacobian is invertible. This fixed-age chart does not supply a classical
inverse PDE at the singular target, nor does it bound age derivatives
there. No general \(q\ge1\) mixture solver, automatic singularity
classification, formal theorem-prover verification or new general
calculus theorem is claimed.
