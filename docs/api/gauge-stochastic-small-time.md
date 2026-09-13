# SU(2) heat kernels at small time

This page proves the uniform image estimates used by the small-time evaluator.
The normalization is Haar mass one and the positive fundamental Casimir is
\(3/4\). The radial coordinate is \(u\in[0,\pi]\), so
\(\operatorname{Tr}U=2\cos u\). The group heat generator is
\(\Delta_G=\tfrac14(\partial_u^2+2\cot u\,\partial_u)\).
All statements below cover \(0<t\le5/64\), including both endpoints. They are
compact-group heat-kernel statements, not assertions about a Wilson continuum
measure or the Kogut–Susskind vacuum.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.stochastic.small_time import (
    replay_small_time_certificate,
    su2_small_time_heat_kernel,
    su2_small_time_score_bounds,
)

uniform = su2_small_time_score_bounds(Q(1, 64))
cut = su2_small_time_heat_kernel(Q(1, 4096), 1)
assert replay_small_time_certificate(uniform["certificate"])
assert replay_small_time_certificate(cut["certificate"])
assert cut["witness"]["arithmetic"]["radial_log_derivative_enclosure"] == ["0", "0"]
```

The point API takes an exact integer or rational `angle_pi` representing
\(u/\pi\). The theorem covers every real angle, while the evaluated inputs are
this rational subset. Times must be exact and in the stated range. The optional
`series_terms` is an integer in `16..128`. Input floats and booleans are refused.
All numerical enclosures use the directed transcendental backend; an exhausted
finite floating-point range is refused. The uniform score certificate uses
only exact rational arithmetic and therefore has no such numerical time floor.
Certificate replay reconstructs the complete source, arithmetic and scope.
The written heat-kernel implication has not been verified by Lean.

## Poisson formula and two regular charts

Poisson summation applied to the Gaussian theta series, followed by one spatial
derivative, gives

\[
K_t(u)=2\sqrt\pi\,e^{t/4}t^{-3/2}\frac{N_t(u)}{\sin u},
\qquad
N_t(u)=\sum_{\ell\in\mathbb Z}
 (u-2\pi\ell)e^{-(u-2\pi\ell)^2/t}.                 \tag{1}
\]

The identity follows as well by expanding
\(K_t=\sum_{n\ge0}(n+1)e^{-tn(n+2)/4}\sin((n+1)u)/\sin u\).
Every derivative of the Gaussian image series converges locally uniformly for
\(t>0\). Zeros in numerator and denominator at \(0,\pi\) are removable.

Write \(C(w)=\cosh\sqrt w\), \(S(w)=\sinh\sqrt w/\sqrt w\), with their entire
power series defining the values at zero.

For \(0\le u\le\pi/2\), put \(y=u^2\). Pair the images at \(\ell=\pm m\):

\[
\frac{N_t(u)}{u e^{-u^2/t}}=1+E_I(y,t),\quad
E_I=\sum_{m\ge1}2e^{-a_m^2/t}
 \left[C(w_m)-\frac{2a_m^2}{t}S(w_m)\right],
\quad a_m=2\pi m,\quad w_m=\frac{4a_m^2y}{t^2}.       \tag{2}
\]

For \(\pi/2\le u\le\pi\), put \(v=\pi-u\), \(y=v^2\), and
\(a_k=(2k+1)\pi\). Then

\[
N_t(u)=2v e^{-(\pi^2+v^2)/t}H_0(y,t)(1+E_C(y,t)),
\quad E_C=\sum_{k\ge1}\frac{H_k}{H_0},                \tag{3}
\]
\[
H_k=e^{-(a_k^2-\pi^2)/t}
 \left[\frac{2a_k^2}{t}S(w_k)-C(w_k)\right],
\qquad w_k=\frac{4a_k^2y}{t^2}.                      \tag{4}
\]

These formulas are even in the local coordinate and therefore cover the
removable endpoints without dividing an uncertain small number by another.

## Uniform image remainder, including derivatives

The following explicit bound is intentionally conservative:

\[
|E|,\ |\partial_yE|,\ |\partial_y^2E|,\
 |\partial_tE|<\varepsilon_*=2^{-99}                 \tag{5}
\]

in either chart; the time derivative holds at fixed physical \(u\), equivalently
fixed local \(v\). Here \(E=E_I\) or \(E_C\).

First, the positive series of \(C,S\) give
\(|C^{(r)}(w)|,|S^{(r)}(w)|\le e^{\sqrt w}\) for
\(w\ge0\) and \(r=0,1,2\). For instance the coefficient of \(w^j\) in
\(C^{(r)}\) is \((j+r)!/[j!(2j+2r)!]\le1/(2j)!\);
the \(S\) coefficient is smaller.

In (2) put \(z=a_m^2/t\). Since \(u\le\pi/2\),
\(-a_m^2/t+2a_mu/t\le-z/2\).
Also \(a_m^2\ge36\), \(4a_m^2/t^2\le z^2/9\), and \(z\ge400\).
For \(r=0,1,2\), each \(\partial_y^r\) summand is bounded by
\(z^5e^{-z/2}\). Direct differentiation at fixed \(u\) bounds its time
derivative by
\((7z+6z^2)t^{-1}e^{-z/2}\le10z^3e^{-z/2}\).
The inequalities

\[
400^5>4^{10}10!,\qquad 400^3>10\,4^6 6!
\]

and the corresponding positive exponential-series terms show that each of
these bounds is at most \(e^{-z/4}\).
Since \(z/4\ge100m\), their sum is below
\(\sum_{m\ge1}e^{-100m}<2^{-99}\).

For the cut chart set \(x=2\pi v/t\), \(\epsilon=t/(2\pi^2)\).
The inequalities \(x\coth x\le1+x\), \(x\le\pi^2/t\), and \(S(x^2)\ge1\)
give

\[
H_0=\left[\frac{2\pi^2}{t}-x\coth x\right]S(x^2)
 \ge\left(\frac{\pi^2}{t}-1\right)S(x^2)\ge1.          \tag{6}
\]

For \(k\ge1\), let \(z=a_k^2/t\). Now
\(a_k^2-a_k\pi-\pi^2\ge a_k^2/2\), \(a_k^2\ge81\), and \(z\ge1000\).
Consequently
\[
|H_k^{(r)}|\le3z(z^2/20)^r e^{-z/2}\quad(r=0,1,2).
\]
Here superscripts mean \(y\)-derivatives. Since
\(e^x/S(x^2)\le1+2x\), equation (6) yields
\[
|H_0'|/H_0\le z^3/100,\qquad
|H_0''|/H_0\le z^5/10000.                             \tag{7}
\]
In this estimate \(4\pi^2/t^2\le z^2/100\) and
\(3(1+2\pi^2/t)\le z\), which hold for every omitted \(k\).
The quotient rule now bounds each \(H_k/H_0\) and its first two \(y\)-derivatives
by \(z^7e^{-z/2}\). For example the second derivative is bounded by
\[
3z\left[z^4/400+z^5/1000+2z^6/10000+z^5/10000\right]e^{-z/2}
 \le z^7e^{-z/2}.
\]
The inequality \(1000^7>4^{14}14!\) implies \(z^7\le e^{z/4}\).
Moreover \(z/4\ge100k\), so the three series again sum to less than \(2^{-99}\).

For completeness the time derivative is controlled without differentiating a
possibly vanishing principal numerator. Define

\[
R=\frac{\pi\cosh x-v\sinh x}{\pi\sinh x-v\cosh x}.
\]

For \(x>0\),
\[
x(R-1)=\frac{2x(1+\epsilon x)}
 {(1-\epsilon x)e^{2x}-(1+\epsilon x)},\qquad
0\le x(R-1)\le\frac32.                               \tag{8}
\]
To prove the upper bound, if \(x\le1\), use \(\cosh1<2\),
\(\sinh x\ge x\), and \(\epsilon<1/8\); the ratio is at most
\((1+\epsilon)/(1-2\epsilon)\le3/2\).
If \(x\ge1\), use \(\epsilon x=v/\pi\le1/2\) and
\(e^{2x}\ge4x+3\). The latter follows from \(e^2>7\) and monotonicity
of \(e^{2x}-4x-3\) on \([1,\infty)\).
The \(x=0\) limit is \(1/(1-\epsilon)\).

It follows that
\[
|\partial_tH_0/H_0|\le\pi^2/t^2+3/(2t).
\]
Direct differentiation of (4) gives
\[
|\partial_tH_k|\le4(z^2+z)t^{-1}e^{-z/2}.
\]
Combining these with \(|H_k|\le3ze^{-z/2}\), \(H_0\ge1\),
\(\pi^2/t\le z/9\), and \(1/t\le z/81\), bounds each
\(\partial_t(H_k/H_0)\) by \(10z^3e^{-z/2}\).
The same exponential majorant and geometric sum finish (5).

All bounds are uniform on the closed chart intervals. Uniform convergence of
the differentiated image series justifies the endpoint limits and derivatives.

## Uniform time-score theorem

Taking the logarithm of (2) gives the principal time score
\(1/4-3/(2t)+u^2/t^2\). Taking the logarithm of (3) gives

\[
\partial_t\log K_t^{\rm principal}(u)
 =\frac14-\frac{3}{2t}+\frac{u^2}{t^2}
   -\frac{x(R-1)}t.                                  \tag{9}
\]

By (5),
\(|\partial_t\log(1+E)|<2^{-98}<1/4\).
Equations (8)–(9) therefore prove, for every \(u\in[0,\pi]\),

\[
\boxed{\frac{u^2}{t^2}-\frac3t
 \ \le\ \partial_t\log K_t(u)\
 \le\ \frac{u^2}{t^2}-\frac{3}{2t}+\frac12}
 \quad(0<t\le5/64).                                  \tag{10}
\]

This is a uniform analytic estimate, not a finite collection of numerical
samples. At the cut locus \(u=\pi\), the time score and log-Laplacian have leading
order \(\pi^2/t^2\); they cannot be bounded globally by a constant times \(1/t\).

## Stable logarithms, scores and group log-Laplacians

Let \(v=u\) in the identity chart and \(v=\pi-u\) in the cut chart.
Write \(y=v^2\), \(r(y)=\sin v/v\). If the principal log kernel is expressed as a
function \(L(y)\), then

\[
\partial_uL=\pm2vL_y,\qquad
\Delta_GL=\left(\frac12+v\cot v\right)L_y+yL_{yy},
\qquad v\cot v=\frac{\cos v}{r(y)}.                    \tag{11}
\]

The sign is positive in the identity chart and negative in the cut chart.
Equation (11) remains valid at \(v=0\), where \(r(0)=1\), \(v\cot v=1\).
Thus both endpoint radial scores vanish exactly. In the cut layer, evaluating
\(S(w)-[t/(2\pi^2)]C(w)\), rather than subtracting two tiny Gaussian images,
avoids loss of the removable zero. Outside that layer, the factored expression
\(u-(2\pi-u)e^{-4\pi(\pi-u)/t}\) avoids exponentially large hyperbolic functions.

From (5), the errors of the principal logarithm, radial log derivative and group
log-Laplacian are respectively bounded by \(2^{-98}\), \(2^{-96}\), and
\(2^{-95}\). Indeed the first two \(y\)-derivatives of \(\log(1+E)\) are bounded
by \(2\varepsilon_*,3\varepsilon_*\), while
\(0\le v\cot v\le1\) and \(v^2<5/2\).

The evaluator returns the logarithm and the rescaled positive value

\[
\mathcal K_t(u)=\frac{t^{3/2}}{2\sqrt\pi}
 e^{u^2/t-t/4}K_t(u),                                \tag{12}
\]

rather than requiring the ordinary kernel to be representable as a floating
point number. Keeping the large negative action in the logarithm avoids
underflow at a nonidentity holonomy as \(t\) becomes small.

For completeness, the finite evaluation of \(C,S,r\) also carries an analytic
tail. For derivative order \(j\in\{0,1,2\}\), their absolute series terms have
coefficients
\[
a_m=\frac{(m)_j W^{m-j}}{(2m+o)!},\qquad o\in\{0,1\},
\]
where \(W\) is an outward upper bound on the nonnegative argument and
\((m)_j=m!/(m-j)!\). If terms through \(m=M\) are retained, then
\[
\frac{a_{m+1}}{a_m}
=\frac{m+1}{m+1-j}
 \frac{W}{(2m+o+1)(2m+o+2)}
\]
decreases with \(m\). Its value \(q\) at \(m=M+1\) gives the omitted-tail
bound \(a_{M+1}/(1-q)\) whenever \(q<1\). The alternating sinc series is
enclosed using this absolute tail. Polynomial evaluation, coefficient
conversion and the tail arithmetic are all outward rounded. This second tail
controls finite evaluation of the principal images; (5) independently controls
every omitted Gaussian image.

## A direct bridge consequence

For \(0<t\le1/64\), let \(X\) have the normalized bridge law
\(K_t(X)K_{4t}(X^{-1}B)/K_{5t}(B)\,dX\).
Differentiating the first convolution time with the second fixed at \(4t\)
and using the heat equation gives
\[
\mathbb E[\partial_t\log K_t(X)]
 =\left.\partial_s\log K_s(B)\right|_{s=5t}.
\]
The times on both sides satisfy (10). Thus
\[
\mathbb E[u_X^2]\le u_B^2/25+27t/10+t^2/2.             \tag{13}
\]
Each of the five conditioned increments has the same class-angle marginal.
Since \(2-\operatorname{Tr}U\le u_U^2\) and
\(u_B^2\le(\pi^2/4)(2-\operatorname{Tr}B)\), equation (13) yields an actual
expected-action contraction with coefficient \(\pi^2/20\) and an additive
\(27t/2+5t^2/2\) term for the sum of five actions. This concerns the specified
compact heat bridge. It does not identify an arbitrary exterior Wilson or
quantum-vacuum conditional law with that bridge.
