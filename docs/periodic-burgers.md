# Prescribed periodic Burgers data with rigorous enclosures

A finite Fourier representation can enclose the viscous Burgers solution for
**prescribed initial velocity**, including velocities with both positive and
negative spatial derivatives. The construction integrates the initial velocity
to a Cole–Hopf potential, approximates its exponential with an exact finite
polynomial, evolves the Fourier modes by the heat equation, and bounds the
discarded exponential terms before taking the logarithmic derivative.

This extends the [positive real exponential family](nonlinear-pde-identities.md)
to a different initial-data class. It uses classical Cole–Hopf and Fourier heat
evolution. The implementation does not establish a new linearization theorem,
universal efficiency, or general solvability of nonlinear PDE systems.

## The initial-value problem

Use period \(2\pi\), viscosity \(\nu>0\), and prescribed scalar initial data

\[
u_0(x)=c+\sum_{n\ge1}\bigl(a_n\cos(nx)+b_n\sin(nx)\bigr),
\]

with finitely many nonzero coefficients. A periodic potential with zero
constant coefficient is

\[
g(x)=\sum_{n\ge1}\left(
-\frac{a_n}{2\nu n}\sin(nx)
+\frac{b_n}{2\nu n}\cos(nx)\right),
\qquad -2\nu g'(x)=u_0(x)-c.
\]

Let \(\phi\) be the periodic heat solution with initial data
\(\phi(x,0)=e^{g(x)}\). Then

\[
u(x,t)=c-2\nu\frac{\phi_x(x-ct,t)}{\phi(x-ct,t)}
\]

solves \(u_t+uu_x=\nu u_{xx}\) and satisfies the prescribed initial condition.
The shift \(x-ct\) matters when the mean \(c\) is nonzero. Omitting it would
solve the wrong transport equation.

In several dimensions, the supported initial data are prescribed **potential
fields**

\[
U_0(x)=c-2\nu\nabla g(x),\qquad x\in\mathbb T^d,
\]

where \(g\) is a real trigonometric polynomial and \(c\) is a constant vector.
The same construction gives
\(U(x,t)=c-2\nu\nabla\phi(x-ct,t)/\phi(x-ct,t)\), satisfying

\[
\partial_tU_i+\sum_jU_j\partial_jU_i
-\nu\sum_j\partial_{jj}U_i=0.
\]

The identity follows by differentiating
\(\phi_t-\nu\Delta\phi=0\) through the Cole–Hopf quotient. Mixed partials
commute, so the fields remain irrotational. General vector data with nonzero
curl are outside this construction. Prescribing the potential still permits
genuine spatial coupling: a term such as \(\cos(x_1+x_2)\) contributes to both
velocity components and their cross derivatives.

For example, \(u_0(x)=\sin x\) has \(u'_0(0)=1\) and
\(u'_0(\pi)=-1\). Such sign variation was excluded by the negative
semidefinite Jacobian of the earlier positive real exponential family. Here
the heat datum is positive because it is an exponential of a real periodic
potential; its Fourier coefficients need not be positive.

## Exact finite heat evolution and a global remainder

Write the potential as

\[
g(x)=\sum_{k\in\mathbb Z^d}\gamma_k e^{ik\cdot x},
\qquad \gamma_{-k}=\overline{\gamma_k},
\]

with finite support. Rational sine and cosine data give rational real and
imaginary parts of every coefficient. A real constant in \(g\) is removed as a
gauge: multiplying the entire heat solution by its positive exponential does
not change any velocity or spatial velocity derivative.

The compiler forms

\[
Q_M(g)=\sum_{j=0}^{M}\frac{g^j}{j!}
=\sum_k q_k e^{ik\cdot x}.
\]

All Fourier coefficient arithmetic in this finite polynomial is exact. Every
product frequency is retained; there is no hidden frequency cutoff or aliasing.
Its heat evolution is the finite expression

\[
\psi_M(x,t)=\sum_kq_k e^{-\nu\lVert k\rVert_2^2t}e^{ik\cdot x}.
\]

It satisfies the heat equation exactly. The difference from \(\phi\) comes
from the exponential terms of degree greater than \(M\), not from a
time-stepping scheme. Numerical evaluation of the finite expression still
requires outward interval arithmetic.

An exact symbolic check of this finite heat expression, or of its Cole–Hopf
residual on a nonzero-denominator domain, checks the finite model only. The
global remainder and positivity argument below are what connect numerical
enclosures to the solution for the prescribed initial velocity.

The exponential remainder is **global**: it can affect retained low Fourier
modes as well as high modes. For example, powers of \(\cos x\) repeatedly
produce a constant coefficient. It is therefore carried separately from the
finite coefficients, rather than declared to be a high-frequency-only tail.

The general `ValidatedFourierSeries` algebra makes the same distinction:
`tail` bounds weighted Fourier error outside its truncation cube, while
`global_remainder` bounds error anywhere. Multiplication can move exterior
uncertainty into low modes, since frequencies \(k\) and \(-k\) produce a
constant. The raw `coeffs` dictionary is the finite representative; `get(k)`
encloses the complete coefficient. A multiplier acting on global uncertainty
requires `global_factor`, a bound over **all** modes, in addition to its
exterior `tail_factor`. The bounded Hilbert, Riesz and Leray helpers supply
that bound automatically.

```python
from omnibias.core.verified.fourier import ValidatedFourierSeries
from omnibias.core.verified.complex_interval import ComplexInterval

# Each input could have unit mass at opposite exterior frequencies.
outside = ValidatedFourierSeries.from_coeffs({}, dim=1, trunc=1, nu=1.0, tail=1.0)
product = outside * outside
assert product.global_remainder.hi >= 1.0
assert product.get((0,)).re.contains(1.0)

# Projection onto the constant mode has exterior bound 0 but global bound 1.
constant_mode = product.apply_multiplier(
    lambda k: ComplexInterval.one() if k == (0,) else ComplexInterval.zero(),
    0.0, global_factor=1.0,
)
assert constant_mode.get((0,)).re.contains(1.0)
```

## A derivative bound for every retained order

Let

\[
A=\sum_k\bigl(|\operatorname{Re}\gamma_k|
              +|\operatorname{Im}\gamma_k|\bigr),
\qquad L=\max_{\gamma_k\ne0}\lVert k\rVert_1.
\]

The zero potential has \(A=L=0\). This \(A\) is an exact upper bound on the
Fourier \(\ell^1\) norm and on \(\lVert g\rVert_\infty\). For a
multi-index \(\alpha\), put \(r=|\alpha|\). Each frequency in \(g^j\)
has \(\ell^1\) norm at most \(jL\). Multiplication is submultiplicative in
the Fourier \(\ell^1\) norm, hence

\[
\lVert D^\alpha g^j\rVert_\infty
\le\lVert D^\alpha g^j\rVert_{\mathcal A}
\le(jL)^r A^j,
\qquad
\lVert f\rVert_{\mathcal A}=\sum_k|\widehat f_k|.
\]

To bound the infinite sum, use the Stirling numbers of the second kind:

\[
j^r=\sum_{\ell=0}^{r}
\left\{\!\begin{matrix}r\\\ell\end{matrix}\!\right\}
\frac{j!}{(j-\ell)!},
\]

where the summand vanishes for \(\ell>j\). Reindexing the exponential tail
gives, with \(q_\ell=\max(M+1-\ell,0)\),

\[
\begin{aligned}
L^r\sum_{j=M+1}^{\infty}\frac{j^r A^j}{j!}
&=L^r\sum_{\ell=0}^{r}
\left\{\!\begin{matrix}r\\\ell\end{matrix}\!\right\}
A^\ell\sum_{n=q_\ell}^{\infty}\frac{A^n}{n!}\\
&\le L^r e^A\sum_{\ell=0}^{r}
\left\{\!\begin{matrix}r\\\ell\end{matrix}\!\right\}
\frac{A^{\ell+q_\ell}}{q_\ell!}
=:R_r.
\end{aligned}
\]

The last inequality follows from \((q+n)!\ge q!n!\). For \(r=0\), use
the convention \(\left\{\!\begin{smallmatrix}0\\0\end{smallmatrix}\!\right\}=1\)
and \(L^0=1\), obtaining
\(R_0=e^A A^{M+1}/(M+1)!\). If \(A=0\), the remainder is exactly zero
at every derivative order.

For \(t\ge0\), every heat multiplier has magnitude at most one, so the same
bound applies after evolution:

\[
\sup_x|D^\alpha\phi(x,t)-D^\alpha\psi_M(x,t)|\le R_{|\alpha|}.
\]

The estimate is uniform in space and nonnegative time. In particular, it is
valid at the initial time. One must compute the derivative remainder of the
required order; differentiating a bare value-error interval does not produce
a valid derivative-error interval.

For fixed \(r\) and fixed potential, \(R_r\to0\) as \(M\to\infty\).
Consequently the finite heat expressions converge to the prescribed heat
solution in every fixed spatial derivative order, uniformly for \(t\ge0\).
This is a mathematical truncation statement. Fixed-precision interval widths,
overflow and finite resource budgets can prevent arbitrarily fine numerical
enclosures.

## Positivity makes the quotient rigorous

Initially \(\phi(x,0)=e^{g(x)}\ge e^{-A}\). The periodic heat kernel is
positive and has unit integral, so convolution preserves that lower bound:

\[
\phi(x,t)\ge m:=e^{-A}>0\qquad(t\ge0).
\]

The evaluator combines a finite-expression interval with the appropriate
\([-R_r,R_r]\) error. For the zero-order derivative it can intersect this
interval with \([m,\infty)\) before dividing. This bounds the **true** heat
solution. It does not require the finite approximation \(\psi_M\) to have
been proved positive. If a positive representable denominator lower bound
cannot be established, the evaluator must refuse the quotient.

All higher velocity derivatives follow from the same enclosed heat derivatives.
Set \(q_i=\phi_i/\phi\). Differentiating \(\phi q_i=\phi_i\) gives the
raw derivative recurrence used by the implementation:

\[
D^\alpha q_i=\frac1\phi\left(D^{\alpha+e_i}\phi-
\sum_{0<\beta\le\alpha}\binom{\alpha}{\beta}
(D^\beta\phi)(D^{\alpha-\beta}q_i)\right).
\]

The velocity derivatives are then

\[
D^\alpha U_i
=\mathbf1_{\alpha=0}c_i
-2\nu D^\alpha q_i,
\]

evaluated at \((x-ct,t)\). A velocity jet through total order \(r\) therefore
requires heat derivatives through order \(r+1\). Returned values are raw
partial derivatives, with the displayed binomial factors retained.

The positive lower bound and convergence of the required heat derivatives also
prove convergence of every fixed velocity derivative order. For sufficiently
large \(M\), \(R_0<m\), so \(\psi_M\) is itself globally positive and its
Cole–Hopf quotient is an exact Burgers solution with nearby initial data. The
validated output instead encloses the solution with the original prescribed
data at every accepted degree, including degrees for which this additional
positivity test on \(\psi_M\) would fail.

## Relation to bias shifts and inverse decisions

For each Fourier mode, a normalized bias-shift stencil has the exact limit

\[
\lim_{h\to0}\frac{1}{h^r}
\sum_{j=0}^r(-1)^{r-j}\binom rj
e^{ik\cdot(x+jh e_\ell)}
=(ik_\ell)^r e^{ik\cdot x}.
\]

The finite representation therefore computes collapsed derivative operators
by multiplying coefficients directly. Its error bound separately controls the
omitted modes of the prescribed-data solution. Neither an unnormalized
difference nor arbitrary activation substitution has this contract.

The [global inverse theorem](inverse-pde-map.md) for positive sums of real
exponentials depends on their positive covariance Hessian. Periodic Fourier
fields do not satisfy that hypothesis. This extension supplies a forward
initial-value solution enclosure; it does not inherit global uniqueness of
velocity preimages or convexity of an inverse objective.

## Exact plan and interval evaluation

`PeriodicHeatPlan` is imported from `omnibias.core.periodic_heat`. Its exact
scalar constructor is
`PeriodicHeatPlan.from_velocity(cosine, sine=None, mean=0, viscosity=1, degree=16, budget=None)`.
The cosine and sine mappings use strictly positive integer frequencies and
exact integer or `Fraction` amplitudes. The compiler integrates the zero-mean
part to \(g\); the `mean` becomes the Galilean drift.

For a prescribed potential in several dimensions, use
`PeriodicHeatPlan.from_log_potential(dimension, cosine, sine=None, viscosity=1, degree=16, drift=None, budget=None)`.
Here frequencies are integer tuples of the declared dimension, and the drift
is a constant vector. The potential is real by its sine/cosine construction.
Removing its constant gauge preserves the requested velocity, but changes the
normalization of the represented heat datum.

The frozen plan exposes `potential_coefficients` and `heat_coefficients` as
triples `(frequency, real_part, imaginary_part)`, together with `norm_bound`
and `max_frequency`. Coefficients are exact rational data. The optional
`ExponentialBudget` limits exact storage and work; `max_total_degree` bounds
the exponential Taylor degree, while `max_derivative_order` applies to the
requested evaluator order.

The interval evaluators are imported from
`omnibias.core.verified.periodic_heat` and require the rigorous transcendental
backend from `omnibias-core[verified]`. Spatial inputs and time are finite
`Interval` objects; a scalar coordinate is wrapped with `Interval.point`.
The entire time interval must be nonnegative. Endpoints are binary64 floats or
integers exactly representable in that format; silently rounding an exact
endpoint could change the requested domain and is refused.

```python
from fractions import Fraction

from omnibias.core.periodic_heat import PeriodicHeatPlan
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.periodic_heat import enclose_burgers_jet

# The prescribed initial velocity is exactly sin(x).
plan = PeriodicHeatPlan.from_velocity(
    {}, {1: 1}, viscosity=Fraction(1, 2), degree=14,
)
jet = enclose_burgers_jet(
    plan, (Interval.point(0.5),), Interval.point(0.5), order=2,
)
print(jet.value, jet.derivative((1,)), jet.derivative((2,)))
assert jet.heat_lower_bound > 0.0

initial = enclose_burgers_jet(
    plan, (Interval.point(0.0),), Interval.point(0.0), order=1,
)
assert initial.value[0].contains(0.0)
assert initial.derivative((1,))[0].contains(1.0)
assert initial.derivative((1,))[0].lo > 0.0
other = enclose_burgers_jet(
    plan, (Interval.point(3.0),), Interval.point(0.0), order=1,
)
assert other.derivative((1,))[0].hi < 0.0
```

`enclose_burgers_jet(plan, point, time, order=0)` evaluates all spatial
multi-indices through the requested total order. Its `BurgersJet` result has
`multi_indices` and aligned `derivatives`; each derivative is a vector of
intervals, one per velocity component. `derivative(alpha)` selects a raw
derivative vector, and `value` selects the zero-order vector. The reported
`heat_lower_bound` is the positive denominator bound used in the quotient.
The evaluator applies the Galilean shift from the plan's drift internally.

`enclose_heat_derivative(plan, point, time, alpha=None)` instead evaluates the
**stationary heat coordinates**, without the drift shift. Its `HeatEnclosure`
contains `finite_part`, a nonnegative `remainder_bound`, and `value` enclosing
the true heat derivative. Omitted `alpha` selects the heat value. Applying the
drift shift twice would evaluate the wrong physical point.

The representation implements derivative extraction directly:
\(D^\alpha e^{ik\cdot x}=(ik)^\alpha e^{ik\cdot x}\). This is the analytic
derivative targeted by properly normalized bias differences. The computation
does not subtract nearly coincident forward values or assume arbitrary
activation substitutions preserve the heat equation.

For a coupled potential, the mixed velocity derivatives can be enclosed by the
same evaluator:

```python
coupled = PeriodicHeatPlan.from_log_potential(
    2,
    {(1, 0): 1, (1, 1): Fraction(1, 2)},
    {(0, 1): Fraction(1, 4)},
    viscosity=Fraction(1, 2), degree=14,
)
at_origin = enclose_burgers_jet(
    coupled, (Interval.point(0.0), Interval.point(0.0)),
    Interval.point(0.0), order=1,
)
assert at_origin.value[0].contains(0.0)
assert at_origin.value[1].contains(-0.25)
assert at_origin.derivative((0, 1))[0].contains(0.5)
assert at_origin.derivative((1, 0))[1].contains(0.5)
assert at_origin.derivative((0, 1))[0].lo > 0.0
assert at_origin.derivative((1, 0))[1].lo > 0.0
```

Here \(g=\cos x+\tfrac12\cos(x+y)+\tfrac14\sin y\), so the initial field
has two nonzero cross derivatives at the origin. The equal mixed derivatives
are consistent with the potential-field restriction, while the mixed frequency
prevents interpreting the field as two independent scalar coordinate problems.

## Scope, numerical limits and prior work

The scalar initial-data class contains every real trigonometric polynomial,
including a nonzero mean. In multiple dimensions it contains periodic potential
velocity fields plus constant drift. No boundary fitting is needed because
periodicity is built into the modes. Other domains, boundary conditions,
forcing, general vortical vector data, and initial data known only at samples
are not handled by this constructor. A sampled residual or an empirical error
estimate would not establish the missing initial-data guarantee.

The Taylor support and rational coefficients grow with the degree and spatial
dimension. Small viscosity increases the initial potential and can make the
positive lower bound tiny, the truncation bounds expensive, and the quotient
poorly conditioned. Interval evaluation over wide boxes can add substantial
dependency error. Increasing the degree does not guarantee a narrower returned
interval at fixed numerical precision. All finite-expression rounding must
remain in addition to the analytic remainder.

The nonlinear identity is the classical
[Cole–Hopf transformation](https://doi.org/10.1090/qam/42889).
The positivity and Fourier-multiplier arguments are standard heat-semigroup
properties; see [Walter Craig's heat-equation chapter](https://ms.mcmaster.ca/craig/craig-pde-chap3.pdf).
Fourier coefficients could alternatively be initialized by validated periodic
quadrature. [Trefethen and Weideman, *The Exponentially Convergent Trapezoidal
Rule*](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2014_149.pdf),
Theorem 3.2, supplies an explicit strip-analytic quadrature bound. The present
finite-polynomial initializer avoids that quadrature step and uses direct
exponential-series remainders. These connections do not establish mathematical
novelty. The capability here is a checked path from specified initial data to
enclosures of the nonlinear solution and its derivatives, under the stated
assumptions; no Lean-kernel proof or universal efficiency claim is made.
