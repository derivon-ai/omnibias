# Confluent power and signed-root calculus

The functions in `omnibias.core.verified.asymptotic_jet` provide concrete
enclosures for two confluent scalar primitives and exact polynomial
derivatives along a fixed-product scale path. Their analytic statements and
proofs follow. They do not identify an actual singular passage with either
primitive, prove a uniform passage remainder, or complete graphic coverage.

## 1. Power compensator through resonance

For real a,b and x>0, put L=log x and

\[
C(a,b;x)=\int_0^1 L e^{(b+t(a-b))L}\,dt.
\]

Integrating the exponential gives (x^a-x^b)/(a-b) when a differs from b.
At a=b the same expression is x^a log x. The integral is entire in a,b,L;
on each compact parameter box all its derivatives are bounded continuously,
so differentiation under this finite integral is justified. For nonnegative
integers p,q, let m=p+q+1. Then

\[
\partial_a^p\partial_b^q C
=L^m\int_0^1 t^p(1-t)^q e^{c(t)L}\,dt,
\qquad c(t)=b+t(a-b).
\]

Since x partial_x=partial_L, its k-th logarithmic derivative is

\[
\sum_{j=0}^{\min(k,m)}\binom{k}{j}\frac{m!}{(m-j)!}L^{m-j}
\int_0^1 t^p(1-t)^q c(t)^{k-j}e^{c(t)L}\,dt. \tag{1}
\]

This is the unnormalized derivative returned by `power_compensator` with
`a_order=p`, `b_order=q`, `log_order=k`. On the exact diagonal the integral
in (1) is `B(p+1,q+1)*a**(k-j)*exp(a*L)`. In particular partial_a C at a=b
is half the derivative of the diagonal function with respect to a: keeping
b fixed matters even at resonance.

For stable evaluation write d=a-b and z=dL. Expand c(t)^ell by the binomial
theorem and use the entire moments

\[
M_{p,q}(z)=\int_0^1t^p(1-t)^q e^{zt}\,dt
=\sum_{n=0}^N\frac{z^n}{n!}\frac{(p+n)!q!}{(p+q+n+1)!}+T_N.
\]

Taylor's exponential remainder gives, uniformly for |z|<=r,

\[
|T_N|\le \frac{p!q!}{(p+q+1)!}
             e^r\frac{r^{N+1}}{(N+1)!}. \tag{2}
\]

Indeed the pointwise exponential remainder is at most
`exp(r)*r**(N+1)*t**(N+1)/(N+1)!`; multiplication by the nonnegative beta
weight and t^(N+1)<=1 proves (2). The implementation uses this bound for
r<=1/2. Elsewhere it partitions [0,1], encloses the whole integrand on
each closed subinterval, and sums interval range times subinterval length.
This is a range proof over a full cover, not sampled quadrature. Interval
inputs crossing a=b need no division by their difference. Underflow remains
enclosed by directed elementary functions and interval operations; nonfinite
intermediate exponent bounds or final endpoints raise explicitly.

## 2. Signed-root primitive through delta=0

Define

\[
S(\delta,v)=\int_0^v\frac{dt}{1+\delta t^2}
=v\int_0^1\frac{dt}{1+\delta v^2t^2}.
\]

For positive delta this equals `atan(sqrt(delta)*v)/sqrt(delta)`; for
negative delta it equals `atanh(sqrt(-delta)*v)/sqrt(-delta)` before the
real pole; at delta=0 it equals v. The interval domain must guarantee a
strictly positive denominator on the entire integration segment. A box
whose denominator might vanish is refused, even if its midpoint is safe.

Differentiation on a pole-free compact gives

\[
\partial_\delta^p S=(-1)^p p!v^{2p+1}
\int_0^1\frac{t^{2p}}{(1+\delta v^2t^2)^{p+1}}\,dt, \tag{3}
\]

and the fundamental theorem of calculus gives

\[
\partial_v\partial_\delta^p S
=\frac{(-1)^p p!v^{2p}}{(1+\delta v^2)^{p+1}}. \tag{4}
\]

The API returns (3) or (4), including negative v. On delta=0, (3) is the
exact polynomial `(-1)**p*p!*v**(2*p+1)/(2*p+1)`. For y=delta*v^2,
the integral in (3) has expansion

\[
\sum_{n=0}^N \frac{\binom{n+p}{p}}{2n+2p+1}(-y)^n+T_N.
\]

For |y|<=r<=1/4, the absolute omitted series has first term

\[
A=\frac{\binom{N+1+p}{p}}{2(N+1)+2p+1}r^{N+1}.
\]

The ratio of subsequent terms is at most
`q=r*(N+p+2)/(N+2)`: the extra denominator ratio is less than one,
and `(n+p+1)/(n+1)` decreases with n. The implementation takes N>=p+1,
so q<1 and |T_N|<=A/(1-q). Outside this central domain it uses the same
complete interval integration cover as above. This proves parameter-uniform
confluence; it does not assert an asymptotic formula near the excluded pole.

## 3. Fixed epsilon and path acceleration

For positive omega,u define omega(s)=omega exp(-s), u(s)=u exp(s).
Their product epsilon=omega*u is fixed for every s. The generator is
E=u partial_u-omega partial_omega and

\[
E^2R=\omega^2R_{\omega\omega}-2\omega uR_{\omega u}
       +u^2R_{uu}+\omega R_\omega+uR_u. \tag{5}
\]

The last two terms arise from the acceleration of this path. In particular
E^2(omega*u)=0; the straight directional Hessian alone returns
-2*omega*u and is not the fixed-epsilon second derivative.

A monomial omega^i u^j is an eigenfunction of E with eigenvalue j-i.
Consequently `weighted_scale_derivative` multiplies every exact rational
coefficient by `(j-i)**order`. Induction proves the result at arbitrary
finite order, within the polynomial's declared arithmetic budgets. Other
named polynomial axes are fixed. `fixed_product_jet` evaluates these exact
polynomials by outward interval arithmetic. `verify_fixed_product_derivative`
recomputes their coefficient identity and rejects the missing-acceleration
polynomial; it accepts no externally supplied correctness flag.

## 4. Corrected singularity-radius contract

`omnibias.difference.singularity.convergence_radius_from_geometric_tail`
assumes the caller has proved |a_k|<=M q^k for every k beyond the supplied
finite prefix, with finite M>=0 and q>0. Cauchy-Hadamard yields only
R>=1/q. The function returns an extended interval with infinite upper
endpoint; for M=0 it returns [infinity,infinity]. The legacy
`certified_singularity_annulus` forwards to this corrected API.

The entire polynomial 1+z, with prefix (1,1) and zero tail, disproves the
old inference of a finite upper radius from the maximum of prefix
coefficient roots. No prefix establishes a positive lower asymptotic
limsup. A finite upper radius requires a proved infinite lower-tail or
noncancellation argument, which this interface deliberately does not infer.

## 5. Evidence and limits

`packages/omnibias-core/tests/verified/test_asymptotic_jet.py` checks the
enclosures against independent high-precision divided differences,
atan/atanh formulas and their derivatives; deterministic grids and random
points in interval boxes include diagonal crossings, tiny opposite signs,
negative endpoints and underflow. It also tests the explicit remainder,
pole refusal, fixed-product identity and wrong-Hessian rejection.

`formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Scale.lean`
proves the actual exponential divided-difference diagonal limit, the exact
fixed-product identity, derivatives of both exponential scale paths, the
second weighted derivative under explicit actual-partial hypotheses, the
product acceleration cancellation and the negative-delta pole margin.
These targeted theorems do not formalize the Python interval evaluator,
the beta/binomial remainder proofs above, or a physical return-map theorem.

Reproduce the focused checks:

```bash
uv run --no-sync python -m pytest packages/omnibias-core/tests/verified/test_asymptotic_jet.py packages/omnibias-difference/tests/test_singularity_tracking.py -q
```

From `formal/omnibias-analytic`:

```bash
lake build OmnibiasAnalytic.Dynamics.Hilbert16Scale
```

An independent adversarial implementation review rederived the beta factors,
logarithmic derivatives, signed-root tail ratio and fixed-product generator;
46 additional 90-digit quadrature probes, including orders 8 and 16, passed.
The eight named Lean theorems built, and their axiom audits reported only
`propext`, `Classical.choice` and `Quot.sound`. Passing these checks does not
replace the missing uniform physical remainder, singular-chart closure or
complete nearby-cycle capture obligations.
