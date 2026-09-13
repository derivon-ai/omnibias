# Stochastic compact-group heat-kernel API

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.stochastic import (
    replay_heat_kernel_certificate,
    su2_heat_kernel_bridge,
    su2_heat_kernel_enclosure,
    su2_wilson_heat_kernel_comparison,
    su3_heat_kernel_torus_enclosure,
)

bridge = su2_heat_kernel_bridge(Q(1, 5), [1, 0, 0, 0], [-1, 0, 0, 0])
assert bridge["density_enclosure_verified"]
assert replay_heat_kernel_certificate(bridge["certificate"])

derivative = su2_heat_kernel_enclosure(Q(3, 5), Q(3, 5), derivative_order=2)
su3 = su3_heat_kernel_torus_enclosure(1, [Q(1, 5), Q(2, 5)])
comparison = su2_wilson_heat_kernel_comparison(4, 32, face_count=4)
assert comparison["finite_coarse_product_comparison_verified"]
assert not comparison["multidimensional_wilson_refinement_verified"]
```

The six public entry points are `su2_heat_kernel_enclosure`,
`su2_heat_kernel_bridge`, `su3_heat_kernel_torus_enclosure`,
`su2_wilson_heat_kernel_comparison`, `normalized_product_tv_bound`, and the
canonical `replay_heat_kernel_certificate`. Times, angles, group coordinates
and ratio bounds are exact integers or `Fraction` objects. Cutoffs and counts
are strict integers. SU(2) derivative orders are 0, 1 or 2; both heat-kernel
cutoffs lie in 0 through 256. Wilson convolution counts lie in 1 through 1024,
face counts in 1 through 10000, and the directed Bessel argument must be at most
600. These are resource scopes, not claims of uniform numerical conditioning.

A `PASS` enclosure always includes the infinite representation tail. The generic
product-TV builder proves an implication from explicit external pointwise
premises. The Wilson-comparison builder independently earns its pointwise
premise when its relative-error gate passes. Parent and formal flags remain
false. The complete analytic justification follows.


This note supplies an actual normalized random holonomy bridge, all-representation
heat-kernel enclosures for SU(2) and SU(3), and a quantitative comparison between
finite convolutions of SU(2) Wilson densities and the heat kernel. The common
measure is normalized Haar. These are compact-group statements. They do not
identify a three- or four-dimensional fine-link integration with independent
coarse face factors, construct the continuum theory, or determine a physical
Hamiltonian mass gap.

The implementation is `omnibias.geometry.gauge.stochastic.heat_kernel`. A finite
computation passing does not complete the continuum scaling stage. Every
interval uses the directed transcendental backend; every finite character sum
has an analytic infinite tail. Numerical reference samples check the arithmetic
but do not supply the tail theorem or a formal proof tier.

## 1. Normalization and an exact stochastic bridge

Let \(C\) be the positive group Casimir, with fundamental value \(3/4\) for
SU(2) and \(4/3\) for SU(3). Write \(K_t\) for the kernel of \(e^{-tC}\). On a
compact connected group and for \(t>0\), this smooth kernel is strictly positive,
central, and has Haar integral one. Peter–Weyl orthogonality gives

\[
K_t*K_s=K_{t+s},\qquad
\int_G K_t(X)K_s(X^{-1}B)\,dX=K_{t+s}(B).                 \tag{1}
\]

Consequently the first increment of five independent \(K_t\)-distributed group
increments, conditioned on their ordered product \(B\), has the probability
density

\[
q_t(X\mid B)=\frac{K_t(X)K_{4t}(X^{-1}B)}{K_{5t}(B)}.    \tag{2}
\]

This is an exact disintegration of a specified probability law, including
\(B=-I\). It is conjugation covariant, since simultaneous conjugation preserves
all three class arguments. It is neither a deterministic fifth root nor an
assertion about the true Kogut–Susskind vacuum. The present API evaluates the
density; it does not implement an exact random sampler.

For SU(2), represent \(X,B\) by exact rational unit quaternions. Their half traces
are their scalar components, and the half trace of \(X^{-1}B\) is their Euclidean
four-vector dot product. Thus no numerical group-membership hypothesis enters
the bridge. The implementation encloses all three kernels and divides only when
the denominator interval has positive lower endpoint. Strict positivity of the
mathematical heat kernel does not ensure that a chosen finite computation can
separate a tiny value from zero: such a computation returns `INCONCLUSIVE` while
retaining the analytic normalization statement (1).

Heat-kernel face measures and their subdivision consistency are standard in
two-dimensional Yang–Mills; see the construction reviewed in
[Yang–Mills Measure and the Master Field on the Sphere](https://link.springer.com/article/10.1007/s00220-020-03773-6).
That two-dimensional construction does not supply the shared-link integration
identity needed for a three- or four-dimensional Wilson refinement.

## 2. SU(2): full-spin values and class-coordinate derivatives

With \(x=\operatorname{Tr}U/2\in[-1,1]\), the doubled-spin label is \(n\ge0\),
its dimension is \(d_n=n+1\), and \(C_n=n(n+2)/4\). Therefore

\[
K_t(x)=\sum_{n=0}^{\infty}(n+1)e^{-tn(n+2)/4}U_n(x),     \tag{3}
\]

where \(U_n\) is the Chebyshev polynomial of the second kind. The finite sum is
evaluated by \(U_{n+1}=2xU_n-U_{n-1}\). Differentiating this polynomial recurrence
once or twice computes the two supported derivatives without nested autodiff.
These are derivatives in the scalar class coordinate \(x\), not group-gradient
norms.

The Gegenbauer derivative identity, or its trigonometric finite-sum form, gives
for every nonnegative derivative order \(r\le n\)

\[
\max_{-1\le x\le1}|U_n^{(r)}(x)|
 =U_n^{(r)}(1)
 =\frac{(n+1)\prod_{j=1}^r((n+1)^2-j^2)}{(2r+1)!!}
 \le\frac{(n+1)^{2r+1}}{(2r+1)!!}.                       \tag{4}
\]

For \(n<r\) the derivative vanishes. Hence, after retaining \(0\le n\le N\), a
uniform remainder for (3) and its \(r\)-th derivative is

\[
T_{N,r}(t)=\frac1{(2r+1)!!}
 \sum_{n=N+1}^{\infty}(n+1)^{2r+2}e^{-tn(n+2)/4}.        \tag{5}
\]

Set \(p=2r+2\). The consecutive ratio of the majorant terms is

\[
q_n=\left(\frac{n+2}{n+1}\right)^p e^{-t(2n+3)/4}.       \tag{6}
\]

Both positive factors decrease in \(n\). If a directed upper bound on
\(q_{N+1}\) is strictly below one, (5) is at most its first term divided by
\(1-q_{N+1}\). This is an all-spin remainder, not a truncation hypothesis.
If the geometric test fails, the API does not assign a finite tail. Endpoint
values in (4) are exact rationals; finite recurrence intervals may safely be
intersected with their global character bounds. The kernel itself may also be
intersected with \([0,\infty)\) by positivity.

## 3. SU(3): every irreducible representation in the tail

On the maximal torus use eigenvalues
\(e^{ia},e^{ib},e^{-i(a+b)}\), with real angles in radians. Every
conjugacy class has such a representative. The API evaluates the subset
with exact rational angles. An irreducible representation has
labels \(p,q\ge0\), dimension and Casimir

\[
d_{p,q}=\frac{(p+1)(q+1)(p+q+2)}2,\qquad
C_{p,q}=\frac{p^2+q^2+pq+3p+3q}{3}.                      \tag{7}
\]

The full heat kernel is
\(\sum_{p,q}d_{p,q}e^{-tC_{p,q}}\chi_{p,q}\). The code retains \(p+q\le N\).
For finite characters, write \(z=\operatorname{Tr}U\) and let \(h_k\) be the
complete symmetric polynomial in the three eigenvalues. Since their product is
one and their second elementary symmetric polynomial is \(\bar z\),

\[
h_k=zh_{k-1}-\bar z h_{k-2}+h_{k-3},\quad h_0=1,
\qquad
\chi_{p,q}=h_{p+q}h_q-h_{p+q+1}h_{q-1},                 \tag{8}
\]

with negative-index \(h\)'s zero. These are exact character identities. Directed
complex interval arithmetic evaluates them, and \(|\chi_{p,q}|\le d_{p,q}\)
bounds each finite contribution. The retained shell is closed under conjugation
\(p\leftrightarrow q\), so taking real parts gives the real truncated kernel.

Put \(m=p+q\). There are \(m+1\) representations in this shell and

\[
d_{p,q}\le\frac{(m+2)^3}{8},\qquad
C_{p,q}\ge\frac{m^2}{4}+m.
\]

The first inequality follows from
\((p+1)(q+1)\le(m+2)^2/4\); the second follows from
\(p^2+q^2+pq\ge3m^2/4\). Consequently the omitted kernel is bounded uniformly
on the group by

\[
T_N^{(3)}(t)\le\frac1{64}\sum_{m=N+1}^{\infty}
 (m+2)^7 e^{-t(m^2/4+m)}.                                \tag{9}
\]

The consecutive ratio is
\(((m+3)/(m+2))^7e^{-t(2m+5)/4}\), again decreasing. The first-term geometric
bound proves the all-representation remainder whenever this ratio is below one.
This includes representation dimensions and shell multiplicity; omitting either
would produce an invalid kernel tail.

## 4. A normalized-product perturbation theorem

Let \(f_i,g_i>0\) be measurable factors on a **common** support and base measure.
Assume their products have finite nonzero normalizers and that actual pointwise
bounds \(0<a_i\le f_i/g_i\le b_i<\infty\) hold. Put
\(L=\prod a_i\), \(U=\prod b_i\). The product ratio and normalizer ratio lie in
\([L,U]\). Thus the normalized density ratio lies in \([L/U,U/L]\).

If a probability density ratio \(r\) lies in \([a,b]\), has mean one, and
\(a\le1\le b\), convexity of \(|r-1|\) gives

\[
\|\mu-\nu\|_{\mathrm{TV}}
 \le\frac{(1-a)(b-1)}{b-a}.
\]

Inserting \(a=L/U,b=U/L\) gives the convenient rational bound

\[
\|\mu-\nu\|_{\mathrm{TV}}\le\frac{U-L}{U+L}.             \tag{10}
\]

Here TV is half the \(L^1\) distance. A bounded observable changes by at most
\(2\|O\|_\infty\) times (10). For \(L=U\), the normalized measures coincide.
No independence of the factors or face variables is needed.

The generic `normalized_product_tv_bound` API checks only this rational
implication. Its caller-provided ratio intervals are explicit analytic premises;
it does not promote them into actual measure bounds. The next section supplies
one independently earned source of those intervals.

## 5. Actual Wilson convolution versus the heat kernel

Define the normalized one-group Wilson density

\[
w_\beta(U)=\frac{e^{\beta\operatorname{Tr}U/2}}{Z_\beta},
\qquad Z_\beta=\frac{2I_1(\beta)}\beta.                   \tag{11}
\]

The SU(2) class Haar integral has weight \(2\sin^2\theta/\pi\). Integrating
\(e^{\beta\cos\theta}\sin((n+1)\theta)\sin\theta\) gives the Fourier multiplier

\[
a_n(\beta)=\frac{I_{n+1}(\beta)}{I_1(\beta)},\qquad
w_\beta^{*m}(x)=\sum_{n\ge0}(n+1)a_n(\beta)^mU_n(x).    \tag{12}
\]

The API compares this **actual normalized \(m\)-fold group convolution** with
\(K_t\), taking \(\beta=2m/t\). This scaling matches the leading fixed-spin
large-\(\beta\) exponent, but the finite error below does not rely on an
uncontrolled Bessel asymptotic expansion.

For retained spins, compute directed enclosures of the Bessel ratios using the
existing verified Bessel implementation. Its current finite argument scope is
\(0<\beta\le600\). Character bounds give

\[
E_{N,m,t}^{\mathrm{finite}}
 =\sum_{n=1}^N(n+1)^2
   |a_n(2m/t)^m-e^{-tC_n}|.                              \tag{13}
\]

To control every omitted Wilson spin, use the positive series

\[
I_j(\beta)=\sum_{k\ge0}
 \frac{(\beta/2)^{2k+j}}{k!(k+j)!}.
\]

Termwise comparison proves, for every integer \(j\ge1\),

\[
\frac{I_{j+1}(\beta)}{I_j(\beta)}
 \le\frac{\beta}{2(j+1)}.                               \tag{14}
\]

Therefore the consecutive ratio of
\((n+1)^2a_n(\beta)^m\) is at most

\[
q_n^{W}=\left(\frac{n+2}{n+1}\right)^2
         \left(\frac{\beta}{2(n+2)}\right)^m,             \tag{15}
\]

which decreases with \(n\). The implementation directly encloses the first
omitted \(a_{N+1}\). If \(q_{N+1}^{W}<1\), the full Wilson tail is at most
that first weighted term divided by \(1-q_{N+1}^{W}\). Adding this tail and (5)
to (13) proves the actual uniform estimate

\[
\sup_{U\in\mathrm{SU}(2)}|w_{2m/t}^{*m}(U)-K_t(U)|
 \le E_{N,m,t}.                                         \tag{16}
\]

Both densities integrate to one; therefore the one-group TV distance is at most
\(\min(1,E_{N,m,t}/2)\). A computable global heat-kernel floor is

\[
L_t=1-\sum_{n=1}^N(n+1)^2e^{-tC_n}-T_{N,0}(t).           \tag{17}
\]

If its directed lower endpoint is positive and \(e=E_{N,m,t}/L_t<1\), (16)
yields the actual ratio interval \([1-e,1+e]\). For any fixed finite family of
measurable holonomy maps \(U_f\) on a common compact Haar configuration space,
apply (10) to
\(f_f=w_{2m/t}^{*m}(U_f)\) and \(g_f=K_t(U_f)\). Compactness, positivity and
boundedness discharge the normalizer assumptions. This remains valid when
holonomy maps share link variables: independence is unnecessary for (10).

At \(t=4,N=32\), the report proves the following conservative decimal upper
bounds (the certificates contain exact outward rational endpoints):

| Convolution steps \(m\) | Uniform density error | One-group TV | Four-factor TV |
|---:|---:|---:|---:|
| 8 | 0.059845 | 0.029923 | 0.291862 |
| 16 | 0.035501 | 0.017751 | 0.176243 |
| 32 | 0.018666 | 0.009333 | 0.093324 |

These three finite cases do not establish a rate valid for every \(m\). They also
do not establish that integrating an arbitrary multidimensional fine Wilson
lattice produces the factors \(w_\beta^{*m}(U_f)\). Shared-link integration can
generate additional interactions even though (10) can compare already specified
interacting factor products. The scaling stage therefore still needs an actual
refinement identity or a controlled replacement, a uniform approximation rate,
and a bound compatible with growth of the number of faces and physical units.

## 6. Replay and independent checks

`replay_heat_kernel_certificate` verifies the seal and rebuilds the entire
canonical report from exact inputs, including finite sums, all tails, scope
flags, and any nested product certificate. Rehashed changes to arithmetic or
scope are rejected. Canonical `INCONCLUSIVE` reports also replay as such; replay
never promotes their failed enclosure gates. Exact-rational inputs reject
booleans and floats. Very small kernel denominators, unsupported Bessel arguments,
or exhausted finite arithmetic can cause an inconclusive result or an explicit
resource-domain refusal.

Independent regressions include full-group rational Haar moments for (1),
integer Chebyshev derivative polynomials, SU(3) Weyl determinants, direct Wilson
Haar integrals, high-precision grid and seeded random evaluations, and rehashed
tampering attempts. The report evaluates 75 SU(2) derivative samples, 11 bridge
samples, 6 SU(3) samples and 75 Wilson-error samples. These checks supplement the
analytic remainder derivations; they do not certify continuum or formal claims.
