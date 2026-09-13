# Actual SU(2) strip marginal hierarchy

The [spatial covariance estimate](gauge-marginal-majorant.md) and
the [invariant Fourier vacuum](gauge-invariant-vacuum-fourier.md) give
a closed weighted Hessian class for the actual vacuum of every finite
open \(1\times n\) SU(2) strip. The bound survives arbitrary repeated
integration of vertical coordinates and carries the exact induced
physical electric form. This is at fixed microscopic coupling.

## Physical source and weighted norm

Use
\[
aH=\frac{\kappa}{2}\sum_e C_e+
\frac{2}{\kappa}\sum_p(2-\operatorname{ReTr}U_p),\qquad
\alpha=\kappa/2,\quad g=4/\kappa^2.
\]
There are \(3n+1\) edges and \(2n+2\) vertices. Every strip has girth
four, plaquette incidence at most two and plaquette ambient line-graph
diameter two. The source constructs the actual positive vacuum
\(\psi_0=e^{S_*+u}\), with weighted correction norm at most \(r\),
if
\[
A=16gb^2,\qquad \tfrac43(A+r)^2\le r,\qquad
\tfrac83(A+r)<1.
\]
All representation sectors of the original links are included in
this construction. No truncated trial vacuum is substituted.

Fix both horizontal rows to the identity, leaving vertical
\(Q_0,\ldots,Q_n\) with common left/right endpoint gauge actions.
This is a forest gauge; its transformations depend only on horizontal
links. The remaining Haar measure is a product, and vertical
covariant Hessian blocks restrict by isometries.
In these coordinates
\[
S_*=\frac g3\sum_{i=1}^n\chi(Q_{i-1}Q_i^{-1}).
\]
Each interior diagonal seed block is bounded by \(g/3\), and each
neighbor block by \(g/6\). The ambient distance of distinct vertical
edges is \(|i-j|+1\), so restricting the source's weighted correction
bound gives a bound in site distance \(|i-j|\) of \(2r/3\).
Therefore
\[
\sup_i\sum_j b^{|i-j|}
 \sup_Q\|\operatorname{Hess}_{ij}S(Q)\|_{\rm op}
\le m_b=\frac{g(1+b)}3+\frac{2r}3.
\]
The source's existing scalar Hessian output is **unweighted**.
The API recomputes this weighted seed bound instead of copying it.

With the SU(2) metric of fundamental Casimir \(3/4\), Ricci is
\(\rho=1/2\). If \(m_b<1/4\), the covariance/Schur theorem preserves
this very same \(m_b\) under any coordinate marginal. It retains all
generated interactions. Every marginal has curvature at least
\(\rho-2m_b\) and covariance kernel weighted row at most
\((\rho-2m_b)^{-1}\). The distance is inherited microscopic distance.
The constants have no dependence on \(n\) or on the elimination depth.

## Exact physical compression

Let \(I=(i_0<\cdots<i_s)\) retain endpoints \(0,n\).
Write \(p_i\) for right gradients and \(q_i=\operatorname{Ad}(Q_i)p_i\)
for left gradients. Endpoint Gauss law gives
\(\sum_i p_i=\sum_i q_i=0\).
Differentiating the original vertical and horizontal edges gives
\[
\Gamma_I(f)=\sum_{i\in I}|p_i|^2+
\sum_{j=0}^{s-1}(i_{j+1}-i_j)
\left(\left|\sum_{h=0}^j p_{i_h}\right|^2+
\left|\sum_{h=0}^j q_{i_h}\right|^2\right).
\]
Each horizontal edge differentiates a suffix in a common left or
right frame; Gauss law turns it into the corresponding prefix.
All cuts between neighboring retained columns have the same prefix,
so their number is exactly \(i_{j+1}-i_j\).
Both horizontal rows contribute; their gradients are not identified.

This tensor depends only on retained variables. Its compression
therefore has no fluctuating kinetic coefficient in the integrated
fiber. Consecutive compressions add adjacent interval lengths, giving
the same tensor as direct compression. Fubini likewise gives the
same actual marginal density. The closed quadratic form is
\[
q_I(f)=\alpha\int\Gamma_I(f)\,d\mu_I,\qquad
\Gamma_I(f)\ge\sum_{i\in I}|\nabla_i f|^2.
\]
Positive smooth densities, compactness, ellipticity and gauge
averaging supply an invariant smooth form core.
The product-metric Poincare inequality then yields
\[
\operatorname{gap}(H_I-E_I)\ge\alpha(\rho-2m_b).
\]
The vacuum-preserving embeddings carry the same fine vacuum at each
stage. This is compression of quadratic forms, not an assertion that
compressed semigroups equal the compression of the fine semigroup.
Rayleigh restriction already preserves a known fine gap; the useful
additional result is the controlled marginal derivatives and explicit
kinetic family.

## Exact witness and replay

At \(\kappa=24,b=5/4,r=1/8\),
\[
A=25/144,\quad r-\tfrac43(A+r)^2=95/15552,\quad
\tfrac83(A+r)=43/54,
\]
\[
m_b=17/192,\quad \rho-2m_b=31/96,\quad
\|K\|_{\text{weighted row}}\le96/31,\quad
\operatorname{gap}(H_I-E_I)\ge31/8.
\]
The gap uses the **original dimensionless \(aH\) units** at every
depth. It is not a continuum physical mass.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_strip_marginal_hierarchy,
    su2_strip_compression_geometry,
    replay_su2_strip_marginal_hierarchy_certificate,
)

result = su2_strip_marginal_hierarchy(
    24, correction_radius=Q(1, 8), decay_base=Q(5, 4),
)
assert result["all_compressed_physical_gap_lower"] == "31/8"
assert replay_su2_strip_marginal_hierarchy_certificate(result["certificate"])
geometry = su2_strip_compression_geometry(8, [0, 3, 8])
assert geometry["cut_weights"] == [3, 5]
```

Source failure or a nonpositive row margin is `INCONCLUSIVE`.
At \(b=1\) the unweighted closure may pass, but no spatial exponential
decay flag is earned. Certificates seal rational premises and the
written analytic implication; they do not earn Lean or Mathlib.

## Exact coupling window of this source criterion

A positive radius satisfying
\(B(A+r)^2\le r\), \(2B(A+r)<1\), \(B=4/3\), exists exactly when
\(4BA<1\). Necessity follows from the maximum of
\(r-B(A+r)^2\), which is \(1/(4B)-A\); at equality its only
zero has contraction one. For sufficiency choose the rational
\(r=A\), giving self-map slack \(A(1-4BA)>0\) and contraction
\(4BA<1\). Thus the exact condition is
\[
3\kappa^2>1024b^2.
\]
Omitting `correction_radius` selects \(r=A\). This witness also
gives \(m_b<1/8+(b+1)/(256b^2)\le17/128<1/4\), so its Hessian
closure gate passes whenever its source criterion passes.
An explicitly smaller radius can improve the bound, as in the
\(31/8\) example. The certificate reports the exact feasibility
slack independently of whether the selected radius passes.

In the adopted Kogut--Susskind normalization \(\kappa=g_0^2\);
the weak-coupling direction is \(\kappa\to0\).
This radius calculation shows precisely why the present source
criterion cannot reach that direction. It does not exclude a gap,
an actual vacuum, or a different constructive estimate.

The result concerns arbitrary depth within each fixed finite strip,
with constants uniform over the strip family. It does not identify
vacua on independently refined graphs, a running coupling, a
four-dimensional block construction, an infinite-volume QFT or
continuum OS axioms. Those parent flags remain false.

## Stronger actual strip bound from conditional spectra

`su2_strip_conditional_hierarchy` uses the
[conditional Poincare Schur theorem](gauge-marginal-majorant.md#extension-using-actual-conditional-poincare-floors)
on the same actual Fourier vacuum. It does not import the preceding
curvature gap.

For one vertical coordinate the oscillation of the actual conditional
log density \(2S\) is at most
\[
\Omega=(16g+8r)/3.
\]
The two incident seed plaquettes contribute \(16g/3\).
In the invariant spin norm, every nonconstant gauge support has
electric energy at least three; the coefficient sum involving a
given vertical is at most \(2r/3\). Its contribution to
\(\operatorname{osc}(2u)\) is therefore at most \(8r/3\).
Comparison with the SU(2) Haar gap gives a uniform actual
one-site conditional floor \((3/4)e^{-\Omega}\).

The mixed-only weighted Hessian row admits the sharper bound
\[
c_b=\frac{gb}{3}+\frac{2r}{3b}.
\]
The second term uses the **extra unit** in the ambient distance
\(d_{\rm line}(V_i,V_j)=|i-j|+1\) for distinct verticals.
This improvement applies only to mixed blocks; a diagonal Hessian
bound cannot receive that factor \(b^{-1}\).

For positive integer \(N\) and \(0\le\Omega<N\), the exact rational
bound
\[
\gamma_N=\frac34(1-\Omega/N)^N\le\frac34e^{-\Omega}
\]
follows from \(\log(1-z)\le-z\). The conditional comparison margin
\[
\eta=\gamma_N-2c_b>0
\]
then survives every marginal Schur complement. Its covariance kernel
has weighted row at most \(1/\eta\), and the already identified
physical electric form gives gap at least \(\alpha\eta\) for every
compression. This propagates conditional floors and mixed bounds;
it does not assert an unchanged absolute diagonal-Hessian ball.

At \(\kappa=24,b=5/4,r=1/8,N=4\),
\[
\Omega=10/27,\quad
\gamma_N=5764801/11337408,\quad c_b=601/8640,
\]
\[
\eta=20937683/56687040,\qquad
\operatorname{gap}(H_I-E_I)\ge
\frac{20937683}{4723920}> \frac{22}{5}.
\]
These are uniform finite-strip-family bounds in the same microscopic
units as before. The exact target slack is
\(20937683/4723920-22/5=30487/944784>0\).

```python
from omnibias.geometry.gauge.transfer import (
    su2_strip_conditional_hierarchy,
    replay_su2_strip_conditional_hierarchy_certificate,
)

result = su2_strip_conditional_hierarchy(
    24, correction_radius=Q(1, 8), decay_base=Q(5, 4), exponent_steps=4,
)
assert result["all_compressed_physical_gap_lower"] == "20937683/4723920"
assert replay_su2_strip_conditional_hierarchy_certificate(result["certificate"])
```

This sharper bound uses the same actual-vacuum source. Its exact
coupling window above has not expanded. A source failure, an invalid
rational exponential domain, or a nonpositive comparison margin is
`INCONCLUSIVE`. No arbitrary supplied conditional gap is accepted
by this actual-strip consumer.
