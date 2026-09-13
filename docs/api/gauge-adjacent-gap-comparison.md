# Actual adjacent-vacuum gaps by density comparison

This note gives a different spectral criterion for the actual vacuum on
the fixed seven-edge SU(2) graph of two adjacent squares. Once an actual
logarithmic correction \(U\) with original Fourier bound \(N(U)\le r\)
has been earned, a bounded-density comparison gives a positive gap for
every finite \(r\), even when the global Ricci-minus-Hessian estimate is
negative. It does not construct that correction or assume that a
reference inverse alone is an actual-vacuum certificate.

The bounded-perturbation framework is classical; primary context is
[Holley–Stroock, *Logarithmic Sobolev inequalities and stochastic Ising
models*, J. Stat. Phys. 46 (1987), 1159–1194](https://link.springer.com/article/10.1007/BF01011161).
The Poincaré comparison needed here is proved directly below. No novelty
or continuum theorem is inferred from this finite-graph calculation.


The public API accepts a canonical actual-vacuum certificate. It inherits
the coupling and correction radius from that source and replays every
parent; the caller cannot supply a gap or an honesty flag.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer import (
    su2_adjacent_cone_vacuum,
    su2_adjacent_vacuum_gap_comparison,
    replay_su2_adjacent_vacuum_gap_comparison_certificate,
)

source = su2_adjacent_cone_vacuum(
    6, correction_radius=Fraction(6, 25),
)
result = su2_adjacent_vacuum_gap_comparison(
    source["certificate"], exponent_steps=4,
)
assert source["actual_vacuum_verified"]
assert not source["curvature_gap_verified"]
assert result["status"] == "PASS"
assert Fraction(result["physical_gap_lower"]) > Fraction(4, 3)
assert result["full_scalar_gap_lower"] == "22709885409121/23619600000000"
assert replay_su2_adjacent_vacuum_gap_comparison_certificate(
    result["certificate"],
)
assert not result["continuum_claim"]
assert not result["yang_mills_mass_gap_claim"]
```

```text
su2_adjacent_vacuum_gap_comparison(
    source_certificate: dict, *, exponent_steps: int = 4,
) -> dict

replay_su2_adjacent_vacuum_gap_comparison_certificate(certificate: dict) -> bool
```

The two supported source types are
```text
su2_adjacent_preconditioned_vacuum_v1
su2_adjacent_cone_vacuum_v1
```
Each type has its own canonical replay. An unsupported or forged source
raises `ValueError`. A canonical source with an unsuccessful nonlinear
gate returns `INCONCLUSIVE` and no gap. A false curvature gate alone
does not prevent the comparison.

The output `physical_gap_lower` is the neutral, gauge-invariant gap in
dimensionless `aH` units; `full_scalar_gap_lower` is the separately
computed bound for the entire scalar seven-link space. These fields
have different exponents and must not be interchanged. The following
sections give the complete analytic implication of the exact replay.

## 1. The exact source and norm

The graph, gauge constraints, and original metric are those in
[the complete preconditioned vacuum source](gauge-adjacent-vacuum.md).
Every one of its seven original edges has electric weight one; its
three invariant paths have lengths \(3,3,1\). Set
\[
 aH=\frac{\kappa}{2}C+\frac2\kappa(4-\chi_p-\chi_q),
 \qquad g=4/\kappa^2,\qquad
 S=S_*+U,\quad S_*=\frac g3(\chi_p+\chi_q).
 \tag{1}
\]
The required premise is that \(e^S\), up to normalization, is the true
positive groundstate of (1), with \(U\) real, Haar-centered, invariant,
and \(N(U)\le r\). This is supplied only by a passing complete source
construction and its canonical replay. It is not supplied by a chosen
positive number \(r\).

In doubled theta labels, the normalized functions are
\[
 b_{abs}=\frac{\operatorname{Tr}
 [P_s(D_{a/2}\otimes D_{b/2})]}{s+1},
 \qquad |a-b|\le s\le a+b,\quad a+b+s\ {\rm even}.
\]
Here \(P_s\) projects onto total spin \(s/2\). Equivalently each basis
function is a matrix coefficient of a unit vector in the diagonal
invariant subspace of the three path representations. Therefore
\[
 |b_{abs}(x)|\le1\quad\hbox{on every configuration}.      \tag{2}
\]
The original-link coefficient nuclear norm and electric energy are
\[
 n_{abs}=(a+1)^2(b+1)^2,\qquad
 E_{abs}=\frac{3a(a+2)+3b(b+2)+s(s+2)}4.
\]
For \(U=\sum_{abs\ne000}u_{abs}b_{abs}\), put
\[
 N_i(U)=\sum_{abs\ne000}w_i(a,b,s)|u_{abs}|,\quad
 w_i=\frac i2 E_{abs}n_{abs},\qquad N=\max(N_a,N_b,N_s).
 \tag{3}
\]
This remains the original norm. No norm is transferred through a gauge
coordinate map by assuming invariance under partial transpose.

## 2. A sharp coefficient bound from an exact positive dual

Take the three rational weights
\[
 (\lambda_a,\lambda_b,\lambda_s)=\frac1{72}(1,1,11).
 \tag{4}
\]
For every nonconstant admissible label,
\[
 \lambda_aw_a+\lambda_bw_b+\lambda_sw_s
 =\frac{E_{abs}n_{abs}(a+b+11s)}{144}\ge1.              \tag{5}
\]
Here is a complete verification. If \(s\ge1\), admissibility forces
\(a+b\ge s\ge1\), hence \(n_{abs}\ge4\). The nonconstant invariant
energy floor is \(E_{abs}\ge3\). Also \(a+b+11s\ge12\), so the
numerator in (5) is at least \(3\cdot4\cdot12=144\).
The floor three follows either from the original girth-four support
argument or directly from the nonnegative doubled labels in the
displayed energy formula. If \(s=0\), admissibility forces
\(a=b\ge1\), and the numerator is
\[
 3a^2(a+2)(a+1)^4\ge3\cdot1\cdot3\cdot16=144.
\]
This covers every spin and every admissible state, including support
with trivial representations on one path.

Summing (5) against \(|u_{abs}|\) gives
\[
 \boxed{\sum_{abs\ne000}|u_{abs}|
 \le\frac{N_a(U)+N_b(U)+11N_s(U)}{72}
 \le\frac{13}{72}N(U).}                               \tag{6}
\]
This coefficient constant is sharp. The three nonzero positive
coefficients
\[
 u_{101}=u_{011}=1/12,\qquad u_{110}=1/72
\]
give \(N_a=N_b=N_s=1\) and coefficient sum \(13/72\).
This sharpness statement concerns (6), not the pointwise oscillation
bound obtained next.

Absolute convergence, (2), and (6) imply
\[
 \|U\|_\infty\le13r/72,\qquad
 \boxed{\operatorname{osc}(2U)\le13r/18.}               \tag{7}
\]
The constant mode would not affect oscillation, but it is removed in
the source norm. The factor four between the coefficient sum and
\(\operatorname{osc}(2U)\) is explicit; optimality of that oscillation
constant is not claimed.

## 3. Elementary bounded-density Poincaré comparison

Let \(\nu\) and \(\mu\) be probability measures on the same configuration
space with \(d\mu=w\,d\nu\) and \(0<m\le w\le M<\infty\).
Suppose
\[
 \operatorname{Var}_\nu f\le\gamma_\nu^{-1}
                    \int\Gamma(f,f)\,d\nu
 \tag{8}
\]
on the required function sector and form domain. Then
\[
 \begin{split}
 \operatorname{Var}_\mu f
 &=\inf_c\int|f-c|^2w\,d\nu\\
 &\le M\operatorname{Var}_\nu f
 \le\frac{M}{\gamma_\nu m}\int\Gamma(f,f)\,d\mu.
 \end{split}
\]
Thus
\[
 \gamma_\mu\ge\gamma_\nu\,m/M.                         \tag{9}
\]
This does not identify the means under the two measures: the variance
is defined by its own minimizing constant. If
\(w=e^{2U}/\nu[e^{2U}]\), normalization cancels from \(M/m\), giving
\[
 \gamma_\mu\ge\gamma_\nu e^{-\operatorname{osc}(2U)}
             \ge\gamma_\nu e^{-13r/18}.               \tag{10}
\]
The same diffusion form \(\Gamma\) occurs on both sides. Since the
source produces a smooth actual \(U\) on the compact graph, bounded
density and form-core approximation justify the extension from smooth
functions to the relevant closed form domain.

## 4. Neutral reference gap with the original electric metric

Let
\[
 d\nu=Z_*^{-1}e^{2S_*}\,dH.
\]
This is the known reference measure, not the actual vacuum measure.
Fix the spanning tree consisting of both top horizontal edges and all
three vertical edges. The two bottom links remain as chords \(X,Y\).
The tree gauge is determined entirely by the five tree links and is
independent of either bottom link. Gauge-invariant functions become
smooth functions \(F(X,Y)\) invariant under simultaneous conjugation.

Haar invariance under the tree-dependent left and right translations
shows that the two chord variables have product Haar law. Each square
character becomes \(\chi(X)\) or \(\chi(Y)\), possibly with an inverse
which does not change the SU(2) character. Therefore
\[
 d\nu_{\rm chord}
 =\frac{e^{(2g/3)\chi(X)}}{Z_g}\,dH(X)\,
  \frac{e^{(2g/3)\chi(Y)}}{Z_g}\,dH(Y).                 \tag{11}
\]
There is no approximation or factorization assumption about the
actual vacuum in this statement: it factorizes this specified reference.

Varying one original bottom link changes only its own chord, through
left/right multiplication isometries independent of that bottom link.
Consequently its original electric gradient has exactly the norm of the
corresponding chord gradient. The other five original electric terms
are nonnegative, so pointwise
\[
 \Gamma_{\rm original}(f,f)
 \ge|\nabla_XF|^2+|\nabla_YF|^2.                       \tag{12}
\]
The original generator is not replaced by the right side. Its extra
terms are retained in the physical energy and only dropped in this
valid lower comparison. Singular quotient strata cause no omission:
the inequalities are proved on the full product \(SU(2)^2\) and then
restricted to invariant functions.

Single-group Haar has gap \(3/4\) in the chosen metric. The log density
of each factor in (11) has oscillation \(8g/3\), since
\(\chi\in[-2,2]\). Equations (8)–(9), tensorization, and (12) give
\[
 \boxed{\gamma_{\nu,\mathrm{neutral}}\ge
                          \frac34e^{-8g/3}.}          \tag{13}
\]
The constant is independent of the sign of any reference curvature.
This is a neutral gauge-invariant reference bound. It does not assert
that the full scalar seven-link diffusion is the two-link product
diffusion.

## 5. Actual neutral gap and a separate full-scalar bound

The actual vacuum probability is
\[
 d\mu=\frac{e^{2(S_*+U)}}{Z}\,dH
     =\frac{e^{2U}}{\nu[e^{2U}]}\,d\nu.
\]
Combine (10) and (13), then apply the actual groundstate form identity:
\[
 \boxed{\Delta_{\rm neutral}(aH)\ge
 \frac{3\kappa}{8}
   \exp\!\left(-\frac{8g}{3}-\frac{13r}{18}\right)>0.}   \tag{14}
\]
This holds for every finite earned \(r\). It requires no positive
lower bound on \({\rm Ric}-2\operatorname{Hess}(S_*+U)\).

A separate conclusion is available on the entire scalar original-link
space. Product Haar on all seven links has gap \(3/4\), and
\[
 \operatorname{osc}(2S_*)\le16g/3.
\]
Applying (9) directly between product Haar and the actual \(\mu\) gives
\[
 \boxed{\Delta_{\rm scalar}(aH)\ge
 \frac{3\kappa}{8}
   \exp\!\left(-\frac{16g}{3}-\frac{13r}{18}\right)>0.}  \tag{15}
\]
Equation (15) is weaker than (14), but covers more functions. A consumer
must keep the two sectors and exponents distinct. In either case the
vacuum subtraction is the true ground energy established by the source,
not a reference energy or a supplied gap estimate.

These bounds are for the fixed adjacent graph. The constant (6) and its
specific three-anchor geometry cannot be copied to growing cubic boxes.

## 6. Rational replay and the new spectral witness

For rational \(\Omega\ge0\) and integer \(m>\Omega\),
\[
 e^{-\Omega}\ge(1-\Omega/m)^m>0.                       \tag{16}
\]
This follows from \(\log(1-x)\le-x\).
It supplies a pure rational floor for (14) and (15); neither a floating
exponential nor an interval backend is required.
For any finite \(\Omega\), the choice
\(m\ge\lfloor\Omega\rfloor+1\) ensures the domain. Strict inequality is
needed; the tangent endpoint \(m=\Omega\) would give zero.

For the already earned source \(\kappa=7,r=1/10\), four steps give
\[
 \Omega_{\rm neutral}=\frac{2557}{8820},\qquad
 \Delta_{\rm neutral}\ge
 \frac{1146601351654183441}{590180693114880000}>1.94.
\]
The analogous full-scalar bound is
\(900268518173886481/590180693114880000>1.52\).
These are additional consequences of that same actual source,
not new source-existence proofs.

The new canonical source from the complete directional-dual inverse and
the \(8/9\) spherical-product bound earns \(\kappa=6,r=6/25\).
Its exact inverse and residual fields satisfy the simple rational caps
\[
 J<57/25,\qquad \varepsilon<1231/10000.
\]
With \(B=8/9\), these give self-map slack at least \(41/250000\)
and contraction at most \(608/625<1\). This is an actual nonlinear
source construction, using every theta spin. Its global-curvature floor is
\[
 \rho=\frac12-\frac8{27}-\frac8{25}=-\frac{157}{1350}.
\]
Replaying that source and applying the comparison gives, with four steps,
\[
 \Omega_{\rm neutral}=\frac{317}{675},\qquad
 \Delta_{\rm neutral}\ge
 \frac{32247508758721}{23619600000000}>\frac43,          \tag{17}
\]
and the full-scalar comparison gives
\(22709885409121/23619600000000>0.96\).
A negative curvature bound does not obstruct these Poincaré estimates.
The source and both comparison gaps pass canonical replay. The source
premise is established by its own complete inverse, residual, and
nonlinear bounds; equation (17) does not replace those bounds.

## 7. Scope of the consumer

The implemented consumer accepts only an explicitly supported canonical
adjacent actual-source certificate. It must replay the source and require
its actual nonlinear fixed point, even if the source's curvature-based
gap flag is false. Coupling, radius, source definition, original graph,
and norm are inherited from that replay. No user-supplied radius,
reference-only certificate, rehashed honesty flag, or finite residual
fit can replace it.

It records the coefficient dual, density oscillation, reference-sector
comparison, rational exponential domain, and resulting physical units.
Its neutral and full-scalar gap flags are distinct; infinite volume,
uniformity in spacing, continuum, Yang–Mills parent, and formal tiers
remain unearned. The next missing implication is a local norm/inverse
mechanism that can be assembled over larger graphs with constants that
remain controlled. This fixed-graph result alone supplies none of that
uniformity.

The implementation is `omnibias.geometry.gauge.transfer.adjacent_gap_comparison`.
Its new nonlinear parent is documented in
[the directional-dual source](gauge-adjacent-cone-vacuum.md); the reference
inverse is documented in [the complete adjacent inverse](gauge-adjacent-resolvent.md).
It accepts the old `su2_adjacent_preconditioned_vacuum_v1` source and
the new `su2_adjacent_cone_vacuum_v1` source explicitly. Its replay
reconstructs the source, exact coefficient dual, both different sectors,
the exponential domain, and every scope flag. The `exponent_steps`
parameter is a positive integer minimum: if necessary the implementation
raises it to \(\lfloor\Omega_{\rm scalar}\rfloor+1\), which ensures a
strictly positive rational floor for both sectors for every finite
earned radius. A canonical source whose nonlinear gate failed produces
`INCONCLUSIVE`, not a claim of zero gap.
