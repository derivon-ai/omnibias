# A two-cycle bound in a compact interior sector

This note assembles the regular and singular passages for one specified
itinerary of a quadratic family. The conclusion is **at most two distinct
cycles in that itinerary**, for sufficiently small parameters with the
strict margins below. It is not full cyclicity of the singular graphic,
the quadratic Hilbert problem, or Hilbert's full sixteenth problem.
The proof uses analytic theorems and exact algebra. No novelty, numerical
remainder certificate, or formal verification of the analytic argument is
claimed.

The regular estimates are in [regular jets](HILBERT16-REGULAR-JETS.md) and
[endpoint passage](HILBERT16-ENDPOINT-PASSAGE.md). The actual singular-map
identification, including time reversal and the parameter-dependent
application of the published entry-exit theorem, is in
[singular transport](HILBERT16-SINGULAR-TRANSPORT.md). That companion
distinguishes the printed theorem from our derivation of its uniform
passive-parameter extension. This distinction is part of the dependency
record of the result below.

## 1. A quantitative criterion for actual maps

Let I=[v0,V] be an interval with v0>0. Suppose H is C2 on I squared,
S is positive and C2 on I, and a,b>0 with L(v)=1-bv>0. Define

\[
 M(v)=\frac{av}{1-bv},\qquad q(v)=S(v)^2-M(v)^2,
 \qquad F(v)=H(v^2)-S(v)^2.
\]

If K2 bounds |H''|, Lmax bounds L from above, and

\[
 4VK_2+\frac{\|q''\|_\infty}{v_0}
       +\frac{\|q'\|_\infty}{v_0^2}
 <\frac{6a^2b}{L_{\max}^4},                              \tag{1}
\]

then F has at most two distinct zeros on I.

**Proof.** Direct differentiation gives the exact identity

\[
 \left(\frac{F'(v)}v\right)'
 =4vH''(v^2)-\frac{6a^2b}{(1-bv)^4}
   -\frac{q''(v)}v+\frac{q'(v)}{v^2}.                   \tag{2}
\]

The right side is strictly negative by (1). Therefore F'/v has at most
one zero. Since v>0, F' has at most one zero. Three distinct zeros of F
would contradict Rolle's theorem. This also excludes an interval of zeros.
No bound on the constant or linear part of H is required.

For explicit entry errors e=S-M with |e^(j)|<=delta_j, j=0,1,2, set

\[
 A_0=a_{\max}V/\ell,\quad A_1=a_{\max}/\ell^2,\quad
 A_2=2a_{\max}b_{\max}/\ell^3,\qquad \ell\le1-bv.
\]

Valid error bounds are

\[
 Q_1=2A_1\delta_0+2A_0\delta_1+2\delta_0\delta_1,
\]
\[
 Q_2=2A_2\delta_0+4A_1\delta_1+2A_0\delta_2
           +2\delta_1^2+2\delta_0\delta_2.              \tag{3}
\]

Use Q1,Q2 in (1). For a compact parameter set the strict right-hand
margin may be replaced by 6*amin^2*bmin/Lmax^4. Failure of this sufficient
inequality is inconclusive; it does not demonstrate three cycles.

## 2. Family and the restricted parameter sector

Use the five-parameter quadratic family in the weighted chart r=-1:

\[
 \dot x=(1+\alpha)x-y+x^2+\nu(p-1)xy+\nu^2my^2,
 \qquad \dot y=Cx+x^2+xy-\nu y^2.                       \tag{4}
\]

Here C ranges in a compact interval inside (1,infinity), alpha is free
in a sufficiently small fixed compact interval, and nu>0. The further
interior sector is specified by the exact analytic normalization in
the singular companion, equations (7)--(9): p=nu*ptilde and m=nu*mtilde,
with lambda in a fixed compact set Lambda satisfying

\[
 4\lambda_0+\lambda_1^2\le-\chi<0.                      \tag{5}
\]

In particular ptilde tends to 3 and mtilde to -2. This specifies a thin
sector of the unfolding, not an entire neighborhood of its origin.
The associated limiting singular map has

\[
 a=\exp\left(\frac{\pi\lambda_1}
                    {\sqrt{-4\lambda_0-\lambda_1^2}}\right),
 \qquad b=\frac{a+1}{3}.
\]

Choose a common compact input interval I=[v0,V], v0>0, with
1-bv>=ell>0 for every parameter and input. Choose a compact positive
output interval J containing all M(I) with a margin. Enlargements used
for composing the maps remain inside these strict domains. Compactness
gives positive amin and bmin. None of the estimates asserts uniformity
as chi, ell, v0, or Cmin-1 tends to zero.

The order of choices is essential: choose the regular section radius
rho>0 sufficiently small, and then choose nu sufficiently small depending
on that fixed rho and the compact sets. Put R=rho/nu>=2. The physical
sections are x=-R and x=+R.

## 3. Exact section coordinates and regular curvature

Let qx=(x^2+2x+C)/2, y0=(x^2-C)/2, and z=(y-y0)/qx. For sigma=+-1
use the following exact coordinate definitions on the finite-nu sections:

\[
 Z_\sigma(t)=\frac{2}{1-\sigma r\rho+
           \sqrt{1-2\sigma r\rho+\rho^2t}}-1,
\]
\[
 K_\sigma(z)=r^2+\frac4{\rho^2}
  \left[(1+z)^{-2}-(1-\sigma r\rho)(1+z)^{-1}\right].   \tag{6}
\]

Here r=-1 and t is a squared fiber label. These are inverse functions
on the branch with 2/(1+z)-(1-sigma*r*rho)>0. For rho<=1/4 and
|z|<=1/4 this reconstructed root is at least 7/20. On compact t ranges,
shrinking rho ensures the radicand is at least 1/4 and the denominator
in Z is at least 1. Direct differentiation then gives

\[
 |Z'|\le2\rho^2,\quad |Z''|\le8\rho^4,\quad
 |K'|\le20\rho^{-2},\quad |K''|\le80\rho^{-2}.          \tag{7}
\]

Let G be the actual regular map from left to right in z coordinates,
at one fixed alpha. The regular-jets proof gives uniform bounds
|G'|<=M1 and |G''|<=M2 for the admitted matching trajectories. For
example M1=theta and M2=theta*T2 in that proof; they depend on the fixed
coefficient box, not on R, nu, or the selected alpha. Define

\[
 H(t)=K_+\big(G(Z_-(t))\big).
\]

On the matching domain |G|<=zeta<=1/4. The chain rule and (7) imply

\[
 |H''|\le C_H\rho^2,\qquad
 C_H=320M_1^2+80M_2+160M_1.                              \tag{8}
\]

For example, the term K''*(G'*Z')^2 is at most
320*M1^2*rho^2, and the two terms multiplied by K' give the other
two summands. No limiting value of G', no reference value H(0), and
no zero-endpoint compensation is needed.

## 4. Which trajectories the regular estimate captures

Choose the fixed endpoint half-width zeta>0 and coarse relative-height
radius eta>0 small enough for the endpoint and regular-jets constants.
They are independent of R and nu. Shrink rho so that Z_-(I squared)
and Z_+(J squared) lie strictly inside [-zeta,zeta]; this follows from
Z_sigma(t)=sigma*r*rho+O(rho^2) uniformly on those compact label ranges.
The coarse-capture lemma in the
regular-jets companion applies to a physical trajectory crossing the
strip -R<=x<=R and satisfying

\[
 |y-y_0|\le\eta(1+|x|)^2,\quad |\alpha|\le\eta,
 \quad |z(-R)|,|z(R)|\le\zeta.                          \tag{9}
\]

Provided eta+nu*(R+1)<=eta_c and
(nu+zeta/R)*(R+1)<=eta_p, that lemma places the trajectory in the
weighted tube of the endpoint theorem. Its increasing x coordinate
makes it a graph. It is therefore the unique endpoint-matching
trajectory at its alpha, and the derivative estimates used in (8)
apply. These sufficient inequalities can be met by choosing eta,zeta,
then rho, and then nu small. The normalized p,m in (4) remain in a
fixed bounded coefficient box.

The domain needed for Rolle's theorem is connected. Indeed, write
alpha_R(z_in,z_out) for the endpoint matching function. Its input
derivative is strictly positive and output derivative strictly negative.
The inputs that admit the chosen fixed alpha are exactly those satisfying

\[
 \alpha_R(z_{in},+\zeta)\le\alpha
                \le\alpha_R(z_{in},-\zeta).             \tag{10}
\]

The first inequality defines an initial interval of inputs, and the
second a final interval. Their intersection is an interval. Z_- is
strictly decreasing, so its inverse image intersected with I squared
is also an interval. Thus, between any two admitted cycle inputs, G
is defined with the same uniform derivative bounds. It is unnecessary
for the intermediate matching trajectories to satisfy the coarser
condition (9): the endpoint construction already places them in the
fine tube used to prove those bounds.

## 5. The actual singular input and closing equation

The singular-transport companion proves, from the entry-exit theorem
and its parameter-family proof, that the normal-forward singular map
between the same finite-nu sections, in positive fiber labels, satisfies

\[
 S_\nu(v)=M(v)+O_{C^2,\rho}(\nu\log(1/\nu)).            \tag{11}
\]

This estimate is uniform on the stated compact sets after rho is fixed.
The proof uses ordinary transverse flow connectors between our sections
and the published theorem's small-height sections. Their finite-nu
corrections are part of S_nu. Formula (6) itself is an exact coordinate
choice, so no approximate coordinate substitution is made in H.

Normal-forward singular time is opposite physical time. Consequently
S_nu runs left-to-right in the section labels, while the physical
singular closing map is S_nu inverse. A cycle with this regular and
singular itinerary consequently satisfies

\[
 \sqrt{H(v^2)}=S_\nu(v),
 \quad\text{or equivalently}\quad H(v^2)-S_\nu(v)^2=0.    \tag{12}
\]

The output is positive and the inverse branch in (6) is fixed, so
squaring loses no admissible roots and introduces none on the physical
output domain. Values of H below zero cannot solve (12). Uniqueness of
the flow gives at most one such cycle for a given section point.

Choose rho so that 4*V*CH*rho^2 is less than half the fixed positive
margin 6*amin^2*bmin. Having fixed rho, (11) makes the Q1,Q2 terms
in (3) smaller than the other half for sufficiently small nu. Criterion
(1) now applies to the connected admitted domain from (10).
It proves **at most two distinct cycles following this itinerary and
meeting (9), with input in I and singular output in J**. The statement
holds uniformly for alpha in its stated compact interval, including
values selected by a compensation that is nonanalytic in nu: alpha is
held fixed when differentiating the section map.

This is an existence-of-a-cutoff result. The singular O(C2) remainder
constant and the final nu cutoff have not been computed as interval
certificates. The analytic proof, including its normal-form ingredients,
is not discharged by the repository's finite rational checker.

## 6. Remaining obligations and replay

The result excludes fibers approaching zero, the singular-map pole,
the discriminant boundary, other weighted parameter charts, and C=1.
It also assumes a specified regular/singular itinerary. No proof here
shows that every cycle near the graphic eventually belongs to one of
finitely many such corridors with a uniform total bound. Those coverage
and boundary problems remain before full graphic cyclicity. Other
quadratic cases, all degrees, configurations, and the algebraic
curve/surface part of Hilbert 16 remain separate.

Run the finite replay from the repository root:

```sh
uv run --no-sync python benchmarks/hilbert16_interior_cyclicity.py \
  --output artifacts/hilbert16/interior_cyclicity.json
uv run --no-sync python -O benchmarks/hilbert16_interior_cyclicity.py \
  --output artifacts/hilbert16/interior_cyclicity_optimized.json
```

It verifies eight symbolic identities, the rational derivative constants,
passing and inconclusive examples of (1), and a two-root example for
abstract maps. That example is not a realization by the physical family.
The test error bounds are declared inputs, not measurements or
certificates for an actual singular passage. The analytic proof above
and its companions supply a different kind of evidence.
