# A quantitative physical derivative gate in the thin center-height layer

This is a written analytic proof for the existing restricted physical
itinerary. It does not prove full graphic cyclicity or Hilbert 16, and no
formal verification or novelty claim is made. Constants are compact-domain
analytic constants, not computed interval cutoffs.

## 1. The achieved extension

Use the exact canonical normal form and physical sections from the
[center-height grazing proof](HILBERT16-GRAZING-PASSAGE.md). Fix `r=-1`, sufficiently small `rho>0`, compact
`C` in `(1,infinity)`, `A=1+nu*abar` with bounded `abar`, and

\[
 -L\le\lambda_0\le-c<0,\qquad |\lambda_1|\le L.
 \tag{1.1}
\]

Let `K>=0` be arbitrary and put `delta=epsilon(1+K)`. There are constants
`delta0,epsilon0,C>0`, fixed by the compact parameter set and sections,
for the following quantitative statement whenever

\[
 0<\epsilon\le\epsilon_0,\qquad \delta=\epsilon(1+K)\le\delta_0.
\]

For the exact normal trajectory with center height `H=h(0)`, consider

\[
 \epsilon^3\exp(-K)\le H\le\epsilon^3.
 \tag{1.2}
\]

The trajectory reaches both selected outer physical sections. Its actual
normal-forward signed-label transition `D_epsilon`, parameterized by those
trajectories, satisfies

\[
 \boxed{|D_\epsilon'-1|\le C\sqrt{\epsilon(1+K)}}
 \tag{1.3}
\]

uniformly on (1.2) and the compact coefficient sets. The derivative is with
respect to the signed physical section label, holding all physical
parameters fixed. The input and output endpoint movement is included.

In particular, let `K_epsilon>=0` be any function satisfying
`epsilon(1+K_epsilon)->0`; no regularity of this function is needed.
Combining (1.3) with the already proved center-height band gives

\[
 D_\epsilon'\longrightarrow1\quad\hbox{uniformly for}\quad
 \epsilon^3e^{-K_\epsilon}\le H\le H_{\max},
 \tag{1.4}
\]

where `Hmax` is any sufficiently small fixed upper height as in that proof.
The convergence on the added thin portion has the explicit rate (1.3);
no rate is asserted here for the previously proved upper band.

Consequently the same captured regular contraction yields at most one
cycle on the connected admitted part of this entire enlarged band for
sufficiently small `epsilon`. Counts are not added across subintervals.
This covers the whole asymptotic region
`epsilon*log(epsilon^3/H)->0` on the thin side. Section 7 also derives a
coarse cycle gate up to a sufficiently small fixed positive value of that
scaled logarithm. The [exponential companion](HILBERT16-EXPONENTIAL-PASSAGE.md)
subsequently proves the actual transition on every fixed finite positive
logarithmic-height band and joins it to this region with a two-cycle
bound. Unbounded logarithmic height and the simultaneous degeneration
`lambda0 -> 0` remain unresolved.

## 2. The exact identity that cancels the large variation

Write the exact system as

\[
 \dot V=f(V,\epsilon)+h g(V,h,\epsilon),\qquad
 \dot h=-Vh,
\]
\[
 f=\epsilon^3\lambda_0+\epsilon^2\lambda_1V
                          +\epsilon V^2\zeta(V,\epsilon),
 \qquad q=-f,\quad k=-g,\quad \mathcal D=q+kh.
\]

Where `mathcal D>0`, use `V` as independent variable:

\[
 h_V=\mathcal R={Vh\over\mathcal D}.
\]

The exact scalar height variation is `J(V)=partial_H h(V;H)>0`, with
`J(0)=1` and `J_V=R_h J`. Direct differentiation gives

\[
 \mathcal R_h={V(q-h^2k_h)\over\mathcal D^2},
\]

and, **along the actual trajectory**,

\[
 {d\over dV}\log{h\over\mathcal D}
 =\mathcal R_h-{q_V+h k_V\over\mathcal D}.
 \tag{2.1}
\]

Here `k_V` is a partial derivative at fixed height. Therefore

\[
 \boxed{
 J(V)={\mathcal D(0,H)\over H}
       {h(V)\over\mathcal D(V,h(V))}
       \exp\!\left(\int_0^V
             {q_V+h k_V\over\mathcal D}\,ds\right).
 }
 \tag{2.2}
\]

The factor `mathcal D(0,H)/H`, which may be exponentially large, is
identical for both endpoint variations and cancels from their ratio.
Bounding the variation equation before making this cancellation would
lose the estimate needed in the thin layer.

On a fixed compact normal-coordinate domain containing the physical
sections, the exact family provides

\[
 |k-1|+|k_V|\le M\epsilon,\qquad |k_h|\le M\epsilon^2,
\]
\[
 |q|\le M(\epsilon V^2+\epsilon^2|V|+\epsilon^3),
 \qquad |q_V|\le M(\epsilon|V|+\epsilon^2).
 \tag{2.3}
\]

These are the uniform analytic coefficient estimates justified in the
grazing-center proof from the exact normal change. In particular the
`k_h` term in (2.1) cancels exactly; it has not been silently discarded.

## 3. A growing core and its escape estimate

Set

\[
 \eta=H/\epsilon^3,\qquad
 K_\epsilon=K,\qquad \delta_\epsilon=\epsilon(1+K),\qquad
 U_\epsilon^2=A(1+K_\epsilon),\qquad
 V_\epsilon=\epsilon^{3/2}U_\epsilon.
 \tag{3.1}
\]

The fixed constant `A` in sections 3–5 is an auxiliary core-size constant,
distinct from the physical quadratic coefficient in section 1. It is
chosen below. Notice the crucial scales

\[
 \sqrt\epsilon U_\epsilon=\sqrt{A\delta_\epsilon},\quad
 \epsilon U_\epsilon^2=A\delta_\epsilon,\quad
 V_\epsilon=\epsilon\sqrt{A\delta_\epsilon}.
 \tag{3.2}
\]

In `V=epsilon^(3/2)u`, `h=epsilon^3w`, the slow denominator coefficient is

\[
 B_\epsilon(u)={q(\epsilon^{3/2}u)\over\epsilon^3}
 =-\lambda_0-\sqrt\epsilon\lambda_1u
                   -\epsilon u^2\zeta(\epsilon^{3/2}u,\epsilon).
\]

For `delta_epsilon` sufficiently small, (3.2) and (1.1) give, for all
`|u|<=U_epsilon`,

\[
 c/2\le B_\epsilon(u)\le B_*:=2L+1,
 \qquad 1/2\le k\le2
 \tag{3.3}
\]

after shrinking `epsilon` and the fixed permitted upper bound `delta0`.
The same estimates hold on both signs of `u`.
To make the orientation explicit, define `w_sigma(x)=w(sigma*x)` for
`x>=0`, `sigma=+-1`. The scalar equation is

\[
 {dw_\sigma\over dx}
 = {x w_\sigma\over B_\epsilon(\sigma x)
                          +k_\sigma(x,w_\sigma)w_\sigma},
 \qquad w_\sigma(0)=\eta.
 \tag{3.4}
\]

Both radial height profiles increase. Their positivity is preserved.
The upper bound in (3.4) is `w_sigma'<=2x`, so

\[
 \eta\le w_\sigma(x)\le1+x^2.
 \tag{3.5}
\]

In physical coordinates this stays in a shrinking fixed-coordinate
neighborhood, since `epsilon^3(1+U_epsilon^2)->0`. Thus the bounds used
to continue the scalar equation are valid on its whole interval. The
normal `V` velocity remains negative; (3.4) on the positive side is a
backward-time construction and on the negative side a forward-time
construction of the same normal trajectory.

For the lower bound define

\[
 \Phi_\eta(w)=B_*\log(w/\eta)+2(w-\eta).
\]

Using (3.3)–(3.4),

\[
 {d\over dx}\Phi_\eta(w_\sigma(x))
 =x{B_*+2w_\sigma\over B_\epsilon(\sigma x)+k_\sigma w_\sigma}
 \ge x.
\]

It follows, with the correct increasing radial orientation on both sides,
that

\[
 \Phi_\eta(w_\sigma(x))\ge x^2/2.
 \tag{3.6}
\]

Start, for example, with `A_base=256(1+B_*)^2`, then increase `A` by a
fixed amount if needed for the tail denominator bound in section 4.
Every `A>=A_base` satisfies `A>=16`, `A>=16B_*`, and
`A>=8B_* log A`. At `A_base`, the last inequality follows from
`log A_base<6+2B_*` and
`A_base/(8B_*)=32(B_*+2+1/B_*)`; it persists for larger `A` because
`A-8B_* log A` is increasing once `A>=16B_*`.
Since `log(1/eta)<=K_epsilon`, supposing
`w_sigma(U_epsilon)<=U_epsilon^2/16` would give

\[
 \begin{split}
 \Phi_\eta(w_\sigma(U_\epsilon))
 &\le B_*\{\log(U_\epsilon^2/16)+K_\epsilon\}
                                      +U_\epsilon^2/8\\
 &\le B_*\{\log A+2K_\epsilon\}+U_\epsilon^2/8\\
 &\le U_\epsilon^2/4,
 \end{split}
\]

contradicting (3.6). Together with (3.5), this proves the uniform escape
height bound

\[
 U_\epsilon^2/16<w_\sigma(U_\epsilon)\le2U_\epsilon^2.
 \tag{3.7}
\]

No assumption about a guessed first-order section threshold was used.

## 4. Fast tails and actual physical-section existence

Choose a fixed `Vmax` beyond both selected limiting outer sections, and
a fixed compact height domain containing `0<=h<=5Vmax^2+1`. At
`|V|=V_epsilon`, (3.7) gives `V^2/16<h<=2V^2`.

On each tail bootstrap

\[
 V^2/32\le h\le4V^2.
\]

By (2.3),

\[
 |k-1|+{|q|\over h}
 \le C\left(\epsilon+{\epsilon^2\over |V|}
                         +{\epsilon^3\over V^2}\right)
 \le C\left(\epsilon+{\sqrt\epsilon\over U_\epsilon}
                         +{1\over U_\epsilon^2}\right)
 \le C\left(\epsilon+{\sqrt\epsilon\over\sqrt A}+{1\over A}\right).
 \tag{4.1}
\]

The last term need not tend to zero when `K_epsilon` is bounded. Choose
the fixed `A` large enough that `C/A<=1/32`, and then choose `epsilon`
small enough that the other two terms total at most `1/32`. This
bound is strictly below `1/8`. Hence `h/mathcal D` lies
between `8/9` and `8/7`. In the radial variable `x=|V|`, the derivative
is again `h_x=x*h/mathcal D`. Integration improves the bootstrap to
`V^2/16<h<=2V^2` and continues the trajectory to `Vmax`. In particular
`mathcal D>0`, and the trajectory reaches both actual outer sections in
the appropriate time directions.

For clarity, convergence of these intersections is also controlled.
On the tails,

\[
 |\mathcal R-V|\le C\left(
 \epsilon|V|+\epsilon^2+{\epsilon^3\over |V|}\right).
\]

Integration, using the core bound `h(V_epsilon)=O(V_epsilon^2)`, gives

\[
 h(V)-V^2/2
 =O\!\left(V_\epsilon^2+\epsilon V^2+\epsilon^2|V|
               +\epsilon^3\log(V_{\max}/V_\epsilon)\right)
 \tag{4.2}
\]

near the outer sections. This is `O(epsilon)` there, because
`V_epsilon^2=A epsilon^2 delta_epsilon` and `delta_epsilon<=delta0`.
The slope there is `R=V+O(epsilon)`. The exact outer section equations

\[
 E_\sigma=V-1+\sigma\rho h
             +\nu\rho^2h^2+C\nu^2\sigma\rho h^2=0
 \tag{4.3}
\]

have limiting transversality `1+sigma rho V=-sqrt(1+2sigma rho)`,
uniformly separated from zero. Their unique nearby outer intersections
therefore exist and differ by `O(epsilon)` from the zero-fiber endpoints.
The statement refers to these specified outer local sections, not to an
earlier crossing of another portion of the same geometric line.

## 5. Quantitative bound on the compensated variation

Abbreviate the exponent in (2.2) as

\[
 \Psi(V)=\int_0^V{q_V+h k_V\over\mathcal D}\,ds.
\]

In the growing core, `mathcal D>=c epsilon^3/2` and `h/mathcal D<=2`.
Equations (2.3) give

\[
 \begin{split}
 \int_{|V|\le V_\epsilon}
 { |q_V|\over\mathcal D}\,dV
 &\le C\left({V_\epsilon\over\epsilon}
                         +{V_\epsilon^2\over\epsilon^2}\right)\\
 &\le C\left(\sqrt{\delta_\epsilon}+\delta_\epsilon\right).
 \end{split}
 \tag{5.1}
\]

On the tails `mathcal D>=c_1 V^2`, so

\[
 \int_{V_\epsilon\le|V|\le V_{\max}}
 { |q_V|\over\mathcal D}\,dV
 \le C\left({\epsilon^2\over V_\epsilon}
                   +\epsilon\log(V_{\max}/V_\epsilon)\right)
 \le C\left({\sqrt\epsilon\over U_\epsilon}
                   +\epsilon\log(1/\epsilon)\right).
 \tag{5.2}
\]

Throughout the passage, `h/mathcal D` is uniformly bounded and
`k_V=O(epsilon)`. The remaining integral is therefore `O(epsilon)`.
Combining (5.1)–(5.2), for each endpoint,

\[
 |\Psi_i|+|\Psi_o|
 \le C\left\{\sqrt{\delta_\epsilon}+\delta_\epsilon
         +{\sqrt\epsilon\over U_\epsilon}
         +\epsilon\log(1/\epsilon)\right\}
 \le C\sqrt{\delta_\epsilon}.
 \tag{5.3}
\]

Indeed `U_epsilon>=sqrt(A)`, `delta_epsilon>=epsilon`, and
`epsilon log(1/epsilon)=O(sqrt(epsilon))`. Once `delta0<=1`, every
displayed term is bounded by a constant times `sqrt(delta_epsilon)`.
The logarithmic tail bound is uniform because `U_epsilon>=sqrt(A)` and
`V_epsilon` stays below the fixed outer section coordinates.

There is no `delta_epsilon*log(1/epsilon)` term in these direct estimates.
Such a term could not in general be absorbed for arbitrary `K_epsilon`,
and must not be inserted into the claimed general bound.

The crucial growing-core contribution is proportional to
`sqrt(epsilon)*U_epsilon`, not `sqrt(epsilon)*U_epsilon^3`.
The latter can appear in an unreduced escape-location expansion, but it
does not control the endpoint derivative ratio after the exact identity
(2.2) cancels the common variation factor.

## 6. Endpoint variation factors and the derivative estimate

On each exact physical section, set

\[
 T_\sigma(h)=(\sigma\rho h-1)^2-2h,
 \qquad
 K_\sigma={T_\sigma'(h_\sigma)\over
                   1+(E_\sigma)_h\mathcal R_\sigma}.
\]

The exact moving-endpoint formula is

\[
 {dt_\sigma\over dH}=K_\sigma J_\sigma.
 \tag{6.1}
\]

By (4.2)–(4.3), `K_sigma=-2+O(epsilon)`, uniformly. Also
`h_sigma/mathcal D_sigma=1+O(epsilon)` at the fixed outer endpoints.
Thus (2.2) gives the **exact cancellation formula**

\[
 D_\epsilon'
 ={K_o\over K_i}
  {{h_o/\mathcal D_o}\over{h_i/\mathcal D_i}}
       \exp(\Psi_o-\Psi_i).
 \tag{6.2}
\]

The potentially exponential factor `mathcal D(0,H)/H` is absent from
(6.2). Its remaining endpoint factors are positive and differ from one
by `O(epsilon)`. Equation (5.3), the inequality
`|exp(z)-1|<=exp(|z|)|z|`, and `delta<=delta0<=1` prove (1.3) with one
fixed constant.

The order of choices is explicit. First fix the coefficient domain and
its bounds (2.3), then the auxiliary `A` large enough for both the escape
and tail inequalities. Next choose `delta0<=1` small enough for (3.3).
Finally choose `epsilon0` small enough for the tail denominator, chart
domain, and physical endpoint inequalities. Since
`V_epsilon<=epsilon sqrt(A delta0)`, the growing core remains before
the physical sections uniformly for every admissible `K`, including
bounded `K`. The final variation constant is fixed on these choices.

For every positive `epsilon`, the section maps are smooth and
`dt_i/dH<0`, `dt_o/dH<0`. The center-height interval therefore maps
monotonically to a single signed input interval, regardless of how small
its width is. The calculations above concern actual derivatives on that
interval; no value-only error term has been differentiated.

## 7. Combining with the upper band and the cycle gate

The previous adaptive-core theorem applies uniformly on
`epsilon^3<=H<=Hmax` by choosing its fixed lower factor `gamma=1`.
It has the same actual section definitions and the same exact center
parametrization. The two intervals meet at `H=epsilon^3`; together they
form one connected center-height band. Their derivative estimates prove
(1.4) on that whole band.

The captured regular map, transported by the exact signed-section
coordinate maps, has an interval as its admitted input domain and
`0<H_reg'<=theta<1`. Intersecting this domain with the connected singular
input interval preserves connectedness. For any function `K_epsilon`
as in (1.4), and sufficiently small `epsilon`,

\[
 (H_{\rm reg}-D_\epsilon)'\le-(1-\theta)/2<0.
\]

The physical cycle equation is `H_reg(t)=D_epsilon(t)` because the
physical singular closing map is the inverse of the normal-forward
passage used here. Thus at most one captured cycle lies in the combined
band. Any such cycle is hyperbolic and attracting, since its physical
return multiplier is `H_reg'/D_epsilon'` strictly between zero and one.

There is also a fixed exponential-sector consequence of the quantitative
gate, without computing an exponential transition formula. Choose a fixed
`kappa_*>0` and `0<epsilon_*<=epsilon0` such that

\[
 \epsilon_*+\kappa_*\le\delta_0,\qquad
 C\sqrt{\epsilon_*+\kappa_*}<(1-\theta)/2.
\]

Take `K=kappa_*/epsilon`. For `epsilon<=epsilon_*`, the thin-band
derivative satisfies `D_epsilon'>(1+theta)/2` on

\[
 \epsilon^3\exp(-\kappa_*/\epsilon)\le H\le\epsilon^3.
\]

Shrink `epsilon_*` once more for the previously proved upper-band
derivative bound. The same strict comparison then gives **at most one
captured cycle on the connected band**

\[
 \epsilon^3\exp(-\kappa_*/\epsilon)\le H\le H_{\max}.
 \tag{7.1}
\]

The constant `kappa_*` depends on the compact coefficient bounds and
the regular contraction margin. Formula (7.1) asserts a coarse derivative
gate for an initial exponential sector; it does not assert convergence
`D_epsilon'->1` at a fixed nonzero `kappa` or identify its limiting slope.

As a rate corollary, choosing `K_epsilon=epsilon^-p` for any fixed
`0<p<1` recovers `D_epsilon'=1+O(epsilon^((1-p)/2))` on the added
thin portion. The general theorem also allows non-power choices such as
`K_epsilon=1/[epsilon*log(1/epsilon)]`.

All original regular capture and section-itinerary assumptions remain
in force. This theorem does not cover arbitrary nearby cycles or assign
an all-degree Hilbert bound.

## 8. The next scale is not a claimed result here

At `H=epsilon^3 exp(-kappa/epsilon)` with fixed `kappa>0`, the above
core length reaches `U=O(epsilon^-1/2)` and (5.1) no longer tends to
zero. It still supplies the small-`kappa_*` comparison in (7.1), but
does not describe the nontrivial slope transition beyond that gate.
A possible intermediate description uses `x=|V|/epsilon` and
`B_sigma(x)=-lambda0-sigma lambda1 x+x^2`, with escape positions
suggested by

\[
 \int_0^{x_\sigma}{s\over B_\sigma(s)}\,ds=\kappa.
\]

The compensated variation suggests a nontrivial endpoint ratio involving
the two `B_sigma` values. Establishing that description requires a
uniform escape/height limit and physical endpoint matching at this
different scale. It is an investigation target, not a proved theorem
or a premise of (1.3).

Primary normal-form context: [Huzak–Kristiansen, section 6](https://documentserver.uhasselt.be/bitstream/1942/49936/1/On_entry_exit_formulas_for_degenerate_turning_point_problems_in_planar_slow_fast_systems.pdf).
The present proof uses the exact family and scalar continuation directly;
it does not invoke the published entry-exit theorem at its zero-fiber
boundary.

The finite identities in sections 2, 3 and 6 are independently checked by
`/tmp/omnibias-hilbert16-subexponential-checks.py`, with output in
`/tmp/omnibias-hilbert16-subexponential-checks.json`. All eight residuals
vanish normally and under `python -O`; the analytic inequalities and limits
are not formalized by those checks.

## Finite checks and reproduction

The [thin-passage benchmark](../../benchmarks/hilbert16_thin_passage.py)
replays 17 exact identities, including the compensated variation, drift
envelope, moving physical endpoints, radial escape inequality, and the
cancellation of the earlier symmetric-window term. Rational examples
check the sufficient escape constants, scale exponents, and a declared
slope margin.

```sh
uv run --no-sync python benchmarks/hilbert16_thin_passage.py \
  --output artifacts/hilbert16/thin_passage.json
uv run --no-sync python -O benchmarks/hilbert16_thin_passage.py \
  --output artifacts/hilbert16/thin_passage_optimized.json
```

The algebra replay does not verify the analytic continuation or uniform
limit arguments. Actual physical cutoff constants are not computed as
interval certificates, and no formal analytic verification flag is earned.
