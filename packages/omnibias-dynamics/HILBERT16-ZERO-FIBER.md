# The zero-fiber boundary: a signed-label obstruction and its inner scales

This is a written analytic derivation, supported by finite symbolic checks and
ordinary floating-point diagnostics. It is not a formally verified theorem, an
independent novelty claim, or a proof of full graphic cyclicity or Hilbert 16.

## 1. Result and scope

The positive fast-fiber coordinate used in the
[interior cyclicity argument](HILBERT16-INTERIOR-CYCLICITY.md) does
**not** extend uniformly to zero. For the actual moving physical sections, the
normal-forward trajectory starting at the old label `B_in=0` has an outgoing
**signed squared** label

\[
 t_{\rm out}=\nu\kappa_{r,\rho}+o(\nu),\qquad \kappa_{r,\rho}<0.
\]

Thus its real positive output label `B_out=sqrt(t_out)` does not exist. This is
stronger than a failure of a relative derivative estimate: the positive-label
map is not defined at that endpoint. The trajectory and its transverse section
intersection exist. It is the positive square-root chart that fails.

The result holds for fixed sufficiently small `rho>0`, `r` in a compact subset
of the negative real axis, `C` in a compact set, `A=1+O(nu)`, and the exact
canonical slow-parameter embedding of
[singular transport](HILBERT16-SINGULAR-TRANSPORT.md), with
`(lambda0,lambda1)` in a bounded set. A strict discriminant is not needed
for this rapid trajectory. The limit order is
fixed `rho`, then `nu -> 0`. Constants may depend on `rho`.

For `r=-1`, the old `B_in=0` point lies strictly inside the
[rapid-passage corridor](HILBERT16-RAPID-PASSAGE.md)
in the new, signed family coordinate. An explicit positive margin is computed
in section 6. This gives a concrete boundary sector to analyze without a
positive square root.

The primary entry-exit theorem does not assert an extension across this
boundary. In Huzak–Kristiansen, Theorem 2.4 uses compact incoming and outgoing
base-point intervals away from zero; in section 6, the incoming interval is
strictly inside the endpoints in (6.12). See the
[published manuscript, Theorem 2.4 and equations (6.8)–(6.12)](https://documentserver.uhasselt.be/bitstream/1942/49936/1/On_entry_exit_formulas_for_degenerate_turning_point_problems_in_planar_slow_fast_systems.pdf).
The boundary calculation below is our derivation for the actual family.

## 2. Actual coordinates and the first variation

Use the same family chart as [singular transport](HILBERT16-SINGULAR-TRANSPORT.md):

\[
\begin{split}
 F&=m+pv-\nu v^3-h+A\nu vh-C\nu^2v^2h,\\
 N&=r+v+\nu v^2+C\nu^2vh,\qquad V=-N,\\
 \ell&=1+2\nu v+C\nu^2h.
\end{split}
\]

After reversing physical time, the exact normal system is

\[
 \dot V=\ell F+C\nu^2vVh,\qquad \dot h=-Vh.
\]

The canonical embedding has
`p=3 r^2 nu+O(nu^2)` and `m=2 r^3 nu+O(nu^2)`. On every fixed compact
`(V,h)` set, uniformly in the admitted passive parameters,

\[
 \dot V=-h+\nu\{V^3+3rV^2+(2-A)(r+V)h\}+O(\nu^2).
 \tag{2.1}
\]

In the coefficient of `nu`, `A` may be replaced by its limit. In particular,
for `A=1+O(nu)` it is `1`. The coefficient `C` first enters at higher order.
The estimate is an expansion on fixed compact sets; it alone does not justify
division by `h` near the turning point. Section 4 supplies that missing bound.

Set `I=V^2-2h`. Whenever `V` decreases monotonically,

\[
 {dI\over dV}=2V-{2Vh\over-\dot V}.
\]

Along the limiting zero fast fiber `h=V^2/2`, the first variation is

\[
 {1\over\nu}{dI\over dV}\longrightarrow
 L_A(V)=-2(4-A)V^2-2(8-A)rV.
 \tag{2.2}
\]

One primitive is

\[
 p_A(V)=-{2\over3}(4-A)V^3-(8-A)rV^2.
 \tag{2.3}
\]

For `A -> 1`, abbreviate `p(V)=-2V^3-7rV^2`.

## 3. The moving physical sections contribute at first order

Fix `x=sigma R`, `R=rho/nu`, `sigma=+-1`, and the old regular coordinate
`z=w/q`. Write

\[
 Y_\sigma={\rho^2(1+z)\over2}
       +\sigma\nu\rho z+{C\nu^2(z-1)\over2},\qquad
 v={\sigma\rho\over Y_\sigma},\quad h={1\over Y_\sigma}.
\]

The old signed label `t=B^2` is defined by setting `z=Z_sigma(t)`, where

\[
 Z_\sigma(t)={2\over1-\sigma r\rho+
                  \sqrt{1-2\sigma r\rho+\rho^2t}}-1.
\]

This formula is analytic also for sufficiently small negative `t`; the
positive square root of `t` is unnecessary. Denote the exact section curve
by `Gamma_sigma,nu(t)`. At `t=0,nu=0`,

\[
 V_i={1+\sqrt{1+2r\rho}\over\rho}>0,\qquad
 V_o=-{1+\sqrt{1-2r\rho}\over\rho}<0,\qquad h=V^2/2.
 \tag{3.1}
\]

At either endpoint the first-order expansion is

\[
 I(\Gamma_{\sigma,\nu}(0))=\nu e_\sigma(V)+O(\nu^2),
\]

\[
 e_\sigma(V)=2V^3-\rho^2V^5+
                {2\sigma V^2\over\rho}-{\sigma\rho V^4\over2}.
 \tag{3.2}
\]

For example, expand `Y=Y0+nu Y1`, with `Y0=2/V^2` and
`Y1=sigma rho z`. Then `v1=-v0 Y1/Y0`, `h1=-Y1/Y0^2`,
`V1=-v1-v0^2`, and `e=2V V1-2h1`, giving (3.2).

The endpoint relation `r+V+sigma rho V^2/2=0` simplifies the difference
between section transport and flow transport to

\[
 E(V):=e_\sigma(V)-p(V)
 ={rV(V^2-4rV-4r^2)\over V+r}
 =rV^2-5r^2V+r^3-{r^4\over V+r}.
 \tag{3.3}
\]

The dependence on `sigma` is through the endpoint `V`.

## 4. Uniform central-height bound and legitimate integration

This supplies an analytic justification of (2.2) through `V=0`; merely
integrating the formal first variation would be insufficient.

First `E(V_i)<0`. Put `u=-r rho`, so `0<u<1/2`, and
`V_i/|r|=(1+sqrt(1-2u))/u>2`. In (3.3), the numerator's quadratic factor
is positive and `V_i+r>0`, whereas `r<0`. On a compact parameter set this
negative value has a uniform margin for fixed `rho`.

Choose a small fixed `d>0`, less than a fixed fraction of `min |r|`, such
that `E(V_i)+p(d)` stays uniformly negative. On the incoming compact portion
`V_i -> d`, the limiting height is at least `d^2/2`; ordinary smooth ODE
dependence applies. Equations (2.2)–(3.3) give

\[
 I(d)=\nu\{E(V_i)+p(d)\}+o(\nu),\qquad
 h(d)\ge d^2/2+c\nu
 \tag{4.1}
\]

for some uniform `c>0` and all sufficiently small `nu`.

The canonical normal form writes

\[
 \dot V=s(V,\nu)+h g(V,h,\nu),
\]

where on `|V|<=d` the exact canonical embedding, with bounded `lambda`, yields

\[
 |s(V,\nu)|\le C_1\nu(V^2+\nu^2),\qquad
 s(V,\nu)\le-c_1\nu V^2+C_1\nu^3.
 \tag{4.2}
\]

Indeed `s=nu(V^3+3rV^2)+O(nu^2 |V|)+O(nu^3)`. Absorb the cubic term
by choosing `d` small and the linear error by the elementary quadratic
inequality. Also (2.1), `A=1+O(nu)`, and `r+V<0` give

\[
 g=-1+\nu(r+V)+O(\nu^2)\le-1-c_2\nu,
 \qquad |g+1|\le C_2\nu.
 \tag{4.3}
\]

These bounds hold uniformly on a fixed height interval containing the
trajectory. Its existence there follows by the continuation estimates below.
Put `D=-s-hg`. Whenever `h>=c nu/2`,

\[
 D-h\ge c_2\nu h+c_1\nu V^2-C_1\nu^3\ge0
\]

for sufficiently small `nu`, so `V` decreases, and

\[
 h_V={Vh\over D},\qquad D\ge h,
 \qquad D\le2h+C_1\nu(V^2+\nu^2).
\]

On the incoming half, bootstrap `h>=V^2/2+c nu/2`. It holds strictly
at `V=d`. In this region `D>=h`, so integrating `h_V<=V` backwards
from (4.1) improves it to `h(V)>=V^2/2+c nu`. Continuation closes the
bootstrap down to zero. Hence `h(0)>=c nu`. For `-d<=V<=0`, the height
increases as `V` decreases, so `h>=c nu`; furthermore

\[
 {h\over D}\ge
 {1\over2+(C_1/c)(V^2+\nu^2)}\ge\gamma>0.
\]

Integration yields `h(V)>=c nu+gamma V^2/2`. On both halves,

\[
 h(V)\ge c_3(\nu+V^2).
 \tag{4.4}
\]

The upper derivative bound `|h_V|<=|V|` also keeps the height in the chosen
compact interval. No loss of existence or monotonicity occurs in this core.

Now the **exact** derivative is

\[
 {1\over\nu}I_V
 ={2V\{h(-g-1)/\nu-s/\nu\}\over D}.
\]

Using (4.2)–(4.4), its modulus is at most a uniform constant times `|V|`.
Consequently `I=O(nu)` across the core, `h(V)->V^2/2`, and for each
nonzero `V` the derivative converges to (2.2). Dominated convergence proves
the integrated first variation through zero. Smooth dependence on the
remaining outgoing compact portion completes the passage to the right
section. All the estimates are uniform over the stated compact parameter
set, with `rho` fixed.

For completeness, the limiting right section is `V+r+rho h=0`; its
derivative along the limiting `V`-parametrized trajectory at `V_o` is
`1+rho V_o=-sqrt(1-2r rho)`, bounded away from zero. Thus the endpoint
intersection persists uniquely. Both its signed label and its section
coordinate are `O(nu)` from their zero-fiber values.

## 5. Exact negative shift on the actual sections

The preceding calculation gives

\[
 I_{\rm out}=\nu\{e_-(V_i)+p(V_o)-p(V_i)\}+o(\nu).
\]

On the outgoing section, `I=t+nu e_+(V_o)+o(nu)` for `t=O(nu)`. Hence

\[
 \boxed{t_{\rm out}=\nu\kappa_{r,\rho}+o(\nu),\quad
        \kappa_{r,\rho}=E(V_i)-E(V_o).}
 \tag{5.1}
\]

The sign is exact. Put `a=sqrt(1+2r rho)` and `b=sqrt(1-2r rho)`.
Algebra using `a^2+b^2=2` gives

\[
 \kappa_{r,\rho}=
 {r^2(2+a+b)\over\rho}\left({4\over a+b}-5\right)
 -2r^4\rho\left({1\over(1+a)^2}+{1\over(1+b)^2}\right)<0.
 \tag{5.2}
\]

Here `sqrt(2)<a+b<=2`, so both displayed contributions are negative.
For small `rho`,

\[
 \kappa_{r,\rho}=-{12r^2\over\rho}+6r^4\rho+O(\rho^3).
\]

Therefore `S_nu(B)/B ~ a/(1-bB)` cannot hold uniformly down to `B=0`
in the old positive-fiber chart, even at the level of existence or relative
`C^0` control. This does not contradict compact-interior convergence.

At `r=-1,rho=0.1`, (5.2) equals `-119.3913074383266`. Ordinary
double-precision integrations of the exactly embedded family, with
`A=1,C=2,lambda0=-1,lambda1=0`, give `t_out/nu=-119.76597` at `nu=1e-5`
and `-119.39508` at `nu=1e-7`, approaching the coefficient. These are
diagnostics, not validated integration certificates.

## 6. Relation to the new signed family label and rapid corridor

Set `r=-1`. The fixed family sections are `v=sigma rho h`. A more useful
label on these sections is

\[
 \tau_{\rm family}=(v-1)^2-2h.
\]

It is independent of `nu` as a function on the family chart; it differs from
both `I=V^2-2h` and the old physical label `B^2` at first order.
Let `V_i=1-v_{i,0}>2`. The old incoming point `B=0` satisfies

\[
 \tau_{\rm family}=\nu\eta_i+O(\nu^2),\qquad
 \eta_i={V_i(2V_i-1)(V_i-2)\over V_i-1}
       =2V_i^2-3V_i-{V_i\over V_i-1}.
 \tag{6.1}
\]

The rapid-passage threshold derived in the family `v` equation is

\[
 \tau_c=(3v_{i,0}-5)(v_{i,0}-1)=3V_i^2+2V_i.
\]

Consequently the exact positive gap is

\[
 \boxed{\tau_c-\eta_i=V_i^2+5V_i+{V_i\over V_i-1}>0.}
 \tag{6.2}
\]

It also equals `-E(V_i)` at `r=-1`, so the central-height mechanism is
identical in the two coordinate descriptions. This verifies that the old
zero endpoint belongs to a rapid-passage sector with a strict first-order
margin; its behavior is not the limiting delayed passage evaluated at zero.

The [rapid-passage proof](HILBERT16-RAPID-PASSAGE.md) establishes its uniform
profile and first-derivative estimates, then combines the passage with the
captured regular map to obtain at most one cycle on a connected admitted
rapid corridor. This note supplies the old zero endpoint's strict margin
inside that corridor and the failure of its positive square-root chart.
The grazing transition between the rapid and delayed sectors remains
unresolved.

## 7. Nested inner scales, after first-order recentering

The order-`nu` signed-label shift first creates a scale `B^2=O(nu)`.
It must be recentered before treating the deepest grazing core. In the exact
canonical normal form

\[
 \dot V=\epsilon[\epsilon^2\lambda_0+
          \epsilon\lambda_1V+V^2\zeta(V,\epsilon)]+hg,
 \quad\dot h=-Vh,
\]

the innermost balance of the constant slow drift with the transverse term is

\[
 V=\epsilon^{3/2}u,\qquad h=\epsilon^3w,
 \qquad\tau=\epsilon^{3/2}t.
\]

On compact `(u,w)` sets this gives

\[
 {du\over d\tau}=\lambda_0-w+
       \epsilon^{1/2}\lambda_1u-\epsilon u^2
       +O(\epsilon w)+O(\epsilon^{5/2}u^3),\qquad
 {dw\over d\tau}=-uw.
\]

Higher coefficient corrections are uniformly smaller on these compact sets.
The leading core has the exact first integral

\[
 J=u^2-2w+2\lambda_0\log w,\qquad w>0.
\]

Since strict discriminant implies `lambda0<0`, its `u` derivative in time is
strictly negative throughout `w>=0`. The symmetric transition between
`u=U` and `u=-U`, where defined, preserves `w`; equivalently, on a common
height it reverses the sign of `u`. This follows from the first integral
because `w -> -2w+2lambda0 log w` is strictly decreasing.

This integrable core does not yield the Mobius multiplier of the delayed
entry-exit map: the `lambda1` contribution is suppressed at this scale. The
intermediate scaling `V=epsilon u,h=epsilon^2w,tau=epsilon t` instead gives
`u'=-w+epsilon(lambda0+lambda1u-u^2)+...`, `w'=-uw`, and itself becomes
singular near `w=0`. A proof covering the threshold must match these layers
and their parameter derivatives. Neither the fixed-interior theorem nor the
rapid-corridor estimate alone supplies that matching.

## 8. Reproducible finite checks

The [benchmark](../../benchmarks/hilbert16_zero_fiber.py) verifies eleven
exact identities with explicit failure checks. Its default output contains
only algebra checks; the optional `--diagnostics` flag also runs ordinary
numerical trajectories using the exact canonical parameter embedding.
Those diagnostics remain nonvalidated and do not replace the analytic proof.

```bash
python benchmarks/hilbert16_zero_fiber.py --output artifacts/hilbert16/zero_fiber.json
python -O benchmarks/hilbert16_zero_fiber.py --output artifacts/hilbert16/zero_fiber_optimized.json
```

The [normal report](../../artifacts/hilbert16/zero_fiber.json) and
[optimized report](../../artifacts/hilbert16/zero_fiber_optimized.json) record
the same finite identities. These checks do not formally verify the analytic
limits, the domain inequalities, or a cyclicity theorem.
