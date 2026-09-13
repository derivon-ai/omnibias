# An actual positive-label anchor for the joint passage

This note proves a value/coordinate anchor for the actual joint passage,
using the existing exact normal field and selected outer physical sections.
It does not invoke a sharper regular multiplier estimate, differentiate a
value remainder, or add cycle counts from separate intervals. This written
analytic proof is not formally verified; no novelty or full-graphic claim
is made.

## 1. Statement

Fix the same compact coefficient set as the [joint matching proof](HILBERT16-JOINT-MATCHING.md), with
`L=-lambda0>0`, bounded `lambda=lambda1`, and
`Delta=4L-lambda^2>=chi>0`. The weighted chart parameter is `r=-1`.
The physical section size `rho>0` is fixed first. Write

\[
 \omega=e^{-\kappa},\qquad u=\epsilon/\omega,
 \quad\epsilon=u\omega,\qquad
 H=\epsilon^3e^{-\kappa/\epsilon},\qquad\beta_0=1/3,
\]

and use the already defined positive compact constants

\[
 d_\sigma=\sqrt L\exp\!\left[
 -{\lambda\over\sqrt\Delta}\arctan{\lambda\over\sqrt\Delta}
 -{\sigma\pi\lambda\over2\sqrt\Delta}\right].
\]

Let `t_sigma` denote the actual signed label at the outer physical endpoint
with radial sign sigma. Thus `t_+` is the normal-forward input and `t_-`
is its output. Their physical section signs are respectively `tau=-1,+1`.
For sufficiently small independent upper cutoffs on omega and u,

\[
 \boxed{
 t_\sigma=
 \left({u d_\sigma\over1+\sigma\beta_0 d_\sigma u}\right)^2
 +O_\rho\!\left(\epsilon[1+u^2\log(1/\epsilon)]\right).}
 \tag{1}
\]

The constants are uniform over the stated compacts. In particular,

\[
 t_\sigma=d_\sigma^2u^2-2\sigma\beta_0d_\sigma^3u^3
 +O_\rho\!\left(u^4+\epsilon[1+u^2\log(1/\epsilon)]\right).
 \tag{2}
\]

This is slightly stronger than retaining an additional `O(u^2*omega)` term,
because that term is at most `O(epsilon)` for u below 1.

For every fixed sufficiently small `0<u_a<u_b`, convergence in (1) is
uniform for `u in [u_a,u_b]` as epsilon tends to zero with
`omega=epsilon/u`. The limiting input and output labels are strictly
positive and stay in a fixed compact interval. This supplies a common
actual orbit interval with the positive-base interior passage.

The estimate does not assert positivity everywhere in the joint corner:
when `u^2` is comparable to epsilon, the physical section correction in
(1) may dominate. Positivity follows uniformly whenever
`omega/u=epsilon/u^2->0`, and in particular on every fixed positive u band.

## 2. Exact transport of the squared radial label

In the exact variables from the joint proof,

\[
 V=\sigma uX,\quad h=\epsilon u^2Y,
 \quad \epsilon Y_X={XY\over b_\sigma+\widehat k_\sigma Y},
 \quad Y(0)=\omega^2e^{-\kappa/\epsilon}.
\]

On a fixed normalized interval `0<=X<=K`, the actual denominator is positive,
`b_sigma` is comparable to `omega^2+X^2`, `1/2<=khat<=2`, and
`|khat-1|<=C*epsilon`. Define

\[
 W(X)=X^2/2-\epsilon Y(X).
\]

Its derivative is exactly

\[
 W_X={X[b_\sigma+(\widehat k_\sigma-1)Y]
                                      \over b_\sigma+\widehat k_\sigma Y}.
 \tag{3}
\]

Let `X_1` be the first level `Y=omega^2`, and `X_2` the first level `Y=1`.
The existing two-level proof supplies common positive bounds for X1,
`X_2-X_1<=C*epsilon*kappa`, and a fixed positive gap between X2 and K.
On `[X_1,X_2]` the right side of (3) is uniformly bounded. On `[X_2,K]`,
the same proof gives

\[
 Y(X)\ge1+c(X-X_2)/\epsilon.
\]

Using bounded X and b, (3) therefore yields

\[
 \left|W(K)-W(X_1)\right|
 \le C[\epsilon\kappa+\epsilon\log(1/\epsilon)+\epsilon]
 \le C\epsilon\log(1/\epsilon).
 \tag{4}
\]

Here `kappa=log(1/omega)<=log(1/epsilon)` because u is below 1.
At the first level `W(X_1)=X_1^2/2-epsilon*omega^2`. Consequently at
the physical radial cut `|V|=Ku`,

\[
 V^2-2h=u^2X_1^2+O(u^2\epsilon\log(1/\epsilon)).
 \tag{5}
\]

This uses the actual height-dependent field, without freezing k at its limit.

## 3. The physical tails and the exact signed sections

Write `s=|V|`, and retain `q=-f`, `k=-g`. The joint passage proof establishes
`h` comparable to `s^2` from `s=Ku` to the selected physical outer sections,
and `q+k*h>0`. The exact radial equation gives

\[
 {d\over ds}(V^2-2h)
 ={2s[q+(k-1)h]\over q+kh}.
 \tag{6}
\]

On the fixed physical chart,

\[
 |q|\le C(\epsilon^3+\epsilon^2s+\epsilon s^2),
 \qquad |k-1|\le C\epsilon.
\]

The absolute value of (6) is therefore bounded by

\[
 C(\epsilon s+\epsilon^2+\epsilon^3/s).
\]

Integrating from Ku to a fixed compact physical endpoint gives `O_rho(epsilon)`.
In particular, `epsilon^3*log(1/u)` is harmless because
`u>=epsilon` and `epsilon^2*log(1/epsilon)` is bounded. This estimate
includes either radial orientation; it bounds absolute tail changes.

The exact normal-coordinate relation is

\[
 V=1-v-\nu v^2-C\nu^2vh,
 \qquad t=(v-1)^2-2h.
\]

Set `A_shift=nu*v^2+C*nu^2*v*h`. Then

\[
 t-(V^2-2h)=2VA_{\rm shift}+A_{\rm shift}^2.
 \tag{7}
\]

The selected physical endpoints stay in a fixed compact chart, so (7) is
`O_rho(nu)=O_rho(epsilon)`. The actual event location is used throughout;
no fixed-V point is silently substituted for the physical section.
Combining (5)–(7) proves

\[
 \boxed{t_\sigma=u^2X_1^2
                  +O_\rho(\epsilon+u^2\epsilon\log(1/\epsilon)).}
 \tag{8}
\]

## 4. Keeping the cubic phase exactly gives a rational anchor

The exact canonical normalization implies

\[
 \zeta(V,\epsilon)=-1+\beta_0V+\epsilon V Z(V,\epsilon),
\]

with Z bounded on the chosen chart. Hence, putting
`Q_sigma=X^2-sigma*lambda*omega*X+L*omega^2`,

\[
 b_\sigma=Q_\sigma-\sigma\beta_0uX^3+O(\epsilon uX^3).
\]

Choose the common u cutoff so that `beta0*u*K<=1/2`. Compare the actual
phase with the limiting quadratic primitive:

\[
 S_{\epsilon,\sigma}(X)-S_\sigma(X/\omega)
 =\sigma\beta_0u\int_0^X{s^4\over Q_\sigma(s)b_\sigma(s)}\,ds
                                      +O(\epsilon u).
 \tag{9}
\]

There is a uniform pointwise estimate

\[
 \left|{s^4\over Q_\sigma(s)b_\sigma(s)}
                   -{1\over1-\sigma\beta_0us}\right|
 \le C\left({\omega\over\omega+s}+\epsilon us\right).
 \tag{10}
\]

To verify it, write `Q_sigma=s^2+E_Q`, where
`|E_Q|<=C*omega*(omega+s)`. The numerator of the difference in (10)
is bounded by
`C[|E_Q|*s^2+E_Q^2+epsilon*u*s^3*Q_sigma]`, and both quadratics are
comparable to `(omega+s)^2`. Also `1-sigma*beta0*u*s>=1/2`.
Division yields (10), including the origin `s=0` by continuity.

Integrating and using the already proved quadratic primitive gives, uniformly
for fixed `a<=X<=K`,

\[
 S_{\epsilon,\sigma}(X)-\kappa
 =\log X+c_\sigma-\log(1-\sigma\beta_0uX)
   +O\!\left(\omega[1+u\log(1/\omega)]\right),
 \tag{11}
\]

where `c_sigma=-log d_sigma`. The exact first-level loss satisfies
`S_epsilon(X_1)-kappa=O(epsilon)`, with its known nonnegative sign.
The derivative of the explicit phase in (11) is

\[
 {1\over X(1-\sigma\beta_0uX)},
\]

uniformly bounded above and away from zero on the common event bracket.
Its exact root is

\[
 X_*(u)={d_\sigma\over1+\sigma\beta_0d_\sigma u}.
\]

The mean value theorem therefore gives

\[
 X_1=X_*(u)+O\!\left(\omega[1+u\log(1/\omega)]\right).
 \tag{12}
\]

The denominators in X* have common positive bounds after reducing u0.
Multiplying the error in (12) by u squared gives
`O(epsilon*u+epsilon*u^2*log(1/omega))`, absorbed in the remainder of
(8). Equations (8) and (12) prove (1). Expanding the rational square in u
gives (2).

## 5. Positive overlap with the existing interior theorem

Fix a small u0 after rho and the coefficient compacts, and restrict
`u in [u0/2,u0]`. The rational anchors in (1) have strictly positive
lower bounds of order `u0^2`, uniform over parameters. For sufficiently
small epsilon the actual signed endpoints are therefore positive and
belong to fixed compact label intervals. This conclusion needs no sign
claim about an uncomputed section shift; its absolute O(epsilon) error
is now small compared with the fixed positive anchor.

Let the limiting positive input and output base labels be

\[
 B_i={u d_+\over1+\beta_0d_+u},\qquad
 B_o={u d_-\over1-\beta_0d_-u}.
\]

With the existing singular theorem's parameters

\[
 a=d_-/d_+,\qquad b=\beta_0(a+1),
\]

one has exactly

\[
 B_o={aB_i\over1-bB_i},\qquad
 1-bB_i={1-\beta_0d_-u\over1+\beta_0d_+u}>0.
 \tag{13}
\]

After a further fixed reduction of u0, input and output intervals and their
small enlargements lie strictly inside the positive-base and pole margins
of the [existing interior theorem](HILBERT16-INTERIOR-CYCLICITY.md). Its cutoff may depend on this fixed u0,
which is compatible with choosing epsilon last. Both descriptions use the
same exact finite-epsilon physical sections and normal-forward orientation.
They therefore concern a common interval of actual trajectories, not merely
two abstract limiting formulas.

This supplies the requested joining anchor. Regular capture, connected
admission, and the appropriate displacement operator remain necessary for
any combined cycle count. No such count is obtained by adding separate
layer bounds in this note.

The [joined positive-base argument](HILBERT16-JOINED-POSITIVE-BASE.md)
uses this anchor to supply overlap and coverage for its stated captured
cycle count. The finite identities can be replayed alongside the
[two-scale correction](HILBERT16-TWO-SCALE-CORRECTION.md); those checks
do not discharge the analytic remainder bounds here.

