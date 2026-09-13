# The actual exponential transition and its derivative in the logarithmic height

This is a written analytic proof for a specified physical section itinerary.
It is not a novelty claim, a formally verified analytic theorem, or a proof
of full graphic cyclicity. The compact-domain bounds are not numerical
interval certificates.

## 1. Statement and precise local parameter coverage

Use the exact canonical quadratic family and normal coordinates from the
[grazing proof](HILBERT16-GRAZING-PASSAGE.md) and the
[quantitative thin-height proof](HILBERT16-SUBEXPONENTIAL-PASSAGE.md). Fix

\[
 r=-1,\quad C\in[C_0,C_1]\Subset(1,\infty),\quad
 A=1+\nu\bar A,\quad |\bar A|\le A_0,
\]
\[
 L=-\lambda_0\in[c,L_0],\quad |\lambda_1|\le L_0,
 \qquad 0<\kappa_0\le\kappa\le\kappa_1<\infty.
 \tag{1.1}
\]

The exact canonical small parameter satisfies `epsilon=nu*k`, with `k`
uniformly positive and analytic. The physical section parameter `rho>0`
is fixed first, sufficiently small for the existing regular-passage
construction. All parameters other than `kappa` are held fixed when
differentiating below.

For each radial sign `sigma=+-1`, define

\[
 B_\sigma(x)=L-\sigma\lambda_1x+x^2,
 \qquad S_\sigma(x)=\int_0^x{s\over B_\sigma(s)}\,ds.
 \tag{1.2}
\]

Assume fixed radial endpoints `K_sigma>0` and uniform strict margins such
that, over the compact parameter set,

\[
 B_\sigma\ge b>0\ \hbox{on }[0,K_\sigma],\qquad
 S_\sigma(K_\sigma)\ge\kappa_1+d_S.
 \tag{1.3}
\]

The endpoints may differ between the two signs. They need not be small,
and there is no condition `x_escape<K_sigma/2`. Positivity and a strict
post-escape margin are sufficient. The unique solutions

\[
 S_\sigma(x_\sigma(\kappa))=\kappa
\]

then have uniform margins `0<x_min<=x_sigma<=K_sigma-d_x`. They depend
smoothly on `kappa`, with

\[
 x_\sigma'(\kappa)={B_\sigma(x_\sigma)\over x_\sigma}.
 \tag{1.4}
\]

Start the **actual** normal trajectory at

\[
 V=0,\qquad H=h(0)=\epsilon^3e^{-\kappa/\epsilon}.
 \tag{1.5}
\]

It reaches the chosen outer physical sections. Write their exact signed
family labels as `t_i(kappa),t_o(kappa)`, using `t=(v-1)^2-2h` on each
section. Let `D_epsilon` be the actual normal-forward section map, so
`D_epsilon(t_i(kappa))=t_o(kappa)`. Then

\[
 \boxed{
 D_\epsilon'(t_i(\kappa))\longrightarrow
 R(\kappa):={B_-(x_-(\kappa))\over B_+(x_+(\kappa))}
 \quad\hbox{in }C^1(\kappa).
 }
 \tag{1.6}
\]

Convergence is uniform over the compact parameter sets in (1.1)–(1.3).
Here `D_epsilon'` is the derivative in the physical signed input label;
`C^1(kappa)` concerns that entire derivative evaluated at `t_i(kappa)`.
Also, with actual moving endpoints included,

\[
 {dt_i\over d\kappa}
 =2\epsilon^2B_+(x_+)(1+o(1)),\qquad
 {dt_o\over d\kappa}
 =2\epsilon^2B_-(x_-)(1+o(1)).
 \tag{1.7}
\]

In particular both label parametrizations increase strictly. The derivative
conclusion in (1.6) is proved by differentiating an exact integral, not by
differentiating a value-only limiting formula.

## 2. Exact radial equation and coefficient derivatives

Write the exact normal field as

\[
 \dot V=f+h g,\qquad\dot h=-Vh,
 \quad f=-L\epsilon^3+\lambda_1\epsilon^2V
                       +\epsilon V^2\zeta(V,\epsilon),
 \quad q=-f,\quad k=-g.
\]

On a fixed compact physical-coordinate domain containing the outer sections,
the exact normal change gives

\[
 k=1+O(\epsilon),\quad k_V=O(\epsilon),\quad
 k_h=O(\epsilon^2),\quad k_{Vh}=O(\epsilon^2).
 \tag{2.1}
\]

These follow from joint analyticity and the first-order expansion
`g=-1+nu(V-1)+O(nu^2)`, whose linear coefficient is independent of `h`.
The normal-coordinate inverse satisfies
`v_V=-1/ell`, `v_h=-C nu^2 v/ell`, with
`ell=1+2nu v+Cnu^2h` uniformly bounded away from zero. Thus the estimates
hold for partial derivatives in the exact normal coordinates, including
at height zero.

For a radial sign put `V=sigma epsilon x`, `h=epsilon^3 y`. Define

\[
 B_{\epsilon,\sigma}(x)
 =L-\sigma\lambda_1x-x^2\zeta(\sigma\epsilon x,\epsilon),
 \quad k_{\epsilon,\sigma}(x,y)
 =k(\sigma\epsilon x,\epsilon^3y,\epsilon).
\]

In the rest of the radial proof suppress `epsilon,sigma` on these exact
coefficients. Since `zeta(0,epsilon)=-1`,

\[
 B_{\epsilon,\sigma}\to B_\sigma\ \hbox{in }C^1([0,K_\sigma]),
 \qquad B_{\epsilon,\sigma}\ge b/2.
\]

The exact scalar equation, with positive radial orientation on both sides, is

\[
 \epsilon y_x={xy\over B(x)+k(x,y)y},\qquad
 y(0)=\eta=e^{-\kappa/\epsilon}.
 \tag{2.2}
\]

Let `W=epsilon*y`. For small `epsilon`, `k>=1/2`, so
`0<=W_x<=2x` and `0<y<=eta+K_sigma^2/epsilon`. Consequently the full
radial solution stays within `h=epsilon^3y<=C epsilon^2`, where all the
coefficient estimates are valid, and it exists on the entire radial interval.
Positivity is preserved; no unproved slow passage is used for existence.

With partial derivatives at fixed `y` or fixed `x` respectively, (2.1) gives

\[
 k_x=O(\epsilon^2),\qquad k_y=O(\epsilon^5),\qquad
 k_{xy}=O(\epsilon^6).
 \tag{2.3}
\]

All these bounds are uniform on `0<=x<=K_sigma`, `0<=y<=C/epsilon`.

## 3. Uniform escape and height limits

Define `S_epsilon(x)=integral_0^x s/B_epsilon(s) ds`, which tends uniformly
to `S_sigma`. Equation (2.2) implies

\[
 \epsilon\log y(x)\le-\kappa+S_\epsilon(x).
 \tag{3.1}
\]

Thus, uniformly on any region separated by a fixed positive distance from
the escape point on its incoming side, `y` is exponentially small.

On the outgoing side, fix a small radial separation `d>0`. Compactness
gives a positive uniform gap `S_sigma(x_sigma+d/3)-kappa`. Choose a small
fixed height level `m>0` such that replacing `B_sigma` by `B_sigma+2m`
reduces the integral by less than half that gap. For small `epsilon`, if
`y` stayed below `m` up to that point, (2.2) would imply

\[
 \epsilon\log y(x)
 \ge-\kappa+\int_0^x{s\over B_\epsilon(s)+2m}\,ds>c_d>0,
\]

contradicting `y<=m`. Hence `y` reaches `m` before `x_sigma+d/3`.
It is nondecreasing. On the next fixed radial interval, bounded away from
zero, its derivative satisfies

\[
 y_x={x\over\epsilon(B/y+k)}\ge{c_m\over\epsilon}.
\]

Therefore `y>=c_d'/epsilon` by `x_sigma+2d/3`. This proves, uniformly
away from the escape point,

\[
 y\to0\quad(x<x_\sigma),\qquad
 y\to\infty\quad(x>x_\sigma).
 \tag{3.2}
\]

Since `W_x=xy/(B+ky)` is bounded by `2x`, it tends to zero before escape
and to `x` after escape. Split an integral into the two regions with a
fixed escape margin and the remaining interval of length `2d`; then let
`epsilon->0` followed by `d->0`. This proves the uniform height limit

\[
 W(x)\longrightarrow {1\over2}(x^2-x_\sigma^2)_+.
 \tag{3.3}
\]

In particular `W(K_sigma)` has a uniform strictly positive lower bound.
For every fixed `0<m<M<infinity`, the inverse level locations exist for
small `epsilon`, and

\[
 x_\epsilon(y,\kappa)\longrightarrow x_\sigma(\kappa)
 \quad\hbox{uniformly for }m\le y\le M.
 \tag{3.4}
\]

## 4. The compensated exponent and its value limit

Set `D=B+ky` and

\[
 F_\epsilon(x,y)={B_x+y k_x\over D},\qquad
 \Psi_\epsilon(x)=\int_0^xF_\epsilon(s,y(s))\,ds.
 \tag{4.1}
\]

The exact compensated scalar variation identity gives

\[
 J_y(x)={A_\epsilon\over\eta}{y(x)\over D(x,y(x))}
                         e^{\Psi_\epsilon(x)},\qquad
 A_\epsilon=B(0)+k(0,\eta)\eta=L+k(0,\eta)\eta.
 \tag{4.2}
\]

Here `J_y` denotes variation with respect to the initial `eta`, at fixed
coefficients. This identity follows by differentiating `log(y/D)` along
(2.2); its partial `k_x` derivative is at fixed height. All exponents are
uniformly bounded since `|B_x|/D` is bounded and `|y k_x|/D=O(epsilon^2)`.

Using (3.2), the first term of the integrand tends to `B_sigma'/B_sigma`
before escape and to zero afterwards. The second is uniformly
`O(epsilon^2)`. Dominated integration with the same split around escape
therefore yields

\[
 \Psi_\epsilon(K_\sigma)\longrightarrow
 \log{B_\sigma(x_\sigma)\over L}.
 \tag{4.3}
\]

The same argument, with the upper endpoint `x_epsilon(y,kappa)` from
(3.4), gives the uniform middle-level limit

\[
 \Psi_\epsilon(x_\epsilon(y,\kappa))\longrightarrow
 \log{B_\sigma(x_\sigma)\over L}
 \quad(m\le y\le M).
 \tag{4.4}
\]

## 5. Differentiating the exact exponent: the actual kernel

For every fixed positive `epsilon`, ordinary smooth ODE dependence allows
differentiation in `kappa`. Since `eta_kappa=-eta/epsilon`, (4.2) gives
the exact formula

\[
 y_\kappa=-{A_\epsilon\over\epsilon}
                     {y\over D}e^{\Psi_\epsilon}.
 \tag{5.1}
\]

Consequently `|y_kappa|<=C/epsilon` everywhere, and
`|y_kappa|<=C y/epsilon` where `y` is small. Do not differentiate the
limiting step profile to obtain this formula.

An exact derivative of the integrand in (4.1) is

\[
 -\partial_yF_\epsilon={P_\epsilon\over D^2},
\]
\[
 P_\epsilon=B_x(k+y k_y)-B k_x-B y k_{xy}
                        -k y^2k_{xy}+y^2k_xk_y.
 \tag{5.2}
\]

Differentiate first on the fixed `x` interval, and only then change
variables by `dx=epsilon D/(xy) dy` on `[a,K_sigma]`, for any fixed
`a>0`. This order introduces no extra moving-height boundary terms.
The exact formula is

\[
 \partial_\kappa\Psi_\epsilon(K_\sigma)
 =\int_0^a (F_\epsilon)_y y_\kappa\,dx
 +A_\epsilon\int_{y(a)}^{y(K_\sigma)}
 {e^{\Psi_\epsilon(x_\epsilon(y))}\over x_\epsilon(y)}
                   {P_\epsilon\over D^2}\,dy.
 \tag{5.3}
\]

The radial-origin portion of this formula is treated in `x`, as follows.
Choose a fixed `a>0` with `S_sigma(a)<=kappa0/2` uniformly. On `[0,a]`,
(3.1) gives `y<=exp(-kappa0/(3epsilon))` for small `epsilon`. The
integrand derivative `F_y` is uniformly bounded there, so its contribution
to `partial_kappa Psi` is at most

\[
 C\epsilon^{-1}e^{-\kappa_0/(3\epsilon)}\longrightarrow0.
 \tag{5.4}
\]

Use (5.3) only on `x>=a`. There `1/x` is bounded, as are `A_epsilon`
and `exp(Psi)`. The exact actual-coefficient error satisfies

\[
 \left|{P_\epsilon-B_xk\over D^2}\right|
 \le C\left\{
 {\epsilon^2\over(1+y)^2}
 +{\epsilon^5\over1+y}+\epsilon^6\right\}.
 \tag{5.5}
\]

Indeed the numerator terms other than `B_xk` are bounded by
`C(epsilon^2+epsilon^5 y+epsilon^6 y^2)`, using (2.3).
On `0<=y<=C/epsilon`, their total integral is
`O(epsilon^2+epsilon^5 log(1/epsilon)+epsilon^5)=o(1)`.
This accounts for the actual `k_x,k_y,k_xy` terms rather than setting
them to zero.

The leading kernel is bounded in absolute value by `C/(1+y)^2`.
Its small-height tail has size at most `Cm`, and its large-height tail
at most `C/M`, uniformly. For `y` in a fixed `[m,M]`, (3.4) and (4.4)
give the uniform kernel limit

\[
 {B_\sigma(x_\sigma)B_\sigma'(x_\sigma)
       \over x_\sigma[B_\sigma(x_\sigma)+y]^2}.
 \tag{5.6}
\]

First let `epsilon->0` on `[m,M]`, then `m->0` and `M->infinity`.
The integral of (5.6) is

\[
 {B_\sigma'(x_\sigma)\over x_\sigma},
\]

since `integral_0^infinity (B+y)^-2 dy=1/B`. Combining with (5.4)–(5.5)
proves the actual derivative limit

\[
 \boxed{
 \partial_\kappa\Psi_\epsilon(K_\sigma)
       \longrightarrow{B_\sigma'(x_\sigma)\over x_\sigma}.}
 \tag{5.7}
\]

All origin, tail, and middle estimates are uniform over (1.1)–(1.3).
Together (4.3) and (5.7) establish `C^1(kappa)` convergence of this
radial exponent, with no differentiation of a `C^0` limit.

## 6. The physical tails and their kappa derivatives

At `V=sigma epsilon K_sigma`, (3.3) gives

\[
 h=\epsilon^2 W(K_\sigma),\qquad
 {h\over V^2}\longrightarrow
 {K_\sigma^2-x_\sigma^2\over2K_\sigma^2}>c_K>0.
\]

The lower ratio may be small but is uniformly positive. It need not
equal any predetermined fraction. Bootstrap `c_1 V^2<=h<=c_2 V^2`
with fixed smaller/larger constants. Since

\[
 {|q|\over h}+|k-1|
 \le C\left(\epsilon+{\epsilon^2\over|V|}
                         +{\epsilon^3\over V^2}\right)=O(\epsilon)
\]

for `|V|>=epsilon K_sigma`, `h/D` tends uniformly to one. The radial
slope `dh/d|V|=|V| h/D` improves the chosen barriers and continues
the actual trajectory to both outer physical sections. Tail integration
gives `h(V)=V^2/2+O(epsilon)` and `R=V+O(epsilon)` near those sections.
In normal time `V` decreases; the radial positive side is its backward
passage and the radial negative side its forward passage.

Let the physical tail exponent integrand be

\[
 Z(V,h)={q_V+h k_V\over q+kh}.
\]

The value of its tail integral tends to zero: its bound is
`C(epsilon^2/(epsilon K_sigma)+epsilon log(1/epsilon)+epsilon)`.
At fixed `V`, (5.1) gives `h_kappa=O(epsilon^2)` at the radial exit.
The ordinary scalar variational equation propagates this bound over the
tail, since

\[
 \int|\mathcal R_h|\,dV
 \le C\{\epsilon\log(1/\epsilon)+\epsilon/K_\sigma
                                      +\epsilon/K_\sigma^2\}=o(1).
 \tag{6.1}
\]

An exact quotient derivative and the actual coefficient bounds yield,
writing `r_V=|V|` just within this estimate,

\[
 |Z_h|\le C\left\{
 {\epsilon\over r_V^3}+{\epsilon^2\over r_V^4}
                   +{\epsilon^2\over r_V^2}+\epsilon^2\right\}.
 \tag{6.2}
\]

For example its numerator is
`q k_V+q h k_Vh+k h^2 k_Vh-q_V k-q_V h k_h-h^2 k_V k_h`.
Multiplication by `h_kappa=O(epsilon^2)` and integration from
`r_V=epsilon K_sigma` to a fixed outer section makes (6.2) `O(epsilon)`.
The lower endpoint is fixed with respect to `kappa`. The upper physical
endpoint has derivative `O(epsilon^2)`, while `Z=O(epsilon)` there,
so its moving-endpoint contribution is also negligible. Consequently
the physical tail exponent is `o(1)` in `C^1(kappa)`.

It follows that the complete physical compensated exponents satisfy

\[
 \Psi_i\to\log[B_+(x_+)/L],\qquad
 \Psi_o\to\log[B_-(x_-)/L]
 \quad\hbox{in }C^1(\kappa).
 \tag{6.3}
\]

## 7. Exact endpoint factors and the physical derivative limit

The physical sections satisfy `v=tau rho h`, where `tau=-1` is the
incoming section and `tau=+1` the outgoing section. Their exact equations
in normal coordinates and their signed labels are

\[
 E_\tau=V-1+\tau\rho h+\nu\rho^2h^2+C\nu^2\tau\rho h^2=0,
 \qquad T_\tau(h)=(\tau\rho h-1)^2-2h.
\]

Their outer branches remain transverse. With `J` the height variation at
fixed `V`, the correct moving-endpoint multiplier is

\[
 {dt_\tau\over dH}=\mathcal K_\tau J_\tau,
 \qquad \mathcal K_\tau={T_\tau'(h_\tau)
                          \over1+(E_\tau)_h\mathcal R_\tau}.
 \tag{7.1}
\]

At the limiting endpoints `v=1-V`, the numerator is
`-2(1+tau rho V)` and the denominator is `1+tau rho V`, so
`mathcal K_tau=-2+O(epsilon)`. Both exact physical endpoints have
`kappa` derivatives `O(epsilon^2)` by the transverse event equation
and the fixed-`V` estimate for `h_kappa`. Hence

\[
 \mathcal K_\tau\to-2\quad\hbox{in }C^1(\kappa),\qquad
 {h_\tau\over q(V_\tau)+k(V_\tau,h_\tau)h_\tau}
             \to1\quad\hbox{in }C^1(\kappa).
 \tag{7.2}
\]

The compensated identity in physical coordinates is

\[
 J(V)={q(0)+k(0,H)H\over H}
             {h\over q+kh}\,e^{\Psi(V)}.
\]

Since `H_kappa=-H/epsilon`, it gives at fixed `V`

\[
 h_\kappa=-\epsilon^2 A_\epsilon {h\over q+kh}e^{\Psi(V)}.
 \tag{7.3}
\]

Now `A_epsilon->L`, and (6.3), (7.1)–(7.3) prove (1.7). This uses the
actual physical events, not fixed-`V` heights as though they were section
coordinates.

Finally the exact ratio is

\[
 D_\epsilon'(t_i(\kappa))
 ={\mathcal K_o\over\mathcal K_i}
  {{h_o/(q_o+k_oh_o)}\over{h_i/(q_i+k_ih_i)}}
                 e^{\Psi_o-\Psi_i}.
 \tag{7.4}
\]

The common exponentially large factor cancels exactly. Equations (6.3)
and (7.2) show that every remaining factor converges in `C^1(kappa)`.
This proves (1.6).

## 8. The explicit slope derivative and what it does not yet cover

Using (1.4), the now justified limiting slope satisfies

\[
 {d\over d\kappa}\log R
 ={B_-'(x_-)\over x_-}-{B_+'(x_+)\over x_+}
 =\lambda_1\left({1\over x_-}+{1\over x_+}\right).
 \tag{8.1}
\]

If `lambda1` is bounded strictly below zero, this has a uniform negative
margin on the compact set. Equations (1.6)–(1.7) then imply the actual
physical curvature limit

\[
 \epsilon^2D_\epsilon''(t_i(\kappa))
       \longrightarrow{R'(\kappa)\over2B_+(x_+)}<0.
 \tag{8.2}
\]

Thus a bounded-curvature regular comparison on a connected admitted
input interval can use strict convexity of `H_reg-D_epsilon` to bound
its zeros by two in this compact exponential window. The regular capture,
common-section, and connected-domain conditions are additional existing
requirements; (8.2) alone is not a global cycle count. The case
`lambda1=0` has zero leading curvature, so this particular strict gate
does not follow there.

Sections 9 and 10 supply the positive-component parameter cover and
the captured two-cycle bound on a whole compact exponential band.
The limit `kappa->infinity`, vanishing parameter faces, and full graphic
capture remain outside these conclusions. Bounds on different windows
are not added.

Primary normal-form context: [Huzak–Kristiansen, section 6](https://documentserver.uhasselt.be/bitstream/1942/49936/1/On_entry_exit_formulas_for_degenerate_turning_point_problems_in_planar_slow_fast_systems.pdf).
The proof above uses the exact family and scalar variational equations,
not an application of the published compact-base theorem at a zero fiber.

## 9. Uniform coverage of every compact positive logarithmic-height band

Fix `0<c<=L<=Lmax`, `|lambda1|<=M` and
`0<kappa0<=kappa<=kappa1<infinity`. No discriminant sign is assumed.
Let `r_sigma` be the first positive zero of `B_sigma`, or infinity if
there is none. On `[0,r_sigma)` the polynomial is positive, and
`S_sigma` increases from zero to infinity. At a simple first zero the
integral diverges logarithmically; at a double first zero it diverges
reciprocally. With no positive zero its integrand is asymptotic to `1/x`.
Every finite positive `kappa` therefore has a unique inverse on this
component. Its defining equation has positive derivative, so local
implicit-function theory gives smooth dependence on all parameters,
including across zero discriminant.

For explicit uniform bounds, set

    Kappa = kappa1 + 1
    A = max(1, M, sqrt(Lmax))
    X = A*exp(3*(Kappa+1))
    d = min(1, c/(2*(M+1)))
    bmin = (c/2)*exp(-M*Kappa/d)
    Bmax = Lmax + M*X + X^2
    xmin = min(d, sqrt(c*kappa0)).

Then, on the entire compact parameter set,

    xmin <= x_sigma(kappa) < X,
    bmin <= B_sigma(x) <= Bmax  for 0<=x<=x_sigma(Kappa).

Indeed, for `x>=A` one has `B_sigma(x)<=3x^2`. Either a positive root
occurs before `X`, or `S_sigma(X)>=log(X/A)/3>Kappa`; both cases give
the upper inverse bound. On `[0,d]`, `B_sigma>=c/2`, hence
`S_sigma(x)<=x^2/c`, which gives the lower inverse bound. Beyond `d`,

\[
 {d\over d\kappa}\log B_\sigma(x_\sigma(\kappa))
       =2-{\sigma\lambda_1\over x_\sigma}\ge-{M\over d}.
\]

Starting at `B_sigma(d)>=c/2` proves the lower polynomial bound;
before `d` it already holds. The upper bound follows from `x<X`.

One may now take branch-specific radial cuts at `x_sigma(Kappa)`.
Their distance after escape in the original band is at least `bmin/X`,
by (1.4). To use fixed cuts as in (1.3), at each coefficient point
choose a cut strictly between `x_sigma(kappa1)` and
`x_sigma(Kappa)`. Continuity preserves strict positive-polynomial and
post-escape margins on a parameter neighborhood; a finite cover of the
compact coefficient set supplies uniform estimates. The two branches
may require different cuts. Requiring a common cut, or requiring escape
before half the cut, could incorrectly discard finite logarithmic heights
near a positive radial root.

The maps in overlapping neighborhoods are the same physical maps by
flow uniqueness, in the same exact section coordinates. Taking a finite
maximum of errors and minimum of cutoffs makes (1.6)–(1.7) uniform over
the full compact set. This is not a sum of chart-dependent cycle counts.

## 10. A two-cycle bound on the entire compact exponential band

Let `I_sigma(x)=integral_0^x ds/B_sigma(s)`. Direct differentiation and
the value at zero give

\[
 S_\sigma(x)={1\over2}\log{B_\sigma(x)\over L}
                      +{\sigma\lambda_1\over2}I_\sigma(x).
\]

At the two escape points, both `S_sigma` equal `kappa`. Consequently

\[
 \log R=\lambda_1 J,\qquad
 J=I_+(x_+)+I_-(x_-),\qquad
 0<J\le {2X\over b_{\min}}=:C_J.
 \tag{10.1}
\]

Also `rmin:=bmin/Bmax<=R<=Bmax/bmin`. The
[regular derivative theorem](HILBERT16-REGULAR-JETS.md), with the exact
[signed-coordinate transport](HILBERT16-INTERIOR-CYCLICITY.md), supplies
on its captured connected domain

\[
 0<H_{\rm reg}'\le\theta_H<1,\qquad |H_{\rm reg}''|\le K_{\rm reg},
 \tag{10.2}
\]

for fixed sufficiently small section parameter `rho`. These bounds
hold uniformly for bounded compensator and the compact physical
coefficients. The actual exponential endpoints converge to the zero-label
outer events, so they lie in that theorem's fixed section-coordinate
range. The regular captured domain may be a proper subinterval.

Choose `a>0` sufficiently small that
`exp(-a*C_J)>(1+theta_H)/2`. If `lambda1>=-a`, (10.1) and the uniform
value part of (1.6) give `D_epsilon'>theta_H` for small `epsilon`.
The displacement `F=H_reg-D_epsilon` is strictly decreasing and has
at most one zero on the admitted interval.

If `lambda1<=-a`, (8.1) gives instead

\[
 R'\le-{2a r_{\min}\over X}=:-c_R<0.
\]

The proved derivative convergence gives
`partial_kappa[D_epsilon'(t_i(kappa))]<=-c_R/2` for small `epsilon`.
Equation (1.7) gives `0<t_i,kappa<=4epsilon^2 Bmax`. Thus

\[
 D_\epsilon''(t_i(\kappa))
      \le-{c_R\over8B_{\max}\epsilon^2}.
 \tag{10.3}
\]

For sufficiently small `epsilon`, this dominates (10.2), and `F''>0`.
Strict convexity bounds the zeros by two. This inference uses the
derivative of the actual map, including its moving physical endpoints.

For each fixed physical parameter, (1.7) maps the whole `kappa` band
increasingly onto one input-label interval. Its intersection with the
regular captured domain is an interval. Each cycle with this itinerary
is a zero of `H_reg-D_epsilon`, since the normal-forward singular map
is the inverse of the physical closing passage. The two alternatives
above partition parameter regimes on this same entire interval.
There are therefore **at most two captured cycles in the specified
compact exponential band**, uniformly over the coefficient compacts.
The count is at most one in the slope-separated regime `lambda1>=-a`.

This conclusion does not cover unbounded `kappa`, `L->0`, unbounded
coefficients or section charts, or other itineraries. It is a written
analytic argument, not a numerical certificate or a formal analytic proof.

## 11. Joining the initial sector without adding cycle counts

For any fixed `kappa1>0`, the same captured itinerary has at most two
cycles on the whole center-height band

\[
 \boxed{\epsilon^3e^{-\kappa_1/\epsilon}\le H\le H_{\max}}
 \tag{11.1}
\]

for sufficiently small `epsilon`, uniformly over the coefficient compacts
with `L>=c>0`. Here `Hmax` is fixed and sufficiently small as in the
grazing proof. The allowed smallness threshold depends on `kappa1`;
this statement gives no uniform conclusion as `kappa1->infinity`.

To prove it, take the constant `kappa_*>0` from section 7 of the
[quantitative thin-height theorem](HILBERT16-SUBEXPONENTIAL-PASSAGE.md).
If `kappa1<=kappa_*`, that theorem already gives at most one cycle.
Otherwise fix `kappa_a=kappa_*/2` and apply sections 9–10 on
`[kappa_a,kappa1]`. The initial theorem gives `F'<0` throughout
`H>=epsilon^3 exp(-kappa_*/epsilon)`, while the new exponential band
extends to `H=epsilon^3 exp(-kappa1/epsilon)` with a nonempty overlap.
All these are the same physical maps, since the center trajectory and
selected outer events are unique.

Throughout the initial band, the old exact endpoint formula gives
`t_i,H<0`; throughout the exponential band this also follows from
`t_i,kappa>0` and `H_kappa<0`. Thus the union maps monotonically to one
input-label interval. Its intersection with the regular captured domain
is again an interval. Increasing the input label moves from the initial
band into the exponential band.

Use the parameter split of section 10. If `lambda1>=-a`, both estimates
give `F'<0` throughout the union, so there is at most one zero. If
`lambda1<=-a`, the displacement is strictly decreasing on the initial
portion and strictly convex on the exponential portion. The portions
overlap where `F'<0`. On their union `F'` therefore has at most one
zero: it is negative initially and strictly increasing on the later
portion. Rolle's theorem bounds the number of distinct zeros of `F`
by two. Restricting to the admitted interval preserves the argument.

This is one displacement argument across an overlap. It does not sum
the earlier one-cycle bound and the exponential two-cycle bound, and
it does not join (11.1) to a delayed passage at unbounded logarithmic
height or to a different itinerary.

The [limiting-tail companion](HILBERT16-MATCHING-LIMIT.md) derives a
uniform analytic expansion at infinite logarithmic height for the
limiting family and encloses its regularized rational tail using existing
verified components. It does not transfer that expansion to the actual
two-parameter passage.

Two subsequent actual estimates now extend the fixed-band results:
[joint matching](HILBERT16-JOINT-MATCHING.md) controls the shrinking
physical-escape corner with 4L-lambda1^2 bounded strictly positive, and
[first-root saddle passage](HILBERT16-ROOT-SADDLE.md) controls all large
admitted kappa in a selected small-label tube when lambda1<0 and
lambda1^2-4L is bounded strictly positive. Their own proofs establish
uniformity; neither follows by sending kappa1 to infinity in (11.1).

## Finite algebra reproduction

The companion benchmark checks the exact compensated-variation and
derivative-kernel identities, limiting slope algebra, and rational
instances of the declared slope/curvature inequalities:

```bash
python benchmarks/hilbert16_exponential_passage.py --output artifacts/hilbert16/exponential_passage.json
```

It does not mechanically verify the limiting arguments, supply numerical
values for their sufficiently small cutoffs, or establish Hilbert 16.

The benchmark includes the independently derived exact checks of the
actual kernel, change of variables, physical tail derivative, limiting
kernel mass and slope identities. These do not formalize the uniform
estimates, parameter coverage or cycle conclusions.
