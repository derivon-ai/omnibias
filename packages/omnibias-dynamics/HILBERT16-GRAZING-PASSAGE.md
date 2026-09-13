# Actual passage through the grazing core, parameterized by center height

This is a written analytic proof for a restricted physical itinerary. It is
not a novelty claim, a formal verification of an analytic theorem, or a proof
of full graphic cyclicity. All constants below are finite compact-domain
constants; no numerical interval cutoff is claimed.

## 1. Result

Fix `0<rho<1/2`, sufficiently small for the existing captured regular-passage
contraction. In the exact quadratic family use `r=-1`, `C` in a compact subset
of `(1,infinity)`, the canonical slow-parameter embedding, and

\[
 -L_0\le\lambda_0\le0,\qquad |\lambda_1|\le L_0,
 \qquad A=1+\nu\bar A,\quad |\bar A|\le A_0.
 \tag{1.1}
\]

The discriminant need not be negative. As usual, `epsilon=nu*k`, with `k`
jointly analytic and uniformly positive near zero. Fix any `gamma>0` and a
small fixed `Hmax>0` satisfying

\[
 1-2\rho-2\rho^2 H_{\max}>0.
 \tag{1.2}
\]

If needed, shrink `Hmax` so that the section labels below lie in the fixed
small label range used for regular-coordinate transport.

For every sufficiently small positive `epsilon`, start the exact normal
trajectory at

\[
 V=0,\qquad h=H,\qquad
 \gamma\epsilon^3\le H\le H_{\max}.
 \tag{1.3}
\]

It reaches both outer physical sections `x=+-rho/nu`, in backward and forward
normal time respectively. Write their **signed family labels** as

\[
 t_i(H)=(v_i-1)^2-2h_i,\qquad
 t_o(H)=(v_o-1)^2-2h_o.
\]

Both functions are strictly decreasing. They define an actual normal-forward
section map `D_epsilon(t_i(H))=t_o(H)` on one connected input interval. The
main derivative conclusion is

\[
 \boxed{
 \sup_{\gamma\epsilon^3\le H\le H_{\max}}
 \left|D_\epsilon'(t_i(H))-1\right|\longrightarrow0.
 }
 \tag{1.4}
\]

The convergence is uniform over (1.1) and the other fixed compact parameter
sets. For any prescribed accuracy, the proof gives an existence cutoff by
choosing a core size, then a fixed height threshold, then `epsilon` small.
It does not require that the actual section-center location be expanded to
error smaller than `epsilon^3`.

Consequently, where the [captured regular map](HILBERT16-REGULAR-JETS.md),
transported to the same signed labels as in
[rapid passage](HILBERT16-RAPID-PASSAGE.md), has
`H_reg'<=theta<1`, there is at most one cycle on the connected admitted part
of the entire band (1.3), for sufficiently small `epsilon`. This is one
count on one connected band; it is not obtained by adding counts of core
and rapid subintervals.

The layer `H/epsilon^3 -> 0` remains unresolved by this proof, including
its joint degeneration with `lambda0 -> 0`. The parameter face `lambda0=0`
is included on the full stated positive-height band.

Subsequent [height comparison](HILBERT16-HEIGHT-COMPARISON.md) removes
the height cutoff for admitted bounded passages with lambda1>=0. The
[subexponential extension](HILBERT16-SUBEXPONENTIAL-PASSAGE.md) reaches
thinner heights for either sign of lambda1 when lambda0 stays strictly
negative. Their remaining exponential and degenerate layers are recorded
in the [program ledger](HILBERT16.md).

## 2. Exact normal coordinates and uniform coefficient bounds

Use the exact family formulas established in
[singular transport](HILBERT16-SINGULAR-TRANSPORT.md)

\[
 F=m+pv-\nu v^3-h+A\nu vh-C\nu^2v^2h,
\]
\[
 V=1-v-\nu v^2-C\nu^2vh,\qquad
 \ell=1+2\nu v+C\nu^2h.
\]

After reversing physical time,

\[
 \dot V=\ell F+C\nu^2vVh
       =f(V,\epsilon)+h g(V,h,\epsilon),\qquad
 \dot h=-Vh,
 \tag{2.1}
\]

where the exact canonical normalization gives

\[
 f=\epsilon^3\lambda_0+\epsilon^2\lambda_1V
                  +\epsilon V^2\zeta(V,\epsilon),
 \qquad \zeta(0,\epsilon)=-1.
 \tag{2.2}
\]

The normal-coordinate inverse is jointly analytic on every fixed compact
`(V,h)` domain for sufficiently small `nu`. Indeed `ell` stays, for example,
between `1/2` and `3/2`, and

\[
 v_V=-1/\ell,\qquad v_h=-C\nu^2v/\ell.
 \tag{2.3}
\]

All needed coordinate derivatives are therefore uniformly bounded there;
the mixed dependence on height first occurs at order `nu^2`. Choosing a
slightly enlarged compact domain gives the same bounds for any fixed finite
number of derivatives.

Factoring the transverse term by analytic division in `h` gives a jointly
analytic `g`. The exact first-order expansion of (2.1) is

\[
 \dot V=-h+\nu\{V^3-3V^2+(V-1)h\}+O(\nu^2)
 \tag{2.4}
\]

under `A=1+nu*bar A`. Its first-order coefficient of `h` is independent of
`h`, so, on each such fixed compact domain,

\[
 |g+1|+|g_V|\le M\epsilon,\qquad
 |g_h|\le M\epsilon^2.
 \tag{2.5}
\]

These are ordinary uniform Taylor estimates, not divisions of an unbounded
remainder by `h`. For instance `g` can be defined by integrating the height
derivative of the right side of (2.1) from `0` to `h`; its derivatives remain
analytic at `h=0`. The constant in the second bound can be chosen as a
fixed multiple of `sup |partial_nu^2 partial_h g|` on the enlarged compact
domain. The weaker `|g_h|<=M epsilon` would already suffice below.

Also, without a sign assumption on the discriminant,

\[
 |f|\le M\{\epsilon V^2+\epsilon^2|V|+\epsilon^3\}.
 \tag{2.6}
\]

Constants can depend on `rho`, since the physical section chart has fixed
coordinates of size `rho^-1` and `rho^-2`. The exact normal change is valid
on that entire fixed compact region once `nu` is chosen sufficiently small.
No local approximation from a small paper-height section is substituted
for these actual coordinates.

## 3. The adaptive core scale

For a given center height `H`, set

\[
 a^2=H+\epsilon^3,\quad
 \xi={\epsilon^3\over a^2},\quad
 \eta={H\over a^2}=1-\xi,
 \quad V=au,\quad h=a^2w,\quad \tau=at.
 \tag{3.1}
\]

On (1.3),

\[
 0\le\xi\le{1\over1+\gamma},\qquad
 {\gamma\over1+\gamma}\le\eta\le1,
 \qquad {\epsilon^2\over a}\le\sqrt\epsilon.
\]

The **exact** rescaled system is

\[
 {du\over d\tau}=\xi\lambda_0+
       {\epsilon^2\over a}\lambda_1u
       +\epsilon u^2\zeta(au,\epsilon)
       +w g(au,a^2w,\epsilon),\qquad
 {dw\over d\tau}=-uw.
 \tag{3.2}
\]

At fixed `|u|<=U`, bounded `w`, and sufficiently small fixed upper bound on
`a`, this converges in every fixed finite derivative order in `(u,w)` to

\[
 u'=-\ell_0-w,\qquad w'=-uw,
 \quad\ell_0=-\lambda_0\xi\ge0,
 \quad w(0)=\eta.
 \tag{3.3}
\]

The coefficient error is `O_U(epsilon^2/a+epsilon)=O_U(sqrt(epsilon))`,
uniformly in the admitted `xi,eta,a` and passive parameters. The leading
path has `w>=eta>=eta_min>0`. Thus, on a slightly enlarged height domain,
the `u` velocity is uniformly negative; dividing by it gives a regular
scalar ODE for `w(u)` for both signs of `u`. Finite-interval ODE dependence
gives a uniform actual solution on `[-U,U]`, its positivity, and convergence
of its height variation. This remains valid at `xi=0`; the positive initial
height removes the apparent degeneracy of (3.3).

The leading solution `w_0` is even in `u` and satisfies

\[
 w_0+\ell_0\log w_0
       =\eta+\ell_0\log\eta+u^2/2.
 \tag{3.4}
\]

The formula includes `ell0=0` by continuity. Uniformly over the parameter
set,

\[
 w_0(U)=U^2/2+O(\log U),\qquad U\longrightarrow\infty.
 \tag{3.5}
\]

For an elementary bound, `eta<=w0<=eta+U^2/2`, and therefore

\[
 w_0(U)\ge U^2/2-L_0\log\frac{1+U^2/2}{\eta_{\min}}.
\]

Choose `U` large so that the limiting endpoint heights have a strict margin
inside `[3U^2/8,5U^2/8]`. For sufficiently small `epsilon`, the actual core
endpoints have those same bounds.

Let `J(V)=partial_H h(V;H)` at fixed `V`, with the physical parameters and
`epsilon` held fixed. When computing this variation in the core, freeze the
scale `a` at the reference trajectory: an increment in `H` is the same
increment in `a^2*w(0)`. Thus the core variation is `partial_eta w` with
`a` and the coefficients held fixed. Differentiating (3.4) gives

\[
 j_0(u)={1+\ell_0/\eta\over1+\ell_0/w_0(u)}.
 \tag{3.6}
\]

In particular the two limiting core endpoint multipliers are equal. Put

\[
 Q(H,\epsilon)=1+{(-\lambda_0)\epsilon^3\over H}.
\]

It lies in `[1,1+L0/gamma]`, and (3.5)–(3.6) imply

\[
 \left|\log{j_0(\pm U)\over Q}\right|\le {C\over U^2}.
 \tag{3.7}
\]

At fixed `U`, actual core variations differ from these values by
`O_U(sqrt(epsilon))`, uniformly. The `a`-dependence of the chosen reference
scale is not differentiated; differentiating it would incorrectly change
the physical variational problem.

## 4. The tails reach the actual physical sections

Write `D=-f-hg` and use `V` as the independent variable. The exact scalar
equation and its height derivative are

\[
 h_V=\mathcal R={Vh\over D},\qquad
 \mathcal R_h={V(-f+h^2g_h)\over D^2}.
 \tag{4.1}
\]

Choose a fixed `Vmax` larger than all the outer limiting section coordinates
for `0<=H<=Hmax`, and a fixed height domain containing `0<=h<=Vmax^2+1`.
All constants in (2.5)–(2.6) are now fixed on an enlarged version of that
domain.

First consider `H` small enough that `aU` lies strictly before both physical
sections. At the core exits `|V|=aU`, the bound in section 3 gives

\[
 3V^2/8\le h\le5V^2/8.
\]

Bootstrap `V^2/4<=h<=3V^2/4` while increasing `|V|` from `aU` to `Vmax`.
In this region, (2.5)–(2.6) imply

\[
 |(-g)-1|+{|f|\over h}
 \le C\left(\epsilon+{\epsilon^2\over |V|}
                         +{\epsilon^3\over V^2}\right)
 \le C\left(\epsilon+{\sqrt\epsilon\over U}+{1\over U^2}\right).
 \tag{4.2}
\]

Choose `U` large, then `epsilon` small, so the right side is at most `1/8`.
It follows that

\[
 {8\over9}\le {h\over D}\le {8\over7}.
\]

For either sign of `V`, the derivative with respect to `x=|V|` is
`h_x=x*h/D`. Integrating improves the bootstrap to

\[
 3V^2/8\le h\le5V^2/8,
\]

because `4/9>3/8` and `4/7<5/8`. This closes continuation to `Vmax`, with
`D>0`, positive height, and decreasing `V` in normal time. Both physical
outer intersections are reached in the appropriate time directions.

The scalar variation on these tails satisfies

\[
 |\mathcal R_h|\le C\left(
 {\epsilon\over |V|}+{\epsilon^2\over V^2}
 +{\epsilon^3\over |V|^3}+\epsilon |V|\right).
\]

We used only the weaker `g_h=O(epsilon)` bound here. Integrating gives

\[
 \int_{aU\le|V|\le V_{\max}}|\mathcal R_h|\,dV
 \le C\left({1\over U^2}+{\sqrt\epsilon\over U}
                  +\epsilon\log(1/\epsilon)+\epsilon\right).
 \tag{4.3}
\]

The constants are independent of `U`, `H`, and `epsilon` after the fixed
compact coefficient domains are chosen. In particular the constant slow
drift contributes `C epsilon^3/(a^2 U^2)<=C/U^2`; it is not discarded.

The same estimates also show convergence of the actual outer intersections
to the limiting ones. More precisely,

\[
 |\mathcal R-V|\le C\left(
 \epsilon|V|+\epsilon^2+{\epsilon^3\over |V|}\right)
 \tag{4.4}
\]

on the tails. In the core, (3.4) gives

\[
 a^2 w_0(U)-\{H+a^2U^2/2\}
       =-(-\lambda_0)\epsilon^3\log(w_0(U)/\eta).
\]

At fixed `U`, the actual core error multiplied by `a^2` is bounded by
`C_U(epsilon^2*a+epsilon*a^2)`. Integration of (4.4) therefore gives
`h(V;H)-(H+V^2/2)->0` uniformly near the fixed outer sections. The slope
there also tends uniformly to `V`.

## 5. The fixed upper height and order of choices

One cannot choose arbitrarily large `U` while keeping `aU` before the
sections for all `H<=Hmax`. The following split avoids that error.

Given a desired derivative accuracy `omega>0`:

1. Fix `U` large enough that the `C/U^2` core and tail bounds are smaller
   than a chosen fraction of `omega`, and that all bootstrap margins hold.
2. Choose a fixed `hstar=hstar(U)>0`, less than `Hmax`, so that
   `U*sqrt(2hstar)` is smaller than a fixed positive lower bound for the
   magnitudes of both outer section coordinates. Also ensure the rescaled
   core's entire compact height domain is contained in the fixed normal
   chart. All these inequalities are strict and possible.
3. On `gamma epsilon^3<=H<=hstar`, choose `epsilon` small enough that
   `epsilon^3<=hstar`. The core and tail arguments apply uniformly, and
   their errors tend to zero apart from the already selected `C/U^2` term.
4. On the **fixed** compact interval `hstar<=H<=Hmax`, the limiting
   trajectory is `h=H+V^2/2>=hstar`. Ordinary smooth ODE dependence and
   transverse event dependence give the actual trajectory and its
   variations uniformly. The limiting multiplier `J` is exactly `1`.
5. Shrink `epsilon` once more to satisfy both pieces. This is one cutoff
   for the original band, whose `Hmax` never depended on `omega`.

For the low-height piece, (3.7), (4.3), and the scalar equation
`J_V=R_h*J` imply

\[
 \sup_{\gamma\epsilon^3\le H\le h_*}
 \left|\log{J(V_i(H))\over Q(H,\epsilon)}\right|
 +\left|\log{J(V_o(H))\over Q(H,\epsilon)}\right|
 \le {C\over U^2}+o_U(1).
\]

On the high-height piece, `J->1` and `Q->1`. Since `omega` was arbitrary,

\[
 {J(V_i(H))\over Q(H,\epsilon)}\to1,\qquad
 {J(V_o(H))\over Q(H,\epsilon)}\to1
 \tag{5.1}
\]

uniformly on the whole band. In particular `J_o/J_i->1`. This proves more
than merely agreement of the compact core multipliers.

## 6. Exact physical section factors

The physical sections are the fixed lines `v=sigma*rho*h`. In the **exact**
normal coordinates they are

\[
 E_\sigma(V,h)=V-1+\sigma\rho h
                  +\nu\rho^2 h^2+C\nu^2\sigma\rho h^2=0.
 \tag{6.1}
\]

Their signed labels as functions of height are

\[
 T_\sigma(h)=(\sigma\rho h-1)^2-2h.
\]

At `epsilon=0`, the selected outer endpoint coordinates are

\[
 V_\sigma^0(H)
   =-{\sigma\over\rho}
      \{1+\sqrt{1+2\sigma\rho-2\rho^2H}\}.
 \tag{6.2}
\]

Condition (1.2) keeps both endpoint transversality factors bounded away
from zero uniformly for `0<=H<=Hmax`:

\[
 1+\sigma\rho V_\sigma^0
   =-\sqrt{1+2\sigma\rho-2\rho^2H}.
\]

Let `V_sigma(H)` be the exact nearby intersection and `J_sigma` the
variation at **fixed** `V`, evaluated there. Differentiating (6.1) and the
trajectory gives the total endpoint-height derivative

\[
 {dh_\sigma\over dH}
     ={J_\sigma\over1+(E_\sigma)_h\mathcal R_\sigma},
 \qquad
 {dt_\sigma\over dH}
     ={T_\sigma'(h_\sigma)\over
                    1+(E_\sigma)_h\mathcal R_\sigma}\,J_\sigma.
 \tag{6.3}
\]

This includes the movement of the physical endpoint and the difference
between fixed `V` and fixed `v` variations. The coefficient derivatives
are exact:

\[
 (E_\sigma)_h=\sigma\rho+2\nu\rho^2h
                                  +2C\nu^2\sigma\rho h,
 \quad T_\sigma'=2\sigma\rho(\sigma\rho h-1)-2.
\]

At the limiting endpoint `v=1-V`, the second expression is
`-2(1+sigma rho V)`. Thus the two nonzero factors in (6.3) have the same
limit on each side, including every `H` in the fixed compact range:

\[
 {T_\sigma'(h_\sigma)\over1+(E_\sigma)_h\mathcal R_\sigma}
       \longrightarrow-2.
 \tag{6.4}
\]

The convergence is uniform, by sections 4–5. Both `J_sigma` are strictly
positive, being scalar variational exponentials. Equations (5.1)–(6.4)
prove that both endpoint labels decrease strictly and that

\[
 {dt_i\over dH}=-2Q(H,\epsilon)(1+o(1)),\qquad
 {dt_o\over dH}=-2Q(H,\epsilon)(1+o(1)).
 \tag{6.5}
\]

Taking their ratio proves (1.4).

## 7. What this says about physical recentering and cycle counts

For a **fixed compact** core-height coordinate `z=H/epsilon^3` inside
`(0,infinity)`, (6.5) supplies the sharper derivative transfer

\[
 {d\over dz}t_{i/o}(\epsilon^3z)
       =-2\epsilon^3\left(1+{L\over z}\right)+o(\epsilon^3),
 \qquad L=-\lambda_0\ge0.
\]

Choose any fixed reference `z_*>0`. Integration on a compact interval gives

\[
 t_{i/o}(\epsilon^3z)=t_{i/o}(\epsilon^3z_*)
 -2\epsilon^3\{z-z_*+L\log(z/z_*)\}+o_{C^1}(\epsilon^3).
 \tag{7.1}
\]

The anchors in this formula are **actual** section intersections, not a
truncated expansion of a guessed threshold. Their separate explicit
asymptotics remain useful, but are unnecessary for the derivative ratio.
Formula (7.1) explains the logarithmic shape of the core's section
coordinate and gives a genuine derivative-level overlap with the physical
sections.

For each fixed positive `epsilon`, `H -> t_i(H)` is a strictly decreasing
smooth parametrization of one interval. The captured regular map's
admitted domain is also an interval, by the opposite strict endpoint
derivatives of the endpoint-matching function. Their intersection is
connected. On it, provided the already established regular-coordinate
bound `0<H_reg'<=theta<1` applies, (1.4) gives

\[
 (H_{\rm reg}-D_\epsilon)'\le-(1-\theta)/2<0
\]

for sufficiently small `epsilon`. The exact physical cycle equation is
`H_reg(t)=D_epsilon(t)`, since the physical closing passage is the inverse
of the normal-forward passage used here. There is at most one zero on
that connected admitted interval.

If such a cycle exists, it is hyperbolic and attracting in physical time.
Its local return map is `P=D_epsilon^{-1} composed with H_reg`. At the
fixed point, the same strict slope margin gives

\[
 0<P'={H_{\rm reg}'\over D_\epsilon'}
       \le {2\theta\over1+\theta}<1.
\]

This is a conditional stability conclusion, not an existence assertion.

This statement retains the original regular capture and section itinerary
assumptions. It does not capture all nearby cycles. In particular, the
center-height layer `H/epsilon^3 -> 0` has `eta -> 0`, invalidating the
uniform positive-height core argument. In that layer the delayed
entry-exit mechanism can survive, and the bounds above supply no uniform
transition or additional count. Although `lambda0=0` is included in the
proved band, the simultaneous limit `lambda0 -> 0`, `H/epsilon^3 -> 0`
has no positive core-height margin and is not settled here.

## 8. Relation to the published source

Huzak–Kristiansen's entry-exit theorem treats compact positive base-point
intervals in its prescribed parameter region; its explicit fast-fiber
formula and quadratic-family normal form supply the surrounding coordinate
context. The present center-height argument is a direct calculation in the
exact family, not an application of that theorem at a zero base point.

Primary source: [Huzak–Kristiansen, Theorem 2.4 and section 6](https://documentserver.uhasselt.be/bitstream/1942/49936/1/On_entry_exit_formulas_for_degenerate_turning_point_problems_in_planar_slow_fast_systems.pdf).
The argument above uses finite-interval smooth ODE dependence, scalar
variation equations, and explicit continuation bounds. It does not assert
a new unquoted singular-transition theorem or formal analytic verification.

## 9. Finite checks and reproduction

The [grazing benchmark](../../benchmarks/hilbert16_grazing_passage.py)
replays 22 exact identities for the adaptive scaling, first integral,
frozen-scale variation, tail bounds, inverse-coordinate derivatives,
physical event factors, and anchored logarithmic derivative. Rational
examples exercise the strict height barriers and slope comparison.

```sh
uv run --no-sync python benchmarks/hilbert16_grazing_passage.py \
  --output artifacts/hilbert16/grazing_passage.json
uv run --no-sync python -O benchmarks/hilbert16_grazing_passage.py \
  --output artifacts/hilbert16/grazing_passage_optimized.json
```

These checks support the displayed algebra. The continuation, uniform
limit estimates, and cycle count remain a written analytic proof. The
benchmark does not compute or certify the physical small-parameter cutoff,
and it sets no formal-verification flag.
