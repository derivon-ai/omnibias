# Actual zeroth-order matching on the shrinking physical escape scale

This is a written analytic estimate for the actual canonical quadratic family
and its established outer physical sections. It is not a novelty claim, a
formal verification of the estimates, a derivative-in-kappa theorem, or full
graphic cyclicity. In particular, the proof below does not obtain its result
by using compact-kappa convergence pointwise on an expanding interval.

## 1. Quantitative two-parameter statement

Fix the existing canonical chart with physical chart parameter `r=-1`,
`C` in a compact subset of `(1,infinity)`, `A=1+nu*Abar` with bounded `Abar`,
and `epsilon=nu*kappa_scale`, where the analytic factor `kappa_scale` stays
uniformly positive. Fix the physical section size `rho>0` first. Take a
compact coefficient set

\[
 L=-\lambda_0>0,\qquad \lambda=\lambda_1,\qquad
 \Delta=4L-\lambda^2\ge\chi>0.
 \tag{1.1}
\]

To avoid confusing the fixed chart parameter with a compactification, write

\[
 \omega=e^{-\kappa},\qquad u=\epsilon e^\kappa,
 \qquad\epsilon=u\omega,
 \qquad H=\epsilon^3e^{-\kappa/\epsilon}.
 \tag{1.2}
\]

There exist common positive `omega0,u0,C_match` such that, whenever
`0<omega<=omega0` and `0<u<=u0`, the actual orbit through `(V,h)=(0,H)`
reaches both selected outer physical section branches, and its physical
normal-forward derivative satisfies

\[
 \boxed{
 \left|\log D_\epsilon'(t_i(\kappa))
             -{2\pi\lambda\over\sqrt\Delta}\right|
       \le C_{\rm match}(\omega+u).}
 \tag{1.3}
\]

Since the coefficient set is compact, the same bound, with a different
constant, holds for

\[
 \left|D_\epsilon'(t_i(\kappa))
                   -\exp{2\pi\lambda\over\sqrt\Delta}\right|.
 \tag{1.4}
\]

In particular the actual slope tends uniformly to the limiting infinite-band
value whenever `kappa->infinity` and `epsilon*exp(kappa)->0`. The two small
parameters may tend to zero at arbitrary relative rates. No derivative in
`kappa` is asserted here.

## 2. Exact field and expanding-domain bounds

Use the exact normal coordinates, not a height-independent replacement:

\[
 \dot V=f(V,\epsilon)+h g(V,h,\epsilon),\qquad
 \dot h=-Vh,
\]
\[
 q=-f=\epsilon^3L-\epsilon^2\lambda V
                     -\epsilon V^2\zeta(V,\epsilon),\qquad
 k=-g.
 \tag{2.1}
\]

The canonical normalization has `zeta(0,epsilon)=-1` exactly. On the fixed
physical compact set containing the existing section itinerary,

\[
 k=1+O(\epsilon),\quad k_V=O(\epsilon),\quad
 k_h=O(\epsilon^2),\quad
 \zeta(V,\epsilon)+1=O(V).
 \tag{2.2}
\]

All coefficients and the finite number of derivatives used below have
common bounds. These estimates follow from the exact analytic normal change;
in particular the first-order coefficient of `g` is independent of `h`.
We retain every actual occurrence of `k(V,h,epsilon)` below. Only its bounds,
not a substitution `k=1`, are used.

For radial sign `sigma=+-1`, put

\[
 V=\sigma uX,\quad h=\epsilon u^2Y,\quad
 b_\sigma(X)=L\omega^2-\sigma\lambda\omega X
                   -X^2\zeta(\sigma uX,\epsilon),
\]
\[
 \widehat k_\sigma(X,Y)
       =k(\sigma uX,\epsilon u^2Y,\epsilon).
 \tag{2.3}
\]

This is the same expanding radial domain as `x=X/omega`, `y=Y/omega^2`
in the intermediate scaling `V=sigma*epsilon*x`, `h=epsilon^3*y`.
The exact radial equation is

\[
 \epsilon Y_X={XY\over b_\sigma(X)+\widehat k_\sigma(X,Y)Y},
 \qquad Y(0)=\omega^2 e^{-\kappa/\epsilon}.
 \tag{2.4}
\]

For a common fixed `K`, strict discriminant and compactness give a constant
`b_*>0` such that

\[
 L\omega^2-\sigma\lambda\omega X+X^2
                  \ge b_*(\omega^2+X^2).
\]

The perturbation in (2.3) has magnitude at most `C*u*X^3`. Choose `u0`
after `K`, sufficiently small. On `0<=X<=K` this gives

\[
 b_\sigma\ge\tfrac12 b_*(\omega^2+X^2),\qquad
 |(b_\sigma)_X|\le C(\omega+X),
\]
\[
 \tfrac12\le\widehat k_\sigma\le2,
 \qquad |(\widehat k_\sigma)_X|\le C\epsilon u.
 \tag{2.5}
\]

The last derivative is a partial derivative at fixed `Y`; it equals
`sigma*u*k_V`. Equation (2.4) gives

\[
 0\le(\epsilon Y)_X\le2X,
 \qquad h=\epsilon u^2Y\le C_Ku^2.
 \tag{2.6}
\]

Thus the full positive solution exists on this radial interval, remains in
the analytic domain, and is strictly increasing for `X>0`. This also
justifies all inverse-height integrals used below. At the endpoint `X=0`
they are interpreted by a one-sided limit.

## 3. The exact level Y=omega^2 and uniform escape localization

Set

\[
 S_{\epsilon,\sigma}(X)=\int_0^X{s\over b_\sigma(s)}\,ds.
\]

Let `B_sigma(x)=x^2-sigma*lambda*x+L` and
`S_sigma(x)=integral_0^x s/B_sigma(s) ds`. Its elementary primitive gives

\[
 S_\sigma(x)=\log x+c_\sigma+O(1/x),
 \qquad d_\sigma=e^{-c_\sigma},
 \tag{3.1}
\]

uniformly on the compact coefficient set. Explicitly,

\[
 d_\sigma=\sqrt L\exp\!\left[
 -{\lambda\over\sqrt\Delta}\arctan{\lambda\over\sqrt\Delta}
 -{\sigma\pi\lambda\over2\sqrt\Delta}\right].
 \tag{3.2}
\]

The actual perturbation satisfies

\[
 |S_{\epsilon,\sigma}(X)-S_\sigma(X/\omega)|\le C_Ku.
 \tag{3.3}
\]

Indeed, before normalization the quadratics differ by at most
`C*epsilon*x^3`, and the difference of the two integrands is bounded by
`C*epsilon*x^4/(1+x^2)^2<=C*epsilon`. Integration to `X/omega` gives
`C*u*X`. Hence on any fixed `0<a<=X<=K`,

\[
 S_{\epsilon,\sigma}(X)-\kappa
                  =\log X+c_\sigma+O(\omega+u).
 \tag{3.4}
\]

Equation (2.4) yields the following exact identity along the actual solution:

\[
 S_{\epsilon,\sigma}(X)
 =\epsilon\log{Y(X)\over Y(0)}
 +\epsilon\int_{Y(0)}^{Y(X)}
        {\widehat k_\sigma(X(Y),Y)\over b_\sigma(X(Y))}\,dY.
 \tag{3.5}
\]

Before and at the level `Y=omega^2` the final integral, multiplied by
`epsilon`, lies in `[0,C*epsilon]`, because
`b_sigma>=c*omega^2` and the integration length is at most `omega^2`.
In particular, if `X_1` is the first point with `Y(X_1)=omega^2`, then

\[
 \boxed{S_{\epsilon,\sigma}(X_1)=\kappa+O(\epsilon),
                  \quad O(\epsilon)\ge0.}
 \tag{3.6}
\]

There are common bounds `0<d_min<=d_sigma<=d_max`. Choose, for example,
a fixed lower bracket below `d_min/2` and an upper bracket above `2*d_max`,
and later choose `K` a fixed distance beyond that upper bracket. Formula
(3.4) gives fixed negative/positive gaps at these brackets. The lower gap
and the nonnegative loss in (3.5) exclude an earlier hit. If no hit had
occurred by the upper bracket, (3.5) would give
`S_epsilon<=kappa+C*epsilon`, contradicting the positive gap. Thus the
event exists with common margins, and (3.4)–(3.6) imply

\[
 \boxed{X_1=d_\sigma+O(\omega+u+\epsilon).}
 \tag{3.7}
\]

This estimate controls the actual height loss on the expanding domain.
It is not a substitution of a limiting escape point in the actual orbit.

## 4. The compensated exponent through the escape region

The exact radial compensated exponent is

\[
 \Psi_\sigma(X)=\int_0^X
 { (b_\sigma)_s+Y(s)(\widehat k_\sigma)_s
                         \over b_\sigma+\widehat k_\sigma Y}\,ds,
 \tag{4.1}
\]

where the partial derivative of `khat` is at fixed height. Its difference
from `log[b_sigma(X)/b_sigma(0)]` is exactly

\[
 -\int_0^X { (b_\sigma)_s\over b_\sigma}
       {\widehat k_\sigma Y\over b_\sigma+\widehat k_\sigma Y}\,ds
 +\int_0^X {Y(\widehat k_\sigma)_s
                         \over b_\sigma+\widehat k_\sigma Y}\,ds.
 \tag{4.2}
\]

Choose a fixed `a>0` below the lower event bracket with
`log a+c_sigma<=-c_a<0` uniformly. The differential upper bound
`epsilon*log(Y/Y(0))<=S_epsilon` and (3.4) show

\[
 Y(X)\le\omega^2e^{-c_a/(2\epsilon)}\quad(0\le X\le a)
 \tag{4.3}
\]

for small common cutoffs. The first error in (4.2) on `[0,a]` is bounded by

\[
 C\omega^2 e^{-c_a/(2\epsilon)}
       \int_0^a {\omega+X\over(\omega^2+X^2)^2}\,dX
 \le C e^{-c_a/(2\epsilon)}.
\]

The last bound follows by `X=omega*t`; the rescaled integral is bounded
by `integral_0^infinity (1+t)/(1+t^2)^2 dt=pi/4+1/2`.
On `[a,X_1]` its bound is `C*omega^2`,
using `Y<=omega^2` and the uniform positive lower bound for `b_sigma`.
The second error is at most `C*epsilon*u` throughout. Consequently

\[
 \Psi_\sigma(X_1)=
 \log{b_\sigma(X_1)\over L\omega^2}
 +O\!\left(\omega^2+\epsilon u+
                e^{-c_a/(2\epsilon)}\right).
 \tag{4.4}
\]

By (3.7), the logarithm on the right equals
`2*kappa+2*log(d_sigma)-log(L)+O(omega+u+epsilon)`.

It remains to bound the actual exponent after `X_1`; setting the height
to zero would be incorrect here. While `omega^2<=Y<=1`, equations
(2.4)–(2.5) and the event margins give

\[
 {d\over dX}\log Y\ge {c\over\epsilon}.
\]

The level `Y=1` is therefore reached at `X_2` with

\[
 0\le X_2-X_1\le C\epsilon\kappa.
 \tag{4.5}
\]

Since `epsilon*kappa=u*omega*log(1/omega)<=u/e`, choose `u0` small
enough that this remains strictly inside the common fixed radial endpoint.
The integrand in (4.1) is uniformly bounded on `[X_1,X_2]`, so its
contribution is `O(epsilon*kappa)`.

After `Y>=1`, the actual equation gives `Y_X>=c/epsilon`. Hence

\[
 Y(X)\ge1+{c(X-X_2)\over\epsilon},
\]

and the remaining contribution in (4.1) is at most

\[
 C\int_{X_2}^K{dX\over1+c(X-X_2)/\epsilon}
              +C\epsilon u
 \le C\epsilon\log(1/\epsilon)+C\epsilon u.
 \tag{4.6}
\]

The residual gap `K-X_2` is uniformly positive; (2.6) and the same lower
slope bound show

\[
 0<c_K\le\epsilon Y(K)\le C_K.
 \tag{4.7}
\]

Combining these bounds gives

\[
 \Psi_\sigma(K)=2\kappa+2\log d_\sigma-\log L
                    +O(\omega+u).
 \tag{4.8}
\]

For completeness the logarithmic estimates used to simplify the remainder
are uniform, without restricting the relative rate of the two parameters:

\[
 \epsilon\log(1/\epsilon)
 =u\omega\log(1/u)+u\omega\log(1/\omega)
                 \le(\omega+u)/e.
 \tag{4.9}
\]

The exponentially small term in (4.4) is also absorbed by `C*epsilon`,
and hence by `C*(omega+u)`. The individual exponents grow
as `2*kappa`; (4.8) is an absolute bound after subtracting that term.

## 5. Actual tails to the physical sections

At the radial exit `|V|=K*u`, (4.7) gives
`c*u^2<=h<=C*u^2`, and therefore a common positive bound for `h/V^2`.
In this paragraph write `s=|V|` for the physical radial coordinate.
The physical scalar equation is

\[
 {dh\over ds}={s h\over q(\sigma s)+k(\sigma s,h)h}.
 \tag{5.1}
\]

Bootstrap fixed smaller/larger barriers `c_1*s^2<=h<=c_2*s^2` with
`c_1<1/2<c_2`, chosen also to enclose the exit ratios strictly. On such
a tube,

\[
 {|q|\over h}+|k-1|
 \le C\left(\epsilon+{\epsilon^2\over s}
                         +{\epsilon^3\over s^2}\right)
 \le C\epsilon\qquad(s\ge Ku),
 \tag{5.2}
\]

because `epsilon/u=omega<=omega0`. The radial derivative in (5.1) is
`s*(1+O(epsilon))`, so it improves both barriers. Continuation reaches
a fixed neighborhood of each prescribed outer event. On every fixed
physical range,

\[
 h(s)=s^2/2+O(u^2+\epsilon).
 \tag{5.3}
\]

The limiting zero-fiber outer intersections are transverse. Equation
(5.3) and ordinary event continuity therefore give the actual events on
the same selected branches. The ambient compact normal chart is chosen
to contain those finite connecting paths; `rho` is fixed before the
small-parameter cutoffs. In particular no claim that a large fixed physical
section lies inside a small slow interval is needed.

The tail contribution to the physical compensated exponent obeys

\[
 \int_{Ku}^{s_{\rm event}}
 \left|{\partial_s q(\sigma s)
               +h\partial_s k(\sigma s,h)
                      \over q(\sigma s)+k(\sigma s,h)h}\right|ds
 \le C\left[\epsilon\log(1/u)+{\epsilon^2\over u}+\epsilon\right]
 \le C(\omega+u).
 \tag{5.4}
\]

Here the endpoint remains in a fixed compact interval bounded away from
zero. The estimates use `q_V=O(epsilon^2+epsilon*|V|)` on the fixed chart,
`k_V=O(epsilon)`, and (5.2). In particular `epsilon*log(1/u)<=omega/e`.
The full incoming/outgoing exponents thus satisfy

\[
 \Psi_i=2\kappa+2\log d_+-\log L+O(\omega+u),
\]
\[
 \Psi_o=2\kappa+2\log d_--\log L+O(\omega+u).
 \tag{5.5}
\]

## 6. Exact physical events and cancellation

The actual physical section equations and labels are

\[
 E_\tau(V,h)=V-1+\tau\rho h+\nu\rho^2h^2
                               +C\nu^2\tau\rho h^2=0,
\]
\[
 T_\tau(h)=(\tau\rho h-1)^2-2h.
 \tag{6.1}
\]

The incoming outer branch has `tau=-1`, radial sign `sigma=+1`; the
outgoing branch has `tau=+1`, radial sign `sigma=-1`. Normal time moves
from the positive radial side to the negative one; the physical singular
closing is its inverse, as in the existing section transport proof.

Let `J` be height variation at fixed `V`, and write
`Rcal=V*h/(q+k*h)`. The exact moving-event derivative is

\[
 {dt_\tau\over dH}=\mathcal K_\tau J_\tau,
 \qquad
 \mathcal K_\tau={T_\tau'(h)\over1+(E_\tau)_h\mathcal R}.
 \tag{6.2}
\]

On these transverse outer events, (5.2) implies
`Rcal=V+O(epsilon)` and `h/(q+k*h)=1+O(epsilon)`. The exact event equation
also gives

\[
 T_\tau'(h)=-2(1+\tau\rho V)+O(\epsilon),\qquad
 1+(E_\tau)_h\mathcal R=1+\tau\rho V+O(\epsilon).
\]

The common transversality margin therefore yields

\[
 \mathcal K_i=-2+O(\epsilon),\qquad
 \mathcal K_o=-2+O(\epsilon).
 \tag{6.3}
\]

The exact compensated scalar variation identity remains valid, including
the actual height dependence of `k`:

\[
 J(V)={q(0)+k(0,H)H\over H}\,
                   {h\over q+kh}\,e^{\Psi(V)}.
 \tag{6.4}
 \]

This physical identity is established before rescaling its integral. When
evaluating the derivative at the reference center height `H`, freeze
`omega,u` at that reference trajectory. No differentiation of an
`H`-dependent change of scale is used to compute the physical variation.

The common, extremely large prefactor cancels before any approximation:

\[
 D_\epsilon'(t_i(\kappa))
 ={\mathcal K_o\over\mathcal K_i}
 {h_o/(q_o+k_oh_o)\over h_i/(q_i+k_ih_i)}
                                      e^{\Psi_o-\Psi_i}.
 \tag{6.5}
\]

All prefactors are positive after taking the ratios and are `1+O(epsilon)`.
Using (5.5),

\[
 \log D_\epsilon'
       =2\log(d_-/d_+)+O(\omega+u)
       ={2\pi\lambda\over\sqrt\Delta}+O(\omega+u).
\]

This proves (1.3)–(1.4) for the actual physical derivative.

## 7. Scope of the result and the remaining matching problem

The result covers the complete two-parameter corner
`exp(-kappa)->0`, `epsilon*exp(kappa)->0`, with a quantitative uniform
value estimate for the physical slope. It does not prove a bound on a
kappa derivative of the error, and it does not resolve the regime where
`epsilon*exp(kappa)` stays away from zero. The latter regime reaches
nonshrinking physical escape coordinates and requires the full slow-fast
entry transport.

If the regular comparison derivative is uniformly separated from the value
`exp(2*pi*lambda/sqrt(Delta))`, this estimate supplies a strict derivative
sign for the composed displacement on a sufficiently small corner.
On a connected admitted interval that yields at most one zero; exclusion
of every zero additionally needs a displacement value or anchoring argument.
Resonant equality of the regular slope with that limiting value is not
settled by this zeroth-order theorem. No full-graphic or all-degree claim
is made.

The exact canonical field and section formulas are established in the
[singular transport](HILBERT16-SINGULAR-TRANSPORT.md),
[grazing passage](HILBERT16-GRAZING-PASSAGE.md), and
[exponential passage](HILBERT16-EXPONENTIAL-PASSAGE.md) notes.
The limiting primitive (3.1)–(3.2) is elementary; its uniform analytic
compactification is in the [limiting-family proof](HILBERT16-MATCHING-LIMIT.md). The argument here supplies the previously missing expanding-domain
estimate instead of assuming it follows from that limiting calculation.

## 8. The regular multiplier on the same signed sections

Here the physical small parameter is `nu`, the canonical one remains
`epsilon=nu*kappa_scale`, and `R=rho/nu`. Set
`c(C)=exp(-4*pi/sqrt(C-1))`. The
[regular derivative theorem](HILBERT16-REGULAR-JETS.md) gives the exact
first variation `A=A0*E`, with

\[
 |\log E(R)|\le2K_w\delta R=2K_w(\rho+\zeta),
 \qquad\delta=\nu+\zeta/R.
\]

Writing `q_R(x)=(x^2+2x+C)/2` and using that theorem's auxiliary
function `D_reg(x)`, the actual normalized regular derivative is

\[
 G_R'=\frac{q_R(R)}{q_R(-R)}
       \frac{D_{\rm reg}(-R)}{D_{\rm reg}(R)}E(R).
 \tag{8.1}
\]

For `R>=4`, the logarithm of the first factor lies in `[0,8/R]`.
The logarithm of the second factor minus `log c(C)` lies in
`[0,8R/(R^2-1)]`, by `atan x<=x` for positive x. Hence

\[
 |\log(G_R'/c(C))|\le19/R+2K_w(\rho+\zeta).
 \tag{8.2}
\]

This is the actual compensated regular trajectory with the splitting
parameter held fixed during differentiation.

The exact physical signed-coordinate maps are

\[
 Z_{\tau,\nu}(t)=\frac{\rho^2Z_\tau(t)+C\nu^2}{\Delta_\tau},
 \quad\Delta_\tau=\rho^2+2\tau\nu\rho+C\nu^2,
\]
\[
 Z_\tau(t)=\frac2{1+\tau\rho+\sqrt{1+2\tau\rho+\rho^2t}}-1.
 \tag{8.3}
\]

Thus `H_reg=Z_{+,nu}^{-1} o G_R o Z_{-,nu}`. Suppose the input and
regular output labels both have magnitude at most one. Choose
`rho<=1/16` and `nu<=rho^2/(1+Cmax)`. With
`A_tau(t)=1+2tau*rho+rho^2*t` and
`J_tau(t)=1+tau*rho+sqrt(A_tau(t))`,

\[
 |Z_\tau'(t)|={\rho^2\over\sqrt{A_\tau(t)}J_\tau(t)^2}.
\]

Use `|A_tau-1|<=3rho`, `|sqrt(A_tau)-1|<=2rho`,
`|Delta_tau/rho^2-1|<=3rho`, and
`|log(1+s)|<=2|s|` for `|s|<=1/2`. The logarithm of the coordinate
derivative ratio has absolute value at most `30rho`: the square roots,
squared J factors and Delta factors contribute at most `6rho`, `12rho`
and `12rho` respectively. Consequently

\[
 \boxed{|\log(H_{\rm reg}'/c(C))|
       \le30\rho+19/R+2K_w(\rho+\zeta).}
 \tag{8.4}
\]

One may choose `zeta=8rho`: (8.3) puts every signed label with magnitude
at most one inside that normalized section box under these cutoffs.
Choosing rho sufficiently small also supplies the endpoint theorem's
sector conditions. Then nu is reduced as necessary. Thus (8.4) can be
made arbitrarily small uniformly over the coefficient compacts and the
captured input domain. It also gives a positive lower regular-slope bound.

## 9. A joined count away from limiting multiplier resonance

Restrict first to `lambda=lambda1<0`, and define

\[
 \Gamma(C,L,\lambda)=\frac{2\pi\lambda}{\sqrt{4L-\lambda^2}}
                       +\frac{4\pi}{\sqrt{C-1}}.
 \tag{9.1}
\]

This difference of limiting logarithmic multipliers has the exact identity

\[
 \Gamma=
 \frac{2\pi[16L-(C+3)\lambda^2]}
 {\sqrt{C-1}\sqrt\Delta[2\sqrt\Delta-\lambda\sqrt{C-1}]}.
 \tag{9.2}
\]

The denominator is positive. The limiting resonance for negative lambda
is therefore `(C+3)*lambda1^2=16L`. This is a limiting multiplier
equality, not an exact center or an actual bifurcation locus.

Fix a coefficient compact with either `Gamma>=gamma>0` throughout or
`Gamma<=-gamma<0` throughout. Choose the regular sections so that (8.4)
is at most `gamma/4`, retaining the original strict regular contraction
and endpoint sector. Let `kappa_*>0` be the corresponding initial-band
cutoff and set `kappa_a=kappa_*/2`. Choose a fixed large
`kappa_b>kappa_*+1` and a fixed small
`u0>0` such that

    exp(-kappa_b)<=omega0,
    C_match*(exp(-kappa_b)+u0)<gamma/4.

The [compact exponential theorem](HILBERT16-EXPONENTIAL-PASSAGE.md)
is used on the fixed band `[kappa_a,kappa_b+1]`, supplying both overlaps
after reducing epsilon. Also require `log(u0/epsilon)>kappa_b+1`.
On the growing interval

\[
 \kappa_b\le\kappa\le\log(u_0/\epsilon),
 \tag{9.3}
\]

the joint estimate (1.3) gives `D_epsilon'-H_reg'` the sign of Gamma.
The upper endpoint of (9.3) tends to infinity; its estimate does not
follow from fixed-band convergence.

If `Gamma>=gamma`, the limiting compact-band slope satisfies
`R(kappa)>=R_infinity` because lambda is negative. Its uniform value
convergence gives `D_epsilon'>H_reg'` throughout the fixed band as well.
The initial thin and grazing estimates supply the same inequality up to
that band. Hence `F=H_reg-D_epsilon` is strictly decreasing on the whole
joined admitted domain and has at most one zero.

If `Gamma<=-gamma`, compactness bounds lambda strictly below zero.
On the fixed positive kappa band the actual singular curvature is
negative of order `epsilon^(-2)`, so `F''>0` there. Initially `F'<0`
by the thin and grazing estimates; on (9.3), `F'>0`. The fixed band
overlaps both regions. Thus F' has at most one zero across their union,
and Rolle's theorem bounds the zeros of F by two. Separate local counts
are not added.

The exact endpoint formula has `t_i,H<0` in every region. The physical
maps agree on overlaps by uniqueness, so the whole center-height band
maps monotonically to one input interval. Its intersection with the
connected captured regular domain is an interval. The resulting bound is

\[
 \boxed{\epsilon^3(\epsilon/u_0)^{1/\epsilon}\le H\le H_{\max}:\quad
 \begin{cases}
  \text{at most one captured cycle},&\Gamma\ge\gamma,\\
  \text{at most two captured cycles},&\Gamma\le-\gamma.
 \end{cases}}
 \tag{9.4}
\]

Here Hmax is fixed and sufficiently small, u0 is fixed after the
coefficient compacts and sections, and epsilon is sufficiently small.
The signed labels lie in the bounded range required in section 8 after
shrinking Hmax and u0, by (5.3) and the prior initial-band estimates.
Restricting the regular output to magnitude at most one also preserves
an interval, since the regular map is increasing.

The nonnegative-lambda case already has the stronger all-height admitted
comparison in [height comparison](HILBERT16-HEIGHT-COMPARISON.md).
The resonant negative-lambda locus, nonshrinking physical escape,
vanishing discriminant or L margins, and other itineraries remain outside
(9.4). This does not count cycles for all quadratic fields or all degrees.

## Finite verification

```bash
python benchmarks/hilbert16_joint_matching.py --output artifacts/hilbert16/joint_matching.json
```

The benchmark replays the scaling, multiplier and resonance algebra and
uses the existing interval register to bound resonance on coefficient
boxes. These checks do not formally verify the analytic theorem or
compute its uniform smallness cutoffs.
