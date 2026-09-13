# Quadratic unfolding: parameter growth and a uniform chart bound

This companion to [the full Hilbert-16 program](HILBERT16.md) records two
proved local estimates and a failed shortcut in singular-map matching.
Neither a complete return-map bound nor quadratic cyclicity is established.
These are written analytic arguments with exact algebra and interval checks;
no novelty or formal verification of their analytic conclusions is claimed.

## 1. Exact first variations of the regular passage

Use the family, fixed sections x=-R and x=R, and normalized coordinate z=w/q
from the main note. Fix C>1 and the incoming value z(-R)=0. Differentiate at
A=1, mu=0, writing alpha=A-1. With

\[
q=(x^2+2x+C)/2,\quad a=\sqrt{C-1},\quad
\theta(x)=\arctan((x+1)/a),\quad
D(x)=\exp\big(4(\theta(x)-\pi/2)/a\big),
\]

one has D'/D=2/q, D(+infinity)=1, and
D(-infinity)=c=exp(-4*pi/a). The parameter variation v of w satisfies

\[
v'=2xv/q+b/q,\quad v(-R)=0,
\]

where the exact forcing polynomials are

\[
b_\alpha=-x^2,\quad b_2=-x^2(x^2-C)/2,\quad
b_3=(C^2-x^4)/4,\quad b_1=-x(x^2-C)^2/4.
\]

The variation of the denominator in the x-time equation contributes zero
because the base numerator wdot vanishes. Solving this linear equation gives

\[
\partial_pG_R(0)=\frac{q(R)}{D(R)}
       \int_{-R}^R\frac{D(x)b_p(x)}{q(x)^3}\,dx. \tag{1}
\]

The section locations, C, and incoming coordinate are fixed in this formula.
Moving sections or parameter-dependent coordinate changes require additional
terms.

The three convergent integrals have exact polynomial antiderivatives. Put

\[
P_\alpha=-\frac{x^4+8x^3+2(C+12)x^2+C(C+12)}{16(C+3)},
\]
\[
P_2=(C+6)P_\alpha+x^3+3x^2+3C/2,\qquad
P_3=3P_\alpha+x^3/2+3x^2/2+Cx/2+C.
\]

Exact polynomial arithmetic verifies qP'_p-2xP_p=b_p. Consequently

\[
\left(\frac{DP_p}{q^2}\right)'=\frac{Db_p}{q^3}.
\]

Taking endpoints gives full-line integrals
-(1-c)/(4(C+3)), (C+6) times this value, and three times this value.
Since q(R)/(R^2 D(R)) tends to 1/2,

\[
\lim_{R\to\infty}R^{-2}
 (\partial_\alpha G_R,\partial_{\mu_2}G_R,\partial_{\mu_3}G_R)
=-\frac{1-c}{8(C+3)}(1,C+6,3). \tag{2}
\]

The exceptional source has unequal logarithmic tails:
Db_1/q^3=-2/x+O(x^-2) at positive infinity and
Db_1/q^3=-2c/x+O(x^-2) at negative infinity. Thus

\[
\int_{-R}^R Db_1/q^3=-2(1-c)\log R+O(1),\qquad
\lim_{R\to\infty}\frac{\partial_{\mu_1}G_R}{R^2\log R}=-(1-c).
\tag{3}
\]

The tail estimates are uniform on fixed compact C-intervals in (1,infinity).
They do not extend the regular-passage proof to C=1.

### Bounds on every cutoff beyond a fixed value

For R>=R0>=6 and C<=R0^2, write (1) divided by R^2 as p_R times
the truncated integral, where

\[
p_R=\left(\frac12+\frac1R+\frac{C}{2R^2}\right)
\exp\left(\frac4a\arctan\frac a{R+1}\right).
\]

Using arctan(u)<=u and monotonicity of the positive factors gives

\[
\frac12\le p_R\le
\left(\frac12+\frac1{R_0}+\frac{C}{2R_0^2}\right)e^{4/(R_0+1)}.
\]

Beyond |x|>=R, b_alpha,b_2,b_3 are nonpositive. Bounds
0<D<=1, q>=(x+1)^2/2 and |x|/|x+1|<=2 imply absolute integrand
bounds 32/|x+1|^4, 64/|x+1|^2 and 32/|x+1|^2. The combined
omitted tails therefore have magnitudes at most

\[
T_\alpha=\frac{64}{3(R_0-1)^3},\qquad
T_2=\frac{128}{R_0-1},\qquad T_3=\frac{64}{R_0-1}.
\]

Each truncated integral belongs to its full-line value plus [0,T_p].
Multiplication by the positive interval for p_R proves the implementation
in `outer_first_variation_ray`. For C in [2,201/100] and every R>=10000,
all three interval upper endpoints are negative. The logarithmic direction
is excluded from this integrable-tail interface.

### From first variations to nonlinear compensation

With mu1=nu^2*m1, mu2=nu*m2, mu3=nu*m3 and R=rho/nu, the linearized
candidate for keeping the regular displacement bounded is

\[
\alpha+\nu[(C+6)m_2+3m_3]
 +8(C+3)\nu^2m_1\log(1/\nu)=O(\nu^2). \tag{4}
\]

Equation (4) records only a balance of first variations. The quadratic
forcing supplies an additional logarithmic term proportional to
m2*(2*m2+m3), as proved in the
[compensation calculation](HILBERT16-COMPENSATION.md). The subsequent
[uniform nonlinear proof](HILBERT16-UNIFORM-PASSAGE.md) and
[common-section extension](HILBERT16-ENDPOINT-PASSAGE.md) establish the
corrected compensation on an explicit small sector with R=rho/nu. They
prove a common passage domain and a uniform remainder for the actual
nonlinear trajectory. The singular closing transition and higher section
jets needed for its complete composition remain unresolved. Without the
compensation, an order-nu splitting is amplified by R^2 in the normalized
coordinate.

## 2. An exact chart for all normalized unfolding directions

For nu>0 set X=nu*x, Y=nu^2*y, tau=t/nu. Direct substitution gives

\[
X_\tau=X^2-Y+(m_2+m_3)XY+m_1Y^2+A\nu X,
\]
\[
Y_\tau=XY+m_3Y^2+\nu X^2+C\nu^2X.
\]

On Y>0 put v=X/Y, z=1/Y and d sigma/d tau=Y. Both time changes are
positive. The exact family-chart field is

\[
v'=F=m_1+m_2v-\nu v^3-z+A\nu vz-C\nu^2v^2z,
\]
\[
z'=-zN,\qquad N=m_3+v+\nu v^2+C\nu^2vz. \tag{5}
\]

At nu=0 this is a continuous boundary extension; the transformation from
the original plane is no longer invertible. At that boundary set
u=v+m3 and d=m1-m2*m3. Then u'=d+m2*u-z, z'=-u*z and

\[
E=u^2/2+d\log z-z,\qquad E'=m_2u^2.
\]

This elementary identity describes the boundary flow. The following bound
holds for the full field, including positive nu.

**Theorem.** Suppose

\[
0\le\nu\le1/100,\quad 1/2\le A\le3/2,\quad1\le C\le3,
\quad |m_1|,|m_2|,|m_3|\le1.
\]

Every connected nonconstant trajectory segment of (5) staying in
the open box -2<v<2, 0<z<4 has at most one height critical point.
It intersects any horizontal level at most twice. No periodic orbit is
wholly contained in this box.

**Proof.** On the closed box define

\[
\ell=N_v=1+2\nu v+C\nu^2z,\quad
k=1-A\nu v+C\nu^2v^2,
\]
\[
F_v=m_2-3\nu v^2+A\nu z-2C\nu^2vz,\quad
D_*=k\ell+C\nu^2vF_v.
\]

One has N(-2,z)<=-24/25 and N(2,z)>=1. Elementary rational bounds are

\[
k\ge1211/1250,\quad2397/2500\le\ell\le2603/2500,
\quad |F_v|\le1481/1250,
\]
\[
D_*\ge5801091/6250000>0.
\]

Thus N=0 is a unique graph v=V(z), with V'=-C*nu^2*V/ell. Along it,
psi(z)=F(V(z),z) satisfies

\[
\psi'=-D_*/\ell\le-5801091/6507500<-4/5.
\]

At a height critical point z'=0, one has z''=-z*ell*psi. If psi=0
the point is an equilibrium, which a nonconstant trajectory cannot traverse
by uniqueness. All critical points are therefore strict extrema. The
decrease of psi implies that every maximum occurs at a lower height than
every minimum. Consecutive extrema along a trajectory require the opposite
height order, a contradiction. Monotonicity on either side proves the
horizontal-level conclusion. A periodic orbit would require at least two
height extrema. This proves the theorem.

The interval implementation in `quadratic_unfolding` gives slightly sharper
bounds, with ell>0.9599, D_*>0.9305 and psi'<-0.8937. It checks the
whole parameter box, including m3=0, without parameter sampling. Broader
input boxes with unsuccessful inequalities leave those premises unresolved.

This theorem does not guarantee a section-to-section passage, bound travel
time, control derivatives near a saddle or z=0, or bound the zeros of a
return displacement that also traverses other charts.

## 3. A singular-map sign shortcut fails

The entry-exit reduction in section 6 of the
[updated Huzak--Kristiansen manuscript](https://documentserver.uhasselt.be/bitstream/1942/49936/1/On_entry_exit_formulas_for_degenerate_turning_point_problems_in_planar_slow_fast_systems.pdf)
reverses time. Let d>0, a>0, b>0 and s=sqrt(x^2-d), with 0<s<1/b.
Its leading normal-forward outgoing-magnitude map has the form

\[
D(x)=\sqrt{d+\left(\frac{as}{1-bs}\right)^2}.
\]

In the squared fast-fiber coordinate u=s^2 this becomes

\[
E(u)=\frac{a^2u}{(1-b\sqrt u)^2},\qquad
\mathcal SE=-\frac{3b(2-b\sqrt u)}{8u^{3/2}(1-b\sqrt u)^2}<0.
\]

The original-flow closing map uses the inverse. Since

\[
\mathcal S(f^{-1})(f(x))=-\mathcal Sf(x)/f'(x)^2,
\]

its Schwarzian sign is positive in those squared coordinates. In the
unsquared fast-fiber coordinate the map and its inverse are both Mobius
and have zero Schwarzian. In the physical x coordinate, the ordinary
Schwarzian has no parameter-independent sign: with d=1/100, a=2 and b=3,
it is positive at s=1/10 and negative at s=1/1000. These facts rule out
inferring the closing-map sign from the forward-entry sign alone.

A conditional alternative illustrates the exact missing obligation. If E
and the regular G=H^{-1}(cH), H(s)=s/(1+s)^2, were the actual transitions
in one common coordinate, E(s)=G(s) would have at most one positive root
on 0<s<min(1,b^-2): E(s)/s is strictly increasing, whereas

\[
\left(G(s)/s\right)'=
\frac{2G(s)(G(s)-s)}{s^2(1+s)(1-G(s))}<0.
\]

But these coordinates have not been identified in the actual unfolding.
One must transport the genuine transitions and bound the resulting weighted
remainder, including its endpoint vanishing order. The conditional
one-zero calculation is not a limit-cycle theorem.

## Reproduction

The benchmark `benchmarks/hilbert16_regular_passage.py` checks the rational
identities and both interval gates. The two dynamics test modules
`test_quadratic_graphic.py` and `test_quadratic_unfolding.py` independently
check integrals, derivatives, coordinate transformations, dense grids,
random rational samples, and unresolved inputs. Numerical tests verify the
implementation; the infinite-parameter analytic conclusions depend on the
written proofs and remain outside the executed formal loop.


## Second-order follow-through

The [compensation companion](HILBERT16-COMPENSATION.md) now controls the actual
finite-section boundary correction and proves the leading logarithmic second
Taylor coefficient. The [polynomial obstruction theorem](HILBERT16-POLYNOMIAL-OBSTRUCTION.md)
isolates its resonant source by an exact finite algorithm. Both results retain
the distinction between fixed-cutoff Taylor coefficients and a uniform nonlinear
estimate when the cutoff diverges. The singular transition and full cyclicity
obligations remain open.
