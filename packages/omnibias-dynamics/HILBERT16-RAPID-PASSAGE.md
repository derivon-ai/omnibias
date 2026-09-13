# A rapid passage through the zero-fiber boundary

This note treats a shrinking signed-fiber corridor omitted by the
[compact-interior result](HILBERT16-INTERIOR-CYCLICITY.md). Its conclusion
is **at most one cycle in the specified rapid corridor**, with the same
regular-passage capture assumptions. The corridor contains the old zero
input fiber and a positive neighborhood of it on its natural parameter
scale. It does not cover the grazing threshold, every nearby cycle, or
Hilbert's full sixteenth problem. This is a written analytic proof with
finite algebra checks; no novelty or formal analytic verification is claimed.

## 1. Exact family chart and sections

Use the physical quadratic family

\[
 \dot x=Ax-y+x^2+\nu(p-1)xy+\nu^2my^2,\qquad
 \dot y=Cx+x^2+xy-\nu y^2,
\]

where C is in a compact interval inside (1,infinity), nu>0 and
|A-1|<=K_A*nu. Use the exact canonical p,m normalization from
[singular transport](HILBERT16-SINGULAR-TRANSPORT.md), with r=-1 and
lambda in any fixed compact set. **Negative discriminant is unnecessary
for this rapid passage.** The normalization gives p=nu*ptilde and
m=nu*mtilde, with ptilde tending to 3 and mtilde tending to -2.

Fix a sufficiently small rho>0 first and let R=rho/nu. In the family chart
v=nu*x/(nu^2*y), h=1/(nu^2*y), the two physical sections x=sigma*R are
exactly the fixed lines v=sigma*rho*h. Reverse the physical chart time.
The field is

\[
 \dot v=bh-a,\qquad \dot h=h(n+ch),
\]
\[
 a=m+pv-\nu v^3,\quad b=1-A\nu v+C\nu^2v^2,
 \quad n=v-1+\nu v^2,\quad c=C\nu^2v.                  \tag{1}
\]

Where bh-a>0 its actual scalar trajectory equation is

\[
 h_v=\mathcal R(v,h,\nu)=\frac{h(n+ch)}{bh-a}.          \tag{2}
\]

Choose the signed section label t=(v-1)^2-2h. It has no square-root
restriction t>=0. The large section branches are

\[
 v_\sigma(t)=1+\frac\sigma\rho+
 \sigma\sqrt{(1+\sigma/\rho)^2-1+t},\qquad
 h_\sigma(t)=\frac{v_\sigma(t)}{\sigma\rho}.             \tag{3}
\]

For rho<1/2 and small |t| these are analytic positive-height sections.
Put vi=v_-(0)<0 and vo=v_+(0)>0. The section label derivative is

\[
 L_\sigma(v)=2\left(v-1-\frac1{\sigma\rho}\right),       \tag{4}
\]

nonzero at the chosen branches. All compact v ranges used below can
depend on the fixed rho.

## 2. The rapid corridor and its profile

Write s=v-1. The exact canonical factorization is

\[
 a=\nu\{-(v-v_0)^2(v+2v_0)+F_1(v-v_0)+F_0\},
\]

where v0-1=O(nu), F1=O(nu), and F0=O(nu^2), uniformly on compact
lambda sets. Consequently, on a fixed v interval,

\[
 a=-\nu s^2(v+2)+O(\nu^2|s|+\nu^3),\quad
 b=1-\nu v+O(\nu^2),\quad n=s+O(\nu),\quad c=O(\nu^2).
                                                               \tag{5}
\]

The condition |A-1|<=K_A*nu is used here. At the limiting zero fiber
h=s^2/2, the first-order scalar slope is

\[
 \mathcal R=s+\nu R_1(v)+O(\nu^2),\qquad
 R_1(v)=-3v+4,\qquad U(v)=-\frac32v^2+4v.               \tag{6}
\]

For general limiting A the polynomial would be
(A-1)v^2-(A+2)v+4. Formula (6) is an algebraic calculation away from
s=0; the estimates below justify its integrated use through s=0.

Define the first-order grazing threshold

\[
 \tau_c=2[U(1)-U(vi)]=(3vi-5)(vi-1)>0.                  \tag{7}
\]

Fix a compact interval of scaled inputs tau in [tau_min,tau_max]
with tau_max<=tau_c-mu for some mu>0, and put t=nu*tau. Then the
actual normal-forward passage between (3) exists for all sufficiently
small nu, and its outgoing signed-label map D_nu satisfies

\[
 D_\nu(t)=t-2\nu[U(vo)-U(vi)]+O(\nu^2\log(1/\nu)),
 \qquad D_\nu'(t)=1+O(\nu\log(1/\nu)).                \tag{8}
\]

The derivative is with respect to t, holding all physical parameters
fixed. Both estimates are uniform on the stated scaled-input interval
and compact coefficient sets. The cutoff can depend on rho, mu and K_A.

**Profile proof.** On an interval slightly larger than [vi,vo], set

\[
 \bar h(v)=\frac{s^2}{2}
       +\nu\{-\tau/2+U(v)-U(vi)\}.
\]

At v=1 the expression in braces is at least mu/2. Continuity gives
a fixed neighborhood of v=1 on which it is at least mu/4. Outside
that neighborhood s^2 has a positive lower bound and dominates the
bounded nu correction. Thus, for some fixed k0,K0>0 and small nu,

\[
 k_0(s^2+\nu)\le\bar h(v)\le K_0(s^2+\nu).             \tag{9}
\]

Bootstrap |h-bar h|<=k0*nu/2. It gives h comparable to s^2+nu and
|s^2-2h|<=K*nu. Equations (5) imply |a|/h=O(nu), b=1+O(nu), so
bh-a>=h/2. Expansion of the exact quotient (2), with this denominator
bound, gives

\[
 |\mathcal R-s-\nu R_1(v)|
 \le K\nu^2\left(1+\frac{|s|}{s^2+\nu}\right).         \tag{10}
\]

To check the potentially singular term, its leading numerator is
-s^2(v+2). Replacing h by s^2/2 changes its contribution to the
slope by at most K*nu*|s|*|s^2-2h|/h. This is precisely the
logarithmic term in (10). The O(nu^2|s|+nu^3) part of a contributes
at most K*(nu^2*s^2+nu^3*|s|)/(s^2+nu), which is O(nu^2).
The remaining numerator and denominator corrections are also O(nu^2).

The actual initial point is v_-(t)=vi+O(nu). At that point
h-bar h=-nu[U(v_-(t))-U(vi)]=O(nu^2). Integrating (10) therefore
bounds the bootstrap error by K*nu^2*log(1/nu), since

\[
 \int_0^d\frac{s}{s^2+\nu}\,ds
       =\frac12\log\frac{d^2+\nu}{\nu}.
\]

This is smaller than k0*nu/2 for sufficiently small nu and closes
continuation across the whole fixed v interval. The denominator stays
positive, so this is the actual forward passage in the reversed chart
time. Its right-section intersection persists by (4) and the implicit
function theorem. The intersection is vo+O(nu); substituting it into
the profile proves the first formula in (8). This is the outer branch
specified in (3). Earlier intersections with other portions of the same
geometric line lie outside the chosen local section and do not define D_nu.

## 3. Derivative control through the moving endpoints

Direct differentiation of (2) gives

\[
 \mathcal R_h=\frac{bc h^2-an-2ach}{(bh-a)^2}.           \tag{11}
\]

Using the already proved height bounds and (5),

\[
 |\mathcal R_h|\le K\left\{\nu^2+
       \frac{\nu(|s|+\nu)}{s^2+\nu}\right\}.
\]

Its integral over the passage is O(nu*log(1/nu)). Let J(v) denote
the variation of h with respect to the input label t at fixed v.
The incoming endpoint varies with t, so its correct initial value is

\[
 J(v_-(t))=
 \frac{-1/\rho-\mathcal R(v_-(t),h_-(t),\nu)}{L_-(v_-(t))}
 =-\frac12+O(\nu).
\]

The scalar variational equation gives J'=R_h*J. At the outgoing
section, differentiating h(v_out(t);t)=v_out(t)/rho yields

\[
 D_\nu'(t)=
 \frac{-L_+(v_{out})J(v_{out})}
      {\mathcal R(v_{out},h_{out},\nu)-1/\rho}.
\]

Both endpoint factors tend to their transverse limiting values, and
exp(integral R_h)=1+O(nu*log(1/nu)). This proves the derivative
formula in (8). Ignoring the moving endpoints would not justify it.

### The second derivative tends to zero

The same rapid passage satisfies D_nu''=O(sqrt(nu)), so no fixed positive
curvature lower bound can hold here. This also explains why the slope
comparison below replaces the compact-interior curvature argument.

The exact second height derivative is

\[
 \mathcal R_{hh}=\frac{2a(bn+ac)}{(bh-a)^3}.
\]

Put kappa_tau=U(1)-U(vi)-tau/2, bounded above and below by positive
constants, and d0=s^2/2+nu*kappa_tau. The profile and coefficient
estimates give

\[
 bh-a=d_0+O(\nu|s|+\nu^2\log(1/\nu)),
\]
\[
 2a(bn+ac)=-6\nu s^3
       +O(\nu s^4+\nu^2s^2+\nu^3|s|+\nu^4).
\]

Both denominators are comparable to s^2+nu. Therefore
R_hh=-6*nu*s^3/d0^3+E2, with integral |E2|=O(sqrt(nu)). The four
numerator errors have integral orders sqrt(nu), sqrt(nu), nu and
nu^(3/2), respectively, by s=sqrt(nu)*u. The inverse-cube error is
bounded by sums proportional to

\[
 \frac{\nu^2|s|^4}{(s^2+\nu)^4},\qquad
 \frac{\nu^3\log(1/\nu)|s|^3}{(s^2+\nu)^4},
\]

whose integrals are O(sqrt(nu)) and O(nu*log(1/nu)). The leading
term is odd and cancels on a common fixed symmetric interval about
s=0 contained in every passage. Outside that interval its integral
is O(nu). In particular,

\[
 \int \mathcal R_{hh}\,dv=O(\sqrt\nu),\qquad
 \int |\mathcal R_{hh}|\,dv=O(1).
\]

Let J2=partial_t^2 h at fixed v. Its incoming value is O(nu), since
the moving section and coefficients are smooth there and its base
value is zero. The equation J2'=R_h*J2+R_hh*J^2 has integration
weight 1/4+O(nu*log(1/nu)) by the first-variation bounds. Thus
J2(v_out)=O(sqrt(nu)). Differentiating the outgoing event twice gives

\[
 v_{out}''=\frac{J_2+2\mathcal R_hJv_{out}'
       +(\mathcal R_v+\mathcal R_h\mathcal R)(v_{out}')^2}
                    {1/\rho-\mathcal R},\qquad
 D_\nu''=2(v_{out}')^2+L_+v_{out}''.
\]

At the outgoing base values J=-1/2, J2=0, R=s, R_v=1 and R_h=0,
the last expression is exactly zero. Its denominators remain separated
from zero; the remaining perturbations are O(sqrt(nu)). This proves
the asserted second-derivative estimate without differentiating the
value-only remainder in (8).

## 4. Exact regular-coordinate transport

Let Z_sigma(t) be the nu-independent formula (6) in the interior note,
with r=-1. The regular normalized height z=(y-y0)/qx on the fixed
family sections (3) is exactly

\[
 Z_{\sigma,\nu}(t)
 =\frac{\rho^2Z_\sigma(t)+C\nu^2}
        {\rho^2+2\sigma\nu\rho+C\nu^2}.                \tag{12}
\]

Indeed 1/h_sigma=rho^2*(1+Z_sigma)/2, and substitution into the
physical section gives (12). Its inverse is K_sigma applied to the
inverse affine change. This transports the actual fixed-parameter
regular map G into signed labels:

\[
H_\nu=Z_{+,\nu}^{-1}\circ G\circ Z_{-,\nu}.           \tag{13}
\]

The endpoint proof gives 0<G'<=theta<1 on its admitted interval.
Choose its fixed endpoint half-width zeta small, then rho small.
For sufficiently small nu, the inverse affine output coordinate zbar
satisfies |zbar|<=2*zeta whenever |G|<=zeta. Set
Delta_sigma=rho^2+2*sigma*nu*rho+C*nu^2. Direct differentiation gives,
for |t|<=1 and rho<=1/4,

\[
 0<H_\nu'\le\theta\frac{\Delta_+}{\Delta_-}
 \frac{1+2\zeta}{(1-2\zeta)^3}
 \frac4{(1-2\rho)(2-3\rho)^2}.                          \tag{14}
\]

For example, the radical in Z_- is at least 1-2rho on this t range.
The other factors follow from
K_+'(z)=-4*[2-(1+rho)(1+z)]/[rho^2*(1+z)^3].
All factors after theta tend to one as zeta,rho and then nu/rho
tend to zero. Thus they can be chosen so that
0<H_nu'<=theta_H<1. Fix these choices before shrinking nu for (8).

## 5. A one-cycle result in this corridor

Assume a cycle uses the regular passage captured by the coarse-tube
lemma of [regular jets](HILBERT16-REGULAR-JETS.md) and the rapid passage
above, with input t=nu*tau in the stated interval. The capture lemma
also supplies the required splitting bound:

\[
 |A-1|\le M_e(\nu+\zeta/R)
       =M_e(1+\zeta/\rho)\nu.
\]

Thus K_A can be fixed to include every such admitted cycle; fixing a
single compensating value of A is unnecessary. The matching relation
alpha_R(z_in,z_out), with opposite strict endpoint derivative signs,
makes the fixed-A regular domain connected, as in the interior proof.
Its inverse image under (12), intersected with the scaled-input interval,
is connected as well.

The normal-forward rapid map runs left-to-right, and its inverse is
the physical closing passage. Hence cycles satisfy

\[
 H_\nu(t)-D_\nu(t)=0.
\]

By (8), for sufficiently small nu, D_nu'>(1+theta_H)/2>theta_H.
The displacement is strictly decreasing on its connected admitted
domain and has **at most one zero**, proving the claim. Signed labels
are ordinary coordinates on positive-height sections; no squaring of
an equation or extraneous-root argument is involved.

## 6. The old zero fiber is covered

Write V_i=1-vi>2. The old physical B_in=0 point, used by the interior
coordinate choice, has new signed family label

\[
 t=\nu\eta_i+O(\nu^2),\qquad
 \eta_i=\frac{V_i(2V_i-1)(V_i-2)}{V_i-1}.
\]

The exact gap to the rapid threshold is

\[
 \tau_c-\eta_i=V_i^2+5V_i+\frac{V_i}{V_i-1}>0.          \tag{15}
\]

Consequently one can choose a compact tau corridor strictly below
tau_c that contains this point and all sufficiently small old input
labels 0<=B_in^2<=c_rho*nu, with c_rho>0. This covers an actual
zero-fiber boundary layer. It does not justify extrapolating the old
positive-output square-root map to zero: that coordinate may cease
to be real even though the physical trajectory continues smoothly.

## 7. Checks and the remaining threshold

The benchmark [rapid passage](../../benchmarks/hilbert16_rapid_passage.py)
replays exact identities and rational coordinate/derivative margins,
including the event cancellation and rescaling exponents for the
second-derivative estimate.
Its example cutoff and derivative inputs are illustrative; the analytic
existence cutoffs in (8) have not been computed as interval certificates.

```sh
uv run --no-sync python benchmarks/hilbert16_rapid_passage.py \
  --output artifacts/hilbert16/rapid_passage.json
uv run --no-sync python -O benchmarks/hilbert16_rapid_passage.py \
  --output artifacts/hilbert16/rapid_passage_optimized.json
```

The proof requires a fixed positive gap mu below the first-order threshold.
It therefore leaves tau approaching tau_c, higher-order recentering of that
threshold, and the transition to a delayed passage unresolved. Balancing
the canonical constant drift with the transverse height suggests the core
V=epsilon^(3/2)*u, h=epsilon^3*w, whose limiting equations are
u'=lambda0-w, w'=-u*w. This local scaling alone does not match the
surrounding layers or bound their return-map zeros. The subsequent
[grazing theorem](HILBERT16-GRAZING-PASSAGE.md) supplies actual tail and
physical-section matching for H=h(V=0)>=gamma*epsilon^3 on compact
lambda0<=0 sets, extending the one-cycle count across that connected band.
The thinner H/epsilon^3->0 layer remains. The other parameter
charts, C=1, global coverage, all degrees and the curve/surface part of
Hilbert 16 remain outside this result.
