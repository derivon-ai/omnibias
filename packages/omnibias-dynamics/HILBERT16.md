# Hilbert's sixteenth problem: active proof program

The overall program concerns the full original problem. It is **not solved** here. The
results below are local reductions, regular-passage estimates, a chart
nonoscillation bound, a two-cycle interior bound, a one-cycle rapid-boundary
bound, a uniform three-cycle bound through varying multiplier detuning
on a strict canonical compact, a two-cycle bound at every height 0<H<=Hmax
admitted by a selected strict first-root saddle itinerary, and boundary reductions,
with an explicit list of the remaining obligations. No novelty is claimed.
The actual passage estimates remain written analytic arguments. Lean now
checks finite rational margin signs and abstract Rolle zero-count
implications; it does not yet check the full physical arguments. This
record distinguishes those scopes from symbolic checks and interval calculations.

The [implementation and research ledger](HILBERT16-PROGRAM.md) records the new
source-bound regular return maps, confluent interval primitives, finite
displacement zero-count calculus, exact projective curve certificates, and
remaining full-graphic obligations. The added tools do not change the scope
of the analytic cycle bounds below.

## Full completion requirements

Hilbert's original problem includes relative positions of real algebraic
curve components, the corresponding surface question, and the number and
configuration of polynomial planar limit cycles. See
[Hilbert's original statement, problem 16](https://people.reed.edu/~davidp/341/resources/hilbert.pdf).

For the dynamical part, one central requirement is

\[
\forall n\ \exists H(n)<\infty\ \forall P,Q\in\mathbb R[x,y],\quad
\max(\deg P,\deg Q)\le n\Longrightarrow
\#\{\text{isolated periodic orbits of }(P,Q)\}\le H(n).
\]

A proof for one vector field, a finite parameter sample, or even all quadratic
fields does not establish the all-degree statement. Relative configurations
also need treatment. The algebraic part needs realizations and complete
exclusions of embedded curve/surface types, with the equivalence relation
specified; abstract component counts are insufficient.

The differentiable-max route has a precise proof obligation. After
normalizing the nonzero coefficient vector of a degree-at-most-n field,
the parameter space is a compact sphere. If an explicit continuous
function U_n on that whole space satisfies

    number_of_isolated_cycles(theta) <= U_n(theta)  for every theta,

then a certified upper bound on max U_n supplies an H(n). Constructing
and validating this majorant, including degenerate coefficients and
cycles approaching singular configurations or infinity, is the central
step. Compactness or smooth optimization alone does not establish that
inequality or local boundedness of the cycle count.

The existing `omnibias.struct.torch.select.soft_max_value` computes the
smooth log-sum-exp relaxation of a finite list. Its certificate proves
max(s)<=lse_beta(s)<=max(s)+log(N)/beta for the supplied N choices.
The exact maximum retains its nondifferentiable ties in the hard limit.
This machinery can optimize candidate majorants and prioritize difficult
parameter regions; its finite-choice gap does not certify that all cycles
or all parameter regimes are represented. The local return-map proofs
below work on that coverage and majorant obligation.

| Obligation | Current evidence | Status |
|---|---|---|
| Curves and surfaces, all degrees, embedded types | Rational-representative reduction below; classical coefficient-space methods | Open in this program |
| Every nearby quadratic field represented in a five-parameter family | Exact determinant and analytic inverse-function argument below | Local reduction proved |
| Regular passage in the base family, including a cutoff tending to infinity | Complex comparison, derivative bounds, interval Schwarzian check below | Proved under stated assumptions |
| Base first parameter variations on all sufficiently large cutoffs | Exact antiderivatives, logarithmic asymptotic, integrable-tail bounds | Proved under stated fixed-section assumptions |
| Height nonoscillation in one full-unfolding chart box | Unique nullcline and strictly decreasing drift, checked over all normalized directions | At most one extremum per nonconstant segment in the box |
| Nonlinear compensation with a growing cutoff | Common contraction domain and logarithmic remainder for epsilon*(R+1) sufficiently small | Local analytic passage theorem |
| Common nonzero sections at one fixed compensated parameter | Uniform shooting sign, endpoint sensitivity, and a strictly contracting normalized passage | Regular passage only; no singular closing map |
| Regular derivatives for any admitted fixed parameter | Uniform first through fourth derivatives, negative Schwarzian, and coarse-tube capture | Proved on the endpoint matching domain |
| All nearby invariant conics in the five-parameter family | Two analytic branches, one explicit rational parabola locus | Local algebraic classification |
| Every polynomial forcing under the base variational operator | Unique exact primitive plus one scalar obstruction | Algebraic theorem; no convergence conclusion |
| Regular/singular matching on one compact interior sector | Exact moving sections, parameter-uniform singular transport, and a strict weighted-curvature criterion | At most two cycles in the specified itinerary; analytic proof, no computed singular cutoff |
| Rapid corridor containing the old zero input fiber | Positive height bootstrap, moving-endpoint derivatives, and a strict slope comparison | At most one cycle below a shifted threshold, with a fixed scaled margin |
| Connected center-height band H>=gamma*epsilon^3, with compact lambda0<=0 | Adaptive core, uniform tails, and exact physical endpoint variations | At most one captured cycle; any such cycle is attracting |
| Admitted bounded passages with lambda1>=0, without a center-height cutoff | Exact inverse-height variation, a signed derivative envelope, physical endpoint factors, and rectangular-domain interval capture | At most one attracting cycle on the connected captured domain; bounded lambda0 allowed under the crossing condition |
| Thin positive center heights with lambda0 bounded away from zero on the negative side | Exact cancellation of the large common variation and a growing-core escape bound | One cycle at most down to H=epsilon^3*exp(-kappa_*/epsilon) for a sufficiently small fixed kappa_*>0 |
| Each fixed finite exponential-height band, with lambda0 bounded strictly negative | Actual C1 logarithmic-height passage, positive-component radial cover across every discriminant sign, and an overlapping displacement argument | At most two captured cycles on epsilon^3*exp(-kappa1/epsilon)<=H<=Hmax for each fixed finite kappa1; the smallness cutoff depends on kappa1 |
| Joint infinite logarithmic height and shrinking physical escape, with 4L-lambda1^2 bounded strictly positive | Actual physical slope error bounded by C*(exp(-kappa)+epsilon*exp(kappa)); exact regular multiplier comparison | Away from the negative-lambda limiting resonance (C+3)*lambda1^2=16L, at most one or two captured cycles on epsilon^3*(epsilon/u0)^(1/epsilon)<=H<=Hmax; no kappa-derivative remainder claim |
| Joint corner joined to the positive-base map on strict positive-Delta, negative-lambda compacts away from resonance | Actual positive-label anchor, positive-base displacement concavity, and one connected admission interval | At most one or three cycles, according to the sign of the limiting gap, across every admitted 0<H<=Hmax in the stated small-label itinerary |
| First signed corrections in the actual joint singular multiplier | Positive omega and u corrections for negative lambda, with O(omega^2+u^2+epsilon*log(1/epsilon)) remainder | Actual singular slope lies above its limiting value; the sharp invariant-parabola comparison supplies the actual regular estimate |
| Exact negative-lambda limiting resonance in the strict positive-Delta canonical compact | Actual two-scale correction, sharp regular multiplier, fixed-band and positive-base comparisons, connected admission | At most one cycle across every admitted 0<H<=Hmax in the selected small-label itinerary; any such cycle is hyperbolic attracting |
| Varying detuning through the negative-lambda resonance on a fixed strict positive-Delta canonical compact | Actual singular C2(kappa) remainder, actual O(u^2) label jets, fixed-field regular logarithmic jets, and one convex-core count | Uniformly at most three distinct cycles for |Gamma|<=Gamma0 and every admitted 0<H<=Hmax in the selected small-label itinerary; analytic estimates remain outside Lean |
| Strict first-root saddle, lambda1<0 and lambda1^2-4L bounded strictly positive | Uniform invariant rectangle, exact event-determinant cancellation, outgoing continuation, incoming lower sensitivity, and connected admission | At most two cycles across all positive heights admitted by the stated small-label itinerary; every height and every nearby cycle are not asserted to be admitted |
| Coalescing-root χ-atlas on the selected small-label itinerary | Exact linear χ-identities, frozen-exponent obstruction, tracked `sep^2` product, fold I-map `dx/dkappa ~ r/kappa^2` | G1 and G4 fail; remainder versus `B_eps`, chart O, and first-hit remain |
| Saddle-node blow-up at vanishing separation | Exact double-root identity and linear `X_exit = 1`; fold I-map leading derivative | Gronwall scale tension is not the leading map; remainder versus `B_eps` open |
| Shrinking-root layer `r1 -> 0` at fixed negative `lambda1` | Exact `r1 = 2 L / (|lambda1| + sep)`; incoming first-hit at `V = u` retained | Outgoing saddle collides with the centre along `L_n = 1/n`; SR2 rematch fails on the selected itinerary |
| Two-blow-up covering in `(u, W)` | Exact `epsilon W = h^epsilon` and outgoing W-ratio; charts WF and WS | Same super-small-sep kill sequence; no C2 remainder; G1 stays failed |
| Nonzero discriminant boundary with compact positive endpoints | Exact central chart and a residence-time contradiction | Such passages are excluded for sufficiently small parameters |
| Leading pole with bounded positive input/output labels | Separate half-map matching supplies a strict pole margin | Reduced to the interior criterion; unbounded-output layer remains |
| Complete return map for the unfolding | Other singular charts, boundary faces, and coverage of all nearby cycles remain | Open |
| Uniform cyclicity near every relevant quadratic graphic | No complete displacement zero bound | Open |
| All quadratic cases and compact-cover closure | Literature inventory incomplete | Open |
| Arbitrary degree and configurations | No degree-uniform theorem | Open |

The [compensation calculation](HILBERT16-COMPENSATION.md) and
[polynomial obstruction theorem](HILBERT16-POLYNOMIAL-OBSTRUCTION.md)
identify the logarithmic source that a uniform singular analysis must retain.
The [uniform-passage proof](HILBERT16-UNIFORM-PASSAGE.md) now controls that
source in an actual nonlinear joint limit. The
[endpoint-passage proof](HILBERT16-ENDPOINT-PASSAGE.md) extends the result to
common nonzero sections and a fixed compensated vector field. The
[regular-jets proof](HILBERT16-REGULAR-JETS.md) gives uniform derivatives on
any admitted fixed-parameter matching interval and captures trajectories
from a fixed coarse tube. The
[singular-transport proof](HILBERT16-SINGULAR-TRANSPORT.md) identifies the
actual singular passage on the same sections, using the published theorem
and an explicit derivation of its passive-parameter uniformity. Together,
the [interior cyclicity proof](HILBERT16-INTERIOR-CYCLICITY.md) bounds the
specified compact-interior itinerary by two cycles. It excludes the
discriminant, zero-fiber and pole boundaries, other parameter charts, and
C=1; it does not prove that these corridors cover all nearby cycles. The
[rapid-passage proof](HILBERT16-RAPID-PASSAGE.md) now handles a shrinking
signed-label corridor containing the old zero input fiber. The
[zero-fiber calculation](HILBERT16-ZERO-FIBER.md) explains why its old
positive square-root coordinate fails. The
[boundary reductions](HILBERT16-BOUNDARY-REDUCTIONS.md) exclude compact
positive-endpoint passages near nonzero discriminant points and force a
pole margin when outputs stay bounded. The
[grazing passage](HILBERT16-GRAZING-PASSAGE.md) now extends the one-cycle
count across the connected band H>=gamma*epsilon^3, including lambda0=0.
The [height comparison](HILBERT16-HEIGHT-COMPARISON.md) removes the
center-height cutoff for admitted bounded passages with lambda1>=0.
The [subexponential passage](HILBERT16-SUBEXPONENTIAL-PASSAGE.md) also
extends the band in the negative-lambda0 region. The
[exponential passage](HILBERT16-EXPONENTIAL-PASSAGE.md) proves the actual
transition slope and its logarithmic-height derivative on every fixed
finite positive band. An overlapping displacement argument extends the
two-cycle bound to the whole band above its lower height. The
[joint matching theorem](HILBERT16-JOINT-MATCHING.md) extends that argument
to a growing logarithmic-height band away from a limiting multiplier
resonance. The [first-root saddle theorem](HILBERT16-ROOT-SADDLE.md)
gives one two-cycle count across all heights 0<H<=Hmax admitted by its
selected small-label itinerary, on strict first-root coefficient compacts.
The [positive-base join](HILBERT16-JOINED-POSITIVE-BASE.md) uses an
[actual positive-label anchor](HILBERT16-JOINT-ANCHOR.md) to give at most
one or three cycles across that same height scope in its distinct
strict-positive-Delta, nonresonant small-label itinerary.
The [varying-detuning synthesis](HILBERT16-VARYING-DETUNING.md) supplies
a uniform three-cycle bound through the limiting resonance within the
strict canonical compact. Degenerating compact margins and other
itineraries are not covered. The
[invariant-conic classification](HILBERT16-INVARIANT-CONICS.md) identifies
two local analytic reference loci; invariance does not establish cyclicity.

The [unfolding companion](HILBERT16-UNFOLDING.md) proves the new local
estimates and records the singular-map orientation obstruction. The
[algebraic-target companion](HILBERT16-ALGEBRAIC-TARGET.md) pins one open
degree-eight realization problem and sufficient exact witness routes;
no witness has been found.

The quadratic route follows the
[Dumortier--Roussarie--Rousseau program](https://doi.org/10.1006/jdeq.1994.1061).
Its local cyclicity obligation has quantifiers
\(\exists U,V,N\ \forall\theta\in V\): at most \(N\) isolated cycles in a
fixed neighborhood \(U\) of the base graphic. Separate bounds for each
parameter are insufficient.

The relevant entry-exit paper is
[Huzak--Kristiansen, Nonlinearity 39 (2026), 085026](https://doi.org/10.1088/1361-6544/ae9443).
The [updated institutional manuscript](https://documentserver.uhasselt.be/bitstream/1942/49936/1/On_entry_exit_formulas_for_degenerate_turning_point_problems_in_planar_slow_fast_systems.pdf),
section 6, treats an interior blow-up chart and still defers further cyclicity
analysis. Its hypotheses exclude several boundary faces. The program below
does not regard that entry-exit theorem as a completed cyclicity theorem.

## 1. Exact local completeness of the quadratic family

Consider

\[
\dot x=Ax-y+x^2+(\mu_2+\mu_3)xy+\mu_1y^2,\qquad
\dot y=Cx+x^2+xy+\mu_3y^2. \tag{1}
\]

At \(A=1,\mu=0\), write \(f_C=(x-y+x^2,Cx+x^2+xy)\).
For every \(C_0\ne0\), every sufficiently close quadratic field is a member
of (1) after a near-identity affine coordinate change and a positive constant
time rescaling.

**Proof.** The analytic coefficient map is

\[
(B,b,t,\theta)\longmapsto e^t B^{-1}f_\theta(Bp+b).
\]

It has six affine variables, one time variable, and five family parameters,
matching the twelve coefficients of a planar quadratic field. Its derivative
at the base has columns \(Df_Cg-Dg f_C\) for
\(g=(1,0),(0,1),(x,0),(y,0),(0,x),(0,y)\), followed by \(f_C\) and
\((x,0),(0,x),(y^2,0),(xy,0),(xy,y^2)\).
In monomial order \((1,x,y,x^2,xy,y^2)\) for each component, its determinant
is exactly \(3C\). The benchmark constructs the matrix and verifies this
identity over the rational-function field, including the pullback sign.
The analytic inverse-function theorem therefore applies. Near the identity,
\(B\) is invertible and \(e^t>0\). These transformations preserve periodic
orbits and local cyclicity. This proves the local reduction, including at
\(C_0=1\). It does not bound the family's cyclicity.

## 2. The actual regular passage

For the rest of the passage argument assume **\(A=1,\mu=0,C>1\)**. Set

\[
w=y-\tfrac12x^2+\tfrac C2,\quad
q(x)=\tfrac12((x+1)^2+C-1),\quad z=w/q(x).
\]

Direct differentiation gives

\[
\dot x=q-w,\quad\dot w=2xw,\qquad
\frac{dw}{dx}=\frac{2xw}{q-w}. \tag{2}
\]

Define \(G_R\) by starting with \(w(-R)=q(-R)z_0\) and returning
\(G_R(z_0)=w(R)/q(R)\). These are **two different fixed x sections**.
This map is not a full Poincare return map.

Put

\[
J=\frac{w}{(q+w)^2},\qquad H(z)=\frac{z}{(1+z)^2}.
\]

Since \(q'=x+1\), equation (2) yields the exact identity

\[
J_x=\frac{w_x(q-w)-2wq'}{(q+w)^3}
   =-\frac{2J}{q+w}. \tag{3}
\]

This identity, rather than a fitted neural approximation, controls the passage.

## 3. Uniform complex domain

Take \(R\ge6\), \(|z_0|\le1/16\), and \(q_0=q(-R)\). Before a
hypothetical first exit through \(|z|=1/2\), the real part of
\(1/(q+w)\) is positive. Equation (3) gives

\[
\frac{d}{dx}|J|^2=-4|J|^2\operatorname{Re}\frac1{q+w}\le0.
\]

On \([-R,R]\),

\[
\frac{q(x)}{q_0}\le\frac{q(R)}{q_0}
=1+\frac{2R}{q_0}\le2.
\]

The last inequality follows from \(R^2-6R+C>0\). Consequently

\[
|H(z(x))|=q(x)|J(x)|\le2|H(z_0)|\le\frac{32}{225}.
\]

But on \(|z|=1/2\), \(|H(z)|\ge2/9>32/225\), a contradiction.
Thus the solutions remain in \(|z|<1/2\). The denominator \(q-w\)
stays separated from zero and the solution is bounded on the finite x
interval, so continuation gives existence throughout \([-R,R]\).
For real data, \(\dot x=q(1-z)>q/2>0\), proving that this is the actual
forward passage in original time.

The same strict bootstrap works on \(|z_0|\le1/15\), because
\(2(15/196)<2/9\). Analytic dependence on initial conditions therefore
holds on an open neighborhood of the closed disk used in the estimates.
This supplies the domain required for Cauchy's derivative estimates.

The improved pointwise bound is

\[
|z(x)|=|H(z(x))|\,|1+z(x)|^2\le\frac{4q(x)}{25q_0}.
\]

It follows that

\[
\left|\int_{-R}^R\left(\frac1{q+w}-\frac1q\right)dx\right|
\le\frac{16R}{25q_0}. \tag{4}
\]

## 4. Limit map and explicit error

Equation (3) implies

\[
H(G_R(z_0))=\frac{q(R)}{q_0}H(z_0)
 \exp\left(-2\int_{-R}^R\frac{dx}{q+w}\right).
\]

Define

\[
c=\exp\left(-\frac{4\pi}{\sqrt{C-1}}\right),\quad
G_c=H^{-1}\circ(cH),\quad
d_R=\frac{82R}{25q_0}+\frac{8R}{R^2-1}. \tag{5}
\]

The inverse branch is the one fixing zero. Since
\(\int_{\mathbb R}dx/q=2\pi/\sqrt{C-1}\), the ratio of the two
exponential factors can be written \(\exp\delta\), where

\[
|\delta|\le\log(q(R)/q_0)+\frac{32R}{25q_0}
 +2\int_{\mathbb R\setminus[-R,R]}\frac{dx}{q}\le d_R.
\]

Here \(\log(1+2R/q_0)\le2R/q_0\), and the tail follows from
\(q(x)\ge(x+1)^2/2\). Therefore

\[
|H(G_R)-cH(z_0)|\le\frac{16}{225}c(e^{d_R}-1).
\]

To pass through the inverse, note that \(H(z)=H(t)\) implies
\((z-t)(1-zt)=0\); thus H is injective on the unit disk. On
\(|z|=1/2\), Rouche's theorem applied to \(z-h(1+z)^2\) gives a
unique root inside that circle whenever \(|h|<2/9\). This gives a
holomorphic inverse throughout that h disk. Both h values above, and their
joining segment, have modulus at most \(32/225<2/9\). On that inverse
branch,

\[
|(H^{-1})'|=\left|\frac{(1+z)^3}{1-z}\right|\le\frac{27}{4}.
\]

Consequently the following is a **uniform bound on a complex disk**:

\[
\sup_{|z_0|\le1/16}|G_R(z_0)-G_c(z_0)|\le
E_R:=\frac{12}{25}c(e^{d_R}-1). \tag{6}
\]

On \(|z_0|\le1/32\), Cauchy's estimate gives, for each fixed integer
\(j\ge0\),

\[
|(G_R-G_c)^{(j)}(z_0)|\le j!\,32^j E_R. \tag{7}
\]

In particular the actual regular maps converge with three controlled
derivatives. The argument is uniform on every compact real C interval
contained in \((1,\infty)\). It does not differentiate in C or cover C=1.

## 5. Negative Schwarzian, including every larger cutoff

For \(Sg=g'''/g'-(3/2)(g''/g')^2\), direct differentiation gives

\[
S G_c(s)=
-\frac{6(1-c)\big((1-c)(1+s)^2+2c(1-s)^2\big)}
 {(1-s)^2\big((1+s)^2-4cs\big)^2}<0
\]

for \(-1<s<1\), \(0<c<1\). The derivative \(G_c'\) is positive.
Thus (7) proves eventual strict negative Schwarzian on every smaller real
compact section. The numerical checker makes particular constants explicit:
it encloses \(G_c',G_c'',G_c'''\) by implicit differentiation of
\(H(G_c)=cH\), adds the three derivative error intervals, verifies a
positive first derivative, and evaluates S with outward rounding on an
exhaustive rational cell cover.

The reproducible accepted instances are:

| C range | Cutoff ray | Normalized real section | Cells |
|---|---|---|---|
| C=2 | Every R >= 1,000,000 | [-1/32,1/32] | 32 |
| 2 <= C <= 201/100 | Every R >= 1,000,000,000 | [-1/32,1/32] | 32 |

To justify the entire ray rather than one cutoff, compute

\[
\frac d{dR}\frac R{q(-R)}=
\frac{2(C-R^2)}{(R^2-2R+C)^2},\qquad
\frac d{dR}\frac R{R^2-1}=-\frac{R^2+1}{(R^2-1)^2}.
\]

Hence \(d_R\), \(E_R\), and all the derivative budgets decrease for
\(R\ge R_0\) if \(C\le R_0^2\). The checker verifies that condition
using an outward lower bound for \(R_0^2\). The real parameter upper
endpoint 201/100 is enclosed from an exact rational before calculation.

The Schwarzian sign depends on transverse coordinates. It cannot be
transported through an arbitrary nonlinear matching chart without its chain
rule terms. This is a regular-passage theorem, not a cycle-count theorem.

## 6. Why generic smooth remainders cannot complete the argument

For \(0<\varepsilon\le1\), let

\[
P_\varepsilon(s)=s+e^{-1/\varepsilon^2}\sin(s/\varepsilon),\qquad P_0(s)=s.
\]

This is jointly smooth and flat in epsilon at zero on each compact s
interval: every mixed derivative of the perturbation is its exponential
factor times finitely many polynomial/Laurent and trigonometric factors.
Also \(P_\varepsilon'\ge1-e^{-1}>0\). Nevertheless its fixed points
\(s=k\pi\varepsilon\) are simple and their number on a fixed interval
grows without bound. Taking epsilon=L/(N*pi) gives increasing selfmaps of
[0,L] with both endpoints fixed and N-1 interior fixed points.

This is **not** a polynomial-vector-field counterexample to Hilbert 16.
It proves that smoothness and arbitrarily small remainder bounds alone
cannot supply the needed uniform cyclicity theorem. The fixed-degree
polynomial origin and the exact singular-transition constraints must enter.

## 7. Algebraic topology remains part of the objective

Classical semialgebraic triviality and elimination can reduce each fixed
degree's coefficient space to finitely many representative ambient pairs.
That background reduction is not an executed all-degree classification.
Degree-eight plane-curve classification remains incomplete in the
[2026 literature](https://arxiv.org/html/2606.21449v1); a recent
[surface paper](https://www.i2m.univ-amu.fr/perso/frederic.mangolte/Real_locus_Fano_3folds.pdf)
also records the unresolved maximal component count for quintic surfaces.

A useful exact reduction is available. For a homogeneous degree-d F on
the unit sphere, put \(T_F=\nabla F-dFx\). If

\[
F^2+\|T_F\|^2\ge\eta^2>0,
\]

and a homogeneous coefficient perturbation E has coefficient l1 norm rho
with \((1+d^2)\rho^2<\eta^2\), then F and F+E have ambient-isotopic
real projective zero sets. Indeed \(|E|\le\rho\) and
\(\|T_E\|\le d\rho\), so the interpolating family has no critical
zero on the sphere. The vector field
\(-E T_{F+tE}/\|T_{F+tE}\|^2\) along its zeros produces the isotopy;
homogeneity makes it antipodally equivariant. Complex nonsingularity,
when required, is a separate open condition.

Thus rational representatives suffice for smooth realizations. This does
not provide a stopping rule for excluding unrealizable types. Even smooth
quadrics \(x^2+y^2-\varepsilon z^2\) have regularity margins tending to
zero as epsilon tends to zero; no fixed positive-margin sampling grid can
silently replace the discriminant analysis.

## 8. Exact phase-chart overlap and the unresolved joint scaling

In the positive-y compactification, write \((x,y)=(v/Z,1/Z)\).
The published phase chart has \(v=r_1v_1,Z=r_1^2,\nu=r_1\nu_1\);
therefore \(v_1=x/\sqrt y\), \(r_1=1/\sqrt y\), and
\(\nu_1=\nu\sqrt y\). At \(x=\sigma R\), \(\sigma=\pm1\), the
exact relation to the normalized regular coordinate is

\[
y=\frac{R^2}{2}\left(1+z+\frac{2\sigma z}{R}
                      +\frac{C(z-1)}{R^2}\right),\qquad
v_1=\sigma\sqrt{\frac{2}{1+z+2\sigma z/R+C(z-1)/R^2}}.
\]

For positive z<1 and R^2>C the y coordinate is positive. Put
\(t=(1-z)/(1+z)\). The finite-cutoff identity, checked symbolically, is

\[
t=\frac{(1+\sigma/R)v_1^2-1}
        {1+(\sigma/R+C/R^2)v_1^2}.
\]

Only in the limit is \(t=v_1^2-1\). Since \(H(z)=(1-t^2)/4\), the
limiting map in t is \(T(t)=\sqrt{1-c+ct^2}\). In the signed phase
coordinate it is

\[
B(v)=\sqrt{1+T(v^2-1)},\qquad -\sqrt2\le v<-1.
\]

This particular nonlinear coordinate change preserves a useful sign, but
that fact needs calculation. With t=v^2-1 and a=1-c,
\(T'=ct/T\), \(T''=ca/T^3\), and
\(T'''=-3c^2at/T^5\). They imply

\[
B B''+(B')^2-\frac{BB'}v
 =\frac{2c(1-c)v^2}{(B^2-1)^3}>0.
\]

They also give

\[
\mathcal ST=-\frac{3a(a+2ct^2)}{2t^2(a+ct^2)^2}<0,
\]
\[
\mathcal SB=4v^2\mathcal ST
 +\frac{3v^2(T')^2}{2(1+T)^2}-\frac{3}{2v^2}<0.
\]

For the last inequality use
\(v^2T'/(1+T)=(t+1)ct/[T(1+T)]<1\), which follows from T>ct.
These are **limit phase-map** inequalities. Derivative transfer to finite
maps needs the corresponding chart derivative bounds; the fixed sections
become moving curves \(r_1=\sigma v_1/R\).

The family chart is \(v=rv_2,Z=r^2z_2,\nu=r\). On its phase overlap,

\[
v_2=v_1/\nu_1,\qquad z_2=1/\nu_1^2,\qquad r=r_1\nu_1.
\]

At the moving sections \(\nu_1\sim\nu R\sqrt{(1+z)/2}\). Our base
calculation has nu=0. Reaching a fixed finite family-chart region requires
nu*R of order one. With \(\mu_1=\nu^2\bar\mu_1\) and
\(\mu_2,\mu_3=\nu\bar\mu_2,\nu\bar\mu_3\), the terms mu2*x*y
and mu1*y^2 have relative sizes nu*R and (nu*R)^2 compared with x^2.
They cannot be discarded in that matching regime. The phase variable is
also not the later slow-fast coordinate or its fast-fiber entrance base point.
Time reversal in the published reduction must be tracked explicitly.

The companion proofs now give a joint-parameter REGULAR passage retaining
nu*R on a sufficiently small positive sector. They do not identify it with
the transition through all resonant and singular charts. Compatible
coordinates and bounds have subsequently been established for the
compact-interior and rapid corridors described below; the other charts
remain outside those results.

A sufficient conditional zero-count gate is available. If the actual outer
and closing maps have compatible section coordinates, nonzero derivatives,
negative/nonpositive Schwarzians, and increasing composition P, then SP<0.
Such a P has at most three fixed points: four would yield three points with
P'=1 by Rolle's theorem, contradicting strict convexity of (P')^(-1/2), whose
second derivative is -(SP)(P')^(-1/2)/2>0. The required inequalities for the
actual closing map remain unproved. This gate is not an achieved cycle bound.
Moreover, the published slow-fast reduction reverses time: the closing map
uses the inverse of its normal-forward entry map. In squared fast-fiber
coordinates that inversion changes its negative Schwarzian to positive.
The [companion analysis](HILBERT16-UNFOLDING.md) explains why the sign cannot
be transferred without computing the actual coordinate changes.

## 9. Nonlinear passage on common sections

Write mu2=epsilon*p, mu3=epsilon*r and mu1=epsilon^2*m, with bounded
p,r,m and C in a compact subinterval of (1,infinity). The new proofs use
an explicitly bounded inverse of q*d/dx-2*x, projected to remove its
growing homogeneous mode, followed by a contraction argument for the exact
trajectory equation. For R>=2, normalized endpoint heights bounded by
zeta, and

    (epsilon+zeta/R)*(R+1) <= eta_p,

where eta_p is a strictly positive explicit parameter-dependent constant,
they construct the matching relation

    alpha_R(epsilon,z_-,z_+)
      = -epsilon*[(C+6)*p+3*r]
        +8*(C+3)*epsilon^2*[p*(2*p+r)-m]*log R
        +O(epsilon/R+epsilon^2+zeta/R^2).

The remainder is uniform on this sector. The selected trajectories have
xdot>=q/2, uniformly bounded physical flight time, and opposite nonzero
matching derivatives in z_- and z_+, both of order R^-2. Their existence
and uniqueness are proved within a stated amplitude tube.

Fixing alpha_*=alpha_R(epsilon,0,0) gives an actual passage map on the
entire interval [-zeta,zeta], satisfying G_R(0)=0 and
0<G_R'<=theta<1. For C in [2,3], theta=6/7 is an admissible conservative
bound after the stated sector restriction. This is a map between two
different sections, so its contraction does not itself give a limit cycle.

The constants can be extremely small. The proofs establish a nonempty
sector and include R=rho/epsilon for sufficiently small fixed rho and
section radius zeta. They do not establish uniform control for every
blow-up direction or across C=1. The exact benchmark checks cover the
finite algebra and rational sufficient inequalities; the analytic arguments
are written proofs, not formal theorem-prover results.

## 10. Interior matching and boundary progress

The [interior theorem](HILBERT16-INTERIOR-CYCLICITY.md) uses a weighted
second derivative of the actual signed-square displacement, rather than
the conditional Schwarzian-composition gate above. Exact coordinates and
a uniform singular C2 remainder make that derivative strictly negative,
giving at most two cycles on its connected compact-interior domain.

The [rapid theorem](HILBERT16-RAPID-PASSAGE.md) uses the exact family
section label t=(v-1)^2-2h. For t=nu*tau and tau in a compact interval
strictly below its shifted threshold tau_c, it proves

    D_nu'(t)=1+O(nu*log(1/nu)), D_nu''(t)=O(sqrt(nu)).

The regular map in the same coordinates has derivative bounded strictly
below one. Their displacement is therefore strictly decreasing and has
at most one zero. An exact positive margin places the old zero input
fiber inside this rapid corridor. This estimate only needs bounded
canonical lambda parameters; it does not assume a negative discriminant.

The [boundary reductions](HILBERT16-BOUNDARY-REDUCTIONS.md) use separate
half-transitions and the exact central chart. With compact positive
input/output labels, passages near nonzero discriminant points are
excluded by a residence-time inequality, without imposing a relative
rate on the two small parameters. On a strict-discriminant compact set,
bounded positive outputs force a positive margin from the leading pole,
returning the matching problem to the interior domain.

The [grazing theorem](HILBERT16-GRAZING-PASSAGE.md) uses the actual
center height H=h(V=0), with H in [gamma*epsilon^3,Hmax]. On a compact
lambda0<=0 parameter set it proves D'=1+o(1) uniformly across this entire
connected band. Its adaptive scale a^2=H+epsilon^3 keeps the core regular;
explicit tail estimates and moving physical endpoints transfer the
variation to the same section labels as the regular map. Comparison gives
one cycle at most on the whole admitted band, and that cycle, if present,
is hyperbolic and attracting. The count includes the rapid corridor in
the overlapping parameter region; it does not add separate subinterval
counts. Relative to exact section anchors, the cubic-height coordinate
also has a uniform C1 logarithmic transfer law.

The [height-comparison theorem](HILBERT16-HEIGHT-COMPARISON.md) gives
an exact signed derivative envelope when lambda1>=0. It cancels the
possibly unbounded center-variation factor and yields D'>=1-o(1) at
admitted bounded passages, without a lower bound on H/epsilon^3. Bounded
positive lambda0 is also allowed when the center has the required crossing
F(0,H)<0. Rectangular-chart comparison proves that centers between two
admitted centers remain admitted, under the stated outer-section interval
hypotheses. This gives at most one attracting cycle on that connected
captured domain. It does not assert that all center heights reach the
sections.

The [subexponential theorem](HILBERT16-SUBEXPONENTIAL-PASSAGE.md) retains
both signs of lambda1 when lambda0 stays strictly negative. Its exact
variation identity removes a common factor before the growing-core and
tail estimates are applied. For arbitrary K>=0 it proves

    |D'-1| <= C*sqrt(epsilon*(1+K))

on epsilon^3*exp(-K)<=H<=epsilon^3 whenever epsilon and
delta=epsilon*(1+K) pass fixed positive smallness gates. Thus it covers
every subexponential logarithmic range, and the strict regular contraction
also gives one captured cycle at most down to
H=epsilon^3*exp(-kappa_*/epsilon) for a sufficiently small fixed kappa_*>0.
This constant is not computed as an interval certificate. The alleged
earlier epsilon^(-1/3) logarithmic threshold was an unsupported symmetric-window inference: actual unequal escape
endpoints cancel that leading term.

The [exponential theorem](HILBERT16-EXPONENTIAL-PASSAGE.md) now establishes
an actual transition law on every fixed compact positive kappa interval,
where H=epsilon^3*exp(-kappa/epsilon). With L=-lambda0>0 and
B_sigma(x)=L-sigma*lambda1*x+x^2, let x_sigma solve
integral_0^x s/B_sigma(s) ds=kappa on the positive component containing
zero. The physical slope converges in C1(kappa) to

    R(kappa)=B_-(x_-)/B_+(x_+),
    (log R)'=lambda1*(1/x_-+1/x_+).

The actual input-label derivative is
2*epsilon^2*B_+(x_+)*(1+o(1)). Uniform radial coverage works for every
discriminant sign on compact sets with L bounded away from zero.
For lambda1 bounded negatively away from zero, the actual singular
curvature is negative of order epsilon^(-2), which dominates bounded
regular curvature. A separate slope comparison handles lambda1 near
zero or positive. The proof differentiates an exact compensated integral,
including height-dependent coefficients and moving physical endpoints.

The initial one-cycle region overlaps this exponential band in the same
physical coordinates. The displacement is decreasing on the initial
portion and either decreasing or strictly convex on the later portion.
Its derivative has at most one zero across the union. This proves at most
two captured cycles on the whole band
H in [epsilon^3*exp(-kappa1/epsilon),Hmax] for every fixed finite kappa1.
The proof does not add separate local counts. Its epsilon cutoff depends
on kappa1; it does not cover kappa growing without bound as epsilon tends
to zero.

A subsequent [joint matching theorem](HILBERT16-JOINT-MATCHING.md) uses
omega=exp(-kappa), u=epsilon/omega and epsilon=omega*u. For
Delta=4L-lambda1^2>=chi>0, it proves for the actual physical slope

    |log D_epsilon' - 2*pi*lambda1/sqrt(Delta)| <= C*(omega+u)

throughout a sufficiently small two-parameter corner, with no restriction
on the relative rates of omega and u. Exact common-factor cancellation
precedes the estimates, and the physical section factors are retained.
The result controls the value of the slope, not its kappa derivative.
The regular derivative has logarithm within
30*rho+19/R+2*K_w*(rho+zeta) of -4*pi/sqrt(C-1), with R=rho/nu.
For negative lambda1 the difference of limiting logarithmic multipliers
has the sign of 16L-(C+3)*lambda1^2. On compact sets separated from zero,
the initial, fixed-band and joint estimates overlap and bound the whole
captured band epsilon^3*(epsilon/u0)^(1/epsilon)<=H<=Hmax by one cycle
on the positive-sign side and two on the negative-sign side.

For the opposite discriminant sign, the
[first-root saddle theorem](HILBERT16-ROOT-SADDLE.md) assumes
lambda1<0, L>=c>0 and lambda1^2-4L>=delta0>0. A fixed invariant
rectangle about the exact first outgoing root and the planar
event-determinant identity prove D_epsilon'<=C_tail*exp(-gamma*kappa)
at every sufficiently large admitted kappa, with one epsilon cutoff.
Outgoing continuation is proved directly; a fixed positive-height incoming
connector supplies a uniform lower input sensitivity. The selected
small-label admission set is one interval. A uniform positive regular
slope, the compact-band curvature theorem and their overlaps then give
at most two cycles across every admitted 0<H<=Hmax. An outer-tail cycle,
if present, is hyperbolic and repelling. The result does not assert that
every positive height or every nearby cycle is admitted by this itinerary.

A further [two-scale correction](HILBERT16-TWO-SCALE-CORRECTION.md)
sharpens the actual singular expansion to

    log D' = 2*pi*lambda1/sqrt(Delta)
             -lambda1*(1/dminus+1/dplus)*omega
             +(dminus+dplus)*u
             +O(omega^2+u^2+epsilon*log(1/epsilon)).

The remainder is o(omega+u) with arbitrary relative rates. Both leading
corrections are positive for negative lambda1. This places the actual
singular derivative above its limiting value in the sufficiently small
corner. The [sharp regular estimate](HILBERT16-REGULAR-ANCHOR.md) now
bounds the actual regular multiplier error by
K_rho*(nu*log(2/nu)+|ti|+|to|), using the exact invariant-parabola
reference and its endpoint cancellation. The
[exact-resonance synthesis](HILBERT16-EXACT-RESONANCE.md) gives at most
one cycle on that slice throughout the selected admitted itinerary; any
such cycle is hyperbolic attracting. The leading detuned slope model
has a balance at omega proportional to sqrt(epsilon) and may have two
slope zeros; those model zeros are not asserted for the actual map.

The [actual anchor](HILBERT16-JOINT-ANCHOR.md) retains the cubic phase
exactly and proves

    t_sigma=[u*d_sigma/(1+sigma*d_sigma*u/3)]^2
              +O_rho(epsilon*(1+u^2*log(1/epsilon))).

On a fixed small positive u band, both endpoints converge to positive
labels satisfying the established Mobius relation. This supplies the
previously missing overlap with the actual compact positive-base map.
On that positive-base interval the singular squared-label map has
strictly positive second derivative, which dominates regular curvature
after rho is reduced. The [joined argument](HILBERT16-JOINED-POSITIVE-BASE.md)
then controls one displacement over every admitted 0<H<=Hmax in the
specified small-label itinerary. It gives at most one cycle for positive
limiting gap and at most three for negative limiting gap. On the latter
side, one displacement critical point can occur in the fixed exponential
band and one in the positive-base interval. Rolle's theorem supplies the
three-zero bound on the connected domain. Separate cycle counts are not
added. Every height and every nearby cycle are not asserted to be admitted.

The [varying-detuning argument](HILBERT16-VARYING-DETUNING.md) now
proves a uniform at-most-three bound on one fixed sufficiently small
|Gamma| interval within the strict canonical compact. Its actual singular
remainder is controlled through two kappa derivatives by
C*(omega^2+u^2+epsilon*log(1/epsilon)); actual input-label jets are O(u^2).
Combined with fixed-field regular jets, this makes the logarithmic
multiplier difference strictly convex on the joint interval. A single
convex sublevel interval joins the negative-at-zero outer comparisons.
The abstract convexity/Rolle implication is checked in Lean; the uniform
physical estimates remain written analytic proofs. See
[the verification manifest](HILBERT16-FORMAL.md).

A total graphic cyclicity bound remains unresolved. Coalescing roots,
shrinking-lambda0 layers, loss of the stated small-label capture,
escaping endpoints, and other
parameter regimes remain. These arguments
give existence of sufficiently small cutoffs, without computed interval
certificates for those cutoffs or formal analytic verification.

## 11. Next unresolved proof obligations

1. The selected strict-canonical varying-detuning bound is now supplied by
   HILBERT16-VARYING-DETUNING.md and its actual singular/regular kappa-jet
   companions, with common constants and cutoff order. Its analytic
   estimates, selection of the clipped convex sublevel interval, and
   complete physical hypotheses are not yet formalized in Lean. The
   abstract whole-interval increasing-core implication is checked.
   The [coalescing-capture companion](HILBERT16-COALESCING-CAPTURE.md)
   records a χ-atlas, a restricted shrinking-rectangle first-derivative
   bound, and an explicit G1/G4 failure. The follow-up
   [saddle-node](HILBERT16-SADDLE-NODE.md),
   [shrinking-root](HILBERT16-SHRINKING-ROOT.md),
   [two-blow-up](HILBERT16-TWO-BLOWUP.md), and
   [next-atlas](HILBERT16-NEXT-ATLAS.md) notes fail on the same named
   sequences (fold-versus-separation scale tension, rewritten as a
   W-ratio and then as a scale dichotomy; outgoing saddle colliding
   with the centre; SR2 rematch failing on the selected itinerary;
   log-intermediate charts LI/WL not a third scale; fold I-map leading
   derivative at `sep = 0`). The
   [chart-cell ledger](HILBERT16-CHART-CELLS.md) records those labels.
   G1 remains failed; G4 is not opened.
   Keep the existing sections, positive-label overlaps and connected
   admission in any extension; finite replay or a fitted neural jet does
   not establish those analytic premises.
2. Extend the compensated common-section sector to the parameter regions
   required by the full graphic, retaining higher section derivatives and
   bounds through the resonant and singular transitions.
3. Cover the discriminant origin, shrinking or unbounded fiber endpoints,
   the incoming slow endpoint, other weighted parameter charts, and the
   extra finite saddle-node at C=1. The current nonzero-discriminant
   exclusion only covers compact positive endpoints.
4. Prove a uniform bound on displacement zeros after those maps are composed,
   including coefficient cancellations and identity/center cases.
5. Close every other quadratic case, then the all-degree and configuration
   questions. Keep the curves/surfaces classification gates separate.

## Reproduction and evidence

From a workspace environment with the dynamics package, mpmath, and SymPy:

```bash
python benchmarks/hilbert16_regular_passage.py --output artifacts/hilbert16/regular-passage.json
python benchmarks/hilbert16_uniform_passage.py --output artifacts/hilbert16/uniform-passage.json
python benchmarks/hilbert16_endpoint_passage.py --output artifacts/hilbert16/endpoint-passage.json
python benchmarks/hilbert16_invariant_conics.py --output artifacts/hilbert16/invariant-conics.json
python benchmarks/hilbert16_exponential_passage.py --output artifacts/hilbert16/exponential_passage.json
python benchmarks/hilbert16_compactified_tail.py --output artifacts/hilbert16/compactified_tail.json
python benchmarks/hilbert16_joint_matching.py --output artifacts/hilbert16/joint_matching.json
python benchmarks/hilbert16_root_saddle.py --output artifacts/hilbert16/root_saddle.json
python benchmarks/hilbert16_coalescing_capture.py --output artifacts/hilbert16/coalescing_capture.json
python benchmarks/hilbert16_saddle_node.py --output artifacts/hilbert16/saddle_node.json
python benchmarks/hilbert16_two_blowup.py --output artifacts/hilbert16/two_blowup.json
python benchmarks/hilbert16_next_atlas.py --output artifacts/hilbert16/next_atlas.json
python -m benchmarks.hilbert16_formal_replay --output artifacts/hilbert16/formal_margin_replay.json
python -m pytest packages/omnibias-dynamics/tests/test_quadratic_graphic.py -q
python -m pytest packages/omnibias-dynamics/tests/test_quadratic_unfolding.py -q
```

The symbolic benchmarks perform exact identities and interval arithmetic.
The formal replay additionally sends six sealed interval-margin sign
obligations through the existing minimal Lean kernel. It verifies the
signs conditional on supplied interval membership, not the interval
evaluator or actual-family enclosure premises. The Mathlib module
[Hilbert16Rolle.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Rolle.lean)
formally proves that at most one or two actual derivative zeros exclude
three or four ordered function zeros, under continuity and derivative
hypotheses. It has no hypothesis asserting the desired function-zero
count. The actual passage estimates, parameter uniformity and capture
arguments supplying its hypotheses remain written proofs.

Tests independently differentiate the limit at high precision,
integrate the complex rational equation on grids and random points, check
the rational coordinate identities, and exercise both successful and
unresolved sign checks. Numerical comparisons test implementation and do
not replace the analytic proof.

The saddle-node, shrinking-root, two-blow-up, and next-atlas increments
are written and all fail, so G1 stays failed and G4 stays closed. A pair
of `(u, W)` charts does not cover `sep = exp(-1/epsilon^2)`; logarithmic
charts LI/WL do not produce a third scale; the kill sequence remains
admitted; and SR2 does not rematch `L_n = 1/n` onto an existing height
theorem on the selected itinerary.
The broader program still seeks a full solution of Hilbert's sixteenth
problem. The present evidence covers a compensated nonlinear regular
passage on common sections, local invariant-conic branches, restricted
cycle bounds through connected initial/exponential/positive-base
regions, a strict first-root saddle itinerary, boundary exclusions, and
a restricted χ-bound on chart D. Full graphic coverage, the all-degree
bound, and curve/surface classification remain unresolved.
