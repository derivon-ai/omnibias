# Joining the growing-height band to the positive-base passage

This written analytic argument concerns the selected small-label itinerary
of the canonical r=-1 quadratic family. It assumes a compact coefficient
set with L=-lambda0>0, lambda=lambda1<0, Delta=4L-lambda^2>=chi>0,
C in a compact subinterval of (1,infinity), and bounded Abar in
A=1+nu*Abar. The exact canonical scale is epsilon=nu*k_scale.
It claims neither full graphic cyclicity nor full Hilbert XVI. Its actual
passage estimates are not formally verified. It uses the existing published
entry-exit theorem through [singular transport](HILBERT16-SINGULAR-TRANSPORT.md), with the
passive-parameter extension justified in that note.

The conclusion below is one count over every center height 0<H<=Hmax
admitted by the specified small-label connector itinerary, away from the
negative-lambda limiting resonance. It is at most one on the positive
limiting-gap side and at most three on the negative side. The increase
from two to three permits a second displacement critical point when the
singular multiplier starts increasing in the positive-base region.

## 1. Common labels and parameter margins

Use the same actual signed section label t=(v-1)^2-2h as joint matching.
Let D map incoming to outgoing labels in normal-forward time; physical
closing is D inverse. At one fixed splitting parameter let Hreg be the
regular map on these exact signed sections. The cycle equation is

    F(t)=Hreg(t)-D(t)=0.

Put a=exp(pi*lambda/sqrt(Delta)), b=(a+1)/3, and
c(C)=exp(-4*pi/sqrt(C-1)). Fix a coefficient compact on one side of

    Gamma=log(a^2/c(C)),

with either Gamma>=gamma>0 or Gamma<=-gamma<0 throughout. All constants
can depend on this compact and its strict margins.

Choose a fixed small Bmax>0, with Bmax^2<1/4 and
1-b*Bmax>=ell>0 uniformly. Choose a small positive output square-label
box whose upper end exceeds all [a*Bmax/(1-b*Bmax)]^2, with a margin,
and is less than 1/4. Input/output boxes also include a small negative
interval about zero. Their upper input end is Bmax^2. Enlargements retain
these bounds. Such choices exist since a,b are bounded and Bmax may be
reduced first. Later positive-base input intervals will have a positive
lower bound Bmin depending on the fixed joining scale.

Choose rho and the normalized matching width zeta=8*rho sufficiently
small to retain the established strict regular contraction, endpoint
sector and coarse-tube capture, as well as the regular logarithmic error
bound below. All physical charts and connectors are then fixed; nu is
reduced afterward. Hmax>0 is fixed sufficiently small for the initial
grazing argument and the chosen signed boxes.

## 2. Regular derivative bounds in the actual coordinates

The joint regular lemma gives, on every admitted fixed-parameter matching
trajectory whose input/output labels are in the boxes,

    |log(Hreg'/c(C))| <= 30*rho+19/R+2*Kw*(rho+zeta), R=rho/nu.

Choose rho and then nu so that this is at most gamma/4. Retain the
original strict contraction Hreg'<=theta<1. Let M1,M2 be the uniform
normalized G',G'' bounds from [regular jets](HILBERT16-REGULAR-JETS.md).

The finite-nu section maps are Zsigma,nu=asigma*Zsigma+bsigma with
asigma=rho^2/Delta_sigma in [1/2,2]. Their inverses are
Ksigma,nu(z)=Ksigma((z-bsigma)/asigma). Shrink zeta and then nu so that each inverse argument
(z-bsigma)/asigma stays in the original |z|<=1/4 derivative box.
The existing coordinate bounds then give

    |Zsigma,nu'|<=4*rho^2, |Zsigma,nu''|<=16*rho^4,
    |Ksigma,nu'|<=40/rho^2, |Ksigma,nu''|<=320/rho^2.

Thus the actual regular derivative has

    |Hreg''|<=CH*rho^2,
    CH=5120*M1^2+640*M2+640*M1.

These constants are uniform in nu, R and the selected splitting, under
the same normalized amplitude tube. A zero-endpoint compensation is not
assumed. This includes the affine corrections between the older positive
fiber coordinate convention and the current actual signed label.

## 3. Strict concavity of the positive-base displacement

On any fixed positive input interval B in [Bmin,Bmax], with Bmin>0,
the existing singular-transport theorem gives

    Snu(B)=a*B/(1-b*B)+O_C2,rho(nu*log(1/nu)).

If its older section convention is used, the exact finite-nu affine
corrections composed with Z and K are identity+O_C3,rho(nu) on the
fixed positive label boxes. Composing those changes preserves the
displayed C2 convergence in the current labels. In particular
D(t)=Snu(sqrt(t))^2 converges in C2 on
[Bmin^2,Bmax^2] to

    D0(t)=a^2*t/(1-b*sqrt(t))^2.

Direct differentiation yields

    D0'(t)=a^2/(1-b*sqrt(t))^3,
    D0''(t)=3*a^2*b/[2*sqrt(t)*(1-b*sqrt(t))^4].

Hence D0''>=m0:=3*amin^2*bmin/(2*Bmax)>0 on every such interval,
since 0<1-b*sqrt(t)<=1. This lower bound does not depend on Bmin.
Choose rho so that CH*rho^2<m0/4 before choosing the joining scale.
After Bmin is fixed, C2 convergence gives D''>=m0/2 for sufficiently
small nu. Therefore

    F''<=-m0/4<0

on the entire admitted positive-base interval. The sufficiently small
nu cutoff can depend on Bmin; this is one fixed interval, not a claim
uniform down to B=0.

## 4. A positive overlap supplied by the actual joint anchor

The [two-height-level anchor proof](HILBERT16-JOINT-ANCHOR.md) gives
an actual signed anchor when
omega=exp(-kappa), u=epsilon/omega. With beta0=1/3 and the established
constants dplus,dminus, the exact cubic phase retained at fixed small u
gives, uniformly on any fixed sufficiently small interval u in [u0/2,u0],

    sqrt(t_i) -> u*dplus/(1+beta0*dplus*u),
    sqrt(t_o) -> u*dminus/(1-beta0*dminus*u)

as epsilon->0. Both limits are strictly positive; u0 is fixed first and
small enough that all denominators and physical boxes have margins.
The anchor proof uses W=X^2/2-epsilon*Y and its exact derivative
X*[b+(khat-1)*Y]/(b+khat*Y), then transfers V^2-2h to the exact
physical label. It does not differentiate a limiting value estimate.
The two limiting anchors satisfy M(Bi)=Bo exactly for M(B)=aB/(1-bB).

First choose the fixed large kappa_b and small u0 as in the [nonresonant joint theorem](HILBERT16-JOINT-MATCHING.md), with Cmatch*(exp(-kappa_b)+u0)<gamma/4. Reduce u0
further so that its positive anchors lie strictly inside the fixed boxes
and below Bmax, uniformly over the coefficient compact. Then choose
Bmin>0 less than half the smallest limiting incoming anchor at u0/2.
The actual anchors for u in [u0/2,u0] lie inside [Bmin,Bmax] for small
nu and also lie in the joint band. This supplies a nonempty overlap
between the growing-height and fixed positive-base maps, in the actual
physical labels. Uniqueness of trajectories makes the maps identical
there. No sum of independent local cycle counts is used.

## 5. Connected admission on 0<H<=Hmax

Fix small-input and small-output connector cuts V=1 and V=-1. At
epsilon=0 a signed label t in the chosen boxes corresponds to height
h=(1-t)/2 on either cut, in a compact positive interval. The ordinary
fast connectors between these cuts and the selected outer physical
sections are transverse and stay in the common normal chart. For small
epsilon their maps chi_sigma from cut height to signed section label
are strictly decreasing with derivatives tending to -2, and identify
each signed box with one cut-height interval J_sigma(epsilon).

In the central rectangle |V|<=1, the exact q=-f obeys

    q(V)>=c1*epsilon*(epsilon^2+V^2)>0.

Indeed the strict quadratic discriminant gives the bound for
L*epsilon^3-lambda*epsilon^2*V+epsilon*V^2, the limiting cubic is
favorable on the negative side, and on 0<=V<=1 the factor 1-V/3 is
uniformly positive. Equivalently use the positive quadratic lower
bound near V=0 and q_V>=0 on [0,1] when lambda<0; fixed enlargements
and small epsilon preserve the inequalities. Also k>=1/2.

For either radial side s=|V| in [0,1], every 0<H<=Hmax gives

    h_s=s*h/(q(sigma*s)+k*h), h(0)=H,
    H<=h(s)<=H+s^2.

There is no blow-up or height loss, and q's lower bound gives finite
travel time for each fixed epsilon and H>0. The cut-height variation
in H is strictly positive by the scalar variational formula. Thus the
preimage of each J_sigma(epsilon) is an interval of center heights;
their intersection with (0,Hmax] is an interval J_sing, possibly empty.
On it both signed endpoint labels are strictly decreasing in H. In
particular the singular input image I_sing is an interval.

The regular fixed-splitting matching interval I_alpha is connected by
the endpoint theorem. Its exact signed preimage, intersected with the
input box and Hreg inverse of the output box, remains an interval,
since Hreg is increasing. Call it I_reg. Then

    I=I_sing intersect I_reg

is an interval containing every cycle under consideration. Intermediate
regular trajectories need only satisfy the fine endpoint tube, as in
the preceding proofs. Any further nonmonotone itinerary restriction
would require an additional connectedness argument and is not inferred.

The initial/growing-height portion covers every admitted
H>=Hcut:=epsilon^3*(epsilon/u0)^(1/epsilon). Its input endpoint at
Hcut has a positive anchor exceeding Bmin^2. Since t_i strictly
decreases with H, every remaining admitted H<Hcut has input larger
than that anchor and at most Bmax^2. It therefore lies in the fixed
positive-base interval. This proves coverage of the stated admission
set by the two established maps; it does not assert admission for
every positive H or coverage of every nearby cycle in the full graphic.

## 6. One global count on that admitted itinerary

If Gamma>=gamma, the initial and fixed exponential regions and the
joint region have F'<0 by the existing joined proof. On the positive
base, D0'>=a^2, whereas the regular logarithmic bound puts
Hreg'<=c(C)*exp(gamma/4)<a^2 with a uniform positive absolute margin.
Actual C1 convergence preserves F'<0 on [Bmin^2,Bmax^2]. This argument
also works if the regular admitted interval does not meet the overlap.
Hence F is strictly decreasing on I and has at most one zero.

If Gamma<=-gamma, the initial region has F'<0. On the fixed positive
kappa band [kappa_a,kappa_b+1], the existing actual C1(kappa) theorem
gives F''>0 for small epsilon. On the joint band beyond kappa_b and
through u0 it gives F'>0. On the positive-base interval section 3
gives F''<0. The fixed band overlaps the initial and joint regions;
the anchor supplies the joint/positive-base overlap. The input label
increases with kappa, so each region is an input interval in that order.

Every zero of F' belongs either to the fixed band or to the positive-base
interval, because it cannot lie in the initial or joint strict-sign
regions. In the fixed band F' is strictly increasing, so it has at
most one zero. In the positive-base interval F' is strictly decreasing,
so it has at most one zero. Intersections with I remain intervals.
Thus F' has at most two zeros on the one connected domain I. Rolle's
theorem implies that F has at most three distinct zeros there. An
interval of zeros is also excluded. A section input determines at most
one physical cycle by uniqueness of the flow.

This is a count of critical points of one displacement across connected
overlaps, not an addition of the earlier two-cycle and interior two-cycle
counts. All cutoffs were chosen in a noncircular finite order. The result
is uniform on each stated coefficient compact, but not as Gamma, Delta,
L, the pole gap, or section-transversality margins vanish. In particular
the limiting resonance, varying near-resonant detuning, coalescing roots,
other parameter charts, C=1, and full graphic/all-degree coverage remain.

## Verification scope

The finite derivative and anchor identities are compatible with the
existing joint/interior benchmark replays. The analytic input consists
of the referenced actual passage theorems, the anchor estimate and the
connectedness proof above. The Mathlib module
[Hilbert16Rolle.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Rolle.lean)
now formally verifies the abstract implication from at most two derivative
zeros to exclusion of four ordered displacement zeros. It does not
formalize the physical derivative and capture hypotheses above.
No full Hilbert XVI conclusion is claimed.
